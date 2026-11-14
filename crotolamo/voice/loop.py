"""Loop de voz concurrente con interrupción por turn_id (de GLaDOS, redo M3).

Arquitectura race-free: cuatro threads (Ear/Stt/Brain/Mouth) coordinados por colas
y un SharedState con turn_id monótono. Cada frase nace con su turn_id; el MouthThread
descarta las de turnos abortados. La interrupción es correcta POR CONSTRUCCIÓN: nada
de Events de "interrupción" que se limpian (races); toda invalidación pasa por
state.new_turn() + state.is_current().

Este módulo quedó como ORQUESTADOR (VoiceLoop, M3.7). Los threads viven en
threads.py y los adapters reales (mic/VAD) en adapters.py; se re-exportan aquí
por compatibilidad con los consumidores que importaban todo desde loop.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
from functools import partial

from crotolamo.logging_setup import get_logger
from crotolamo.voice.adapters import _RealMic, _SileroVad
from crotolamo.voice.notify import DirWatch, Wakeable
from crotolamo.voice.state import SharedState, read_control_enabled
from crotolamo.voice.threads import (
    END,
    STOP,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
    _drain_queue,
)

# Re-exports de compatibilidad: los nombres históricos de loop.py siguen
# importables desde aquí aunque ahora vivan en threads.py / adapters.py.
__all__ = [
    "END",
    "Utterance",
    "MouthThread",
    "BrainThread",
    "SttThread",
    "EarThread",
    "VoiceLoop",
    "_drain_queue",
    "_RealMic",
    "_SileroVad",
]

log = get_logger("voice.loop")

# Canal de control (panel -> loop) vigilado con inotify: el hilo principal
# duerme hasta que control.json cambia, un hilo termina o llega el apagado.
# Igual se re-mira cada _CONTROL_SAFETY_S por si se perdiera un evento (no
# debería: un desborde de la cola del kernel también despierta).
_CONTROL_SAFETY_S = 30.0
# Sin inotify (o si deja de funcionar, o control.json es un enlace): el sondeo
# de siempre, un stat() cada 0.3 s.
_CONTROL_POLL_S = 0.3


def _publisher_with_ducker(hud_publisher, ducker):
    """Publisher de SharedState que además avisa al Ducker del modo vigente.

    SharedState publica un dict con "mode" en cada transición; el Ducker baja
    la música al salir de idle y la restaura al volver (ver media_aware.py).
    Se compone AQUÍ, sin tocar state.py: el ducking es una preocupación del
    orquestador, no del estado. on_mode solo encola (los hilos de audio que
    publican no esperan ningún subprocess) y un fallo suyo nunca impide la
    publicación al HUD.
    """

    def _publish(state_dict: dict) -> None:
        try:
            ducker.on_mode(state_dict.get("mode", "idle"))
        except Exception as error:  # noqa: BLE001 - el ducking no tumba el HUD
            log.warning("ducker: %s", error)
        if hud_publisher is not None:
            hud_publisher(state_dict)

    return _publish


class VoiceLoop:
    """Orquestador: arma los 4 threads y los apaga limpio (M3.7).

    Args:
        hud_publisher: callable(dict) -> None para publicar el estado del HUD en
                       tiempo real. Si es None (default), no se publica nada.
                       Usar ``make_file_publisher(path)`` de state.py para escritura
                       atómica a archivo.
        ducker: Ducker de media_aware.py (opcional). Atenúa la música mientras
                el loop no está en idle y la restaura al volver; se engancha al
                publisher del estado. None (default) = sin ducking, nada cambia.
    """

    def __init__(self, agent, stt, tts, wake_detector, *, allow_barge_in: bool = False,
                 silence_ms: int = 640, mic=None, wake_fn=None, vad_fn=None,
                 to_wav=None, hud_publisher=None, control_path=None, ducker=None) -> None:
        self.ducker = ducker
        publisher = hud_publisher
        if ducker is not None:
            publisher = _publisher_with_ducker(hud_publisher, ducker)
        self.state = SharedState(publisher=publisher)
        # Sigue siendo un threading.Event (un shutdown.set() a secas apaga
        # todo), pero además DESPIERTA: mete un STOP en cada cola, así
        # Mouth/Brain/Stt esperan sin timeout (ver threads._get_timeout), y
        # avisa al bucle principal de run().
        self.shutdown = Wakeable()
        # Canal de control inverso (panel -> loop). None = sin canal (tests).
        self.control_path = control_path
        # Firma (mtime, tamaño, inodo) del control.json la última vez que se
        # leyó: solo se relee y parsea cuando cambia (ver _poll_control).
        self._control_sig: tuple[int, int, int] | None = None
        # Nombre que se espera en los eventos de inotify de su directorio.
        self._control_names: set[str] = (
            set() if control_path is None else {os.path.basename(os.fspath(control_path))})
        self.stt_q: queue.Queue = queue.Queue()
        self.cmd_q: queue.Queue = queue.Queue()
        self.tts_q: queue.Queue = queue.Queue()
        for q in (self.stt_q, self.cmd_q, self.tts_q):
            self.shutdown.add_waker(partial(q.put, STOP))
        # "Hay algo que mirar YA" para el bucle principal: el apagado o un hilo
        # que terminó. Con inotify también despierta su select (ver run()).
        self._aviso = Wakeable()
        self.shutdown.add_waker(self._aviso.set)
        # Hilos cuyo run() ya volvió. is_alive() sigue en True un instante
        # después del aviso (el hilo aún no terminó de cerrarse): sin esto el
        # bucle podía despertar, ver al hilo "vivo" y volver a dormir 30 s.
        self._terminados: set[threading.Thread] = set()
        # fd de despertar de señales que había antes de enrutarlas al vigía (None =
        # no se tocó). Ver _senales_al_vigia.
        self._fd_senales_previo: int | None = None
        self.tts = tts
        # M3.8: parámetros de mitigación de eco desde la config (config-first).
        try:
            from crotolamo.settings import get_settings

            vcfg = get_settings().voice
        except Exception as error:  # noqa: BLE001 — sin config, defaults
            log.warning("no pude cargar la config de voz (%s); uso defaults", error)
            vcfg = {}
        # Adapters reales por defecto; los tests inyectan fakes (no abre audio).
        # input_device: None (default) = micrófono default del sistema. Apúntalo a
        # "crotolamo_aec_source" tras activar el AEC (ver desktop/aec.sh).
        self._mic = mic or _RealMic(
            sample_rate=vcfg.get("sample_rate", 16000),
            device=vcfg.get("input_device"),
        )
        ear = EarThread(
            self._mic,
            wake_fn or wake_detector.feed,
            vad_fn or _SileroVad(),
            # Camino caliente EN MEMORIA: los frames van directo a Whisper como
            # ndarray (SttThread lo acepta), sin WAV temporal por comando.
            to_wav or stt._frames_to_audio,
            tts, self.stt_q, self.tts_q, self.state, self.shutdown,
            allow_barge_in=allow_barge_in, silence_ms=silence_ms,
            vad_threshold=vcfg.get("vad_threshold", 0.8),
            barge_in_grace_ms=vcfg.get("barge_in_grace_ms", 400),
            barge_in_threshold_margin=vcfg.get("barge_in_threshold_margin", 0.1),
            barge_in_min_chunks=vcfg.get("barge_in_min_chunks", 5),
            # None (default) = sin histéresis, comportamiento idéntico al previo.
            vad_neg_threshold=vcfg.get("vad_neg_threshold"),
            # Lo que se antepone a la orden cuando el patrón calla un rato tras
            # el wake (V1); mismo valor y default que el modo simple.
            preactivation_ms=vcfg.get("vad_preactivation_ms", 800),
        )
        # El oído es el ÚNICO lector del micrófono: stop() lo espera antes de
        # cerrar el stream (ver stop()).
        self._ear = ear
        self.threads = [
            ear,
            SttThread(stt, self.stt_q, self.cmd_q, self.state, self.shutdown,
                      hotwords=vcfg.get("hotwords",
                                        "Crotolamo, Tletl, Huevonitis")),
            BrainThread(agent, self.cmd_q, self.tts_q, self.state, self.shutdown),
            MouthThread(tts, self.tts_q, self.state, self.shutdown),
        ]
        for hilo in self.threads:
            self._avisar_al_terminar(hilo)

    def _avisar_al_terminar(self, hilo: threading.Thread) -> None:
        """Al terminar run() (el oído que se rindió, una excepción, el apagado)
        el hilo despierta al bucle principal, que así lo ve al instante en vez
        de mirar is_alive() cada 0.3 s (V2: un hilo muerto apaga el loop)."""
        run = hilo.run

        def _run() -> None:
            try:
                run()
            finally:
                self._terminados.add(hilo)
                self._aviso.set()

        hilo.run = _run  # type: ignore[method-assign]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def _poll_control(self) -> None:
        """Mira el canal de control (panel -> loop) y lo relee SOLO si cambió.

        Se compara la firma (mtime, tamaño, inodo) con la de la última lectura:
        si no cambió, no se lee ni se parsea el JSON. El mtime solo no basta
        desde que el loop despierta con el evento: lee el archivo en cuanto se
        CREA, antes de que el escritor termine, y si crear y escribir caen en
        el mismo tic del reloj del sistema de archivos el mtime no cambia y la
        pausa se perdía para siempre. Un reemplazo atómico (el panel) cambia el
        inodo; una escritura en el sitio, el tamaño (true/false).
        Archivo ausente => escucha activa (default-seguro), igual que antes.
        """
        try:
            st = os.stat(self.control_path)
        except FileNotFoundError:
            # Sin archivo de control: default-seguro (escucha activa), como
            # hacía read_control_enabled. Si reaparece, la firma nueva forzará
            # la relectura.
            self._control_sig = None
            self.state.set_enabled(True)
            return
        except OSError:
            # E/S rara (permisos, etc.): mismo default-seguro que la lectura.
            self.state.set_enabled(True)
            return
        firma = (st.st_mtime_ns, st.st_size, st.st_ino)
        if firma == self._control_sig:
            return
        self._control_sig = firma
        self.state.set_enabled(read_control_enabled(self.control_path))

    def _abrir_vigia_del_control(self) -> DirWatch | None:
        """inotify sobre el directorio de control.json (el panel lo reemplaza
        por rename). None = sin canal, o el sondeo de siempre si no se puede."""
        if self.control_path is None:
            return None
        ruta = os.fspath(self.control_path)
        if os.path.islink(ruta):
            # El panel reemplaza el DESTINO del enlace (dotfiles), en otro
            # directorio: vigilar este no lo vería.
            log.info("%s es un enlace: lo miro cada %.1f s", ruta, _CONTROL_POLL_S)
            return None
        try:
            directorio = os.path.dirname(os.path.abspath(ruta))
            os.makedirs(directorio, exist_ok=True)
            watch = DirWatch.open(directorio)
        except OSError as error:
            log.warning("no puedo vigilar %s con inotify (%s): lo miro cada %.1f s, "
                        "como antes", ruta, error, _CONTROL_POLL_S)
            return None
        self._aviso.add_waker(watch.wake)
        return watch

    def _senales_al_vigia(self, watch: DirWatch | None) -> None:
        """Con el vigía, el hilo principal duerme en su select hasta
        _CONTROL_SAFETY_S. CPython solo corre los manejadores de Python (el
        _graceful_exit del listener) en el hilo principal, y el kernel entrega la
        señal del proceso a OTRO hilo si el principal ya tiene una pendiente (el
        SIGCHLD de un playerctl): el select no se enteraba y el SIGTERM de systemd
        esperaba hasta 30 s (TimeoutStopSec=10: SIGKILL, sin apagado limpio). Con
        set_wakeup_fd el manejador de C escribe en el pipe de despertar del vigía,
        el select vuelve y el manejador corre al instante. Solo en el hilo
        principal (set_wakeup_fd no se puede llamar desde otro; los tests corren
        run() en hilos)."""
        if watch is None or threading.current_thread() is not threading.main_thread():
            return
        try:
            self._fd_senales_previo = signal.set_wakeup_fd(
                watch.wake_fd, warn_on_full_buffer=False)
        except (ValueError, OSError) as error:
            log.debug("no pude enrutar las señales al vigía: %s", error)

    def _cerrar_vigia(self, watch: DirWatch) -> None:
        """Devuelve el fd de despertar de señales ANTES de cerrar el pipe: si no,
        el manejador de C escribiría en un fd cerrado o, peor, ya reutilizado por
        otra cosa (un socket, el micro)."""
        previo, self._fd_senales_previo = self._fd_senales_previo, None
        if previo is not None:
            try:
                signal.set_wakeup_fd(previo)
            except (ValueError, OSError) as error:
                log.debug("no pude devolver el fd de despertar de señales: %s", error)
        watch.close()

    def _esperar(self, watch: DirWatch | None) -> DirWatch | None:
        """Duerme hasta que haya algo que mirar: control.json cambió, un hilo
        terminó o llegó el apagado (con inotify, o re-chequeo de seguridad cada
        _CONTROL_SAFETY_S). Devuelve el vigía para la vuelta siguiente: None si
        dejó de servir, y desde ahí el sondeo de siempre."""
        if watch is not None:
            watch.wait(self._control_names, _CONTROL_SAFETY_S)
            # Limpiar ANTES de mirar: lo que avise después vuelve a despertar.
            self._aviso.clear()
            if watch.broken:
                log.warning("dejé de poder vigilar %s con inotify (¿se borró o movió el "
                            "directorio?): lo miro cada %.1f s, como antes",
                            self.control_path, _CONTROL_POLL_S)
            elif os.path.islink(self.control_path):
                log.info("%s pasó a ser un enlace: lo miro cada %.1f s",
                         self.control_path, _CONTROL_POLL_S)
            else:
                return watch
            self._cerrar_vigia(watch)
            return None
        if self.control_path is None:
            self._aviso.wait()
        else:
            self._aviso.wait(_CONTROL_POLL_S)
        self._aviso.clear()
        return None

    def run(self) -> int:
        """Corre el loop hasta el apagado. Devuelve 0 si se apagó por las buenas
        y 1 si murió alguno de sus hilos (V2): el listener lo usa como código de
        salida para que systemd (Restart=on-failure) reinicie el servicio."""
        # El vigía se abre ANTES de la primera lectura: un cambio entre las dos
        # no se pierde (llega como evento).
        watch = self._abrir_vigia_del_control()
        self._senales_al_vigia(watch)
        # Estado inicial de la escucha desde el canal de control (default: activa).
        if self.control_path is not None:
            self._poll_control()
        # Publicar el estado inicial SIEMPRE: pisa un hud_state.json rancio de
        # un proceso anterior que murió a medio turno (el HUD se quedaba en
        # "PENSANDO" hasta el siguiente cambio de modo).
        self.state.publish_now()
        # Los hilos arrancan DESPUÉS: el oído ya ve la pausa del panel desde
        # su primer chunk (antes podía aceptar un wake con la escucha pausada)
        # y lo primero que se publica es el estado inicial, no un wake que se
        # le adelantó.
        self.start()
        codigo = 0
        try:
            while not self.shutdown.is_set():
                watch = self._esperar(watch)
                # Canal de control (panel -> loop). set_enabled solo publica si
                # el valor cambia, y _poll_control solo relee el JSON si el
                # archivo cambió. Sin thread extra: reutilizamos esta espera.
                if self.control_path is not None:
                    self._poll_control()
                # Vigilar a los hilos en la misma espera (V2): con uno muerto (el
                # oído que se rindió ante el micro, la boca...) el loop seguía
                # vivo, sordo o mudo, y systemd lo veía sano. Los muertos se miran
                # ANTES que shutdown: un hilo que salió por el apagado no cuenta.
                muertos = [t.name for t in self.threads
                           if t in self._terminados or not t.is_alive()]
                if muertos and not self.shutdown.is_set():
                    log.error("murió el hilo %s del loop de voz: me apago",
                              ", ".join(muertos))
                    codigo = 1
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if watch is not None:
                self._cerrar_vigia(watch)
            self.stop()
        return codigo

    def stop(self) -> None:
        """Apagado ordenado con un límite conocido (B1): si el EarThread está
        BLOQUEADO dentro de mic.read() (PortAudio sin timeout), el join expira
        a los 2s y el thread queda vivo; al ser daemon, muere con el proceso
        (el atajo de teclado usa os._exit justamente por esto).

        El mic se cierra DESPUÉS de esperar al oído (V17): PortAudio no admite
        cerrar un stream mientras otro hilo está dentro de read() sobre él
        (comportamiento indefinido: error en el oído o un crash nativo). El
        oído sale de read() en un frame (~32 ms), ve shutdown y termina; solo
        si sigue dentro al expirar el join (lectura colgada) se cierra igual
        desde aquí, como último recurso para desbloquearla.
        """
        self.shutdown.set()                       # 1) avisar a todos
        try:
            self.tts.stop()                       # 2) cortar audio en curso
        finally:
            # 3) devolver la música ANTES de esperar a los hilos: el join del
            # oído puede tardar hasta 2s (PortAudio) y no hay razón para que
            # Spotify siga bajito mientras tanto. close() primero (procesa lo
            # ya encolado por on_mode, en orden, y deja de aceptar más) y
            # restore() después, síncrono: garantía final pase lo que pase.
            if self.ducker is not None:
                self.ducker.close()
                self.ducker.restore()
        self._ear.join(timeout=2.0)               # 4) el oído suelta el mic...
        if hasattr(self._mic, "close"):
            self._mic.close()                     # 5) ...y recién entonces se cierra
        for t in self.threads:                    # 6) esperar (todos miran shutdown)
            if t is not self._ear:
                t.join(timeout=2.0)
