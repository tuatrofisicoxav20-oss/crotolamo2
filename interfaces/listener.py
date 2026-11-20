"""Bucle de voz wake-word sobre el agente nuevo. Reemplaza C1::listener.main.

Flujo: escucha ambiente -> wake.is_wake_word -> graba la orden (VAD) -> stt ->
agent.handle_turn -> tts. Confirmación por voz para tools inseguras.

Requiere la extra [voice] (faster-whisper, sounddevice, piper). Sin ella, da un
mensaje claro en personaje en vez de reventar.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any, Callable

from crotolamo.logging_setup import get_logger
from crotolamo.settings import Settings, get_settings
from crotolamo.voice import wake
from crotolamo.voice.capture import MicFailed, MicSession
from crotolamo.voice.media_aware import (
    Ducker,
    MediaBackend,
    MediaMonitor,
    PlayerctlBackend,
    sane_media_threshold,
)
from crotolamo.voice.state import (
    CONTROL_PATH,
    HUD_STATE_PATH,
    Mode,
    SharedState,
    make_file_publisher,
    read_control_enabled,
)
from crotolamo.voice.stt import STT, VoiceUnavailable, warm_vad
from crotolamo.voice.threads import WakePreconnect
from crotolamo.voice.tts import TTS, StreamSpeaker, _AvisoEnFraseAparte
from crotolamo.voice.vad import DEFAULTS
from crotolamo.voice.wakeword import WakeWordDetector

from interfaces.shell import build_agent

log = get_logger("listener")

# Ruta canónica del archivo de estado para el HUD. Se define UNA vez en
# crotolamo/voice/state.py, junto a CONTROL_PATH (el otro lado del contrato con
# desktop/common/ipc.py); aquí solo se toma, para que los tests la redirijan.
_HUD_STATE_PATH = HUD_STATE_PATH

# Cadencia del sondeo del canal de control (panel -> loop) en el modo simple:
# la misma del loop concurrente cuando no tiene inotify (loop._CONTROL_POLL_S).
_CONTROL_POLL_S = 0.3

# Edad mínima para barrer un temporal de la escritura atómica (ver
# _sweep_stale_tmp): una publicación tarda milisegundos, así que uno más viejo
# que esto es un huérfano; uno reciente puede ser una escritura EN CURSO.
_TMP_STALE_S = 60.0

# Lo que se dice cuando un turno revienta en el loop simple: la misma frase que
# el BrainThread del loop concurrente (crotolamo/voice/threads.py) en el mismo caso.
_TURNO_ROTO = "Se me trabaron los cables, patrón. Vuelve a intentarlo."

# Banda bajo el umbral del wake difuso que todavía se loguea a INFO (ver
# _log_heard): los mishears reales de "crotolamo" medidos dan 0.57-0.67 y las
# cabezas de la tele 0.13-0.50; con el umbral de fábrica (0.72) entran desde 0.55.
_CASI_WAKE = 0.17

# Espera máxima, al salir run_listen, a la carga de modelo que el calentador
# (WarmVoice) tenga en curso. Lo que quede en cola ya no se carga; esto solo evita
# que el intérprete se apague con una carga nativa a medias, sin colgar la salida.
_WARM_STOP_S = 2.0


@dataclass
class ListenerConfig:
    """Ajustes del bucle de voz, leídos UNA vez de la config.

    Antes vivían como una maraña de variables locales al principio de
    run_listen; agrupados aquí se pueden pasar enteros al loop simple. Los
    valores de fábrica que comparte con stt/threads/loop salen de la tabla
    DEFAULTS (crotolamo/voice/vad.py).
    """

    # Wake word.
    threshold: float = 0.67
    variants: Any = None
    wake_silence_ms: int = 400
    wake_max_s: float = 8.0
    # Grabación del comando (VAD / endpointing inteligente).
    silence_ms: int = DEFAULTS.silence_ms_listener
    smart_endpoint: bool = True
    endpoint_rounds: int = DEFAULTS.endpoint_rounds
    endpoint_continue_s: float = DEFAULTS.endpoint_continue_s
    followup_s: float = 6.0
    hotwords: str | None = None
    # Velocidad percibida.
    ack: str = "beep"  # "beep" | "voz" | "off"
    stream_speak: bool = True
    # El loop concurrente habla en streaming (BrainThread): clave propia, apagada.
    stream_speak_concurrent: bool = False
    # Wake con música sonando (ver crotolamo/voice/media_aware.py).
    threshold_media: float = 0.85
    media_poll_s: float = 1.5
    media_watch: str = "follow"  # "follow" (eventos) | "poll" (el sondeo de antes)
    duck: bool = True
    duck_volume: float = 0.2
    # Captura (ver crotolamo/voice/capture.py y _abrir_sesion_mic).
    audio_session: bool = False
    mic_give_up_s: float = 90.0
    # Constantes de los loops: no son clave de la config (antes eran literales
    # sueltos en cada loop); cambiar una es cambiar esta línea.
    command_start_timeout_s: float = DEFAULTS.command_start_timeout_s  # espera de la orden
    wake_window_s: float = 6  # ventana del wake difuso esperando voz
    min_audio_s: float = 0.3  # una grabación más corta no pasa por Whisper (simple)
    confirm_max_s: float = 5  # tope de la respuesta de voice_confirm
    post_reply_guard_s: float = 0.5  # guarda tras responder, antes del followup (simple)
    post_speak_grace_s: float = DEFAULTS.post_speak_grace_s  # oído concurrente tras un turno
    # Tope de una orden: el oído concurrente lo recibe de aquí; el loop simple usa
    # el de STT (el mismo de la tabla) para no cambiar sus llamadas.
    max_command_s: float = DEFAULTS.max_command_s

    @classmethod
    def from_settings(cls, settings: Settings) -> "ListenerConfig":
        voice = settings.voice
        threshold = settings.wake.get("threshold", 0.67)
        return cls(
            threshold=threshold,
            variants=settings.wake.get("variants"),
            # Con música en Playing el wake difuso exige más (la música mete
            # falsos despertares) y el reproductor se atenúa mientras dura la
            # interacción. media_poll_s: cadencia del sondeo de respaldo;
            # media_watch: eventos de playerctl --follow o el sondeo de antes.
            threshold_media=sane_media_threshold(
                threshold, voice.get("wake_threshold_media", 0.85), "[voice].wake_threshold_media",
            ) or threshold,
            media_poll_s=voice.get("media_poll_s", 1.5),
            media_watch=voice.get("media_watch", "follow"),
            duck=voice.get("wake_duck", True),
            duck_volume=voice.get("wake_duck_volume", 0.2),
            wake_silence_ms=voice.get("wake_silence_ms", 400),
            wake_max_s=voice.get("wake_max_seconds", 8.0),
            silence_ms=voice.get("vad_silence_ms", DEFAULTS.silence_ms_listener),
            # Endpointing inteligente: si la frase parece a medias (pausa de
            # pensar), reabrir la escucha y concatenar.
            smart_endpoint=voice.get("smart_endpoint", True),
            endpoint_rounds=voice.get("endpoint_max_rounds", DEFAULTS.endpoint_rounds),
            endpoint_continue_s=voice.get("endpoint_continue_s", DEFAULTS.endpoint_continue_s),
            # followup_s > 0 = tras responder, seguir escuchando esa ventana
            # SIN exigir "crotolamo" de nuevo (conversación).
            followup_s=voice.get("followup_s", 6.0),
            # Hotwords solo para COMANDOS (post-wake), igual que el loop concurrente.
            # Sin la clave, este loop no sesga y el concurrente sí (ver
            # DEFAULTS.hotwords_concurrent): diferencia de hoy, decisión K.
            hotwords=voice.get("hotwords") or None,
            # ack = acuse tras el wake ("beep" instantáneo, "voz" habla "Te
            # escucho" ~1s, "off" nada). stream_speak habla cada frase conforme
            # el LLM la genera. El wake usa su propia ventana de silencio (una
            # palabra corta no necesita los ~600ms de los comandos) y un tope
            # de grabación chico.
            ack=voice.get("ack", "beep"),
            stream_speak=voice.get("stream_speak", True),
            # stream_speak es solo del loop simple (de fábrica true); el
            # concurrente habla en streaming solo con su propia clave.
            stream_speak_concurrent=bool(voice.get("stream_speak_concurrent", False)),
            # audio_session: el micrófono abierto toda la sesión de escucha (false
            # = un stream por grabación, lo de siempre). mic_give_up_s: con la
            # sesión y bajo systemd, cuánto fallo seguido aguanta antes de salir.
            audio_session=bool(voice.get("audio_session", False)),
            mic_give_up_s=voice.get("mic_give_up_s", 90.0),
        )


def _write_idle_hud(path: Path = _HUD_STATE_PATH) -> None:
    """Escribe un estado idle final en el archivo HUD (usado al salir).

    Se llama en el manejador de señal (os._exit) y en el finally de run_listen;
    debe ser defensiva y no propagar nunca.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        state = {
            "mode": "idle",
            "turn_id": 0,
            "text": "",
            # Mismo esquema que SharedState._publish: sin `enabled` el panel
            # caía al default (activa) aunque la escucha estuviera pausada.
            "enabled": read_control_enabled(),
            "ts": time.time(),
            "pid": os.getpid(),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


class _HudGate:
    """Publicador de hud_state.json con compuerta de apagado.

    El idle final tiene que ser lo ÚLTIMO que se escribe. Sin compuerta, otro
    hilo (la Boca arrancando la siguiente frase, el Cerebro) podía publicar
    DESPUÉS del idle —o tener su os.replace a medias cuando se escribía— y el
    HUD, que no sabe si el proceso sigue vivo, se quedaba en "HABLANDO" hasta
    el siguiente arranque. close_with_idle() cierra el paso, espera a la
    publicación en curso y solo entonces escribe el idle; lo que llegue después
    se descarta.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._write = make_file_publisher(self.path)
        # RLock y no Lock: el manejador de señal corre en el hilo principal, que
        # pudo ser interrumpido A MEDIA publicación (en el loop simple publica
        # él) o dentro de otro manejador (dos Ctrl-C seguidos). Con un Lock
        # normal el cierre esperaría un candado que tiene su propio hilo.
        self._lock = threading.RLock()
        self._closed = False

    def publish(self, state_dict: dict) -> None:
        with self._lock:
            if not self._closed:
                self._write(state_dict)

    def close_with_idle(self, lock_timeout_s: float = 1.0) -> None:
        """Cierra la compuerta y escribe el idle final. Nunca propaga.

        La espera del candado está acotada (como Ducker.emergency_restore): si
        un hilo se quedó colgado a media escritura, con el proceso saliendo es
        mejor escribir el idle sin el candado que no escribirlo.
        """
        self._closed = True
        acquired = self._lock.acquire(timeout=lock_timeout_s)
        try:
            _write_idle_hud(self.path)
        finally:
            if acquired:
                self._lock.release()


# Compuerta viva (si la hay), para que el apagado por señal cierre las
# publicaciones antes del idle final. Ver _graceful_exit.
_ACTIVE_HUD_GATE: _HudGate | None = None


def _final_idle_best_effort() -> None:
    """Idle final desde el manejador de señal: por la compuerta si el loop ya
    publica; si la señal llegó antes (cargando modelos), el idle a secas."""
    gate = _ACTIVE_HUD_GATE
    try:
        if gate is not None:
            gate.close_with_idle()
        else:
            _write_idle_hud()
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


def _sweep_stale_tmp(directory: Path, older_than_s: float = _TMP_STALE_S) -> int:
    """Borra los temporales huérfanos de la escritura atómica; devuelve cuántos.

    os._exit a media publicación (entre mkstemp y os.replace) deja un
    tmpXXXXXXXX.tmp en ~/.crotolamo que nadie limpiaba. Solo se tocan archivos
    con el patrón de mkstemp(suffix=".tmp") y más viejos que `older_than_s`:
    uno reciente puede ser una escritura en curso del panel (control.json usa
    el mismo esquema). Best-effort: jamás lanza.
    """
    removed = 0
    try:
        limit = time.time() - older_than_s
        for tmp in Path(directory).glob("tmp*.tmp"):
            try:
                if tmp.is_file() and tmp.stat().st_mtime < limit:
                    tmp.unlink()
                    removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


def _start_control_watcher(
    hud_state: SharedState,
    control_path: Path,
    poll_s: float = _CONTROL_POLL_S,
) -> Callable[[], None]:
    """Vigía del canal de control para el loop SIMPLE; devuelve su stop().

    Ahí el hilo principal se queda BLOQUEADO esperando el wake (openWakeWord
    con timeout None, o la ventana de Whisper del difuso) y no puede sondear
    control.json: sin esto, hud_state.json seguía con enabled=true hasta el
    siguiente wake y el switch "Escuchar por voz" del panel se revertía solo.
    El vigía solo PUBLICA el flag; quien decide no escuchar es el loop.

    Igual que VoiceLoop._poll_control: el JSON se relee solo si cambió el mtime,
    y archivo ausente o ilegible => escucha activa (default-seguro).
    """
    stop = threading.Event()

    def _watch() -> None:
        last_mtime: float | None = None
        while not stop.wait(poll_s):
            try:
                mtime = os.stat(control_path).st_mtime
            except OSError:
                last_mtime = None
                hud_state.set_enabled(True)
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                hud_state.set_enabled(read_control_enabled(control_path))

    thread = threading.Thread(target=_watch, name="ControlWatch", daemon=True)
    thread.start()

    def _stop() -> None:
        stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=1.0)

    return _stop


# Ducker vivo (si lo hay), para que el apagado por señal devuelva la música a
# su volumen: os._exit se salta los finally que la restaurarían. Ver _graceful_exit.
_ACTIVE_DUCKER: Ducker | None = None
# MediaMonitor vivo: su `playerctl --follow` es un hijo de larga vida que
# os._exit no cerraría. Ver _graceful_exit.
_ACTIVE_MEDIA: MediaMonitor | None = None


def _restore_media_best_effort() -> None:
    """Repone el volumen de la música desde el manejador de señal (best-effort).

    Igual que _write_idle_hud: defensiva, acotada en tiempo y jamás propaga. Si
    el hilo principal murió a medio duck con el lock tomado, emergency_restore
    espera un instante y restaura sin él (el proceso está saliendo de todos modos).
    """
    ducker = _ACTIVE_DUCKER
    if ducker is None:
        return
    try:
        ducker.emergency_restore()
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


def _graceful_exit(signum, _frame) -> None:
    """Apagado a prueba de core-dumps cuando corre como servicio (systemd).

    Con los modelos de voz cargados (onnxruntime/torch) y el micrófono abierto por
    PortAudio, el cierre "limpio" del intérprete de Python puede segfaultear (el hilo
    del oído está dentro de una lectura nativa bloqueante). Dejamos que el sistema
    operativo libere mic y audio, y salimos con os._exit(0): salida instantánea,
    código 0 y sin volcado de memoria. Es lo correcto para un Ctrl-C o un stop.

    NOTA: os._exit(0) omite finally/atexit, por eso escribimos el idle HUD AQUÍ,
    antes de salir, en vez de confiar en stop() o un finally exterior. Por la
    misma razón se restaura aquí la música atenuada (ducking): un Ctrl-C a media
    orden no debe dejar Spotify bajito. Y se espera (acotado) la extracción de
    memoria del último turno, que el atexit de la Memoria ya no alcanzaría.

    El idle va AL FINAL, pegado a os._exit: los demás hilos siguen corriendo
    mientras se restaura la música y se cierran los servers MCP, y lo que
    publicaran en ese rato pisaba un idle escrito antes.
    """
    try:
        print("Crotolamo apagado, patrón.", flush=True)
    except Exception:  # noqa: BLE001
        pass
    # El seguidor de playerctl PRIMERO (no espera a nadie): con systemctl stop
    # muere a la vez que nosotros, y el monitor tiene que saberse parado antes
    # de que reponer la música (hasta segundos con D-Bus colgado) lo haga pasar
    # por una caída del seguidor (WARNING falso y un playerctl más al salir).
    _terminate_media_follower_best_effort()
    _restore_media_best_effort()
    _terminate_mcp_best_effort()
    _flush_memoria_best_effort()
    _final_idle_best_effort()
    os._exit(0)


def _flush_memoria_best_effort(timeout_s: float = 2.0) -> None:
    """os._exit se salta el vaciado atexit de la Memoria: el hecho que el patrón
    contó en el último turno (su extracción corre en un hilo de fondo) se perdía
    al parar el servicio. Acotado y sin propagar.

    No construye nada: si nadie usó la Memoria (o su módulo ni se importó) no
    hay nada pendiente. flush() vuelve en el acto con la cola vacía, así que un
    apagado sin extracciones pendientes no tarda más. Va en un hilo aparte con
    join acotado: el manejador corre en el hilo principal, que pudo quedar
    interrumpido DENTRO de un put() a la cola de la Memoria (en el loop simple
    el turno se atiende en él) con el candado de la cola tomado, y un flush()
    ahí mismo lo esperaría para siempre.
    """
    try:
        modulo = sys.modules.get("crotolamo.core.memoria")
        memoria = getattr(modulo, "_MEMORIA", None)
        if memoria is None:
            return
        hilo = threading.Thread(target=memoria.flush, kwargs={"timeout_s": timeout_s},
                                name="MemoriaFlush", daemon=True)
        hilo.start()
        hilo.join(timeout=timeout_s + 0.5)
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


def _terminate_mcp_best_effort() -> None:
    """Los servers MCP (M4) se cierran por atexit, que os._exit se salta: aquí
    se les manda SIGTERM al grupo sin esperar. Sin esto, un wrapper (npx) o un
    server que ignora el EOF de stdin quedaba huérfano al parar el servicio."""
    try:
        from crotolamo.mcp import bridge

        bridge.terminate_all()
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


def _terminate_media_follower_best_effort() -> None:
    """SIGTERM al `playerctl --follow` del MediaMonitor (lo mismo que a los
    servers MCP): os._exit se salta el stop() que lo terminaría."""
    media = _ACTIVE_MEDIA
    if media is None:
        return
    try:
        media.terminate_children()
    except Exception:  # noqa: BLE001 — nunca propagar desde aquí
        pass


def _install_signal_handlers() -> None:
    """SIGINT (Ctrl-C), SIGTERM (systemd stop) y SIGHUP (cerrar la terminal)
    -> apagado limpio e inmediato.

    SIGHUP: el .desktop lanza `kitty --hold -e launch.sh` y cerrar esa ventana
    manda SIGHUP; su acción por defecto mataba el proceso sin _graceful_exit
    (Spotify bajito, HUD en "HABLANDO", servers MCP huérfanos en su propia
    sesión). Salvo que llegue ignorada (`nohup`): eso es pedir sobrevivir al
    cierre de la terminal, y no se pisa.
    """
    sigs = [signal.SIGINT, signal.SIGTERM]
    try:
        if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
            sigs.append(signal.SIGHUP)
    except (ValueError, OSError):
        pass
    for sig in sigs:
        try:
            signal.signal(sig, _graceful_exit)
        except (ValueError, OSError):
            pass  # p.ej. si no estamos en el hilo principal


class _Reintentos:
    """Cuánto espera el loop simple tras un error y si lo vuelve a imprimir.

    Antes cada vuelta fallida era print + sleep(1): con el micrófono caído (el
    stream no abre y revienta al instante) eran 119 líneas a stdout por minuto,
    171.360 al día en el journal, y otras tantas aperturas del dispositivo. Ahora
    la espera crece 1, 2, 4... hasta max_s y cada mensaje DISTINTO se imprime una
    vez por racha. La racha acaba con el primer éxito (ok()) o si el reintento
    anduvo bien más de max_s antes de fallar: eso es otro incidente, no el mismo
    que sigue (openWakeWord por llamada solo vuelve con un wake, así que su
    escucha buena no llega a llamar a ok()).

    El reloj va aparte del `time` del módulo (monotonic importado tal cual): la
    espera se mide aunque time.sleep esté sustituido.
    """

    def __init__(self, min_s: float = 1.0, max_s: float = 30.0,
                 reloj: Callable[[], float] = _monotonic) -> None:
        self.min_s = min_s
        self.max_s = max_s
        self._reloj = reloj
        self._espera = min_s
        self._vistos: set[str] = set()
        self._reintento_en: float | None = None  # cuándo arrancó el último reintento

    def ok(self) -> None:
        self._espera = self.min_s
        self._vistos.clear()
        self._reintento_en = None

    def fallo(self, error: BaseException) -> tuple[bool, bool, float]:
        """Anota un fallo: (primero de la racha, mensaje nuevo, segundos a esperar)."""
        ahora = self._reloj()
        if self._reintento_en is not None and ahora - self._reintento_en >= self.max_s:
            self.ok()
        primero = self._reintento_en is None
        mensaje = str(error)
        nuevo = mensaje not in self._vistos
        self._vistos.add(mensaje)
        espera = self._espera
        self._espera = min(espera * 2, self.max_s)
        self._reintento_en = ahora + espera
        return primero, nuevo, espera


def _avisar_wake(on_wake: Callable[[], None]) -> None:
    """El aviso del wake (on_wake de _run_simple_loop) desde la grabación de la
    orden: un fallo suyo no toca la orden, queda en DEBUG."""
    try:
        on_wake()
    except Exception as error:  # noqa: BLE001 - es una optimización
        log.debug("el aviso del wake falló: %s", error)


def _log_heard(heard: str, threshold: float, variants: Any, activated: bool) -> None:
    """Diagnóstico del wake difuso sin volcar al journal lo que se habla en casa.

    Se logueaba a INFO la transcripción ENTERA de cada ventana con voz: con la
    tele puesta, 12 líneas por minuto (720 por hora) citando lo que se dice en el
    cuarto, aunque la escucha estuviera en pausa. Lo que sirve para afinar
    [wake].variants es cómo transcribe Whisper el "crotolamo" del patrón: a INFO
    van solo las activaciones y los casi-wakes (score de las 3 primeras palabras,
    las que mira split_wake_command, >= umbral - _CASI_WAKE), con su score y solo
    esas 3 palabras. La transcripción entera va a DEBUG, o a INFO como siempre
    con CROTOLAMO_WAKE_DEBUG=1 (validación del micro: el journal de antes).
    """
    if os.environ.get("CROTOLAMO_WAKE_DEBUG", "") not in ("", "0"):
        log.info("wake difuso oyó: %r", heard)
        return
    log.debug("wake difuso oyó: %r", heard)
    head = " ".join(heard.split()[:3])
    score, _ = wake.wake_score(head, variants)
    if activated or score >= threshold - _CASI_WAKE:
        log.info("wake difuso: score=%.2f (umbral=%.2f) cabeza=%r -> %s",
                 score, threshold, head, "ACTIVA" if activated else "no")


@dataclass
class _TurnoSimple:
    """Lo que voice_confirm necesita del loop simple, que se arma DESPUÉS que él
    (build_agent recibe el confirm_fn antes de que el loop exista).

    estado: el SharedState del loop, lo que ve el HUD (None hasta que se crea).
    hablar_pendiente: con stream_speak, dice ENTERO lo que el turno en curso ya
    soltó por on_token (ver _hablar_pendiente); None fuera de un turno así.
    """

    estado: SharedState | None = None
    hablar_pendiente: Callable[[], None] | None = None


def _hablar_pendiente(on_token: _AvisoEnFraseAparte, speaker: StreamSpeaker) -> None:
    """Lo que el turno ya soltó por on_token se habla entero antes de volver: lo
    que on_token retenía, la frase a medio cerrar y la cola del StreamSpeaker."""
    on_token.soltar()
    speaker.drain()


def _run_simple_loop(
    agent: Any,
    stt: STT,
    wake_stt: STT,
    tts: TTS,
    wake_detector: Any,
    use_oww: bool,
    cfg: ListenerConfig,
    hud_state: SharedState,
    say: Callable[[str], None],
    media: MediaMonitor | None = None,
    ducker: Ducker | None = None,
    pregate: _PreFiltro | None = None,
    mic: MicSession | None = None,
    on_wake: Callable[[], None] | None = None,
    turno: _TurnoSimple | None = None,
) -> int:
    """Bucle secuencial clásico (--simple o sin openWakeWord): espera el wake,
    graba la orden, la atiende y vuelve a esperar. Extraído de run_listen para
    que el arranque (settings/agente) y el loop vivan separados.

    media/ducker (media_aware.py) son opcionales: con None el bucle se comporta
    EXACTAMENTE como antes. Con monitor, el wake difuso usa su umbral dinámico
    (más alto con música sonando); con ducker, la música se atenúa durante toda
    la interacción y se restaura pase lo que pase.

    pregate ([wake].pregate, ver _choose_pregate): compuerta de openWakeWord
    delante del wake difuso. None (de fábrica) = la grabación de siempre.

    mic ([voice].audio_session, ver _abrir_sesion_mic): la sesión que tiene el
    micrófono abierto todo el loop; todas las grabaciones leen de ella. None (de
    fábrica) = cada grabación abre y cierra su stream, lo de siempre. Con ella
    se oye lo que antes caía en los huecos entre grabaciones (Whisper
    decodificando el wake o la primera parte de una orden, el atenuado de la
    música); lo que el stream cerrado tampoco oía a propósito (Crotolamo
    hablando, la guarda tras responder, la pausa del panel) se tira o se cierra.

    on_wake (el WakePreconnect del agente, ver _listen_with_warmer): se llama con
    la 1ª voz de cada orden que se GRABA aparte: la que viene detrás de un wake
    aceptado (no la pegada al wake) y la de cada ventana de followup. El motor
    pre-conecta mientras el patrón la dice. Un falso wake (el wake y silencio) o
    una ventana de followup que acaba en silencio no avisan. Debe volver al
    instante; su dueño es quien lo pasa. None = nada, lo de siempre.

    turno (ver _TurnoSimple): el enlace con voice_confirm. En cada turno hablado
    en streaming el loop deja ahí cómo decir lo ya soltado, para que la pregunta
    de confirmación de una tool no se cuele por delante. None = nada.
    """
    # Sin sesión, las grabaciones reciben exactamente los argumentos de siempre.
    fuente: dict[str, Any] = {} if mic is None else {"source": mic}
    # El aviso del wake viaja con la grabación de la orden (on_voice del STT) y
    # sale con su 1ª voz: antes salía con el wake y cada falso wake abría con la
    # nube una conexión (TCP+TLS, vivo 30 s) que ninguna petición usaba. Mismo
    # criterio que el oído concurrente (EarThread). Sin aviso, nada cambia.
    # Lo llevan la orden tras el wake y la de cada ventana de followup.
    voz_de_la_orden: dict[str, Any] = {} if on_wake is None else {
        "on_voice": partial(_avisar_wake, on_wake)}

    def descartar_captura() -> None:
        """Lo capturado hasta aquí no es del patrón (la voz de Crotolamo, la
        guarda tras responder): fuera, como cuando el stream estaba cerrado."""
        if mic is not None:
            mic.flush()

    def listen_command(start_timeout_s: float = cfg.command_start_timeout_s,
                       **aviso: Any) -> str:
        # min_audio_s: si nadie habló, NO se transcribe el silencio (Whisper
        # paddea a 30s; eso costaba ~1s de mic sordo por ventana vacía).
        # aviso: voz_de_la_orden (la orden tras el wake y la del followup).
        if cfg.smart_endpoint:
            return stt.listen_smart(
                silence_ms=cfg.silence_ms, start_timeout_s=start_timeout_s,
                rounds=cfg.endpoint_rounds, continue_timeout_s=cfg.endpoint_continue_s,
                hotwords=cfg.hotwords, min_audio_s=cfg.min_audio_s, **fuente, **aviso,
            )
        # hotwords también aquí: sin endpointing inteligente la orden sigue
        # siendo un COMANDO (post-wake) y lleva el mismo sesgo que en la rama
        # smart y en el loop concurrente.
        return stt.listen_once(silence_ms=cfg.silence_ms, start_timeout_s=start_timeout_s,
                               hotwords=cfg.hotwords, min_audio_s=cfg.min_audio_s, **fuente,
                               **aviso)

    def atender(command: str, on_token: Callable[[str], None] | None = None) -> tuple[str, bool]:
        """agent.handle_turn que no propaga: devuelve (respuesta, reventó).

        Como el BrainThread: un turno roto (un hook, la DB, la memoria; los
        LLMError ya vuelven como respuesta) se DICE en personaje. Antes caía al
        `except` de afuera, que solo imprimía "Error en el listener": el patrón
        daba la orden y oía silencio, y se perdía la ventana de followup.
        """
        try:
            if on_token is None:
                return agent.handle_turn(command), False
            return agent.handle_turn(command, on_token=on_token), False
        except Exception as error:  # noqa: BLE001 - un turno roto no mata el loop
            log.exception("el turno reventó: %s", error)
            return _TURNO_ROTO, True

    def escucha_activa() -> bool:
        """transcribe_if del wake difuso: la pausa del panel que llega A MEDIA
        ventana (hasta 6 s esperando voz + 8 s de grabación) ya no paga Whisper.
        Ese texto se tiraba igual (tras el wake se vuelve a mirar el flag), así
        que no cambia ninguna decisión. Publica la pausa como esa comprobación.
        Con sesión también decide si se sigue esperando el wake en ella (on_idle
        del difuso, should_stop de openWakeWord). Y cierra la ventana de followup,
        que es una orden sin wake."""
        if read_control_enabled():
            return True
        hud_state.set_enabled(False)
        return False

    # El aviso de espera sale UNA vez por periodo de espera (como con
    # openWakeWord, que bloquea hasta el wake). El difuso vuelve al tope del
    # bucle tras cada ventana de 6 s sin activación, y reimprimirlo ahí eran
    # dos líneas a stdout/journald cada 6 s de silencio. Se rearma con lo que
    # cierra la espera: un wake, una pausa del panel o el primer error de una racha.
    espera_anunciada = False
    reintentos = _Reintentos()
    while True:
        try:
            # Escucha pausada desde el panel: se consulta el flag en CADA vuelta,
            # antes de abrir el micrófono. Antes solo se miraba tras un wake: la
            # pausa no se publicaba (el switch del panel se revertía solo) y el
            # wake difuso seguía transcribiendo el ambiente con la escucha
            # "pausada". En pausa no se escucha nada: se duerme y se vuelve a mirar.
            enabled = read_control_enabled()
            hud_state.set_enabled(enabled)
            if not enabled:
                espera_anunciada = False
                # Con sesión, el micrófono se cierra mientras dure la pausa: sin
                # ella, en pausa no se abría nada.
                if mic is not None:
                    mic.suspend()
                time.sleep(_CONTROL_POLL_S)
                continue
            if not espera_anunciada:
                # Espera nueva: lo capturado antes (la respuesta, una pausa, un
                # error) es lo que un stream recién abierto no habría oído.
                if mic is not None:
                    mic.resume()
                    mic.flush()
                print("\nEsperando palabra de activación...", flush=True)
                espera_anunciada = True
                # Espera nueva = hubo un hueco sin escuchar (arranque, turno,
                # pausa, error): el pre-filtro olvida lo de antes en su próxima
                # ventana. Entre ventanas de la misma espera no (ver _PreFiltro).
                if pregate is not None:
                    pregate.hueco()
            try:
                if use_oww:
                    # Detector dedicado: bloquea hasta la activación; no transcribe,
                    # así que el comando se graba SIEMPRE después. Con sesión mira
                    # el panel cada segundo (la pausa cierra el micrófono arriba).
                    espera_oww: dict[str, Any] = {} if mic is None else {
                        "source": mic, "should_stop": lambda: not escucha_activa()}
                    disparo = wake_detector.listen_for_wake(timeout_s=None, **espera_oww)
                    reintentos.ok()
                    if not disparo:
                        continue
                    inline_cmd = ""  # openWakeWord no transcribe: no hay orden pegada
                else:
                    # Fallback difuso: Whisper sobre el ambiente. Silencio corto
                    # (una palabra no necesita 600ms) pero tope amplio: así
                    # "crotolamo pausa la música" DE CORRIDO cabe completo.
                    # min_audio_s: si nadie habló, NO se transcribe el silencio
                    # (eso costaba ~1s de mic sordo por ciclo vacío); con la
                    # escucha pausada mientras grababa, tampoco (escucha_activa).
                    # Con pre-filtro, la ventana solo llega a Whisper si
                    # openWakeWord disparó en ella (ver _choose_pregate); si se
                    # rompió, se quita y el difuso sigue como sin él.
                    # Con sesión, agotados los 6 s sin voz se sigue esperando en
                    # la misma (on_idle, si el panel no pausó) sin perder la
                    # pre-activación, en vez de volver aquí a abrir otra ventana.
                    compuerta = {} if pregate is None or pregate.roto else {"gate": pregate}
                    sesion: dict[str, Any] = {} if mic is None else {
                        "source": mic, "on_idle": escucha_activa}
                    heard = wake_stt.listen_once(silence_ms=cfg.wake_silence_ms,
                                                 max_seconds=cfg.wake_max_s,
                                                 start_timeout_s=cfg.wake_window_s,
                                                 min_audio_s=cfg.min_audio_s,
                                                 transcribe_if=escucha_activa,
                                                 **compuerta, **sesion)
                    reintentos.ok()
                    # Activación: el wake debe venir AL INICIO de la frase (3
                    # primeras palabras) — protege de música/ruido igual que el
                    # viejo guard de "máximo 3 palabras", pero SIN rechazar
                    # "crotolamo <orden>" dicho de corrido: esa orden pegada se
                    # atiende directo, sin segundo turno de escucha.
                    # Umbral dinámico: con música en Playing el monitor pide
                    # más parecido (las letras de canciones alucinan wakes).
                    # Lee un caché: no bloquea.
                    threshold = media.threshold() if media is not None else cfg.threshold
                    activated, inline_cmd = wake.split_wake_command(
                        heard, threshold, cfg.variants
                    )
                    if heard:
                        _log_heard(heard, threshold, cfg.variants, activated)
                    if not activated:
                        continue
            except VoiceUnavailable as error:
                print(str(error))
                return 1
            # Hubo wake: pase lo que pase después, la próxima espera se anuncia.
            espera_anunciada = False

            # Escucha pausada desde el panel MIENTRAS se esperaba el wake: ignora
            # este wake por completo (mismo efecto visible que en el loop
            # concurrente: decir "crotolamo" no hace nada). El poll bloqueante
            # del wake no se puede interrumpir, así que el flag se vuelve a
            # consultar AQUÍ, justo tras dispararse.
            if not read_control_enabled():
                hud_state.set_enabled(False)
                continue
            hud_state.set_enabled(True)

            # TODA la interacción (acuse, escucha de la orden, bucle de
            # followup) va dentro de este try: la música se atenúa al entrar y
            # el finally la restaura en CUALQUIER salida (error, timeout, orden
            # vacía, KeyboardInterrupt). Nunca dejar Spotify bajito.
            try:
                if ducker is not None:
                    ducker.duck()

                # Convocado: el HUD debe APARECER (listening). Si la orden venía
                # pegada al wake, se atiende de inmediato (ni bip hace falta). Si
                # no, acuse según config: "beep" suena SIN bloquear (la grabación
                # abre de inmediato y el patrón puede hablar ya); "voz" es el
                # clásico "Te escucho" (~1s de espera); "off" nada — solo el HUD.
                # Turno nuevo Y a escuchar en una publicación, como el loop
                # concurrente: limpia el texto (la respuesta anterior) que
                # set_mode republicaba bajo "ESCUCHANDO" hasta transcribir la
                # orden nueva.
                hud_state.new_turn(mode=Mode.LISTENING)
                if inline_cmd:
                    command = inline_cmd
                else:
                    if cfg.ack == "voz":
                        say("Te escucho, patrón.")
                        descartar_captura()  # el acuse hablado no es la orden
                    elif cfg.ack != "off":
                        tts.beep()
                        print("Te escucho, patrón.", flush=True)
                    # La orden viene detrás: mientras se dice, el handshake con la
                    # nube sale del turno (voz_de_la_orden). Con la orden pegada al
                    # wake el turno empieza ya: no hay rato que aprovechar.
                    command = listen_command(**voz_de_la_orden)

                if not command.strip():
                    say("No te escuché claro, patrón.")
                    continue  # el finally publica idle -> HUD se oculta

                # Conversación: atender el comando y, si followup_s > 0, seguir
                # escuchando esa ventana sin exigir el wake word otra vez. La
                # ventana se cierra sola si el patrón ya no dice nada (la
                # grabación corta por start_timeout sin voz -> texto vacío).
                while command.strip():
                    print(f"Orden: {command}", flush=True)
                    hud_state.set_text(command)
                    hud_state.set_mode(Mode.THINKING)
                    # stream_speak: habla cada frase EN CUANTO el LLM la cierra,
                    # en vez de esperar la respuesta completa (la primera palabra
                    # suena segundos antes). Fallback al camino clásico si el
                    # streamer no llegó a hablar (una respuesta que no pasó por
                    # on_token) o si el turno reventó (ver atender).
                    if cfg.stream_speak:
                        speaker = StreamSpeaker(
                            tts, on_first=lambda: hud_state.set_mode(Mode.SPEAKING)
                        )
                        on_token = _AvisoEnFraseAparte(speaker.feed)
                        # Para voice_confirm, si este turno pide confirmar una
                        # tool: lo ya soltado se dice antes que la pregunta.
                        if turno is not None:
                            turno.hablar_pendiente = partial(
                                _hablar_pendiente, on_token, speaker)
                        try:
                            reply, roto = atender(command, on_token=on_token)
                            on_token.cerrar(reply)
                            # La respuesta al HUD (y a la consola) en cuanto se
                            # conoce, ANTES de finish(): este espera a que se
                            # hable la cola entera, y publicarla después dejaba
                            # la orden del patrón bajo "HABLANDO" y la respuesta
                            # ya dicha.
                            hud_state.set_text(reply)
                            print(reply, flush=True)
                        finally:
                            if turno is not None:
                                turno.hablar_pendiente = None
                            spoke = speaker.finish()
                        # roto: el aviso nunca pasó por el streamer; lo que este
                        # haya dicho es la respuesta cortada, así que va aparte.
                        if (roto or not spoke) and reply.strip():
                            hud_state.set_mode(Mode.SPEAKING)
                            try:
                                tts.speak_sentences(reply)
                            except Exception as error:  # noqa: BLE001
                                log.warning("voz falló: %s", error)
                    else:
                        reply, _ = atender(command)
                        print(reply, flush=True)
                        hud_state.set_text(reply)
                        hud_state.set_mode(Mode.SPEAKING)
                        try:
                            tts.speak_sentences(reply)  # Fase 6: TTS por frases
                        except Exception as error:  # noqa: BLE001
                            log.warning("voz falló: %s", error)
                    time.sleep(cfg.post_reply_guard_s)
                    # Con sesión: la respuesta y esta guarda quedaron capturadas.
                    descartar_captura()

                    if cfg.followup_s <= 0:
                        break
                    # La ventana de followup es una orden nueva SIN wake, así que la
                    # pausa del panel la bloquea como a un wake: el turno en curso
                    # no se corta, pero no se abre otro. Sin esto, pausar mientras
                    # Crotolamo pensaba o hablaba no paraba nada: la ventana se
                    # abría, lo oído en el cuarto se ejecutaba y cada respuesta
                    # abría otra (la pausa llegaba tras 6 s de silencio).
                    if not escucha_activa():
                        break
                    # Turno nuevo y texto limpio, igual que tras un wake.
                    hud_state.new_turn(mode=Mode.LISTENING)
                    print("(sigo escuchando...)", flush=True)
                    # El followup también pre-conecta con su 1ª voz. El keep-alive
                    # de un stream solo se reutiliza 15 s tras su fin
                    # (HTTPTransport._REUSE_MAX_IDLE_S) y con stream_release="final"
                    # la respuesta entera se habla después: una de 8-13 s más esta
                    # orden pasan de 15 s y el handshake con la nube se pagaba
                    # dentro del turno. Precio: un handshake en segundo plano por
                    # followup aunque el keep-alive siguiera valiendo, el mismo que
                    # tras un wake. La ventana que acaba en silencio no abre nada.
                    command = wake.strip_wake_word(
                        listen_command(start_timeout_s=cfg.followup_s, **voz_de_la_orden)
                    )
                    # La pausa llegó con la ventana ya grabando: lo dicho en ella
                    # tampoco se atiende (como el wake de la espera, ver arriba).
                    if not escucha_activa():
                        break
            finally:
                if ducker is not None:
                    ducker.restore()

        except KeyboardInterrupt:
            say("Crotolamo apagado, patrón.")
            return 0
        except MicFailed as error:
            # La sesión de captura se rindió (solo bajo systemd, ver
            # _abrir_sesion_mic): código 1 y Restart=on-failure reinicia el servicio.
            print(f"Error en el listener: {error}", flush=True)
            return 1
        except Exception as error:  # noqa: BLE001
            primero, nuevo, espera = reintentos.fallo(error)
            if nuevo:
                print(f"Error en el listener: {error}", flush=True)
            if primero:
                espera_anunciada = False
            # Cada error es un hueco sin escuchar para el pre-filtro (antes lo
            # traía el reanuncio de la espera, que ahora es uno por racha).
            if pregate is not None:
                pregate.hueco()
            time.sleep(espera)
            descartar_captura()  # con sesión: lo capturado durante la espera
        finally:
            # idle en TODAS las salidas del cuerpo (continue, fin de turno,
            # excepción) para que el overlay no se quede abierto. El guard evita
            # reescribir idle en cada ciclo de espera ambiente.
            if hud_state.get_mode() != Mode.IDLE:
                hud_state.set_mode(Mode.IDLE)


def make_deny_with_notice(say: Callable[[str], None]) -> Callable[[str], bool]:
    """confirm_fn del modo CONCURRENTE: deniega con aviso, sin abrir micrófono.

    En el modo concurrente el EarThread ya tiene el mic abierto de forma
    persistente; el listen_once de voice_confirm abriría un SEGUNDO
    sd.InputStream sobre el mismo device y chocan. Hasta integrar la
    confirmación al loop del EarThread, una tool delicada se rechaza avisando
    en personaje por dónde sí confirmarla (el shell). La factory recibe SOLO
    say() a propósito: sin STT a la mano, no hay forma de regresar el bug.
    """

    def deny_with_notice(reason: str) -> bool:
        say(reason + " Esa está delicada, patrón: dímela por el shell y ahí la confirmo.")
        return False

    return deny_with_notice


def _make_notice_say(
    say: Callable[[str], None],
    tts: Any,
    get_state: Callable[[], SharedState | None],
    get_brain: Callable[[], Any] | None = None,
) -> Callable[[str], None]:
    """say() del aviso de make_deny_with_notice en el loop CONCURRENTE (V9).

    El aviso se dice desde el Cerebro, DENTRO de agent.handle_turn, y el modo
    seguía en THINKING: el HUD decía "pensando" mientras sonaba y, con
    --barge-in, el oído juzgaba el eco con las reglas de THINKING (sin gracia
    ni margen de umbral), así que el eco del propio aviso lo cortaba y abría un
    turno fantasma. Aquí suena en SPEAKING, como una frase de la Boca, y al
    terminar se vuelve a THINKING: el LLM sigue con el turno (solo si el SPEAKING
    lo puso el aviso; en streaming puede ser de la Boca, ver el finally).

    El aviso es del turno que el Cerebro ATIENDE (`get_brain().turno_en_curso`),
    no del vigente cuando se dice: tras un barge-in el turno viejo termina en
    silencio y, si ya no es el vigente, lo suyo se descarta, como su respuesta.
    Mirar el turno vigente solo lo descartaba mientras el patrón dictaba
    (LISTENING): con la orden nueva ya cerrada (THINKING otra vez, esperando al
    Cerebro) el aviso viejo sonaba entero sobre ella, con el HUD en "hablando".
    `get_state` da el estado del VoiceLoop, que se arma DESPUÉS del agente; sin
    él, say() a secas. Sin Cerebro a la vista, el turno vigente, como antes.
    """

    def _say(text: str) -> None:
        state = get_state()
        if state is None:
            say(text)
            return
        brain = get_brain() if get_brain is not None else None
        turn = getattr(brain, "turno_en_curso", None)
        if turn is None:
            turn = state.turn_id
        # [voice].stream_speak_concurrent: lo que este turno ya le mandó a la Boca
        # se dice ANTES que el aviso, que si no se colaba a media explicación (el
        # turno de voz solo lo pone detrás de la frase que suena). Si esa espera
        # se suelta por un barge-in o por el apagado, ya no hay aviso que decir.
        procede = state.is_current(turn) and (brain is None or brain.hablar_lo_encolado())
        # Turno y modo: solo se pasa a SPEAKING el THINKING de ESTE turno.
        cambio = procede and state.is_current(turn) and state.compare_and_set_mode(
            Mode.THINKING, Mode.SPEAKING)
        if not cambio and not (
                # [voice].stream_speak_concurrent: la Boca ya dijo el principio de
                # ESTE turno y el modo es suyo; el aviso suena detrás, no se tira.
                procede and state.get_mode() is Mode.SPEAKING and state.is_current(turn)):
            log.info("aviso de un turno ya interrumpido, no lo digo: %s", text)
            return
        print(text, flush=True)
        try:
            # El corte va atado al turno: un stop() del barge-in que cae justo
            # antes de que el TTS arranque no se pierde (V4).
            tts.speak(text, should_abort=lambda: not state.is_current(turn))
        except Exception as error:  # noqa: BLE001
            log.warning("voz falló: %s", error)
        finally:
            # Solo si el SPEAKING lo puso este aviso y nadie cambió el modo
            # mientras sonaba: el LISTENING de un barge-in no se pisa. Si era de
            # la Boca (streaming), es suyo: una frase de este turno que esperaba
            # turno de voz detrás del aviso sonaba en THINKING (HUD "pensando",
            # eco juzgado sin gracia) y, si era la última, el END ya no la
            # dejaba en IDLE.
            if cambio:
                state.compare_and_set_mode(Mode.SPEAKING, Mode.THINKING)

    return _say


class _Warmer:
    """El hilo WarmVoice: carga, en orden y de una en una, lo que run_listen encola.

    Antes era una lista fija (Whisper de la orden, Whisper del wake y Piper) que
    arrancaba ANTES de saber el modo: el concurrente cargaba y guardaba todo el
    día un 'tiny' que jamás usa (+64 MB), Piper iba al final aunque el saludo es
    lo primero que suena (el hilo principal esperaba su carga), y Silero no se
    calentaba: caía en el oído (o en la primera grabación) tras el primer wake.
    Ahora run_listen encola Piper de entrada y, decidido el modo, lo que ese loop
    usa en el orden en que lo usa (_warm_plan). Cada carga pasa por el cargador
    thread-safe del componente (TTS._get_voice, STT._get_model, stt.warm_vad): si
    quien lo usa llega antes, espera esa carga en vez de repetirla.

    Un cargador que lanza se avisa con WARNING, como antes de este hilo: sin el
    extra [voice], el journal (nivel INFO de fábrica) dice ya al arrancar qué falta
    y cómo instalarlo, en vez de callarlo hasta el primer wake. Una misma causa (los
    dos Whisper sin faster_whisper) se avisa una vez, como antes.

    Dueño: run_listen. close() = ya no hay más que cargar (el hilo termina al
    vaciar la cola); stop() = salir sin cargar lo que quede, con espera acotada.
    """

    def __init__(self) -> None:
        self._cola: queue.SimpleQueue[tuple[str, Callable[[], object]] | None] = (
            queue.SimpleQueue())
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._run, name="WarmVoice", daemon=True)
        self._hilo.start()

    def add(self, nombre: str, cargar: Callable[[], object]) -> None:
        self._cola.put((nombre, cargar))

    def close(self) -> None:
        self._cola.put(None)

    def join(self, timeout_s: float | None = None) -> None:
        if self._hilo is not threading.current_thread():
            self._hilo.join(timeout_s)

    def stop(self) -> None:
        """Lo que quede en cola ya no se carga; la carga en curso se espera como
        mucho _WARM_STOP_S (no se puede cortar una carga nativa a medias)."""
        self._parar.set()
        self._cola.put(None)
        self.join(_WARM_STOP_S)

    def _run(self) -> None:
        cargados: list[str] = []
        sin_cargar: list[str] = []
        avisados: set[str] = set()
        while True:
            item = self._cola.get()
            if item is None or self._parar.is_set():
                break
            nombre, cargar = item
            try:
                ok = cargar()
            except Exception as error:  # noqa: BLE001 - un cargador roto no para a los demás
                # El texto de antes ('no pude precalentar la voz: <error>') abre el aviso.
                aviso = log.debug if str(error) in avisados else log.warning
                aviso("no pude precalentar la voz: %s (%s)", error, nombre)
                avisados.add(str(error))
                ok = False
            # False = no cargó sin error (sin .onnx de Piper, o warm_vad sin silero-vad)
            # y lo decidirá su primer uso real.
            (sin_cargar if ok is False else cargados).append(nombre)
        if self._parar.is_set():
            log.debug("calentamiento cortado al salir (cargado: %s)", ", ".join(cargados))
            return
        resumen = ", ".join(cargados) or "nada"
        if sin_cargar:
            log.info("voz caliente (%s); sin precalentar (lo decide su primer uso): %s",
                     resumen, ", ".join(sin_cargar))
        else:
            log.info("voz caliente (%s)", resumen)


def _choose_wake(settings: Any, detector: Any, simple: bool) -> tuple[bool, bool]:
    """(use_oww, concurrente): qué wake y qué loop corren. Se decide UNA vez, al
    arrancar y antes de calentar lo demás: de esto dependen el confirm_fn y lo
    que carga _warm_plan.

    M1: wake word dedicado con openWakeWord; si no está, fallback al difuso
    (Whisper). [wake].use_oww = false fuerza el wake DIFUSO, que sí reconoce
    "crotolamo" (vía variants), a costa de más latencia/CPU y sin barge-in; con
    false no se mira nada más (ni se importa ni se carga openWakeWord).
    openWakeWord (true, default) es rápido y fiable pero dispara con "hey jarvis"
    (modelo puente) hasta entrenar un modelo propio de "crotolamo".

    El modelo se carga AQUÍ (can_load), antes de elegir el loop: se cargaba en el
    primer chunk, dentro del hilo del oído, y con un oww_model malo (ruta que no
    existe, pesos que faltan) el asistente decía "Crotolamo escuchando" y el oído
    se rendía al instante: sordo, o reiniciado por systemd sin fin. No es trabajo
    extra: el primer chunk lo pagaba igual. Si no carga, el wake difuso (Whisper)
    no lo necesita. Un detector sin can_load (solo la interfaz de wake) se usa
    tal cual.
    """
    use_oww = bool(settings.wake.get("use_oww", True)) and detector.available()
    can_load = getattr(detector, "can_load", None) if use_oww else None
    if callable(can_load) and not can_load():
        error = getattr(detector, "load_error", None)
        log.warning("openWakeWord no carga (%s); uso el wake difuso", error)
        print(str(error), flush=True)
        use_oww = False
    return use_oww, not simple and use_oww


class _PreFiltro:
    """Compuerta del wake difuso con [wake].pregate='oww': el WakeWordDetector
    que oye cada chunk de la ventana (ver stt._GateRun). Si su rearm() o su
    feed() revientan queda `roto` y el loop deja de pasarlo: el difuso sigue
    como sin pre-filtro (el grabador ya avisó con WARNING, una vez, y esa
    ventana fue entera a Whisper)."""

    def __init__(self, detector: Any) -> None:
        self._detector = detector
        self.roto = False
        # La primera ventana también va tras un hueco: lo que tenga el detector
        # es de antes de este loop (recién cargado no hay nada y no cuesta nada).
        self._hueco = True

    def hueco(self) -> None:
        """El loop dejó de escuchar (espera nueva: arranque, wake, pausa del
        panel, error): la próxima ventana olvida lo de antes (ver rearm)."""
        self._hueco = True

    def rearm(self) -> None:
        """Lo pide _GateRun al abrir CADA ventana; solo llega al detector tras
        un hueco. Entre dos ventanas seguidas no lo hay: sin disparo no pasó
        nada por Whisper y la siguiente abre en milisegundos, así que lo que oyó
        el detector es lo último que sonó, como en el oído del loop concurrente.
        Tras un disparo feed() ya reseteó. Vaciarlo en cada ventana costaba en
        openwakeword 0.6 un reset de ~66 ms de CPU (recalcula 4 s de
        embeddings) en 13 de 14 ventanas por minuto de tele, sin quitar nada
        viejo."""
        if not self._hueco:
            return
        self._hueco = False
        try:
            self._detector.rearm()
        except Exception:
            self.roto = True
            raise

    def feed(self, chunk: Any) -> bool:
        try:
            return bool(self._detector.feed(chunk))
        except Exception:
            self.roto = True
            raise


def _choose_pregate(settings: Any, detector: Any, use_oww: bool) -> _PreFiltro | None:
    """[wake].pregate: un detector barato y continuo delante del wake difuso.

    '' (de fábrica): nada, Whisper transcribe cada ventana con voz. 'oww': el
    modelo de [wake].oww_model oye cada chunk de esa ventana; si no dispara, la
    ventana no llega a Whisper, y si dispara se recorta a ~1.3 s antes del
    disparo. Whisper y split_wake_command siguen siendo el verificador. Solo
    sirve con un modelo que reconozca el "crotolamo" del patrón (decisión del
    dueño, ver config); con use_oww=true no hay wake difuso que filtrar. Un
    modelo que no carga se avisa y el difuso sigue sin pre-filtro.
    """
    modo = settings.wake.get("pregate", "") or ""
    if not modo or use_oww:
        return None
    if modo != "oww":
        log.warning("[wake].pregate=%r no existe ('' u 'oww'); sigo sin pre-filtro", modo)
        return None
    if not detector.available():
        log.warning("[wake].pregate='oww' sin openwakeword importable; sigo sin pre-filtro")
        return None
    can_load = getattr(detector, "can_load", None)
    if callable(can_load) and not can_load():
        log.warning("[wake].pregate='oww': el modelo no carga (%s); sigo sin pre-filtro",
                    getattr(detector, "load_error", None))
        return None
    log.info("pre-filtro openWakeWord delante del wake difuso (umbral %.2f)",
             detector.threshold)
    return _PreFiltro(detector)


def _warm_plan(concurrente: bool, use_oww: bool, vad_silero: bool, stt: Any,
               wake_stt: Any) -> list[tuple[str, Callable[[], object]]]:
    """Lo que el loop YA elegido usa, en el orden en que lo usa por primera vez.

    Piper no está: va antes, en todos los modos (el saludo es lo primero que
    suena). El modelo de openWakeWord tampoco: lo carga el hilo principal al
    decidir el modo (si no carga, se cae al difuso; ver run_listen).
    - concurrente: el oído usa Silero tras el wake (SIEMPRE: _SileroVad no mira
      vad_backend) y el hilo Stt, el Whisper de la orden.
    - simple con openWakeWord: la grabación de la orden (Silero si vad_backend es
      'silero') y el Whisper de la orden, que también transcribe la confirmación
      por voz (C5). El 'tiny' del wake no lo usa nadie: no se carga.
    - difuso: la grabación del wake (Silero si 'silero') y los dos Whisper (del
      mismo tamaño, una sola carga, como siempre). El de la orden va ANTES que el
      del wake, como antes: con 'tiny' primero el proceso queda con ~40-50 MB más
      de RSS para siempre (medido: mismos modelos, otro orden), y los dos están
      cargados mucho antes de que acabe el saludo, así que el orden no retrasa
      el primer wake.
    Whisper va por su cargador (_get_model), no por STT.warm(): su error (falta
    faster_whisper, con la instrucción para instalarlo) llega así a la frontera de
    _Warmer, que lo avisa con WARNING como antes; no se memoriza, así que el primer
    uso real lo reintenta y da el mismo error. Silero sí va por warm_vad: su primer
    uso decide caer a energía y lo avisa una vez (calentar no añade otro aviso).
    Los cargadores se resuelven ya en el hilo del calentador: un STT sin ellos no
    tumba el arranque, lo avisa la frontera de _Warmer.
    """
    plan: list[tuple[str, Callable[[], object]]] = []
    if concurrente or vad_silero:
        plan.append(("silero", warm_vad))
    plan.append((f"whisper:{stt.model_size}", lambda: stt._get_model() is not None))
    if not use_oww and wake_stt.model_size != stt.model_size:
        plan.append((f"whisper:{wake_stt.model_size}",
                     lambda: wake_stt._get_model() is not None))
    return plan


def _abrir_sesion_mic(settings: Any, cfg: ListenerConfig) -> MicSession | None:
    """[voice].audio_session: UN dueño del micrófono para todo el loop (ver
    crotolamo/voice/capture.py) en vez de un stream por grabación.

    None = lo de siempre: la clave en false (de fábrica), o la sesión no abre el
    micrófono; eso se avisa UNA vez y el loop sigue con un stream por grabación
    (que dará su error de siempre si el micrófono de verdad no está).

    La rendición tras mic_give_up_s de fallos seguidos (MicFailed: el loop sale
    con código 1) solo corre como servicio (INVOCATION_ID, lo pone systemd): ahí
    Restart=on-failure reinicia al listener. En una terminal no hay quien lo
    reinicie, así que se reintenta siempre (cada 30 s como mucho), como hoy.
    """
    if not cfg.audio_session:
        return None
    rendirse_s = float(cfg.mic_give_up_s) if os.environ.get("INVOCATION_ID") else 0.0
    voice = settings.voice
    sesion = MicSession(voice.get("sample_rate", DEFAULTS.sample_rate), voice.get("input_device"),
                        give_up_s=rendirse_s)
    try:
        sesion.start()
    except Exception as error:  # noqa: BLE001 - sin sesión, el camino de siempre
        log.warning("[voice].audio_session: no pude abrir el micrófono para toda la "
                    "sesión (%s); grabo con un stream por grabación, como antes", error)
        sesion.close()
        return None
    log.info("micrófono abierto para toda la sesión de escucha")
    return sesion


def _llm_necesita_calentarse(llm: Any) -> bool:
    """needs_warmup del cliente (LLMClient: True, Ollama carga el modelo en RAM;
    GLMClient: False, la nube no carga nada). Un cliente que no lo declare sigue
    la regla de siempre: se calienta solo el local (LLMClient)."""
    necesita = getattr(llm, "needs_warmup", None)
    if necesita is not None:
        return bool(necesita)
    from crotolamo.core.llm import LLMClient

    return isinstance(llm, LLMClient)


def _warm_llm(settings: Any) -> None:
    """Calienta el LLM primario si es un modelo local. Corre en el hilo WarmLLM y
    nunca propaga.

    El PRIMER comando con Ollama arranca el modelo en frío (~60-120s en CPU) y
    deja la voz "pensando" un buen rato: una llamada dummy al arrancar lo deja
    residente (keep_alive lo mantiene). Con GLM no hay nada que cargar: era una
    completion real a la API por arranque, en un cliente desechable cuyo socket
    no reutiliza nadie (el HTTPTransport es por hilo). build_primary_llm sigue
    siendo la única regla de cuál es el primario: glm sin API key es Ollama, y
    ese sí se calienta.

    Cliente PROPIO, no agent.llm: el del loop puede ser FallbackLLM y compartir
    su HTTPTransport entre este hilo y BrainThread corrompía el socket; además,
    un fallo del warm-up abriría el circuit breaker 60s.
    """
    try:
        from crotolamo.core.engine import build_primary_llm

        llm = build_primary_llm(settings)
        if not _llm_necesita_calentarse(llm):
            log.info("cerebro en la nube: no hay LLM que calentar")
            return
        llm.chat([{"role": "user", "content": "di solo: ok"}])
        log.info("LLM caliente, patrón.")
    except Exception as error:  # noqa: BLE001
        log.warning("no pude calentar el LLM: %s", error)


def run_listen(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    # Temporales que dejó un apagado anterior a media publicación del HUD.
    _sweep_stale_tmp(_HUD_STATE_PATH.parent)
    settings = get_settings()
    cfg = ListenerConfig.from_settings(settings)
    # M3: por defecto, loop concurrente. --simple = modo secuencial viejo (fallback).
    # Barge-in conservador: half-duplex por defecto; --barge-in lo activa (auriculares).
    # Se lee lo primero: el modo decide qué modelos se calientan y en qué orden.
    argv = argv or []
    simple = "--simple" in argv
    allow_barge_in = ("--barge-in" in argv) and ("--no-barge-in" not in argv)

    tts = TTS.from_settings(settings)
    # Precalentar la VOZ en segundo plano (hilo WarmVoice, ver _Warmer): sin esto
    # la primera orden pagaba la carga de los modelos (sorda al transcribir, muda
    # al responder). Piper va PRIMERO en todos los modos y se encola antes de
    # decidir el modo: el saludo es lo primero que suena, y así su carga va a la
    # vez que la de openWakeWord en este hilo, en vez de detrás de dos Whisper.
    warmer = _Warmer()
    # UN solo dueño para el calentador: salga run_listen como salga (su return, un
    # error al decidir el modo o al armar el monitor de música, el import del loop,
    # un KeyboardInterrupt en build_agent o en el saludo, SystemExit), lo que quede
    # en cola ya no se carga y la carga en curso se espera, acotado. Un intérprete
    # que se apaga con una carga nativa a medias (la sesión ONNX de Piper o de
    # Silero) aborta (SIGABRT, rc 134, con core) en vez de salir con su código. Va
    # después del finally de cada loop, que ya publicó el idle del HUD: su espera
    # no lo retrasa.
    try:
        # Piper por su cargador, como Whisper (ver _warm_plan): un .onnx que no abre
        # se avisa al arrancar. Sin .onnx no hay voz ni nada que cargar (False).
        warmer.add("piper", lambda: tts.available() and tts._get_voice() is not None)
        return _listen_with_warmer(settings, cfg, tts, warmer, simple=simple,
                                   allow_barge_in=allow_barge_in)
    finally:
        warmer.stop()
        # Dueño también del hilo de pre-síntesis del TTS ([voice].tts_presynth),
        # si llegó a arrancar. Un TTS falso sin close() no tiene nada que parar.
        cerrar_tts = getattr(tts, "close", None)
        if callable(cerrar_tts):
            cerrar_tts()


def _listen_with_warmer(settings: Any, cfg: ListenerConfig, tts: TTS, warmer: _Warmer, *,
                        simple: bool, allow_barge_in: bool) -> int:
    """El resto de run_listen, con el calentamiento ya en marcha (Piper en cola);
    run_listen es el dueño del calentador y lo para al salir."""
    wake_detector = WakeWordDetector.from_settings(settings)
    # El modo queda decidido aquí (ver _choose_wake): el confirm_fn (más abajo) y
    # el resto del calentamiento dependen de él.
    use_oww, concurrente = _choose_wake(settings, wake_detector, simple)
    # [wake].pregate (apagado de fábrica): solo con el wake difuso.
    pregate = _choose_pregate(settings, wake_detector, use_oww)

    stt = STT.from_settings(settings)  # 'base' para el comando
    # I3: modelo barato solo para escuchar la palabra de activación.
    # initial_prompt=None es CLAVE: con pista, Whisper la regurgita sobre el
    # silencio (incluye "crotolamo") y dispara falsos wakes. Construirlo no carga
    # nada: su modelo solo se calienta si el wake es el difuso (ver _warm_plan).
    wake_stt = STT(
        model_size=settings.voice.get("wake_whisper_model", "tiny"),
        sample_rate=settings.voice.get("sample_rate", DEFAULTS.sample_rate),
        initial_prompt=None,
    )
    # Lo demás, en el orden en que lo usa el loop elegido. Mismo default que la
    # grabación del modo simple (stt._record_frames): sin clave, energía.
    vad_silero = settings.voice.get("vad_backend", DEFAULTS.vad_backend) == "silero"
    for nombre, cargar in _warm_plan(concurrente, use_oww, vad_silero, stt, wake_stt):
        warmer.add(nombre, cargar)
    warmer.close()

    # Wake consciente de la música (media_aware.py), compartido por los dos
    # loops: un sondeo en segundo plano dice si algo suena (umbral de wake más
    # alto, en el difuso y en openWakeWord) y el ducker atenúa el reproductor
    # mientras Crotolamo escucha y responde. Sin playerctl, todo es no-op.
    global _ACTIVE_DUCKER, _ACTIVE_MEDIA
    media_backends: list[MediaBackend] = [PlayerctlBackend()]
    media = MediaMonitor(media_backends, poll_s=cfg.media_poll_s,
                         threshold_normal=cfg.threshold,
                         threshold_media=cfg.threshold_media,
                         watch=cfg.media_watch)
    ducker = Ducker(media_backends, factor=cfg.duck_volume, enabled=cfg.duck)
    _ACTIVE_DUCKER = ducker  # para el apagado por señal (os._exit)
    _ACTIVE_MEDIA = media
    wake_detector.attach_media(media)
    # El monitor (un playerctl --follow vivo, o un playerctl cada media_poll_s)
    # solo tiene sentido si alguna de las dos defensas está activa; si no, ni se arranca
    # (is_playing() queda en False y todo sigue como sin música).
    oww_media = wake_detector.threshold_media
    media_util = (
        cfg.duck
        or cfg.threshold_media > cfg.threshold
        or (oww_media is not None and oww_media > wake_detector.threshold)
    )
    if media_util:
        media.start()
    else:
        log.info("wake con música desactivado (sin ducking y sin umbral distinto): no sondeo")

    def say(text: str) -> None:
        print(text, flush=True)
        try:
            tts.speak(text)
        except Exception as error:  # noqa: BLE001
            log.warning("voz falló: %s", error)

    # La sesión de captura del loop simple ([voice].audio_session), si la hay: se
    # abre más abajo, tras el saludo, y voice_confirm (que corre dentro de un
    # turno) escucha la respuesta en ella.
    sesion_mic: list[MicSession] = []
    # Y lo demás que voice_confirm necesita de ese loop, que tampoco existe aún:
    # su estado (el HUD) y lo que el turno en curso ya soltó a la voz.
    turno_simple = _TurnoSimple()

    def voice_confirm(reason: str) -> bool:
        # stream_release="sentence": lo que el turno ya soltó se dice ENTERO antes
        # de preguntar. say() solo espera a la frase que suena: la pregunta se
        # colaba a media explicación y el resto del preámbulo sonaba con el
        # micrófono de la respuesta abierto (en altavoces, una respuesta sucia,
        # que vale "no"). De fábrica ("final") no hay nada soltado.
        if turno_simple.hablar_pendiente is not None:
            turno_simple.hablar_pendiente()
        # El HUD acompaña, como con el aviso del loop concurrente (V9): "hablando"
        # mientras suena la pregunta y "escuchando" mientras se graba la
        # respuesta. Se quedaba en "pensando" todo el rato, con el patrón
        # esperado al micrófono. Al acabar, el modo que traía el turno
        # (THINKING, o SPEAKING si el streaming ya había empezado a hablar).
        estado = turno_simple.estado
        previo = None if estado is None else estado.get_mode()
        try:
            if estado is not None:
                estado.set_mode(Mode.SPEAKING)
            say(reason + " Di 'confirmo' o 'cancela', patrón.")
            fuente: dict[str, Any] = {}
            if sesion_mic:
                # La pregunta que acaba de sonar (y lo de antes) no es la respuesta:
                # fuera, como con el stream que antes se abría aquí.
                sesion_mic[0].flush()
                fuente = {"source": sesion_mic[0]}
            if estado is not None:
                estado.set_mode(Mode.LISTENING)
            try:
                # C5: la respuesta la transcribe el modelo de la ORDEN, no el 'tiny'
                # del wake (solo para la palabra de activación): un mishear vale "no"
                # (lo seguro) y el patrón tenía que repetir la orden entera. Mismas
                # ventanas (respuesta de una palabra) y sin hotwords: no es un
                # comando. min_audio_s: si nadie contestó, "" (un "no") sin pasar el
                # silencio por Whisper.
                answer = stt.listen_once(silence_ms=cfg.wake_silence_ms,
                                         max_seconds=cfg.confirm_max_s,
                                         min_audio_s=cfg.min_audio_s, **fuente)
            except (VoiceUnavailable, MicFailed):
                # Sin micrófono no hay "sí": lo seguro. El loop se entera del
                # MicFailed en su siguiente escucha.
                return False
            return wake.confirmation_from_answer(answer)
        finally:
            if estado is not None and previo is not None:
                estado.set_mode(previo)

    # El modo (decidido arriba) va ANTES de build_agent porque el confirm_fn
    # depende de él: concurrente → deny_with_notice (el EarThread ya tiene el mic;
    # un segundo InputStream de voice_confirm chocaría con él); simple → voice_confirm.
    # El aviso concurrente suena con el modo del VoiceLoop (ver _make_notice_say),
    # que todavía no existe: se engancha al crearlo, más abajo.
    voice_loops: list[Any] = []
    notice_say = _make_notice_say(
        say, tts, lambda: voice_loops[0].state if voice_loops else None,
        lambda: getattr(voice_loops[0], "brain", None) if voice_loops else None,
    )
    confirm_fn = make_deny_with_notice(notice_say) if concurrente else voice_confirm
    try:
        agent, _ = build_agent(confirm_fn=confirm_fn)
    except Exception as error:  # noqa: BLE001
        print(f"No pude armar el agente, patrón: {error}")
        media.stop()  # sin dejar vivo su playerctl --follow
        return 1

    # Calentar el LLM en segundo plano (solo si es un modelo local: ver _warm_llm).
    # No bloquea el arranque.
    threading.Thread(target=_warm_llm, args=(settings,), name="WarmLLM", daemon=True).start()

    # HUD: los dos loops publican el estado en tiempo real al archivo canónico,
    # por la compuerta que el apagado cierra antes del idle final (_HudGate).
    global _ACTIVE_HUD_GATE
    hud_gate = _HudGate(_HUD_STATE_PATH)
    _ACTIVE_HUD_GATE = hud_gate  # para el apagado por señal (os._exit)

    if concurrente:
        from crotolamo.voice.loop import VoiceLoop

        modo_bi = "barge-in (auriculares)" if allow_barge_in else "half-duplex"
        say(f"Crotolamo escuchando, patrón. (concurrente, {modo_bi})")
        # [voice].audio_session: el oído lee de la sesión de captura, que ante un
        # fallo del dispositivo lo cierra, avisa una vez por mensaje y lo reabre
        # con espera creciente, sin salir del proceso; si se rinde (solo bajo
        # systemd), el oído recibe MicFailed y el loop sale con 1. Sin ella (de
        # fábrica), el _RealMic de siempre: el oído se rinde a los ~10 fallos
        # seguidos y systemd reinicia el servicio.
        sesion = _abrir_sesion_mic(settings, cfg)
        con_sesion: dict[str, Any] = {} if sesion is None else {"mic": sesion}
        # [voice].stream_speak_concurrent (apagada de fábrica): sin ella el
        # VoiceLoop recibe los argumentos de siempre.
        if cfg.stream_speak_concurrent:
            con_sesion["stream_speak"] = True
        try:
            voice_loop = VoiceLoop(agent, stt, tts, wake_detector,
                                   allow_barge_in=allow_barge_in, silence_ms=cfg.silence_ms,
                                   hud_publisher=hud_gate.publish, control_path=CONTROL_PATH,
                                   ducker=ducker, start_timeout_s=cfg.command_start_timeout_s,
                                   max_command_s=cfg.max_command_s,
                                   post_speak_grace_s=cfg.post_speak_grace_s, **con_sesion)
            voice_loops.append(voice_loop)
            # run() devuelve != 0 si murió un hilo del loop (el oído que se
            # rindió ante un micro que no vuelve): ese código sale del proceso
            # para que systemd (Restart=on-failure) reinicie al listener sordo.
            # Antes se devolvía 0 siempre y el servicio quedaba "inactive", sin
            # reintento. Un run() sin código cuenta como apagado normal.
            codigo = voice_loop.run() or 0
        finally:
            # Escribir idle final al salir del loop (KeyboardInterrupt, stop(), etc.).
            # Si _graceful_exit llegó primero (os._exit), este bloque no ejecuta;
            # el idle ya se escribió desde el manejador de señal. Por la
            # compuerta: stop() hace join con timeout y un hilo rezagado (la Boca
            # a media frase) todavía puede publicar después.
            if sesion is not None:
                sesion.close()  # stop() del loop ya la cerró; esto cubre un fallo antes
            media.stop()
            hud_gate.close_with_idle()
        return codigo

    modo = "openWakeWord" if use_oww else "wake difuso (Whisper)"
    say(f"Crotolamo escuchando, patrón. (modo simple, wake: {modo})")

    # HUD: el modo simple también publica el estado en tiempo real (el bucle
    # concurrente ya lo hacía). Reutilizamos SharedState para garantizar el MISMO
    # esquema JSON que el HUD espera.
    hud_state = SharedState(publisher=hud_gate.publish)
    turno_simple.estado = hud_state
    # Estado inicial de la escucha desde el canal de control (default: activa).
    hud_state.set_enabled(read_control_enabled())
    # Publicar SIEMPRE el estado inicial: pisa un hud_state.json rancio de un
    # proceso anterior que murió a medio turno (set_enabled solo publica si cambia).
    hud_state.publish_now()
    # Mientras el loop está bloqueado esperando el wake, la pausa del panel la
    # publica este vigía (ver _start_control_watcher).
    stop_control_watch = _start_control_watcher(hud_state, CONTROL_PATH)
    # Tras el saludo: lo que el micrófono capture empieza aquí. Sin sesión
    # (audio_session=false o si no abre) el loop recibe los argumentos de siempre.
    sesion = _abrir_sesion_mic(settings, cfg)
    con_sesion = {}
    if sesion is not None:
        sesion_mic.append(sesion)
        con_sesion = {"mic": sesion}
    # El aviso del wake: con un motor que sabe pre-conectar (GLM), el handshake con
    # la nube se paga mientras el patrón dice la orden. Este bloque es su dueño: al
    # salir espera, acotado, a la pre-conexión que siga en curso. Sin un motor así,
    # el loop recibe los argumentos de siempre.
    preconexion = WakePreconnect.del_agente(agent)
    if preconexion is not None:
        con_sesion["on_wake"] = preconexion

    try:
        return _run_simple_loop(
            agent=agent, stt=stt, wake_stt=wake_stt, tts=tts,
            wake_detector=wake_detector, use_oww=use_oww, cfg=cfg,
            hud_state=hud_state, say=say, media=media, ducker=ducker, pregate=pregate,
            turno=turno_simple, **con_sesion,
        )
    finally:
        if preconexion is not None:
            preconexion.join()
        stop_control_watch()
        if sesion is not None:
            sesion.close()
        media.stop()
        hud_gate.close_with_idle()


if __name__ == "__main__":
    from crotolamo.logging_setup import setup_logging

    # Como script (`python interfaces/listener.py --simple`) no se pasa por
    # crotolamo.__main__: run_listen() a secas ignoraba --simple/--barge-in, y sin
    # setup_logging los INFO se perdían y los WARNING salían sin formato.
    setup_logging()
    raise SystemExit(run_listen(sys.argv[1:]))
