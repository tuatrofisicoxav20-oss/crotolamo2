"""Bucle de voz wake-word sobre el agente nuevo. Reemplaza C1::listener.main.

Flujo: escucha ambiente -> wake.is_wake_word -> graba la orden (VAD) -> stt ->
agent.handle_turn -> tts. Confirmación por voz para tools inseguras.

Requiere la extra [voice] (faster-whisper, sounddevice, piper). Sin ella, da un
mensaje claro en personaje en vez de reventar.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from crotolamo.logging_setup import get_logger
from crotolamo.settings import Settings, get_settings
from crotolamo.voice import wake
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
from crotolamo.voice.stt import STT, VoiceUnavailable
from crotolamo.voice.tts import TTS, StreamSpeaker
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


@dataclass
class ListenerConfig:
    """Ajustes del bucle de voz, leídos UNA vez de la config.

    Antes vivían como una maraña de variables locales al principio de
    run_listen; agrupados aquí se pueden pasar enteros al loop simple.
    """

    # Wake word.
    threshold: float = 0.67
    variants: Any = None
    wake_silence_ms: int = 400
    wake_max_s: float = 8.0
    # Grabación del comando (VAD / endpointing inteligente).
    silence_ms: int = 800
    smart_endpoint: bool = True
    endpoint_rounds: int = 2
    endpoint_continue_s: float = 2.5
    followup_s: float = 6.0
    hotwords: str | None = None
    # Velocidad percibida.
    ack: str = "beep"  # "beep" | "voz" | "off"
    stream_speak: bool = True
    # Wake con música sonando (ver crotolamo/voice/media_aware.py).
    threshold_media: float = 0.85
    media_poll_s: float = 1.5
    media_watch: str = "follow"  # "follow" (eventos) | "poll" (el sondeo de antes)
    duck: bool = True
    duck_volume: float = 0.2

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
            silence_ms=voice.get("vad_silence_ms", 800),
            # Endpointing inteligente: si la frase parece a medias (pausa de
            # pensar), reabrir la escucha y concatenar.
            smart_endpoint=voice.get("smart_endpoint", True),
            endpoint_rounds=voice.get("endpoint_max_rounds", 2),
            endpoint_continue_s=voice.get("endpoint_continue_s", 2.5),
            # followup_s > 0 = tras responder, seguir escuchando esa ventana
            # SIN exigir "crotolamo" de nuevo (conversación).
            followup_s=voice.get("followup_s", 6.0),
            # Hotwords solo para COMANDOS (post-wake), igual que el loop concurrente.
            hotwords=voice.get("hotwords") or None,
            # ack = acuse tras el wake ("beep" instantáneo, "voz" habla "Te
            # escucho" ~1s, "off" nada). stream_speak habla cada frase conforme
            # el LLM la genera. El wake usa su propia ventana de silencio (una
            # palabra corta no necesita los ~600ms de los comandos) y un tope
            # de grabación chico.
            ack=voice.get("ack", "beep"),
            stream_speak=voice.get("stream_speak", True),
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


class _AvisoEnFraseAparte:
    """on_token del loop simple: el aviso de un LLM que se cae a media respuesta
    se habla como frase PROPIA, no pegado a la frase cortada.

    El agente lo emite como su último token, on_token("\\n" + aviso), y devuelve
    el aviso. StreamSpeaker no corta en un salto de línea suelto (cómo suenan
    los saltos en general es decisión pendiente, voice-runtime#18), así que se
    oía "La capital de Francia es GLM no respondió a tiempo...". Por eso un
    token que empieza por salto se retiene hasta el siguiente, y entonces pasa
    tal cual (la misma segmentación de siempre, un token más tarde), o hasta el
    final del turno: si era "\\n" + la respuesta, va tras un salto doble, que sí
    cierra frase.
    """

    def __init__(self, feed: Callable[[str], None]) -> None:
        self._feed = feed
        self._retenido: str | None = None

    def __call__(self, token: str) -> None:
        retenido, self._retenido = self._retenido, None
        if retenido is not None:
            self._feed(retenido)
        if token.startswith("\n"):
            self._retenido = token
        else:
            self._feed(token)

    def cerrar(self, reply: str) -> None:
        """Fin del turno: suelta lo retenido, como frase aparte si era el aviso."""
        retenido, self._retenido = self._retenido, None
        if retenido is not None:
            self._feed("\n" + retenido if retenido == "\n" + reply else retenido)


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
) -> int:
    """Bucle secuencial clásico (--simple o sin openWakeWord): espera el wake,
    graba la orden, la atiende y vuelve a esperar. Extraído de run_listen para
    que el arranque (settings/agente) y el loop vivan separados.

    media/ducker (media_aware.py) son opcionales: con None el bucle se comporta
    EXACTAMENTE como antes. Con monitor, el wake difuso usa su umbral dinámico
    (más alto con música sonando); con ducker, la música se atenúa durante toda
    la interacción y se restaura pase lo que pase.
    """

    def listen_command(start_timeout_s: float = 4.0) -> str:
        # min_audio_s: si nadie habló, NO se transcribe el silencio (Whisper
        # paddea a 30s; eso costaba ~1s de mic sordo por ventana vacía).
        if cfg.smart_endpoint:
            return stt.listen_smart(
                silence_ms=cfg.silence_ms, start_timeout_s=start_timeout_s,
                rounds=cfg.endpoint_rounds, continue_timeout_s=cfg.endpoint_continue_s,
                hotwords=cfg.hotwords, min_audio_s=0.3,
            )
        # hotwords también aquí: sin endpointing inteligente la orden sigue
        # siendo un COMANDO (post-wake) y lleva el mismo sesgo que en la rama
        # smart y en el loop concurrente.
        return stt.listen_once(silence_ms=cfg.silence_ms, start_timeout_s=start_timeout_s,
                               hotwords=cfg.hotwords, min_audio_s=0.3)

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

    # El aviso de espera sale UNA vez por periodo de espera (como con
    # openWakeWord, que bloquea hasta el wake). El difuso vuelve al tope del
    # bucle tras cada ventana de 6 s sin activación, y reimprimirlo ahí eran
    # dos líneas a stdout/journald cada 6 s de silencio. Se rearma con lo que
    # cierra la espera: un wake, una pausa del panel o un error.
    espera_anunciada = False
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
                time.sleep(_CONTROL_POLL_S)
                continue
            if not espera_anunciada:
                print("\nEsperando palabra de activación...", flush=True)
                espera_anunciada = True
            try:
                if use_oww:
                    # Detector dedicado: bloquea hasta la activación; no transcribe,
                    # así que el comando se graba SIEMPRE después.
                    if not wake_detector.listen_for_wake(timeout_s=None):
                        continue
                    inline_cmd = ""  # openWakeWord no transcribe: no hay orden pegada
                else:
                    # Fallback difuso: Whisper sobre el ambiente. Silencio corto
                    # (una palabra no necesita 600ms) pero tope amplio: así
                    # "crotolamo pausa la música" DE CORRIDO cabe completo.
                    # min_audio_s: si nadie habló, NO se transcribe el silencio
                    # (eso costaba ~1s de mic sordo por ciclo vacío).
                    heard = wake_stt.listen_once(silence_ms=cfg.wake_silence_ms,
                                                 max_seconds=cfg.wake_max_s,
                                                 start_timeout_s=6,
                                                 min_audio_s=0.3)
                    # Log de lo que oyó el wake difuso: sirve para AFINAR la lista
                    # de variantes de "crotolamo" según cómo lo transcribe Whisper.
                    if heard:
                        log.info("wake difuso oyó: %r", heard)
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
                    elif cfg.ack != "off":
                        tts.beep()
                        print("Te escucho, patrón.", flush=True)
                    command = listen_command()

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
                    time.sleep(0.5)

                    if cfg.followup_s <= 0:
                        break
                    # La ventana de followup es una orden nueva (sin wake): turno
                    # nuevo y texto limpio, igual que tras un wake.
                    hud_state.new_turn(mode=Mode.LISTENING)
                    print("(sigo escuchando...)", flush=True)
                    command = wake.strip_wake_word(
                        listen_command(start_timeout_s=cfg.followup_s)
                    )
            finally:
                if ducker is not None:
                    ducker.restore()

        except KeyboardInterrupt:
            say("Crotolamo apagado, patrón.")
            return 0
        except Exception as error:  # noqa: BLE001
            print(f"Error en el listener: {error}", flush=True)
            espera_anunciada = False
            time.sleep(1)
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
) -> Callable[[str], None]:
    """say() del aviso de make_deny_with_notice en el loop CONCURRENTE (V9).

    El aviso se dice desde el Cerebro, DENTRO de agent.handle_turn, y el modo
    seguía en THINKING: el HUD decía "pensando" mientras sonaba y, con
    --barge-in, el oído juzgaba el eco con las reglas de THINKING (sin gracia
    ni margen de umbral), así que el eco del propio aviso lo cortaba y abría un
    turno fantasma. Aquí suena en SPEAKING, como una frase de la Boca, y al
    terminar se vuelve a THINKING: el LLM sigue con el turno.

    Si el modo ya no es THINKING, un barge-in abrió otro turno (el patrón está
    dictando la orden nueva) y lo del turno viejo se descarta, como su
    respuesta: hablar encima metía el aviso en esa grabación. `get_state` da el
    estado del VoiceLoop, que se arma DESPUÉS del agente; sin él, say() a secas.
    """

    def _say(text: str) -> None:
        state = get_state()
        if state is None:
            say(text)
            return
        turn = state.turn_id
        if not state.compare_and_set_mode(Mode.THINKING, Mode.SPEAKING):
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
            # Solo si nadie cambió el modo mientras sonaba: el LISTENING de un
            # barge-in no se pisa.
            state.compare_and_set_mode(Mode.SPEAKING, Mode.THINKING)

    return _say


def run_listen(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    # Temporales que dejó un apagado anterior a media publicación del HUD.
    _sweep_stale_tmp(_HUD_STATE_PATH.parent)
    settings = get_settings()
    cfg = ListenerConfig.from_settings(settings)
    stt = STT.from_settings(settings)  # 'base' para el comando
    # I3: modelo barato solo para escuchar la palabra de activación.
    # initial_prompt=None es CLAVE: con pista, Whisper la regurgita sobre el
    # silencio (incluye "crotolamo") y dispara falsos wakes.
    wake_stt = STT(
        model_size=settings.voice.get("wake_whisper_model", "tiny"),
        sample_rate=settings.voice.get("sample_rate", 16000),
        initial_prompt=None,
    )
    tts = TTS.from_settings(settings)

    # Precalentar la VOZ en segundo plano: Whisper (comando y wake) y Piper se
    # cargaban en el PRIMER uso, así que la primera orden pagaba varios segundos
    # de carga de modelos (sorda al transcribir, muda al responder). Cargarlos
    # al arrancar deja el primer turno igual de rápido que los demás.
    def _warm_voice() -> None:
        try:
            stt._get_model()
            if wake_stt.model_size != stt.model_size:
                wake_stt._get_model()
            if tts.available():
                tts._get_voice()
            log.info("voz caliente (Whisper y Piper cargados)")
        except Exception as error:  # noqa: BLE001 - sin extra [voice], se avisa al usar
            log.warning("no pude precalentar la voz: %s", error)

    threading.Thread(target=_warm_voice, name="WarmVoice", daemon=True).start()

    # M1: wake word dedicado con openWakeWord; si no está, fallback al difuso (Whisper).
    # [wake].use_oww = false fuerza el wake DIFUSO (Whisper), que sí reconoce
    # "crotolamo" (vía variants), a costa de más latencia/CPU y sin barge-in.
    # openWakeWord (true, default) es rápido y fiable pero dispara con "hey jarvis"
    # (modelo puente) hasta entrenar un modelo propio de "crotolamo".
    wake_detector = WakeWordDetector.from_settings(settings)
    use_oww = settings.wake.get("use_oww", True) and wake_detector.available()
    if use_oww:
        # El modelo se carga AQUÍ, antes de elegir el loop: se cargaba en el
        # primer chunk, dentro del hilo del oído, y con un oww_model malo (ruta
        # que no existe, pesos que faltan) el asistente decía "Crotolamo
        # escuchando" y el oído se rendía al instante: sordo, o reiniciado por
        # systemd sin fin. No es trabajo extra: el primer chunk lo pagaba igual.
        # Si no carga, el wake difuso (Whisper) no lo necesita. Un detector sin
        # carga explícita (solo la interfaz de wake) se usa tal cual.
        cargar_modelo = getattr(wake_detector, "_get_model", None)
        if callable(cargar_modelo):
            try:
                cargar_modelo()
            except VoiceUnavailable as error:
                log.warning("openWakeWord no carga (%s); uso el wake difuso", error)
                print(str(error), flush=True)
                use_oww = False

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

    def voice_confirm(reason: str) -> bool:
        say(reason + " Di 'confirmo' o 'cancela', patrón.")
        try:
            # C5: la respuesta la transcribe el modelo de la ORDEN, no el 'tiny'
            # del wake (solo para la palabra de activación): un mishear vale "no"
            # (lo seguro) y el patrón tenía que repetir la orden entera. Mismas
            # ventanas (respuesta de una palabra) y sin hotwords: no es un
            # comando. min_audio_s: si nadie contestó, "" (un "no") sin pasar el
            # silencio por Whisper.
            answer = stt.listen_once(silence_ms=cfg.wake_silence_ms, max_seconds=5,
                                     min_audio_s=0.3)
        except VoiceUnavailable:
            return False
        return wake.confirmation_from_answer(answer)

    # M3: por defecto, loop concurrente. --simple = modo secuencial viejo (fallback).
    # Barge-in conservador: half-duplex por defecto; --barge-in lo activa (auriculares).
    # La decisión va ANTES de build_agent porque el confirm_fn depende del modo:
    # concurrente → deny_with_notice (el EarThread ya tiene el mic; un segundo
    # InputStream de voice_confirm chocaría con él); simple → voice_confirm.
    argv = argv or []
    simple = "--simple" in argv
    allow_barge_in = ("--barge-in" in argv) and ("--no-barge-in" not in argv)
    concurrente = not simple and use_oww

    # El aviso concurrente suena con el modo del VoiceLoop (ver _make_notice_say),
    # que todavía no existe: se engancha al crearlo, más abajo.
    voice_loops: list[Any] = []
    notice_say = _make_notice_say(
        say, tts, lambda: voice_loops[0].state if voice_loops else None
    )
    confirm_fn = make_deny_with_notice(notice_say) if concurrente else voice_confirm
    try:
        agent, _ = build_agent(confirm_fn=confirm_fn)
    except Exception as error:  # noqa: BLE001
        print(f"No pude armar el agente, patrón: {error}")
        media.stop()  # sin dejar vivo su playerctl --follow
        return 1

    # Calentar el LLM en segundo plano: el PRIMER comando arranca llama en frío
    # (~60-120s en CPU), lo que deja la voz "pensando" un buen rato. Una llamada
    # dummy al arrancar lo deja residente (keep_alive lo mantiene) para que la
    # primera orden responda rápido. No bloquea el arranque.
    def _warm_llm() -> None:
        # Cliente PROPIO, no agent.llm: el del loop puede ser FallbackLLM y
        # compartir su HTTPTransport entre este hilo y BrainThread corrompía el
        # socket; además, un fallo del warm-up abriría el circuit breaker 60s.
        try:
            from crotolamo.core.engine import build_primary_llm

            build_primary_llm(settings).chat(
                [{"role": "user", "content": "di solo: ok"}]
            )
            log.info("LLM caliente, patrón.")
        except Exception as error:  # noqa: BLE001
            log.warning("no pude calentar el LLM: %s", error)

    threading.Thread(target=_warm_llm, name="WarmLLM", daemon=True).start()

    # HUD: los dos loops publican el estado en tiempo real al archivo canónico,
    # por la compuerta que el apagado cierra antes del idle final (_HudGate).
    global _ACTIVE_HUD_GATE
    hud_gate = _HudGate(_HUD_STATE_PATH)
    _ACTIVE_HUD_GATE = hud_gate  # para el apagado por señal (os._exit)

    if concurrente:
        from crotolamo.voice.loop import VoiceLoop

        modo_bi = "barge-in (auriculares)" if allow_barge_in else "half-duplex"
        say(f"Crotolamo escuchando, patrón. (concurrente, {modo_bi})")
        try:
            voice_loop = VoiceLoop(agent, stt, tts, wake_detector,
                                   allow_barge_in=allow_barge_in, silence_ms=cfg.silence_ms,
                                   hud_publisher=hud_gate.publish, control_path=CONTROL_PATH,
                                   ducker=ducker)
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
            media.stop()
            hud_gate.close_with_idle()
        return codigo

    modo = "openWakeWord" if use_oww else "wake difuso (Whisper)"
    say(f"Crotolamo escuchando, patrón. (modo simple, wake: {modo})")

    # HUD: el modo simple también publica el estado en tiempo real (el bucle
    # concurrente ya lo hacía). Reutilizamos SharedState para garantizar el MISMO
    # esquema JSON que el HUD espera.
    hud_state = SharedState(publisher=hud_gate.publish)
    # Estado inicial de la escucha desde el canal de control (default: activa).
    hud_state.set_enabled(read_control_enabled())
    # Publicar SIEMPRE el estado inicial: pisa un hud_state.json rancio de un
    # proceso anterior que murió a medio turno (set_enabled solo publica si cambia).
    hud_state.publish_now()
    # Mientras el loop está bloqueado esperando el wake, la pausa del panel la
    # publica este vigía (ver _start_control_watcher).
    stop_control_watch = _start_control_watcher(hud_state, CONTROL_PATH)

    try:
        return _run_simple_loop(
            agent=agent, stt=stt, wake_stt=wake_stt, tts=tts,
            wake_detector=wake_detector, use_oww=use_oww, cfg=cfg,
            hud_state=hud_state, say=say, media=media, ducker=ducker,
        )
    finally:
        stop_control_watch()
        media.stop()
        hud_gate.close_with_idle()


if __name__ == "__main__":
    from crotolamo.logging_setup import setup_logging

    # Como script (`python interfaces/listener.py --simple`) no se pasa por
    # crotolamo.__main__: run_listen() a secas ignoraba --simple/--barge-in, y sin
    # setup_logging los INFO se perdían y los WARNING salían sin formato.
    setup_logging()
    raise SystemExit(run_listen(sys.argv[1:]))
