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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from crotolamo.logging_setup import get_logger
from crotolamo.settings import Settings, get_settings
from crotolamo.voice import wake
from crotolamo.voice.media_aware import Ducker, MediaBackend, MediaMonitor, PlayerctlBackend
from crotolamo.voice.state import (
    CONTROL_PATH,
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

# Ruta canónica del archivo de estado para el HUD.
_HUD_STATE_PATH = Path.home() / ".crotolamo" / "hud_state.json"


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
    duck: bool = True
    duck_volume: float = 0.2

    @classmethod
    def from_settings(cls, settings: Settings) -> "ListenerConfig":
        voice = settings.voice
        return cls(
            threshold=settings.wake.get("threshold", 0.67),
            variants=settings.wake.get("variants"),
            # Con música en Playing el wake difuso exige más (la música mete
            # falsos despertares) y el reproductor se atenúa mientras dura la
            # interacción. media_poll_s: cadencia del sondeo en segundo plano.
            threshold_media=voice.get("wake_threshold_media", 0.85),
            media_poll_s=voice.get("media_poll_s", 1.5),
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


# Ducker vivo (si lo hay), para que el apagado por señal devuelva la música a
# su volumen: os._exit se salta los finally que la restaurarían. Ver _graceful_exit.
_ACTIVE_DUCKER: Ducker | None = None


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
    orden no debe dejar Spotify bajito.
    """
    try:
        print("Crotolamo apagado, patrón.", flush=True)
    except Exception:  # noqa: BLE001
        pass
    _restore_media_best_effort()
    _write_idle_hud()
    os._exit(0)


def _install_signal_handlers() -> None:
    """SIGINT (Ctrl-C) y SIGTERM (systemd stop) -> apagado limpio e inmediato."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _graceful_exit)
        except (ValueError, OSError):
            pass  # p.ej. si no estamos en el hilo principal


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
        return stt.listen_once(silence_ms=cfg.silence_ms, start_timeout_s=start_timeout_s,
                               min_audio_s=0.3)

    while True:
        try:
            print("\nEsperando palabra de activación...", flush=True)
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

            # Escucha pausada desde el panel: ignora este wake por completo (mismo
            # efecto visible que en el loop concurrente: decir "crotolamo" no hace
            # nada). El poll bloqueante del wake no se puede interrumpir, así que
            # consultamos el flag AQUÍ, justo tras dispararse.
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
                hud_state.set_mode(Mode.LISTENING)
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
                    # streamer no llegó a hablar (p.ej. LLMError devuelto como
                    # texto sin pasar por on_token).
                    if cfg.stream_speak:
                        speaker = StreamSpeaker(
                            tts, on_first=lambda: hud_state.set_mode(Mode.SPEAKING)
                        )
                        try:
                            reply = agent.handle_turn(command, on_token=speaker.feed)
                        finally:
                            spoke = speaker.finish()
                        print(reply, flush=True)
                        hud_state.set_text(reply)
                        if not spoke and reply.strip():
                            hud_state.set_mode(Mode.SPEAKING)
                            try:
                                tts.speak_sentences(reply)
                            except Exception as error:  # noqa: BLE001
                                log.warning("voz falló: %s", error)
                    else:
                        reply = agent.handle_turn(command)
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
                    hud_state.set_mode(Mode.LISTENING)
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


def run_listen(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
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

    # Wake consciente de la música (media_aware.py), compartido por los dos
    # loops: un sondeo en segundo plano dice si algo suena (umbral de wake más
    # alto, en el difuso y en openWakeWord) y el ducker atenúa el reproductor
    # mientras Crotolamo escucha y responde. Sin playerctl, todo es no-op.
    global _ACTIVE_DUCKER
    media_backends: list[MediaBackend] = [PlayerctlBackend()]
    media = MediaMonitor(media_backends, poll_s=cfg.media_poll_s,
                         threshold_normal=cfg.threshold,
                         threshold_media=cfg.threshold_media)
    ducker = Ducker(media_backends, factor=cfg.duck_volume, enabled=cfg.duck)
    _ACTIVE_DUCKER = ducker  # para el apagado por señal (os._exit)
    wake_detector.attach_media(media)
    media.start()

    def say(text: str) -> None:
        print(text, flush=True)
        try:
            tts.speak(text)
        except Exception as error:  # noqa: BLE001
            log.warning("voz falló: %s", error)

    def voice_confirm(reason: str) -> bool:
        say(reason + " Di 'confirmo' o 'cancela', patrón.")
        try:
            answer = wake_stt.listen_once(silence_ms=cfg.wake_silence_ms, max_seconds=5)
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

    confirm_fn = make_deny_with_notice(say) if concurrente else voice_confirm
    try:
        agent, _ = build_agent(confirm_fn=confirm_fn)
    except Exception as error:  # noqa: BLE001
        print(f"No pude armar el agente, patrón: {error}")
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

    if concurrente:
        from crotolamo.voice.loop import VoiceLoop

        modo_bi = "barge-in (auriculares)" if allow_barge_in else "half-duplex"
        say(f"Crotolamo escuchando, patrón. (concurrente, {modo_bi})")
        # HUD: activar publicación de estado en tiempo real al archivo canónico.
        hud_publisher = make_file_publisher(_HUD_STATE_PATH)
        try:
            VoiceLoop(agent, stt, tts, wake_detector,
                      allow_barge_in=allow_barge_in, silence_ms=cfg.silence_ms,
                      hud_publisher=hud_publisher, control_path=CONTROL_PATH,
                      ducker=ducker).run()
        finally:
            # Escribir idle final al salir del loop (KeyboardInterrupt, stop(), etc.).
            # Si _graceful_exit llegó primero (os._exit), este bloque no ejecuta;
            # _write_idle_hud() ya fue llamado desde el manejador de señal.
            media.stop()
            _write_idle_hud()
        return 0

    modo = "openWakeWord" if use_oww else "wake difuso (Whisper)"
    say(f"Crotolamo escuchando, patrón. (modo simple, wake: {modo})")

    # HUD: el modo simple también publica el estado en tiempo real (el bucle
    # concurrente ya lo hacía). Reutilizamos SharedState para garantizar el MISMO
    # esquema JSON que el HUD espera.
    hud_state = SharedState(publisher=make_file_publisher(_HUD_STATE_PATH))
    # Estado inicial de la escucha desde el canal de control (default: activa).
    hud_state.set_enabled(read_control_enabled())
    # Publicar SIEMPRE el estado inicial: pisa un hud_state.json rancio de un
    # proceso anterior que murió a medio turno (set_enabled solo publica si cambia).
    hud_state.publish_now()

    try:
        return _run_simple_loop(
            agent=agent, stt=stt, wake_stt=wake_stt, tts=tts,
            wake_detector=wake_detector, use_oww=use_oww, cfg=cfg,
            hud_state=hud_state, say=say, media=media, ducker=ducker,
        )
    finally:
        media.stop()
        _write_idle_hud()


if __name__ == "__main__":
    raise SystemExit(run_listen())
