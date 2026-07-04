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
from pathlib import Path

from crotolamo.logging_setup import get_logger
from crotolamo.settings import get_settings
from crotolamo.voice import wake
from crotolamo.voice.state import Mode, SharedState, make_file_publisher
from crotolamo.voice.stt import STT, VoiceUnavailable
from crotolamo.voice.tts import TTS, StreamSpeaker
from crotolamo.voice.wakeword import WakeWordDetector

from interfaces.shell import build_agent

log = get_logger("listener")

# Ruta canónica del archivo de estado para el HUD.
_HUD_STATE_PATH = Path.home() / ".crotolamo" / "hud_state.json"


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


def _graceful_exit(signum, _frame) -> None:
    """Apagado a prueba de core-dumps cuando corre como servicio (systemd).

    Con los modelos de voz cargados (onnxruntime/torch) y el micrófono abierto por
    PortAudio, el cierre "limpio" del intérprete de Python puede segfaultear (el hilo
    del oído está dentro de una lectura nativa bloqueante). Dejamos que el sistema
    operativo libere mic y audio, y salimos con os._exit(0): salida instantánea,
    código 0 y sin volcado de memoria. Es lo correcto para un Ctrl-C o un stop.

    NOTA: os._exit(0) omite finally/atexit, por eso escribimos el idle HUD AQUÍ,
    antes de salir, en vez de confiar en stop() o un finally exterior.
    """
    try:
        print("Crotolamo apagado, patrón.", flush=True)
    except Exception:  # noqa: BLE001
        pass
    _write_idle_hud()
    os._exit(0)


def _install_signal_handlers() -> None:
    """SIGINT (Ctrl-C) y SIGTERM (systemd stop) -> apagado limpio e inmediato."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _graceful_exit)
        except (ValueError, OSError):
            pass  # p.ej. si no estamos en el hilo principal


def run_listen(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    settings = get_settings()
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
    threshold = settings.wake.get("threshold", 0.67)
    variants = settings.wake.get("variants")
    silence_ms = settings.voice.get("vad_silence_ms", 800)
    # Endpointing inteligente: si la frase parece a medias (pausa de pensar),
    # reabrir la escucha y concatenar. followup_s > 0 = tras responder, seguir
    # escuchando esa ventana SIN exigir "crotolamo" de nuevo (conversación).
    smart_endpoint = settings.voice.get("smart_endpoint", True)
    endpoint_rounds = settings.voice.get("endpoint_max_rounds", 2)
    endpoint_continue_s = settings.voice.get("endpoint_continue_s", 2.5)
    followup_s = settings.voice.get("followup_s", 6.0)
    # Hotwords solo para COMANDOS (post-wake), igual que el loop concurrente.
    hotwords = settings.voice.get("hotwords") or None
    # Velocidad percibida: ack = acuse tras el wake ("beep" instantáneo, "voz"
    # habla "Te escucho" ~1s, "off" nada). stream_speak habla cada frase
    # conforme el LLM la genera, en vez de esperar la respuesta completa. El
    # wake usa su propia ventana de silencio (una palabra corta no necesita
    # los ~600ms de los comandos) y un tope de grabación chico.
    ack = settings.voice.get("ack", "beep")
    stream_speak = settings.voice.get("stream_speak", True)
    wake_silence_ms = settings.voice.get("wake_silence_ms", 400)
    wake_max_s = settings.voice.get("wake_max_seconds", 3.0)

    def listen_command(start_timeout_s: float = 4.0) -> str:
        if smart_endpoint:
            return stt.listen_smart(
                silence_ms=silence_ms, start_timeout_s=start_timeout_s,
                rounds=endpoint_rounds, continue_timeout_s=endpoint_continue_s,
                hotwords=hotwords,
            )
        return stt.listen_once(silence_ms=silence_ms, start_timeout_s=start_timeout_s)

    # M1: wake word dedicado con openWakeWord; si no está, fallback al difuso (Whisper).
    # [wake].use_oww = false fuerza el wake DIFUSO (Whisper), que sí reconoce
    # "crotolamo" (vía variants), a costa de más latencia/CPU y sin barge-in.
    # openWakeWord (true, default) es rápido y fiable pero dispara con "hey jarvis"
    # (modelo puente) hasta entrenar un modelo propio de "crotolamo".
    wake_detector = WakeWordDetector.from_settings(settings)
    use_oww = settings.wake.get("use_oww", True) and wake_detector.available()

    def say(text: str) -> None:
        print(text, flush=True)
        try:
            tts.speak(text)
        except Exception as error:  # noqa: BLE001
            log.warning("voz falló: %s", error)

    def voice_confirm(reason: str) -> bool:
        say(reason + " Di 'confirmo' o 'cancela', patrón.")
        try:
            answer = wake_stt.listen_once(silence_ms=wake_silence_ms, max_seconds=5)
        except VoiceUnavailable:
            return False
        if wake.contains_any(answer, wake.CANCEL_VARIANTS):
            return False
        return wake.contains_any(answer, wake.CONFIRM_VARIANTS)

    try:
        agent, _ = build_agent(confirm_fn=voice_confirm)
    except Exception as error:  # noqa: BLE001
        print(f"No pude armar el agente, patrón: {error}")
        return 1

    # Calentar el LLM en segundo plano: el PRIMER comando arranca llama en frío
    # (~60-120s en CPU), lo que deja la voz "pensando" un buen rato. Una llamada
    # dummy al arrancar lo deja residente (keep_alive lo mantiene) para que la
    # primera orden responda rápido. No bloquea el arranque.
    def _warm_llm() -> None:
        try:
            agent.llm.chat([{"role": "user", "content": "di solo: ok"}])
            log.info("LLM caliente, patrón.")
        except Exception as error:  # noqa: BLE001
            log.warning("no pude calentar el LLM: %s", error)

    threading.Thread(target=_warm_llm, name="WarmLLM", daemon=True).start()

    # M3: por defecto, loop concurrente. --simple = modo secuencial viejo (fallback).
    # Barge-in conservador: half-duplex por defecto; --barge-in lo activa (auriculares).
    argv = argv or []
    simple = "--simple" in argv
    allow_barge_in = ("--barge-in" in argv) and ("--no-barge-in" not in argv)
    if not simple and use_oww:
        from crotolamo.voice.loop import VoiceLoop

        modo_bi = "barge-in (auriculares)" if allow_barge_in else "half-duplex"
        say(f"Crotolamo escuchando, patrón. (concurrente, {modo_bi})")
        # HUD: activar publicación de estado en tiempo real al archivo canónico.
        hud_publisher = make_file_publisher(_HUD_STATE_PATH)
        try:
            VoiceLoop(agent, stt, tts, wake_detector,
                      allow_barge_in=allow_barge_in, silence_ms=silence_ms,
                      hud_publisher=hud_publisher).run()
        finally:
            # Escribir idle final al salir del loop (KeyboardInterrupt, stop(), etc.).
            # Si _graceful_exit llegó primero (os._exit), este bloque no ejecuta;
            # _write_idle_hud() ya fue llamado desde el manejador de señal.
            _write_idle_hud()
        return 0

    modo = "openWakeWord" if use_oww else "wake difuso (Whisper)"
    say(f"Crotolamo escuchando, patrón. (modo simple, wake: {modo})")

    # HUD: el modo simple también publica el estado en tiempo real (el bucle
    # concurrente ya lo hacía). Reutilizamos SharedState para garantizar el MISMO
    # esquema JSON que el HUD espera.
    hud_state = SharedState(publisher=make_file_publisher(_HUD_STATE_PATH))

    while True:
        try:
            print("\nEsperando palabra de activación...", flush=True)
            try:
                if use_oww:
                    # Detector dedicado: bloquea hasta la activación; no transcribe,
                    # así que el comando se graba SIEMPRE después.
                    if not wake_detector.listen_for_wake(timeout_s=None):
                        continue
                else:
                    # Fallback difuso: Whisper sobre el ambiente. Ventana de
                    # silencio corta y tope chico: "crotolamo" es una palabra,
                    # y cuanto menos audio, más rápida la transcripción.
                    heard = wake_stt.listen_once(silence_ms=wake_silence_ms,
                                                 max_seconds=wake_max_s,
                                                 start_timeout_s=6)
                    # Log de lo que oyó el wake difuso: sirve para AFINAR la lista
                    # de variantes de "crotolamo" según cómo lo transcribe Whisper.
                    if heard:
                        log.info("wake difuso oyó: %r", heard)
                    # Guard anti-alucinación: "crotolamo" es UNA palabra; una frase
                    # larga (4+ palabras) es casi siempre música/ruido transcrito,
                    # no la palabra de activación. La rechazamos de plano.
                    if heard and len(heard.split()) > 3:
                        continue
                    if not heard or not wake.is_wake_word(heard, threshold, variants):
                        continue
            except VoiceUnavailable as error:
                print(str(error))
                return 1

            # Convocado: el HUD debe APARECER (listening). Acuse según config:
            # "beep" suena SIN bloquear (la grabación abre de inmediato y el
            # patrón puede hablar ya); "voz" es el clásico "Te escucho" (~1s
            # de espera); "off" nada — solo el HUD.
            hud_state.set_mode(Mode.LISTENING)
            if ack == "voz":
                say("Te escucho, patrón.")
            elif ack != "off":
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
                if stream_speak:
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

                if followup_s <= 0:
                    break
                hud_state.set_mode(Mode.LISTENING)
                print("(sigo escuchando...)", flush=True)
                command = wake.strip_wake_word(
                    listen_command(start_timeout_s=followup_s)
                )

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


if __name__ == "__main__":
    raise SystemExit(run_listen())
