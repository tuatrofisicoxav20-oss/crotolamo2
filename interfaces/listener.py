"""Bucle de voz wake-word sobre el agente nuevo. Reemplaza C1::listener.main.

Flujo: escucha ambiente -> wake.is_wake_word -> graba la orden (VAD) -> stt ->
agent.handle_turn -> tts. Confirmación por voz para tools inseguras.

Requiere la extra [voice] (faster-whisper, sounddevice, piper). Sin ella, da un
mensaje claro en personaje en vez de reventar.
"""

from __future__ import annotations

import time

from crotolamo.logging_setup import get_logger
from crotolamo.settings import get_settings
from crotolamo.voice import wake
from crotolamo.voice.stt import STT, VoiceUnavailable
from crotolamo.voice.tts import TTS
from crotolamo.voice.wakeword import WakeWordDetector

from interfaces.shell import build_agent

log = get_logger("listener")


def run_listen(argv: list[str] | None = None) -> int:
    settings = get_settings()
    stt = STT.from_settings(settings)  # 'base' para el comando
    # I3: modelo 'tiny' barato solo para escuchar la palabra de activación.
    wake_stt = STT(
        model_size=settings.voice.get("wake_whisper_model", "tiny"),
        sample_rate=settings.voice.get("sample_rate", 16000),
    )
    tts = TTS.from_settings(settings)
    threshold = settings.wake.get("threshold", 0.67)
    variants = settings.wake.get("variants")
    silence_ms = settings.voice.get("vad_silence_ms", 800)

    # M1: wake word dedicado con openWakeWord; si no está, fallback al difuso (Whisper).
    wake_detector = WakeWordDetector.from_settings(settings)
    use_oww = wake_detector.available()

    def say(text: str) -> None:
        print(text, flush=True)
        try:
            tts.speak(text)
        except Exception as error:  # noqa: BLE001
            log.warning("voz falló: %s", error)

    def voice_confirm(reason: str) -> bool:
        say(reason + " Di 'confirmo' o 'cancela', patrón.")
        try:
            answer = wake_stt.listen_once(silence_ms=silence_ms, max_seconds=5)
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

    # M3: por defecto, loop concurrente. --simple = modo secuencial viejo (fallback).
    # Barge-in conservador: half-duplex por defecto; --barge-in lo activa (auriculares).
    argv = argv or []
    simple = "--simple" in argv
    allow_barge_in = ("--barge-in" in argv) and ("--no-barge-in" not in argv)
    if not simple and use_oww:
        from crotolamo.voice.loop import VoiceLoop

        modo_bi = "barge-in (auriculares)" if allow_barge_in else "half-duplex"
        say(f"Crotolamo escuchando, patrón. (concurrente, {modo_bi})")
        VoiceLoop(agent, stt, tts, wake_detector,
                  allow_barge_in=allow_barge_in, silence_ms=silence_ms).run()
        return 0

    modo = "openWakeWord" if use_oww else "wake difuso (Whisper)"
    say(f"Crotolamo escuchando, patrón. (modo simple, wake: {modo})")

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
                    # Fallback difuso: Whisper 'tiny' sobre el ambiente.
                    heard = wake_stt.listen_once(silence_ms=silence_ms, max_seconds=5,
                                                 start_timeout_s=6)
                    if not heard or not wake.is_wake_word(heard, threshold, variants):
                        continue
            except VoiceUnavailable as error:
                print(str(error))
                return 1

            say("Te escucho, patrón.")
            command = stt.listen_once(silence_ms=silence_ms)

            if not command.strip():
                say("No te escuché claro, patrón.")
                continue

            print(f"Orden: {command}", flush=True)
            reply = agent.handle_turn(command)
            print(reply, flush=True)
            try:
                tts.speak_sentences(reply)  # Fase 6: TTS por frases
            except Exception as error:  # noqa: BLE001
                log.warning("voz falló: %s", error)
            time.sleep(0.5)

        except KeyboardInterrupt:
            say("Crotolamo apagado, patrón.")
            return 0
        except Exception as error:  # noqa: BLE001
            print(f"Error en el listener: {error}", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(run_listen())
