"""Smoke test de voz REAL, determinista y sin micrófono (cierra C1).

Circular: sintetiza una frase con el TTS (Piper) -> resamplea a 16 kHz -> la pasa
por el STT (Whisper) -> verifica que el texto no quede vacío. Así se prueba de
verdad la cadena TTS+STT sin depender de un micrófono.

  python scripts/smoke_voz.py          # circular (TTS -> STT)
  python scripts/smoke_voz.py --mic    # graba 4-5 s reales del micrófono

Nota: la API de piper-tts 1.4.2 expone voice.synthesize() (no synthesize_stream_raw);
por eso reutilizamos TTS.synthesize_pcm(), que ya devuelve el PCM int16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from crotolamo.settings import get_settings
    from crotolamo.voice.stt import STT, VoiceUnavailable
    from crotolamo.voice.tts import TTS
except ModuleNotFoundError:  # ejecutado como script suelto (launch.sh sin .venv)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crotolamo.settings import get_settings
    from crotolamo.voice.stt import STT, VoiceUnavailable
    from crotolamo.voice.tts import TTS

FRASE = "crotolamo prueba uno dos tres"


def circular() -> int:
    settings = get_settings()
    tts = TTS.from_settings(settings)
    stt = STT.from_settings(settings)

    if not tts.available():
        print("FALLO: no encuentro la voz .onnx en", tts.voice_model)
        return 1

    # Lo que suelta la cadena sin ser un bug del smoke: un módulo que falta
    # (piper, scipy -> ImportError; faster-whisper -> VoiceUnavailable) o un
    # modelo que no carga/descarga (OSError). Antes salía traceback en vez de
    # "FALLO: ...".
    try:
        import numpy as np
        from scipy.signal import resample_poly

        t0 = time.perf_counter()
        pcm, src_rate = tts.synthesize_pcm(FRASE)
        t_tts = time.perf_counter() - t0
        if pcm.size == 0:
            print("FALLO: Piper no produjo audio")
            return 1

        # Resamplea de la tasa de Piper (22050) a los 16 kHz que espera Whisper y
        # se lo pasa EN MEMORIA como float32 normalizado, igual que el daemon
        # (record_audio_until_silence -> transcribe(ndarray)). Antes iba por un
        # WAV temporal: con av 19 (Python >= 3.12) faster-whisper revienta al
        # abrir el archivo (TypeError por metadata_errors), un «voz rota» falso
        # por un camino que producción no usa; y el .astype(int16) volteaba los
        # picos, porque el resampleo se pasa del rango int16.
        audio = (resample_poly(pcm.astype(np.float32), 16000, src_rate)
                 / 32768.0).astype(np.float32)

        t1 = time.perf_counter()
        texto = stt.transcribe(audio)
        t_stt = time.perf_counter() - t1
    except (VoiceUnavailable, ImportError, OSError) as error:
        print("FALLO:", error)
        return 1
    except Exception as error:
        # Es el diagnóstico de arriba del todo: un error que no esperábamos de la
        # cadena (librería nueva, modelo raro) también es "FALLO: ...", con su
        # tipo para poder buscarlo, nunca un traceback.
        print(f"FALLO: {type(error).__name__}: {error}")
        return 1

    print(f"TTS dijo:      '{FRASE}'")
    print(f"STT devolvió:  '{texto}'")
    print(f"Latencia: TTS={t_tts:.2f}s  STT={t_stt:.2f}s  total={t_tts + t_stt:.2f}s")

    if not texto.strip():
        print("FALLO: el STT devolvió texto vacío")
        return 1
    print("OK: la cadena TTS->STT funciona.")
    return 0


def with_mic() -> int:
    settings = get_settings()
    stt = STT.from_settings(settings)
    # sounddevice se importa aquí para poder NOMBRAR PortAudioError (sin
    # micrófono o con [voice].input_device inválido), que no hereda de OSError:
    # antes salía como traceback igual que el OSError de un libportaudio ausente.
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        print("FALLO: no puedo usar el micrófono:", error)
        return 1
    print("Habla ahora, patrón (grabo hasta el silencio)...")
    try:
        texto = stt.listen_once(max_seconds=5)
    except (VoiceUnavailable, ImportError, OSError, sd.PortAudioError) as error:
        print("FALLO:", error)
        return 1
    print(f"STT (micrófono) devolvió: '{texto}'")
    return 0 if texto.strip() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test de voz de Crotolamo.")
    parser.add_argument("--mic", action="store_true", help="grabar del micrófono real")
    args = parser.parse_args()
    return with_mic() if args.mic else circular()


if __name__ == "__main__":
    sys.exit(main())
