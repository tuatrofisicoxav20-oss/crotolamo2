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
import tempfile
import time
import wave
from pathlib import Path

from crotolamo.settings import get_settings
from crotolamo.voice.stt import STT, VoiceUnavailable
from crotolamo.voice.tts import TTS

FRASE = "crotolamo prueba uno dos tres"


def _write_wav_16k(path: Path, pcm16) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16.tobytes())


def circular() -> int:
    settings = get_settings()
    tts = TTS.from_settings(settings)
    stt = STT.from_settings(settings)

    if not tts.available():
        print("FALLO: no encuentro la voz .onnx en", tts.voice_model)
        return 1

    import numpy as np
    from scipy.signal import resample_poly

    t0 = time.perf_counter()
    pcm, src_rate = tts.synthesize_pcm(FRASE)
    t_tts = time.perf_counter() - t0
    if pcm.size == 0:
        print("FALLO: Piper no produjo audio")
        return 1

    # Resamplea de la tasa de Piper (22050) a los 16 kHz que espera Whisper.
    pcm16 = resample_poly(pcm.astype(np.float32), 16000, src_rate).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        wav_path = Path(t.name)
    _write_wav_16k(wav_path, pcm16)

    t1 = time.perf_counter()
    texto = stt.transcribe(wav_path)
    t_stt = time.perf_counter() - t1
    wav_path.unlink(missing_ok=True)

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
    print("Habla ahora, patrón (grabo hasta el silencio)...")
    try:
        texto = stt.listen_once(max_seconds=5)
    except VoiceUnavailable as error:
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
