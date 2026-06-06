from pathlib import Path
import tempfile
import re
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
SECONDS = 8
MODEL_SIZE = "small"
LANGUAGE = "es"

_model = None


def get_model() -> WhisperModel:
    global _model

    if _model is None:
        print("Cargando modelo Whisper. La primera vez puede tardar, patrón...")
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )

    return _model


def normalize_text(text: str) -> str:
    """
    Corrige errores comunes de Whisper con comandos de Crotolamo.
    No es magia, es cinta adhesiva inteligente. Como media ingeniería humana.
    """
    text = text.strip()
    text = text.replace("  ", " ")

    lower = text.lower()

    replacements = {
        "mis hit hoyo": "mi escritorio",
        "mis sitio": "mi escritorio",
        "mi escrit hoyo": "mi escritorio",
        "mi escrito yo": "mi escritorio",
        "mi escriptorio": "mi escritorio",
        "en escritorio": "en mi escritorio",
        "prueba de oído": "prueba_oido",
        "prueba oído": "prueba_oido",
        "prueba oido": "prueba_oido",
        "coto y amo": "crotolamo",
        "coto lamo": "crotolamo",
        "croto lamo": "crotolamo",
    }

    for wrong, right in replacements.items():
        lower = lower.replace(wrong, right)

    lower = re.sub(
        r"^(qué es|que es|que se|es)\s+una\s+carpeta",
        "crea una carpeta",
        lower,
    )

    lower = re.sub(
        r"^crear\s+una\s+carpeta",
        "crea una carpeta",
        lower,
    )

    lower = lower.replace("llamada prueba_oido", "llamada prueba_oido")

    return lower.strip()


def write_wav_int16(path: Path, sample_rate: int, audio_int16: np.ndarray) -> None:
    """
    Escribe WAV mono de 16 bits sin SciPy.

    Antes esto usaba scipy.io.wavfile.write solo para guardar un WAV.
    Eso metía una dependencia enorme para una tarea que Python puede hacer solo.
    La civilización sobrevive otro día.
    """
    audio_int16 = np.asarray(audio_int16, dtype=np.int16)
    audio_int16 = np.ascontiguousarray(audio_int16.reshape(-1))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16 = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def record_audio(seconds: int = SECONDS) -> Path:
    print(f"Escuchando durante {seconds} segundos, patrón...")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    audio = np.squeeze(audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)

    max_val = np.max(np.abs(audio)) if getattr(audio, "size", 0) else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    audio_int16 = np.int16(audio * 32767)
    write_wav_int16(path, SAMPLE_RATE, audio_int16)

    return path


def transcribe_audio(path: Path) -> str:
    model = get_model()

    segments, info = model.transcribe(
        str(path),
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=(
            "Comandos de voz en español mexicano para un asistente llamado Crotolamo. "
            "Frases comunes: crea una carpeta, abre archivos, mi escritorio, prueba_oido, patrón."
        ),
    )

    text_parts = []

    for segment in segments:
        text_parts.append(segment.text.strip())

    raw_text = " ".join(text_parts).strip()
    clean_text = normalize_text(raw_text)

    return clean_text


def listen_once(seconds: int = SECONDS) -> str:
    path = record_audio(seconds)

    try:
        text = transcribe_audio(path)
        return text
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    text = listen_once()
    print(f"Transcripción: {text}")
