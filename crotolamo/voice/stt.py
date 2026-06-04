"""Speech-to-text con faster-whisper + VAD real. Migrado/ampliado de C1::voice_in.

Mejora clave vs C1: en vez de grabar una duración FIJA de 8s, el VAD corta al
silencio (record_until_silence). Eso es lo que más mejora la sensación de 'vivo'.

Las dependencias pesadas (numpy, sounddevice, faster_whisper) se importan de forma
perezosa: este módulo se puede importar sin tenerlas instaladas; solo al transcribir
o grabar se exige la extra [voice].
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from crotolamo.voice.normalize import normalize_text

_INITIAL_PROMPT = (
    "Comandos de voz en español mexicano para un asistente llamado Crotolamo. "
    "Frases comunes: crea una carpeta, abre archivos, mi escritorio, patrón."
)

_model = None


class VoiceUnavailable(RuntimeError):
    """Faltan las dependencias de voz. Instala con: pip install -e '.[voice]'."""


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as error:
        raise VoiceUnavailable(
            f"Falta '{module}', patrón. Instala la voz con: pip install -e '.[voice]'."
        ) from error


class STT:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000,
                 language: str = "es") -> None:
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.language = language

    @classmethod
    def from_settings(cls, settings) -> "STT":
        voice = settings.voice
        return cls(
            model_size=voice.get("whisper_model", "small"),
            sample_rate=voice.get("sample_rate", 16000),
        )

    def _get_model(self):
        global _model
        if _model is None:
            faster_whisper = _require("faster_whisper")
            print("Cargando Whisper, patrón. La primera vez tarda...")
            _model = faster_whisper.WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
        return _model

    # --- grabación ---
    def _write_wav(self, path: Path, audio_int16) -> None:
        np = _require("numpy")
        data = np.ascontiguousarray(np.asarray(audio_int16, dtype=np.int16).reshape(-1))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(data.tobytes())

    def record_until_silence(self, silence_ms: int = 800, max_seconds: float = 12.0,
                             start_timeout_s: float = 4.0) -> Path:
        """Graba hasta detectar `silence_ms` de silencio tras haber hablado (VAD por energía).

        Args:
            silence_ms: silencio sostenido que cierra la grabación.
            max_seconds: tope duro de duración.
            start_timeout_s: si no se detecta voz en este tiempo, corta.
        """
        np = _require("numpy")
        sd = _require("sounddevice")

        chunk_ms = 30
        chunk = int(self.sample_rate * chunk_ms / 1000)
        silence_chunks = max(1, int(silence_ms / chunk_ms))
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        start_chunks = int(start_timeout_s * 1000 / chunk_ms)

        frames: list = []
        speaking = False
        silent_run = 0
        # Umbral de energía adaptativo a partir del primer chunk de ambiente.
        threshold = None

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(block)
                frames.append(block)
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)

                if threshold is None:
                    threshold = max(rms * 3.0, 0.01)
                    continue

                if rms >= threshold:
                    speaking = True
                    silent_run = 0
                elif speaking:
                    silent_run += 1
                    if silent_run >= silence_chunks:
                        break

                if not speaking and i >= start_chunks:
                    break

        audio = np.concatenate(frames) if frames else np.zeros(1, dtype="float32")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self._write_wav(path, np.int16(audio * 32767))
        return path

    # --- transcripción ---
    def transcribe(self, path: Path) -> str:
        model = self._get_model()
        segments, _ = model.transcribe(
            str(path), language=self.language, beam_size=5, vad_filter=True,
            condition_on_previous_text=False, initial_prompt=_INITIAL_PROMPT,
        )
        raw = " ".join(seg.text.strip() for seg in segments).strip()
        return normalize_text(raw)

    def listen_once(self, **vad_kwargs) -> str:
        path = self.record_until_silence(**vad_kwargs)
        try:
            return self.transcribe(path)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
