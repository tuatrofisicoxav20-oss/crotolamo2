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

from crotolamo.logging_setup import get_logger
from crotolamo.voice.normalize import normalize_text

log = get_logger("voice.stt")

_INITIAL_PROMPT = (
    "Comandos de voz en español mexicano para un asistente llamado Crotolamo. "
    "Frases comunes: crea una carpeta, abre archivos, mi escritorio, patrón."
)

# Caché de modelos Whisper POR tamaño: así 'base' (comando) y 'tiny' (wake) pueden
# coexistir sin pisarse el uno al otro.
_models: dict[str, object] = {}


class VoiceUnavailable(RuntimeError):
    """Faltan las dependencias de voz. Instala con: pip install -e '.[voice]'."""


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as error:
        raise VoiceUnavailable(
            f"Falta '{module}', patrón. Instala la voz con: pip install -e '.[voice]'."
        ) from error


def _voice_cfg() -> dict:
    """Sección [voice] de la config (o {} si no se puede cargar)."""
    try:
        from crotolamo.settings import get_settings

        return get_settings().voice
    except Exception:
        return {}


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
        if self.model_size not in _models:
            faster_whisper = _require("faster_whisper")
            log.info("Cargando Whisper '%s' (la primera vez tarda)", self.model_size)
            _models[self.model_size] = faster_whisper.WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
        return _models[self.model_size]

    # --- grabación ---
    def _write_wav(self, path: Path, audio_int16) -> None:
        np = _require("numpy")
        data = np.ascontiguousarray(np.asarray(audio_int16, dtype=np.int16).reshape(-1))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(data.tobytes())

    def record_until_silence(self, silence_ms: int | None = None, max_seconds: float = 12.0,
                             start_timeout_s: float = 4.0) -> Path:
        """Graba hasta el silencio. Backend según [voice].vad_backend: 'silero' (M2)
        o 'energy' (fallback). Silero suma un buffer de pre-activación para no
        perder las primeras sílabas.
        """
        voice = _voice_cfg()
        if silence_ms is None:
            silence_ms = voice.get("vad_silence_ms", 640)
        backend = voice.get("vad_backend", "energy")

        if backend == "silero":
            try:
                return self._record_silero(silence_ms, max_seconds, start_timeout_s, voice)
            except VoiceUnavailable:
                raise
            except Exception as error:  # noqa: BLE001 - si silero falla, caemos a energía
                log.warning("VAD silero falló (%s); uso energía", error)
        return self._record_energy(silence_ms, max_seconds, start_timeout_s)

    def _frames_to_wav(self, frames: list) -> Path:
        """Normaliza los frames y los escribe a un WAV temporal."""
        np = _require("numpy")
        audio = np.concatenate(frames) if frames else np.zeros(1, dtype="float32")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self._write_wav(path, np.int16(audio * 32767))
        return path

    def _record_energy(self, silence_ms: int, max_seconds: float,
                       start_timeout_s: float) -> Path:
        """VAD por energía RMS. Calibra el piso de ruido con ~10 chunks (M1 audit)."""
        np = _require("numpy")
        sd = _require("sounddevice")

        chunk_ms = 30
        chunk = int(self.sample_rate * chunk_ms / 1000)
        silence_chunks = max(1, int(silence_ms / chunk_ms))
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        start_chunks = int(start_timeout_s * 1000 / chunk_ms)
        calib_chunks = 10

        frames: list = []
        speaking = False
        silent_run = 0
        threshold = None
        calib_rms: list[float] = []

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(block)
                frames.append(block)
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)

                if i < calib_chunks:
                    calib_rms.append(rms)
                    continue
                if threshold is None:
                    noise_floor = float(np.mean(calib_rms)) if calib_rms else 0.0
                    threshold = max(noise_floor * 3.0, 0.01)

                if rms >= threshold:
                    speaking = True
                    silent_run = 0
                elif speaking:
                    silent_run += 1
                    if silent_run >= silence_chunks:
                        break

                if not speaking and i >= start_chunks:
                    break

        return self._frames_to_wav(frames)

    def _record_silero(self, silence_ms: int, max_seconds: float,
                       start_timeout_s: float, voice: dict) -> Path:
        """VAD neuronal Silero (de GLaDOS, M2). Probabilidad de voz por chunk de
        32 ms + buffer circular de pre-activación para no perder el inicio.
        """
        np = _require("numpy")
        sd = _require("sounddevice")
        torch = _require("torch")
        from collections import deque

        from silero_vad import load_silero_vad

        model = load_silero_vad(onnx=True)
        model.reset_states()
        threshold = voice.get("vad_threshold", 0.8)
        pre_ms = voice.get("vad_preactivation_ms", 800)

        chunk = 512  # Silero exige 512 muestras a 16 kHz (32 ms).
        chunk_ms = chunk * 1000 / self.sample_rate
        silence_chunks = max(1, int(silence_ms / chunk_ms))
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        start_chunks = int(start_timeout_s * 1000 / chunk_ms)
        pre_chunks = max(1, int(pre_ms / chunk_ms))

        pre_buffer: deque = deque(maxlen=pre_chunks)
        frames: list = []
        speaking = False
        silent_run = 0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(np.asarray(block, dtype=np.float32))
                prob = float(model(torch.from_numpy(block.copy()), self.sample_rate))

                if not speaking:
                    pre_buffer.append(block)
                    if prob >= threshold:
                        speaking = True
                        frames.extend(pre_buffer)  # M2.2: anteponer la pre-activación
                        silent_run = 0
                    elif i >= start_chunks:
                        break
                else:
                    frames.append(block)
                    if prob >= threshold:
                        silent_run = 0
                    else:
                        silent_run += 1
                        if silent_run >= silence_chunks:
                            break

        return self._frames_to_wav(frames)

    # --- transcripción ---
    def transcribe(self, path: Path) -> str:
        model = self._get_model()
        # I4: vad_filter=False — ya recortamos por energía en record_until_silence;
        # el doble VAD (energía + el de Whisper) se comía audio.
        # Anti-alucinación: temperature=0 (determinista, sin "inventar" sobre música
        # o silencio) en vez del fallback 0..1 por defecto, que es la causa de los
        # fantasmas tipo "yo te voy a amar" cuando suena Spotify.
        segments, _ = model.transcribe(
            str(path), language=self.language, beam_size=5, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=_INITIAL_PROMPT,
            temperature=0.0,
        )
        # Descarta segmentos alucinados: Whisper marca cada segmento con la prob de
        # "no es voz" (no_speech_prob) y su confianza media (avg_logprob). Si el
        # segmento es muy probablemente NO-voz o de baja confianza, lo tiramos —
        # eso es lo que transcribe de la música/ruido ambiente.
        kept: list[str] = []
        for seg in segments:
            no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
            avg_logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
            if no_speech > 0.6 or avg_logprob < -1.2:
                continue
            text = seg.text.strip()
            if text:
                kept.append(text)
        raw = " ".join(kept).strip()
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
