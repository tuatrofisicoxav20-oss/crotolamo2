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

# Pista de transcripción SOLO para comandos. OJO: Whisper REGURGITA este texto
# literal cuando oye silencio/ruido; por eso NO debe contener "crotolamo" (haría
# que el detector de wake se autodisparara) ni ejemplos de comandos (los inventaría
# como órdenes fantasma). El detector de wake usa initial_prompt=None (sin pista).
_INITIAL_PROMPT = "Transcripción de habla en español de México."

# Caché de modelos Whisper POR tamaño: así 'base' (comando) y 'tiny' (wake) pueden
# coexistir sin pisarse el uno al otro.
_models: dict[str, object] = {}

# Caché del VAD Silero. Antes se hacía load_silero_vad() en CADA grabación (y hasta
# 3 veces por orden con smart_endpoint), releyendo el ONNX del disco cada vez. El
# modelo no tiene configuración: una sola instancia sirve a todos los turnos; su
# estado de RNN se limpia con reset_states() en cada uso.
_silero_vad: object | None = None


def _get_silero_vad():
    """OJO: instancia ÚNICA compartida con `loop._SileroVad`. Silero es una RNN con
    estado; esto es seguro solo porque los dos caminos se excluyen — el modo síncrono
    usa `_record_silero`, y el modo `listen` (hilos) usa el VAD del EarThread, cuyo
    hilo de STT transcribe con Whisper, no con Silero. Si algún día ambos corren a la
    vez, hay que dar una instancia por hilo o el estado se pisará.
    """
    global _silero_vad
    if _silero_vad is None:
        from silero_vad import load_silero_vad

        _silero_vad = load_silero_vad(onnx=True)
    return _silero_vad


class VoiceUnavailable(RuntimeError):
    """Faltan las dependencias de voz. Instala con: pip install -e '.[voice]'."""


def _wav_seconds(path: Path) -> float:
    """Duración del WAV en segundos (0.0 si no se puede leer)."""
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else 0.0
    except (OSError, wave.Error):
        return 0.0


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
    except Exception as error:  # noqa: BLE001 — sin config usable, defaults
        log.debug("no pude cargar la config de voz (%s); uso defaults", error)
        return {}


class STT:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000,
                 language: str = "es", initial_prompt: str | None = _INITIAL_PROMPT,
                 beam_size: int = 5) -> None:
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.language = language
        # El detector de wake debe pasar initial_prompt=None: si no, Whisper
        # regurgita la pista (con "crotolamo") sobre silencio y dispara solo.
        self.initial_prompt = initial_prompt
        # beam_size: haces de búsqueda del decoder. 5 = más preciso; 1-2 = mucho
        # más rápido en CPU con pérdida mínima en comandos cortos. El wake puede
        # ir en 1 (solo busca UNA palabra); comandos, según [voice].beam_size.
        self.beam_size = beam_size

    @classmethod
    def from_settings(cls, settings) -> "STT":
        voice = settings.voice
        return cls(
            model_size=voice.get("whisper_model", "small"),
            sample_rate=voice.get("sample_rate", 16000),
            beam_size=voice.get("beam_size", 5),
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

    def _record_frames(self, silence_ms: int | None, max_seconds: float,
                       start_timeout_s: float) -> list:
        """Graba hasta el silencio y devuelve los frames float32 crudos. Backend
        según [voice].vad_backend: 'silero' (M2) o 'energy' (fallback). Silero
        suma un buffer de pre-activación para no perder las primeras sílabas.
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

    def record_until_silence(self, silence_ms: int | None = None, max_seconds: float = 12.0,
                             start_timeout_s: float = 4.0) -> Path:
        """Compat: graba y devuelve un WAV temporal. El camino caliente ya no
        pasa por aquí (usa record_audio_until_silence, sin tocar disco)."""
        return self._frames_to_wav(
            self._record_frames(silence_ms, max_seconds, start_timeout_s)
        )

    def record_audio_until_silence(self, silence_ms: int | None = None,
                                   max_seconds: float = 12.0,
                                   start_timeout_s: float = 4.0):
        """Graba hasta el silencio y devuelve el audio float32 EN MEMORIA.

        Misión velocidad: faster-whisper acepta el ndarray directo, así que el
        WAV temporal (escribir a disco + relerlo + borrarlo por frase) era costo
        puro. Este es el camino caliente de todos los comandos.
        """
        return self._frames_to_audio(
            self._record_frames(silence_ms, max_seconds, start_timeout_s)
        )

    def _frames_to_audio(self, frames: list):
        """Concatena y normaliza los frames a un float32 mono listo para Whisper."""
        np = _require("numpy")
        audio = np.concatenate(frames) if frames else np.zeros(1, dtype="float32")
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        return audio

    def _frames_to_wav(self, frames: list) -> Path:
        """Normaliza los frames y los escribe a un WAV temporal (compat/tests)."""
        np = _require("numpy")
        audio = self._frames_to_audio(frames)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self._write_wav(path, np.int16(audio * 32767))
        return path

    def _record_energy(self, silence_ms: int, max_seconds: float,
                       start_timeout_s: float) -> list:
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

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            device=_voice_cfg().get("input_device")) as stream:
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

        return frames

    def _record_silero(self, silence_ms: int, max_seconds: float,
                       start_timeout_s: float, voice: dict) -> list:
        """VAD neuronal Silero (de GLaDOS, M2). Probabilidad de voz por chunk de
        32 ms + buffer circular de pre-activación para no perder el inicio.
        """
        np = _require("numpy")
        sd = _require("sounddevice")
        torch = _require("torch")
        from collections import deque

        from crotolamo.voice.vad import is_voice, resolve_neg_threshold

        model = _get_silero_vad()
        model.reset_states()
        threshold = voice.get("vad_threshold", 0.8)
        # Histéresis (misma decisión que el loop concurrente, ver voice/vad.py):
        # ausente/None = sin histéresis, comportamiento idéntico al histórico.
        neg_threshold = resolve_neg_threshold(threshold, voice.get("vad_neg_threshold"))
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

        # input_device: None = default del sistema; "crotolamo_aec_source" tras el AEC.
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            device=voice.get("input_device")) as stream:
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(np.asarray(block, dtype=np.float32))
                # ascontiguousarray evita la copia extra que hacía .copy(): solo
                # copia si el bloque no es ya un array float32 contiguo.
                prob = float(model(
                    torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)),
                    self.sample_rate,
                ))

                if not speaking:
                    pre_buffer.append(block)
                    # Entrar exige el umbral ALTO (no despertar con cualquier ruido).
                    if is_voice(prob, speaking=False, threshold=threshold,
                                neg_threshold=neg_threshold):
                        speaking = True
                        frames.extend(pre_buffer)  # M2.2: anteponer la pre-activación
                        silent_run = 0
                    elif i >= start_chunks:
                        break
                else:
                    frames.append(block)
                    # Mantenerse basta con el umbral BAJO (histéresis).
                    if is_voice(prob, speaking=True, threshold=threshold,
                                neg_threshold=neg_threshold):
                        silent_run = 0
                    else:
                        silent_run += 1
                        if silent_run >= silence_chunks:
                            break

        return frames

    # --- transcripción ---
    def transcribe(self, audio, hotwords: str | None = None) -> str:
        """Transcribe un WAV (Path/str) o un ndarray float32 16kHz EN MEMORIA.

        `hotwords` (faster-whisper >= 1.0.2) sesga el decoder hacia nombres
        propios ("Crotolamo", "Tletl"...) SOLO en la llamada que lo pida — el
        caller decide; aquí no se hardcodea nada. La ruta de wake NO debe
        pasarlo: igual que initial_prompt, un sesgo con "crotolamo" sobre
        silencio/ruido aumentaría falsos despertares.
        """
        model = self._get_model()
        if isinstance(audio, (str, Path)):
            audio = str(audio)  # faster-whisper acepta la ruta tal cual
        # I4: vad_filter=False — ya recortamos por energía en record_until_silence;
        # el doble VAD (energía + el de Whisper) se comía audio.
        # Anti-alucinación: temperature=0 (determinista, sin "inventar" sobre música
        # o silencio) en vez del fallback 0..1 por defecto, que es la causa de los
        # fantasmas tipo "yo te voy a amar" cuando suena Spotify.
        segments, _ = model.transcribe(
            audio, language=self.language, beam_size=self.beam_size, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=self.initial_prompt,
            temperature=0.0, hotwords=hotwords,
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
        return self._listen_transcribe(**vad_kwargs)

    def _listen_transcribe(self, hotwords: str | None = None,
                           min_audio_s: float = 0.0, **vad_kwargs) -> str:
        """Graba y transcribe EN MEMORIA (sin WAV temporal). min_audio_s: si la
        grabación quedó más corta (el VAD nunca detectó voz), devuelve "" SIN
        pasar por Whisper — el modelo paddea a ventanas de 30s, así que
        transcribir silencio cuesta ~1s con el micrófono cerrado. En el bucle
        del wake eso era una ventana SORDA cada ciclo silencioso; ahora el mic
        reabre de inmediato.
        """
        audio = self.record_audio_until_silence(**vad_kwargs)
        if min_audio_s > 0 and (audio.size / self.sample_rate) < min_audio_s:
            return ""
        return self.transcribe(audio, hotwords=hotwords)

    def listen_smart(self, silence_ms: int | None = None, max_seconds: float = 12.0,
                     start_timeout_s: float = 4.0, rounds: int = 2,
                     continue_timeout_s: float = 2.5,
                     hotwords: str | None = None,
                     min_audio_s: float = 0.0) -> str:
        """Escucha con endpointing inteligente: si la frase parece INCOMPLETA
        (pausa de pensar: termina en "de", "para", coma...), reabre la escucha
        `continue_timeout_s` segundos y concatena la continuación, hasta
        `rounds` veces. Si la continuación sale vacía (de verdad ya terminó),
        se queda con lo que hay. Con rounds=0 equivale a listen_once.
        """
        from crotolamo.voice.endpoint import seems_incomplete

        text = self._listen_transcribe(
            hotwords=hotwords, min_audio_s=min_audio_s, silence_ms=silence_ms,
            max_seconds=max_seconds, start_timeout_s=start_timeout_s,
        )
        for _ in range(max(0, rounds)):
            if not text.strip() or not seems_incomplete(text):
                break
            extra = self._listen_transcribe(
                hotwords=hotwords, min_audio_s=min_audio_s, silence_ms=silence_ms,
                max_seconds=max_seconds, start_timeout_s=continue_timeout_s,
            )
            if not extra.strip():
                break
            text = f"{text.strip()} {extra.strip()}"
        return text
