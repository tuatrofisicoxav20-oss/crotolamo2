"""Speech-to-text con faster-whisper + VAD real. Migrado/ampliado de C1::voice_in.

Mejora clave vs C1: en vez de grabar una duración FIJA de 8s, el VAD corta al
silencio (record_until_silence). Eso es lo que más mejora la sensación de 'vivo'.

Las dependencias pesadas (numpy, sounddevice, faster_whisper) se importan de forma
perezosa: este módulo se puede importar sin tenerlas instaladas; solo al transcribir
o grabar se exige la extra [voice].
"""

from __future__ import annotations

import math
import tempfile
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crotolamo.logging_setup import get_logger
from crotolamo.voice import runtime
from crotolamo.voice.normalize import normalize_text
from crotolamo.voice.vad import DEFAULTS, FIN, SIN_VOZ, Endpointer, resolve_neg_threshold

log = get_logger("voice.stt")

# Pista de transcripción SOLO para comandos. OJO: Whisper REGURGITA este texto
# literal cuando oye silencio/ruido; por eso NO debe contener "crotolamo" (haría
# que el detector de wake se autodisparara) ni ejemplos de comandos (los inventaría
# como órdenes fantasma). El detector de wake usa initial_prompt=None (sin pista).
_INITIAL_PROMPT = "Transcripción de habla en español de México."

# Caché de modelos Whisper POR tamaño: así 'base' (comando) y 'tiny' (wake) pueden
# coexistir sin pisarse el uno al otro.
_models: dict[str, object] = {}

# Un lock de carga POR tamaño. El hilo WarmVoice del listener precalienta los
# modelos mientras el loop ya escucha: sin lock, el primer transcribe() que
# llegaba a media carga construía una SEGUNDA copia del mismo modelo (segundos de
# CPU y ~137 MB de pico para nada). Por tamaño y no global: cargar (o descargar)
# 'base' no debe dejar esperando al wake, que usa 'tiny'.
_model_locks: dict[str, threading.Lock] = {}
_model_locks_guard = threading.Lock()


def _model_lock(model_size: str) -> threading.Lock:
    with _model_locks_guard:
        return _model_locks.setdefault(model_size, threading.Lock())


# Caché del VAD Silero. Antes se hacía load_silero_vad() en CADA grabación (y hasta
# 3 veces por orden con smart_endpoint), releyendo el ONNX del disco cada vez. El
# modelo no tiene configuración: una sola instancia sirve a todos los turnos; su
# estado de RNN se limpia con reset_states() en cada uso.
_silero_vad: object | None = None

# Lock de carga de Silero. Igual que con Whisper: WarmVoice y el primer uso real
# (el oído o una grabación del modo simple) podían ver la caché vacía a la vez y
# construir cada uno su sesión ONNX, tirando una.
_silero_lock = threading.Lock()

# Motivo por el que CARGAR Silero falló (falta silero-vad u onnxruntime, ONNX
# corrupto), o None. Ese fallo es permanente en el proceso: antes cada grabación
# del modo simple reintentaba el import/carga y escribía un WARNING — uno cada
# <=6 s, para siempre, en el bucle del wake — para acabar igual en el VAD por
# energía. Mismo criterio que el fallback permanente de adapters._SileroVad.
_silero_load_error: str | None = None


def _get_silero_vad(*, remember_error: bool = True):
    """OJO: instancia ÚNICA compartida con `loop._SileroVad`. Silero es una RNN con
    estado; esto es seguro solo porque los dos caminos se excluyen — el modo síncrono
    usa `_record_silero`, y el modo `listen` (hilos) usa el VAD del EarThread, cuyo
    hilo de STT transcribe con Whisper, no con Silero. Si algún día ambos corren a la
    vez, hay que dar una instancia por hilo o el estado se pisará.

    remember_error=False (solo warm_vad): un fallo de carga NO se memoriza, para que
    el primer uso real lo reintente y lo avise él (ver warm_vad).
    """
    global _silero_vad, _silero_load_error
    # Camino caliente sin lock: con el modelo ya cargado es leer un global.
    if _silero_vad is not None:
        return _silero_vad
    with _silero_lock:
        # Doble comprobación: otro hilo pudo cargarlo (o fallar) mientras esperábamos.
        if _silero_vad is None:
            if _silero_load_error is not None:
                raise RuntimeError(f"Silero no cargó: {_silero_load_error}")
            try:
                # Sin torch: el mismo silero_vad.onnx con numpy + onnxruntime, mismas
                # probabilidades (ver runtime.SileroOnnx). Antes load_silero_vad()
                # importaba torch (+218 MB) solo como contenedor de tensores.
                _silero_vad = runtime.load_silero()
            except Exception as error:
                if remember_error:
                    _silero_load_error = f"{type(error).__name__}: {error}"
                raise
    return _silero_vad


def warm_vad() -> bool:
    """Precalienta Silero: la misma carga que haría su primer uso real (el primer
    frame del oído en el loop concurrente, o la primera grabación del modo simple),
    pero en el hilo de quien llama y antes de que haga falta. Sin esto, la carga
    caía DENTRO del hilo del oído tras el primer wake, con el micrófono sin leer.

    Nunca lanza, y si no carga devuelve False SIN memorizar el fallo: la decisión
    de caer a energía (y su único aviso en el log) la sigue tomando el primer uso
    real, idéntica a la de no haber precalentado. True si el modelo quedó en caché.
    """
    try:
        _get_silero_vad(remember_error=False)
    except Exception as error:  # noqa: BLE001 - lo decide y lo avisa el primer uso real
        log.debug("no precalenté el VAD Silero (%s); lo decidirá su primer uso", error)
        return False
    return True


class VoiceUnavailable(RuntimeError):
    """Faltan las dependencias de voz. Instala con: pip install -e '.[voice]'."""


# Lo que el pre-filtro del wake ([wake].pregate) deja ANTES de su disparo: la
# ventana que puntúa openWakeWord (16 embeddings = 1.28 s). Así la palabra encabeza
# lo que transcribe Whisper (split_wake_command mira las 3 primeras palabras)
# aunque se dijera a media ventana.
_GATE_LOOKBACK_S = 1.28


class _GateRun:
    """Una grabación con pre-filtro (`gate`: rearm() y feed(chunk) -> bool, el
    WakeWordDetector o un envoltorio). Al abrir el stream se le pide rearm() (el
    del listener solo olvida tras un hueco del loop: ver listener._PreFiltro),
    recibe CADA chunk leído hasta su primer disparo (el pre-roll incluido) y
    recuerda en qué chunk fue.

    Sin gate (None) no hace nada: la grabación es la de siempre. Un gate que lanza
    se avisa y la ventana sigue SIN él, entera a Whisper como sin pre-filtro: el
    pre-filtro solo ahorra pasadas, no puede dejar sordo al wake.
    """

    def __init__(self, gate: Any, lookback_chunks: int) -> None:
        self.gate = gate
        self.lookback = lookback_chunks
        self.fired_at: int | None = None
        if gate is not None:
            self._call(gate.rearm)

    def feed(self, index: int, block) -> None:
        if self.gate is not None and self.fired_at is None and self._call(self.gate.feed, block):
            self.fired_at = index

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 - un pre-filtro roto no tumba la escucha
            log.warning("pre-filtro del wake falló (%s); esta ventana va entera a Whisper",
                        error)
            self.gate = None
            return False

    def trim(self, frames: list, first_index: int) -> list:
        """Lo que se devuelve: los frames tal cual sin gate; [] si nunca disparó
        (nada para Whisper); si disparó, desde `lookback` chunks antes del disparo.
        `first_index` es el chunk del que viene frames[0]."""
        if self.gate is None:
            return frames
        if self.fired_at is None:
            return []
        return frames[max(0, self.fired_at - self.lookback - first_index):]


def _leer_fuente(source: Any, frames: int, on_idle: Callable[[], bool] | None) -> Any:
    """Siguiente bloque de `frames` muestras de una fuente YA abierta (el
    contrato de capture.MicSession). Si en 1 s no llega audio (el micrófono
    reintentando, la sesión suspendida o cerrada) se pregunta a on_idle: True =
    seguir esperando; sin on_idle o False = None, y la grabación acaba ahí."""
    while True:
        block = source.read(timeout=1.0, frames=frames)
        if block is not None:
            return block
        # Cerrada no entregará nada más: su read() vuelve al instante, y seguir
        # preguntando a on_idle sería girar en vacío.
        if getattr(source, "closed", False) or on_idle is None or not on_idle():
            return None


class _FuenteVigilada:
    """La fuente de una grabación, recordando si fue ELLA la que lanzó.

    Con el stream propio, cualquier error dentro de la grabación con Silero se
    toma por un fallo del VAD y se reintenta con energía (y otro stream). Con
    una fuente ya abierta, un fallo del micrófono no es del VAD ni lo arregla
    otro VAD: _record_frames lo deja subir tal cual."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.error: BaseException | None = None

    @property
    def closed(self) -> bool:
        return bool(getattr(self._source, "closed", False))

    def read(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._source.read(*args, **kwargs)
        except Exception as error:
            self.error = error
            raise

    def pending(self, *args: Any, **kwargs: Any) -> int:
        try:
            return self._source.pending(*args, **kwargs)
        except Exception as error:
            self.error = error
            raise


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


def _whisper_cpu_threads() -> int:
    """[voice].whisper_cpu_threads: hilos de CTranslate2 por modelo Whisper.

    0 (o ausente) = el default de faster-whisper, el comportamiento de siempre.
    Un valor inválido no impide cargar: se avisa y se usa el default.
    """
    value = _voice_cfg().get("whisper_cpu_threads", 0)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    log.warning("[voice].whisper_cpu_threads=%r no es un entero >= 0; uso el default de "
                "faster-whisper", value)
    return 0


class STT:
    def __init__(self, model_size: str = "small", sample_rate: int = DEFAULTS.sample_rate,
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
            sample_rate=voice.get("sample_rate", DEFAULTS.sample_rate),
            beam_size=voice.get("beam_size", 5),
        )

    def _get_model(self):
        # Camino caliente sin lock: con el modelo ya cargado es un dict lookup.
        if self.model_size not in _models:
            with _model_lock(self.model_size):
                # Doble comprobación: otro hilo pudo cargarlo mientras esperábamos.
                if self.model_size not in _models:
                    faster_whisper = _require("faster_whisper")
                    log.info("Cargando Whisper '%s' (la primera vez tarda)",
                             self.model_size)
                    # Primero de la caché local (sin preguntar al Hub en cada
                    # arranque) y sin el hilo monitor de tqdm; si no está en caché,
                    # la llamada de siempre, que lo descarga (ver runtime.load_whisper).
                    _models[self.model_size] = runtime.load_whisper(
                        faster_whisper, self.model_size,
                        cpu_threads=_whisper_cpu_threads(),
                    )
        return _models[self.model_size]

    def warm(self) -> bool:
        """Precalienta este tamaño de Whisper: la misma carga que el primer
        transcribe(), en el hilo de quien llama. Si otro hilo ya lo está cargando,
        espera a esa carga (el lock por tamaño) en vez de repetirla.

        Nunca lanza: si no carga no queda nada cacheado, así que el primer uso real
        lo reintenta y da su error de siempre. True si el modelo quedó cargado.
        """
        try:
            self._get_model()
        except Exception as error:  # noqa: BLE001 - lo decide y lo avisa el primer uso real
            log.debug("no precalenté Whisper '%s' (%s); lo decidirá su primer uso",
                      self.model_size, error)
            return False
        return True

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
                       start_timeout_s: float, gate: Any = None, source: Any = None,
                       on_idle: Callable[[], bool] | None = None) -> list:
        """Graba hasta el silencio y devuelve los frames float32 crudos. Backend
        según [voice].vad_backend: 'silero' (M2) o 'energy' (fallback). Silero
        suma un buffer de pre-activación para no perder las primeras sílabas.
        gate: pre-filtro del wake difuso (ver _GateRun); None = lo de siempre.
        source/on_idle: fuente ya abierta (capture.MicSession) en vez de un
        stream propio, ver _silero_de_fuente; None = lo de siempre. Con fuente,
        un fallo de la FUENTE sube tal cual: no es del VAD (ver _FuenteVigilada).
        """
        voice = _voice_cfg()
        if silence_ms is None:
            silence_ms = voice.get("vad_silence_ms", DEFAULTS.silence_ms)
        backend = voice.get("vad_backend", DEFAULTS.vad_backend)
        # Sin fuente, los grabadores reciben exactamente los argumentos de siempre.
        fuente = None if source is None else _FuenteVigilada(source)
        extra: dict[str, Any] = {} if fuente is None else {"source": fuente,
                                                            "on_idle": on_idle}

        if backend == "silero":
            if _silero_load_error is not None:
                # Ya no carga ni cargará en este proceso, y se avisó UNA vez (abajo).
                log.debug("VAD silero no disponible (%s); uso energía", _silero_load_error)
            else:
                try:
                    return self._record_silero(silence_ms, max_seconds, start_timeout_s,
                                               voice, gate=gate, **extra)
                except VoiceUnavailable:
                    raise
                except Exception as error:  # noqa: BLE001 - si silero falla, caemos a energía
                    if fuente is not None and fuente.error is error:
                        raise
                    log.warning("VAD silero falló (%s); uso energía", error)
        return self._record_energy(silence_ms, max_seconds, start_timeout_s, gate=gate,
                                   **extra)

    def record_until_silence(self, silence_ms: int | None = None,
                             max_seconds: float = DEFAULTS.max_command_s,
                             start_timeout_s: float = DEFAULTS.command_start_timeout_s) -> Path:
        """Compat: graba y devuelve un WAV temporal. El camino caliente ya no
        pasa por aquí (usa record_audio_until_silence, sin tocar disco)."""
        return self._frames_to_wav(
            self._record_frames(silence_ms, max_seconds, start_timeout_s)
        )

    def record_audio_until_silence(self, silence_ms: int | None = None,
                                   max_seconds: float = DEFAULTS.max_command_s,
                                   start_timeout_s: float = DEFAULTS.command_start_timeout_s,
                                   gate: Any = None,
                                   source: Any = None,
                                   on_idle: Callable[[], bool] | None = None):
        """Graba hasta el silencio y devuelve el audio float32 EN MEMORIA.

        Misión velocidad: faster-whisper acepta el ndarray directo, así que el
        WAV temporal (escribir a disco + relerlo + borrarlo por frase) era costo
        puro. Este es el camino caliente de todos los comandos.

        source/on_idle: leer de una fuente ya abierta (capture.MicSession) en vez
        de abrir el micrófono; ver _silero_de_fuente. None = lo de siempre.
        """
        extra: dict[str, Any] = {} if source is None else {"source": source,
                                                            "on_idle": on_idle}
        return self._frames_to_audio(
            self._record_frames(silence_ms, max_seconds, start_timeout_s, gate=gate, **extra)
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
                       start_timeout_s: float, gate: Any = None, source: Any = None,
                       on_idle: Callable[[], bool] | None = None) -> list:
        """VAD por energía RMS. Calibra el piso de ruido con ~10 chunks (M1 audit).
        Si en la ventana solo hubo ruido estable, devuelve [] (igual que Silero).
        gate: pre-filtro del wake difuso (ver _GateRun).
        source/on_idle: fuente ya abierta en vez de un stream propio (ver
        _energia_de_fuente); None = lo de siempre."""
        if source is not None:
            return self._energia_de_fuente(silence_ms, max_seconds, start_timeout_s,
                                           gate, source, on_idle)
        np = _require("numpy")
        sd = _require("sounddevice")

        chunk_ms = 30
        chunk = int(self.sample_rate * chunk_ms / 1000)
        silence_chunks = max(1, int(silence_ms / chunk_ms))
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        start_chunks = int(start_timeout_s * 1000 / chunk_ms)
        calib_chunks = 10
        ratio = 3.0  # voz = RMS >= max(piso de ruido * ratio, 0.01)

        frames: list = []
        speaking = False
        silent_run = 0
        threshold = None
        calib_rms: list[float] = []
        # Chunk más bajo CON señal (los ceros digitales del arranque del stream o
        # de un mic silenciado no dicen nada del ruido de la sala) y el más alto.
        rms_floor = rms_max = 0.0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            device=_voice_cfg().get("input_device")) as stream:
            gated = _GateRun(gate, math.ceil(_GATE_LOOKBACK_S * self.sample_rate / chunk))
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(block)
                frames.append(block)
                gated.feed(i, block)
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
                if rms > 1e-6 and (rms_floor == 0.0 or rms < rms_floor):
                    rms_floor = rms
                rms_max = max(rms_max, rms)

                if i < calib_chunks:
                    calib_rms.append(rms)
                    continue
                if threshold is None:
                    noise_floor = float(np.mean(calib_rms)) if calib_rms else 0.0
                    threshold = max(noise_floor * ratio, 0.01)

                if rms >= threshold:
                    speaking = True
                    silent_run = 0
                elif speaking:
                    silent_run += 1
                    if silent_run >= silence_chunks:
                        break

                if not speaking and i >= start_chunks:
                    break

        # Solo ruido de fondo estable (o silencio digital): nada que transcribir,
        # [] como Silero, y min_audio_s lo corta antes de Whisper. Antes se
        # devolvía la ventana ENTERA (6 s en el wake difuso), Whisper la
        # transcribía normalizada a 0.9 y el micrófono quedaba sordo ~1 s en cada
        # ciclo vacío. Se juzga con el criterio MÁS permisivo, no con `speaking`:
        # si la voz (o el bip) cayó en los chunks de calibración, el umbral quedó
        # en 3x la propia voz; y una voz bajita (mic con poca ganancia) no cruza
        # el piso absoluto de 0.01. En ambos casos `speaking` no se enciende pero
        # algún chunk supera 3x el más bajo, y el audio llega a Whisper como antes.
        if not speaking and (rms_floor == 0.0 or rms_max < rms_floor * ratio):
            return []
        return gated.trim(frames, 0)

    def _energia_de_fuente(self, silence_ms: int, max_seconds: float,
                           start_timeout_s: float, gate: Any, source: Any,
                           on_idle: Callable[[], bool] | None) -> list:
        """_record_energy sobre una fuente ya abierta: mismas decisiones chunk a
        chunk (bloques de 30 ms que la fuente re-encuadra), con lo propio de leer
        de una fuente que explica _silero_de_fuente. Al pedir seguir tras una
        espera agotada, recalibra y empieza los frames ahí, como una llamada
        nueva (la energía no tiene pre-activación que conservar)."""
        np = _require("numpy")

        chunk_ms = 30
        chunk = int(self.sample_rate * chunk_ms / 1000)
        silence_chunks = max(1, int(silence_ms / chunk_ms))
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        start_chunks = int(start_timeout_s * 1000 / chunk_ms)
        calib_chunks = 10
        ratio = 3.0  # voz = RMS >= max(piso de ruido * ratio, 0.01)

        frames: list = []
        speaking = False
        silent_run = 0
        threshold = None
        calib_rms: list[float] = []
        rms_floor = rms_max = 0.0
        origen = 0  # chunk del que viene frames[0] (y con el que empieza la calibración)
        # Chunk desde el que cuentan los plazos: el de la llamada, tras lo que la
        # fuente ya tenía capturado (como si el stream se abriera aquí).
        ventana = int(source.pending(chunk))
        lookback = math.ceil(_GATE_LOOKBACK_S * self.sample_rate / chunk)
        gated = _GateRun(gate, lookback)
        i = -1
        while True:
            i += 1
            if i - ventana >= max_chunks:
                break
            block = _leer_fuente(source, chunk, on_idle)
            if block is None:
                break
            block = np.squeeze(block)
            frames.append(block)
            gated.feed(i, block)
            rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
            if rms > 1e-6 and (rms_floor == 0.0 or rms < rms_floor):
                rms_floor = rms
            rms_max = max(rms_max, rms)

            if i - origen < calib_chunks:
                calib_rms.append(rms)
                continue
            if threshold is None:
                noise_floor = float(np.mean(calib_rms)) if calib_rms else 0.0
                threshold = max(noise_floor * ratio, 0.01)

            if rms >= threshold:
                speaking = True
                silent_run = 0
            elif speaking:
                silent_run += 1
                if silent_run >= silence_chunks:
                    break

            if not speaking and i - ventana >= start_chunks:
                if on_idle is None or not on_idle():
                    break
                frames, calib_rms, threshold = [], [], None
                rms_floor = rms_max = 0.0
                origen = ventana = i + 1
                # El gate de la grabación, no el original: roto ya es None y sigue
                # fuera (ver _silero_de_fuente).
                gated = _GateRun(gated.gate, lookback)

        if not speaking and (rms_floor == 0.0 or rms_max < rms_floor * ratio):
            return []
        return gated.trim(frames, origen)

    def _record_silero(self, silence_ms: int, max_seconds: float,
                       start_timeout_s: float, voice: dict, gate: Any = None,
                       source: Any = None, on_idle: Callable[[], bool] | None = None) -> list:
        """VAD neuronal Silero (de GLaDOS, M2). Probabilidad de voz por chunk de
        32 ms + buffer circular de pre-activación para no perder el inicio.
        gate: pre-filtro del wake difuso (ver _GateRun).
        source/on_idle: fuente ya abierta en vez de un stream propio (ver
        _silero_de_fuente); None = lo de siempre.
        """
        if source is not None:
            return self._silero_de_fuente(silence_ms, max_seconds, start_timeout_s, voice,
                                          gate, source, on_idle)
        np = _require("numpy")
        sd = _require("sounddevice")

        model = _get_silero_vad()
        model.reset_states()
        chunk = 512  # Silero exige 512 muestras a 16 kHz (32 ms).
        ep, max_chunks = self._endpointer(silence_ms, max_seconds, start_timeout_s, voice,
                                          chunk)

        # input_device: None = default del sistema; "crotolamo_aec_source" tras el AEC.
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            device=voice.get("input_device")) as stream:
            gated = _GateRun(gate, math.ceil(_GATE_LOOKBACK_S * self.sample_rate / chunk))
            for i in range(max_chunks):
                block, _ = stream.read(chunk)
                block = np.squeeze(np.asarray(block, dtype=np.float32))
                gated.feed(i, block)
                # ascontiguousarray evita la copia extra que hacía .copy(): solo
                # copia si el bloque no es ya un array float32 contiguo. El modelo
                # recibe numpy: ya no hace falta torch (antes, sin torch, esto
                # lanzaba VoiceUnavailable y el servicio salía en bucle).
                prob = float(model(np.ascontiguousarray(block, dtype=np.float32),
                                   self.sample_rate))
                if ep.feed(block, prob) in (FIN, SIN_VOZ):
                    break

        return gated.trim(ep.frames, ep.first)

    def _endpointer(self, silence_ms: int, max_seconds: float, start_timeout_s: float,
                    voice: dict, chunk: int, ahead: int = 0) -> tuple[Endpointer, int]:
        """El Endpointer (voice/vad.py) de una grabación con Silero y su tope en chunks.

        El modo simple siempre esperó la voz UN chunk más que su plazo (su `i >=
        start_chunks` contaba los chunks ANTERIORES al actual; el oído cuenta el
        actual): de ahí el + 1, que conserva cada decisión tal cual. ahead: chunks
        ya capturados por la fuente al empezar (ver _silero_de_fuente).
        """
        threshold = voice.get("vad_threshold", DEFAULTS.vad_threshold)
        # Histéresis (misma decisión que el loop concurrente, ver voice/vad.py):
        # ausente/None = sin histéresis, comportamiento idéntico al histórico.
        neg_threshold = resolve_neg_threshold(
            threshold, voice.get("vad_neg_threshold", DEFAULTS.vad_neg_threshold))
        pre_ms = voice.get("vad_preactivation_ms", DEFAULTS.vad_preactivation_ms)
        chunk_ms = chunk * 1000 / self.sample_rate
        max_chunks = int(max_seconds * 1000 / chunk_ms)
        ep = Endpointer(threshold=threshold, neg_threshold=neg_threshold,
                        silence_chunks=max(1, int(silence_ms / chunk_ms)),
                        max_chunks=max_chunks,
                        start_chunks=int(start_timeout_s * 1000 / chunk_ms) + 1,
                        pre_chunks=max(1, int(pre_ms / chunk_ms)), ahead=ahead)
        return ep, max_chunks

    def _silero_de_fuente(self, silence_ms: int, max_seconds: float,
                          start_timeout_s: float, voice: dict, gate: Any, source: Any,
                          on_idle: Callable[[], bool] | None) -> list:
        """_record_silero sobre una fuente ya abierta (capture.MicSession): las
        mismas decisiones chunk a chunk que con el stream propio (lo comparan
        los tests de tests/test_capture.py), más lo que solo existe porque el
        micrófono no se abre aquí:

        - Los plazos (start_timeout_s, max_seconds) cuentan desde ESTA llamada,
          como cuando el stream se abría aquí: lo que la fuente ya tenía (lo que
          sonó mientras Whisper decodificaba) se lee primero y los alarga igual.
        - Agotada la espera sin voz, o si en 1 s no llega audio, se pregunta a
          on_idle(): True = seguir esperando en la misma fuente (tras la espera
          agotada, como una llamada nueva: Silero en blanco y plazos desde ahí,
          pero CONSERVANDO la pre-activación, que una llamada nueva perdía); sin
          on_idle o False, la grabación acaba ahí: [] sin voz, lo grabado con ella.
        - Un error de la fuente sube tal cual (ver _FuenteVigilada).
        """
        np = _require("numpy")

        model = _get_silero_vad()
        model.reset_states()
        chunk = 512  # Silero exige 512 muestras a 16 kHz (32 ms).
        # Los plazos cuentan desde la llamada, tras lo ya capturado (ahead) y, si
        # on_idle pide seguir, desde el chunk siguiente a la espera agotada.
        ep, _ = self._endpointer(silence_ms, max_seconds, start_timeout_s, voice, chunk,
                                 ahead=int(source.pending(chunk)))
        lookback = math.ceil(_GATE_LOOKBACK_S * self.sample_rate / chunk)
        gated = _GateRun(gate, lookback)
        i = -1
        while True:
            i += 1
            if ep.expired():
                break
            block = _leer_fuente(source, chunk, on_idle)
            if block is None:
                break
            block = np.squeeze(np.asarray(block, dtype=np.float32))
            gated.feed(i, block)
            prob = float(model(np.ascontiguousarray(block, dtype=np.float32),
                               self.sample_rate))
            estado = ep.feed(block, prob)
            if estado == SIN_VOZ:
                if on_idle is None or not on_idle():
                    break
                model.reset_states()
                ep.restart()
                # El gate de ESTA grabación, no el original: uno que ya lanzó es None
                # y sigue fuera hasta que la grabación vuelva al listener (que es
                # quien deja de pasarlo). Rearmar el original cada ventana lo hacía
                # fallar y avisar de nuevo cada ~6 s mientras on_idle siguiera.
                gated = _GateRun(gated.gate, lookback)
            elif estado == FIN:
                break

        return gated.trim(ep.frames, ep.first)

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
                           min_audio_s: float = 0.0,
                           transcribe_if: Callable[[], bool] | None = None,
                           **vad_kwargs) -> str:
        """Graba y transcribe EN MEMORIA (sin WAV temporal). min_audio_s: si la
        grabación quedó más corta (el VAD nunca detectó voz), devuelve "" SIN
        pasar por Whisper — el modelo paddea a ventanas de 30s, así que
        transcribir silencio cuesta ~1s con el micrófono cerrado. En el bucle
        del wake eso era una ventana SORDA cada ciclo silencioso; ahora el mic
        reabre de inmediato.

        transcribe_if: se consulta con la grabación YA hecha; si dice False,
        devuelve "" sin Whisper. Es para quien tiraría el texto de todos modos
        (el wake difuso con la escucha pausada a media ventana): no cambia
        ninguna transcripción, solo se ahorra la que nadie iba a usar.
        """
        audio = self.record_audio_until_silence(**vad_kwargs)
        if min_audio_s > 0 and (audio.size / self.sample_rate) < min_audio_s:
            return ""
        if transcribe_if is not None and not transcribe_if():
            return ""
        return self.transcribe(audio, hotwords=hotwords)

    def listen_smart(self, silence_ms: int | None = None,
                     max_seconds: float = DEFAULTS.max_command_s,
                     start_timeout_s: float = DEFAULTS.command_start_timeout_s,
                     rounds: int = DEFAULTS.endpoint_rounds,
                     continue_timeout_s: float = DEFAULTS.endpoint_continue_s,
                     hotwords: str | None = None,
                     min_audio_s: float = 0.0, source: Any = None) -> str:
        """Escucha con endpointing inteligente: si la frase parece INCOMPLETA
        (pausa de pensar: termina en "de", "para", coma...), reabre la escucha
        `continue_timeout_s` segundos y concatena la continuación, hasta
        `rounds` veces. Si la continuación sale vacía (de verdad ya terminó),
        se queda con lo que hay. Con rounds=0 equivale a listen_once.

        source: fuente ya abierta (capture.MicSession) para TODAS las rondas.
        Con el stream propio, la continuación abría el suyo tras decodificar la
        primera parte y el arranque de la frase que seguía se perdía; con la
        fuente, eso queda capturado y se lee primero. None = lo de siempre.
        """
        from crotolamo.voice.endpoint import seems_incomplete

        fuente: dict[str, Any] = {} if source is None else {"source": source}
        text = self._listen_transcribe(
            hotwords=hotwords, min_audio_s=min_audio_s, silence_ms=silence_ms,
            max_seconds=max_seconds, start_timeout_s=start_timeout_s, **fuente,
        )
        for _ in range(max(0, rounds)):
            if not text.strip() or not seems_incomplete(text):
                break
            extra = self._listen_transcribe(
                hotwords=hotwords, min_audio_s=min_audio_s, silence_ms=silence_ms,
                max_seconds=max_seconds, start_timeout_s=continue_timeout_s, **fuente,
            )
            if not extra.strip():
                break
            text = f"{text.strip()} {extra.strip()}"
        return text
