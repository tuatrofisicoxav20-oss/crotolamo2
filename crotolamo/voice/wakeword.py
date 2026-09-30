"""Wake word ligero con openWakeWord (mejora I2 / M1).

Antes: el listener corría Whisper completo en bucle solo para oír "crotolamo"
(lento, quema CPU). openWakeWord es un detector dedicado y barato que escucha en
chunks de 80 ms y solo despierta a Whisper cuando hay activación.

NOTA HONESTA: "crotolamo" no tiene modelo pre-entrenado en openWakeWord. Usamos un
modelo PUENTE soportado ("hey_jarvis" por defecto) hasta entrenar uno propio de
"crotolamo" (paso aparte). El detector difuso de wake.py se conserva como fallback
y sigue usándose para strip_wake_word del comando.

Imports pesados (openwakeword, sounddevice, numpy) perezosos: el módulo importa sin
la extra [voice]; solo al escuchar se exige.
"""

from __future__ import annotations

import glob
import os
import time
from typing import TYPE_CHECKING, Any

from crotolamo.logging_setup import get_logger
from crotolamo.voice.stt import VoiceUnavailable, _require

if TYPE_CHECKING:
    from crotolamo.voice.media_aware import MediaMonitor

log = get_logger("voice.wakeword")

# openWakeWord espera frames de 1280 muestras (80 ms a 16 kHz).
_FRAME = 1280
_SAMPLE_RATE = 16000


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5,
                 debug: bool = False, threshold_media: float | None = None,
                 media: MediaMonitor | None = None) -> None:
        self.model_name = model_name
        self.threshold = threshold
        # Umbral con MÚSICA sonando (ver voice/media_aware.py): la música mete
        # falsos disparos, así que se exige más. None = un solo umbral, siempre.
        self.threshold_media = threshold_media
        self._media = media
        # Modo debug (validación de mic): imprime CADA score en tiempo real por
        # stdout para calibrar el umbral viendo qué da la voz real vs el ruido.
        self.debug = debug
        self._model = None  # se carga una sola vez
        self._np: Any = None  # numpy cacheado: score() corre en CADA chunk (~31/s)

    @classmethod
    def from_settings(cls, settings) -> "WakeWordDetector":
        wake = settings.wake
        return cls(
            model_name=wake.get("oww_model", "hey_jarvis"),
            threshold=wake.get("oww_threshold", 0.5),
            threshold_media=wake.get("oww_threshold_media", 0.7),
            debug=os.environ.get("CROTOLAMO_WAKE_DEBUG", "") not in ("", "0"),
        )

    def attach_media(self, monitor: MediaMonitor | None) -> None:
        """Engancha el MediaMonitor: con música en Playing rige threshold_media."""
        self._media = monitor

    def effective_threshold(self) -> float:
        """Umbral vigente para ESTE chunk: el de música si el monitor dice que
        suena algo (lee su caché: no bloquea), el normal en cualquier otro caso.
        Sin monitor o sin threshold_media, idéntico al umbral fijo de siempre.
        """
        if self.threshold_media is None or self._media is None:
            return self.threshold
        try:
            playing = self._media.is_playing()
        except Exception as error:  # noqa: BLE001 - un monitor roto no cambia el wake
            log.debug("media monitor falló (%s); uso el umbral normal", error)
            return self.threshold
        return self.threshold_media if playing else self.threshold

    def _resolve_model_path(self) -> str:
        """Resuelve el nombre del modelo (p.ej. 'hey_jarvis') a la ruta de su .onnx.

        Algunas versiones de openWakeWord cargan modelos por NOMBRE y otras solo por
        RUTA al .onnx. Si model_name ya apunta a un .onnx existente (modelo propio),
        se usa tal cual; si no, se busca '{model_name}*.onnx' entre los modelos
        pre-entrenados que openWakeWord trae en resources/models
        (p.ej. 'hey_jarvis' -> 'hey_jarvis_v0.1.onnx').
        """
        if self.model_name.endswith(".onnx") and os.path.isfile(self.model_name):
            return self.model_name
        try:
            import openwakeword

            base = os.path.join(
                os.path.dirname(openwakeword.__file__), "resources", "models"
            )
            matches = sorted(glob.glob(os.path.join(base, f"{self.model_name}*.onnx")))
            if matches:
                return matches[0]
        except Exception as error:  # noqa: BLE001 - sin resolver, el nombre crudo
            log.debug("no pude resolver la ruta del modelo '%s': %s",
                      self.model_name, error)
        return self.model_name

    def _get_model(self):
        """Carga el modelo de openWakeWord una sola vez (descarga el puente si falta).

        También cachea numpy en self._np: score() corre en cada chunk de audio y
        resolver el import ahí (aunque esté cacheado en sys.modules) era costo
        por frame sin motivo.
        """
        if self._np is None:
            self._np = _require("numpy")
        if self._model is not None:
            return self._model
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise VoiceUnavailable(
                "Falta openwakeword, patrón. Instala con: pip install -e '.[voice]'."
            ) from error

        # Descarga perezosa del modelo puente pre-entrenado (si falta).
        try:
            from openwakeword.utils import download_models

            download_models([self.model_name])
        except Exception as error:  # noqa: BLE001
            # Ya descargado, o el nombre es una ruta a un .onnx propio.
            log.debug("descarga del modelo '%s' omitida: %s", self.model_name, error)

        path = self._resolve_model_path()

        # La firma del constructor cambia entre versiones de openWakeWord: unas aceptan
        # el NOMBRE (wakeword_models) y/o inference_framework; otras solo la RUTA al
        # .onnx (wakeword_model_paths) y son ONNX puro. Probamos en orden y nos
        # quedamos con la primera variante que cargue.
        attempts = (
            {"wakeword_models": [self.model_name], "inference_framework": "onnx"},
            {"wakeword_model_paths": [path], "inference_framework": "onnx"},
            {"wakeword_model_paths": [path]},
            {"wakeword_models": [path]},
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                self._model = Model(**kwargs)
                return self._model
            except Exception as error:  # noqa: BLE001 - TypeError o NoSuchFile, etc.
                last_error = error
                continue
        raise VoiceUnavailable(
            f"No pude cargar el wake word '{self.model_name}', patrón: {last_error}"
        )

    def available(self) -> bool:
        """True si openwakeword (y sounddevice) están importables.

        Coherencia con VoiceLoop / listener.py:
        - openWakeWord es REQUERIDO para el modo concurrente; no tiene fallback
          en el EarThread (wake_fn=wake_detector.feed).
        - silero-vad es OPCIONAL: _SileroVad en loop.py cae a energía RMS si falta.
          Por eso silero NO forma parte de esta comprobación.
        - Si available() devuelve False, listener.py cae al modo simple (Whisper
          difuso), que no usa openWakeWord.
        """
        try:
            import openwakeword  # noqa: F401
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False

    def feed(self, chunk) -> bool:
        """Alimenta un chunk al detector y devuelve True si supera el umbral (M3.6).

        openWakeWord espera frames de 1280 muestras (80 ms a 16 kHz) IDEALMENTE,
        pero su preprocessor interno mantiene estado entre llamadas y acepta chunks
        más cortos (verificado en openwakeword/model.py:predict, rama ``else`` en
        línea 219). Con el frame=512 de _RealMic (32 ms), el preprocessor acumula
        historia y hace inferencia correctamente; la detección puede ser ligeramente
        menos reactiva (latencia ~80 ms en vez de ~32 ms) pero es funcional.

        Si el comportamiento fuera problemático, la solución es bufferizar aquí
        hasta 1280 muestras antes de llamar a model.predict. Por ahora se documenta
        como tolerable y no se cambia el frame del micrófono (Silero exige 512).

        openWakeWord espera int16; convertimos si llega en float.
        """
        mx = self.score(chunk)
        # Umbral EFECTIVO (sube con música sonando): es el que se loguea, para
        # que el log explique por qué un score que ayer disparaba hoy no.
        threshold = self.effective_threshold()
        fired = mx >= threshold
        if self.debug:
            print(f"[wake] score={mx:.3f} umbral={threshold:.2f}"
                  f"{'  <<< DISPARA' if fired else ''}", flush=True)
        # Log de diagnóstico: solo cuando hay señal (>0.05), para ver qué score da
        # la voz REAL del patrón y afinar el umbral sin inundar el log con silencio.
        if mx > 0.05:
            log.info("wake score=%.3f (umbral=%.2f) -> %s",
                     mx, threshold, "DISPARA" if fired else "no")
        return fired

    def score(self, chunk) -> float:
        """Score crudo de openWakeWord para un chunk (máximo entre los modelos).

        Separado de feed() para que las herramientas de calibración
        (scripts/wake_debug.py) puedan medir sin duplicar la conversión ni el
        estado del preprocessor.
        """
        model = self._get_model()
        np = self._np  # cacheado por _get_model(): no re-resolver en el hot path
        arr = np.asarray(chunk)
        if arr.dtype != np.int16:
            arr = (arr * 32767).astype(np.int16)
        scores = model.predict(arr)
        return float(max(scores.values())) if scores else 0.0

    def listen_for_wake(self, timeout_s: float | None = None) -> bool:
        """Escucha el micrófono y devuelve True al detectar la palabra de activación.

        Lee en chunks de 80 ms y los alimenta al detector. Si `timeout_s` se agota
        sin activación, devuelve False.
        """
        sd = _require("sounddevice")
        np = _require("numpy")
        model = self._get_model()

        start = time.monotonic()
        # input_device: None = default del sistema; "crotolamo_aec_source" tras el AEC
        # (ver desktop/aec.sh). Es el mismo micrófono que usa el resto de la voz.
        try:
            from crotolamo.settings import get_settings

            _device = get_settings().voice.get("input_device")
        except Exception as error:  # noqa: BLE001 — sin config, el default del sistema
            log.debug("sin config de voz (%s); uso el micrófono default", error)
            _device = None
        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16",
                            device=_device) as stream:
            while timeout_s is None or (time.monotonic() - start) < timeout_s:
                block, _ = stream.read(_FRAME)
                audio = np.squeeze(np.asarray(block, dtype=np.int16))
                scores = model.predict(audio)
                # Umbral efectivo por frame: la música puede arrancar o parar
                # mientras esperamos el wake (el monitor lo refleja en su caché).
                if scores and max(scores.values()) >= self.effective_threshold():
                    return True
        return False
