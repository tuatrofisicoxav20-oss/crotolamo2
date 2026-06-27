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

from crotolamo.logging_setup import get_logger
from crotolamo.voice.stt import VoiceUnavailable, _require

log = get_logger("voice.wakeword")

# openWakeWord espera frames de 1280 muestras (80 ms a 16 kHz).
_FRAME = 1280
_SAMPLE_RATE = 16000


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = None  # se carga una sola vez

    @classmethod
    def from_settings(cls, settings) -> "WakeWordDetector":
        wake = settings.wake
        return cls(
            model_name=wake.get("oww_model", "hey_jarvis"),
            threshold=wake.get("oww_threshold", 0.5),
        )

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
        except Exception:  # noqa: BLE001 - sin resolver, devolvemos el nombre crudo
            pass
        return self.model_name

    def _get_model(self):
        """Carga el modelo de openWakeWord una sola vez (descarga el puente si falta)."""
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
        except Exception:
            pass  # ya descargado, o el nombre es una ruta a un .onnx propio

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
        np = _require("numpy")
        model = self._get_model()
        arr = np.asarray(chunk)
        if arr.dtype != np.int16:
            arr = (arr * 32767).astype(np.int16)
        scores = model.predict(arr)
        mx = max(scores.values()) if scores else 0.0
        # Log de diagnóstico: solo cuando hay señal (>0.05), para ver qué score da
        # la voz REAL del patrón y afinar el umbral sin inundar el log con silencio.
        if mx > 0.05:
            log.info("wake score=%.3f (umbral=%.2f) -> %s",
                     mx, self.threshold, "DISPARA" if mx >= self.threshold else "no")
        return bool(mx >= self.threshold)

    def listen_for_wake(self, timeout_s: float | None = None) -> bool:
        """Escucha el micrófono y devuelve True al detectar la palabra de activación.

        Lee en chunks de 80 ms y los alimenta al detector. Si `timeout_s` se agota
        sin activación, devuelve False.
        """
        sd = _require("sounddevice")
        np = _require("numpy")
        model = self._get_model()

        start = time.monotonic()
        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            while timeout_s is None or (time.monotonic() - start) < timeout_s:
                block, _ = stream.read(_FRAME)
                audio = np.squeeze(np.asarray(block, dtype=np.int16))
                scores = model.predict(audio)
                if scores and max(scores.values()) >= self.threshold:
                    return True
        return False
