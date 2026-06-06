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

import time

from crotolamo.voice.stt import VoiceUnavailable, _require

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

        # Descarga perezosa del modelo puente pre-entrenado.
        try:
            from openwakeword.utils import download_models

            download_models([self.model_name])
        except Exception:
            pass  # ya descargado, o el nombre es una ruta a un .onnx propio

        # La API varía entre versiones: por nombre (wakeword_models) o por ruta.
        try:
            self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except TypeError:
            self._model = Model(wakeword_model_paths=[self.model_name], inference_framework="onnx")
        return self._model

    def available(self) -> bool:
        try:
            import openwakeword  # noqa: F401
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False

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
