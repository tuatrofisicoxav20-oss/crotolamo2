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
        # Buffers del preprocessor recién cargado (openwakeword < 0.6): reset()
        # los restaura de esta copia en vez de recalcular ~200 ms de embeddings.
        self._blank_state: dict[str, Any] | None = None
        # (score, umbral) del chunk anterior: feed() no repite en el log una
        # predicción que openWakeWord no recalculó (ver feed).
        self._last_score_key: tuple[float, float] | None = None

    @classmethod
    def from_settings(cls, settings) -> "WakeWordDetector":
        from crotolamo.voice.media_aware import sane_media_threshold

        wake = settings.wake
        threshold = wake.get("oww_threshold", 0.5)
        return cls(
            model_name=wake.get("oww_model", "hey_jarvis"),
            threshold=threshold,
            threshold_media=sane_media_threshold(
                threshold, wake.get("oww_threshold_media", 0.7), "[wake].oww_threshold_media",
            ),
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
        own = self._own_model_path()
        if own is not None:
            return own
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

    def _own_model_path(self) -> str | None:
        """Ruta al .onnx PROPIO de [wake].oww_model si existe; None si no.

        Se expanden ~ y $VARS como en el resto de rutas de la config (settings.py):
        aquí se usaba la cadena cruda y "~/modelos/x.onnx" no existía. Una ruta
        relativa se busca primero desde el CWD (lo que ya funcionaba no cambia) y
        luego desde la raíz del repo: la documentada
        (wakeword_training/my_custom_model/crotolamo.onnx) solo cargaba lanzando
        el proceso desde ahí, como hace el .service; desde otra terminal, no.
        """
        if not self.model_name.endswith(".onnx"):
            return None
        path = os.path.expandvars(os.path.expanduser(self.model_name))
        if os.path.isfile(path):
            return path
        if not os.path.isabs(path):
            from crotolamo.settings import REPO_ROOT

            anchored = os.path.join(REPO_ROOT, path)
            if os.path.isfile(anchored):
                return anchored
        return None

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
        # Un .onnx propio va por su ruta YA resuelta también en el primer intento:
        # 0.6+ solo acepta wakeword_models y busca ahí un archivo existente.
        own = self._own_model_path()

        # La firma del constructor cambia entre versiones de openWakeWord: unas aceptan
        # el NOMBRE (wakeword_models) y/o inference_framework; otras solo la RUTA al
        # .onnx (wakeword_model_paths) y son ONNX puro. Probamos en orden y nos
        # quedamos con la primera variante que cargue.
        attempts = (
            {"wakeword_models": [own or self.model_name], "inference_framework": "onnx"},
            {"wakeword_model_paths": [path], "inference_framework": "onnx"},
            {"wakeword_model_paths": [path]},
            {"wakeword_models": [path]},
        )
        cause: Exception | None = None
        for kwargs in attempts:
            try:
                self._model = Model(**kwargs)
            except Exception as error:  # noqa: BLE001 - TypeError o NoSuchFile, etc.
                # Un TypeError es una firma que no es de ESTA versión: se informa
                # el primer error que no lo sea (archivo inexistente, pesos que
                # faltan...). Antes salía el del ÚLTIMO intento, un TypeError de
                # firma que apuntaba a incompatibilidad y escondía la causa real.
                if cause is None or (isinstance(cause, TypeError)
                                     and not isinstance(error, TypeError)):
                    cause = error
                continue
            self._remember_blank_state()
            return self._model
        raise VoiceUnavailable(
            f"No pude cargar el wake word '{self.model_name}', patrón: {cause}"
        )

    def _remember_blank_state(self) -> None:
        """Copia los buffers del preprocessor recién cargado (antes de oír nada).

        Solo hace falta en openwakeword < 0.6, donde Model.reset() no toca el
        preprocessor; en 0.6+ el preprocessor trae su propio reset().
        """
        pre = getattr(self._model, "preprocessor", None)
        if pre is None or callable(getattr(pre, "reset", None)):
            return
        self._blank_state = {
            name: getattr(pre, name).copy()
            for name in ("melspectrogram_buffer", "feature_buffer", "raw_data_remainder")
            if hasattr(getattr(pre, name, None), "copy")
        }

    def reset(self) -> None:
        """Olvida el audio ya escuchado: el modelo queda como recién cargado.

        Se llama al DISPARAR. openWakeWord puntúa las últimas features (~1.3 s de
        audio) que guarda su preprocessor; tras un wake el oído deja de
        alimentarlo (LISTENING, THINKING, SPEAKING, gracia) y, al volver, el
        primer chunk de SILENCIO todavía puntuaba con el wake dentro: turno
        fantasma tras turno fantasma (un solo wake real abría más de 20), y en
        el modo simple listen_for_wake() devolvía True en su primer frame.
        Model.reset() solo vacía las predicciones en 0.4/0.5; los buffers del
        preprocessor se restauran aparte desde la copia en blanco.
        """
        model = self._model
        if model is None:
            return
        model_reset = getattr(model, "reset", None)
        if callable(model_reset):
            model_reset()  # prediction_buffer (en 0.6+ también el preprocessor)
        pre = getattr(model, "preprocessor", None)
        if pre is None or self._blank_state is None:
            return
        for name, value in self._blank_state.items():
            setattr(pre, name, value.copy())
        raw = getattr(pre, "raw_data_buffer", None)
        if raw is not None:
            raw.clear()
        if hasattr(pre, "accumulated_samples"):
            pre.accumulated_samples = 0

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
        except (ImportError, OSError):
            # OSError: sounddevice importa pero no encuentra libportaudio
            # ("PortAudio library not found"). Sin esto el listener reventaba
            # con traceback en vez de caer al modo simple.
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
        # Con chunks de 512 openWakeWord solo predice de nuevo cada 1280 muestras;
        # entre medias devuelve el MISMO score y la misma línea salía 2-3 veces
        # seguidas. Se omite la repetición exacta (mismo score y mismo umbral);
        # un disparo se loguea siempre: cada uno abre un turno.
        key = (mx, threshold)
        if mx > 0.05 and (fired or key != self._last_score_key):
            log.info("wake score=%.3f (umbral=%.2f) -> %s",
                     mx, threshold, "DISPARA" if fired else "no")
        self._last_score_key = key
        if fired:
            self.reset()  # el wake ya se atendió: que no vuelva a puntuar (ver reset)
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
                    # Sin reset, la siguiente espera disparaba en su PRIMER frame
                    # con este mismo wake todavía en los buffers (ver reset).
                    self.reset()
                    return True
        return False
