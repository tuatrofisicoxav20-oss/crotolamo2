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
import inspect
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from itertools import islice
from typing import TYPE_CHECKING, Any

from crotolamo.logging_setup import get_logger
from crotolamo.voice import runtime
from crotolamo.voice.stt import VoiceUnavailable, _require

if TYPE_CHECKING:
    from crotolamo.voice.media_aware import MediaMonitor

log = get_logger("voice.wakeword")

# openWakeWord espera frames de 1280 muestras (80 ms a 16 kHz).
_FRAME = 1280
_SAMPLE_RATE = 16000
# Predicciones que openWakeWord anula (0.0) tras cargar el modelo o tras
# Model.reset(): "Zero predictions for first 5 frames" en 0.4 y en 0.6. Cuenta
# LLAMADAS a predict, no frames: feed() las hace chunk a chunk, como antes, para
# que esos primeros scores sean los mismos (ver _stream_score).
_OWW_WARMUP_CALLS = 5
# Tope del audio crudo que se deja en el preprocessor de openWakeWord (ver
# _shrink_raw): medio segundo, sobra para la cola que lee cada frame.
_RAW_MAX = 8000
# Contexto que el melspectrograma del paquete toma de antes de las muestras sin
# frame (list(raw)[-acumulado - 160 * 3:], en 0.4 y en 0.6).
_MEL_CONTEXT = 160 * 3


def _shrink_raw(model: Any) -> None:
    """Recorta el audio crudo que guarda el preprocessor de openWakeWord.

    El paquete (0.4 y 0.6) lo guarda en un deque de 10 s y en CADA frame lo
    copia entero a una lista para quedarse con la cola: con el detector
    alimentado sin parar (el oído del loop concurrente; el pre-filtro, que ya
    no se vacía en cada ventana) eran 160000 muestras copiadas cada 80 ms,
    0.6-0.8 s de CPU por minuto (medido en 60 s de tele: 2.2 -> 1.4 CPU-s en
    0.4, 2.1 -> 1.6 en 0.6). Antes de cada predict solo se deja lo que ese
    frame puede leer: las muestras que aún no dieron frame (accumulated_samples)
    más el contexto del melspectrograma, y un frame de margen. Es lo único que
    el paquete lee de ese búfer, así que los scores son idénticos (medido chunk
    a chunk en 0.4 y en 0.6). Un preprocessor sin ese búfer no se toca.
    """
    pre: Any = getattr(model, "preprocessor", None)
    raw = getattr(pre, "raw_data_buffer", None)
    if not isinstance(raw, deque) or len(raw) <= _RAW_MAX:
        return
    try:
        keep = int(pre.accumulated_samples) + _MEL_CONTEXT + _FRAME
    except (AttributeError, TypeError, ValueError):
        return
    if keep < len(raw):
        pre.raw_data_buffer = deque(islice(raw, len(raw) - keep, None), maxlen=raw.maxlen)


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5,
                 debug: bool = False, threshold_media: float | None = None,
                 media: MediaMonitor | None = None, exact_frames: bool = False,
                 rearm_after_gaps: bool = True) -> None:
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
        # Lock de carga (doble comprobación en _get_model): WarmVoice puede estar
        # cargándolo cuando el oído pide su primer score; sin lock cada uno
        # construía su propio modelo y uno se tiraba.
        self._load_lock = threading.Lock()
        self._np: Any = None  # numpy cacheado: score() corre en CADA chunk (~31/s)
        # Buffers del preprocessor recién cargado (openwakeword < 0.6): reset()
        # los restaura de esta copia en vez de recalcular ~200 ms de embeddings.
        self._blank_state: dict[str, Any] | None = None
        # Episodio de señal en curso para el log de feed() (ver _log_score): pico
        # (None = sin episodio), último score que salió a INFO, decisión del
        # último chunk y su umbral.
        self._episode_peak: float | None = None
        self._episode_logged = 0.0
        self._episode_fired = False
        self._episode_threshold = threshold
        # Re-encuadre de feed() (ver _stream_score). exact_frames ([wake].
        # oww_exact_frames, decisión del dueño): frames exactos de 1280, los de
        # listen_for_wake y wake_debug.py; en 0.4 cambia scores frente al modo
        # de siempre, que reproduce lo que el paquete hace con chunks de 512.
        self.exact_frames = exact_frames
        # ¿El preprocessor guarda él mismo el resto que no completa frame?
        # (openwakeword 0.5+: raw_data_remainder). Se mira al cargar.
        self._carry = False
        self._held: list[Any] = []  # chunks aún no entregados a predict
        self._held_n = 0
        # Muestras que el paquete ya tiene y todavía no le dieron frame nuevo.
        self._lib_pending = 0
        self._calls = 0  # predicts desde la carga o el último reset
        self._cached = 0.0  # score del último predict: el de los chunks sin frame
        # [wake].oww_rearm: tras un hueco en la alimentación (escucha pausada,
        # una espera nueva de listen_for_wake) se olvida lo de antes (ver rearm).
        self.rearm_after_gaps = rearm_after_gaps
        self._dirty = False  # ¿oyó algo desde la carga o el último reset?
        # Por qué no cargó el modelo la última vez que se probó (can_load).
        self.load_error: str | None = None

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
            exact_frames=bool(wake.get("oww_exact_frames", False)),
            rearm_after_gaps=bool(wake.get("oww_rearm", True)),
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
            # La carpeta SIN ejecutar el __init__ del paquete (que arrastra sklearn
            # y scipy): runtime.package_file la localiza con find_spec.
            base = runtime.package_file("openwakeword", "resources", "models")
            if base is not None:
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

    def _get_model(self, download: bool = True):
        """Carga el modelo de openWakeWord una sola vez (descarga el puente si falta).

        También cachea numpy en self._np: score() corre en cada chunk de audio y
        resolver el import ahí (aunque esté cacheado en sys.modules) era costo
        por frame sin motivo. download=False: sin la descarga del paquete (ver
        can_load).
        """
        if self._np is None:
            self._np = _require("numpy")
        # Camino caliente sin lock: con el modelo ya cargado es leer un atributo.
        if self._model is not None:
            return self._model
        with self._load_lock:
            # Doble comprobación: otro hilo pudo cargarlo mientras esperábamos. Se
            # publica ya listo (estado en blanco copiado, arena soltado): nadie
            # puntúa con un modelo a medio preparar.
            if self._model is None:
                self._model = (self._load_model() if download
                               else self._load_model(download=False))
        return self._model

    def warm(self) -> bool:
        """Precalienta el modelo de openWakeWord: la misma carga que el primer
        score(), en el hilo de quien llama, para que no caiga en el primer chunk del
        oído (si otro hilo ya lo está cargando, espera a esa carga).

        Nunca lanza: si no carga no queda nada cacheado, y el primer uso real lo
        reintenta y da el mismo VoiceUnavailable de siempre (con él decide el
        listener caer al wake difuso). True si el modelo quedó cargado.
        """
        try:
            self._get_model()
        except Exception as error:  # noqa: BLE001 - lo decide y lo avisa el primer uso real
            log.debug("no precalenté el wake '%s' (%s); lo decidirá su primer uso",
                      self.model_name, error)
            return False
        return True

    def can_load(self, *, download: bool = True) -> bool:
        """¿Carga DE VERDAD el modelo configurado? available() solo dice que el
        paquete se importa: con un oww_model que no carga (ruta que no existe, un
        .onnx al que le falta su .onnx.data) se elegía el loop concurrente y el
        oído moría en su primer chunk.

        Es la misma carga única de _get_model (la que luego usa el oído: no se
        carga dos veces) y nunca lanza: si no carga devuelve False y deja la
        causa en load_error, el texto del VoiceUnavailable de siempre. Avisar es
        cosa de quien decide qué hacer con eso (el listener cae al wake difuso).
        Como _get_model, un fallo no se memoriza: otra llamada lo reintenta.

        download=False (el doctor): sin la descarga de openwakeword.utils, que va
        por red sin timeout; un modelo que no está en disco no carga, con su causa.
        Con los archivos presentes es la misma carga (la descarga no hace nada).
        """
        try:
            # Sin argumentos en la carga de siempre: la firma que conocen los
            # dobles de prueba y el bench que sustituyen _get_model.
            if download:
                self._get_model()
            else:
                self._get_model(download=False)
        except VoiceUnavailable as error:
            self.load_error = str(error)
            return False
        self.load_error = None
        return True

    def pending_downloads(self) -> list[str]:
        """Los .onnx que esta carga necesita y que la descarga del paquete (la de
        _load_model, al arrancar el servicio) aún tiene que bajar. [] si no falta
        nada o si el paquete no descarga (0.4 los trae en la rueda).

        Para el doctor, que no descarga (red sin timeout): con un openwakeword
        0.5+ recién instalado no están ni hey_jarvis ni los modelos de features,
        la carga sin descarga falla y el servicio, que sí descarga, carga bien.
        Imita la decisión de openwakeword.utils.download_models (baja X.tflite y
        X.onnx si falta X.tflite) y solo cuenta un .onnx que falte y que esa
        descarga traería. Nunca lanza: ante un paquete que no se entiende, []
        (y el doctor prueba la carga de siempre).
        """
        try:
            oww = runtime.import_openwakeword()
            from openwakeword.utils import download_models
        except ImportError:
            return []
        try:
            target = inspect.signature(download_models).parameters["target_directory"].default
            urls = [spec["download_url"] for spec in oww.FEATURE_MODELS.values()]
            # El VAD también se descarga, pero el wake no lo carga. Del modelo
            # pedido, la primera URL oficial cuyo archivo contiene el nombre: un
            # .onnx propio o un nombre mal escrito no casan con ninguna.
            official = [spec["download_url"] for spec in oww.MODELS.values()]
            urls += [url for url in official if self.model_name in url.split("/")[-1]][:1]
            pending = []
            for url in urls:
                fetched = url.split("/")[-1]
                onnx = fetched.replace(".tflite", ".onnx")
                if not (os.path.exists(os.path.join(target, fetched))
                        or os.path.exists(os.path.join(target, onnx))):
                    pending.append(onnx)
            return pending
        except Exception as error:  # noqa: BLE001 - otra versión del paquete: no se adivina
            log.debug("no sé qué descargaría openwakeword para '%s' (%s)", self.model_name,
                      error)
            return []

    def _load_model(self, download: bool = True):
        try:
            # Sin el módulo de ENTRENAMIENTO del paquete (sklearn + scipy: +87 MB y
            # +3 hilos de OpenBLAS para nada); ver runtime.import_openwakeword.
            runtime.import_openwakeword()
            from openwakeword.model import Model
        except ImportError as error:
            raise VoiceUnavailable(
                "Falta openwakeword, patrón. Instala con: pip install -e '.[voice]'."
            ) from error

        # Descarga perezosa del modelo puente pre-entrenado (si falta).
        try:
            if download:
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
        # .onnx (wakeword_model_paths) y son ONNX puro. Va primero la variante que
        # declara la firma instalada (en 0.4 los intentos a ciegas construían 6
        # sesiones ONNX que se tiraban) y detrás la lista de siempre, en orden; nos
        # quedamos con la primera que cargue.
        attempts = runtime.oww_constructor_kwargs(Model, own or self.model_name, path)
        cause: Exception | None = None
        for kwargs in attempts:
            try:
                model = Model(**kwargs)
            except Exception as error:  # noqa: BLE001 - TypeError o NoSuchFile, etc.
                # Un TypeError es una firma que no es de ESTA versión: se informa
                # el primer error que no lo sea (archivo inexistente, pesos que
                # faltan...). Antes salía el del ÚLTIMO intento, un TypeError de
                # firma que apuntaba a incompatibilidad y escondía la causa real.
                if cause is None or (isinstance(cause, TypeError)
                                     and not isinstance(error, TypeError)):
                    cause = error
                continue
            self._remember_blank_state(model)
            self._carry = hasattr(getattr(model, "preprocessor", None), "raw_data_remainder")
            # El calentamiento de AudioFeatures (10 s de ceros) deja su pico en el
            # arena de ORT para siempre (~200 MB): se devuelve. No cambia scores.
            runtime.release_oww_warmup_arena(model)
            return model
        raise VoiceUnavailable(
            f"No pude cargar el wake word '{self.model_name}', patrón: {cause}"
        )

    def _remember_blank_state(self, model) -> None:
        """Copia los buffers del preprocessor recién cargado (antes de oír nada).

        Solo hace falta en openwakeword < 0.6, donde Model.reset() no toca el
        preprocessor; en 0.6+ el preprocessor trae su propio reset().
        """
        pre = getattr(model, "preprocessor", None)
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

        Lo que el re-encuadre de feed() guardaba también es audio de antes: se
        tira, y el calentamiento del paquete (que Model.reset() reabre) vuelve
        a ir chunk a chunk. Un episodio de señal a medias (el rearm de una
        pausa) se cierra en el log con su pico (ver _log_score).
        """
        self._close_episode()
        self._held = []
        self._held_n = 0
        self._lib_pending = 0
        self._calls = 0
        self._cached = 0.0
        self._dirty = False
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

    def rearm(self) -> None:
        """Olvida lo oído antes de un HUECO: el detector dejó de recibir audio
        (escucha pausada desde el panel, una espera nueva de listen_for_wake con
        su stream recién abierto) y lo que guarda no es lo que suena ahora.

        La ventana que puntúa openWakeWord son las últimas 16 features (1.28 s)
        y sin alimentarlo no avanza: medido con el modelo real (0.4 y 0.6), un
        "Hey Jarvis" cortado por la pausa entre 32 y 192 ms antes de su disparo
        abría un turno fantasma a los 0.03-0.3 s de reanudar, durase lo que
        durase la pausa (15 de 15 casos; con rearm, 0).

        Es el mismo reset() del disparo, y solo si oyó algo desde el último:
        tras un turno (que empezó con un disparo) no hace nada, así que lo que
        pasa al volver de un turno no cambia. Un wake dicho justo al reanudar
        dispara como uno dicho al volver de un turno (en 18 de 18 casos medidos,
        en el chunk de un modelo recién cargado). [wake].oww_rearm=false lo
        apaga: la ventana vieja se queda, como antes.
        """
        if self.rearm_after_gaps and self._dirty:
            self.reset()

    def available(self) -> bool:
        """True si openwakeword (y sounddevice) están importables.

        Coherencia con VoiceLoop / listener.py:
        - openWakeWord es REQUERIDO para el modo concurrente; no tiene fallback
          en el EarThread (wake_fn=wake_detector.feed).
        - silero-vad es OPCIONAL: _SileroVad en loop.py cae a energía RMS si falta.
          Por eso silero NO forma parte de esta comprobación.
        - Si available() devuelve False, listener.py cae al modo simple (Whisper
          difuso), que no usa openWakeWord. Si el MODELO configurado carga lo
          dice can_load(), no esto.
        """
        try:
            # Sin sklearn/scipy (runtime.import_openwakeword): esta decisión de modo
            # corre en el hilo principal al arrancar y costaba ~2 s y +87 MB.
            # sounddevice sí se importa de verdad: su OSError es lo que se diagnostica.
            runtime.import_openwakeword()
            import sounddevice  # noqa: F401

            return True
        except (ImportError, OSError):
            # OSError: sounddevice importa pero no encuentra libportaudio
            # ("PortAudio library not found"). Sin esto el listener reventaba
            # con traceback en vez de caer al modo simple.
            return False

    def feed(self, chunk) -> bool:
        """Alimenta un chunk al detector y devuelve True si supera el umbral (M3.6).

        El micrófono entrega chunks de 512 muestras (32 ms: el frame que exige
        Silero) y openWakeWord solo calcula features nuevas cada 1280 muestras
        (80 ms). Medido en la 0.4 instalada: predict() corre el clasificador en
        CADA llamada, pero el preprocessor solo saca un embedding nuevo al
        juntar >= 1280 muestras, así que con 512 (uno cada 1536) 2 de cada 3
        predicciones re-puntuaban las MISMAS features; la 0.5+ guarda el resto y
        devuelve la predicción anterior. El score sale siempre de las últimas
        16 features (1.28 s de audio), no del chunk.

        Por eso feed() junta los chunks y llama a predict solo cuando el paquete
        calcularía features nuevas; entre medias devuelve el score guardado, que
        es exactamente el que habría devuelto el paquete. Mismos scores y mismas
        decisiones en cada chunk con una llamada por frame: en 0.4, un tercio de
        las de antes (ver _stream_score). En CPU es poco: las llamadas que se
        ahorran solo corrían el clasificador (~0.08 ms; lo caro, melspec y
        embedding, ya iba una vez por frame): 2-4 % del wake en 0.4 (0.05-0.10 s
        de CPU por minuto alimentado) y <1 % en 0.5+, que ya no recalculaba nada.
        score() sigue siendo la llamada cruda, una por chunk.

        openWakeWord espera int16; convertimos si llega en float.
        """
        mx, fresh = self.stream_score(chunk)
        # Umbral EFECTIVO (sube con música sonando): es el que se loguea, para
        # que el log explique por qué un score que ayer disparaba hoy no.
        threshold = self.effective_threshold()
        fired = mx >= threshold
        if self.debug:
            print(f"[wake] score={mx:.3f} umbral={threshold:.2f}"
                  f"{'  <<< DISPARA' if fired else ''}", flush=True)
        self._log_score(mx, threshold, fired, fresh=fresh)
        if fired:
            # El disparo cierra el episodio: su línea ya salió y no lleva "fin".
            self._episode_peak = None
            self.reset()  # el wake ya se atendió: que no vuelva a puntuar (ver reset)
        return fired

    def _log_score(self, mx: float, threshold: float, fired: bool, fresh: bool) -> None:
        """Diagnóstico proporcional a los EVENTOS, no al audio.

        Sirve para ver qué score da la voz REAL del patrón y afinar el umbral.
        Antes salía una línea INFO por cada predicción nueva con señal (> 0.05):
        con música o voz que rondan el modelo, una cada 80-96 ms en el journal
        mientras durase. Ahora un episodio de señal da, a INFO, su primer score,
        cada cambio de decisión (dispara / no dispara; también el que causa el
        umbral de la música con el mismo score) y su fin con el pico, si el
        pico no salió ya (un chispazo de un solo score es una línea, como
        antes). Un disparo sale siempre: o abre su episodio o es un cambio de
        decisión. Los scores de en medio van a DEBUG, uno por predicción nueva
        (no por chunk: el score guardado de feed() no es información nueva).
        """
        if mx <= 0.05:
            self._close_episode()
            return
        if self._episode_peak is None or fired != self._episode_fired:
            log.info("wake score=%.3f (umbral=%.2f) -> %s",
                     mx, threshold, "DISPARA" if fired else "no")
            self._episode_logged = mx
        elif fresh:
            log.debug("wake score=%.3f (umbral=%.2f) -> %s",
                      mx, threshold, "DISPARA" if fired else "no")
        self._episode_peak = max(mx, self._episode_peak or 0.0)
        self._episode_fired = fired
        self._episode_threshold = threshold

    def _close_episode(self) -> None:
        if self._episode_peak is not None and self._episode_peak > self._episode_logged:
            log.info("wake pico=%.3f (umbral=%.2f): fin de la señal",
                     self._episode_peak, self._episode_threshold)
        self._episode_peak = None
        self._episode_fired = False

    def stream_score(self, chunk) -> tuple[float, bool]:
        """(score, nuevo) de un chunk por el MISMO re-encuadre que feed(), sin
        umbral, log ni reset: lo que mide scripts/wake_debug.py --como-produccion.
        nuevo=False es el score guardado de un chunk que no completó frame (el
        mismo que el anterior: el paquete no calculó nada).
        """
        calls = self._calls
        mx = self._stream_score(chunk)
        return mx, self._calls != calls

    def score(self, chunk) -> float:
        """Score crudo de openWakeWord para un chunk (máximo entre los modelos).

        Separado de feed() para que las herramientas de calibración
        (scripts/wake_debug.py) puedan medir sin duplicar la conversión ni el
        estado del preprocessor.
        """
        model = self._get_model()
        self._dirty = True
        return self._predict(model, self._as_int16(chunk))

    def _as_int16(self, chunk):
        np = self._np  # cacheado por _get_model(): no re-resolver en el hot path
        arr = np.asarray(chunk)
        if arr.dtype != np.int16:
            arr = (arr * 32767).astype(np.int16)
        return arr

    @staticmethod
    def _predict(model, arr) -> float:
        _shrink_raw(model)
        scores = model.predict(arr)
        return float(max(scores.values())) if scores else 0.0

    def _stream_score(self, chunk) -> float:
        """Score de feed() para este chunk, con un predict solo cuando hace falta.

        Imita el encuadre del paquete en vez de pelear con él: lleva la cuenta
        de las muestras que el preprocessor tiene sin frame (_lib_pending) y
        guarda los chunks hasta el que completa uno; entonces entrega todo
        junto, que es lo que el preprocessor habría hecho con ellos por
        separado (mismo búfer crudo, mismo frame, mismo chunk). Tras el frame,
        0.4 descarta la cuenta y 0.5+ guarda el resto (_carry).

        Las _OWW_WARMUP_CALLS primeras predicciones tras cargar o resetear las
        anula el propio paquete contando LLAMADAS: van chunk a chunk, como
        antes, más una viva para que el score guardado sea uno real. Sin esto,
        el arranque y cada reset dejaban sorda la ventana 3 veces más tiempo.

        exact_frames: frames de 1280 exactos y el resto espera aquí (el
        encuadre de listen_for_wake y de wake_debug.py). Cambia scores en 0.4;
        en 0.5+ el preprocessor ya encuadra así y el camino normal es idéntico.
        """
        model = self._get_model()
        self._dirty = True
        arr = self._as_int16(chunk)
        if self.exact_frames and not self._carry:
            return self._exact_frame_score(model, arr)
        total = self._lib_pending + self._held_n + len(arr)
        fresh = total >= _FRAME
        if not fresh and self._calls > _OWW_WARMUP_CALLS:
            # Sin frame nuevo el paquete devolvería lo mismo: 0.4 re-corre el
            # clasificador sobre las mismas features, 0.5+ repite la anterior.
            self._held.append(arr)
            self._held_n += len(arr)
            return self._cached
        if self._held:
            self._held.append(arr)
            arr = self._np.concatenate(self._held)
            self._held = []
            self._held_n = 0
        self._cached = self._predict(model, arr)
        self._calls += 1
        if fresh:
            self._lib_pending = total % _FRAME if self._carry else 0
        else:
            self._lib_pending = total
        return self._cached

    def _exact_frame_score(self, model, arr) -> float:
        if self._held:
            self._held.append(arr)
            arr = self._np.concatenate(self._held)
        usable = len(arr) - len(arr) % _FRAME
        rest = arr[usable:]
        self._held = [rest] if len(rest) else []
        self._held_n = len(rest)
        if usable:
            self._cached = self._predict(model, arr[:usable])
            self._calls += 1
        return self._cached

    def listen_for_wake(self, timeout_s: float | None = None, source: Any = None,
                        should_stop: Callable[[], bool] | None = None) -> bool:
        """Escucha el micrófono y devuelve True al detectar la palabra de activación.

        Lee en chunks de 80 ms y los alimenta al detector. Si `timeout_s` se agota
        sin activación, devuelve False.

        source: fuente ya abierta (capture.MicSession) en vez de un stream propio:
        el modelo recibe los mismos frames de 1280 en int16 (la sesión re-encuadra
        sus chunks float32 de 512; ver _wake_de_fuente) y lo que sigue al frame
        que dispara se queda en ella para la orden. should_stop (solo con source): se consulta en un tick de
        1 s (o cuando pasa 1 s sin audio) y con True devuelve False: con
        timeout_s=None la espera ya no es ciega a una pausa o un apagado.
        """
        sd = _require("sounddevice")
        np = _require("numpy")
        model = self._get_model()
        # El stream de la espera anterior se cerró: lo que el modelo oyó ahí es
        # de antes del hueco (ver rearm). Tras un disparo no hace nada.
        self.rearm()

        start = time.monotonic()
        if source is not None:
            return self._wake_de_fuente(model, np, source, timeout_s, should_stop, start)
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
                _shrink_raw(model)
                scores = model.predict(audio)
                self._dirty = True
                # Umbral efectivo por frame: la música puede arrancar o parar
                # mientras esperamos el wake (el monitor lo refleja en su caché).
                if scores and max(scores.values()) >= self.effective_threshold():
                    # Sin reset, la siguiente espera disparaba en su PRIMER frame
                    # con este mismo wake todavía en los buffers (ver reset).
                    self.reset()
                    return True
        return False

    def _wake_de_fuente(self, model: Any, np: Any, source: Any, timeout_s: float | None,
                        should_stop: Callable[[], bool] | None, start: float) -> bool:
        """listen_for_wake sobre una fuente ya abierta: mismo frame, mismo predict
        y misma decisión por frame que con el stream propio de int16."""
        tick = start
        while timeout_s is None or (time.monotonic() - start) < timeout_s:
            block = source.read(timeout=1.0, frames=_FRAME)
            if should_stop is not None:
                ahora = time.monotonic()
                if block is None or ahora - tick >= 1.0:
                    tick = ahora
                    if should_stop():
                        return False
            if block is None:
                # Una fuente cerrada vuelve al instante y no traerá más audio.
                if getattr(source, "closed", False):
                    return False
                continue
            # float32 -> int16 como _as_int16 en feed() (el oído del loop
            # concurrente, que también lee float32); el clip solo evita que algo
            # fuera de [-1, 1] dé la vuelta. Un stream de int16 lo convierte
            # PortAudio, que puede redondear 1 LSB distinto: eso solo se ve en el
            # dispositivo real.
            audio = np.clip(np.asarray(block, dtype=np.float32) * 32767,
                            -32768, 32767).astype(np.int16)
            _shrink_raw(model)
            scores = model.predict(audio)
            self._dirty = True
            if scores and max(scores.values()) >= self.effective_threshold():
                self.reset()
                return True
        return False
