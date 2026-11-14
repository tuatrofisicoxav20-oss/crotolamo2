"""Dueño ÚNICO de cómo se cargan los modelos de voz en el proceso 24/7.

Por qué existe: cada modelo (Silero, openWakeWord, Whisper, Piper) entraba por el
paquete de conveniencia de su vendor, con su superficie de imports y sus defaults
de sesión, pensados para scripts cortos o para entrenar, no para un servicio que
vive semanas en un portátil sin GPU. Medido con los cargadores reales:

- Silero: silero_vad importa torch (+218 MB, ~950 módulos, en el hilo del oído)
  solo como contenedor de tensores de un ONNX de 2.3 MB que onnxruntime ya corre.
- openWakeWord: `import openwakeword` ejecuta custom_verifier_model, que arrastra
  sklearn y scipy (+87 MB, +3 hilos de OpenBLAS) para una función de ENTRENAR;
  el bucle de firmas construía 9 sesiones ONNX para quedarse con 3; y el arena
  de ORT retenía para siempre el pico del calentamiento de 10 s de ceros.
- Whisper: resolvía el nombre en el Hub en cada arranque, y las barras de tqdm
  DESACTIVADAS de faster-whisper arrancaban un hilo monitor que no termina.
- Piper: el arena retiene el pico de la frase más larga dicha jamás.

Reglas de este módulo: a nivel de módulo solo la stdlib (numpy y onnxruntime se
importan dentro de cada función: importar esto no cuesta nada), y cada atajo cae
SOLO al camino del vendor de siempre si la versión instalada no encaja (archivo
movido, firma distinta, API nueva). Lo que hay aquí no cambia ninguna salida:
mismas probabilidades, mismos scores, mismo audio.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
import types
from typing import Any

from crotolamo.logging_setup import get_logger

log = get_logger("voice.runtime")

_MISSING = object()

# Avisos que deben salir UNA vez por proceso (un respaldo al cargador del vendor
# no cambia mientras el proceso viva: repetirlo solo ensucia el journal).
_warned: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    log.warning(msg, *args)


# --- localizar sin importar ---------------------------------------------------


def module_available(name: str) -> bool:
    """¿Se puede importar `name`? Sin importarlo (find_spec). Para nombres de
    primer nivel: con un nombre con puntos, find_spec SÍ importa el paquete padre.

    Mira sys.modules antes que find_spec para respetar los dos idiomas de los
    tests: `sys.modules[name] = None` (falta) y un `types.ModuleType` falso sin
    __spec__ (está; find_spec lanzaría ValueError con él).
    """
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def package_file(package: str, *parts: str) -> str | None:
    """Ruta de un archivo (o carpeta) de datos que trae `package`, o None.

    SIN ejecutar el __init__ del paquete: localizar el ONNX de Silero o la carpeta
    de modelos de openWakeWord importando el paquete entero es justo lo que trae
    torch o sklearn. Si el paquete ya está en sys.modules (real o un falso de los
    tests) se usan su __path__ o su __file__, como hacía el código de antes.
    """
    module: Any = sys.modules.get(package, _MISSING)
    if module is None:
        return None
    bases: list[str]
    if module is not _MISSING:
        bases = list(getattr(module, "__path__", None) or [])
        if not bases and getattr(module, "__file__", None):
            bases = [os.path.dirname(module.__file__)]
    else:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            return None
        bases = list(spec.submodule_search_locations or []) if spec is not None else []
    for base in bases:
        path = os.path.join(base, *parts)
        if os.path.exists(path):
            return path
    return None


# --- sesiones ONNX Runtime ----------------------------------------------------

# Al terminar un run con esta opción, ORT devuelve al sistema los bloques del
# arena que ese run dejó libres. Sin ella el arena se queda, de por vida, con el
# pico de memoria del run más grande que haya visto.
_SHRINK_KEY = "memory.enable_memory_arena_shrinkage"


def shrink_run_options():
    """RunOptions que encogen el arena de CPU al terminar el run."""
    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry(_SHRINK_KEY, "cpu:0")
    return options


class ArenaShrinkingSession:
    """Proxy de InferenceSession que encoge el arena tras los runs LARGOS.

    Un run cuya entrada principal (`input_feed[input_name]`, último eje) mide más
    de `keep_len` devuelve al terminar la memoria que necesitó; los cortos dejan
    el arena como hoy, porque encoger en CADA run costó +10 a 25 % de tiempo por
    frase (páginas que se vuelven a pedir). keep_len=None: encoge siempre. Unas
    run_options del llamador nunca se pisan. Todo lo demás se delega.

    Si ORT rechaza el encogido (una sesión sin arena de CPU: un ORT con mimalloc,
    o un vendor que lo apague), ese run se repite tal cual lo hace el vendor y el
    proxy deja de encoger: sin esto, toda frase larga reventaba y se callaba.
    """

    def __init__(self, session, keep_len: int | None = None,
                 input_name: str = "input") -> None:
        self._session = session
        self._keep_len = keep_len
        self._input_name = input_name
        self._shrink: Any = None  # RunOptions perezosas: el primer run largo
        self._shrink_rejected = False

    def _is_long(self, input_feed) -> bool:
        if self._keep_len is None:
            return True
        try:
            return int(input_feed[self._input_name].shape[-1]) > self._keep_len
        except Exception:  # noqa: BLE001 - entrada rara: como hoy, sin encoger
            return False

    def run(self, output_names, input_feed, run_options=None):
        if run_options is not None:
            return self._session.run(output_names, input_feed, run_options)
        if self._shrink_rejected or not self._is_long(input_feed):
            return self._session.run(output_names, input_feed)
        try:
            if self._shrink is None:
                self._shrink = shrink_run_options()
            return self._session.run(output_names, input_feed, self._shrink)
        except Exception as error:  # noqa: BLE001 - abajo: el run de siempre decide
            rejected = error
        # ORT valida la lista de encogido ANTES de calcular, así que repetir cuesta
        # ~0. Fuera del except: si el run de siempre también falla, sube SU error,
        # el mismo que daría el vendor, sin el rechazo encadenado.
        outputs = self._session.run(output_names, input_feed)
        self._shrink_rejected = True
        _warn_once("arena-shrink", "ONNX Runtime no deja encoger el arena de esta sesión "
                   "(%s): la uso sin encoger, igual que el vendor", rejected)
        return outputs

    def __getattr__(self, name: str):
        if name == "_session":  # copy/pickle antes de __init__: sin recursión
            raise AttributeError(name)
        return getattr(self._session, name)


def ort_session(path: str, threads: int = 1):
    """InferenceSession de CPU con hilos EXPLÍCITOS (intra=threads, inter=1).

    El default (0 = todos los núcleos) dimensiona cada pool como si el proceso
    tuviera la máquina para él solo.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options,
                                providers=["CPUExecutionProvider"])


# --- Silero VAD -----------------------------------------------------------------


class SileroOnnx:
    """El mismo cálculo que silero_vad.utils_vad.OnnxWrapper, con numpy en vez de torch.

    Mismo archivo (el data/silero_vad.onnx que el paquete carga con onnx=True),
    mismas opciones de sesión (CPU, 1 hilo intra y 1 inter), mismo estado
    (2, 1, 128), mismo contexto de 64 muestras a 16 kHz (32 a 8 kHz) y mismas
    entradas input/state/sr: las probabilidades salen idénticas bit a bit. NO
    vale el silero_vad_v6.onnx de faster-whisper: es otro modelo (cambian
    decisiones), y los umbrales de [voice] están afinados con este.

    Un chunk mono por llamada (lo único que usa Crotolamo): `x` es cualquier cosa
    que acepte np.asarray, de 512 muestras a 16 kHz; devuelve la probabilidad.
    """

    def __init__(self, path: str) -> None:
        import numpy as np

        self._np = np
        self.session = ort_session(path, threads=1)
        # Un ONNX con OTRAS entradas (otra versión del paquete) debe fallar aquí, al
        # cargar, para que load_silero caiga al cargador del vendor; si no, fallaría
        # en el primer chunk y el VAD quedaría en energía para todo el proceso.
        names = sorted(i.name for i in self.session.get_inputs())
        if names != ["input", "sr", "state"]:
            raise ValueError(f"entradas inesperadas en {path}: {names}")
        # Como el vendor: un modelo '16k' solo acepta 16 kHz.
        self.sample_rates = [16000] if "16k" in path else [8000, 16000]
        self.reset_states()

    def reset_states(self) -> None:
        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((0, 0), dtype=np.float32)
        self._last_sr = 0

    def __call__(self, x, sr: int) -> float:
        np = self._np
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"SileroOnnx procesa un chunk mono, no {x.shape}")
        if sr != 16000 and sr % 16000 == 0:
            x = x[:, ::sr // 16000]
            sr = 16000
        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} "
                             "(or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} "
                             "(Supported values: 256 for 8000 sample rate, 512 for 16000)")
        context_size = 64 if sr == 16000 else 32
        if self._last_sr and self._last_sr != sr:
            self.reset_states()
        if not self._context.size:
            self._context = np.zeros((1, context_size), dtype=np.float32)
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self.session.run(
            None, {"input": x, "state": self._state, "sr": np.array(sr, dtype=np.int64)},
        )
        self._context = x[:, -context_size:]
        self._last_sr = sr
        return float(out[0][0])


class _VendorSileroAdapter:
    """Respaldo: el OnnxWrapper de silero_vad (torch) detrás del MISMO contrato
    que SileroOnnx (numpy de entrada, float de salida), para que stt y adapters
    no sepan cuál de los dos les tocó."""

    def __init__(self, model) -> None:
        import numpy as np
        import torch  # ya importado por load_silero_vad: aquí no cuesta

        self._model = model
        self._np = np
        self._torch = torch

    def reset_states(self) -> None:
        self._model.reset_states()

    def __call__(self, x, sr: int) -> float:
        arr = self._np.ascontiguousarray(x, dtype=self._np.float32)
        return float(self._model(self._torch.from_numpy(arr), sr))


def load_silero():
    """Silero VAD sin torch: SileroOnnx sobre el ONNX que trae el paquete silero_vad.

    - Sin numpy u onnxruntime: ImportError hacia arriba (el cargador del vendor
      los necesita también; el llamador cae a energía como siempre).
    - Sin el archivo, o si no abre con estas entradas (otra versión del paquete):
      el cargador de siempre, load_silero_vad(onnx=True), con UN aviso. Si ese
      también falla, su error sube tal cual (y lo avisa quien llama, una vez).
    - Sin silero_vad instalado: el ImportError de siempre.
    """
    path = package_file("silero_vad", "data", "silero_vad.onnx")
    if path is not None and os.path.isfile(path):
        try:
            return SileroOnnx(path)
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - deriva del archivo/API: el vendor
            reason = f"no pude abrir {path} directo ({type(error).__name__}: {error})"
    else:
        reason = "silero_vad no trae data/silero_vad.onnx"
    from silero_vad import load_silero_vad

    model = _VendorSileroAdapter(load_silero_vad(onnx=True))
    _warn_once("silero", "VAD: %s; uso el cargador de silero_vad, que importa torch "
               "(~+200 MB)", reason)
    return model


# --- openWakeWord ----------------------------------------------------------------

_OWW_VERIFIER = "openwakeword.custom_verifier_model"

# available() corre en el hilo principal y _get_model() en WarmVoice o en el oído:
# sin lock, dos hilos podían registrar cada uno su stub y pisarse.
_oww_import_lock = threading.Lock()


def import_openwakeword():
    """`import openwakeword` sin su módulo de entrenamiento (sklearn + scipy).

    El __init__ del paquete hace `from openwakeword.custom_verifier_model import
    train_custom_verifier`, que importa sklearn y scipy solo para ENTRENAR un
    verificador. Se pre-registra un stub de ese módulo cuya train_custom_verifier
    cambia el stub por el módulo real en su primera llamada: quien entrene sigue
    llegando a la función de verdad. Solo si ni el paquete ni ese módulo estaban
    ya en sys.modules (un falso de los tests, o un import previo, se respetan);
    si con el stub el import falla, se reintenta sin él (otra versión del paquete).
    """
    with _oww_import_lock:
        if "openwakeword" not in sys.modules and _OWW_VERIFIER not in sys.modules:
            stub = types.ModuleType(_OWW_VERIFIER)

            def train_custom_verifier(*args, **kwargs):
                if sys.modules.get(_OWW_VERIFIER) is stub:
                    del sys.modules[_OWW_VERIFIER]
                real = importlib.import_module(_OWW_VERIFIER).train_custom_verifier
                package = sys.modules.get("openwakeword")
                if getattr(package, "train_custom_verifier", None) is train_custom_verifier:
                    package.train_custom_verifier = real  # type: ignore[union-attr]
                return real(*args, **kwargs)

            stub.train_custom_verifier = train_custom_verifier  # type: ignore[attr-defined]
            sys.modules[_OWW_VERIFIER] = stub
            try:
                import openwakeword

                return openwakeword
            except ImportError as error:
                if sys.modules.get(_OWW_VERIFIER) is stub:
                    del sys.modules[_OWW_VERIFIER]
                log.debug("openwakeword no importó con el stub del verificador (%s); "
                          "reintento sin él", error)
        import openwakeword

        return openwakeword


def oww_constructor_kwargs(model_cls, name: str, path: str) -> list[dict[str, Any]]:
    """Variantes de kwargs para openwakeword.Model, la de SU firma primero.

    La firma cambia entre versiones (0.4: wakeword_model_paths y **kwargs; 0.5/0.6:
    wakeword_models e inference_framework). Probar a ciegas era caro: en 0.4 el
    primer intento cargaba los 5 modelos pre-entrenados y el segundo uno más, y
    los dos reventaban DESPUÉS, al pasarle a AudioFeatures los kwargs que su
    **kwargs se había tragado (9 sesiones construidas para quedarse con 3). Tras
    la elegida por firma va la lista de siempre, sin repetir, como respaldo.
    `name` es el nombre (o la ruta propia ya resuelta) y `path` la ruta al .onnx.
    """
    import inspect  # aquí y no arriba: ~7 ms de import que solo paga quien carga el wake

    legacy: list[dict[str, Any]] = [
        {"wakeword_models": [name], "inference_framework": "onnx"},
        {"wakeword_model_paths": [path], "inference_framework": "onnx"},
        {"wakeword_model_paths": [path]},
        {"wakeword_models": [path]},
    ]
    try:
        params = inspect.signature(model_cls.__init__).parameters
    except (TypeError, ValueError):
        return legacy
    first: dict[str, Any] | None = None
    if "wakeword_models" in params:
        first = {"wakeword_models": [name]}
        if "inference_framework" in params:
            first["inference_framework"] = "onnx"
    elif "wakeword_model_paths" in params:
        first = {"wakeword_model_paths": [path]}
    if first is None:
        return legacy
    return [first] + [kwargs for kwargs in legacy if kwargs != first]


def release_oww_warmup_arena(model) -> None:
    """Devuelve al sistema el pico que dejó el calentamiento de openWakeWord.

    AudioFeatures.__init__ corre melspec + embedding sobre 10 s de ceros y el
    arena de ORT se queda ese pico para siempre (~200 MB). Un run mínimo con
    shrink_run_options() en cada sesión de features lo suelta. Solo toca las
    sesiones (las salidas se tiran): los buffers del preprocessor no cambian, y
    por eso tampoco los scores. Es una optimización: nunca lanza.
    """
    pre = getattr(model, "preprocessor", None)
    sessions: list[tuple[str, Any]] = []
    for attr in ("melspec_model", "embedding_model"):
        session = getattr(pre, attr, None)
        if callable(getattr(session, "run", None)):
            sessions.append((attr, session))
    if not sessions:
        return
    try:
        import numpy as np

        options = shrink_run_options()
    except Exception as error:  # noqa: BLE001
        log.debug("sin onnxruntime/numpy para soltar el arena de openWakeWord: %s", error)
        return
    for attr, session in sessions:
        try:
            spec = session.get_inputs()[0]
            if attr == "melspec_model":
                shape = [1, 1280]  # un frame de 80 ms
            else:
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
            session.run(None, {spec.name: np.zeros(shape, dtype=np.float32)}, options)
        except Exception as error:  # noqa: BLE001 - optimización: el modelo sigue igual
            log.debug("no solté el arena de openWakeWord (%s): %s", attr, error)


# --- Whisper ---------------------------------------------------------------------


def _quiet_tqdm_monitor() -> None:
    """Que tqdm no arranque su hilo monitor (TMonitor).

    tqdm.__new__ lo arranca aunque la barra esté desactivada, y faster-whisper
    crea barras desactivadas en cada transcripción: un hilo que despierta cada
    10 s para siempre (8 640 veces al día) para vigilar barras que no se pintan.
    """
    try:
        import tqdm

        tqdm.tqdm.monitor_interval = 0
    except Exception as error:  # noqa: BLE001 - sin tqdm no hay monitor que parar
        log.debug("no pude desactivar el monitor de tqdm: %s", error)


def load_whisper(faster_whisper, size: str, cpu_threads: int = 0):
    """WhisperModel(size) primero desde la caché local, sin preguntar al Hub.

    Antes cada arranque resolvía el nombre en el Hub de Hugging Face (una conexión
    por modelo, y hasta ~10 s por modelo con la red caída) aunque el modelo ya
    estuviera descargado. Con cualquier fallo (no está en la caché: primera vez)
    se hace EXACTAMENTE la llamada de siempre, que lo descarga. cpu_threads=0 es
    el default de la librería (no se pasa).
    """
    _quiet_tqdm_monitor()
    extra: dict[str, Any] = {"cpu_threads": cpu_threads} if cpu_threads else {}
    try:
        return faster_whisper.WhisperModel(size, device="cpu", compute_type="int8",
                                           local_files_only=True, **extra)
    except Exception as error:  # noqa: BLE001 - sin caché (o API vieja): el de siempre
        log.debug("Whisper '%s' no cargó solo de la caché local (%s); lo resuelvo "
                  "en el Hub", size, error)
    return faster_whisper.WhisperModel(size, device="cpu", compute_type="int8", **extra)


# --- Piper -----------------------------------------------------------------------

# ~115 caracteres / ~8 s de audio: las frases típicas conservan el arena (misma
# velocidad que hoy) y las largas devuelven su pico al terminar. Es el default de
# [voice].piper_arena_keep_phonemes (lo lee TTS.from_settings).
PIPER_KEEP_PHONEMES = 256


def _has_cpu_arena(session) -> bool:
    """¿La sesión dice tener arena de CPU? Sin él no hay pico retenido que devolver
    y ORT rechaza la opción de encogido. Si no sabe decirlo, se asume que sí: el
    proxy cae solo al run de siempre si ORT la rechaza."""
    try:
        return bool(session.get_session_options().enable_cpu_mem_arena)
    except Exception:  # noqa: BLE001 - otra API (o un falso): que decida run()
        return True


def wrap_piper_session(voice, keep_phonemes: int = PIPER_KEEP_PHONEMES):
    """Envuelve voice.session en ArenaShrinkingSession (no-op si no hay sesión, o si
    la sesión no tiene arena de CPU: entonces queda la de fábrica)."""
    session = getattr(voice, "session", None)
    if (session is None or isinstance(session, ArenaShrinkingSession)
            or not callable(getattr(session, "run", None))):
        return voice
    if not _has_cpu_arena(session):
        log.debug("la sesión de Piper no tiene arena de CPU: nada que acotar")
        return voice
    try:
        voice.session = ArenaShrinkingSession(session, keep_len=keep_phonemes)
    except Exception as error:  # noqa: BLE001 - optimización: la voz sigue igual
        log.debug("no pude envolver la sesión de Piper: %s", error)
    return voice
