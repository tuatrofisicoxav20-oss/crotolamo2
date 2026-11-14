"""Memoria semántica: Crotolamo te conoce con el tiempo (mem0 + Chroma + fastembed).

POR QUÉ: la memoria de hechos en SQLite (persistence/facts.py) guarda frases
literales y las busca por solape de palabras; sirve para "anota que...", pero no
para que Crotolamo RECUERDE solo lo que le has ido contando y lo traiga cuando
viene al caso. Esta capa hace eso: recupera recuerdos relevantes en CADA turno
(embeddings multilingües) y, tras responder, deja que mem0 extraiga hechos
nuevos de la conversación en segundo plano.

Tres capas de memoria conviven:
  - core/memory.Conversation: historial de la sesión (ventana de turnos).
  - persistence/facts.py: hechos literales en SQLite, categoría y búsqueda fuzzy.
    Con [memoria].enabled=true sus tools se retiran de la vista del modelo (una
    sola familia de "recordar/olvidar") y `crotolamo memoria migrar` los copia aquí.
  - esta: recuerdos semánticos por turno (mem0), fuera del repo
    (~/.local/share/crotolamo/memoria), con extracción automática.

Decisiones que vienen de MEDIR (scripts/memoria_calibrar.py):
  - Los scores de mem0 NO son coseno: con Chroma calcula 1/(1 + distancia L2) de
    los vectores tal cual, así que un par con coseno 0.63 sale como 0.047. El
    `threshold=0.1` por defecto de mem0.search elimina TODOS los aciertos; 0.04
    perdía 3 de 10; 0.02-0.03 conservaba 10/10. Default: 0.03, y la calidad la
    pone el ranking con top_k chico, no el umbral. Cambiar el embedder obliga a
    recalibrar.
  - Telemetría de mem0 (PostHog) apagada SIEMPRE: se fija aquí, antes de que
    nadie importe mem0, no depende del entorno del patrón.

Regla de contenido (en las tools y en el prompt de extracción): aquí van SOLO
hechos sobre el patrón (gustos, personas, costumbres, preferencias). Ni estado de
proyectos, ni pendientes, ni specs (eso irá a Notion). NUNCA secretos: la
heurística `parece_secreto` rechaza un turno entero antes de mandarlo a mem0.

Todo lo pesado es perezoso y opcional: el módulo importa sin mem0 instalado; si
falta o falla, Crotolamo sigue sin memoria semántica con un WARNING (una vez).
"""

from __future__ import annotations

import os

# Telemetría de mem0 (PostHog) APAGADA SIEMPRE. Va antes de cualquier import de
# mem0 (todos son perezosos, más abajo) y no es configurable a propósito.
os.environ["MEM0_TELEMETRY"] = "False"

import atexit  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future  # noqa: E402
from concurrent.futures import TimeoutError as FutureTimeout  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Any, Callable, Protocol  # noqa: E402

from crotolamo.core.turn import DaemonPool, TurnInfo, wants_turn  # noqa: E402
from crotolamo.logging_setup import get_logger  # noqa: E402

if TYPE_CHECKING:
    from crotolamo.core.memory import Conversation

log = get_logger("core.memoria")

DEFAULT_RUTA = "~/.local/share/crotolamo/memoria"
DEFAULT_UMBRAL = 0.03
DEFAULT_TOP_K = 3
DEFAULT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDER_DIMS = 384
DEFAULT_LLM_PROVIDER = "auto"
# Vacío = el mismo modelo que [llm.glm].model (el cerebro en la nube).
DEFAULT_LLM_MODEL = ""
_GROQ_HOST = "api.groq.com"
_GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_BUSQUEDA_S = 1.5
DEFAULT_USUARIO = "patron"
# Lo más que el proceso espera AL SALIR a que termine la extracción pendiente (el
# mismo tope que cerrar()): una nube colgada no puede dejar el shell sin terminar.
_ESPERA_EXTRACCION_AL_SALIR_S = 5.0
# Breaker de la búsqueda POR TURNO (la del pre-hook), como el de FallbackLLM con
# el cerebro (mismos 60 s): tras BREAKER_FALLOS seguidas sin respuesta a tiempo,
# los turnos de los próximos BREAKER_PAUSA_S responden sin recuerdos y sin
# esperarla. Las búsquedas explícitas (tools, CLI) no pasan por él.
BREAKER_FALLOS = 2
BREAKER_PAUSA_S = 60.0
# Lo mismo para la extracción en segundo plano (ver Memoria._drenar): tras
# EXTRAER_FALLOS seguidas, el hilo espera EXTRAER_PAUSA_S antes de la siguiente.
EXTRAER_FALLOS = 3
EXTRAER_PAUSA_S = 120.0

# La regla de contenido, UNA sola vez: la leen el modelo (descripciones de las
# tools) y mem0 (instrucciones del prompt de extracción).
REGLA_CONTENIDO = (
    "Solo hechos sobre el patrón como persona: gustos, personas de su vida, "
    "costumbres, preferencias, cómo quiere que le hablen. NO guardes estado de "
    "proyectos, pendientes, tareas ni especificaciones técnicas (eso va a Notion). "
    "NUNCA guardes secretos: API keys, contraseñas, tokens ni credenciales."
)

# Instrucciones que mem0 añade a su prompt de extracción de hechos (sección
# "Custom Instructions", máxima prioridad). En español porque el patrón habla
# español y los hechos deben quedar en su idioma.
INSTRUCCIONES_EXTRACCION = f"""\
Extrae únicamente hechos duraderos sobre el usuario (a quien el asistente llama
"patrón"), en español, en frases cortas en primera persona del usuario o en tercera
persona neutra ("Le gusta el café de olla sin azúcar"). {REGLA_CONTENIDO}
Ignora lo que diga el asistente salvo que confirme un hecho del usuario. Ignora
órdenes puntuales ("abre Spotify", "pausa la música"), preguntas, saludos, y
cualquier detalle de proyectos, código, tareas o pendientes. Si el mensaje contiene
algo con pinta de contraseña, clave o token, no extraigas nada de él. Si no hay
hechos personales duraderos, devuelve una lista vacía."""


class MemoriaNoDisponible(RuntimeError):
    """La memoria semántica no puede usarse (falta mem0, config rota, backend caído)."""


class BusquedaFallida(RuntimeError):
    """Una búsqueda no terminó (tardó de más o el backend falló) con la memoria en
    pie. NO es "sin recuerdos": quien contesta con el resultado debe decirlo."""


class SecretoRechazado(ValueError):
    """El texto tiene pinta de credencial y no se guarda."""


# ---------------------------------------------------------------------------
# Detección de secretos (heurística, conservadora con el español cotidiano)
# ---------------------------------------------------------------------------

_PREFIJOS_KEY = re.compile(
    r"\b(?:(?:sk|gsk|spak|ghp|gho|ghu|ghs|github_pat|xox[abps]|hf|glpat|npm)"
    r"[_\-][A-Za-z0-9_\-]{12,}"
    # AWS y Google NO llevan separador tras el prefijo: dentro del grupo de arriba
    # (que exige `_` o `-`) estas tres alternativas no casaban nunca.
    r"|AKIA[0-9A-Z]{16}|AIza[A-Za-z0-9_\-]{30,}|ya29\.[A-Za-z0-9_\-.]{12,})"
)
_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}")
# Token largo y "aleatorio": 24+ caracteres sin espacios con letras Y dígitos.
_ALEATORIO = re.compile(
    r"(?<![\w/.:@-])(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{24,}(?![\w/])"
)
# "mi contraseña es X", "la api key: X", "token=X". Palabras concretas de
# credencial (no "clave" a secas: "palabra clave", "la clave del éxito").
_PALABRA_CREDENCIAL = re.compile(
    r"(?i)\b(?:contrase[ñn]a|password|passwd|api[\s_-]?key|apikey|access[\s_-]?token|"
    r"token|secret[o]?|credencial(?:es)?|clave\s+(?:de\s+)?(?:acceso|api|secreta|privada|"
    r"del?\s+wifi))\b"
)
_VALOR_TRAS_CREDENCIAL = re.compile(
    r"(?i)(?:[:=]\s*|\b(?:es|son|ser[ií]a|era)\s+)(?=\S*[\dA-Z_\-@#$%!*])\S{6,}"
)
# Lo mismo con un valor CORTO si lleva dígitos ("mi contraseña es 1234"): los 6
# caracteres de arriba dejaban pasar PINs y contraseñas de 4-5. Un porcentaje
# ("es 100% actitud") no cuenta.
_VALOR_CON_DIGITO = re.compile(
    r"(?i)(?:[:=]\s*|\b(?:es|son|ser[ií]a|era)\s+)(?!\d+(?:[.,]\d+)?%)(?=\S*\d)\S{4,}"
)
# Valor pegado a la palabra, sin "es" ni ":" ("usa la contraseña Gato2024! para
# entrar", "mi password Tr0ub4dor&3xx"): solo si mezcla letras y dígitos.
_VALOR_PEGADO = re.compile(r"\s+(?=\S*\d)(?=\S*[A-Za-z])\S{6,}")
# "clave" a secas y los códigos numéricos (NIP, PIN, código de seguridad) SOLO
# cuentan con un valor con dígitos detrás ("la clave del banco es 482913", "mi
# nip es 4821"). Sin dígitos siguen pasando "palabra clave" y "la clave del éxito".
_PALABRA_CODIGO = re.compile(
    r"(?i)\b(?:clave|nip|pin|c[oó]digo\s+de\s+(?:seguridad|acceso))\b"
)


def parece_secreto(texto: str) -> bool:
    """True si el texto trae algo con pinta de credencial.

    Prefiere un falso positivo (no guardar un hecho inocente) a un falso
    negativo (guardar una key en una base vectorial sin cifrar).
    """
    if not texto:
        return False
    if (_PREFIJOS_KEY.search(texto) or _PEM.search(texto) or _BEARER.search(texto)
            or _JWT.search(texto) or _ALEATORIO.search(texto)):
        return True
    for m in _PALABRA_CREDENCIAL.finditer(texto):
        fin = min(len(texto), m.end() + 60)
        if (_VALOR_TRAS_CREDENCIAL.search(texto, m.end(), fin)
                or _VALOR_CON_DIGITO.search(texto, m.end(), fin)
                or _VALOR_PEGADO.match(texto, m.end())):
            return True
    for m in _PALABRA_CODIGO.finditer(texto):
        if _VALOR_CON_DIGITO.search(texto, m.end(), min(len(texto), m.end() + 60)):
            return True
    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MemoriaConfig:
    enabled: bool = False
    umbral: float = DEFAULT_UMBRAL
    top_k: int = DEFAULT_TOP_K
    ruta: Path = Path(DEFAULT_RUTA).expanduser()
    usuario: str = DEFAULT_USUARIO
    modelo_embeddings: str = DEFAULT_EMBEDDER
    dims_embeddings: int = DEFAULT_EMBEDDER_DIMS
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    timeout_busqueda_s: float = DEFAULT_TIMEOUT_BUSQUEDA_S
    extraer: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "MemoriaConfig":
        raw = settings if isinstance(settings, dict) else getattr(settings, "memoria", None)
        raw = raw if isinstance(raw, dict) else {}
        ruta = raw.get("ruta") or DEFAULT_RUTA
        return cls(
            # Solo un booleano TOML cuenta: `enabled = "false"` (string) es truthy.
            enabled=raw.get("enabled") is True,
            umbral=float(raw.get("umbral", DEFAULT_UMBRAL)),
            top_k=max(1, int(raw.get("top_k", DEFAULT_TOP_K))),
            ruta=Path(os.path.expandvars(os.path.expanduser(str(ruta)))),
            usuario=str(raw.get("usuario") or DEFAULT_USUARIO),
            modelo_embeddings=str(raw.get("modelo_embeddings") or DEFAULT_EMBEDDER),
            dims_embeddings=int(raw.get("dims_embeddings", DEFAULT_EMBEDDER_DIMS)),
            llm_provider=str(raw.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower(),
            llm_model=str(raw.get("llm_model") or DEFAULT_LLM_MODEL),
            timeout_busqueda_s=float(raw.get("timeout_busqueda_s", DEFAULT_TIMEOUT_BUSQUEDA_S)),
            extraer=bool(raw.get("extraer", True)),
        )


@dataclass
class Recuerdo:
    id: str
    texto: str
    score: float = 0.0


def _clave_recuerdo(texto: str) -> str:
    """Forma canónica para saber si dos recuerdos dicen LO MISMO: sin mayúsculas,
    espacios de más ni punto final."""
    return " ".join(texto.split()).casefold().rstrip(" .")


class Backend(Protocol):
    """Lo que la Memoria necesita de mem0 (o de un doble en los tests)."""

    def guardar(self, texto: str) -> list[str]: ...
    def extraer(self, usuario: str, respuesta: str) -> list[str]: ...
    def buscar(self, query: str, top_k: int, umbral: float) -> list[Recuerdo]: ...
    def todas(self) -> list[Recuerdo]: ...
    def borrar(self, memoria_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Backend real: mem0 + Chroma + fastembed
# ---------------------------------------------------------------------------

def _glm_section(settings: Any) -> dict[str, Any]:
    llm_raw = getattr(settings, "llm", None) if settings is not None else None
    llm_raw = llm_raw if isinstance(llm_raw, dict) else {}
    glm_raw = llm_raw.get("glm")
    return glm_raw if isinstance(glm_raw, dict) else {}


def resolver_provider(cfg: MemoriaConfig, settings: Any) -> str:
    """Proveedor efectivo del LLM de extracción.

    "auto" (default) sigue al cerebro en la nube: si [llm.glm].base_url es Groq,
    usa el SDK de Groq; si es cualquier otro endpoint (Z.ai de fábrica), el
    proveedor OpenAI-compatible con ESE base_url. Así la key de
    CROTOLAMO_GLM_API_KEY siempre va al proveedor que la emitió: mandar la key de
    Z.ai a Groq hacía fallar TODAS las extracciones en silencio.
    """
    if cfg.llm_provider != "auto":
        return cfg.llm_provider
    base_url = str(_glm_section(settings).get("base_url") or "")
    return "groq" if _GROQ_HOST in base_url else "openai"


def _llm_config(cfg: MemoriaConfig, settings: Any) -> dict[str, Any]:
    """Config del LLM que mem0 usa para EXTRAER hechos (no es el cerebro de Crotolamo).

    La key de la nube (CROTOLAMO_GLM_API_KEY, ver core/glm.py) solo se reutiliza
    con el proveedor que corresponde a [llm.glm].base_url. "groq" explícito con
    otro base_url usa GROQ_API_KEY (no la de la nube); "ollama" usa el modelo
    local de [llm] (sin key, más lento).
    """
    from crotolamo.core.glm import DEFAULT_BASE_URL, DEFAULT_MODEL, _find_api_key

    llm_raw = getattr(settings, "llm", None) if settings is not None else None
    llm_raw = llm_raw if isinstance(llm_raw, dict) else {}
    glm = _glm_section(settings)
    base_url = str(glm.get("base_url") or DEFAULT_BASE_URL)
    cloud_model = str(glm.get("model") or DEFAULT_MODEL)
    provider = resolver_provider(cfg, settings)

    if provider == "ollama":
        return {"provider": "ollama", "config": {
            "model": cfg.llm_model or llm_raw.get("model", "qwen2.5-coder:7b"),
            "ollama_base_url": llm_raw.get("host", "http://localhost:11434"),
            "temperature": 0.0,
        }}
    if provider == "openai":
        config: dict[str, Any] = {"model": cfg.llm_model or cloud_model, "temperature": 0.0,
                                  "openai_base_url": base_url}
        key = _find_api_key()
        if key:
            config["api_key"] = key
        return {"provider": "openai", "config": config}

    # Groq. La key de la nube solo si la nube ES Groq; si no, GROQ_API_KEY.
    groq_es_la_nube = _GROQ_HOST in base_url
    modelo = cfg.llm_model or (cloud_model if groq_es_la_nube else _GROQ_MODEL_FALLBACK)
    config = {"model": modelo, "temperature": 0.0}
    key = _find_api_key() if groq_es_la_nube else os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        config["api_key"] = key
    elif not groq_es_la_nube:
        log.warning("memoria: llm_provider='groq' pero [llm.glm].base_url no es Groq y no hay "
                    "GROQ_API_KEY; la extracción va a fallar (usa llm_provider='auto')")
    return {"provider": "groq", "config": config}


# Variables de entorno de las que el SDK de cada proveedor saca la key cuando la
# config de mem0 no trae una (mem0/llms/openai.py y groq.py). Ollama no usa key.
_ENVS_KEY_SDK: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "groq": ("GROQ_API_KEY",),
}
_KEY_RELLENO = "sin-key"


def _sin_key(llm: dict[str, Any]) -> bool:
    """True si el SDK del LLM de extracción se quedaría sin key: ni en la config
    de mem0 ni en las variables de las que la toma él solo. Ollama no usa key."""
    envs = _ENVS_KEY_SDK.get(llm["provider"])
    if not envs:
        return False
    return not llm["config"].get("api_key") and not any(
        os.environ.get(env, "").strip() for env in envs)


def extraccion_sin_key(cfg: MemoriaConfig, settings: Any) -> bool:
    """¿Arrancaría mem0 sin key para extraer (y con la extracción apagada)?

    El mismo criterio que Mem0Backend._config_arrancable, en un solo sitio: el
    doctor lo usa para decir lo que de verdad va a pasar al arrancar.
    """
    return _sin_key(_llm_config(cfg, settings))


def fijar_cache_modelos(ruta: Path) -> None:
    """Hace que fastembed guarde el modelo de embeddings en `ruta`/modelos.

    Por defecto usa <tmp>/fastembed_cache, y en Fedora /tmp es tmpfs: los ~240 MB
    del modelo ocupaban RAM y se volvían a descargar en cada arranque (sin red, la
    memoria no arrancaba). fastembed lee la variable al crear el modelo, así que
    basta fijarla antes de Memory.from_config. setdefault: si el patrón ya la
    exporta, manda la suya.
    """
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ruta / "modelos"))


class Mem0Backend:
    """Adaptador fino sobre `mem0.Memory`; todo perezoso (importa al primer uso)."""

    def __init__(self, cfg: MemoriaConfig, settings: Any = None) -> None:
        self.cfg = cfg
        self._settings = settings
        self._mem: Any = None
        self._lock = threading.Lock()
        # True (tras arrancar) si el LLM de extracción no tiene API key: la memoria
        # funciona, pero extraer() no hace nada. Ver _config_arrancable.
        self.sin_llm = False

    def config_mem0(self) -> dict[str, Any]:
        ruta = self.cfg.ruta
        return {
            "vector_store": {"provider": "chroma", "config": {
                "collection_name": "crotolamo", "path": str(ruta / "chroma"),
            }},
            "embedder": {"provider": "fastembed", "config": {
                "model": self.cfg.modelo_embeddings,
                "embedding_dims": self.cfg.dims_embeddings,
            }},
            "llm": _llm_config(self.cfg, self._settings),
            "history_db_path": str(ruta / "history.db"),
            "custom_instructions": INSTRUCCIONES_EXTRACCION,
        }

    def _config_arrancable(self) -> dict[str, Any]:
        """config_mem0() con una key de relleno si el LLM de extracción no tiene ninguna.

        mem0 construye el cliente del LLM al ARRANCAR y los SDK (openai, groq)
        lanzan si no hay key: no tener key de nube tumbaba TODA la memoria, también
        buscar y recordar (que no usan LLM) y aunque extraer=false. Con el relleno
        mem0 arranca y solo se apaga la extracción (ver `extraer`).
        """
        conf = self.config_mem0()
        llm = conf["llm"]
        if _sin_key(llm):
            llm["config"]["api_key"] = _KEY_RELLENO
            self.sin_llm = True
            if self.cfg.extraer:
                log.warning("memoria: no hay API key para el LLM de extracción (%s); recordar "
                            "y buscar funcionan, la extracción automática queda apagada",
                            llm["provider"])
        return conf

    def _memory(self) -> Any:
        with self._lock:
            if self._mem is None:
                try:
                    from mem0 import Memory
                except ImportError as error:
                    raise MemoriaNoDisponible(
                        "falta mem0 (pip install -e '.[memoria]')") from error
                # CUALQUIER fallo al arrancar (Chroma ilegible, falta chromadb,
                # fastembed o el SDK del proveedor, modelo sin descargar y sin red)
                # es "memoria no disponible": la Memoria la marca rota UNA vez y las
                # tools caen a la libreta SQLite. Con la excepción cruda no había
                # respaldo y mem0 entero (modelo ONNX incluido) se reconstruía en
                # cada turno, dos veces: búsqueda y extracción.
                try:
                    self.cfg.ruta.mkdir(parents=True, exist_ok=True)
                    fijar_cache_modelos(self.cfg.ruta)
                    self._mem = Memory.from_config(self._config_arrancable())
                except Exception as error:  # noqa: BLE001
                    raise MemoriaNoDisponible(f"mem0 no arrancó: {error}") from error
            return self._mem

    @staticmethod
    def _rows(result: Any) -> list[dict[str, Any]]:
        rows = result.get("results", []) if isinstance(result, dict) else result
        return [r for r in (rows or []) if isinstance(r, dict)]

    def guardar(self, texto: str) -> list[str]:
        result = self._memory().add(texto, user_id=self.cfg.usuario, infer=False)
        return [str(r.get("id")) for r in self._rows(result) if r.get("id")]

    def extraer(self, usuario: str, respuesta: str) -> list[str]:
        mem = self._memory()
        if self.sin_llm:
            # Sin key no hay con qué extraer (ya se avisó al arrancar): ni se
            # intenta, para no mandar cada turno a la nube con la key de relleno.
            return []
        messages = [{"role": "user", "content": usuario},
                    {"role": "assistant", "content": respuesta}]
        result = mem.add(messages, user_id=self.cfg.usuario, infer=True)
        return [str(r.get("memory", "")) for r in self._rows(result)
                if r.get("event") in ("ADD", "UPDATE")]

    def buscar(self, query: str, top_k: int, umbral: float) -> list[Recuerdo]:
        # mem0 >= 2.x: user_id va en `filters`; `limit` se llama `top_k`.
        result = self._memory().search(
            query, filters={"user_id": self.cfg.usuario}, top_k=top_k, threshold=umbral,
        )
        out = [Recuerdo(str(r.get("id")), str(r.get("memory", "")), float(r.get("score") or 0.0))
               for r in self._rows(result) if r.get("memory")]
        return sorted(out, key=lambda r: r.score, reverse=True)

    def todas(self) -> list[Recuerdo]:
        result = self._memory().get_all(filters={"user_id": self.cfg.usuario}, top_k=1000)
        return [Recuerdo(str(r.get("id")), str(r.get("memory", "")))
                for r in self._rows(result) if r.get("memory")]

    def borrar(self, memoria_id: str) -> None:
        self._memory().delete(memoria_id)


# ---------------------------------------------------------------------------
# La memoria que usa Crotolamo
# ---------------------------------------------------------------------------

class Memoria:
    """Recuperación rápida (con presupuesto de tiempo) + extracción en segundo plano.

    - `buscar`/`contexto` corren en un pool y se rinden a los `timeout_busqueda_s`:
      la respuesta hablada NUNCA espera a la memoria; si tarda o falla, se sigue
      sin recuerdos y se loguea WARNING.
    - `extraer_en_fondo` solo ENCOLA; un hilo daemon llama a mem0 (que llama al LLM)
      después de que Crotolamo ya habló. Al salir del proceso se espera, acotado, a
      que termine lo encolado (atexit; cerrar() hace lo mismo).
    - Un backend roto se marca UNA vez y todo pasa a no-op silencioso (INFO).
    """

    def __init__(self, cfg: MemoriaConfig, backend: Backend | None = None,
                 settings: Any = None) -> None:
        self.cfg = cfg
        self._backend: Backend | None = backend
        self._settings = settings
        self._backend_lock = threading.Lock()
        self._roto: str | None = None
        # Hilos daemon (DaemonPool): una búsqueda de mem0 colgada, que ni el
        # presupuesto ni cancel() pueden parar, no impide que el proceso salga.
        self._pool = DaemonPool(2, "Memoria")
        self._cola: queue.Queue[Any] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._vaciado_apuntado = False
        self._cerrada = False
        # Salud de la búsqueda por turno (ver _buscar_turno): fallos seguidos,
        # hasta cuándo no se intenta, la búsqueda en vuelo y si mem0 aún carga.
        self._salud_lock = threading.Lock()
        self._fallos_turno = 0
        self._pausa_hasta = 0.0
        self._breaker_abierto = False
        self._busqueda_turno: Future[list[Recuerdo]] | None = None
        self._calentando = False
        # cerrar() y el vaciado al salir lo ponen: corta la pausa del hilo de
        # extracción y no deja empezar otra (ver _drenar). _en_pausa: el hilo está
        # en ella, sin ningún item en la mano (ver flush).
        self._parar = threading.Event()
        self._en_pausa = threading.Event()

    # --- estado ---
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled) and self._roto is None and not self._cerrada

    @property
    def motivo_no_disponible(self) -> str | None:
        return self._roto

    def _get_backend(self) -> Backend:
        with self._backend_lock:
            if self._roto is not None:
                raise MemoriaNoDisponible(self._roto)
            if self._backend is None:
                # El constructor solo guarda la config (no puede fallar): mem0
                # arranca, y falla, al primer uso, en Mem0Backend._memory.
                self._backend = Mem0Backend(self.cfg, self._settings)
            return self._backend

    def _marcar_rota(self, motivo: str) -> None:
        if self._roto is None:
            self._roto = motivo
            log.warning("memoria semántica no disponible (%s); sigo sin ella", motivo)

    # --- recuperación ---
    def _buscar_sync(self, query: str, top_k: int) -> list[Recuerdo]:
        backend = self._get_backend()
        return backend.buscar(query, top_k, self.cfg.umbral)

    def buscar_estricto(self, query: str, top_k: int | None = None,
                        timeout_s: float | None = None) -> list[Recuerdo]:
        """Como buscar(), pero [] significa SOLO "nada pasó el umbral".

        Para quien va a CONTESTAR con el resultado (tools, CLI, olvidar): con el []
        de buscar() un fallo o una tardanza se contaban como "no recuerdo nada" y
        eso era lo que el modelo le repetía al patrón con el recuerdo guardado.
        Lanza MemoriaNoDisponible si la memoria no está (y la marca rota) y
        BusquedaFallida si esta búsqueda tardó de más o falló.
        """
        if not query.strip():
            return []
        if not self.enabled:
            raise MemoriaNoDisponible(self._roto or "la memoria semántica está apagada")
        k = top_k or self.cfg.top_k
        limite = self.cfg.timeout_busqueda_s if timeout_s is None else timeout_s
        future: Future[list[Recuerdo]] = self._pool.submit(self._buscar_sync, query, k)
        try:
            return future.result(timeout=limite)
        except FutureTimeout:
            # Si aún no empezó (pool ocupado), ya no corre: nadie va a leerla.
            future.cancel()
            log.warning("memoria: la búsqueda tardó más de %.1fs; respondo sin recuerdos", limite)
            raise BusquedaFallida(f"la búsqueda tardó más de {limite:.1f}s") from None
        except MemoriaNoDisponible as error:
            self._marcar_rota(str(error))
            raise
        except Exception as error:  # noqa: BLE001 - se traduce: el llamador decide qué decir
            log.warning("memoria: búsqueda falló (%s); respondo sin recuerdos", error)
            raise BusquedaFallida(str(error) or type(error).__name__) from error

    def buscar(self, query: str, top_k: int | None = None,
               timeout_s: float | None = None) -> list[Recuerdo]:
        """Recuerdos relevantes, o [] si no hay memoria, tarda o falla (WARNING)."""
        if not self.enabled:
            return []
        try:
            return self.buscar_estricto(query, top_k, timeout_s)
        except (MemoriaNoDisponible, BusquedaFallida):
            # Ya está logueado. La versión del pre-hook: la memoria nunca tumba
            # ni retrasa la respuesta hablada.
            return []

    def _buscar_turno(self, query: str) -> list[Recuerdo]:
        """buscar() del pre-hook, que aprende de sus fallos como FallbackLLM.

        El presupuesto de antes (Future.result con timeout en el pool de 2)
        acotaba la ESPERA, no el trabajo: la búsqueda vencida seguía corriendo;
        con dos colgadas el pool se llenaba y cada turno esperaba el presupuesto
        entero sin que su búsqueda llegara a empezar (y todas corrían después).
        Ahora:
        - una a la vez: si la del turno anterior sigue en curso no se encola
          otra; cuenta como fallo y el turno sigue sin recuerdos al instante;
        - al vencer el presupuesto se cancela (si aún no empezó, ya no corre);
        - BREAKER_FALLOS seguidas sin respuesta a tiempo pausan la búsqueda
          BREAKER_PAUSA_S: esos turnos no esperan nada. Pasada la pausa se prueba
          una (medio abierto): un fallo la reabre, un acierto la cierra.
        Con la memoria sana es buscar() tal cual: mismos recuerdos y presupuesto.
        El breaker nunca la marca rota: eso sigue siendo MemoriaNoDisponible.
        """
        if not self.enabled or not query.strip():
            return []
        future: Future[list[Recuerdo]] | None = None
        with self._salud_lock:
            if self._pausa_hasta > time.monotonic():
                return []
            previa = self._busqueda_turno
            if previa is None or previa.done():
                future = self._pool.submit(self._buscar_sync, query, self.cfg.top_k)
                self._busqueda_turno = future
        if future is None:
            log.debug("memoria: la búsqueda del turno anterior sigue en curso; "
                      "respondo sin recuerdos")
            self._fallo_turno()
            return []
        limite = self.cfg.timeout_busqueda_s
        try:
            recuerdos = future.result(timeout=limite)
        except FutureTimeout:
            future.cancel()
            log.warning("memoria: la búsqueda tardó más de %.1fs; respondo sin recuerdos", limite)
            self._fallo_turno()
            return []
        except MemoriaNoDisponible as error:
            self._marcar_rota(str(error))
            return []
        except Exception as error:  # noqa: BLE001 - la memoria nunca tumba el turno
            log.warning("memoria: búsqueda falló (%s); respondo sin recuerdos", error)
            self._fallo_turno()
            return []
        self._exito_turno()
        return recuerdos

    def _fallo_turno(self) -> None:
        """Una búsqueda por turno sin respuesta a tiempo (o rota). Mientras mem0
        carga (precalentar) no cuenta: es el arranque, no un almacén enfermo, y
        dos turnos en esos segundos dejaban los recuerdos en pausa un minuto."""
        with self._salud_lock:
            if self._calentando:
                return
            self._fallos_turno += 1
            if self._fallos_turno < BREAKER_FALLOS:
                return
            self._pausa_hasta = time.monotonic() + BREAKER_PAUSA_S
            # Medio abierto: tras la pausa, UN fallo más basta para reabrirla.
            self._fallos_turno = BREAKER_FALLOS - 1
            self._breaker_abierto = True
        log.warning("memoria: la búsqueda no responde a tiempo; los próximos %.0f s respondo "
                    "sin recuerdos y sin esperarla", BREAKER_PAUSA_S)

    def _exito_turno(self) -> None:
        with self._salud_lock:
            volvio = self._breaker_abierto
            self._fallos_turno = 0
            self._breaker_abierto = False
        if volvio:
            log.info("memoria: la búsqueda volvió a responder a tiempo")

    def contexto(self, query: str, ya_visto: Callable[[str], bool] | None = None) -> str:
        """Bloque de texto para el LLM con los recuerdos relevantes ('' si nada).

        Es la búsqueda POR TURNO (la del pre-hook): pasa por _buscar_turno.
        `ya_visto(linea)`: True si esa línea ("- texto") ya está a la vista del
        modelo; se omite (ver make_memoria_prehook).
        """
        recuerdos = self._buscar_turno(query)
        if not recuerdos:
            return ""
        lineas = [f"- {r.texto}" for r in recuerdos]
        if ya_visto is not None:
            lineas = [linea for linea in lineas if not ya_visto(linea)]
        return "\n".join(lineas)

    # --- escritura ---
    def recordar(self, texto: str) -> list[str]:
        """Guarda un hecho tal cual (sin LLM). Lanza SecretoRechazado si huele a key."""
        texto = " ".join(texto.split())
        if not texto:
            return []
        if parece_secreto(texto):
            raise SecretoRechazado("eso tiene pinta de credencial y no lo guardo")
        try:
            backend = self._get_backend()
            repetido = self._ya_guardado(backend, texto)
            if repetido is not None:
                return [repetido]
            return backend.guardar(texto)
        except MemoriaNoDisponible as error:
            # Como buscar(): se marca rota aquí para que la siguiente llamada vaya
            # directa al respaldo SQLite sin volver a arrancar mem0.
            self._marcar_rota(str(error))
            raise

    def _ya_guardado(self, backend: Backend, texto: str) -> str | None:
        """Id del recuerdo que ya dice exactamente esto, o None.

        mem0.add(infer=False) crea SIEMPRE un recuerdo nuevo: repetir "acuérdate de
        que X" (o `memoria migrar`) dejaba copias que ocupaban huecos del top_k y
        que "olvida lo de X" borraba de una en una. Se mira el backend EN VIVO, sin
        caché: otro proceso o un "olvida" pueden haberlo borrado, y contestar "ya
        quedó" sin guardar sería peor que una copia. Por lo mismo, si la consulta
        falla, el hecho se guarda igual.
        """
        clave = _clave_recuerdo(texto)
        try:
            for recuerdo in backend.todas():
                if _clave_recuerdo(recuerdo.texto) == clave:
                    return recuerdo.id
        except MemoriaNoDisponible:
            raise
        except Exception as error:  # noqa: BLE001 - recordar no depende de listar
            log.warning("memoria: no pude mirar si ya lo tenía (%s); lo guardo igual", error)
        return None

    def extraer_en_fondo(self, usuario: str, respuesta: str) -> bool:
        """Encola el turno para que mem0 extraiga hechos DESPUÉS de hablar.

        True si se encoló. Un turno con pinta de secreto no se manda entero: la
        heurística corre aquí, en el hilo del agente, en microsegundos.
        """
        if not self.enabled or not self.cfg.extraer:
            return False
        if not usuario.strip():
            return False
        if parece_secreto(usuario) or parece_secreto(respuesta):
            log.info("memoria: el turno tiene pinta de credencial; no lo mando a extraer")
            return False
        self._cola.put((usuario, respuesta))
        self._asegurar_worker()
        return True

    def olvidar(self, descripcion: str) -> Recuerdo | None:
        """Borra el recuerdo que MEJOR encaja con la descripción ("lo de mi perro").

        Devuelve el recuerdo borrado o None si nada pasó el umbral. Solo uno por
        llamada: olvidar por voz debe ser conservador. Si la búsqueda no se pudo
        hacer LANZA (MemoriaNoDisponible / BusquedaFallida): "no pude buscar" no
        es "no había nada que olvidar".
        """
        if not self.enabled or not descripcion.strip():
            return None
        candidatos = self.buscar_estricto(descripcion, top_k=3, timeout_s=10.0)
        if not candidatos:
            return None
        mejor = candidatos[0]
        self._get_backend().borrar(mejor.id)
        return mejor

    def todas(self) -> list[Recuerdo]:
        if not self.enabled:
            return []
        return self._get_backend().todas()

    # --- hilo de extracción ---
    def _asegurar_worker(self) -> None:
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drenar, name="MemoriaExtraccion", daemon=True,
                )
                self._worker.start()
            if not self._vaciado_apuntado:
                # El hilo es daemon: al salir (run_shell vuelve con "salir", sin
                # cerrar()) el intérprete lo mataba a media llamada al LLM y lo que
                # el patrón contó en el último turno no se guardaba nunca.
                atexit.register(self._vaciar_al_salir)
                self._vaciado_apuntado = True

    def _vaciar_al_salir(self, timeout_s: float | None = None) -> bool:
        # Corre en atexit, cuando los hilos daemon aún viven (os._exit se lo salta:
        # el apagado por señal del listener vacía la cola él mismo). Al salir no
        # hay "después de la pausa" (ver _drenar): se corta, y lo que llegó durante
        # ella se intenta ya, dentro de este mismo presupuesto, como antes de que
        # existiera. Sin esto flush() esperaba los 5 s enteros a un hilo dormido, y
        # lo encolado se perdía sin intentarse.
        self._parar.set()
        return self.flush(timeout_s=_ESPERA_EXTRACCION_AL_SALIR_S if timeout_s is None
                          else timeout_s)

    def _drenar(self) -> None:
        # Extracciones seguidas que fallaron. Antes, tras un fallo se pasaba
        # directo al siguiente turno: un proveedor que contesta 429 o rechaza la
        # key recibía otra petición tras CADA turno, contra el mismo límite que el
        # cerebro (un 429 largo en el cerebro lo tira al modelo local 60 s). Tras
        # EXTRAER_FALLOS el hilo espera EXTRAER_PAUSA_S; lo que llega mientras se
        # queda en la cola, en orden, y se extrae al acabar la pausa. Después basta
        # UN fallo más para otra pausa (medio abierto); un acierto pone el contador
        # a cero. MemoriaNoDisponible no cuenta: ya la marca rota, como siempre.
        fallos = 0
        while True:
            item = self._cola.get()
            try:
                if item is None:
                    return
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                usuario, respuesta = item
                t0 = time.monotonic()
                try:
                    nuevos = self._get_backend().extraer(usuario, respuesta)
                except MemoriaNoDisponible as error:
                    self._marcar_rota(str(error))
                except Exception as error:  # noqa: BLE001 - un turno roto no mata el hilo
                    log.warning("memoria: la extracción falló (%s)", error)
                    fallos += 1
                else:
                    fallos = 0
                    log.debug("memoria: extracción en %.1fs, %d hecho(s) nuevo(s)",
                              time.monotonic() - t0, len(nuevos))
            finally:
                self._cola.task_done()
            # Con _parar puesto (cerrar, la salida) ya no se pausa: el aviso de
            # "pauso 120 s" salía tras cada fallo sin que hubiera pausa.
            if fallos >= EXTRAER_FALLOS and not self._parar.is_set():
                log.warning("memoria: %d extracciones seguidas fallaron; pauso la extracción "
                            "%.0f s (lo encolado espera, no se pierde)", fallos, EXTRAER_PAUSA_S)
                # Sin tener ningún item en la mano (flush() espera, como con una
                # extracción colgada, a lo encolado DURANTE la pausa); cerrar() y el
                # vaciado al salir ponen el Event y la cortan.
                self._en_pausa.set()
                try:
                    self._parar.wait(EXTRAER_PAUSA_S)
                finally:
                    self._en_pausa.clear()
                fallos = EXTRAER_FALLOS - 1

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Espera a que el hilo procese lo encolado hasta ahora (tests, CLI, salida)."""
        if self._worker is None or not self._worker.is_alive():
            return self._cola.empty()
        if self._en_pausa.is_set() and self._cola.empty():
            # En pausa el hilo no tiene ningún item en la mano: con la cola vacía,
            # todo lo encolado hasta ahora ya se intentó. Sin esto el Event de abajo
            # quedaba detrás de la pausa y flush() solo podía agotar su tiempo (los
            # 2 s del apagado del listener) con nada pendiente.
            return True
        listo = threading.Event()
        self._cola.put(listo)
        return listo.wait(timeout_s)

    def precalentar(self) -> None:
        """Carga mem0 y el modelo de embeddings en segundo plano al arrancar.

        La primera búsqueda carga el modelo (~4-9 s) y se pasaría del presupuesto:
        mejor pagar eso antes del primer "crotolamo", como WarmVoice con Whisper.
        """
        if not self.enabled:
            return

        def _calentar() -> None:
            fallo = False
            try:
                self._buscar_sync("hola", 1)
                log.info("memoria semántica lista (%s)", self.cfg.ruta)
            except MemoriaNoDisponible as error:
                self._marcar_rota(str(error))
            except Exception as error:  # noqa: BLE001
                log.warning("memoria: no pude precalentar (%s)", error)
                fallo = True
            finally:
                with self._salud_lock:
                    self._calentando = False
            if fallo:
                # Un almacén que ya falla al arrancar: con un fallo más del
                # primer turno, la búsqueda por turno se pausa (ver _buscar_turno).
                self._fallo_turno()

        with self._salud_lock:
            self._calentando = True
        threading.Thread(target=_calentar, name="MemoriaWarm", daemon=True).start()

    def cerrar(self, timeout_s: float = 5.0) -> None:
        self._cerrada = True
        # Despierta al hilo de extracción si está en pausa (ver _drenar): sigue
        # vaciando la cola hasta el None, acotado por el join de abajo.
        self._parar.set()
        # cerrar() ya vacía la cola (None va detrás de lo encolado): el vaciado de
        # atexit sobra, y soltarlo no deja la Memoria referenciada hasta el final.
        atexit.unregister(self._vaciar_al_salir)
        if self._worker is not None and self._worker.is_alive():
            self._cola.put(None)
            self._worker.join(timeout=timeout_s)
        # Lo que quedó en la cola del pool (búsquedas que ya nadie espera) no corre.
        self._pool.shutdown(cancel_futures=True)


def mem0_instalado() -> bool:
    """True si mem0 se puede importar (sin importarlo: cuesta segundos)."""
    import importlib.util

    try:
        return importlib.util.find_spec("mem0") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Singleton perezoso (tools y shell) + hooks del agente
# ---------------------------------------------------------------------------

_MEMORIA: Memoria | None = None
_MEMORIA_LOCK = threading.Lock()


def get_memoria() -> Memoria:
    """La memoria de este proceso, construida desde [memoria] de la config."""
    global _MEMORIA
    with _MEMORIA_LOCK:
        if _MEMORIA is None:
            from crotolamo.settings import get_settings

            settings = get_settings()
            _MEMORIA = Memoria(MemoriaConfig.from_settings(settings), settings=settings)
        return _MEMORIA


def set_memoria(memoria: Memoria | None) -> None:
    """Sustituye el singleton (tests y CLI)."""
    global _MEMORIA
    with _MEMORIA_LOCK:
        _MEMORIA = memoria


def _ya_en(textos: list[str]) -> Callable[[str], bool]:
    """¿Está `linea` como línea ENTERA en alguno de `textos`? La última del bloque
    de recuerdos lleva pegado el "]" de cierre."""
    envueltos = [f"\n{texto}\n" for texto in textos]

    def visto(linea: str) -> bool:
        return any(f"\n{linea}\n" in t or f"\n{linea}]\n" in t for t in envueltos)

    return visto


def make_memoria_prehook(
    memoria: Memoria,
    conversation: Conversation | None = None,
    ventana_entera: Callable[[], bool] | None = None,
) -> Callable[[str], str]:
    """Pre-hook por turno: antepone los recuerdos relevantes al mensaje del patrón.

    Va PRIMERO en la cadena de pre-hooks para buscar con el texto limpio. Si la
    memoria no está, tarda o falla, devuelve el texto tal cual (buscar() ya
    logueó el WARNING): la voz nunca espera.

    Con `conversation`, un recuerdo que ya va en un mensaje del patrón que
    seguirá en la ventana no se repite: el bloque se guarda en el historial con
    el mensaje, y antes cada turno sumaba otra copia de los mismos recuerdos
    (54 líneas repetidas en el request del turno 25 con top_k=3). Cuando esa
    copia sale de la ventana, el recuerdo vuelve a entrar en cuanto se recupere.
    `ventana_entera()` dice si el motor que respondería ve la ventana entera:
    si no (el local de 2048 tira los turnos viejos del request), se repite
    como siempre, porque la copia vieja podría no llegarle.

    Con el TurnInfo del agente (`turn`), un comando anclado ("pausa la música",
    "abre opera": la orden COMPLETA casa con una regla del fast-path) no busca:
    su acción la fija la regla y los recuerdos no la cambian. Antes pagaba la
    búsqueda (hasta el presupuesto entero, ANTES de que el fast-path corriera
    la tool) y el bloque solo acababa en el historial. Sin `turn` (llamada
    directa), como siempre.
    """

    @wants_turn
    def hook(text: str, turn: TurnInfo | None = None) -> str:
        if turn is not None and turn.command is not None:
            return text
        if conversation is None or (ventana_entera is not None and not ventana_entera()):
            ctx = memoria.contexto(text)
        else:
            ctx = memoria.contexto(text, _ya_en(conversation.surviving_user_texts()))
        if not ctx:
            return text
        return f"[recuerdos sobre el patrón que vienen al caso:\n{ctx}]\n\n{text}"

    return hook


def make_memoria_posthook(memoria: Memoria) -> Callable[[str, str], None]:
    """After-turn hook: encola el turno para extraer hechos. No bloquea.

    Un comando anclado (ver make_memoria_prehook) no se encola: cada extracción
    es una llamada al LLM de la nube (con la key y el límite del cerebro) y el
    prompt de extracción ya manda ignorar esas órdenes, así que no se pierde
    ningún hecho. Sin `turn`, como siempre.
    """

    @wants_turn
    def hook(usuario: str, respuesta: str, turn: TurnInfo | None = None) -> None:
        if turn is not None and turn.command is not None:
            return
        memoria.extraer_en_fondo(usuario, respuesta)

    return hook


# ---------------------------------------------------------------------------
# CLI: python -m crotolamo memoria (migrar|listar|buscar|olvidar|calibrar)
# ---------------------------------------------------------------------------

def _cargar_calibrar() -> Callable[[list[str]], int] | None:
    """El `main` de scripts/memoria_calibrar.py, o None si el script no está.

    `scripts` no se empaqueta (pyproject solo incluye crotolamo* e interfaces*):
    solo es importable con `python -m crotolamo` DESDE la raíz del repo. Con el
    comando `crotolamo` o desde otra carpeta, `import scripts` reventaba con un
    traceback; ahí el script se carga por su ruta en el repo (REPO_ROOT, la misma
    ancla con la que settings encuentra config/), que no depende de dónde estés.
    """
    try:
        from scripts.memoria_calibrar import main
    except ImportError:
        pass
    else:
        return main
    import importlib.util

    from crotolamo.settings import REPO_ROOT

    ruta = REPO_ROOT / "scripts" / "memoria_calibrar.py"
    if not ruta.is_file():
        return None
    spec = importlib.util.spec_from_file_location("crotolamo_memoria_calibrar", ruta)
    if spec is None or spec.loader is None:
        return None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    calibrar: Callable[[list[str]], int] = modulo.main
    return calibrar


def run_cli(argv: list[str]) -> int:
    from crotolamo.settings import get_settings

    settings = get_settings()
    cfg = MemoriaConfig.from_settings(settings)
    sub = argv[0] if argv else "listar"
    resto = " ".join(argv[1:]).strip()

    if sub == "calibrar":
        calibrar = _cargar_calibrar()
        if calibrar is None:
            print("No encuentro scripts/memoria_calibrar.py junto al paquete: la calibración "
                  "necesita el checkout completo del repo.")
            return 1
        return calibrar(argv[1:])

    if not cfg.enabled:
        print("La memoria semántica está apagada: pon [memoria].enabled = true en la config.")
        return 1
    memoria = Memoria(cfg, settings=settings)
    try:
        if sub == "migrar":
            from crotolamo.persistence import facts

            hechos = facts.recall()
            # `migrar` se puede repetir: es la única vía de traer lo que las tools
            # apuntaron en SQLite mientras la memoria semántica estuvo caída, y
            # antes duplicaba todo lo ya migrado. Lo que ya está se salta con UNA
            # lectura de la memoria (recordar() tampoco lo duplicaría, pero
            # listándola entera por cada hecho).
            presentes = {_clave_recuerdo(r.texto) for r in memoria.todas()}
            n = repetidos = 0
            for row in hechos:
                texto = str(row["texto"])
                if _clave_recuerdo(texto) in presentes:
                    repetidos += 1
                    continue
                try:
                    if memoria.recordar(texto):
                        n += 1
                        presentes.add(_clave_recuerdo(texto))
                except SecretoRechazado:
                    print(f"  saltado (pinta de secreto): #{row['id']}")
            ya_estaban = f" ({repetidos} ya estaban)" if repetidos else ""
            print(f"Migrados {n} de {len(hechos)} hechos de SQLite a la memoria "
                  f"semántica{ya_estaban}.")
            print("Los hechos siguen en SQLite; con [memoria].enabled = true ya no se inyectan.")
            return 0
        if sub == "listar":
            recuerdos = memoria.todas()
            if not recuerdos:
                print("No tengo recuerdos todavía, patrón.")
                return 0
            for r in recuerdos:
                print(f"- {r.texto}   [{r.id}]")
            return 0
        if sub == "buscar":
            if not resto:
                print("Uso: crotolamo memoria buscar <pregunta>")
                return 2
            # Estricto: sin mem0 (o si la búsqueda falla) esto no imprimía nada y
            # salía con 0, igual que una búsqueda sin coincidencias.
            recuerdos = memoria.buscar_estricto(resto, timeout_s=30.0)
            if not recuerdos:
                print("No encontré nada parecido.")
            for r in recuerdos:
                print(f"{r.score:.3f}  {r.texto}")
            return 0
        if sub == "olvidar":
            if not resto:
                print("Uso: crotolamo memoria olvidar <descripción>")
                return 2
            borrado = memoria.olvidar(resto)
            print(f"Olvidado: {borrado.texto}" if borrado else "No encontré nada parecido.")
            return 0
        print("Uso: crotolamo memoria [migrar|listar|buscar <q>|olvidar <q>|calibrar]")
        return 2
    except MemoriaNoDisponible as error:
        print(f"Memoria no disponible: {error}")
        return 1
    except BusquedaFallida as error:
        print(f"No pude buscar en la memoria: {error}")
        return 1
    finally:
        memoria.cerrar()
