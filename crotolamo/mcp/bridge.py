"""Puente entre los servers MCP y el registry/router de Crotolamo (M4).

Lee `[mcp]` de la config, lanza cada server, le pide sus tools y las registra
como `Tool` normales: el LLM las ve con el mismo formato, el guard las revisa
igual y el patrón confirma igual. El resto del agente no sabe que son MCP.

DECISIONES, y por qué:

- **Un server caído NUNCA impide arrancar.** Cada server se lanza bajo un
  presupuesto (`startup_timeout_s` cubre handshake + tools/list); si falla, se
  avisa por log y se salta. Crotolamo con menos tools es mejor que Crotolamo mudo.
  Los handshakes de varios servers corren A LA VEZ (el arranque cuesta el más
  lento, no la suma); lanzarlos y registrar sus tools sigue en el hilo del
  llamador y en el orden de la config, así los nombres no dependen de quién
  contesta antes.

- **Soft-errors en personaje, no excepciones.** Un timeout o un fallo del server
  vuelven al LLM como texto ("El server MCP 'x' tardó..."), con un prefijo que
  NO está en `tool_parsing.HARD_ERROR_PREFIXES`: es texto apto para decírselo al
  patrón, no un fallo de la capa de ejecución.

- **Strikes.** Un timeout cuenta un strike; al segundo CONSECUTIVO se retiran
  todas las tools del server y se cierra: un server colgado no debe volver a
  costar 20s por cada intento del modelo. Un éxito resetea los strikes. Un fallo
  de TRANSPORTE (proceso muerto, pipe rota) desregistra de inmediato: no hay
  nada que reintentar. También sin llamada en curso: el cliente avisa (on_dead).

- **Confirmación por hints, con letra chica.** Bajo `confirm = "destructive"`
  una tool corre sin preguntar solo si el server la declara de solo lectura
  (`readOnlyHint: true`) o no destructiva (`destructiveHint: false`). Los hints
  los declara EL SERVER y son advisory: un server mentiroso se salta la
  confirmación. Para servers no confiables: `confirm = "always"`.

- **`strict_args = False`.** El esquema de una tool MCP lo dicta el server
  (anidado, `additionalProperties`...); el filtro de kwargs del agente, pensado
  para los inventos del 3B sobre tools planas, mutilaría llamadas válidas.
"""

from __future__ import annotations

import atexit
import itertools
import os
import re
import signal
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from crotolamo.core import router
from crotolamo.logging_setup import get_logger
from crotolamo.mcp.client import MCPError, MCPTimeout, MCPTransportError, StdioMCPClient
from crotolamo.tools.base import Registry, Tool, normalize_key, truncate_for_context

log = get_logger("mcp.bridge")

CONFIRM_MODES = ("destructive", "always", "never")
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_STARTUP_TIMEOUT_S = 15.0
DEFAULT_CONFIRM = "destructive"

# Los nombres de tool que aceptan Ollama/OpenAI/GLM: [A-Za-z0-9_], máx. 64.
MAX_TOOL_NAME = 64
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")
# Recorte de la descripción del server: cada tool a la vista cuesta tokens en
# CADA turno, y algunos servers escriben párrafos enteros.
DESCRIPTION_CAP = 300
# Timeouts consecutivos antes de desconectar un server.
MAX_STRIKES = 2
# Tope de keywords derivadas por server, para que un server verborreico no
# matchee cualquier frase del patrón.
MAX_DERIVED_KEYWORDS = 40

# Palabras comunes (es/en) que no dicen nada del server: fuera de las keywords
# derivadas de las descripciones.
_STOPWORDS = frozenset({
    "sobre", "desde", "hasta", "para", "entre", "cuando", "donde", "como", "pero",
    "este", "esta", "estos", "estas", "otro", "otra", "todos", "todas", "tambien",
    "puede", "pueden", "debe", "deben", "tiene", "tienen", "hacer", "usar", "segun",
    "with", "from", "that", "this", "which", "their", "there", "about", "after",
    "before", "where", "these", "those", "into", "using", "given", "returns",
    "return", "value", "values", "string", "optional", "required", "true", "false",
    "null", "object", "array", "number", "boolean", "integer", "default", "example",
    "tool", "tools", "server", "input", "output", "result", "results", "specified",
    "provided", "otherwise", "should", "would", "could", "will", "must", "each",
    "only", "also", "when", "then", "than", "other", "based", "within", "without",
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    confirm: str = DEFAULT_CONFIRM
    # Vacía = derivar del nombre del server y de sus tools (ver default_keywords).
    keywords: list[str] = field(default_factory=list)
    prefix: str = ""


@dataclass
class MCPConfig:
    enabled: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    confirm: str = DEFAULT_CONFIRM
    servers: list[MCPServerConfig] = field(default_factory=list)


_UNEXPANDED = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _expand(value: str, where: str = "[mcp]") -> str:
    """~ y $VARS, como el resto de rutas de la config (settings._expand).

    Una variable sin definir se queda LITERAL ("$TOKEN") y se manda tal cual al
    server: mejor avisar que descubrirlo por un 401 opaco.
    """
    out = os.path.expandvars(os.path.expanduser(value))
    if _UNEXPANDED.search(out):
        log.warning("%s: variable de entorno sin definir en %r; se manda tal cual", where, value)
    return out


def _positive_number(raw: Any, default: float, key: str, where: str) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    log.warning("%s: %s=%r no es un número positivo; uso %g", where, key, raw, default)
    return default


def _confirm_mode(raw: Any, default: str, where: str) -> str:
    """`confirm` validado. Ausente => `default`; INVÁLIDO => "always".

    Fail-closed a propósito: heredar el default ante un typo convertía un
    `confirm = "allways"` (el patrón quería MÁS confirmación) bajo un global
    "never" en un server que borra sin preguntar.
    """
    if raw is None:
        return default
    mode = str(raw).strip().lower()
    if mode in CONFIRM_MODES:
        return mode
    log.warning(
        "%s: confirm=%r desconocido (vale %s); uso 'always' (lo seguro ante un typo)",
        where, raw, "/".join(CONFIRM_MODES),
    )
    return "always"


def _parse_server(name: str, table: Any, defaults: MCPConfig) -> MCPServerConfig | None:
    where = f"[mcp.servers.{name}]"
    if not isinstance(table, dict):
        log.warning("%s no es una tabla; lo ignoro", where)
        return None

    command = table.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(c, str) and c for c in command)
    ):
        log.warning("%s: falta `command` (lista de strings no vacía); lo ignoro", where)
        return None

    env_raw = table.get("env", {})
    env: dict[str, str] = {}
    if isinstance(env_raw, dict):
        for key, value in env_raw.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                env[str(key)] = _expand(str(value), f"{where}.env")
            else:
                log.warning("%s: env.%s no es un escalar; lo ignoro", where, key)
    elif env_raw:
        log.warning("%s: `env` debe ser una tabla; lo ignoro", where)

    cwd_raw = table.get("cwd")
    cwd: str | None = None
    if isinstance(cwd_raw, str) and cwd_raw:
        cwd = _expand(cwd_raw, f"{where}.cwd")
    elif cwd_raw is not None:
        log.warning("%s: `cwd` debe ser un string; lo ignoro", where)

    keywords_raw = table.get("keywords", [])
    keywords: list[str] = []
    if isinstance(keywords_raw, list):
        keywords = [str(k) for k in keywords_raw if isinstance(k, str) and k.strip()]
    elif keywords_raw:
        log.warning("%s: `keywords` debe ser una lista; la ignoro", where)

    prefix_raw = table.get("prefix")
    prefix = prefix_raw.strip() if isinstance(prefix_raw, str) and prefix_raw.strip() else name

    return MCPServerConfig(
        name=name,
        command=[_expand(c, f"{where}.command") for c in command],
        env=env,
        cwd=cwd,
        timeout_s=_positive_number(table.get("timeout_s"), defaults.timeout_s, "timeout_s", where),
        startup_timeout_s=_positive_number(
            table.get("startup_timeout_s"), defaults.startup_timeout_s, "startup_timeout_s", where,
        ),
        confirm=_confirm_mode(table.get("confirm"), defaults.confirm, where),
        keywords=keywords,
        prefix=prefix,
    )


def load_mcp_config(settings: Any) -> MCPConfig:
    """Lee y valida `[mcp]`. Acepta el objeto Settings o directamente el dict.

    Nunca lanza por config mala: cada valor inválido se sustituye por su default
    con un WARNING (salvo `confirm`, que cae en "always": un typo no debe BAJAR
    la confirmación), y un server sin `command` válido se ignora. La config es
    del patrón; un typo no debe tumbar el asistente.
    """
    raw = settings if isinstance(settings, dict) else getattr(settings, "mcp", None)
    if not isinstance(raw, dict):
        raw = {}

    enabled_raw = raw.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        # `enabled = "false"` (string) es truthy en Python: activaría MCP por un
        # typo de comillas. Solo un booleano TOML cuenta.
        log.warning("[mcp].enabled=%r no es booleano (true/false sin comillas); queda apagado",
                    enabled_raw)
        enabled_raw = False
    cfg = MCPConfig(
        enabled=enabled_raw,
        timeout_s=_positive_number(raw.get("timeout_s"), DEFAULT_TIMEOUT_S, "timeout_s", "[mcp]"),
        startup_timeout_s=_positive_number(
            raw.get("startup_timeout_s"), DEFAULT_STARTUP_TIMEOUT_S, "startup_timeout_s", "[mcp]",
        ),
        confirm=_confirm_mode(raw.get("confirm"), DEFAULT_CONFIRM, "[mcp]"),
    )

    servers_raw = raw.get("servers", {})
    if not isinstance(servers_raw, dict):
        log.warning("[mcp].servers debe ser tablas nombradas [mcp.servers.<nombre>]; lo ignoro")
        servers_raw = {}
    for name, table in servers_raw.items():
        server = _parse_server(str(name), table, cfg)
        if server is not None:
            cfg.servers.append(server)
    return cfg


# ---------------------------------------------------------------------------
# Traducción tool MCP -> Tool de Crotolamo
# ---------------------------------------------------------------------------

def sanitize_tool_name(prefix: str, remote_name: str, taken: set[str]) -> str:
    """`mcp_<prefix>_<tool>` con solo [A-Za-z0-9_], <= 64 chars y sin chocar con
    `taken` (sufijo numérico si hace falta). El sufijo cabe siempre: se recorta
    la base para no pasarse del tope."""
    base = _INVALID_NAME_CHARS.sub("_", f"mcp_{prefix}_{remote_name}")[:MAX_TOOL_NAME]
    candidate = base
    n = 2
    while candidate in taken:
        suffix = f"_{n}"
        candidate = base[: MAX_TOOL_NAME - len(suffix)] + suffix
        n += 1
    return candidate


def normalize_schema(schema: Any) -> dict[str, Any]:
    """inputSchema del server -> esquema de objeto que aceptan Ollama/GLM.

    Garantiza `type: object` y un `properties` dict; quita `$schema` (a algunos
    motores les estorba). Lo demás (anidados, additionalProperties, enum...) se
    respeta tal cual: es la parte que el server necesita que llegue intacta.
    """
    out: dict[str, Any] = dict(schema) if isinstance(schema, dict) else {}
    out.pop("$schema", None)
    out["type"] = "object"
    props = out.get("properties")
    out["properties"] = dict(props) if isinstance(props, dict) else {}
    if "required" in out and not isinstance(out["required"], list):
        out.pop("required")
    return out


def tool_is_safe(confirm: str, annotations: Any) -> bool:
    """Política de confirmación (ver docstring del módulo).

    OJO: bajo "destructive" nos fiamos de lo que declare el server. Un
    `readOnlyHint: true` mentiroso se salta la confirmación; ese es el precio de
    no preguntar por cada lectura. Servers dudosos => "always".
    """
    if confirm == "never":
        return True
    if confirm == "always":
        return False
    hints = annotations if isinstance(annotations, dict) else {}
    return hints.get("readOnlyHint") is True or hints.get("destructiveHint") is False


def describe_tool(raw_tool: dict[str, Any], server_name: str) -> str:
    text = str(raw_tool.get("description") or raw_tool.get("name") or "tool MCP")
    text = " ".join(text.split())
    if len(text) > DESCRIPTION_CAP:
        text = text[:DESCRIPTION_CAP].rstrip() + "…"
    return f"{text} [MCP: {server_name}]"


def default_keywords(server: MCPServerConfig, raw_tools: list[dict[str, Any]]) -> list[str]:
    """Keywords de routing cuando el patrón no las define: nombre del server,
    su prefijo, los trozos de los nombres de sus tools y las palabras largas
    de sus descripciones (minúsculas, sin acentos, sin relleno, con tope).

    Es un default de compromiso: las descripciones suelen venir en inglés y el
    patrón habla en español, así que lo que de verdad enruta es el nombre del
    server ("jira", "archivos"). Para afinar, `keywords` en la config.
    """
    words: list[str] = []

    def add(token: str, min_len: int) -> None:
        token = normalize_key(token)
        if len(token) >= min_len and token not in _STOPWORDS and token not in words:
            words.append(token)

    add(server.name, 2)
    add(server.prefix, 2)
    # Trozos de 5+ letras: "get", "set", "run" o "list" (3) matcheaban DENTRO de
    # palabras españolas ("resetea", "target") y convertían charla en turnos con
    # tools. Con 5 se quedan "issue", "search", "create"...; lo que de verdad
    # enruta sigue siendo el nombre del server.
    for raw_tool in raw_tools:
        for piece in re.split(r"[^a-z0-9]+", normalize_key(str(raw_tool.get("name", "")))):
            add(piece, 5)
    for raw_tool in raw_tools:
        for piece in re.findall(r"[a-z]{5,}", normalize_key(str(raw_tool.get("description", "")))):
            if len(words) >= MAX_DERIVED_KEYWORDS:
                return words
            add(piece, 5)
    return words[:MAX_DERIVED_KEYWORDS]


# ---------------------------------------------------------------------------
# Estado de los servers conectados
# ---------------------------------------------------------------------------

@dataclass
class _ServerState:
    config: MCPServerConfig
    client: StdioMCPClient
    registry: Registry
    tools: list[Tool]
    group: str
    strikes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Todos los registries donde se repusieron estas tools (el camino
    # idempotente puede recibir otro registry, p.ej. build_registry()): al
    # desconectar hay que retirarlas de TODOS, no solo del primero.
    registries: list[Registry] = field(default_factory=list)


_STATE: dict[str, _ServerState] = {}
_STATE_LOCK = threading.RLock()
# Clientes lanzados que aún no están en _STATE (handshake + tools/list, hasta
# startup_timeout_s cada uno, y con varios servers la espera a su turno de
# registro). Sin esto, un SIGTERM/Ctrl+C del arranque (npx descargando...) no
# los veía terminate_all()/close_all(): huérfanos.
_STARTING: set[StdioMCPClient] = set()
# Clientes con su Popen en curso. Una señal atendida justo tras el fork corre
# con el proceso ya creado y el cliente sin su pid: terminate_now() no lo
# alcanza y lo busca _kill_unspawned, por la marca de entorno de su lanzamiento
# ("<pid de Crotolamo>.<n>", una distinta cada vez).
_SPAWNING: set[StdioMCPClient] = set()
_MARCA_ENV = "CROTOLAMO_MCP_LANZAMIENTO"
_LANZAMIENTOS = itertools.count(1)
_ATEXIT_REGISTERED = False


def connected_servers() -> list[str]:
    """Nombres de los servers MCP hoy conectados (doctor/tests)."""
    with _STATE_LOCK:
        return list(_STATE)


def strikes_for(name: str) -> int:
    """Timeouts consecutivos acumulados por un server (0 si no está conectado)."""
    with _STATE_LOCK:
        state = _STATE.get(name)
    return state.strikes if state is not None else 0


def _soft(server_name: str, text: str) -> str:
    # Prefijo fijo de TODOS los soft-errors MCP. No está (ni debe estar) en
    # tool_parsing.HARD_ERROR_PREFIXES: es texto para el patrón.
    return f"El server MCP '{server_name}' {text}"


def _retire(state: _ServerState, registry: Registry, why: str | None = None) -> bool:
    """Saca `state` de _STATE, retira sus tools y su grupo y cierra su proceso.

    Pueden pedirlo a la vez el hilo del agente (la llamada que vio el transporte
    roto) y el aviso del cliente (`on_dead`), así que todo lo que el agente ve
    (estado, tools, grupo) se quita en UNA sección bajo _STATE_LOCK: el que
    llega segundo no encuentra nada y nadie vuelve con las tools a medio
    retirar. Y solo si `state` sigue siendo el conectado: un aviso tardío de un
    cliente viejo no debe tumbar al que lo sustituyó con el mismo nombre.
    """
    name = state.config.name
    with _STATE_LOCK:
        if _STATE.get(name) is not state:
            return False
        del _STATE[name]
        if why is not None:
            log.warning("MCP '%s': %s; retiro sus %d tools", name, why, len(state.tools))
        targets = [registry, state.registry, *state.registries]
        for tool in state.tools:
            seen: list[Registry] = []
            for target in targets:
                if any(target is s for s in seen):
                    continue
                seen.append(target)
                target.unregister(tool.name)
        router.unregister_group(state.group)
    # Fuera del lock: cerrar puede tardar segundos (SIGTERM -> SIGKILL).
    state.client.close()
    log.info("MCP '%s': desconectado (%d tools retiradas)", name, len(state.tools))
    return True


def _drop(state: _ServerState, why: str) -> None:
    _retire(state, state.registry, why)


def _why_dead(client: StdioMCPClient) -> str:
    # La causa real (EOF, basura en stdout, pipe rota) la sabe el cliente; antes
    # el log decía siempre "el proceso murió", aunque siguiera vivo.
    reason = client.dead_reason
    return f"transporte roto ({reason})" if reason else "el proceso murió"


def _on_transport_dead(client: StdioMCPClient, reason: str) -> None:
    """`on_dead` del cliente: su transporte murió SOLO (EOF, basura en stdout).

    Desregistra ya, como promete el docstring del módulo, en vez de esperar a
    que el modelo llame una tool muerta: hasta entonces seguían en el registry
    (GLM recibía sus esquemas), ocupaban hueco en el router y, tras basura en
    stdout, el proceso seguía vivo sin lector. En un hilo aparte porque el
    retiro cierra el cliente y espera a su hilo lector, que es quien avisa.
    """
    with _STATE_LOCK:
        state = _STATE.get(client.name)
    if state is None or state.client is not client:
        return  # en pleno handshake (lo ve _connect) o ya retirado/sustituido
    threading.Thread(
        target=_drop, args=(state, f"transporte roto ({reason})"),
        name=f"mcp-{client.name}-retiro", daemon=True,
    ).start()


def _on_timeout(state: _ServerState) -> str:
    name = state.config.name
    with state.lock:
        state.strikes += 1
        strikes = state.strikes
    if strikes >= MAX_STRIKES:
        _drop(state, f"{strikes} timeouts seguidos de {state.config.timeout_s:g}s")
        return _soft(
            name, f"se colgó {strikes} veces seguidas; lo desconecté y retiré sus tools, patrón.",
        )
    log.warning("MCP '%s': timeout (%d de %d)", name, strikes, MAX_STRIKES)
    return _soft(name, "tardó demasiado en responder, patrón. Si vuelve a pasar, lo desconecto.")


def _make_caller(server_name: str, remote_name: str) -> Callable[..., str]:
    """Wrapper que ejecuta la tool remota. Busca el estado EN CADA llamada: si
    el server ya se desconectó, responde en personaje en vez de reventar."""

    def call(**arguments: Any) -> str:
        with _STATE_LOCK:
            state = _STATE.get(server_name)
        if state is None:
            return _soft(server_name, "ya no está conectado, patrón.")
        if not state.client.alive:
            _drop(state, _why_dead(state.client))
            return _soft(server_name, "se murió por su cuenta, patrón. Retiré sus tools.")
        try:
            text, is_error = state.client.call_tool(
                remote_name, arguments, timeout=state.config.timeout_s,
            )
        except MCPTimeout:
            return _on_timeout(state)
        except MCPTransportError as error:
            _drop(state, f"transporte roto ({error})")
            return _soft(server_name, "se cayó, patrón. Lo desconecté y retiré sus tools.")
        except MCPError as error:
            # Un error JSON-RPC (args inválidos, tool desconocida) es una
            # RESPUESTA del server: está vivo. Resetea los strikes igual que un
            # éxito; si no, timeout -> error -> timeout lo desconectaba aunque
            # contestó en medio.
            with state.lock:
                state.strikes = 0
            return _soft(
                server_name, f"devolvió un error, patrón: {truncate_for_context(str(error))}",
            )

        with state.lock:
            state.strikes = 0
        if is_error:
            detail = truncate_for_context(text.strip()) or "sin detalles"
            return _soft(server_name, f"reportó un fallo, patrón: {detail}")
        if not text.strip():
            return _soft(server_name, "respondió sin contenido, patrón.")
        return truncate_for_context(text)

    call.__name__ = f"mcp_{server_name}_{remote_name}"
    return call


def _build_tools(
    registry: Registry, server: MCPServerConfig, raw_tools: list[dict[str, Any]],
) -> list[Tool]:
    taken = set(registry.names())
    tools: list[Tool] = []
    for raw_tool in raw_tools:
        remote_name = raw_tool.get("name")
        if not isinstance(remote_name, str) or not remote_name:
            log.warning("MCP '%s': tool sin nombre en tools/list; la salto", server.name)
            continue
        name = sanitize_tool_name(server.prefix, remote_name, taken)
        taken.add(name)
        tools.append(Tool(
            name=name,
            func=_make_caller(server.name, remote_name),
            description=describe_tool(raw_tool, server.name),
            parameters=normalize_schema(raw_tool.get("inputSchema")),
            safe=tool_is_safe(server.confirm, raw_tool.get("annotations")),
            direct=False,
            strict_args=False,
        ))
    return tools


def _remaining(deadline: float) -> float:
    return max(0.05, deadline - time.monotonic())


def _abandon(client: StdioMCPClient) -> None:
    """Cierra un cliente que no llegó a registrarse y lo quita de _STARTING
    (después de cerrarlo: hasta entonces terminate_all() debe seguir viéndolo)."""
    try:
        client.close()
    finally:
        with _STATE_LOCK:
            _STARTING.discard(client)


def _abandon_all(clients: list[StdioMCPClient]) -> None:
    """_abandon de varios clientes A LA VEZ, esperando a todos. Cada close() de
    un server que ignora el EOF tarda segundos (espera, SIGTERM, SIGKILL): uno
    tras otro, un Ctrl+C con N handshakes en vuelo esperaba N cierres. La espera
    está acotada porque close() lo está (~6 s el peor caso)."""

    def cerrar(client: StdioMCPClient) -> None:
        try:
            _abandon(client)
        except Exception as error:  # noqa: BLE001 — en su propio hilo: que quede en el log
            log.warning("MCP '%s': no pude cerrarlo (%s)", client.name, error)

    hilos: list[threading.Thread] = []
    sin_hilo: list[StdioMCPClient] = []
    for c in clients:
        hilo = threading.Thread(target=cerrar, args=(c,), name=f"mcp-{c.name}-cierre", daemon=True)
        try:
            hilo.start()
        except Exception as error:  # noqa: BLE001 — p.ej. "can't start new thread"
            # Sin hilo, se cierra aquí, en serie como antes. Si no, el error
            # tapaba el Ctrl+C (o el registro ya hecho) y el server seguía vivo.
            log.debug("MCP '%s': cierre en serie (%s)", c.name, error)
            sin_hilo.append(c)
            continue
        hilos.append(hilo)
    for c in sin_hilo:
        cerrar(c)
    for hilo in hilos:
        hilo.join()


# Estados de /proc/<pid>/stat de un proceso que ya murió (zombi, o saliendo).
_MUERTO = (b"Z", b"X")
# Tope de la espera a que un hijo a medio exec publique su entorno: lo normal
# son µs; bajo carga, ms. Solo la paga un hijo directo, líder de su grupo, vivo
# y sin entorno, y solo en el apagado o un Ctrl+C con un Popen cortado.
_EXEC_EN_CURSO_S = 0.2


def _proc_stat(pid: int) -> list[bytes]:
    """Los campos de /proc/<pid>/stat tras el ")" del nombre: estado, ppid,
    grupo... (el nombre puede llevar espacios y paréntesis)."""
    with open(f"/proc/{pid}/stat", "rb") as f:
        return f.read().rsplit(b")", 1)[1].split()


def _environ_publicado(pid: int) -> bytes:
    """El environ de un hijo vivo, esperando (acotado) a que exista. Popen
    vuelve con el hijo a medio execve: el kernel ya le cambió la memoria pero
    publica su entorno más tarde, y hasta entonces /proc/<pid>/environ se lee
    VACÍO (medido: 178 de 300 lecturas justo tras volver Popen entero). Leído
    una sola vez, el server cortado en pleno Popen se daba por ajeno y quedaba
    huérfano. Un proceso que sale vacío hasta el tope (una app lanzada con el
    entorno vacío) no es nuestro y no se mata; uno que muere mientras, tampoco."""
    fin = time.monotonic() + _EXEC_EN_CURSO_S
    while True:
        with open(f"/proc/{pid}/environ", "rb") as f:
            entorno = f.read()
        if entorno or time.monotonic() >= fin or _proc_stat(pid)[0] in _MUERTO:
            return entorno
        time.sleep(0.002)


def _kill_unspawned(clients: list[StdioMCPClient]) -> None:
    """SIGTERM al grupo de los servers de `clients` cuyo Popen cortó una señal
    o un Ctrl+C justo tras el fork: el proceso existe, pero el cliente aún no
    sabe su pid y ni terminate_now() ni close() lo alcanzan (huérfano). Se
    buscan en /proc los hijos DIRECTOS de este proceso, líderes de su grupo
    (start_new_session) y con la marca de entorno de uno de esos lanzamientos.
    La marca es imprescindible: las apps que abre run_detached (Blender, el
    terminal) también son hijos directos y líderes de su sesión, y parar
    Crotolamo no debe matarlas. Con vfork (Linux, Python >= 3.11) Popen vuelve
    del fork con el hijo ya DENTRO de su exec, pero no con el exec acabado: ver
    _environ_publicado. Best-effort, nunca lanza."""
    try:
        marcas = {
            f"{_MARCA_ENV}={c.env[_MARCA_ENV]}".encode() for c in clients if _MARCA_ENV in c.env
        }
        if not marcas:
            return
        yo = os.getpid()
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                campos = _proc_stat(pid)
                # Un zombi no es un exec en curso: ya murió (y su environ se
                # lee vacío siempre; esperarle solo retrasaría el apagado).
                if int(campos[1]) != yo or int(campos[2]) != pid or campos[0] in _MUERTO:
                    continue
                if marcas.isdisjoint(_environ_publicado(pid).split(b"\0")):
                    continue
                os.killpg(pid, signal.SIGTERM)
            except (OSError, ValueError, IndexError):
                continue  # ya no existe, o no es nuestro
    except Exception as error:  # noqa: BLE001 — en pleno apagado: que no tumbe nada
        log.debug("MCP: no pude barrer los servers a medio lanzar (%s)", error)


def _launch(server: MCPServerConfig) -> tuple[StdioMCPClient, float] | None:
    """Crea el cliente, lo apunta en _STARTING y lanza el proceso del server.

    SIEMPRE en el hilo del llamador, también con los handshakes en paralelo:
    la señal de apagado del listener se atiende en ese hilo, así que cuando
    terminate_all() hace su foto todo proceso que exista ya está en _STARTING,
    o en _SPAWNING si la señal cortó su Popen justo tras el fork (el cliente
    aún no sabe el pid; lo busca _kill_unspawned). Un Popen en un hilo
    ayudante podía acabar justo después de la foto y dejar un huérfano.
    Lanzar cuesta milisegundos; lo que tarda (que el server conteste) va en
    _handshake.

    Devuelve (cliente, fin de su presupuesto de arranque), o None si no se pudo
    lanzar (avisado y cerrado). Ctrl+C/SystemExit se propagan tras cerrarlo.
    """
    client = StdioMCPClient(
        server.name, server.command,
        {**server.env, _MARCA_ENV: f"{os.getpid()}.{next(_LANZAMIENTOS)}"},
        server.cwd, on_dead=_on_transport_dead,
    )
    with _STATE_LOCK:
        _STARTING.add(client)
        _SPAWNING.add(client)
    # startup_timeout_s es un PRESUPUESTO para handshake + tools/list juntos.
    deadline = time.monotonic() + server.startup_timeout_s
    try:
        client.start()
    except Exception as error:  # noqa: BLE001 — un server caído no impide arrancar
        log.warning("MCP '%s': no arrancó, lo salto (%s)", server.name, error)
        _abandon(client)
        return None
    except BaseException:
        # Ctrl+C (KeyboardInterrupt no es Exception) o SystemExit a mitad del
        # arranque: nadie más conoce este cliente; sin cerrarlo quedaba huérfano.
        # Sin proceso a la vista, el Ctrl+C pudo cortar su Popen tras el fork.
        if not client.alive:
            _kill_unspawned([client])
        _abandon(client)
        raise
    finally:
        with _STATE_LOCK:
            _SPAWNING.discard(client)
    return client, deadline


def _handshake(
    server: MCPServerConfig, client: StdioMCPClient, deadline: float,
    abortando: threading.Event | None = None,
) -> list[dict[str, Any]] | None:
    """initialize + tools/list de un server ya lanzado, en lo que quede de su
    presupuesto. No toca registry, router ni _STATE, así que puede correr en un
    hilo ayudante a la vez que el de otros servers. Devuelve las tools crudas,
    o None si falló o no ofrece ninguna (avisado, cerrado y fuera de
    _STARTING). Ctrl+C/SystemExit se propagan tras cerrarlo.

    `abortando` lo marca register_mcp_tools antes de cerrar lo que ya no va a
    registrar (un Ctrl+C a mitad): ese fallo lo provocó él, no el server, y va
    a DEBUG; si no, cada Ctrl+C dejaba un "no arrancó" por server en vuelo."""
    try:
        client.initialize(timeout=_remaining(deadline))
        raw_tools = client.list_tools(timeout=_remaining(deadline))
    except Exception as error:  # noqa: BLE001 — un server caído no impide arrancar
        if abortando is not None and abortando.is_set():
            log.debug("MCP '%s': handshake abortado (%s)", server.name, error)
        else:
            log.warning("MCP '%s': no arrancó, lo salto (%s)", server.name, error)
        _abandon(client)
        return None
    except BaseException:
        _abandon(client)
        raise
    if not raw_tools:
        log.warning("MCP '%s': no ofrece ninguna tool; lo cierro", server.name)
        _abandon(client)
        return None
    return raw_tools


# Un server lanzado por register_mcp_tools con su handshake en un hilo ayudante.
_Launched = tuple[StdioMCPClient, Future[list[dict[str, Any]] | None]]


def _handshake_into(
    future: Future[list[dict[str, Any]] | None], server: MCPServerConfig,
    client: StdioMCPClient, deadline: float, abortando: threading.Event,
) -> None:
    """Cuerpo del hilo de un handshake en paralelo. Los fallos del server ya
    los avisa _handshake; lo demás (también Ctrl+C/SystemExit) queda en
    `future` y _connect lo re-lanza en el hilo del llamador, en su turno."""
    try:
        future.set_result(_handshake(server, client, deadline, abortando))
    except BaseException as error:  # noqa: BLE001 — se re-lanza en el llamador
        future.set_exception(error)


def _connect(
    registry: Registry, server: MCPServerConfig, launched: _Launched | None = None,
) -> list[str]:
    """Registra las tools de un server. Sin `launched` lo lanza y hace el
    handshake aquí mismo; con él, recoge el handshake que register_mcp_tools
    corrió en paralelo. El registro va SIEMPRE en el hilo del llamador y en el
    orden de la config: los nombres (y el sufijo _2 de un choque) salen
    idénticos al arranque en serie. Nunca lanza (salvo Ctrl+C/SystemExit, que
    se propagan tras cerrar el server; con `launched` lo cierra el llamador)."""
    if launched is None:
        own = _launch(server)
        if own is None:
            return []
        client, deadline = own
    else:
        client, handshake = launched
    caller_closes = False
    try:
        if launched is None:
            raw_tools = _handshake(server, client, deadline)
        else:
            raw_tools = handshake.result()
        if raw_tools is None:
            return []

        tools = _build_tools(registry, server, raw_tools)
        group = f"mcp:{server.name}"
        keywords = server.keywords or default_keywords(server, raw_tools)
        # Comprobar que sigue vivo y publicarlo en UNA sección: si el transporte
        # muere antes, su aviso (on_dead) no encuentra estado que retirar y el
        # server quedaba registrado muerto; si muere después, el aviso espera al
        # lock y lo encuentra.
        with _STATE_LOCK:
            alive = client.alive
            if alive:
                for tool in tools:
                    registry.register(tool)
                router.register_group(group, [t.name for t in tools], keywords)
                _STATE[server.name] = _ServerState(
                    config=server, client=client, registry=registry, tools=tools, group=group,
                )
                _STARTING.discard(client)
        if not alive:
            log.warning("MCP '%s': se cayó tras el handshake, lo salto (%s)",
                        server.name, _why_dead(client))
            client.close()
            return []
    except BaseException:
        # Ctrl+C (KeyboardInterrupt no es Exception) o SystemExit a mitad del
        # arranque: nadie más conoce este cliente; sin cerrarlo quedaba huérfano.
        # Si lo lanzó register_mcp_tools, lo cierra él A LA VEZ que a los demás
        # que tiene a medias (este solo y luego el resto eran dos rondas de
        # close(), ~1 s cada una con un server que ignora el EOF); hasta
        # entonces sigue en _STARTING, al alcance de terminate_all().
        if launched is None:
            client.close()
        else:
            caller_closes = True
        raise
    finally:
        if not caller_closes:
            with _STATE_LOCK:
                _STARTING.discard(client)
    log.info(
        "MCP '%s': %d tools registradas (%s)",
        server.name, len(tools), ", ".join(t.name for t in tools),
    )
    return [t.name for t in tools]


def _ensure_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(close_all)
        _ATEXIT_REGISTERED = True


def register_mcp_tools(registry: Registry, settings: Any) -> list[str]:
    """Conecta los servers de `[mcp]` y registra sus tools. Devuelve los nombres.

    Idempotente: un server ya conectado no se relanza (sus tools se reponen en
    `registry` si faltan). Con `[mcp].enabled = false` no hace nada. Nunca
    propaga el fallo de un server: se loguea y se sigue con el siguiente.
    """
    cfg = load_mcp_config(settings)
    if not cfg.enabled:
        log.debug("MCP desactivado ([mcp].enabled = false)")
        return []
    _ensure_atexit()

    # Con más de un server por lanzar, sus handshakes corren A LA VEZ en hilos
    # ayudantes: el arranque cuesta el más lento, no la suma (15 s por cada
    # server colgado). Lanzar (_launch) y registrar (_connect) siguen en este
    # hilo y en el orden de la config. Un server que sigue en _STATE con el
    # proceso muerto se relanza abajo en serie, como siempre, tras retirar el
    # viejo: dos procesos del mismo server a la vez podían pisarse (un lockfile).
    with _STATE_LOCK:
        to_launch = [s for s in cfg.servers if s.name not in _STATE]
    # Se marca antes de cerrar lo lanzado que ya no se va a registrar (ver _handshake).
    abortando = threading.Event()
    hilos: list[threading.Thread] = []
    # nombre -> server lanzado con su handshake en curso, hasta que se registra;
    # None = ni se pudo lanzar (ya avisado): no se reintenta abajo.
    pending: dict[str, _Launched | None] = {}
    try:
        if len(to_launch) > 1:
            for server in to_launch:
                own = _launch(server)
                if own is None:
                    pending[server.name] = None
                    continue
                client, deadline = own
                handshake: Future[list[dict[str, Any]] | None] = Future()
                hilo = threading.Thread(
                    target=_handshake_into, args=(handshake, server, client, deadline, abortando),
                    name=f"mcp-start-{server.name}", daemon=True,
                )
                # Apuntados ANTES de arrancar el hilo: un Ctrl+C dentro de
                # start() lo deja corriendo, y el finally cierra el server y
                # espera al hilo como a los demás.
                pending[server.name] = (client, handshake)
                hilos.append(hilo)
                try:
                    hilo.start()
                except Exception as error:  # noqa: BLE001 — p.ej. "can't start new thread"
                    # Sin hilo, este server hace su handshake aquí, en serie
                    # como antes: ni se queda sin dueño ni tumba a los demás.
                    hilos.remove(hilo)
                    log.debug("MCP '%s': handshake en serie (%s)", server.name, error)
                    handshake.set_result(_handshake(server, client, deadline, abortando))

        registered: list[str] = []
        for server in cfg.servers:
            with _STATE_LOCK:
                existing = _STATE.get(server.name)
            if existing is not None:
                if existing.client.alive:
                    for tool in existing.tools:
                        if registry.get(tool.name) is None:
                            registry.register(tool)
                    if registry is not existing.registry and registry not in existing.registries:
                        existing.registries.append(registry)
                    registered.extend(t.name for t in existing.tools)
                    continue
                _drop(existing, _why_dead(existing.client))
            if server.name not in pending:
                registered.extend(_connect(registry, server))
                continue
            # Sigue en `pending` mientras se registra: si un Ctrl+C corta aquí,
            # el finally lo cierra A LA VEZ que a los demás.
            if (launched := pending[server.name]) is not None:
                registered.extend(_connect(registry, server, launched))
            del pending[server.name]
        return registered
    finally:
        # Lanzado pero sin registrar: un Ctrl+C a mitad, o un server que otro
        # hilo conectó entretanto. Se cierra ANTES de esperar a los hilos, así
        # su handshake aborta al momento en vez de agotar su presupuesto.
        abortando.set()
        _abandon_all([p[0] for p in pending.values() if p is not None])
        for hilo in hilos:
            try:
                hilo.join()
            except RuntimeError:
                pass  # el Ctrl+C llegó antes de que arrancase: nada que esperar


def unregister_server(registry: Registry, name: str) -> bool:
    """Quita las tools de un server del registry y del router y cierra su
    proceso. True si estaba conectado. Idempotente."""
    with _STATE_LOCK:
        state = _STATE.get(name)
    if state is None:
        return False
    return _retire(state, registry)


def close_all() -> None:
    """Cierra todos los servers conectados o arrancando (atexit y tests)."""
    with _STATE_LOCK:
        states = list(_STATE.values())
        starting = list(_STARTING)
    for state in states:
        unregister_server(state.registry, state.config.name)
    for client in starting:
        client.close()  # su handshake falla y _connect lo da por no arrancado


def terminate_all() -> None:
    """Para el apagado por SEÑAL del listener, que sale con os._exit y se salta
    atexit: manda SIGTERM al grupo de cada server (también los que siguen en el
    handshake) sin esperar a nadie. Sin esto, un wrapper (npx, sh -c) o un
    server que ignora el EOF de stdin quedaba huérfano al parar el servicio.
    Al que la señal pilla en pleno Popen lo busca _kill_unspawned.
    Best-effort, nunca lanza."""
    with _STATE_LOCK:
        clients = [state.client for state in _STATE.values()] + list(_STARTING)
        spawning = list(_SPAWNING)
    for client in clients:
        try:
            client.terminate_now()
        except Exception:  # noqa: BLE001 - el proceso está saliendo
            pass
    if spawning:
        _kill_unspawned(spawning)
