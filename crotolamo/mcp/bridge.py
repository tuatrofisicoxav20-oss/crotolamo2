"""Puente entre los servers MCP y el registry/router de Crotolamo (M4).

Lee `[mcp]` de la config, lanza cada server, le pide sus tools y las registra
como `Tool` normales: el LLM las ve con el mismo formato, el guard las revisa
igual y el patrón confirma igual. El resto del agente no sabe que son MCP.

DECISIONES, y por qué:

- **Un server caído NUNCA impide arrancar.** Cada server se lanza bajo un
  presupuesto (`startup_timeout_s` cubre handshake + tools/list); si falla, se
  avisa por log y se salta. Crotolamo con menos tools es mejor que Crotolamo mudo.

- **Soft-errors en personaje, no excepciones.** Un timeout o un fallo del server
  vuelven al LLM como texto ("El server MCP 'x' tardó..."), con un prefijo que
  NO está en `tool_parsing.HARD_ERROR_PREFIXES`: es texto apto para decírselo al
  patrón, no un fallo de la capa de ejecución.

- **Strikes.** Un timeout cuenta un strike; al segundo CONSECUTIVO se retiran
  todas las tools del server y se cierra: un server colgado no debe volver a
  costar 20s por cada intento del modelo. Un éxito resetea los strikes. Un fallo
  de TRANSPORTE (proceso muerto, pipe rota) desregistra de inmediato: no hay
  nada que reintentar.

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
import os
import re
import threading
import time
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
    if raw is None:
        return default
    mode = str(raw).strip().lower()
    if mode in CONFIRM_MODES:
        return mode
    log.warning(
        "%s: confirm=%r desconocido (vale %s); uso %r",
        where, raw, "/".join(CONFIRM_MODES), default,
    )
    return default


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
    con un WARNING, y un server sin `command` válido se ignora. La config es del
    patrón; un typo no debe tumbar el asistente.
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


def _drop(state: _ServerState, why: str) -> None:
    log.warning("MCP '%s': %s; retiro sus %d tools", state.config.name, why, len(state.tools))
    unregister_server(state.registry, state.config.name)


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
            _drop(state, "el proceso murió")
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


def _connect(registry: Registry, server: MCPServerConfig) -> list[str]:
    """Lanza un server, hace el handshake y registra sus tools. Nunca lanza."""
    client = StdioMCPClient(server.name, server.command, server.env, server.cwd)
    # startup_timeout_s es un PRESUPUESTO para handshake + tools/list juntos.
    deadline = time.monotonic() + server.startup_timeout_s
    try:
        client.start()
        client.initialize(timeout=_remaining(deadline))
        raw_tools = client.list_tools(timeout=_remaining(deadline))
    except Exception as error:  # noqa: BLE001 — un server caído no impide arrancar
        log.warning("MCP '%s': no arrancó, lo salto (%s)", server.name, error)
        client.close()
        return []
    if not raw_tools:
        log.warning("MCP '%s': no ofrece ninguna tool; lo cierro", server.name)
        client.close()
        return []

    tools = _build_tools(registry, server, raw_tools)
    for tool in tools:
        registry.register(tool)
    group = f"mcp:{server.name}"
    keywords = server.keywords or default_keywords(server, raw_tools)
    router.register_group(group, [t.name for t in tools], keywords)
    with _STATE_LOCK:
        _STATE[server.name] = _ServerState(
            config=server, client=client, registry=registry, tools=tools, group=group,
        )
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
            _drop(existing, "el proceso murió")
        registered.extend(_connect(registry, server))
    return registered


def unregister_server(registry: Registry, name: str) -> bool:
    """Quita las tools de un server del registry y del router y cierra su
    proceso. True si estaba conectado. Idempotente."""
    with _STATE_LOCK:
        state = _STATE.pop(name, None)
    if state is None:
        return False
    targets = [registry, state.registry, *state.registries]
    for tool in state.tools:
        seen: list[Registry] = []
        for target in targets:
            if any(target is s for s in seen):
                continue
            seen.append(target)
            target.unregister(tool.name)
    router.unregister_group(state.group)
    state.client.close()
    log.info("MCP '%s': desconectado (%d tools retiradas)", name, len(state.tools))
    return True


def close_all() -> None:
    """Cierra todos los servers conectados (atexit y tests)."""
    with _STATE_LOCK:
        states = list(_STATE.values())
    for state in states:
        unregister_server(state.registry, state.config.name)


def terminate_all() -> None:
    """Para el apagado por SEÑAL del listener, que sale con os._exit y se salta
    atexit: manda SIGTERM al grupo de cada server sin esperar a nadie. Sin
    esto, un wrapper (npx, sh -c) o un server que ignora el EOF de stdin
    quedaba huérfano al parar el servicio. Best-effort, nunca lanza."""
    with _STATE_LOCK:
        states = list(_STATE.values())
    for state in states:
        try:
            state.client.terminate_now()
        except Exception:  # noqa: BLE001 - el proceso está saliendo
            pass
