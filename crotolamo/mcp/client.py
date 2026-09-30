"""Cliente MCP por stdio: JSON-RPC 2.0 sobre las tuberías de un subproceso.

Transporte según la spec MCP (2025-06-18): el cliente lanza el server como
subproceso, le escribe mensajes JSON por stdin y lee sus respuestas por stdout,
UN mensaje por línea (sin saltos de línea embebidos). Stderr es libre para logs
del server. Stdlib puro: subprocess + threading + json, cero SDKs.

DISEÑO, y por qué:

1. **Hilo lector + futuros por id.** Las respuestas pueden llegar en cualquier
   orden (y entre medias el server manda notificaciones o sus propios requests).
   Un hilo daemon lee stdout línea a línea y despacha cada respuesta al
   `threading.Event` que espera por ese `id`. Así varias llamadas desde hilos
   distintos (BrainThread, un warm-up...) no se pisan.

2. **Un lock para escribir.** stdin es un único canal; dos hilos escribiendo a
   la vez intercalarían bytes y romperían el JSON de ambos.

3. **Stderr drenado siempre.** Un pipe que nadie lee se llena (~64 KB) y el
   server se BLOQUEA al escribir su siguiente log: parecería colgado. Un hilo
   lo vacía y lo manda al log en DEBUG.

4. **Timeout = excepción propia, no cuelgue.** Un `tools/call` que no responde
   levanta `MCPTimeout`; la entrada pendiente se descarta y si la respuesta llega
   tarde se ignora (DEBUG). El bridge decide qué hacer con los reincidentes.

5. **Transporte roto = todos los que esperan se enteran.** EOF en stdout,
   proceso muerto o pipe rota marcan el cliente como muerto y despiertan a todo
   el que esperaba con `MCPTransportError`, en vez de dejarlos hasta su timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any

from crotolamo import __version__
from crotolamo.logging_setup import get_logger

log = get_logger("mcp.client")

# Versión del protocolo que negociamos. Si el server contesta otra, seguimos:
# tools/list y tools/call son estables entre las versiones publicadas.
PROTOCOL_VERSION = "2025-06-18"

# Tope de páginas en tools/list: protege de un server que devuelva siempre
# nextCursor (bucle infinito).
_MAX_LIST_PAGES = 100


class MCPError(Exception):
    """Fallo lógico del server: respuesta JSON-RPC con `error`, o protocolo raro."""


class MCPTimeout(MCPError):
    """El server no respondió a tiempo. El cliente sigue vivo."""


class MCPTransportError(MCPError):
    """Proceso muerto, EOF en stdout, pipe rota o basura irrecuperable en stdout.
    El cliente queda inservible: hay que cerrarlo y, si acaso, crear otro."""


class _Pending:
    """Futuro casero para una respuesta JSON-RPC: un Event y el mensaje."""

    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None


def _summarize_content(content: Any, structured: Any = None) -> str:
    """Aplana el `content` de un tools/call a texto para el LLM.

    Los items `text` se unen con saltos de línea; lo que no es texto (imágenes,
    audio, recursos) se resume en una marca para que el modelo sepa que hubo
    algo, sin meterle base64 al contexto.
    """
    parts: list[str] = []
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "image":
            parts.append(f"[imagen {item.get('mimeType', 'sin tipo')}]")
        elif kind == "audio":
            parts.append(f"[audio {item.get('mimeType', 'sin tipo')}]")
        elif kind == "resource":
            raw_resource = item.get("resource")
            resource: dict[str, Any] = raw_resource if isinstance(raw_resource, dict) else {}
            marca = f"[recurso {resource.get('uri', 'sin uri')}]"
            texto = resource.get("text")
            parts.append(f"{marca}\n{texto}" if isinstance(texto, str) and texto else marca)
        elif kind == "resource_link":
            parts.append(f"[recurso {item.get('uri') or item.get('name') or 'sin uri'}]")
        else:
            parts.append(f"[{kind or 'contenido'} sin texto]")
    text = "\n".join(parts)
    # 2025-06-18: un server puede responder solo con structuredContent (JSON).
    if not text.strip() and structured is not None:
        try:
            text = json.dumps(structured, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(structured)
    return text


class StdioMCPClient:
    """Cliente de UN server MCP lanzado como subproceso (transporte stdio).

    Ciclo de vida: `start()` -> `initialize()` -> `list_tools()` / `call_tool()`
    -> `close()`. Seguro entre hilos para las llamadas; `close()` es idempotente.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("el server MCP necesita un comando (argv) no vacío")
        self.name = name
        self.command = list(command)
        self.env = dict(env or {})
        self.cwd = cwd
        # Lo que el server dice de sí mismo en el handshake.
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 0
        self._dead = False
        self._dead_reason = ""
        self._closed = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        """True si el proceso corre y el transporte no se ha dado por roto."""
        proc = self._proc
        return proc is not None and not self._dead and proc.poll() is None

    def start(self) -> None:
        """Lanza el proceso y los hilos lector (stdout) y drenador (stderr)."""
        if self._closed:
            raise MCPTransportError(f"el cliente MCP '{self.name}' ya se cerró")
        if self._proc is not None:
            return  # ya arrancado: idempotente
        env = {**os.environ, **self.env}
        try:
            # start_new_session: el server no hereda el grupo de proceso de la
            # terminal, así el Ctrl+C del patrón no lo mata "por debajo" y el
            # cierre lo hacemos nosotros, ordenado (stdin -> SIGTERM -> SIGKILL).
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.cwd or None,
                start_new_session=True,
            )
        except (OSError, ValueError) as error:
            self._dead = True
            self._dead_reason = f"no pude lanzar {self.command[0]!r}: {error}"
            raise MCPTransportError(
                f"no pude lanzar el server MCP '{self.name}' ({error})"
            ) from error

        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.name}-stdout", daemon=True,
        )
        self._reader.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"mcp-{self.name}-stderr", daemon=True,
        )
        self._stderr_thread.start()
        log.debug("MCP '%s': lanzado pid=%s (%s)", self.name, self._proc.pid, self.command)

    def close(self) -> None:
        """Cierra el server: stdin, luego SIGTERM, luego SIGKILL si se resiste.

        Idempotente. Cerrar stdin primero es la señal "educada" de la spec: un
        server bien hecho termina solo al ver EOF; el resto es red de seguridad.
        """
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        self._mark_dead("cliente cerrado")
        if proc is None:
            return
        for step in (self._close_stdin, proc.terminate):
            try:
                step()
            except (OSError, ValueError):
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            log.debug("MCP '%s': no terminó con SIGTERM; SIGKILL", self.name)
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        # Los hilos ven EOF al morir el proceso; les damos un respiro para no
        # dejar descriptores abiertos a medias.
        for thread in (self._reader, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1)
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except (OSError, ValueError):
                pass
        log.debug("MCP '%s': cerrado (returncode=%s)", self.name, proc.returncode)

    def _close_stdin(self) -> None:
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()

    # ------------------------------------------------------------------
    # Protocolo
    # ------------------------------------------------------------------

    def initialize(self, timeout: float = 15.0) -> dict[str, Any]:
        """Handshake: request `initialize` + notificación `initialized`."""
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "crotolamo2", "version": __version__},
            },
            timeout,
        )
        if not isinstance(result, dict):
            raise MCPError(f"el server MCP '{self.name}' respondió un initialize raro")
        self.server_info = result.get("serverInfo") or {}
        self.server_capabilities = result.get("capabilities") or {}
        self.protocol_version = result.get("protocolVersion")
        if self.protocol_version != PROTOCOL_VERSION:
            log.info(
                "MCP '%s': el server habla protocolo %s (yo %s); sigo, tools/* es estable",
                self.name, self.protocol_version, PROTOCOL_VERSION,
            )
        self._notify("notifications/initialized", {})
        log.info(
            "MCP '%s': conectado a %s %s",
            self.name, self.server_info.get("name", "?"), self.server_info.get("version", ""),
        )
        return result

    def list_tools(self, timeout: float = 15.0) -> list[dict[str, Any]]:
        """`tools/list` completo, siguiendo la paginación por nextCursor."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(_MAX_LIST_PAGES):
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params, timeout)
            if not isinstance(result, dict):
                raise MCPError(f"el server MCP '{self.name}' respondió un tools/list raro")
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(t for t in page if isinstance(t, dict))
            cursor = result.get("nextCursor") or None
            if not cursor:
                break
            if cursor in seen_cursors:
                log.warning("MCP '%s': nextCursor repetido en tools/list; corto", self.name)
                break
            seen_cursors.add(cursor)
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        timeout: float = 20.0,
    ) -> tuple[str, bool]:
        """`tools/call`. Devuelve (texto aplanado, is_error).

        Un error JSON-RPC (tool inexistente, params inválidos) levanta MCPError;
        un fallo "de negocio" que el server reporta con `isError: true` llega
        como texto + True, para que el LLM lo lea y reaccione.
        """
        result = self._request(
            "tools/call", {"name": name, "arguments": dict(arguments or {})}, timeout,
        )
        if not isinstance(result, dict):
            raise MCPError(f"el server MCP '{self.name}' respondió un tools/call raro")
        text = _summarize_content(result.get("content"), result.get("structuredContent"))
        return text, bool(result.get("isError", False))

    # ------------------------------------------------------------------
    # JSON-RPC
    # ------------------------------------------------------------------

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        pending = _Pending()
        with self._pending_lock:
            if self._dead:
                raise MCPTransportError(self._dead_message())
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = pending
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        except MCPTransportError:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        if not pending.event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            # Si murió justo mientras esperábamos, eso es lo relevante.
            if self._dead:
                raise MCPTransportError(self._dead_message())
            raise MCPTimeout(
                f"el server MCP '{self.name}' no respondió a '{method}' en {timeout:g}s"
            )

        response = pending.response
        if response is None:
            raise MCPTransportError(self._dead_message())
        if "error" in response:
            error = response.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", "error sin mensaje"))
                code = error.get("code")
                raise MCPError(f"{message} (código {code})" if code is not None else message)
            raise MCPError(str(error))
        return response.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict[str, Any]) -> None:
        # json.dumps nunca emite saltos de línea crudos (van escapados dentro de
        # los strings), así que UNA línea = UN mensaje, como exige la spec.
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            proc = self._proc
            if proc is None or proc.stdin is None or self._dead:
                raise MCPTransportError(self._dead_message())
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                # ValueError: "write to closed file" tras un close().
                self._mark_dead(f"pipe rota al escribir: {error}")
                raise MCPTransportError(self._dead_message()) from error

    def _dead_message(self) -> str:
        return f"el server MCP '{self.name}' no está disponible ({self._dead_reason or 'muerto'})"

    def _mark_dead(self, reason: str) -> None:
        """Marca el transporte como roto y despierta a todo el que esperaba."""
        with self._pending_lock:
            if not self._dead:
                self._dead = True
                self._dead_reason = reason
            waiting = list(self._pending.values())
            self._pending.clear()
        for pending in waiting:
            pending.event.set()

    # ------------------------------------------------------------------
    # Hilos
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        reason = "EOF en stdout (el server terminó)"
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # La spec exige stdout LIMPIO (solo mensajes MCP). Una línea
                    # que no es JSON significa que el server escribe basura por
                    # el canal del protocolo: no hay forma de resincronizar.
                    reason = f"basura no-JSON en stdout: {line[:80]!r}"
                    log.warning("MCP '%s': %s", self.name, reason)
                    break
                if isinstance(message, list):  # batch JSON-RPC (versiones viejas)
                    for item in message:
                        self._dispatch(item)
                else:
                    self._dispatch(message)
        except (OSError, ValueError) as error:
            reason = f"error leyendo stdout: {error}"
        finally:
            if not self._closed:
                log.debug("MCP '%s': transporte cerrado: %s", self.name, reason)
            self._mark_dead(reason)

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            log.debug("MCP '%s': mensaje que no es objeto, ignorado: %r", self.name, message)
            return
        has_id = "id" in message and message.get("id") is not None
        if "method" in message:
            self._handle_incoming(message, has_id)
            return
        if not has_id:
            log.debug("MCP '%s': mensaje sin id ni method, ignorado", self.name)
            return
        with self._pending_lock:
            pending = self._pending.pop(message["id"], None)
        if pending is None:
            # Respuesta tardía a una llamada que ya expiró por timeout, o id ajeno.
            log.debug("MCP '%s': respuesta huérfana id=%r, ignorada", self.name, message["id"])
            return
        pending.response = message
        pending.event.set()

    def _handle_incoming(self, message: dict[str, Any], has_id: bool) -> None:
        """Requests/notificaciones que manda EL SERVER. Solo atendemos ping."""
        method = message.get("method")
        if method == "ping" and has_id:
            try:
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            except MCPTransportError:
                pass
            return
        if has_id:
            # Un request que no soportamos (sampling, roots, elicitation...): se
            # responde "método no encontrado" para que el server no se quede
            # esperando una respuesta que nunca llegará.
            log.debug("MCP '%s': request del server no soportado: %s", self.name, method)
            try:
                self._send({
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": f"método no soportado: {method}"},
                })
            except MCPTransportError:
                pass
            return
        log.debug("MCP '%s': notificación del server ignorada: %s", self.name, method)

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in proc.stderr:
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    log.debug("MCP '%s' stderr: %s", self.name, text)
        except (OSError, ValueError):
            pass
