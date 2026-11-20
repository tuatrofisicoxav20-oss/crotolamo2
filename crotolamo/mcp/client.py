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

2. **Un lock para escribir, con plazo.** stdin es un único canal; dos hilos
   escribiendo a la vez intercalarían bytes y romperían el JSON de ambos. Y es
   NO bloqueante: con un server que deja de leer, un mensaje mayor que la
   tubería (64 KiB) bloqueaba write() para siempre, fuera del timeout.

3. **Stderr drenado siempre.** Un pipe que nadie lee se llena (~64 KB) y el
   server se BLOQUEA al escribir su siguiente log: parecería colgado. Un hilo
   lo vacía y lo manda al log en DEBUG, y guarda la cola: si el server muere
   SOLO, su causa (falta un token, un módulo) va con el código de salida en
   el motivo de la muerte, que sí llega al log por defecto.

4. **Timeout = excepción propia, no cuelgue.** Un `tools/call` que no responde
   levanta `MCPTimeout`; la entrada pendiente se descarta, al server se le manda
   `notifications/cancelled` y si la respuesta llega tarde se ignora (DEBUG). El
   bridge decide qué hacer con los reincidentes.

5. **Transporte roto = todos los que esperan se enteran.** EOF en stdout,
   proceso muerto o pipe rota marcan el cliente como muerto y despiertan a todo
   el que esperaba con `MCPTransportError`, en vez de dejarlos hasta su timeout.
   Y aunque nadie esperase, `on_dead` avisa al dueño (el bridge) para que
   retire el server ya, no en la próxima llamada del modelo.
"""

from __future__ import annotations

import collections
import json
import os
import select
import signal
import subprocess
import threading
import time
from typing import Any, Callable

from crotolamo import __version__
from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import replace_surrogates

log = get_logger("mcp.client")

# Versión del protocolo que negociamos. Si el server contesta otra, seguimos:
# tools/list y tools/call son estables entre las versiones publicadas.
PROTOCOL_VERSION = "2025-06-18"

# Tope de páginas en tools/list: protege de un server que devuelva siempre
# nextCursor (bucle infinito).
_MAX_LIST_PAGES = 100

# Respiro tras cerrar stdin antes del SIGTERM (la spec: cerrar la entrada y
# ESPERAR a que el server salga solo). Un server bien hecho tarda milisegundos;
# solo el que ignora el EOF paga este segundo de más al cerrarse.
_EOF_GRACE_S = 1.0
# Lo que se espera tras el SIGTERM antes del SIGKILL. La misma en close() y en
# el apagado por señal (bridge.kill_survivors): un server lento pero bien hecho
# no muere antes por un camino que por el otro.
SIGTERM_GRACE_S = 2.0

# Cola de stderr que se guarda para explicar una muerte SOLA: el server dice
# por stderr por qué sale (falta un token, un módulo) y eso iba solo a DEBUG.
# Con tope de líneas y de largo: un volcado minificado no mete megas en el log.
_STDERR_TAIL_LINES = 20
_STDERR_LINE_CAP = 200
# Lo que se espera, al ver la muerte, a que el drenador lea lo último de stderr
# y el proceso dé su código. Suele ser inmediato (el kernel cierra las dos
# tuberías al salir); el plazo es para el que cierra stdout y sigue vivo.
_EXIT_DETAIL_WAIT_S = 0.5


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


def _write_all(fd: int, data: bytes, deadline: float | None) -> int:
    """Escribe `data` en el fd NO bloqueante sin pasarse de `deadline` (None =
    sin plazo). Devuelve los bytes escritos: menos que len(data) = venció el
    plazo con la tubería llena (el server no lee). poll y no select: select no
    admite fds >= 1024."""
    view = memoryview(data)
    sent = 0
    poller = None
    while sent < len(view):
        try:
            sent += os.write(fd, view[sent:])
            continue
        except BlockingIOError:
            pass
        wait_ms = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_ms = remaining * 1000
        if poller is None:
            poller = select.poll()
            poller.register(fd, select.POLLOUT)
        # Con el server muerto (nadie lee) poll da POLLERR y el write, EPIPE.
        poller.poll(wait_ms)
    return sent


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
        on_dead: Callable[["StdioMCPClient", str], None] | None = None,
    ) -> None:
        if not command:
            raise ValueError("el server MCP necesita un comando (argv) no vacío")
        self.name = name
        self.command = list(command)
        self.env = dict(env or {})
        self.cwd = cwd
        # Aviso al dueño (el bridge) cuando el transporte muere SOLO, sin un
        # close() nuestro: sin él, un server muerto seguía anunciando sus tools
        # hasta que el modelo llamaba una, y tras basura en stdout el proceso
        # seguía vivo sin lector. Se llama desde el hilo lector, una vez.
        self._on_dead = on_dead
        # Lo que el server dice de sí mismo en el handshake.
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
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

    @property
    def dead_reason(self) -> str:
        """Por qué se dio el transporte por roto ("" mientras no lo esté)."""
        return self._dead_reason

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
            # cierre lo hacemos nosotros, ordenado (stdin, espera, SIGTERM, SIGKILL).
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
        # stdin NO bloqueante: _send escribe con plazo (ver _write_all). Con un
        # write() bloqueante, un server que deja de leer y un mensaje mayor que
        # la tubería colgaban la llamada sin timeout, y close() detrás.
        assert self._proc.stdin is not None
        os.set_blocking(self._proc.stdin.fileno(), False)

        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.name}-stdout", daemon=True,
        )
        self._reader.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"mcp-{self.name}-stderr", daemon=True,
        )
        self._stderr_thread.start()
        # Solo el ejecutable: el argv completo puede llevar un token expandido
        # de $VARS y acabaría en el log.
        log.debug("MCP '%s': lanzado pid=%s (%s)", self.name, self._proc.pid, self.command[0])

    def _signal_group(self, proc: subprocess.Popen[bytes], sig: int) -> None:
        """Señal a TODO el grupo de procesos del server (somos su líder por
        start_new_session). Con un wrapper (npx, uvx, sh -c) el hijo directo es
        el launcher; el nieto que habla el protocolo retiene NUESTRAS tuberías,
        y matar solo al hijo dejaba al lector sin EOF y el cierre colgado."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            else:  # pragma: no cover - solo Linux en este proyecto
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass

    def _pipe_threads(self) -> list[threading.Thread]:
        """Hilos lector y drenador que hay que ver terminar (no el actual: un
        close() desde el propio hilo lector no puede esperarse a sí mismo).

        Solo los vivos: sin hilos libres ("can't start new thread") start()
        deja el Thread asignado sin arrancar, y su join() lanza; close()
        reventaba sin cerrar las tuberías y el error tumbaba el registro de
        TODOS los servers en bridge._launch. Uno que ya terminó no hay que
        esperarlo. is_alive() y no `ident`: este se pone antes de que el hilo
        cuente como arrancado, y en ese hueco join() también lanza."""
        return [
            t for t in (self._reader, self._stderr_thread)
            if t is not None and t.is_alive() and t is not threading.current_thread()
        ]

    def _wait_exit(self, proc: subprocess.Popen[bytes], timeout: float) -> bool:
        """Espera hasta `timeout` a que el hijo termine Y sus hilos vean EOF.

        Que muera el hijo directo no basta: con un wrapper (sh -c, npx, uvx) el
        nieto que habla el protocolo retiene NUESTRAS tuberías y, si sobrevive
        (ignora SIGTERM, se cuelga), dar el cierre por bueno lo dejaba huérfano
        (sobrevive a Crotolamo), con el lector bloqueado y las tuberías sin
        cerrar. EOF en ambas = nadie del grupo las retiene ya."""
        deadline = time.monotonic() + timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        for thread in self._pipe_threads():
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in self._pipe_threads())

    def _stop_process(self, proc: subprocess.Popen[bytes], sig: int, timeout: float) -> bool:
        """Manda `sig` al grupo si hace falta y espera (ver _wait_exit). True si terminó."""
        if proc.poll() is None or any(t.is_alive() for t in self._pipe_threads()):
            self._signal_group(proc, sig)
        return self._wait_exit(proc, timeout)

    def close(self) -> None:
        """Cierra el server: stdin, espera, luego SIGTERM, luego SIGKILL si se resiste.

        Idempotente y ACOTADO (~6 s peor caso). Cerrar stdin primero es la señal
        "educada" de la spec: un server bien hecho termina solo al ver EOF, y se
        le da `_EOF_GRACE_S` para hacerlo (guardar estado, borrar un lockfile)
        antes del SIGTERM; el resto es red de seguridad. Las señales van al
        grupo entero (ver _signal_group) y las tuberías solo se cierran si sus
        hilos ya salieron: cerrar un BufferedReader con el lector bloqueado
        dentro de read() se cuelga sin plazo (espera el lock del buffer), y eso
        colgaba el arranque entero cuando un server no completaba el handshake.
        """
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        self._mark_dead("cliente cerrado")
        if proc is None:
            return
        # Un escritor atascado (server que no lee) retiene stdin hasta su plazo:
        # sin EOF posible, directos al SIGTERM; al morir el server el escritor
        # ve EPIPE y suelta stdin, que se cierra al final.
        try:
            stdin_closed = self._close_stdin(_EOF_GRACE_S)
            # Antes el SIGTERM salía microsegundos después del EOF: la limpieza
            # que la spec le deja hacer al server moría a medias en cada cierre.
            if not (stdin_closed and self._wait_exit(proc, _EOF_GRACE_S)):
                if not self._stop_process(proc, signal.SIGTERM, timeout=SIGTERM_GRACE_S):
                    log.debug("MCP '%s': no terminó con SIGTERM; SIGKILL", self.name)
                    self._stop_process(proc, signal.SIGKILL, timeout=2.0)
        except BaseException:
            # Ctrl-C o SystemExit (el SIGTERM/SIGHUP del shell) A MEDIO cierre: este
            # close() ya no se puede repetir (_closed) y quien lo llamó suelta al
            # cliente (_retire, _abandon), así que close_all(), en atexit, no lo ve.
            # El server (colgado: por eso se le retiraba) sobrevivía a Crotolamo sin
            # que hubiera salido ni el SIGTERM. Quien corta se va: SIGKILL al grupo,
            # sin esperar, con el mismo criterio que _stop_process.
            if proc.poll() is None or any(t.is_alive() for t in self._pipe_threads()):
                self._signal_group(proc, signal.SIGKILL)
            raise
        if not stdin_closed:
            self._close_stdin(_EOF_GRACE_S)
        for thread, pipe in ((self._reader, proc.stdout), (self._stderr_thread, proc.stderr)):
            if pipe is None or (thread is not None and thread.is_alive()):
                # Alguien retiene el otro extremo (un nieto inmortal): el hilo
                # daemon muere con el proceso; cerrar aquí se colgaría.
                continue
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
        log.debug("MCP '%s': cerrado (returncode=%s)", self.name, proc.returncode)

    def terminate_now(self) -> None:
        """Apagado por señal del listener (os._exit se salta atexit): cierra
        stdin y manda SIGTERM al grupo SIN esperar. Best-effort, nunca lanza."""
        self._closed = True
        proc = self._proc
        self._mark_dead("apagado")
        if proc is None:
            return
        # Sin esperar: con un escritor atascado se salta el EOF; el SIGTERM basta.
        self._close_stdin(0)
        self._signal_group(proc, signal.SIGTERM)

    def kill_if_survives(self, deadline: float) -> bool:
        """La otra mitad de terminate_now(): espera hasta `deadline`
        (time.monotonic) a que el server salga y suelte las tuberías y, si
        sigue ahí, SIGKILL al grupo, sin esperar más. True si hubo que matarlo.

        terminate_now() no escala: el server que ignora el SIGTERM (o el nieto
        de un wrapper, que es quien retiene las tuberías) sobrevivía al os._exit
        del listener, en su propia sesión y sin nadie que lo recogiese. Mismo
        criterio que close() (ver _wait_exit). Con el server ya muerto vuelve
        en el acto; quien llama sale justo después, así que no se recoge nada."""
        proc = self._proc
        if proc is None:
            return False
        if self._wait_exit(proc, max(0.0, deadline - time.monotonic())):
            return False
        self._signal_group(proc, signal.SIGKILL)
        return True

    def _close_stdin(self, timeout: float) -> bool:
        """Cierra stdin (EOF para el server) si ningún escritor lo usa en
        `timeout` s. True si quedó cerrado. Bajo _write_lock: cerrar el fd con
        un escritor dentro de poll()/write() dejaría que este escribiera en
        otro archivo que reutilizase ese número de fd."""
        if not self._write_lock.acquire(timeout=timeout):
            return False
        try:
            proc = self._proc
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        finally:
            self._write_lock.release()
        return True

    # ------------------------------------------------------------------
    # Protocolo
    # ------------------------------------------------------------------

    def initialize(self, timeout: float = 15.0) -> dict[str, Any]:
        """Handshake: request `initialize` + notificación `initialized`."""
        deadline = time.monotonic() + timeout
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
        self._notify("notifications/initialized", {}, deadline)
        log.info(
            "MCP '%s': conectado a %s %s",
            self.name, self.server_info.get("name", "?"), self.server_info.get("version", ""),
        )
        return result

    def list_tools(self, timeout: float = 15.0) -> list[dict[str, Any]]:
        """`tools/list` completo, siguiendo la paginación por nextCursor.

        `timeout` es el plazo de la lista ENTERA, no de cada página: el bridge
        le pasa lo que queda del presupuesto de arranque, y con un plazo por
        página un catálogo paginado lento bloqueaba el arranque páginas x plazo
        (hasta _MAX_LIST_PAGES veces) en vez de saltarse el server.
        """
        deadline = time.monotonic() + timeout
        # El aviso nombra el plazo TOTAL: "en 0.4s" (lo que le quedaba a una
        # página) haría creer que el presupuesto configurado es ese.
        expired = f"el server MCP '{self.name}' no respondió a 'tools/list' en {timeout:g}s"
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(_MAX_LIST_PAGES):
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPTimeout(expired)
            try:
                result = self._request("tools/list", params, remaining)
            except MCPTimeout:
                raise MCPTimeout(expired) from None
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
        # UN plazo para escribir la petición y esperar su respuesta: antes solo
        # contaba la espera, y una escritura atascada no tenía límite.
        deadline = time.monotonic() + timeout
        pending = _Pending()
        with self._pending_lock:
            if self._dead:
                raise MCPTransportError(self._dead_message())
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = pending
        try:
            self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                deadline,
            )
        except BaseException:
            # CUALQUIER fallo al mandar (también un TypeError de json.dumps o un
            # Ctrl+C) retira la entrada: antes solo el de transporte, y el
            # resto dejaba una huérfana en _pending por intento.
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        if not pending.event.wait(max(0.0, deadline - time.monotonic())):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            if not self._dead:
                self._cancel(request_id, method, timeout)
            # Si murió justo mientras esperábamos, eso es lo relevante.
            if self._dead:
                raise MCPTransportError(self._dead_message())
            raise MCPTimeout(
                f"el server MCP '{self.name}' no respondió a '{method}' en {timeout:g}s"
            )

        response = pending.response
        if response is None:
            raise MCPTransportError(self._dead_message())
        # `is not None`, no `"error" in`: librerías laxas (y la JSON-RPC 1.0)
        # mandan "error": null junto al result de un ÉXITO, y se descartaba el
        # resultado bueno por un MCPError('None') (handshake caído incluido).
        error = response.get("error")
        if error is not None:
            if isinstance(error, dict):
                message = str(error.get("message", "error sin mensaje"))
                code = error.get("code")
                raise MCPError(f"{message} (código {code})" if code is not None else message)
            raise MCPError(str(error))
        return response.get("result")

    def _notify(self, method: str, params: dict[str, Any], deadline: float | None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params}, deadline)

    def _cancel(self, request_id: int, method: str, timeout: float) -> None:
        """`notifications/cancelled` para una petición que dejamos de esperar.

        La spec: el emisor DEBERÍA avisar al rendirse por timeout. Sin el aviso,
        el server seguía con una operación cuyo resultado ya nadie lee (y uno
        que atiende de uno en uno encolaba detrás la llamada siguiente). El
        `initialize` no se cancela nunca: la spec lo prohíbe. Best-effort: si la
        tubería está rota, _send ya marcó el transporte y el llamador lo ve; y
        sin esperar (plazo = ahora): con la tubería llena el server no lee, y
        no la vería, así que la llamada ya vencida no paga otro plazo por ella.
        """
        if method == "initialize":
            return
        try:
            self._notify("notifications/cancelled", {
                "requestId": request_id, "reason": f"sin respuesta en {timeout:g}s",
            }, time.monotonic())
        except MCPError:
            pass

    def _send(self, message: dict[str, Any], deadline: float | None = None) -> None:
        """Escribe UN mensaje en stdin. `deadline` (time.monotonic, None = sin
        plazo) acota la espera del lock Y la escritura: vencido sin escribir
        nada, MCPTimeout (el stream sigue entero); a medio mensaje, el stream ya
        no se puede resincronizar y el transporte se da por roto."""
        # json.dumps nunca emite saltos de línea crudos (van escapados dentro de
        # los strings), así que UNA línea = UN mensaje, como exige la spec.
        text = json.dumps(message, ensure_ascii=False) + "\n"
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            # Un sustituto suelto (medio emoji del modelo, que json.loads deja
            # pasar) no existe en UTF-8: reventaba la llamada. U+FFFD, como el
            # resto de tools (y lo que acabaría haciendo un server en JS).
            data = replace_surrogates(text).encode("utf-8")
        wait = -1.0 if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._write_lock.acquire(timeout=wait):
            raise MCPTimeout(f"el server MCP '{self.name}' no lee su entrada (stdin atascado)")
        try:
            proc = self._proc
            if proc is None or proc.stdin is None or self._dead:
                raise MCPTransportError(self._dead_message())
            try:
                sent = _write_all(proc.stdin.fileno(), data, deadline)
            except (OSError, ValueError) as error:
                # ValueError: "I/O operation on closed file" tras un close().
                # Sin esperar al drenador: aquí se tiene tomado el lock de stdin.
                detail = "" if self._closed else self._exit_detail(proc, 0.0)
                self._mark_dead(f"pipe rota al escribir: {error}{'; ' + detail if detail else ''}")
                raise MCPTransportError(self._dead_message()) from error
            if sent == 0:
                raise MCPTimeout(f"el server MCP '{self.name}' no lee su entrada (stdin lleno)")
            if sent < len(data):
                self._mark_dead("el server dejó de leer stdin a mitad de un mensaje")
                raise MCPTransportError(self._dead_message())
        finally:
            self._write_lock.release()

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
                try:
                    if isinstance(message, list):  # batch JSON-RPC (versiones viejas)
                        for item in message:
                            self._dispatch(item)
                    else:
                        self._dispatch(message)
                except Exception as error:  # noqa: BLE001 - un mensaje raro no mata el lector
                    log.warning("MCP '%s': mensaje que no pude despachar (%s): %.80r",
                                self.name, error, line)
            else:
                # EOF de verdad (no basura): sin el código de salida y la cola
                # de stderr, el aviso era un "terminó" mudo y la causa solo se
                # veía reiniciando con CROTOLAMO_LOG=DEBUG.
                detail = "" if self._closed else self._exit_detail(proc, _EXIT_DETAIL_WAIT_S)
                if detail:
                    reason = f"EOF en stdout (el server terminó; {detail})"
        except (OSError, ValueError) as error:
            reason = f"error leyendo stdout: {error}"
        finally:
            if not self._closed:
                log.debug("MCP '%s': transporte cerrado: %s", self.name, reason)
            self._mark_dead(reason)
            if not self._closed and self._on_dead is not None:
                try:
                    self._on_dead(self, self._dead_reason)
                except Exception as error:  # noqa: BLE001 - p.ej. sin hilos al salir
                    log.debug("MCP '%s': el aviso de muerte falló: %s", self.name, error)

    def _exit_detail(self, proc: subprocess.Popen[bytes], wait_s: float) -> str:
        """Por qué murió el server que se murió solo: su código de salida (o la
        señal que lo mató) y la cola de su stderr ("" si no hay nada). Solo
        para el log: el texto que ve el modelo lo pone el bridge, fijo. Espera
        como mucho `wait_s` a que el drenador acabe y el proceso salga."""
        deadline = time.monotonic() + wait_s
        stderr_thread = self._stderr_thread
        # Un drenador sin arrancar (sin hilos libres, o aún dentro de su start()
        # mientras el lector ya ve morir al server) no se espera: su join()
        # lanza y el lector moría sin dar el motivo (ver _pipe_threads).
        if (stderr_thread is not None and stderr_thread.is_alive()
                and stderr_thread is not threading.current_thread()):
            stderr_thread.join(wait_s)
        parts: list[str] = []
        try:
            code = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass  # cerró sus tuberías pero sigue vivo: no hay código que dar
        else:
            if code >= 0:
                parts.append(f"código de salida {code}")
            else:
                try:
                    parts.append(f"matado por {signal.Signals(-code).name}")
                except ValueError:
                    parts.append(f"matado por la señal {-code}")
        tail = list(self._stderr_tail)
        if tail:
            parts.append("stderr: " + " | ".join(tail))
        return "; ".join(parts)

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
        if not isinstance(message["id"], int) or isinstance(message["id"], bool):
            # Nuestros ids son enteros: cualquier otra cosa (una lista, un dict,
            # un string) no puede ser respuesta a algo nuestro y, si no es
            # hasheable, reventaría el hilo lector al buscarla en _pending.
            log.debug("MCP '%s': respuesta con id inválido %r, ignorada", self.name, message["id"])
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
        # Las respuestas de aquí salen del hilo lector, sin plazo (deadline
        # None): si el server no lee, close() no espera a este escritor (ver
        # _close_stdin) y el SIGTERM lo desatasca con EPIPE.
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
                    self._stderr_tail.append(text[:_STDERR_LINE_CAP])
                    log.debug("MCP '%s' stderr: %s", self.name, text)
        except (OSError, ValueError):
            pass
