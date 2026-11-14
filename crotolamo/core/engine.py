"""Selección del motor de inferencia: Ollama (local) o GLM (nube).

Un único punto donde se decide con qué LLM habla Crotolamo. Ambos clientes
exponen el mismo contrato (`LLMEngine`: chat, chat_stream -> ChatResponse), así
que el resto del código —`ToolAgent`, `Conversation`, las tools— no distingue
cuál corre.

Se elige con [llm].backend en la config, o con la env CROTOLAMO_LLM_BACKEND
(que manda sobre el toml, útil para probar sin editar archivos).

Aquí vive también `HTTPTransport`, la base HTTP con keep-alive que comparten
los dos clientes.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import threading
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable, Protocol

from crotolamo.logging_setup import get_logger

if TYPE_CHECKING:
    from crotolamo.core.llm import ChatResponse

log = get_logger("core.engine")

OLLAMA = "ollama"
GLM = "glm"
VALID_BACKENDS = (OLLAMA, GLM)

# Reloj del transporte (tope del drenaje y edad de un keep-alive). En Linux,
# CLOCK_BOOTTIME: sigue contando mientras el portátil está suspendido, y
# monotonic() no; tras despertar, un socket de "hace 5 s" puede tener horas.
_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", None)


def _reloj() -> float:
    if _BOOTTIME is not None:
        return time.clock_gettime(_BOOTTIME)
    return time.monotonic()


class LLMEngine(Protocol):
    """Contrato que cumple todo motor: LLMClient, GLMClient y FallbackLLM.

    `ToolAgent` y las interfaces solo dependen de esto; nada más del cliente
    concreto (host, base_url, api_key...) es parte del contrato.
    """

    @property
    def model(self) -> str:
        """Nombre del modelo que respondería ahora mismo."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> "ChatResponse":
        ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        # El motor ignora lo que devuelva on_token; solo FallbackLLM lo mira
        # (False = "retenido, no sonó"), por eso no se exige `-> None`. Si
        # on_token trae `checkpoint` (LiveStreamer), el motor lo consulta con
        # cada línea del stream: es por donde se cancela un turno a medias.
        on_token: Callable[[str], object] | None = None,
    ) -> "ChatResponse":
        ...


class HTTPTransport:
    """Conexión HTTP(S) persistente (keep-alive) compartida por los clientes LLM.

    POR QUÉ: `urllib.request.urlopen` abre TCP+TLS nuevos en CADA petición; un
    turno GLM con tools hace 2+ llamadas HTTPS secuenciales pagando el handshake
    completo cada vez. `http.client` reutiliza el socket entre peticiones.

    Qué maneja:
    - Reconexión transparente si el servidor cerró el keep-alive entre turnos
      (un único reintento ante `BadStatusLine`/`ConnectionError`, y el
      `ssl.SSLEOFError` con que Python 3.11/3.12 avisa de lo mismo sobre TLS).
    - Streaming: la respuesta devuelta se puede iterar línea a línea (SSE y
      NDJSON). El parser se detiene en el fin LÓGICO ([DONE] / done:true), un
      paso antes del fin FÍSICO del cuerpo (el chunk terminal). Si el cliente
      lo avisa con `finish(resp)`, la siguiente petición de ese hilo lee esa
      cola, acotada en bytes y en tiempo total, y reutiliza el keep-alive.
      Sin el aviso, si la cola no llega a tiempo o si la conexión lleva más
      de `_REUSE_MAX_IDLE_S` quieta, la conexión sucia se tira y se abre
      otra, como antes.
    - Un fallo a media petición (timeout esperando la respuesta, por ejemplo)
      cierra la conexión en el acto: con el socket abierto el servidor seguía
      generando para nadie hasta la siguiente petición de ese hilo.
    - Timeouts (de conexión y de lectura, vía el timeout del socket).

    SEGURO ENTRE HILOS vía conexión POR HILO (threading.local): en el modo de
    voz concurrente, WarmLLM y BrainThread pueden llamar chat() sobre el mismo
    cliente —y por tanto este transport— a la vez; con un socket único eso lo
    corrompía (CannotSendRequest, respuestas cruzadas). Cada hilo tiene ahora
    su propio keep-alive, sin locks ni contención. close() cierra SOLO la
    conexión del hilo que lo llama (nadie necesita hoy un cierre global; los
    sockets de hilos muertos los recoge el GC).

    Stdlib puro, igual que el resto del proyecto: cero dependencias.
    """

    def __init__(self, base_url: str, timeout: float = 60) -> None:
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"URL inválida para el transporte HTTP: {base_url!r}")
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port  # None = puerto por defecto del esquema
        self.base_path = parts.path.rstrip("/")
        self.timeout = timeout
        # Estado por hilo: .conn (la conexión keep-alive), .last (la última
        # respuesta, para detectar streams a medias), .finishable (el
        # consumidor avisó que .last llegó a su fin lógico) y .finished_at
        # (cuándo, con `_reloj`). Ver docstring.
        self._local = threading.local()

    @property
    def _conn(self) -> http.client.HTTPConnection | None:
        return getattr(self._local, "conn", None)

    @_conn.setter
    def _conn(self, value: http.client.HTTPConnection | None) -> None:
        self._local.conn = value

    @property
    def _last(self) -> http.client.HTTPResponse | None:
        return getattr(self._local, "last", None)

    @_last.setter
    def _last(self, value: http.client.HTTPResponse | None) -> None:
        self._local.last = value

    def _connect(self) -> http.client.HTTPConnection:
        if self._https:
            return http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)

    # Cola de un stream ya terminado lógicamente: línea en blanco + chunk
    # terminal, unos pocos bytes que para la siguiente petición ya llegaron.
    # Los topes acotan el caso raro (un servidor que sigue mandando o que
    # retrasa el chunk terminal): pasado cualquiera, se descarta como antes.
    # El de tiempo es del drenaje ENTERO, no de cada lectura: un goteo más
    # rápido que eso retenía la petición hasta juntar los 64 KiB.
    _DRAIN_BYTES = 65536
    _DRAIN_TIMEOUT_S = 0.05
    # Un stream terminado solo deja su keep-alive a la petición que llega en
    # estos segundos (las llamadas de un turno con tools, una réplica rápida).
    # Un socket muerto en silencio (NAT, cambio de wifi) no avisa: escribir
    # "funciona" y la respuesta no llega nunca, así que costaría el timeout de
    # lectura entero y abriría el breaker. Antes del finish() los streams
    # nunca reutilizaban; pasado este tope, conexión nueva como entonces.
    _REUSE_MAX_IDLE_S = 15.0

    def finish(self, resp: Any) -> None:
        """El consumidor llegó al fin LÓGICO de `resp` ([DONE] / done:true).

        Solo lo ANOTA, así que el turno en curso no paga nada: la lectura de
        la cola la hace la siguiente petición de este hilo (ver `_drain`).
        Un aviso sobre otra respuesta, o desde otro hilo, no cuenta.
        """
        if resp is not None and resp is self._last:
            self._local.finishable = True
            self._local.finished_at = _reloj()

    def _drain(self, resp: Any) -> bool:
        """Lee la cola de `resp`, acotada; True si el socket quedó limpio.

        read1 lee lo de UN chunk cada vez, con el timeout del socket recortado
        a lo que queda del tope: entre lectura y lectura se mira el reloj, así
        que un goteo no estira el total más allá de _DRAIN_TIMEOUT_S (más la
        última espera, acotada por lo que quedaba).
        """
        if getattr(resp, "will_close", False):
            return False  # `Connection: close`: el servidor no lo va a reutilizar
        sock = getattr(self._conn, "sock", None)
        limite = _reloj() + self._DRAIN_TIMEOUT_S
        leidos = 0
        try:
            previous = sock.gettimeout() if sock is not None else None
            try:
                while not resp.isclosed():
                    queda = limite - _reloj()
                    if queda <= 0 or leidos >= self._DRAIN_BYTES:
                        return False
                    if sock is not None:
                        sock.settimeout(queda)
                    data = resp.read1(self._DRAIN_BYTES - leidos)
                    if not data:
                        break
                    leidos += len(data)
            finally:
                if sock is not None:
                    sock.settimeout(previous)
        except Exception:  # noqa: BLE001 - si no se puede drenar, se descarta
            return False
        return bool(resp.isclosed())

    def close(self) -> None:
        """Suelta la conexión actual (la siguiente petición abrirá otra)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - cerrar es best-effort
                pass
            self._conn = None
        self._last = None
        self._local.finishable = False

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> http.client.HTTPResponse:
        """POST JSON a `base_path + path` reutilizando la conexión si se puede.

        Devuelve la respuesta SIN leer el cuerpo: el caller decide si hace
        `.read()` completo o la itera línea a línea (streaming). No lanza por
        status >= 400; eso lo mapea cada cliente a sus mensajes en personaje.
        """
        # errors="replace": el historial puede traer sustitutos sueltos (el
        # nombre de un archivo con bytes no-UTF-8 que listó una tool, un
        # "\ud83d" suelto de un JSON ajeno). Con el codec estricto eso lanzaba
        # UnicodeEncodeError aquí, en ESTE turno y en todos los siguientes
        # mientras el mensaje siguiera en el historial. Lo irrepresentable sale
        # como "?"; el texto normal se codifica byte a byte igual que antes.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
        hdrs = {"Content-Type": "application/json", **(headers or {})}

        # Si la respuesta anterior quedó a medias, el socket tiene bytes
        # pendientes y no es reutilizable: se tira y se abre conexión nueva.
        # Salvo que el consumidor avisara (finish) que llegó al fin lógico: lo
        # que falta es solo la cola del cuerpo, que se lee y el socket sigue.
        last = self._last
        finishable = getattr(self._local, "finishable", False) and (
            _reloj() - self._local.finished_at <= self._REUSE_MAX_IDLE_S)
        self._local.finishable = False
        if last is not None and not last.isclosed():
            if not (finishable and self._drain(last)):
                self.close()

        url = f"{self.base_path}{path}"
        for intento in (1, 2):
            if self._conn is None:
                self._conn = self._connect()
            try:
                self._conn.request("POST", url, body=body, headers=hdrs)
                resp = self._conn.getresponse()
            except (http.client.BadStatusLine, http.client.CannotSendRequest,
                    ConnectionError, ssl.SSLEOFError) as error:
                # El servidor cerró el keep-alive mientras esperábamos: se
                # reconecta y se reintenta UNA vez. A la segunda, se propaga.
                # SSLEOFError es ese mismo socket muerto visto por TLS en Python
                # 3.11/3.12 (en 3.13 ya llega como ConnectionError).
                self.close()
                if intento == 2:
                    raise error
                continue
            except BaseException:
                # Cualquier otro fallo (timeout esperando la respuesta, red
                # caída): la petición quedó a medias y el servidor seguiría
                # generando para nadie. Se cuelga ya; el error sigue su camino.
                self.close()
                raise
            self._last = resp
            return resp
        raise AssertionError("inalcanzable")  # pragma: no cover


def resolve_backend(settings) -> str:
    """Nombre del backend a usar. La env pisa el toml; un valor raro cae a ollama."""
    raw = os.environ.get("CROTOLAMO_LLM_BACKEND") or settings.llm.get("backend", OLLAMA)
    backend = str(raw).strip().lower()
    if backend not in VALID_BACKENDS:
        log.warning("backend '%s' desconocido; uso '%s'", backend, OLLAMA)
        return OLLAMA
    return backend


def build_primary_llm(settings):
    """Cliente primario SIN envolver en FallbackLLM, con transport propio.

    Para usos auxiliares (hoy: el warm-up del listener) que no deben compartir
    ni el socket ni el circuit breaker del cliente del loop: un fallo del
    warm-up sobre el FallbackLLM compartido abriría el breaker 60s y degradaría
    el primer comando real al modelo local sin motivo.
    """
    backend = resolve_backend(settings)
    if backend == GLM:
        from crotolamo.core.glm import GLMClient, _find_api_key

        if _find_api_key() is not None:
            return GLMClient.from_settings(settings)

    from crotolamo.core.llm import LLMClient

    return LLMClient.from_settings(settings)


def build_llm(settings):
    """Construye el cliente del backend configurado.

    Con GLM, el cliente va envuelto en `FallbackLLM`: si la nube falla EN CALIENTE
    (se cayó el wifi, la key caducó, un 429), los turnos siguientes los atiende
    Ollama local. Sin eso, un corte de internet dejaba a Crotolamo respondiendo
    "¿Hay internet, patrón?" en vez de usar el modelo que ya está instalado.

    Si se pide GLM y no hay API key, ni siquiera se intenta: se usa Ollama directo.
    Más vale un Crotolamo lento que un Crotolamo mudo.
    """
    backend = resolve_backend(settings)

    from crotolamo.core.llm import LLMClient

    local = LLMClient.from_settings(settings)

    if backend == GLM:
        from crotolamo.core.glm import GLMClient, _find_api_key

        if _find_api_key() is None:
            log.warning(
                "backend=glm pero no hay API key (CROTOLAMO_GLM_API_KEY). "
                "Caigo a Ollama local."
            )
        else:
            from crotolamo.core.fallback import FallbackLLM

            remote = GLMClient.from_settings(settings)
            cooldown = settings.llm.get("glm", {}).get("fallback_cooldown_s", 60.0)
            log.info(
                "motor: GLM (%s) vía %s, con respaldo local (%s)",
                remote.model, remote.base_url, local.model,
            )
            return FallbackLLM(remote, local, cooldown_s=cooldown)

    log.info("motor: Ollama (%s) en %s", local.model, local.host)
    return local
