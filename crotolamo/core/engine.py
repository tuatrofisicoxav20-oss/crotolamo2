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
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable, Protocol

from crotolamo.logging_setup import get_logger

if TYPE_CHECKING:
    from crotolamo.core.llm import ChatResponse

log = get_logger("core.engine")

OLLAMA = "ollama"
GLM = "glm"
VALID_BACKENDS = (OLLAMA, GLM)


class LLMEngine(Protocol):
    """Contrato que cumple todo motor: LLMClient, GLMClient y FallbackLLM.

    `ToolAgent` y las interfaces solo dependen de esto; nada más del cliente
    concreto (host, base_url, api_key...) es parte del contrato.
    """

    @property
    def model(self) -> str:
        """Nombre del modelo activo (lo leen el logging y el doctor)."""
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
        on_token: Callable[[str], None] | None = None,
    ) -> "ChatResponse":
        ...


class HTTPTransport:
    """Conexión HTTP(S) persistente (keep-alive) compartida por los clientes LLM.

    POR QUÉ: `urllib.request.urlopen` abre TCP+TLS nuevos en CADA petición; un
    turno GLM con tools hace 2+ llamadas HTTPS secuenciales pagando el handshake
    completo cada vez. `http.client` reutiliza el socket entre peticiones.

    Qué maneja:
    - Reconexión transparente si el servidor cerró el keep-alive entre turnos
      (un único reintento ante `BadStatusLine`/`ConnectionError`).
    - Streaming: la respuesta devuelta se puede iterar línea a línea (SSE y
      NDJSON). Si el caller no la drena entera, la siguiente petición detecta
      la conexión sucia y abre una nueva.
    - Timeouts (de conexión y de lectura, vía el timeout del socket).

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
        self._conn: http.client.HTTPConnection | None = None
        self._last: http.client.HTTPResponse | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._https:
            return http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)

    def close(self) -> None:
        """Suelta la conexión actual (la siguiente petición abrirá otra)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - cerrar es best-effort
                pass
            self._conn = None
        self._last = None

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
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        hdrs = {"Content-Type": "application/json", **(headers or {})}

        # Si la respuesta anterior quedó a medias (p.ej. un stream que cortó en
        # [DONE] sin drenar el resto), el socket tiene bytes pendientes y no es
        # reutilizable: se tira y se abre conexión nueva.
        if self._last is not None and not self._last.isclosed():
            self.close()

        url = f"{self.base_path}{path}"
        for intento in (1, 2):
            if self._conn is None:
                self._conn = self._connect()
            try:
                self._conn.request("POST", url, body=body, headers=hdrs)
                resp = self._conn.getresponse()
            except (http.client.BadStatusLine, http.client.CannotSendRequest,
                    ConnectionError) as error:
                # El servidor cerró el keep-alive mientras esperábamos: se
                # reconecta y se reintenta UNA vez. A la segunda, se propaga.
                self.close()
                if intento == 2:
                    raise error
                continue
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
