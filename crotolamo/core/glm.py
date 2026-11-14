"""Cliente GLM (Z.ai / Zhipu) con tool-calling, hablando el contrato OpenAI.

POR QUÉ EXISTE: en esta lap (12 núcleos, 15 GiB, GPU integrada) inferir un 7B en
CPU cuesta ~17-22s por turno con tool, y deja la RAM en swap. GLM-4.7-Flash es
gratuito en Z.ai y responde en ~1-2s, sacando la inferencia de la máquina.

DISEÑO: esto es un ADAPTADOR. La traducción de mensajes Ollama<->OpenAI vive en
`openai_adapter.py` y el parseo del stream en `sse.py`; aquí queda solo el
cliente HTTP, que devuelve un `ChatResponse` idéntico al de `LLMClient`. Así
`Conversation`, `ToolAgent` y las tools no se enteran de con qué motor hablan.

Cero dependencias: stdlib, igual que `llm.py`. La red va por `HTTPTransport`
(keep-alive), compartiendo conexión entre las 2+ llamadas de un turno con tools.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from typing import Any, NoReturn

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.llm import (
    ChatResponse,
    LLMError,
    TransientLLMError,
    _parse_tool_calls,
    _read_body,
)
from crotolamo.core.openai_adapter import from_openai_message, to_openai_messages
from crotolamo.core.sse import SSEError, SSETruncated, parse_sse
from crotolamo.logging_setup import get_logger

log = get_logger("core.glm")

_PROF = os.environ.get("CROTOLAMO_LLM_PROF") == "1"

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.7-flash"  # gratuito para usuarios registrados
# Variables de entorno donde se busca la API key, en orden. NUNCA se guarda la
# key en el toml (que va a git); la env es el sitio correcto.
API_KEY_ENVS = ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY")

# Un 429 con Retry-After hasta este tope se espera y reintenta in-place, en vez
# de tumbar el turno (y con él, degradar 60s al modelo local vía FallbackLLM).
_MAX_RETRY_AFTER_S = 3.0


class GLMAuthError(LLMError):
    """Falta la API key o el servidor la rechazó."""


def _find_api_key() -> str | None:
    for env in API_KEY_ENVS:
        key = os.environ.get(env)
        if key and key.strip():
            return key.strip()
    return None


def _retry_after_s(resp) -> float | None:
    """Segundos del header Retry-After, o None si falta o no es numérico."""
    raw = resp.getheader("Retry-After")
    if not raw:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None  # formato HTTP-date: lo tratamos como "espera larga"


class GLMClient:
    """Mismo contrato público que `LLMClient`: chat(), chat_stream(), from_settings()."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        timeout: float = 60,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> None:
        self.api_key = api_key or _find_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # 60s sobra: la nube responde en ~1-2s. El timeout largo de Ollama (180s)
        # existía por el arranque en frío del modelo local, que aquí no aplica.
        self.timeout = timeout
        self.max_tokens = max_tokens
        # `thinking`: GLM-4.7 razona ANTES de responder y lo trae activado por
        # defecto. MEDIDO contra la API real: para un "hola" gasta ~460 tokens de
        # reasoning y tarda 6.9s, contra 1.2s sin él (5.6x). El tool-calling acierta
        # igual con y sin (3/3). En un asistente de VOZ esos 5s de más son veneno,
        # así que aquí va apagado. Además reasoning_tokens cuenta contra el rate
        # limit del tier gratuito. Actívalo si quieres razonamiento en preguntas
        # difíciles y no te importa esperar.
        self.thinking = thinking
        # Conexión HTTPS persistente: un turno con tools hace 2+ llamadas
        # seguidas; reutilizar el socket ahorra el handshake TCP+TLS de cada una.
        self._transport = HTTPTransport(self.base_url, timeout=self.timeout)

    @classmethod
    def from_settings(cls, settings) -> "GLMClient":
        llm = settings.llm
        glm = llm.get("glm", {}) if isinstance(llm.get("glm"), dict) else {}
        return cls(
            base_url=glm.get("base_url", DEFAULT_BASE_URL),
            model=glm.get("model", DEFAULT_MODEL),
            temperature=llm.get("temperature", 0.2),
            timeout=glm.get("timeout", 60),
            max_tokens=glm.get("max_tokens"),
            thinking=glm.get("thinking", False),
        )

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise GLMAuthError(
                "No traigo API key de GLM, patrón. Exporta CROTOLAMO_GLM_API_KEY "
                "(la sacas gratis en https://z.ai) o cambia [llm].backend a "
                '"ollama" en la config.'
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            # `Tool.schema()` ya emite el formato OpenAI ({"type":"function",...}).
            payload["tools"] = tools
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if not self.thinking:
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _raise_http(self, status: int, body: str, location: str | None = None) -> NoReturn:
        if status < 400:
            # Permanente, como un 400: es la config y no una caída. Esperar 60s
            # con el modelo local no cambia a dónde apunta base_url.
            raise LLMError(
                f"GLM me mandó a otra dirección ({status}) en vez de responder, patrón. "
                f"Revisa [llm.glm].base_url; me redirige a {location or 'no sé dónde'}."
            )
        if status in (401, 403):
            raise GLMAuthError(
                f"GLM me rechazó la credencial ({status}), patrón. "
                f"Revisa la API key. {body}"
            )
        if status == 429:
            raise TransientLLMError(
                "GLM me está limitando el ritmo (429), patrón. Aguanta tantito."
            )
        if status >= 500:
            raise TransientLLMError(f"GLM respondió {status}, patrón: {body}")
        # 4xx restantes (400 payload roto, etc.): bug de cliente, NO abre el breaker.
        raise LLMError(f"GLM respondió {status}, patrón: {body}")

    def _post(self, payload: dict[str, Any]) -> http.client.HTTPResponse:
        """POST a /chat/completions con los errores de red ya en personaje."""
        headers = self._headers()  # puede lanzar GLMAuthError sin tocar la red
        try:
            return self._transport.post_json("/chat/completions", payload, headers)
        except TimeoutError as error:
            raise TransientLLMError(
                f"GLM no respondió a tiempo, patrón. ({error})") from error
        except (http.client.HTTPException, OSError) as error:
            raise TransientLLMError(
                f"No pude hablar con GLM en {self.base_url}, patrón. "
                f"¿Hay internet? ({error})"
            ) from error

    def _send(self, payload: dict[str, Any]) -> http.client.HTTPResponse:
        """Envía y valida el status. Un 429 con Retry-After corto se reintenta
        UNA vez in-place: sin esto, un límite transitorio de segundos tumbaba el
        turno y `FallbackLLM` degradaba 60s al modelo local (lento en CPU).
        """
        resp = self._post(payload)
        if resp.status == 429:
            wait = _retry_after_s(resp)
            if wait is not None and 0 <= wait <= _MAX_RETRY_AFTER_S:
                _read_body(resp)  # drenar para poder reutilizar la conexión
                time.sleep(wait)
                resp = self._post(payload)
        # >= 300 y no >= 400: http.client NO sigue redirecciones, así que el 301
        # de un base_url con http:// (o con la ruta vieja) pasaba por respuesta
        # y su cuerpo vacío, por una contestación en blanco.
        if resp.status >= 300:
            self._raise_http(resp.status, _read_body(resp), resp.getheader("Location"))
        return resp

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        t0 = time.time() if _PROF else 0.0
        resp = self._send(self._payload(messages, tools, stream=False))
        try:
            raw = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, OSError) as error:
            # El cuerpo no llegó entero: el socket queda a medias. Se cuelga ya
            # en vez de dejarlo abierto hasta la siguiente petición de este hilo.
            self._transport.close()
            raise TransientLLMError(
                f"GLM no respondió a tiempo, patrón. ({error})") from error
        except http.client.HTTPException as error:
            # Cuerpo cortado a medias (IncompleteRead): es HTTPException, NO
            # OSError. Sin esta rama salía cruda y FallbackLLM ni abría el
            # breaker ni probaba el modelo local (ver LLMClient.chat).
            self._transport.close()
            raise TransientLLMError(
                f"GLM me cortó la respuesta a medias, patrón. ({error})") from error
        except ValueError as error:
            # JSONDecodeError o UnicodeDecodeError (cuerpo no-UTF-8): ambas ValueError.
            raise LLMError("GLM me devolvió basura no-JSON, patrón.") from error

        if _PROF:
            usage = raw.get("usage", {}) or {}
            print(
                f"[LLM-PROF/glm] {time.time() - t0:5.1f}s | "
                f"prompt={usage.get('prompt_tokens', '?')} "
                f"gen={usage.get('completion_tokens', '?')} | "
                f"tools_enviadas={len(tools) if tools else 0}",
                flush=True,
            )

        choices = raw.get("choices") or []
        message = (choices[0].get("message", {}) or {}) if choices else {}
        normalized = from_openai_message(message)
        return ChatResponse(
            content=(normalized.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(normalized),
            raw_message=normalized,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token=None,
    ) -> ChatResponse:
        """Como chat() pero consumiendo el SSE de OpenAI (`data: {...}` por línea).

        Cierra el ciclo de la respuesta: al llegar el fin lógico avisa al
        transporte (`finish`) para reutilizar el keep-alive; ante un corte, un
        fallo o un consumidor que aborta, cuelga el socket para que la nube
        deje de generar para nadie.
        """
        resp = self._send(self._payload(messages, tools, stream=True))
        try:
            content, message, complete = parse_sse(resp, on_token)
        except (TimeoutError, OSError) as error:
            self._transport.close()
            raise TransientLLMError(
                f"GLM no respondió a tiempo, patrón. ({error})") from error
        except (http.client.HTTPException, SSETruncated) as error:
            # Stream cortado: a medio chunk (IncompleteRead, ver chat()) o entre
            # dos chunks sin dejar nada usable (ver parse_sse).
            self._transport.close()
            raise TransientLLMError(
                f"GLM me cortó la respuesta a medias, patrón. ({error})") from error
        except SSEError as error:
            # Transitorio: el 200 ya salió (la petición y la key pasaron), así que
            # lo que falla es el servicio: límite de ritmo, sobrecarga, inferencia.
            self._transport.close()
            raise TransientLLMError(
                f"GLM falló a media respuesta, patrón: {error}") from error
        except BaseException:
            # El consumidor abortó (on_token lanzó): colgar YA, o la nube sigue
            # generando y el socket queda a medias hasta otro turno.
            self._transport.close()
            raise
        if complete:
            self._transport.finish(resp)  # la cola se drena en la próxima petición
        else:
            # Final limpio sin la marca: se acepta (ver parse_sse), pero queda
            # dicho en el log. Esa conexión no se reutiliza: lo normal es que
            # la cortara el servidor o un proxy, y reusarla solo costaría una
            # escritura fallida más el reintento.
            log.warning(
                "GLM cerró el stream sin [DONE] ni finish_reason: respuesta "
                "posiblemente truncada (%d caracteres, %d tool_calls)",
                len(content), len(message.get("tool_calls") or []),
            )
            self._transport.close()

        return ChatResponse(
            content=content.strip(),
            tool_calls=_parse_tool_calls(message),
            raw_message=message,
        )
