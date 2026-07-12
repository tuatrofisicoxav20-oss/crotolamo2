"""Cliente de Ollama con tool-calling. Reescritura de C1::ask_ollama.

Habla /api/chat por HTTP (stdlib, sin dependencias) sobre una conexión
persistente (`HTTPTransport`, keep-alive). Soporta el campo `tools` para
tool-calling nativo de qwen2.5-coder. Los errores se devuelven en personaje.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.tool_parsing import parse_arguments

# Instrumentación opcional: con CROTOLAMO_LLM_PROF=1 imprime tiempo y tokens por
# llamada a /api/chat (prompt_eval_count, eval_count). Útil para diagnosticar el
# costo de los tool-schemas en CPU. Cero overhead si la env no está puesta.
_PROF = os.environ.get("CROTOLAMO_LLM_PROF") == "1"


class LLMError(RuntimeError):
    """Error hablando con Ollama, ya con mensaje en personaje."""


@dataclass
class ChatResponse:
    content: str = ""
    # Lista de tool calls pedidos: [{"name": str, "arguments": dict}, ...]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # El mensaje crudo del asistente (para reinyectar al historial tal cual).
    raw_message: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        # Ollama suele mandar arguments como dict; algunos modelos lo mandan
        # como string JSON. Normalizamos a dict.
        args = parse_arguments(fn.get("arguments", {}))
        calls.append({"name": fn.get("name", ""), "arguments": args})
    return calls


def _read_body(resp) -> str:
    """Cuerpo de una respuesta de error, best-effort y acotado."""
    try:
        return resp.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001 - el cuerpo del error es best-effort
        return ""


class LLMClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.2,
        timeout: float = 120,
        keep_alive: str = "15m",
        num_ctx: int | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # num_ctx: ventana de contexto que Ollama reserva. En CPU, una ventana
        # más chica reduce memoria y trabajo del KV-cache; con el routing de
        # tools (prompts cortos) 2048 sobra. None = default del modelo.
        self.num_ctx = num_ctx
        # keep_alive: cuánto mantiene Ollama el modelo (y su cache de prefijo KV)
        # residente tras una respuesta. En CPU es CLAVE: el primer turno paga el
        # prompt-eval completo de los tool-schemas (~lento), pero si el modelo
        # sigue caliente los turnos siguientes reusan el cache y son baratos.
        # "15m" balancea rapidez en sesión vs. liberar RAM cuando no se usa.
        self.keep_alive = keep_alive
        # Conexión HTTP persistente: evita abrir TCP nuevo en cada petición.
        self._transport = HTTPTransport(self.host, timeout=self.timeout)

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        llm = settings.llm
        return cls(
            host=llm.get("host", "http://localhost:11434"),
            model=llm.get("model", "qwen2.5-coder:7b"),
            temperature=llm.get("temperature", 0.2),
            timeout=llm.get("timeout", 120),
            keep_alive=llm.get("keep_alive", "15m"),
            num_ctx=llm.get("num_ctx"),
        )

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        return options

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _raise_http(self, status: int, body: str) -> NoReturn:
        """Mapea un status HTTP >= 400 de Ollama a un LLMError útil.

        Antes un 404/500 caía en la rama genérica de "¿Está vivo el servicio?",
        que despista: el servicio SÍ está vivo, es la petición la que falló.
        """
        if status == 404:
            raise LLMError(
                f"Ollama respondió 404, patrón. ¿Existe el modelo '{self.model}'? "
                f"Prueba `ollama pull {self.model}`. {body}".rstrip()
            )
        if status >= 500:
            raise LLMError(
                f"Ollama tropezó por dentro ({status}), patrón: {body}"
            )
        raise LLMError(f"Ollama respondió {status}, patrón: {body}")

    def _send(self, payload: dict[str, Any]) -> http.client.HTTPResponse:
        """POST a /api/chat con los errores de transporte ya en personaje."""
        try:
            resp = self._transport.post_json("/api/chat", payload)
        except TimeoutError as error:
            raise LLMError(
                f"Ollama no respondió a tiempo, patrón. ({error})"
            ) from error
        except (http.client.HTTPException, OSError) as error:
            raise LLMError(
                f"No pude hablar con Ollama en {self.host}, patrón. "
                f"¿Está vivo el servicio? ({error})"
            ) from error
        if resp.status >= 400:
            self._raise_http(resp.status, _read_body(resp))
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
            raise LLMError(
                f"Ollama no respondió a tiempo, patrón. ({error})"
            ) from error
        except json.JSONDecodeError as error:
            raise LLMError("Ollama me devolvió basura no-JSON, patrón.") from error

        if _PROF:
            n_tools = len(tools) if tools else 0
            print(
                f"[LLM-PROF] {time.time() - t0:5.1f}s | "
                f"prompt_eval={raw.get('prompt_eval_count', '?')} "
                f"gen={raw.get('eval_count', '?')} | tools_enviadas={n_tools}",
                flush=True,
            )

        message = raw.get("message", {}) or {}
        return ChatResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message),
            raw_message=message,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token=None,
    ) -> ChatResponse:
        """Como chat() pero con stream=True: invoca on_token(chunk) por cada delta.

        Acumula el contenido y captura tool_calls del último mensaje. Útil para
        respuesta token-a-token (Fase 6). Funciona mejor con GPU o modelos que usan
        el campo nativo tool_calls.
        """
        resp = self._send(self._payload(messages, tools, stream=True))
        try:
            content, last_message = self._consume_stream(resp, on_token)
        except (TimeoutError, OSError) as error:
            raise LLMError(f"Ollama no respondió a tiempo, patrón. ({error})") from error

        return ChatResponse(
            content=content.strip(),
            tool_calls=_parse_tool_calls(last_message),
            raw_message=last_message,
        )

    @staticmethod
    def _consume_stream(resp, on_token) -> tuple[str, dict[str, Any]]:
        """Parsea el JSONL de /api/chat?stream=true. Reusable y testeable."""
        parts: list[str] = []
        last_message: dict[str, Any] = {}
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message", {}) or {}
            if msg:
                last_message = msg
            delta = msg.get("content") or ""
            if delta:
                parts.append(delta)
                if on_token:
                    on_token(delta)
            if obj.get("done"):
                break
        return "".join(parts), last_message
