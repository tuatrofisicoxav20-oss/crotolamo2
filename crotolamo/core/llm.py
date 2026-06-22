"""Cliente de Ollama con tool-calling. Reescritura de C1::ask_ollama.

Habla /api/chat por HTTP (stdlib, sin dependencias). Soporta el campo `tools`
para tool-calling nativo de qwen2.5-coder. Los errores se devuelven en personaje.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

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
        args = fn.get("arguments", {})
        # Ollama suele mandar arguments como dict; algunos modelos lo mandan
        # como string JSON. Normalizamos a dict.
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": fn.get("name", ""), "arguments": args or {}})
    return calls


class LLMClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.2,
        timeout: float = 120,
        keep_alive: str = "15m",
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # keep_alive: cuánto mantiene Ollama el modelo (y su cache de prefijo KV)
        # residente tras una respuesta. En CPU es CLAVE: el primer turno paga el
        # prompt-eval completo de los tool-schemas (~lento), pero si el modelo
        # sigue caliente los turnos siguientes reusan el cache y son baratos.
        # "15m" balancea rapidez en sesión vs. liberar RAM cuando no se usa.
        self.keep_alive = keep_alive

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        llm = settings.llm
        return cls(
            host=llm.get("host", "http://localhost:11434"),
            model=llm.get("model", "qwen2.5-coder:7b"),
            temperature=llm.get("temperature", 0.2),
            timeout=llm.get("timeout", 120),
            keep_alive=llm.get("keep_alive", "15m"),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time() if _PROF else 0.0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise LLMError(
                f"No pude hablar con Ollama en {self.host}, patrón. "
                f"¿Está vivo el servicio? ({error.reason})"
            ) from error
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
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content, last_message = self._consume_stream(resp, on_token)
        except urllib.error.URLError as error:
            raise LLMError(
                f"No pude hablar con Ollama en {self.host}, patrón. ({error.reason})"
            ) from error
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
