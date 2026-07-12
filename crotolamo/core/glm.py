"""Cliente GLM (Z.ai / Zhipu) con tool-calling, hablando el contrato OpenAI.

POR QUÉ EXISTE: en esta lap (12 núcleos, 15 GiB, GPU integrada) inferir un 7B en
CPU cuesta ~17-22s por turno con tool, y deja la RAM en swap. GLM-4.7-Flash es
gratuito en Z.ai y responde en ~1-2s, sacando la inferencia de la máquina.

DISEÑO: esto es un ADAPTADOR. Traduce el formato de mensajes de Ollama al de
OpenAI y devuelve un `ChatResponse` idéntico al de `LLMClient`, con las
tool_calls ya normalizadas al formato Ollama. Así `Conversation`, `ToolAgent` y
las tools no se enteran de con qué motor hablan.

Las dos incompatibilidades reales del contrato:
  1. OpenAI exige `tool_call_id` en los mensajes de rol "tool" (Ollama usa `name`).
  2. OpenAI manda `function.arguments` como string JSON (Ollama, como dict).

Cero dependencias: stdlib, igual que `llm.py`.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from crotolamo.core.llm import ChatResponse, LLMError, _parse_tool_calls

_PROF = os.environ.get("CROTOLAMO_LLM_PROF") == "1"

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.7-flash"  # gratuito para usuarios registrados
# Variables de entorno donde se busca la API key, en orden. NUNCA se guarda la
# key en el toml (que va a git); la env es el sitio correcto.
API_KEY_ENVS = ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY")


class GLMAuthError(LLMError):
    """Falta la API key o el servidor la rechazó."""


def _find_api_key() -> str | None:
    for env in API_KEY_ENVS:
        key = os.environ.get(env)
        if key and key.strip():
            return key.strip()
    return None


def _call_id(index: int) -> str:
    """ID determinista para correlacionar un tool_call con su resultado.

    El historial de `Conversation` no guarda ids (Ollama no los usa), así que los
    reconstruimos por posición al traducir. Es correcto porque los resultados de
    tools se añaden SIEMPRE en el mismo orden en que se pidieron las llamadas
    (ver `ToolAgent.handle_turn`).
    """
    return f"call_{index}"


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Traduce el historial estilo Ollama al contrato de OpenAI.

    - assistant con tool_calls: añade `id` y `type`, y serializa `arguments` a str.
    - tool: sustituye `name` por el `tool_call_id` de la llamada correspondiente.

    Los ids se asignan por orden dentro de cada bloque assistant→tools, que es el
    orden en que `ToolAgent` los produce.
    """
    out: list[dict[str, Any]] = []
    pending: list[str] = []  # ids de las tool_calls del último assistant
    counter = 0

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            pending = []
            calls_out = []
            for call in msg["tool_calls"]:
                fn = call.get("function", {}) or {}
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                cid = call.get("id") or _call_id(counter)
                counter += 1
                pending.append(cid)
                calls_out.append({
                    "id": cid,
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            # OpenAI acepta content vacío cuando hay tool_calls, pero exige la clave.
            out.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls_out,
            })
            continue

        if role == "tool":
            if not pending:
                # `tool` sin un assistant-con-tool_calls que lo preceda EN ESTE
                # payload. Hoy `Conversation._trim()` recorta por bloques completos
                # y esto no ocurre, pero si ocurriera emitiríamos un tool_call_id
                # colgando y OpenAI/GLM devolvería un 400 opaco. Lo descartamos: no
                # dependemos de una invariante que mantiene otro módulo.
                continue
            # Consumimos los ids en el mismo orden en que se pidieron las llamadas.
            out.append({
                "role": "tool",
                "tool_call_id": pending.pop(0),
                "content": msg.get("content", ""),
            })
            continue

        # system / user / assistant sin tools: pasan tal cual (mismo contrato).
        out.append({"role": role, "content": msg.get("content", "")})

    return out


def _from_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normaliza el mensaje de OpenAI al formato Ollama que espera `ToolAgent`.

    `ToolAgent` reinyecta `response.raw_message["tool_calls"]` al historial tal
    cual, y `Conversation` lo vuelve a pasar por `to_openai_messages`. Guardarlo
    en formato Ollama (arguments como dict) mantiene un único formato canónico
    en memoria, independientemente del motor.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {}) or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append({
            "id": call.get("id"),
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    out: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if calls:
        out["tool_calls"] = calls
    return out


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
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

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

    def _request(self, payload: dict[str, Any]):
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def _raise_http(self, error: urllib.error.HTTPError) -> None:
        body = ""
        try:
            body = error.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - el cuerpo del error es best-effort
            pass
        if error.code in (401, 403):
            raise GLMAuthError(
                f"GLM me rechazó la credencial ({error.code}), patrón. "
                f"Revisa la API key. {body}"
            ) from error
        if error.code == 429:
            raise LLMError(
                "GLM me está limitando el ritmo (429), patrón. Aguanta tantito."
            ) from error
        raise LLMError(f"GLM respondió {error.code}, patrón: {body}") from error

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        req = self._request(self._payload(messages, tools, stream=False))

        t0 = time.time() if _PROF else 0.0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            self._raise_http(error)
        except urllib.error.URLError as error:
            raise LLMError(
                f"No pude hablar con GLM en {self.base_url}, patrón. "
                f"¿Hay internet? ({error.reason})"
            ) from error
        except (TimeoutError, OSError) as error:
            raise LLMError(f"GLM no respondió a tiempo, patrón. ({error})") from error
        except json.JSONDecodeError as error:
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
        normalized = _from_openai_message(message)
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
        """Como chat() pero consumiendo el SSE de OpenAI (`data: {...}` por línea)."""
        req = self._request(self._payload(messages, tools, stream=True))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content, message = self._consume_sse(resp, on_token)
        except urllib.error.HTTPError as error:
            self._raise_http(error)
        except urllib.error.URLError as error:
            raise LLMError(
                f"No pude hablar con GLM en {self.base_url}, patrón. ({error.reason})"
            ) from error
        except (TimeoutError, OSError) as error:
            raise LLMError(f"GLM no respondió a tiempo, patrón. ({error})") from error

        return ChatResponse(
            content=content.strip(),
            tool_calls=_parse_tool_calls(message),
            raw_message=message,
        )

    @staticmethod
    def _consume_sse(resp, on_token) -> tuple[str, dict[str, Any]]:
        """Parsea el Server-Sent Events de /chat/completions?stream=true.

        A diferencia del JSONL de Ollama, aquí los tool_calls llegan FRAGMENTADOS
        entre deltas: cada delta trae un trozo de `arguments` y un `index` que dice
        a qué llamada pertenece. Hay que reensamblarlos por índice.
        """
        parts: list[str] = []
        # index -> {"id": str, "name": str, "args": [fragmentos]}
        acc: dict[int, dict[str, Any]] = {}

        for raw_line in resp:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}

            text = delta.get("content") or ""
            if text:
                parts.append(text)
                if on_token:
                    on_token(text)

            for call in delta.get("tool_calls") or []:
                idx = call.get("index", 0)
                slot = acc.setdefault(idx, {"id": None, "name": "", "args": []})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function", {}) or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"].append(fn["arguments"])

        message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
        if acc:
            calls = []
            for idx in sorted(acc):
                slot = acc[idx]
                joined = "".join(slot["args"])
                try:
                    args = json.loads(joined) if joined.strip() else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id": slot["id"],
                    "function": {"name": slot["name"], "arguments": args},
                })
            message["tool_calls"] = calls

        return "".join(parts), message
