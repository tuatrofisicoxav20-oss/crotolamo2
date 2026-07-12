"""Traducción del contrato de mensajes Ollama <-> OpenAI.

Extraído de `glm.py`: es lógica pura de formato, sin red. Las dos
incompatibilidades reales del contrato:
  1. OpenAI exige `tool_call_id` en los mensajes de rol "tool" (Ollama usa `name`).
  2. OpenAI manda `function.arguments` como string JSON (Ollama, como dict).
"""

from __future__ import annotations

import json
from typing import Any

from crotolamo.core.tool_parsing import parse_arguments


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


def from_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normaliza el mensaje de OpenAI al formato Ollama que espera `ToolAgent`.

    `ToolAgent` reinyecta `response.raw_message["tool_calls"]` al historial tal
    cual, y `Conversation` lo vuelve a pasar por `to_openai_messages`. Guardarlo
    en formato Ollama (arguments como dict) mantiene un único formato canónico
    en memoria, independientemente del motor.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {}) or {}
        calls.append({
            "id": call.get("id"),
            "function": {
                "name": fn.get("name", ""),
                "arguments": parse_arguments(fn.get("arguments", {})),
            },
        })
    out: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if calls:
        out["tool_calls"] = calls
    return out
