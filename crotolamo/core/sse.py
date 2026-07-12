"""Parseo de Server-Sent Events del endpoint /chat/completions?stream=true.

Extraído de `glm.py`: es lógica pura de parseo, testeable sin red. `resp` es
cualquier iterable de líneas (bytes o str): un `http.client.HTTPResponse` en
producción, una lista en los tests.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from crotolamo.core.tool_parsing import parse_arguments


def consume_sse(
    resp: Any,
    on_token: Callable[[str], None] | None,
) -> tuple[str, dict[str, Any]]:
    """Parsea el SSE de OpenAI (`data: {...}` por línea) hasta `[DONE]`.

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
            calls.append({
                "id": slot["id"],
                "function": {
                    "name": slot["name"],
                    "arguments": parse_arguments("".join(slot["args"])),
                },
            })
        message["tool_calls"] = calls

    return "".join(parts), message
