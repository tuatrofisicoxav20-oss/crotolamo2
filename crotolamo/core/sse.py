"""Parseo de Server-Sent Events del endpoint /chat/completions?stream=true.

Extraído de `glm.py`: es lógica pura de parseo, testeable sin red. `resp` es
cualquier iterable de líneas (bytes o str): un `http.client.HTTPResponse` en
producción, una lista en los tests.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from crotolamo.core.tool_parsing import parse_arguments


class SSEError(Exception):
    """El servidor mandó un objeto `error` DENTRO del stream, ya con el 200 dado."""


class SSETruncated(Exception):
    """El stream se acabó sin `[DONE]` ni `finish_reason`: lo cortaron a medias."""


def _error_text(error: Any) -> str:
    """Lo legible de un objeto `error`: su mensaje (y código), acotado."""
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error, ensure_ascii=False)
        code = error.get("code")
        error = f"{message} ({code})" if code else message
    return str(error)[:300]


def consume_sse(
    resp: Any,
    on_token: Callable[[str], None] | None,
) -> tuple[str, dict[str, Any]]:
    """Parsea el SSE de OpenAI (`data: {...}` por línea) hasta `[DONE]`.

    A diferencia del JSONL de Ollama, aquí los tool_calls llegan FRAGMENTADOS
    entre deltas: cada delta trae un trozo de `arguments` y un `index` que dice
    a qué llamada pertenece. Hay que reensamblarlos por índice.

    Lanza `SSEError` si el servidor avisa de un fallo dentro del stream y
    `SSETruncated` si el stream se acaba sin su cierre. Antes las dos cosas
    pasaban por una respuesta buena: vacía, cortada a media frase, o con un
    tool_call cuyos argumentos a medio llegar se degradaban a {} y se ejecutaba.
    El cliente las traduce a su error en personaje.
    """
    parts: list[str] = []
    # index -> {"id": str, "name": str, "args": [fragmentos]}
    acc: dict[int, dict[str, Any]] = {}
    # ¿Llegó el cierre? Vale `[DONE]` o un `finish_reason` (hay servidores que
    # cierran tras ese chunk sin mandar `[DONE]`). Que el HTTP no se queje no
    # basta: un FIN entre dos chunks acaba la iteración por líneas sin error.
    complete = False

    for raw_line in resp:
        # "replace": una línea que ni es UTF-8 (el HTML en latin-1 de un portal
        # cautivo) es basura como cualquier otra y la descartan los filtros de
        # abajo; con el codec estricto reventaba aquí con un UnicodeDecodeError.
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            complete = True
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue  # JSON válido que no es un evento (null, una lista): basura
        if obj.get("error"):
            raise SSEError(_error_text(obj["error"]))

        choices = obj.get("choices") or []
        if not choices:
            continue
        if choices[0].get("finish_reason"):
            complete = True
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

    if not complete:
        raise SSETruncated("el stream acabó sin [DONE]")

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
