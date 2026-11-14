"""Parseo de Server-Sent Events del endpoint /chat/completions?stream=true.

Extraído de `glm.py`: es lógica pura de parseo, testeable sin red. `resp` es
cualquier iterable de líneas (bytes o str): un `http.client.HTTPResponse` en
producción, una lista en los tests.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from crotolamo.core.tool_parsing import arguments_complete, parse_arguments


class SSEError(Exception):
    """El servidor mandó un objeto `error` DENTRO del stream, ya con el 200 dado."""


class SSETruncated(Exception):
    """El stream se acabó sin `[DONE]` ni `finish_reason` y sin nada usable."""


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
    """Como `parse_sse`, sin el aviso de fin lógico (para tests y llamadores viejos)."""
    content, message, _complete = parse_sse(resp, on_token)
    return content, message


def parse_sse(
    resp: Any,
    on_token: Callable[[str], None] | None,
    usage: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], bool]:
    """Parsea el SSE de OpenAI (`data: {...}` por línea) hasta `[DONE]`.

    A diferencia del JSONL de Ollama, aquí los tool_calls llegan FRAGMENTADOS
    entre deltas: cada delta trae un trozo de `arguments` y un `index` que dice
    a qué llamada pertenece. Hay que reensamblarlos por índice.

    El tercer valor dice si llegó el fin LÓGICO del stream (`[DONE]` o un
    `finish_reason`): con él, el cliente le avisa al transporte que el socket
    se puede reutilizar.

    Lanza `SSEError` si el servidor avisa de un fallo dentro del stream. Si el
    stream se acaba sin su cierre, lanza `SSETruncated` cuando no deja nada
    usable: ni texto ni tool_calls, o un tool_call cuyos argumentos no llegaron
    enteros (degradados a {} se ejecutaba con sus defaults). Con algo usable
    devuelve la respuesta y False: un final limpio sin la marca es un cierre
    del servidor, y tratarlo como error rompería TODAS las respuestas de un
    proveedor que no la mande. El cliente lo avisa en el log. Un corte de red
    a medio chunk no llega aquí: http.client lanza IncompleteRead.

    Si on_token trae `checkpoint` (LiveStreamer), se consulta con CADA línea
    leída y lo que lance (TurnCancelled) sale tal cual: un tool-call o un
    razonamiento no traen texto, no llegan a on_token y el turno cancelado
    esperaba el stream entero.

    `usage`: si se pasa un dict, se llena con el objeto `usage` del stream si
    algún chunk lo trae (suele ir en el último, a veces con `choices` vacío).
    Solo para el log del cliente; lo que se devuelve no cambia.
    """
    checkpoint = getattr(on_token, "checkpoint", None)
    parts: list[str] = []
    # index -> {"id": str, "name": str, "args": [fragmentos]}
    acc: dict[int, dict[str, Any]] = {}
    # ¿Llegó el cierre? Vale `[DONE]` o un `finish_reason` (hay servidores que
    # cierran tras ese chunk sin mandar `[DONE]`). Que el HTTP no se queje no
    # basta: un FIN entre dos chunks acaba la iteración por líneas sin error.
    complete = False

    for raw_line in resp:
        if checkpoint is not None:
            checkpoint()
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
        if usage is not None and isinstance(obj.get("usage"), dict):
            usage.update(obj["usage"])

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

    content = "".join(parts)
    # Sin cierre vale lo que llegó si es usable (el cliente avisa del posible
    # truncado), pero jamás un tool_call con los argumentos a medio llegar.
    if not complete and (
        not (content.strip() or acc)
        or not all(arguments_complete("".join(slot["args"])) for slot in acc.values())
    ):
        raise SSETruncated("el stream acabó sin [DONE]")

    message: dict[str, Any] = {"role": "assistant", "content": content}
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

    return content, message, complete
