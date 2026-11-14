"""Presupuesto de tamaño de lo que viaja al LLM: el único dueño de sus unidades.

POR QUÉ existe: cada límite del camino tool -> request lo puso un autor distinto,
en su capa y en su unidad (READ_CAP en caracteres de una tool, max_turns en
turnos de usuario, num_ctx en tokens que nadie comprobaba), y ninguno sabía de
los demás ni del motor que iba a recibir el request. Aquí viven las unidades
(caracteres de un resultado de tool, bytes JSON, tokens estimados) y la única
operación que adapta una lista de mensajes a una ventana: `fit`.

Puro y sin red: no muta su entrada, y si el request ya cabe devuelve el MISMO
objeto (la llamada normal no paga copias ni cambia un byte).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from crotolamo.tools.base import truncate_for_context

# Tope de un resultado de tool tal como entra al historial (ToolAgent). Está por
# encima de todo lo que ya devuelven las tools con tope propio (read_file 20 000
# + marca, read_project_file + cabecera y ruta, read_page ~6 100 + URL, errores
# MCP ~20 100): a esas no les cambia un byte. Acota las que no tienen tope
# (find_in_project, recall_facts, list_project_tree, list_processes...).
TOOL_RESULT_CAP = 24_000

# Bytes de JSON por token, calibrado con el tokenizer real de Qwen2.5: 3.43 la
# persona, 3.64 los esquemas, 3.78 el código, 2.92 la charla en español. Se usa
# el lado bajo para que la estimación peque de grande y no de chica.
BYTES_PER_TOKEN = 3.3

# Tokens que se dejan libres para la respuesta del modelo.
REPLY_RESERVE = 256

# Suelo al recortar a mitades los resultados del bloque en curso: por debajo el
# modelo ya no tiene con qué contestar y sobra la llamada.
MIN_TOOL_CHARS = 300

# Ventana de la nube cuando la config no dice otra (red de seguridad, no dieta):
# por encima, el proveedor contesta un 400 permanente que tumbaba cada turno.
DEFAULT_CLOUD_CONTEXT_TOKENS = 128_000

# Parte de una ventana LOCAL que puede ocupar un solo resultado de tool, y su
# suelo. Con num_ctx=2048 la persona (~830 tokens) y 8 esquemas (~700) dejan
# ~500: un resultado entero de 20 000 caracteres no cabe ni solo, y sin tope el
# ajuste tiraría primero toda la historia para luego recortarlo igual.
LOCAL_TOOL_SHARE = 0.35
MIN_LOCAL_TOOL_CHARS = 1_200

# Lo que se queda de cada resultado de tool al mandar turnos viejos al resumidor
# de [memory].compaction en segundo plano: el resumen guarda hechos y decisiones,
# no el archivo que se leyó (un bloque con una lectura pesaba ~7 KB por llamada).
SUMMARY_TOOL_CHARS = 500


def _json_bytes(obj: Any) -> int:
    # Igual que serializa el transporte (engine.HTTPTransport): mismo tamaño.
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8", errors="replace"))


def _tokens(nbytes: int) -> int:
    return math.ceil(nbytes / BYTES_PER_TOKEN)


def est_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
) -> int:
    """Tokens estimados de un request: bytes JSON de mensajes + tools, / 3.3.

    Sin tools no se cuentan: los clientes omiten la clave "tools" si va vacía.
    """
    return _tokens(_json_bytes(messages) + (_json_bytes(tools) if tools else 0))


@dataclass(frozen=True)
class Fit:
    """Lo que hizo `fit`, para que el cliente lo pueda contar en el log."""

    messages: list[dict[str, Any]]
    est_before: int
    est_after: int
    dropped_blocks: int
    clipped_tools: int
    fits: bool
    # Huecos anclados tras la persona (hechos, resumen) que se recortaron.
    clipped_pinned: int = 0


def _content(msg: dict[str, Any]) -> str | None:
    content = msg.get("content")
    return content if isinstance(content, str) else None


def _is_tool(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "tool" and _content(msg) is not None


def _list_bytes(sizes_sum: int, count: int) -> int:
    # json.dumps de una lista: "[" + ", ".join(elementos) + "]". Llevar la suma
    # a mano evita reserializar todo el request cada vez que se tira un bloque.
    return 2 + sizes_sum + 2 * max(0, count - 1)


def _clip_slot(msg: dict[str, Any], max_bytes: int) -> dict[str, Any] | None:
    """Un hueco anclado (hechos, resumen) en <= `max_bytes` de JSON, o None si
    no tiene líneas que quitar.

    Se quedan su primera línea (dice qué es el bloque), una marca con lo que se
    quitó y las líneas del FINAL que quepan, enteras: facts_context pone el
    hecho más nuevo al final. Si ni la cabecera con la marca cabe, se queda eso.
    """
    lines = (_content(msg) or "").split("\n")
    header, body = lines[0], lines[1:]
    clipped = msg
    for keep in range(len(body) - 1, -1, -1):
        mark = f"...[recortado, {len(body) - keep} de {len(body)} líneas; quedan las más nuevas]"
        clipped = {**msg, "content": "\n".join([header, mark, *body[len(body) - keep:]])}
        if _json_bytes(clipped) <= max_bytes:
            break
    # Un hueco de una línea, o tan corto que la marca pesa más que lo quitado.
    return clipped if _json_bytes(clipped) < _json_bytes(msg) else None


def fit(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    tool_cap: int | None = None,
) -> Fit:
    """Adapta `messages` a una ventana de `max_tokens` (dejando REPLY_RESERVE).

    Si ya cabe y ningún 'tool' pasa de `tool_cap`, devuelve la MISMA lista. Si
    no, trabaja sobre una copia (los dicts que cambia son nuevos):
      1. recorta cada 'tool' a `tool_cap` con truncate_for_context (la marca de
         siempre, con el total original);
      2. conserva toda la cabeza de mensajes 'system' (persona, fijos, resumen)
         y el bloque en curso (del último 'user' en adelante);
      3. tira la historia vieja por bloques ENTEROS (un user con su cadena de
         assistant/tool), del más viejo al más nuevo, hasta que quepa: un 'tool'
         nunca se queda sin el assistant que lo pidió;
      4. si aun así no cabe, recorta a mitades los 'tool' del bloque en curso,
         siempre desde el original, hasta MIN_TOOL_CHARS;
      5. y si TAMPOCO, recorta los huecos anclados de la cabeza (del último al
         primero: el resumen antes que los hechos), dejando de cada uno su
         primera línea y las líneas más nuevas (_clip_slot). La persona, nunca.
    Si ni con eso cabe (la persona sola no entra), devuelve lo mejor posible.
    """
    budget = max_tokens - REPLY_RESERVE
    tools_bytes = _json_bytes(tools) if tools else 0
    est_before = _tokens(_json_bytes(messages) + tools_bytes)
    over_cap = tool_cap is not None and any(
        _is_tool(m) and len(_content(m) or "") > tool_cap for m in messages
    )
    if est_before <= budget and not over_cap:
        return Fit(messages, est_before, est_before, 0, 0, True)

    out = list(messages)
    clipped: set[int] = set()
    if tool_cap is not None:
        for i, msg in enumerate(out):
            text = _content(msg)
            if _is_tool(msg) and text is not None and len(text) > tool_cap:
                out[i] = {**msg, "content": truncate_for_context(text, tool_cap)}
                clipped.add(i)
    sizes = [_json_bytes(m) for m in out]

    head = 0
    while head < len(out) and out[head].get("role") == "system":
        head += 1
    users = [i for i in range(head, len(out)) if out[i].get("role") == "user"]
    current = users[-1] if users else head

    def total(start: int) -> int:
        # Cabeza + historia desde `start` (lo anterior, tirado) + tools.
        kept = sizes[:head] + sizes[start:]
        return _list_bytes(sum(kept), len(kept)) + tools_bytes

    # Inicios de bloque de la historia vieja (entre la cabeza y el bloque en
    # curso): cada user, y lo que hubiera antes del primero como un bloque más.
    # Tirar el bloque k es mover `start` al inicio del k+1.
    starts = sorted({head, *(i for i in users if i < current)}) if head < current else []
    start = head
    dropped = 0
    for nxt in starts[1:] + ([current] if starts else []):
        if _tokens(total(start)) <= budget:
            break
        start = nxt
        dropped += 1

    if _tokens(total(start)) > budget:
        live = [i for i in range(current, len(out)) if _is_tool(messages[i])]
        if live:
            cap = tool_cap if tool_cap is not None else max(
                len(_content(messages[i]) or "") for i in live
            )
            while _tokens(total(start)) > budget and cap > MIN_TOOL_CHARS:
                cap = max(MIN_TOOL_CHARS, cap // 2)
                for i in live:
                    text = _content(messages[i]) or ""
                    if len(text) > cap:
                        out[i] = {**messages[i], "content": truncate_for_context(text, cap)}
                        sizes[i] = _json_bytes(out[i])
                        clipped.add(i)

    # Paso 5. Con [memory].pin_facts los hechos del patrón viajan en la cabeza;
    # intocables, con ~8 hechos el request local pasaba de num_ctx aun sin
    # historia, y Ollama cortaba el PRINCIPIO del prompt: la persona.
    pinned = 0
    limit = math.floor(budget * BYTES_PER_TOKEN)  # los bytes que caben en `budget`
    for i in range(head - 1, 0, -1):
        excess = total(start) - limit
        if excess <= 0:
            break
        slot = _clip_slot(out[i], sizes[i] - excess) if _content(out[i]) else None
        if slot is not None:
            out[i] = slot
            sizes[i] = _json_bytes(slot)
            pinned += 1

    result = out[:head] + out[start:]
    est_after = _tokens(total(start))
    return Fit(
        result,
        est_before,
        est_after,
        dropped,
        sum(1 for i in clipped if i < head or i >= start),
        est_after <= budget,
        pinned,
    )


def fit_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    tool_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Los mensajes de `fit` (ver allí el contrato)."""
    return fit(messages, tools, max_tokens, tool_cap).messages


def local_tool_cap(num_ctx: int) -> int:
    """Tope de un resultado de tool en una ventana local de `num_ctx` tokens.

    El 35 % de la ventana en caracteres (2 365 con 2048), nunca menos de 1 200.
    """
    return max(MIN_LOCAL_TOOL_CHARS, int(LOCAL_TOOL_SHARE * num_ctx * BYTES_PER_TOKEN))


def fit_window(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    tool_cap: int | None = None,
) -> Fit:
    """Lo que hace un cliente con el request antes de mandarlo a SU motor.

    Si el request cabe en la ventana sale el MISMO objeto, aunque algún 'tool'
    pase de `tool_cap`: lo que el motor ya aceptaba entero no se toca. Solo si
    no cabe se ajusta con `fit`, y entonces sí con el tope de tool: recortar
    resultados viejos antes de tirar turnos enteros deja más conversación.
    """
    est = est_tokens(messages, tools)
    if est <= max_tokens - REPLY_RESERVE:
        return Fit(messages, est, est, 0, 0, True)
    return fit(messages, tools, max_tokens, tool_cap)
