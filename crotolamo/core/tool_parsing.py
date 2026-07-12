"""Parsing compartido de tool-calls, independiente del motor.

Aquí vive lo que antes estaba repartido entre `agent.py`, `llm.py` y `glm.py`:
la coerción de tool-calls emitidos como texto, la detección de fallos duros de
la capa de ejecución y la normalización de `arguments` a dict.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_arguments(args: Any) -> dict[str, Any]:
    """Normaliza el campo `arguments` de un tool-call a dict.

    Ollama suele mandarlo como dict; OpenAI/GLM lo mandan como string JSON, y
    algunos modelos emiten basura. Cualquier cosa que no acabe siendo un dict
    (JSON roto, un array, None) se degrada a {} en vez de reventar el turno.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def coerce_text_tool_calls(content: str, known_names: set[str]) -> list[dict[str, Any]]:
    """Fallback: algunos modelos (qwen2.5-coder en Ollama) emiten el tool-call
    como JSON dentro de `content` en vez de en el campo nativo `tool_calls`.

    Parseamos ese texto y, solo si referencia una tool conocida, lo tratamos
    como llamada. Devuelve [] si el contenido es texto conversacional normal.
    """
    if not content or "{" not in content:
        return []

    # Quitar cercas de código ```json ... ``` si las hay.
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Intentar el bloque {...} o [...] más externo.
    start = min((text.find(c) for c in "{[" if c in text), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        return []

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    candidates = parsed if isinstance(parsed, list) else [parsed]
    calls: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        args = item.get("arguments", item.get("args", {}))
        if name in known_names and isinstance(args, dict):
            calls.append({"name": name, "arguments": args})
    return calls


# Prefijos con los que la capa de ejecución (Registry.run) marca un fallo DURO de
# la tool (excepción no controlada o argumentos inválidos). En esos casos NO se
# hace short-circuit: dejamos que el modelo reaccione. Los "soft-errors" en
# personaje de las propias tools ("No pude leer /proc/meminfo, patrón.") SÍ son
# texto listo para el patrón, así que esos sí se devuelven directos.
HARD_ERROR_PREFIXES: tuple[str, ...] = (
    "La tool '",            # "...reventó, patrón: ..."
    "Argumentos inválidos para '",
    "No tengo una tool ",   # nombre desconocido (defensivo; no debería pasar aquí)
)


def is_hard_error(result: str) -> bool:
    """True si el resultado de una tool es un fallo duro (no apto para short-circuit)."""
    if not result or not result.strip():
        return True
    return result.lstrip().startswith(HARD_ERROR_PREFIXES)
