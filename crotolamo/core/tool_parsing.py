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


def arguments_complete(args: Any) -> bool:
    """¿Llegaron ENTEROS los `arguments` de un tool-call? Un dict, o un texto
    que es un JSON completo.

    Solo importa en un stream que acabó sin su cierre ([DONE] / done:true):
    ahí un texto vacío o a medias no se puede degradar a {} como hace
    `parse_arguments`, porque la tool correría con sus defaults en vez de con
    lo que pidió el modelo.
    """
    if isinstance(args, dict):
        return True
    if isinstance(args, str):
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return False
        return True
    return False


_JSON_OPEN = re.compile(r"[{\[]")
_DECODER = json.JSONDecoder()


def _json_documents(text: str) -> list[Any]:
    """Los documentos JSON que hay en `text`, uno tras otro y en orden.

    Un modelo que pide dos tools las escribe como dos objetos seguidos: uno por
    línea, o cada uno entre <tool_call></tool_call> (el formato de qwen). Eso no
    es UN documento, y un `json.loads` del primer `{` al último `}` lo rechazaba
    entero: no corría ninguna y al patrón le llegaba el JSON crudo.

    Todo o nada: si un `{`/`[` no abre JSON válido, el texto no es una lista
    limpia de llamadas (prosa con llaves, un objeto cortado a medias) y se
    devuelve []. Nada de pescar el primer trozo que parsee dentro de otra cosa.
    """
    docs: list[Any] = []
    pos = 0
    while (opening := _JSON_OPEN.search(text, pos)) is not None:
        try:
            doc, pos = _DECODER.raw_decode(text, opening.start())
        except json.JSONDecodeError:
            return []
        docs.append(doc)
    return docs


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

    # Del primer { o [ al último } o ]: un bloque, o varios seguidos.
    start = min((text.find(c) for c in "{[" if c in text), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        return []

    candidates: list[Any] = []
    for doc in _json_documents(text[start:end + 1]):
        candidates.extend(doc if isinstance(doc, list) else [doc])

    calls: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        # "parameters" es como los nombra el formato de texto de Llama 3.x (el
        # mismo que `strip_leaked_tool_json` ya trata como tool-call): sin
        # leerlo, la llamada se ejecutaba con {} y la tool corría con defaults.
        args = item.get("arguments", item.get("parameters", item.get("args", {})))
        if isinstance(args, str):
            # Contrato OpenAI: `arguments` como string JSON ("" = sin argumentos).
            # Solo vale si trae un objeto. Un string ilegible NO se degrada a {}
            # como hace `parse_arguments` con el campo nativo: aquí nada asegura
            # que esto sea una llamada, y la tool correría con sus defaults.
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                continue
        # isinstance(name, str): un JSON de DATOS puede traer un dict o una lista
        # bajo "name"/"tool", y buscarlo en el set lanzaba TypeError (unhashable)
        # que salía cruda de handle_turn.
        if isinstance(name, str) and name in known_names and isinstance(args, dict):
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
