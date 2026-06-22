"""Hooks pre/post LLM (M4, inspirado en los pipelines de Open WebUI).

Factories que producen funciones str->str para enchufar en ToolAgent.pre_hooks /
post_hooks. Mínimo y en personaje; NO es el motor entero de Open WebUI.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

# Preámbulos meta que el 3B a veces antepone ("parece que la herramienta 'x' no
# acepta parámetros. Te resumo el resultado: ..."). Patrones ANCLADOS al inicio,
# conservadores, para no tocar respuestas normales. Cada patrón consume su frase
# y la puntuación/espacio que la sigue.
_META_PREAMBLES: tuple[re.Pattern[str], ...] = (
    # "(Patrón, )parece que la herramienta '...' no acepta (ningún )parámetro(s)."
    re.compile(
        r"^\s*(?:patrón[,\s]+)?(?:parece que )?la herramienta\b[^.!?\n]*?"
        r"no acepta[^.!?\n]*?par[áa]metros?[.!?\s]*",
        re.IGNORECASE,
    ),
    # "Te resumo el resultado:" / "Aquí (te) (va )el resultado:"
    re.compile(
        r"^\s*(?:te\s+)?resumo el resultado\s*[:.\-]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*aqu[íi](?:\s+te)?(?:\s+va)?\s+el resultado\s*[:.\-]?\s*",
        re.IGNORECASE,
    ),
)


def meta_preamble_cleaner(reply: str) -> str:
    """Post-hook: quita preámbulos meta al INICIO de la respuesta del modelo.

    Conservador: solo recorta patrones evidentes anclados al comienzo y, si tras
    recortar no queda nada útil, devuelve la respuesta original (mejor algo de
    ruido que una respuesta vacía).
    """
    if not reply:
        return reply
    cleaned = reply
    # Aplicamos cada patrón una vez, en cadena (pueden venir encadenados:
    # "...no acepta parámetros. Te resumo el resultado: ...").
    for pattern in _META_PREAMBLES:
        new = pattern.sub("", cleaned, count=1)
        if new != cleaned:
            cleaned = new.lstrip()
    cleaned = cleaned.strip()
    return cleaned if cleaned else reply


def strip_leaked_tool_json(reply: str) -> str:
    """Post-hook: evita que un tool-call JSON CRUDO se filtre como respuesta final.

    El 3B a veces, en la 2ª llamada, en vez de redactar intenta re-llamar una tool
    y emite algo como `{"name": "...", "parameters": {}}` que se cuela tal cual al
    patrón. Detectamos esa forma (JSON con clave 'name' y 'parameters'/'arguments',
    o lista de esos) y la sustituimos por un mensaje en personaje, en vez de
    mostrar basura. Conservador: SOLO dispara si TODA la respuesta es ese JSON; el
    texto normal (aunque mencione llaves) no se toca.
    """
    if not reply:
        return reply
    stripped = reply.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return reply
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return reply  # no es JSON puro: es texto normal, no lo tocamos

    items = parsed if isinstance(parsed, list) else [parsed]
    looks_like_tool_call = any(
        isinstance(it, dict)
        and ("name" in it or "tool" in it)
        and ("parameters" in it or "arguments" in it or "args" in it)
        for it in items
    )
    if looks_like_tool_call:
        return ("Lo tengo, patrón, pero se me trabó la lengua al redactarlo. "
                "Pregúntamelo otra vez y te lo digo derecho.")
    return reply


def make_facts_prehook():
    """Pre-hook que inyecta los hechos recordados del patrón UNA sola vez (primer
    turno), sustituyendo la inyección manual en el system prompt (M4.2).
    """
    state = {"done": False}

    def hook(text: str) -> str:
        if state["done"]:
            return text
        state["done"] = True
        try:
            from crotolamo.persistence import facts

            ctx = facts.facts_context()
        except Exception:
            ctx = ""
        if ctx:
            return f"[contexto que ya sabes sobre el patrón:\n{ctx}]\n\n{text}"
        return text

    return hook


def datetime_prehook(text: str) -> str:
    """Pre-hook trivial de demostración: antepone la fecha/hora actual al contexto."""
    return f"[ahora: {datetime.now():%Y-%m-%d %H:%M}]\n{text}"
