"""Hooks pre/post LLM (M4, inspirado en los pipelines de Open WebUI).

Factories que producen funciones str->str para enchufar en ToolAgent.pre_hooks /
post_hooks. Mínimo y en personaje; NO es el motor entero de Open WebUI.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from crotolamo.logging_setup import get_logger

log = get_logger("core.hooks")

# Preámbulos meta que el 3B a veces antepone ("parece que la herramienta 'x' no
# acepta parámetros. Te resumo el resultado: ..."). Patrones ANCLADOS al inicio,
# conservadores, para no tocar respuestas normales. Cada patrón consume su frase
# y la puntuación/espacio que la sigue.
# El separador tras la frase es OBLIGATORIO: opcional, el patrón casaba con el
# principio de una frase normal y se la comía ("Aquí el resultado final fue 3 a
# 1" -> "final fue 3 a 1"; "...no acepta parámetros negativos. Dime..." ->
# "negativos. Dime...").
_META_PREAMBLES: tuple[re.Pattern[str], ...] = (
    # "(Patrón, )parece que la herramienta '...' no acepta (ningún )parámetro(s)."
    # La frase acaba en "parámetro(s)": con su puntuación o con el salto de línea.
    re.compile(
        r"^\s*(?:patrón[,\s]+)?(?:parece que )?la herramienta\b[^.!?\n]*?"
        r"no acepta[^.!?\n]*?par[áa]metros?(?:\s*[.!?]+|[ \t]*\n)\s*",
        re.IGNORECASE,
    ),
    # "Te resumo el resultado:" / "Aquí (te) (va )el resultado:"
    re.compile(
        r"^\s*(?:te\s+)?resumo el resultado\s*[:.\-]\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*aqu[íi](?:\s+te)?(?:\s+va)?\s+el resultado\s*[:.\-]\s*",
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


def _es_tool_call_json(text: str) -> bool:
    """True si `text` completo parsea como un tool-call JSON (o lista de ellos)."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    items = parsed if isinstance(parsed, list) else [parsed]
    return bool(items) and any(
        isinstance(it, dict)
        and ("name" in it or "tool" in it)
        and ("parameters" in it or "arguments" in it or "args" in it)
        for it in items
    )


# Inicio de línea que abre un posible bloque JSON al FINAL de la respuesta.
_JSON_BLOCK_START = re.compile(r"^[ \t]*[\[{]", re.MULTILINE)


def strip_leaked_tool_json(reply: str) -> str:
    """Post-hook: evita que un tool-call JSON CRUDO se filtre como respuesta final.

    El 3B a veces, en la 2ª llamada, en vez de redactar intenta re-llamar una tool
    y emite algo como `{"name": "...", "parameters": {}}` que se cuela tal cual al
    patrón — a veces solo, a veces DESPUÉS de una frase de prosa. Dos casos:

    - Respuesta 100% JSON de tool-call: se sustituye por un mensaje en personaje.
    - Prosa + bloque JSON de tool-call al final: se conserva la prosa y se tira
      el bloque.

    Conservador contra falsos positivos: el bloque final debe (a) empezar una
    línea, (b) parsear como JSON VÁLIDO completo hasta el final y (c) tener la
    forma de tool-call (name/tool + parameters/arguments/args). Texto con llaves
    sueltas o JSON de datos legítimo no se toca.
    """
    if not reply:
        return reply
    stripped = reply.strip()
    if _es_tool_call_json(stripped):
        return ("Lo tengo, patrón, pero se me trabó la lengua al redactarlo. "
                "Pregúntamelo otra vez y te lo digo derecho.")
    # Bloque de tool-call al final tras prosa: probar cada inicio de línea con
    # { o [ y quedarnos con el primero cuyo resto sea un tool-call completo.
    for m in _JSON_BLOCK_START.finditer(stripped):
        prosa = stripped[:m.start()].strip()
        if prosa and _es_tool_call_json(stripped[m.start():]):
            return prosa
    return reply


def _facts_block() -> str | None:
    """El bloque de hechos del patrón tal como lo lee el modelo. "" sin hechos;
    None si no se pudo leer la base (se reintenta en el próximo turno)."""
    try:
        from crotolamo.persistence import facts

        ctx = facts.facts_context()
    except Exception as error:  # noqa: BLE001 - sin hechos, pero el turno sigue
        log.warning("no pude leer los hechos del patrón (reintento en el próximo turno): %s",
                    error)
        return None
    return f"[contexto que ya sabes sobre el patrón:\n{ctx}]" if ctx else ""


def make_facts_prehook():
    """Pre-hook que inyecta los hechos recordados del patrón UNA sola vez (primer
    turno), sustituyendo la inyección manual en el system prompt (M4.2).

    Es la de [memory].pin_facts = false: el bloque viaja dentro del primer
    mensaje guardado y se pierde cuando ese mensaje sale de la ventana o con
    /reset. De fábrica va anclado (make_pinned_facts_prehook).
    """
    state = {"done": False}

    def hook(text: str) -> str:
        if state["done"]:
            return text
        block = _facts_block()
        if block is None:
            # NO se marca como hecho: la marca iba antes de leer y un fallo
            # puntual en el primer turno (sqlite bloqueada) dejaba al proceso
            # sin hechos hasta reiniciar, sin una línea en el log. Se reintenta
            # en el turno siguiente.
            return text
        state["done"] = True
        if block:
            return f"{block}\n\n{text}"
        return text

    return hook


def make_pinned_facts_prehook(conversation):
    """Pre-hook que mantiene los hechos del patrón en un hueco ANCLADO de la
    Conversation ([memory].pin_facts, de fábrica) y devuelve el texto intacto.

    Mismo bloque que make_facts_prehook, pero tras el system prompt en vez de
    dentro del primer mensaje: lo ve el modelo en TODOS los turnos, también
    pasados max_turns turnos y tras /reset. Se relee en cada turno (una consulta
    SQLite de una tabla chica): un hecho que se guarda o se olvida se nota en el
    turno siguiente, y mientras no cambien, los bytes del hueco tampoco. Si la
    base falla, se queda el bloque que ya estaba.
    """

    def hook(text: str) -> str:
        block = _facts_block()
        if block is not None:
            conversation.set_pinned("facts", block or None)
        return text

    return hook


# Memoria semántica (mem0): los hooks viven junto a la Memoria, en core/memoria.py
# (make_memoria_prehook por turno y make_memoria_posthook tras responder).
from crotolamo.core.memoria import (  # noqa: E402,F401 - re-export
    make_memoria_posthook,
    make_memoria_prehook,
)


def datetime_prehook(text: str) -> str:
    """Pre-hook trivial de demostración: antepone la fecha/hora actual al contexto."""
    return f"[ahora: {datetime.now():%Y-%m-%d %H:%M}]\n{text}"
