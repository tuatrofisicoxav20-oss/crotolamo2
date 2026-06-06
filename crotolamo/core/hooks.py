"""Hooks pre/post LLM (M4, inspirado en los pipelines de Open WebUI).

Factories que producen funciones str->str para enchufar en ToolAgent.pre_hooks /
post_hooks. Mínimo y en personaje; NO es el motor entero de Open WebUI.
"""

from __future__ import annotations

from datetime import datetime


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
