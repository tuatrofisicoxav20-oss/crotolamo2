"""
Plugin de memoria para Crotolamo v7.
Puede usarse desde el plugin registry si el runtime lo descubre.
También funciona importándolo directo desde el runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.local_memory import handle_memory_command, memory_summary
from core.session_history import history_summary


PLUGIN_NAME = "memory"
PLUGIN_DESCRIPTION = "Memoria local, notas, aliases e historial de sesión."


ACTIONS = {
    "memory.summary": {
        "name": "memory.summary",
        "mode": "general",
        "description": "Muestra resumen de memoria local.",
        "triggers": ["memoria", "ver memoria", "mostrar memoria"],
    },
    "memory.history": {
        "name": "memory.history",
        "mode": "general",
        "description": "Muestra historial reciente.",
        "triggers": ["historial", "ver historial", "sesión", "sesion"],
    },
}


def can_handle(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low in {"memoria", "memory", "ver memoria", "mostrar memoria", "historial", "ver historial", "history", "sesión", "sesion", "notas", "ver notas", "mis notas"}
        or low.startswith("recuerda que ")
        or low.startswith("nota ")
        or low.startswith("guardar nota ")
        or low.startswith("memoria nota ")
        or low.startswith("alias ")
        or low.startswith("olvida alias ")
        or low.startswith("dato ")
        or low.startswith("olvida dato ")
    )


def run(text: str, root: Path | None = None) -> str:
    low = (text or "").strip().lower()
    if low in {"historial", "ver historial", "history", "sesión", "sesion"}:
        return history_summary(root)
    result = handle_memory_command(text, root)
    return result or "No entendí ese comando de memoria."


def get_actions() -> dict[str, dict[str, Any]]:
    return ACTIONS


def run_action(action_name: str, root: Path | None = None, **kwargs: Any) -> str:
    if action_name == "memory.summary":
        return memory_summary(root)
    if action_name == "memory.history":
        return history_summary(root)
    return f"Acción de memoria no reconocida: {action_name}"
