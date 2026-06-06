from __future__ import annotations

from pathlib import Path
from typing import Any

from core.context_engine import context_summary, handle_context_command
from core.config_manager import config_summary, handle_config_command


PLUGIN_NAME = "context"
PLUGIN_DESCRIPTION = "Motor de contexto y configuración local."


ACTIONS = {
    "context.summary": {
        "name": "context.summary",
        "mode": "general",
        "description": "Muestra el contexto local que Crotolamo usará.",
        "triggers": ["contexto", "ver contexto", "contexto actual"],
    },
    "context.config": {
        "name": "context.config",
        "mode": "general",
        "description": "Muestra configuración local.",
        "triggers": ["config", "ajustes", "configuración"],
    },
}


def can_handle(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low in {"contexto", "ver contexto", "context engine", "contexto actual", "config", "configuración", "configuracion", "ver config", "ajustes"}
        or low.startswith("config ")
        or low.startswith("usar modelo ")
        or low.startswith("timeout ollama ")
        or low in {"contexto on", "contexto off", "context engine on", "context engine off"}
    )


def run(text: str, root: Path | None = None) -> str:
    return (
        handle_context_command(text, root)
        or handle_config_command(text, root)
        or "No entendí ese comando de contexto/configuración."
    )


def get_actions() -> dict[str, dict[str, Any]]:
    return ACTIONS


def run_action(action_name: str, root: Path | None = None, **kwargs: Any) -> str:
    if action_name == "context.summary":
        return context_summary(root)
    if action_name == "context.config":
        return config_summary(root)
    return f"Acción de contexto no reconocida: {action_name}"
