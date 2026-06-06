from __future__ import annotations

from pathlib import Path
from typing import Any

from core.project_inspector import handle_project_inspector_command, inspect_project, format_report


PLUGIN_NAME = "project_inspector"
PLUGIN_DESCRIPTION = "Inspección avanzada de proyectos locales."


ACTIONS = {
    "project.inspect": {
        "name": "project.inspect",
        "mode": "general",
        "description": "Inspecciona un proyecto local con el índice v10.",
        "triggers": ["inspeccionar", "auditar", "analizar proyecto"],
    }
}


def can_handle(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low in {
            "inspeccionar", "auditar", "inspeccionar proyecto", "auditar proyecto",
            "inspeccionar huevonitis", "auditar huevonitis",
            "inspeccionar tletl", "auditar tletl",
        }
        or low.startswith("inspeccionar ")
        or low.startswith("auditar ")
        or low.startswith("analizar proyecto ")
        or low.startswith("dependencias ")
        or low.startswith("entradas ")
        or low.startswith("pendientes ")
    )


def run(text: str, root: Path | None = None) -> str:
    return handle_project_inspector_command(text, root) or "No entendí ese comando de inspección."


def get_actions() -> dict[str, dict[str, Any]]:
    return ACTIONS


def run_action(action_name: str, root: Path | None = None, **kwargs: Any) -> str:
    if action_name == "project.inspect":
        project = kwargs.get("project") or kwargs.get("name") or "crotolamo"
        return format_report(inspect_project(project, root))
    return f"Acción inspector no reconocida: {action_name}"
