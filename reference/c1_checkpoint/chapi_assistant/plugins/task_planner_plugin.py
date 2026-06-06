from __future__ import annotations

from pathlib import Path
from typing import Any

from core.task_planner import handle_task_planner_command, create_task_plan, format_plan, list_recent_plans


PLUGIN_NAME = "task_planner"
PLUGIN_DESCRIPTION = "Planificador de tareas técnicas por proyecto."


ACTIONS = {
    "task.plan": {
        "name": "task.plan",
        "mode": "general",
        "description": "Crea un plan técnico seguro para un objetivo.",
        "triggers": ["plan", "planificar", "crear plan"],
    },
    "task.recent": {
        "name": "task.recent",
        "mode": "general",
        "description": "Lista planes recientes.",
        "triggers": ["planes", "planes recientes"],
    },
}


def can_handle(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low in {"plan", "planificar", "crear plan", "planes", "ver planes", "planes recientes"}
        or low.startswith("plan ")
        or low.startswith("planificar ")
        or low.startswith("crear plan ")
        or low.startswith("hacer plan ")
        or low.startswith("planea ")
        or low.startswith("planear ")
    )


def run(text: str, root: Path | None = None) -> str:
    return handle_task_planner_command(text, root) or "No entendí ese comando de planificador."


def get_actions() -> dict[str, dict[str, Any]]:
    return ACTIONS


def run_action(action_name: str, root: Path | None = None, **kwargs: Any) -> str:
    if action_name == "task.recent":
        return list_recent_plans(root)
    if action_name == "task.plan":
        objective = kwargs.get("objective") or kwargs.get("text") or "mejorar proyecto"
        project = kwargs.get("project")
        return format_plan(create_task_plan(str(objective), project=project, root=root))
    return f"Acción planner no reconocida: {action_name}"
