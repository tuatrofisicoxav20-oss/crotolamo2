from __future__ import annotations

from pathlib import Path
from typing import Any

from core.project_indexer import handle_project_index_command, index_summary, scan_project, list_known_projects


PLUGIN_NAME = "project_index"
PLUGIN_DESCRIPTION = "Indexador de proyectos locales, mapas y búsqueda."


ACTIONS = {
    "project.index": {
        "name": "project.index",
        "mode": "general",
        "description": "Indexa un proyecto local.",
        "triggers": ["indexar", "indexar proyecto"],
    },
    "project.map": {
        "name": "project.map",
        "mode": "general",
        "description": "Muestra mapa de proyecto.",
        "triggers": ["mapa", "mapa proyecto"],
    },
    "project.known": {
        "name": "project.known",
        "mode": "general",
        "description": "Lista proyectos conocidos.",
        "triggers": ["proyectos", "proyectos conocidos"],
    },
}


def can_handle(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low in {
            "proyectos", "rutas proyectos", "proyectos conocidos",
            "index", "índice", "indice", "indexar", "indexar proyecto",
            "mapa", "mapa proyecto", "mapa crotolamo",
            "mapa huevonitis", "indexar huevonitis",
            "mapa tletl", "indexar tletl",
        }
        or low.startswith("indexar ")
        or low.startswith("mapa ")
        or low.startswith("buscar archivo ")
        or low.startswith("buscar texto ")
    )


def run(text: str, root: Path | None = None) -> str:
    return handle_project_index_command(text, root) or "No entendí ese comando de indexador."


def get_actions() -> dict[str, dict[str, Any]]:
    return ACTIONS


def run_action(action_name: str, root: Path | None = None, **kwargs: Any) -> str:
    project = kwargs.get("project") or kwargs.get("name") or "crotolamo"
    if action_name == "project.index":
        idx = scan_project(project, root)
        return index_summary(idx.name, root)
    if action_name == "project.map":
        return index_summary(project, root)
    if action_name == "project.known":
        return list_known_projects(root)
    return f"Acción de indexador no reconocida: {action_name}"
