from __future__ import annotations

from pathlib import Path

from plugins.base_plugin import BasePlugin, PluginInfo, PluginResult


class ProjectsPlugin(BasePlugin):
    info = PluginInfo(name="projects", description="Rutas rápidas de Crotolamo/Huevonitis/Tletl")

    def __init__(self, project_paths: dict[str, str]) -> None:
        self.project_paths = {k.lower(): Path(v).expanduser() for k, v in project_paths.items()}

    def can_handle(self, text: str) -> bool:
        lower = text.lower().strip()
        return lower.startswith("ruta ") or lower.startswith("donde esta ") or lower.startswith("dónde está ")

    def handle(self, text: str) -> PluginResult:
        lower = text.lower().strip()
        for name, path in self.project_paths.items():
            if name in lower:
                exists = "sí existe" if path.exists() else "no existe todavía"
                return PluginResult(True, f"{name}: {path} ({exists})")
        return PluginResult(True, "No reconocí ese proyecto. Tengo: " + ", ".join(sorted(self.project_paths)))
