"""
Crotolamo Plugin Registry v6.

Centro de acciones para que Crotolamo deje de improvisar cada vez que quieres
trabajar en Huevonitis, Tletl, Fedora o el laboratorio. Sí, organización. Qué
concepto tan subversivo.
"""

from __future__ import annotations

import importlib
import shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


PLUGIN_MODULES = [
    "plugins.general_actions",
    "plugins.huevonitis_plugin",
    "plugins.tletl_plugin",
    "plugins.fedora_actions",
    "plugins.school_plugin",
    "plugins.laboratory_plugin",
]


@dataclass(frozen=True)
class PluginAction:
    key: str
    title: str
    description: str
    modes: list[str]
    aliases: list[str]
    commands: list[str]
    risk: str = "safe"
    direct_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _q(path: str | Path) -> str:
    return shlex.quote(str(Path(str(path)).expanduser()))


class PluginRegistry:
    version = "v6"

    def __init__(self, project_root: str | Path, mode_manager: Any, settings: dict[str, Any] | None = None) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.mode_manager = mode_manager
        self.settings = settings or {}
        self.import_errors: dict[str, str] = {}
        self.actions: list[PluginAction] = []
        self.reload()

    def reload(self) -> None:
        self.import_errors.clear()
        actions: list[PluginAction] = []
        for module_name in PLUGIN_MODULES:
            try:
                module = importlib.import_module(module_name)
                raw_actions = getattr(module, "ACTIONS", [])
                for raw in raw_actions:
                    action = self._coerce_action(raw)
                    if action:
                        actions.append(action)
            except Exception as error:  # pragma: no cover
                self.import_errors[module_name] = str(error)
        # Evita duplicados por key, gana el último cargado solo si insiste. Drama mínimo.
        by_key: dict[str, PluginAction] = {}
        for action in actions:
            by_key[action.key] = action
        self.actions = list(by_key.values())

    def _coerce_action(self, raw: Any) -> PluginAction | None:
        if not isinstance(raw, dict):
            return None
        key = str(raw.get("key", "")).strip()
        title = str(raw.get("title", key)).strip()
        if not key or not title:
            return None
        modes = raw.get("modes", ["*"])
        aliases = raw.get("aliases", [])
        commands = raw.get("commands", [])
        if isinstance(modes, str):
            modes = [modes]
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(commands, str):
            commands = [commands]
        return PluginAction(
            key=key,
            title=title,
            description=str(raw.get("description", "")).strip(),
            modes=[str(m).strip().lower() for m in modes if str(m).strip()],
            aliases=[str(a).strip().lower() for a in aliases if str(a).strip()],
            commands=[str(c).strip() for c in commands if str(c).strip()],
            risk=str(raw.get("risk", "safe")).strip().lower() or "safe",
            direct_text=str(raw.get("direct_text", "")).strip(),
        )

    def active_mode_key(self) -> str:
        try:
            return str(self.mode_manager.active_key())
        except Exception:
            return "crotolamo"

    def active_mode_path(self) -> Path:
        try:
            mode = self.mode_manager.get_mode()
            raw = mode.get("path") or self.project_root
            return Path(str(raw)).expanduser()
        except Exception:
            return self.project_root

    def actions_for_mode(self, mode_key: str | None = None) -> list[PluginAction]:
        mode_key = (mode_key or self.active_mode_key()).lower()
        result: list[PluginAction] = []
        for action in self.actions:
            if "*" in action.modes or mode_key in action.modes:
                result.append(action)
        return sorted(result, key=lambda a: ("*" not in a.modes, a.key))

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "active_mode": self.active_mode_key(),
            "active_path": str(self.active_mode_path()),
            "total_actions": len(self.actions),
            "active_actions": [a.to_dict() for a in self.actions_for_mode()],
            "import_errors": dict(self.import_errors),
        }

    def summary_text(self, only_active: bool = False) -> str:
        active = self.active_mode_key()
        actions = self.actions_for_mode(active) if only_active else sorted(self.actions, key=lambda a: a.key)
        lines = ["Crotolamo Actions / Plugins v6", f"Modo activo: {active}", f"Ruta activa: {self.active_mode_path()}", ""]
        if not actions:
            lines.append("No hay acciones cargadas. La nada organizada, básicamente.")
        else:
            for action in actions:
                mode_label = ",".join(action.modes)
                alias = f" | alias: {', '.join(action.aliases[:3])}" if action.aliases else ""
                lines.append(f"- {action.key:<24} [{mode_label}] {action.title}{alias}")
                if action.description:
                    lines.append(f"  {action.description}")
        lines.append("")
        lines.append("Uso: 'acciones', 'acciones modo', 'accion <clave>' o un alias como 'revisar huevonitis'.")
        if self.import_errors:
            lines.append("\nErrores de plugins:")
            for name, error in self.import_errors.items():
                lines.append(f"- {name}: {error}")
        return "\n".join(lines)

    def _match_action(self, text: str) -> PluginAction | None:
        lower = _norm(text)
        explicit_prefixes = ("accion ", "acción ", "ejecuta accion ", "ejecuta acción ", "correr accion ", "correr acción ")
        requested = lower
        for prefix in explicit_prefixes:
            if lower.startswith(prefix):
                requested = lower[len(prefix):].strip()
                break

        active_actions = self.actions_for_mode()
        all_actions = sorted(self.actions, key=lambda a: a.key)
        candidates = active_actions + [a for a in all_actions if a not in active_actions]

        for action in candidates:
            names = [action.key.lower(), action.title.lower()] + action.aliases
            if requested in names:
                return action
        for action in candidates:
            names = [action.key.lower(), action.title.lower()] + action.aliases
            if any(alias and alias in lower for alias in names):
                return action
        return None

    def _format_commands(self, commands: list[str]) -> list[str]:
        active_path = self.active_mode_path()
        values = {
            "root": _q(self.project_root),
            "active_path": _q(active_path),
            "active": _q(active_path),
            "mode_key": self.active_mode_key(),
        }
        formatted: list[str] = []
        for cmd in commands:
            try:
                formatted.append(cmd.format(**values))
            except Exception:
                formatted.append(cmd)
        return formatted

    def handle_command(self, text: str) -> dict[str, Any] | None:
        lower = _norm(text)
        if lower in {"plugins", "acciones", "actions", "centro de acciones"}:
            return {"kind": "direct", "text": self.summary_text(only_active=False), "meta": {"plugins": self.payload()}}
        if lower in {"acciones modo", "acciones del modo", "acciones activas", "plugins modo"}:
            return {"kind": "direct", "text": self.summary_text(only_active=True), "meta": {"plugins": self.payload()}}

        action = self._match_action(text)
        if not action:
            return None

        commands = self._format_commands(action.commands)
        if not commands and action.direct_text:
            return {"kind": "direct", "text": action.direct_text, "meta": {"action": action.to_dict(), "plugins": self.payload()}}

        explanation = (
            f"Acción v6: {action.title}\n"
            f"{action.description}\n\n"
            f"Modo activo: {self.active_mode_key()}\n"
            f"Ruta activa: {self.active_mode_path()}\n\n"
            "Revisa los comandos antes de ejecutar. Crotolamo ya no va a hacer acrobacias con sudo solo porque sí."
        ).strip()
        return {
            "kind": "plan",
            "safe": action.risk != "blocked",
            "explanation": explanation,
            "commands": commands,
            "meta": {"source": "plugins", "action": action.to_dict(), "plugins": self.payload()},
        }
