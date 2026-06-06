"""
Crotolamo Context Engine v8.

Construye contexto compacto usando:
- memoria local
- configuración
- modo activo
- historial reciente
- estado del sistema
- rutas de proyectos

Meta: que Crotolamo no responda como pez dorado cyberpunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import platform


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_imports():
    imports: dict[str, Any] = {}
    try:
        from core.local_memory import load_memory, list_notes
        imports["load_memory"] = load_memory
        imports["list_notes"] = list_notes
    except Exception:
        pass
    try:
        from core.session_history import recent_events
        imports["recent_events"] = recent_events
    except Exception:
        pass
    try:
        from core.config_manager import load_config
        imports["load_config"] = load_config
    except Exception:
        pass
    try:
        from core.project_modes import get_current_mode, get_mode_context, list_modes
        imports["get_current_mode"] = get_current_mode
        imports["get_mode_context"] = get_mode_context
        imports["list_modes"] = list_modes
    except Exception:
        pass
    try:
        from core.system_probe import get_system_snapshot
        imports["get_system_snapshot"] = get_system_snapshot
    except Exception:
        pass
    return imports


@dataclass
class ContextSnapshot:
    user_text: str
    mode: str
    profile: dict[str, Any]
    preferences: dict[str, Any]
    project_paths: dict[str, str]
    notes: list[dict[str, str]]
    facts: dict[str, Any]
    history: list[dict[str, Any]]
    system: dict[str, Any]
    config: dict[str, Any]


def _simple_system_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "home": str(Path.home()),
    }


def _get_mode(root: Path, imports: dict[str, Any]) -> str:
    fn = imports.get("get_current_mode")
    if callable(fn):
        try:
            mode = fn(root)
            if isinstance(mode, str) and mode:
                return mode
            if hasattr(mode, "name"):
                return str(mode.name)
        except Exception:
            pass

    # fallback simple: buscar config/mode_state.json o config/crotolamo_modes.json
    for rel in ("config/current_mode.txt", "config/crotolamo_current_mode.txt"):
        p = root / rel
        if p.exists():
            try:
                value = p.read_text(encoding="utf-8").strip()
                if value:
                    return value
            except Exception:
                pass
    return "general"


def collect_context(user_text: str, root: Path | None = None) -> ContextSnapshot:
    root = root or _root()
    imports = _safe_imports()

    config = {}
    if callable(imports.get("load_config")):
        try:
            config = imports["load_config"](root)
        except Exception:
            config = {}

    memory = {}
    if callable(imports.get("load_memory")):
        try:
            memory = imports["load_memory"](root)
        except Exception:
            memory = {}

    notes = []
    max_notes = int(config.get("runtime", {}).get("max_notes_for_context", 8) or 8)
    if callable(imports.get("list_notes")):
        try:
            notes = imports["list_notes"](root, limit=max_notes)
        except Exception:
            notes = list(reversed(memory.get("notes", [])[-max_notes:]))

    history = []
    max_hist = int(config.get("runtime", {}).get("max_history_events_for_context", 8) or 8)
    if callable(imports.get("recent_events")):
        try:
            history = imports["recent_events"](root, limit=max_hist)
        except Exception:
            history = []

    system = {}
    if callable(imports.get("get_system_snapshot")):
        try:
            snap = imports["get_system_snapshot"]()
            if isinstance(snap, dict):
                system = snap
            else:
                system = getattr(snap, "__dict__", {}) or {}
        except Exception:
            system = _simple_system_snapshot()
    else:
        system = _simple_system_snapshot()

    return ContextSnapshot(
        user_text=user_text,
        mode=_get_mode(root, imports),
        profile=memory.get("profile", {}),
        preferences=memory.get("preferences", {}),
        project_paths=memory.get("project_paths", {}),
        notes=notes,
        facts=memory.get("facts", {}),
        history=history,
        system=system,
        config=config,
    )


def _compact_value(value: Any, max_len: int = 180) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_context_block(snapshot: ContextSnapshot) -> str:
    lines: list[str] = []
    profile = snapshot.profile or {}
    config = snapshot.config or {}

    lines.append("### CONTEXTO LOCAL DE CROTOLAMO")
    lines.append(f"Usuario: {profile.get('display_name', 'Caos Orbital')}")
    lines.append(f"Asistente: {profile.get('assistant_name', 'Crotolamo')}")
    lines.append(f"Tono preferido: {profile.get('tone', 'directo, útil y sin inventar')}")
    lines.append(f"Modo activo: {snapshot.mode}")
    lines.append(f"Modelo preferido: {config.get('ollama', {}).get('model', profile.get('preferred_model', 'no definido'))}")

    if snapshot.project_paths:
        lines.append("")
        lines.append("Rutas de proyectos:")
        for name, path in snapshot.project_paths.items():
            lines.append(f"- {name}: {path}")

    if snapshot.facts:
        lines.append("")
        lines.append("Datos guardados:")
        for key, item in list(snapshot.facts.items())[:12]:
            if isinstance(item, dict):
                lines.append(f"- {key}: {_compact_value(item.get('value', ''))}")
            else:
                lines.append(f"- {key}: {_compact_value(item)}")

    if snapshot.notes:
        lines.append("")
        lines.append("Notas recientes:")
        for note in snapshot.notes[:8]:
            lines.append(f"- {note.get('text', '')}")

    if snapshot.history:
        lines.append("")
        lines.append("Historial reciente:")
        for e in snapshot.history[:8]:
            kind = e.get("kind", "?")
            content = _compact_value(e.get("content", ""), 160)
            lines.append(f"- [{kind}] {content}")

    if snapshot.system:
        lines.append("")
        lines.append("Estado del sistema resumido:")
        useful_keys = ("python", "platform", "cwd", "home", "ram_percent", "battery_percent", "disk_home_percent", "ollama_ok")
        added = False
        for key in useful_keys:
            if key in snapshot.system:
                lines.append(f"- {key}: {_compact_value(snapshot.system[key])}")
                added = True
        if not added:
            for key, value in list(snapshot.system.items())[:8]:
                lines.append(f"- {key}: {_compact_value(value)}")

    lines.append("")
    lines.append("Reglas de respuesta:")
    lines.append("- Usa el contexto solo si ayuda.")
    lines.append("- No inventes archivos, rutas, resultados ni estados.")
    lines.append("- Si falta evidencia, dilo.")
    lines.append("- Para comandos peligrosos, no ejecutes ni sugieras sin explicar riesgo.")
    lines.append("- Responde en español, claro y directo.")

    return "\n".join(lines)


def build_enriched_prompt(user_text: str, root: Path | None = None) -> str:
    snapshot = collect_context(user_text, root)
    block = build_context_block(snapshot)
    return f"{block}\n\n### PETICIÓN DEL USUARIO\n{user_text.strip()}"


def context_summary(root: Path | None = None) -> str:
    snapshot = collect_context("resumen", root)
    return build_context_block(snapshot)


def should_skip_context(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return True
    prefixes = (
        "memoria",
        "historial",
        "notas",
        "recuerda que ",
        "nota ",
        "guardar nota ",
        "alias ",
        "olvida alias ",
        "dato ",
        "olvida dato ",
        "config",
        "usar modelo ",
        "timeout ollama ",
        "contexto ",
        "modo ",
        "modos",
        "acciones",
        "accion ",
        "diagnóstico",
        "diagnostico",
        "runtime",
    )
    return any(low == p.strip() or low.startswith(p) for p in prefixes)


def handle_context_command(text: str, root: Path | None = None) -> str | None:
    low = (text or "").strip().lower()
    if low in {"contexto", "ver contexto", "context engine", "contexto actual"}:
        return context_summary(root)
    return None
