"""
Crotolamo local memory v7.
Memoria local simple, segura y editable en JSON.

No intenta ser una base de datos gigante ni una caja negra dramática.
Guarda preferencias, aliases, notas y rutas útiles para que el runtime pueda
responder con más contexto sin depender de magia.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROFILE = {
    "display_name": "Caos Orbital",
    "assistant_name": "Crotolamo",
    "tone": "directo, útil, con humor, sin dar la razón si no la tiene",
    "default_mode": "general",
    "preferred_model": "qwen2.5-coder:7b",
}

DEFAULT_PREFERENCES = {
    "voice_enabled": False,
    "confirm_risky_commands": True,
    "save_history": True,
    "show_memes": True,
    "ui_theme": "orbital-neon",
}

DEFAULT_PROJECT_PATHS = {
    "crotolamo": "~/Documentos/chapi_assistant",
    "huevonitis": "~/Documentos/huevonitis version 2.1",
    "tletl": "~/Documentos/tletl_control_v4_1_ai_integrado",
}

DEFAULT_ALIASES = {
    "doctor": "accion general.doctor",
    "diag": "accion general.doctor",
    "huevo": "modo huevonitis",
    "tletl": "modo tletl",
    "fedora": "modo fedora",
    "escuela": "modo escuela",
    "lab": "modo laboratorio",
}

DEFAULT_MEMORY = {
    "version": 7,
    "profile": DEFAULT_PROFILE,
    "preferences": DEFAULT_PREFERENCES,
    "project_paths": DEFAULT_PROJECT_PATHS,
    "aliases": DEFAULT_ALIASES,
    "notes": [],
    "facts": {},
    "updated_at": None,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def memory_path(root: Path | None = None) -> Path:
    root = root or _root()
    return root / "data" / "memory" / "local_memory.json"


def ensure_memory(root: Path | None = None) -> Path:
    path = memory_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data = dict(DEFAULT_MEMORY)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_memory(root: Path | None = None) -> dict[str, Any]:
    path = ensure_memory(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        backup = path.with_suffix(".json.broken")
        path.replace(backup)
        data = dict(DEFAULT_MEMORY)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    changed = False
    for key, value in DEFAULT_MEMORY.items():
        if key not in data:
            data[key] = value
            changed = True
    if changed:
        save_memory(data, root)
    return data


def save_memory(data: dict[str, Any], root: Path | None = None) -> Path:
    path = ensure_memory(root)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def get_alias(text: str, root: Path | None = None) -> str | None:
    text = (text or "").strip().lower()
    data = load_memory(root)
    aliases = data.get("aliases", {})
    return aliases.get(text)


def set_alias(name: str, command: str, root: Path | None = None) -> None:
    data = load_memory(root)
    data.setdefault("aliases", {})[name.strip().lower()] = command.strip()
    save_memory(data, root)


def forget_alias(name: str, root: Path | None = None) -> bool:
    data = load_memory(root)
    aliases = data.setdefault("aliases", {})
    key = name.strip().lower()
    existed = key in aliases
    aliases.pop(key, None)
    save_memory(data, root)
    return existed


def add_note(note: str, root: Path | None = None) -> None:
    data = load_memory(root)
    data.setdefault("notes", []).append({
        "text": note.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_memory(data, root)


def list_notes(root: Path | None = None, limit: int = 12) -> list[dict[str, str]]:
    data = load_memory(root)
    notes = data.get("notes", [])
    return list(reversed(notes[-limit:]))


def set_fact(key: str, value: str, root: Path | None = None) -> None:
    data = load_memory(root)
    data.setdefault("facts", {})[key.strip()] = {
        "value": value.strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_memory(data, root)


def forget_fact(key: str, root: Path | None = None) -> bool:
    data = load_memory(root)
    facts = data.setdefault("facts", {})
    existed = key in facts
    facts.pop(key, None)
    save_memory(data, root)
    return existed


def memory_summary(root: Path | None = None) -> str:
    data = load_memory(root)
    profile = data.get("profile", {})
    prefs = data.get("preferences", {})
    aliases = data.get("aliases", {})
    project_paths = data.get("project_paths", {})
    notes = data.get("notes", [])
    facts = data.get("facts", {})

    lines = [
        "Memoria local Crotolamo v7",
        f"Usuario: {profile.get('display_name', 'sin nombre')}",
        f"Asistente: {profile.get('assistant_name', 'Crotolamo')}",
        f"Modelo preferido: {profile.get('preferred_model', 'no definido')}",
        f"Modo por defecto: {profile.get('default_mode', 'general')}",
        "",
        "Preferencias:",
    ]
    for k, v in prefs.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Rutas de proyectos:")
    for k, v in project_paths.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append(f"Aliases: {len(aliases)}")
    for k, v in sorted(aliases.items()):
        lines.append(f"- {k} -> {v}")

    lines.append("")
    lines.append(f"Notas guardadas: {len(notes)}")
    for item in list_notes(root, limit=5):
        lines.append(f"- {item.get('created_at', '?')}: {item.get('text', '')}")

    lines.append("")
    lines.append(f"Facts guardados: {len(facts)}")
    for k, item in facts.items():
        if isinstance(item, dict):
            lines.append(f"- {k}: {item.get('value', '')}")
        else:
            lines.append(f"- {k}: {item}")
    return "\n".join(lines)


def handle_memory_command(text: str, root: Path | None = None) -> str | None:
    raw = (text or "").strip()
    low = raw.lower()

    if low in {"memoria", "memory", "ver memoria", "mostrar memoria"}:
        return memory_summary(root)

    if low in {"notas", "ver notas", "mis notas"}:
        notes = list_notes(root)
        if not notes:
            return "No hay notas guardadas."
        return "Notas guardadas:\n" + "\n".join(
            f"- {n.get('created_at', '?')}: {n.get('text', '')}" for n in notes
        )

    prefixes_note = ("recuerda que ", "nota ", "guardar nota ", "memoria nota ")
    for prefix in prefixes_note:
        if low.startswith(prefix):
            note = raw[len(prefix):].strip()
            if not note:
                return "No guardé nada porque la nota venía vacía. Qué eficiencia tan trágica."
            add_note(note, root)
            return f"Nota guardada: {note}"

    if low.startswith("alias "):
        # alias nombre = comando
        body = raw[6:].strip()
        if "=" not in body:
            return "Formato: alias nombre = comando"
        name, command = body.split("=", 1)
        if not name.strip() or not command.strip():
            return "Formato: alias nombre = comando"
        set_alias(name, command, root)
        return f"Alias guardado: {name.strip().lower()} -> {command.strip()}"

    if low.startswith("olvida alias "):
        name = raw[len("olvida alias "):].strip()
        ok = forget_alias(name, root)
        return f"Alias eliminado: {name}" if ok else f"No existía el alias: {name}"

    if low.startswith("dato "):
        # dato clave = valor
        body = raw[5:].strip()
        if "=" not in body:
            return "Formato: dato clave = valor"
        key, value = body.split("=", 1)
        if not key.strip() or not value.strip():
            return "Formato: dato clave = valor"
        set_fact(key, value, root)
        return f"Dato guardado: {key.strip()} = {value.strip()}"

    if low.startswith("olvida dato "):
        key = raw[len("olvida dato "):].strip()
        ok = forget_fact(key, root)
        return f"Dato eliminado: {key}" if ok else f"No existía el dato: {key}"

    return None
