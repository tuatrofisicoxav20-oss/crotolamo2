"""
Crotolamo config manager v8.
Configuración local simple para no andar quemando valores en el código como cavernícolas con teclado.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 8,
    "ollama": {
        "model": "qwen2.5-coder:7b",
        "fallback_model": "llama3.2:latest",
        "timeout_seconds": 30,
        "use_context_engine": True,
    },
    "runtime": {
        "default_mode": "general",
        "save_history": True,
        "max_history_events_for_context": 8,
        "max_notes_for_context": 8,
    },
    "ui": {
        "theme": "orbital-neon",
        "show_context_panel": True,
        "show_memes": True,
    },
    "voice": {
        "enabled_by_default": False,
        "push_to_talk": True,
    },
    "updated_at": None,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path | None = None) -> Path:
    root = root or _root()
    return root / "config" / "crotolamo_settings.json"


def ensure_config(root: Path | None = None) -> Path:
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data = dict(DEFAULT_CONFIG)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _deep_merge(default: dict[str, Any], current: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    result = dict(current)
    for key, value in default.items():
        if key not in result:
            result[key] = value
            changed = True
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            merged, sub_changed = _deep_merge(value, result[key])
            result[key] = merged
            changed = changed or sub_changed
    return result, changed


def load_config(root: Path | None = None) -> dict[str, Any]:
    path = ensure_config(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        backup = path.with_suffix(".json.broken")
        path.replace(backup)
        data = dict(DEFAULT_CONFIG)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    data, changed = _deep_merge(DEFAULT_CONFIG, data)
    if changed:
        save_config(data, root)
    return data


def save_config(data: dict[str, Any], root: Path | None = None) -> Path:
    path = ensure_config(root)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def get_value(key_path: str, default: Any = None, root: Path | None = None) -> Any:
    data = load_config(root)
    cur: Any = data
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_value(key_path: str, value: Any, root: Path | None = None) -> None:
    data = load_config(root)
    cur = data
    parts = key_path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    save_config(data, root)


def parse_value(raw: str) -> Any:
    value = raw.strip()
    low = value.lower()
    if low in {"true", "sí", "si", "on", "yes"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def config_summary(root: Path | None = None) -> str:
    data = load_config(root)
    lines = ["Configuración Crotolamo v8:"]
    lines.append(f"- Modelo Ollama: {data.get('ollama', {}).get('model')}")
    lines.append(f"- Modelo fallback: {data.get('ollama', {}).get('fallback_model')}")
    lines.append(f"- Timeout Ollama: {data.get('ollama', {}).get('timeout_seconds')}s")
    lines.append(f"- Context engine: {data.get('ollama', {}).get('use_context_engine')}")
    lines.append(f"- Modo por defecto: {data.get('runtime', {}).get('default_mode')}")
    lines.append(f"- Historial en contexto: {data.get('runtime', {}).get('max_history_events_for_context')}")
    lines.append(f"- Notas en contexto: {data.get('runtime', {}).get('max_notes_for_context')}")
    lines.append(f"- Tema UI: {data.get('ui', {}).get('theme')}")
    lines.append(f"- Voz por defecto: {data.get('voice', {}).get('enabled_by_default')}")
    return "\n".join(lines)


def handle_config_command(text: str, root: Path | None = None) -> str | None:
    raw = (text or "").strip()
    low = raw.lower()

    if low in {"config", "configuración", "configuracion", "ver config", "ajustes"}:
        return config_summary(root)

    if low.startswith("config "):
        body = raw[len("config "):].strip()
        if "=" not in body:
            return "Formato: config ruta.clave = valor\nEjemplo: config ollama.timeout_seconds = 20"
        key, value = body.split("=", 1)
        key = key.strip()
        parsed = parse_value(value)
        set_value(key, parsed, root)
        return f"Configuración actualizada: {key} = {parsed}"

    if low.startswith("usar modelo "):
        model = raw[len("usar modelo "):].strip()
        if not model:
            return "Formato: usar modelo nombre_del_modelo"
        set_value("ollama.model", model, root)
        return f"Modelo principal actualizado: {model}"

    if low.startswith("timeout ollama "):
        value = raw[len("timeout ollama "):].strip()
        try:
            seconds = int(value)
        except Exception:
            return "Formato: timeout ollama 30"
        if seconds < 5:
            return "No lo puse menor a 5s porque eso ya es sabotaje con pasos extra."
        set_value("ollama.timeout_seconds", seconds, root)
        return f"Timeout de Ollama actualizado: {seconds}s"

    if low in {"contexto on", "context engine on"}:
        set_value("ollama.use_context_engine", True, root)
        return "Context engine activado."

    if low in {"contexto off", "context engine off"}:
        set_value("ollama.use_context_engine", False, root)
        return "Context engine desactivado."

    return None
