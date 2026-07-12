"""Lógica pura del HUD (sin GTK, sin gi) — testeable headless.

El parseo defensivo del JSON vive en common.ipc (compartido con el panel);
aquí queda la interpretación de los campos del contrato.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.ipc import HUD_STATE_FILE, parse_hud_state, read_hud_state

__all__ = [
    "HUD_STATE_FILE",
    "VALID_MODES",
    "parse_hud_state",
    "read_hud_file",
    "extract_mode",
    "extract_text",
]

# Modos reconocidos (igual que los valores JSON del contrato de integración)
VALID_MODES = frozenset({"idle", "listening", "thinking", "speaking"})


def read_hud_file(path: Path) -> dict[str, Any]:
    """Lee y parsea hud_state.json de forma defensiva."""
    return read_hud_state(path)


def extract_mode(data: dict[str, Any]) -> str:
    """Extrae y valida el campo 'mode'; devuelve 'idle' si no es válido."""
    mode = data.get("mode", "idle")
    if mode not in VALID_MODES:
        return "idle"
    return mode


def extract_text(data: dict[str, Any]) -> str:
    """Extrae el campo 'text' de forma segura."""
    text = data.get("text", "")
    if not isinstance(text, str):
        return ""
    return text.strip()
