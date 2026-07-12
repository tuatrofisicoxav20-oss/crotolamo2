"""Detección del emulador de terminal disponible — sin gi, testeable headless."""

from __future__ import annotations

import shutil

_CANDIDATES: list[tuple[str, list[str]]] = [
    ("kitty", ["kitty", "-e"]),
    ("ptyxis", ["ptyxis", "--"]),
    ("gnome-terminal", ["gnome-terminal", "--"]),
    ("konsole", ["konsole", "-e"]),
    ("foot", ["foot"]),
    ("alacritty", ["alacritty", "-e"]),
    ("xterm", ["xterm", "-e"]),
]


def _find_terminal() -> list[str] | None:
    """Prefijo de comando del primer terminal instalado, o None si no hay."""
    for binary, prefix in _CANDIDATES:
        if shutil.which(binary):
            return prefix
    return None
