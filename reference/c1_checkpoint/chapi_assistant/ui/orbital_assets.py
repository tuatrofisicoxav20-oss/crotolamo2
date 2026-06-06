from __future__ import annotations

from pathlib import Path
from tkinter import PhotoImage

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "orbital_ui"

_CACHE: dict[str, PhotoImage] = {}

def get_asset(name: str):
    path = ASSET_DIR / name
    if not path.exists():
        return None
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = PhotoImage(file=key)
    return _CACHE[key]
