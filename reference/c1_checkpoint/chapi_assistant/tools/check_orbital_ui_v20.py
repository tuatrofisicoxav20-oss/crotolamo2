#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO ORBITAL UI v20 CHECK")
print("=" * 36)

for rel in [
    "ui/orbital_theme.py",
    "ui/orbital_widgets_v17.py",
    "ui/orbital_assets.py",
    "ui/orbital_ui_pro_v20_polish.py",
    "launch_orbital_ui_pro.py",
    "launch_orbital_ui_pro_v20.py",
]:
    print(("OK " if (ROOT / rel).exists() else "NO ") + rel)

for mod in [
    "ui.orbital_theme",
    "ui.orbital_widgets_v17",
    "ui.orbital_assets",
    "ui.orbital_ui_pro_v20_polish",
]:
    try:
        importlib.import_module(mod)
        print(f"{mod:34} OK")
    except Exception as e:
        print(f"{mod:34} NO ({type(e).__name__}: {e})")
