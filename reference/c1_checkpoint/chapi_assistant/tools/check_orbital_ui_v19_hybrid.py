#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO ORBITAL UI v19 HYBRID CHECK")
print("=" * 46)

mods = [
    "ui.orbital_theme",
    "ui.orbital_widgets_v17",
    "ui.orbital_assets",
    "ui.orbital_ui_pro_v19_hybrid",
]
for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:34} OK")
    except Exception as e:
        print(f"{mod:34} NO ({type(e).__name__}: {e})")

for rel in [
    "launch_orbital_ui_pro.py",
    "launch_orbital_ui_pro_v19.py",
    "ui/orbital_ui_pro_v19_hybrid.py",
]:
    print(("OK " if (ROOT / rel).exists() else "NO ") + rel)
