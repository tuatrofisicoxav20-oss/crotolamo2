#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO UI ASSETS FIX CHECK")
print("=" * 38)

for rel in [
    "ui/orbital_assets.py",
    "ui/orbital_ui_pro_v19_hybrid.py",
    "launch_orbital_ui_pro.py",
]:
    p = ROOT / rel
    print(("OK " if p.exists() else "NO ") + rel)

for mod in [
    "ui.orbital_assets",
    "ui.orbital_ui_pro_v19_hybrid",
]:
    try:
        importlib.import_module(mod)
        print(f"{mod:32} OK")
    except Exception as e:
        print(f"{mod:32} NO ({type(e).__name__}: {e})")

asset_dir = ROOT / "assets" / "orbital_ui"
print(f"\nAsset dir: {asset_dir}")
print("OK assets/orbital_ui existe" if asset_dir.exists() else "AVISO assets/orbital_ui no existe")
