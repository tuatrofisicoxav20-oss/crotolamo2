#!/usr/bin/env python3
from pathlib import Path
import sys, importlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print("CROTOLAMO ORBITAL UI ASSETS v18 CHECK")
print("=" * 48)
for rel in ["assets/orbital_ui/galaxy_bg.png","assets/orbital_ui/pezlin_3000.png","assets/orbital_ui/paloma_suprema_67.png","assets/orbital_ui/hacker_mode.png","assets/orbital_ui/sticker_strip.png"]:
    print(("OK " if (ROOT/rel).exists() else "NO ") + rel)
for mod in ["ui.orbital_assets","ui.orbital_ui_pro_v18_assets"]:
    try:
        importlib.import_module(mod)
        print(f"{mod:34} OK")
    except Exception as e:
        print(f"{mod:34} NO ({type(e).__name__}: {e})")
