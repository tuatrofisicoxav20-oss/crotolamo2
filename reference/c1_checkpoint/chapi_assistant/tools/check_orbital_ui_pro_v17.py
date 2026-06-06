#!/usr/bin/env python3
from pathlib import Path
import sys, importlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print("CROTOLAMO ORBITAL UI PRO v17 CHECK")
print("=" * 45)
for mod in ["ui.orbital_theme", "ui.orbital_widgets_v17", "ui.orbital_ui_pro_v17", "core.crotolamo_runtime"]:
    try:
        importlib.import_module(mod)
        print(f"{mod:32} OK")
    except Exception as e:
        print(f"{mod:32} NO ({type(e).__name__}: {e})")
for rel in ["launch_orbital_ui_pro.py", "launch_orbital_ui_pro_v17.py"]:
    print(("OK " if (ROOT/rel).exists() else "NO ") + rel)
