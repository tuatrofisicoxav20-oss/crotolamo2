#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO ORBITAL UI PRO CHECK")
print("=" * 42)

checks = [
    "ui.orbital_ui_pro",
    "core.crotolamo_runtime",
]

for mod in checks:
    try:
        importlib.import_module(mod)
        print(f"{mod:30} OK")
    except Exception as e:
        print(f"{mod:30} NO ({type(e).__name__}: {e})")

print("\nArchivos:")
for rel in ["launch_orbital_ui_pro.py", "ui/orbital_ui_pro.py"]:
    p = ROOT / rel
    print(f"{'OK' if p.exists() else 'NO'} {rel}")
