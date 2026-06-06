#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO DOCTOR v11 EXTRA")
print("=" * 42)
print(f"Raíz: {ROOT}")

mods = [
    "core.project_inspector",
    "plugins.project_inspector_plugin",
    "core.project_indexer",
    "core.brain_engine",
    "core.crotolamo_runtime",
]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:34} OK")
    except Exception as e:
        print(f"{mod:34} NO ({e})")

try:
    from core.project_inspector import inspect_project, format_report
    print("\nInspección Crotolamo:")
    print("-" * 42)
    print(format_report(inspect_project("crotolamo", ROOT)))
except Exception as e:
    print(f"\nInspector: NO ({e})")
