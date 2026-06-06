#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO DOCTOR v10 EXTRA")
print("=" * 42)
print(f"Raíz: {ROOT}")

mods = [
    "core.project_indexer",
    "plugins.project_index_plugin",
    "core.brain_engine",
    "core.context_engine",
    "core.crotolamo_runtime",
]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:30} OK")
    except Exception as e:
        print(f"{mod:30} NO ({e})")

try:
    from core.project_indexer import list_known_projects, index_summary
    print("\nProyectos:")
    print("-" * 42)
    print(list_known_projects(ROOT))
    print("\nMapa Crotolamo:")
    print("-" * 42)
    print(index_summary("crotolamo", ROOT))
except Exception as e:
    print(f"\nIndexador: NO ({e})")
