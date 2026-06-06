#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO DOCTOR v7 EXTRA")
print("=" * 42)
print(f"Raíz: {ROOT}")

mods = [
    "core.local_memory",
    "core.session_history",
    "core.project_modes",
    "core.plugin_registry",
    "core.crotolamo_runtime",
]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:30} OK")
    except Exception as e:
        print(f"{mod:30} NO ({e})")

try:
    from core.local_memory import ensure_memory, memory_summary
    path = ensure_memory(ROOT)
    print(f"\nMemoria local: {path}")
    print("-" * 42)
    print(memory_summary(ROOT))
except Exception as e:
    print(f"\nMemoria local: NO ({e})")

try:
    from core.session_history import history_summary
    print("\n" + history_summary(ROOT, limit=5))
except Exception as e:
    print(f"\nHistorial: NO ({e})")
