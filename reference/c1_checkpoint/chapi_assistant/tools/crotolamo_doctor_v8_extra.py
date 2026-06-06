#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO DOCTOR v8 EXTRA")
print("=" * 42)
print(f"Raíz: {ROOT}")

mods = [
    "core.config_manager",
    "core.context_engine",
    "plugins.context_plugin",
    "core.local_memory",
    "core.session_history",
    "core.crotolamo_runtime",
]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:30} OK")
    except Exception as e:
        print(f"{mod:30} NO ({e})")

try:
    from core.config_manager import ensure_config, config_summary
    path = ensure_config(ROOT)
    print(f"\nConfig: {path}")
    print("-" * 42)
    print(config_summary(ROOT))
except Exception as e:
    print(f"\nConfig: NO ({e})")

try:
    from core.context_engine import context_summary
    print("\nContexto:")
    print("-" * 42)
    print(context_summary(ROOT))
except Exception as e:
    print(f"\nContexto: NO ({e})")
