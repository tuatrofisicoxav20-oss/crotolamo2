#!/usr/bin/env python3
from pathlib import Path
import sys, importlib
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
print("CROTOLAMO DOCTOR v13-v15 EXTRA")
print("="*46); print(f"Raíz: {ROOT}")
mods=["core.patch_builder","core.test_runner","core.safe_executor","plugins.patch_builder_plugin","plugins.test_runner_plugin","plugins.safe_executor_plugin","core.crotolamo_runtime"]
for m in mods:
    try: importlib.import_module(m); print(f"{m:34} OK")
    except Exception as e: print(f"{m:34} NO ({e})")
try:
    from core.safe_executor import classify
    print("\nClasificación rápida:")
    for c in ["ls","python -m py_compile core/crotolamo_runtime.py","sudo dnf install cowsay","rm -rf /"]:
        print(f"- {classify(c).upper():8} {c}")
except Exception as e: print(f"\nSafe classify: NO ({e})")
