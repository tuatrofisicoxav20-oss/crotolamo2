#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("CROTOLAMO DOCTOR v12 EXTRA")
print("=" * 42)
print(f"Raíz: {ROOT}")

mods = [
    "core.task_planner",
    "plugins.task_planner_plugin",
    "core.project_inspector",
    "core.project_indexer",
    "core.crotolamo_runtime",
]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod:34} OK")
    except Exception as e:
        print(f"{mod:34} NO ({e})")

try:
    from core.task_planner import create_task_plan, format_plan
    plan = create_task_plan("probar planificador de crotolamo", project="crotolamo", root=ROOT)
    print("\nPlan de prueba:")
    print("-" * 42)
    print(format_plan(plan).splitlines()[0])
    print("Plan guardado en data/task_plans/")
except Exception as e:
    print(f"\nPlanner: NO ({e})")
