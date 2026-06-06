#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.task_planner import handle_task_planner_command, create_task_plan, format_plan, list_recent_plans

def main() -> int:
    if len(sys.argv) == 1:
        print(list_recent_plans(ROOT))
        print()
        print("Uso:")
        print("  python tools/crotolamo_task_planner_cli.py plan mejorar extractor de huevonitis")
        return 0

    text = " ".join(sys.argv[1:]).strip()
    result = handle_task_planner_command(text, ROOT)
    if result:
        print(result)
        return 0

    print(format_plan(create_task_plan(text, root=ROOT)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
