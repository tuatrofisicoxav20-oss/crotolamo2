#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.project_inspector import handle_project_inspector_command, inspect_project, format_report

def main() -> int:
    if len(sys.argv) == 1:
        print(format_report(inspect_project("crotolamo", ROOT)))
        return 0

    text = " ".join(sys.argv[1:]).strip()
    result = handle_project_inspector_command(text, ROOT)
    if result:
        print(result)
        return 0

    print(format_report(inspect_project(text, ROOT)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
