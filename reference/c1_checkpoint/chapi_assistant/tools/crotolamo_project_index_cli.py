#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.project_indexer import (
    handle_project_index_command,
    index_summary,
    scan_project,
    list_known_projects,
)

def main() -> int:
    if len(sys.argv) == 1:
        print(list_known_projects(ROOT))
        print()
        print(index_summary("crotolamo", ROOT))
        return 0

    text = " ".join(sys.argv[1:]).strip()
    result = handle_project_index_command(text, ROOT)
    if result:
        print(result)
        return 0

    # fallback: treat args as project name and index it
    idx = scan_project(text, ROOT)
    print(index_summary(idx.name, ROOT))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
