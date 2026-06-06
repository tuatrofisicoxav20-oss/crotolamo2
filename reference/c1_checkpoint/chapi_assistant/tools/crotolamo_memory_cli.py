#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.local_memory import handle_memory_command, memory_summary, ensure_memory
from core.session_history import history_summary


def main() -> int:
    ensure_memory(ROOT)
    if len(sys.argv) == 1:
        print(memory_summary(ROOT))
        return 0

    text = " ".join(sys.argv[1:]).strip()
    if text in {"historial", "history"}:
        print(history_summary(ROOT))
        return 0

    result = handle_memory_command(text, ROOT)
    print(result or "Comando de memoria no reconocido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
