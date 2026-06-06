#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.context_engine import context_summary, build_enriched_prompt
from core.config_manager import handle_config_command, config_summary


def main() -> int:
    if len(sys.argv) == 1:
        print(context_summary(ROOT))
        return 0

    text = " ".join(sys.argv[1:]).strip()
    result = handle_config_command(text, ROOT)
    if result:
        print(result)
        return 0

    if text.lower() in {"prompt", "enriched"}:
        print(build_enriched_prompt("hola, prueba de contexto", ROOT))
        return 0

    print(build_enriched_prompt(text, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
