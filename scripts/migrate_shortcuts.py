"""Importa el learned_commands.json viejo de Crotolamo 1 a la SQLite de C2.

Uso:
    python scripts/migrate_shortcuts.py [ruta_al_learned_commands.json]

Por defecto busca ~/Documentos/chapi_assistant/config/learned_commands.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from crotolamo.persistence import db
    from crotolamo.tools.desktop import normalize_key
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crotolamo.persistence import db
    from crotolamo.tools.desktop import normalize_key

_DEFAULT = Path.home() / "Documentos" / "chapi_assistant" / "config" / "learned_commands.json"


def migrate(source: Path) -> int:
    if not source.exists():
        print(f"No encontré el archivo viejo, patrón: {source}")
        return 1
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"No pude leer {source}, patrón: {error}")
        return 1

    count = 0
    for alias, action in data.items():
        if not isinstance(action, dict):
            continue
        action_type = action.get("type", "search")
        payload = {k: v for k, v in action.items() if k != "type"}
        db.save_shortcut(normalize_key(alias), action_type, payload)
        count += 1

    print(f"Migré {count} atajo(s) a la SQLite, patrón.")
    return 0


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    return migrate(source)


if __name__ == "__main__":
    raise SystemExit(main())
