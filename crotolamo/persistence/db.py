"""SQLite para Crotolamo. En Fase 3 respalda los atajos aprendidos (antes un
JSON suelto en C1). La Fase 4 añadirá facts y session_log a este mismo módulo.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shortcuts (
    alias        TEXT PRIMARY KEY,
    action_type  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _default_db_path() -> Path:
    from crotolamo.settings import get_settings

    return get_settings().paths.get("db", Path.home() / ".crotolamo" / "crotolamo.sqlite")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- atajos ---
def save_shortcut(alias: str, action_type: str, payload: dict, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO shortcuts (alias, action_type, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(alias) DO UPDATE SET action_type=excluded.action_type, "
            "payload=excluded.payload",
            (alias, action_type, json.dumps(payload, ensure_ascii=False)),
        )


def get_shortcut(alias: str, db_path: Path | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT action_type, payload FROM shortcuts WHERE alias = ?", (alias,)
        ).fetchone()
    if row is None:
        return None
    return {"type": row["action_type"], **json.loads(row["payload"])}


def all_shortcuts(db_path: Path | None = None) -> dict[str, dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT alias, action_type, payload FROM shortcuts").fetchall()
    return {r["alias"]: {"type": r["action_type"], **json.loads(r["payload"])} for r in rows}
