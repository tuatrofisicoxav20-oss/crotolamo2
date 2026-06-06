"""
Historial local de sesiones para Crotolamo v7.
Guarda JSONL sencillo, sin ponerse a inventar una infraestructura bancaria.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def history_dir(root: Path | None = None) -> Path:
    root = root or _root()
    return root / "data" / "history"


def today_file(root: Path | None = None) -> Path:
    d = history_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"session_{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def log_event(kind: str, content: Any, root: Path | None = None, extra: dict[str, Any] | None = None) -> None:
    path = today_file(root)
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "content": content,
        "extra": extra or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def recent_events(root: Path | None = None, limit: int = 30) -> list[dict[str, Any]]:
    d = history_dir(root)
    if not d.exists():
        return []
    files = sorted(d.glob("session_*.jsonl"), reverse=True)
    events: list[dict[str, Any]] = []
    for file in files[:7]:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            try:
                events.append(json.loads(line))
            except Exception:
                continue
            if len(events) >= limit:
                return events
    return events


def history_summary(root: Path | None = None, limit: int = 15) -> str:
    events = recent_events(root, limit)
    if not events:
        return "No hay historial todavía."
    lines = ["Historial reciente:"]
    for e in events:
        ts = e.get("ts", "?")
        kind = e.get("kind", "?")
        content = str(e.get("content", "")).replace("\n", " ")
        if len(content) > 140:
            content = content[:137] + "..."
        lines.append(f"- {ts} [{kind}] {content}")
    return "\n".join(lines)


def handle_history_command(text: str, root: Path | None = None) -> str | None:
    low = (text or "").strip().lower()
    if low in {"historial", "ver historial", "history", "sesión", "sesion"}:
        return history_summary(root)
    return None
