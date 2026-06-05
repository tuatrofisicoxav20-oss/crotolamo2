"""Hechos persistentes sobre el patrón. Lógica de almacenamiento + formateo.

Las tools viven en crotolamo/tools/facts.py; aquí está lo reusable (incluida la
inyección de hechos en el system prompt al arrancar la sesión).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from crotolamo.persistence import db


def _score(query: str, texto: str) -> float:
    """Relevancia ligera (sin embeddings): mezcla solape de palabras + similitud."""
    q = query.lower()
    t = texto.lower()
    q_words = set(re.findall(r"\w+", q))
    t_words = set(re.findall(r"\w+", t))
    overlap = len(q_words & t_words) / len(q_words) if q_words else 0.0
    ratio = SequenceMatcher(None, q, t).ratio()
    return 0.7 * overlap + 0.3 * ratio


def remember(texto: str, categoria: str = "general", db_path: Path | None = None) -> int:
    return db.add_fact(texto.strip(), categoria.strip() or "general", db_path=db_path)


def recall(categoria: str | None = None, db_path: Path | None = None) -> list[dict]:
    return db.get_facts(categoria, db_path=db_path)


def forget(fact_id: int, db_path: Path | None = None) -> bool:
    return db.delete_fact(fact_id, db_path=db_path)


def search(query: str, top: int = 5, db_path: Path | None = None) -> list[dict]:
    """Busca hechos por relevancia fuzzy (memoria semántica ligera, Fase 6).

    Si las facts crecen mucho, aquí se enchufarían embeddings reales; por ahora
    un ranking offline sin dependencias.
    """
    rows = recall(db_path=db_path)
    ranked = sorted(rows, key=lambda r: _score(query, r["texto"]), reverse=True)
    return [r for r in ranked if _score(query, r["texto"]) > 0.0][:top]


def facts_context(db_path: Path | None = None, limit: int = 30) -> str:
    """Devuelve los hechos formateados para inyectar en el system prompt."""
    facts = recall(db_path=db_path)[:limit]
    if not facts:
        return ""
    lines = [f"- ({f['categoria']}) {f['texto']}" for f in facts]
    return "\n".join(lines)


# Detección ligera de "acuérdate que ...": permite que el agente o la interfaz
# guarden un hecho sin depender del tool-calling del modelo.
_REMEMBER_RE = re.compile(
    r"^\s*(?:acu[eé]rdate|recuerda|anota|ten en cuenta)\s+(?:de\s+|que\s+)?(.+)$",
    re.IGNORECASE,
)


def detect_remember(text: str) -> str | None:
    """Si el texto pide recordar algo, devuelve el hecho a guardar; si no, None."""
    match = _REMEMBER_RE.match(text.strip())
    if not match:
        return None
    fact = match.group(1).strip(" .")
    return fact or None
