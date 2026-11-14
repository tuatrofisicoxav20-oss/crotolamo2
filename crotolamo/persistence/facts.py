"""Hechos persistentes sobre el patrón. Lógica de almacenamiento + formateo.

Las tools viven en crotolamo/tools/facts.py; aquí está lo reusable (incluida la
inyección de hechos en el system prompt al arrancar la sesión).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from crotolamo.core.memoria import _clave_recuerdo
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
    # `or ""`: la categoría es opcional y el modelo la manda a veces como `null`;
    # con None.strip() la tool reventaba y el hecho no se guardaba.
    # Sin copias: "acuérdate que uso Fedora" dos veces (o el modelo volviendo a
    # guardar un hecho que ya ve anclado) dejaba dos filas que viajaban en CADA
    # request, ocupaban huecos de la ventana de facts_context y "olvida lo de
    # Fedora" borraba una y dejaba la otra viva. La clave es la de la memoria
    # semántica (Memoria._ya_guardado, `memoria migrar`): lo que allí es el mismo
    # recuerdo, aquí también.
    return db.add_fact_unique(
        texto.strip(), (categoria or "").strip() or "general", _clave_recuerdo, db_path=db_path
    )


def recall(categoria: str | None = None, db_path: Path | None = None) -> list[dict]:
    return db.get_facts(categoria, db_path=db_path)


def forget(fact_id: int, db_path: Path | None = None) -> bool:
    return db.delete_fact(fact_id, db_path=db_path)


def search(query: str, top: int | None = 5, db_path: Path | None = None) -> list[dict]:
    """Busca hechos por relevancia fuzzy (memoria semántica ligera, Fase 6).

    Si las facts crecen mucho, aquí se enchufarían embeddings reales; por ahora
    un ranking offline sin dependencias. `top=None` devuelve el ranking entero.

    OJO: es un RANKING, no un filtro. El `> 0.0` no descarta casi nada (dos
    frases cualesquiera comparten alguna letra y SequenceMatcher ya da > 0):
    quien vaya a actuar sobre el resultado debe comprobar que el hecho trata de
    lo pedido (ver tools/memoria._olvidar_sqlite).
    """
    rows = recall(db_path=db_path)
    # El score se calcula UNA vez por hecho: antes dos (al ordenar y al filtrar)
    # y SequenceMatcher es lo caro. sort() es estable: el orden no cambia.
    scored = [(_score(query, r["texto"]), r) for r in rows]
    scored.sort(key=lambda par: par[0], reverse=True)
    return [r for score, r in scored if score > 0.0][:top]


def facts_context(db_path: Path | None = None, limit: int = 30) -> str:
    """Devuelve los hechos formateados para inyectar en el system prompt.

    Van los `limit` MÁS RECIENTES, en orden cronológico. recall() ordena por id
    ascendente, así que `[:limit]` eran los más VIEJOS: a partir del hecho 31
    ningún hecho nuevo entraba al prompt y Crotolamo "olvidaba" lo recién
    aprendido.
    """
    if limit <= 0:
        return ""
    facts = recall(db_path=db_path)[-limit:]
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
