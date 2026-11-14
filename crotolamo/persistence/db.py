"""SQLite para Crotolamo. Respalda los atajos aprendidos (Fase 3, antes un JSON
suelto en C1) y la memoria persistente de hechos (Fase 4).
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shortcuts (
    alias        TEXT PRIMARY KEY,
    action_type  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    texto        TEXT NOT NULL,
    categoria    TEXT NOT NULL DEFAULT 'general',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Espera ante un fichero bloqueado (dos escritores: shell + listener sobre el
# mismo .sqlite) antes de rendirse con "database is locked". Se fija AL ABRIR:
# sqlite3.connect() ya trae 5 s de fábrica, así que el `PRAGMA busy_timeout=3000`
# que se ejecutaba en cada operación no añadía espera; la bajaba a estos mismos
# 3 s con una sentencia de más.
_BUSY_TIMEOUT_S = 3.0


def _default_db_path() -> Path:
    from crotolamo.settings import get_settings

    return get_settings().paths.get("db", Path.home() / ".crotolamo" / "crotolamo.sqlite")


# Rutas cuyo schema ya se ejecutó en ESTE proceso. Re-correr el CREATE TABLE
# IF NOT EXISTS completo en cada operación era trabajo tirado; un set por ruta
# (y no un bool) porque los tests abren DBs temporales distintas. El caché se
# ignora si el fichero ya no está (ver _db_file_present).
_SCHEMA_READY: set[Path] = set()


def _db_file_present(path: Path) -> bool:
    """False si el .sqlite no existe o está vacío (= DB nueva, sin tablas).

    Hay que mirarlo ANTES de sqlite3.connect(): el driver crea el fichero (de
    0 bytes) nada más abrirlo.
    """
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Si el fichero desapareció en caliente (borrado a mano, limpieza de
    # ~/.crotolamo...) el caché de schema miente: connect recreaba un fichero
    # vacío y toda tool de hechos/atajos respondía "no such table" hasta
    # reiniciar. Fichero ausente o vacío => el schema se vuelve a ejecutar.
    fresh = not _db_file_present(path)
    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    try:
        # WAL deja LEER mientras se escribe; queda persistido en el fichero, pero
        # re-pedirlo es idempotente y barato.
        conn.execute("PRAGMA journal_mode=WAL")
        key = path.resolve()
        if fresh or key not in _SCHEMA_READY:
            conn.executescript(_SCHEMA)
            _SCHEMA_READY.add(key)
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


# --- hechos (memoria persistente) ---
# La categoría por defecto (la del schema y la de remember_fact sin categoría),
# en la forma canónica de _clave_categoria.
_GENERAL = "general"


def add_fact(texto: str, categoria: str = "general", db_path: Path | None = None) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO facts (texto, categoria) VALUES (?, ?)", (texto, categoria)
        )
        return int(cur.lastrowid or 0)


def add_fact_unique(
    texto: str, categoria: str, clave: Callable[[str], str], db_path: Path | None = None
) -> int:
    """Guarda el hecho SIN copias y devuelve su id: las filas que dicen lo mismo
    (misma `clave`) se sustituyen por esta, que queda como la más reciente.

    Sustituir y no devolver el id de la que ya estaba: facts_context ancla los
    MÁS RECIENTES, y repetir un hecho que ya salió de esa ventana tiene que
    volver a meterlo (como hacía la copia nueva); si no, Crotolamo diría
    "Anotado" y el hecho seguiría fuera del prompt. De paso recoge las copias que
    traiga una libreta de antes (add_fact sigue siendo el INSERT crudo).

    Mirar y escribir van en UNA transacción con el cerrojo de escritura tomado
    desde el principio: shell y listener escriben el mismo fichero, y con la
    lectura fuera de ella los dos podían mirar a la vez, no ver el hecho del
    otro y dejar dos copias. La clave se compara en Python porque casefold no
    tiene equivalente en SQLite (lower() solo pliega ASCII).
    """
    buscada = clave(texto)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        filas = conn.execute("SELECT id, texto, categoria FROM facts").fetchall()
        repetidas = [f for f in filas if clave(f["texto"]) == buscada]
        conn.executemany("DELETE FROM facts WHERE id = ?", [(f["id"],) for f in repetidas])
        # "general" es lo que pone remember_fact cuando el modelo no manda
        # categoría, no una etiqueta elegida: re-guardar sin ella un hecho de
        # "preferencias" lo dejaba en "(general)" y recall_facts("preferencias")
        # ya no lo encontraba (antes de quitar copias, la fila etiquetada se
        # quedaba). Hereda la de la sustituida más reciente que tenía una; una
        # categoría explícita sigue mandando.
        if _clave_categoria(categoria) == _GENERAL:
            etiquetadas = [f for f in repetidas if _clave_categoria(f["categoria"]) != _GENERAL]
            if etiquetadas:
                categoria = max(etiquetadas, key=lambda f: f["id"])["categoria"]
        cur = conn.execute(
            "INSERT INTO facts (texto, categoria) VALUES (?, ?)", (texto, categoria)
        )
        return int(cur.lastrowid or 0)


def _clave_categoria(categoria: str) -> str:
    """Forma canónica de una etiqueta: sin mayúsculas, acentos ni espacios de más."""
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", categoria) if unicodedata.category(c) != "Mn"
    )
    return " ".join(sin_acentos.split()).casefold()


def get_facts(categoria: str | None = None, db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, texto, categoria, created_at FROM facts ORDER BY id"
        ).fetchall()
    facts = [dict(r) for r in rows]
    if not categoria:
        return facts
    # La etiqueta la escribe el modelo en cada llamada y no siempre igual
    # ("Preferencias" al guardar, "preferencias" al consultar): con `categoria = ?`
    # exacto la consulta decía que no había hechos habiéndolos. Se compara en
    # Python porque NOCASE y lower() de SQLite solo pliegan ASCII ("MÚSICA" no
    # casaría con "música"); sin índice sobre categoria, SQLite ya recorría toda
    # la tabla.
    clave = _clave_categoria(categoria)
    return [f for f in facts if _clave_categoria(f["categoria"]) == clave]


def delete_fact(fact_id: int, db_path: Path | None = None) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0
