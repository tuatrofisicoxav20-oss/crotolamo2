from crotolamo.persistence import db, facts


def test_add_and_recall(tmp_path):
    dbp = tmp_path / "m.sqlite"
    db.add_fact("mi proyecto principal es Huevonitis 4", "proyectos", db_path=dbp)
    rows = db.get_facts(db_path=dbp)
    assert len(rows) == 1
    assert "Huevonitis 4" in rows[0]["texto"]
    assert rows[0]["categoria"] == "proyectos"


def test_recall_filters_by_category(tmp_path):
    dbp = tmp_path / "m.sqlite"
    db.add_fact("uso Fedora", "sistema", db_path=dbp)
    db.add_fact("me gusta el café", "preferencias", db_path=dbp)
    assert len(db.get_facts("sistema", db_path=dbp)) == 1
    assert len(db.get_facts(db_path=dbp)) == 2


def test_delete_fact(tmp_path):
    dbp = tmp_path / "m.sqlite"
    fid = db.add_fact("dato temporal", db_path=dbp)
    assert db.delete_fact(fid, db_path=dbp) is True
    assert db.get_facts(db_path=dbp) == []


def test_facts_context_formats(tmp_path):
    dbp = tmp_path / "m.sqlite"
    db.add_fact("X", "proyectos", db_path=dbp)
    ctx = facts.facts_context(db_path=dbp)
    assert "(proyectos) X" in ctx


def test_detect_remember_variants():
    assert facts.detect_remember("acuérdate que mi laptop es Fedora") == "mi laptop es Fedora"
    assert facts.detect_remember("recuerda de comprar pan") == "comprar pan"
    assert facts.detect_remember("anota que el wifi es lento") == "el wifi es lento"


def test_detect_remember_ignores_normal_text():
    assert facts.detect_remember("abre youtube") is None
    assert facts.detect_remember("¿qué hora es?") is None


# --- T5: schema una sola vez por ruta + WAL + busy_timeout ---

def test_schema_solo_se_ejecuta_una_vez_por_ruta(tmp_path, monkeypatch):
    """Tras la primera operación, connect() no re-ejecuta el CREATE TABLE:
    se sabotea _SCHEMA y la segunda operación debe seguir funcionando. Una
    ruta NUEVA sí lo ejecuta (y revienta con el schema saboteado)."""
    import sqlite3

    import pytest

    from crotolamo.persistence import db

    dbp = tmp_path / "memoria.sqlite"
    db.add_fact("uno", db_path=dbp)
    monkeypatch.setattr(db, "_SCHEMA", "ESTO NO ES SQL;")
    db.add_fact("dos", db_path=dbp)  # no revienta: el schema ya no se re-corre
    assert len(db.get_facts(db_path=dbp)) == 2

    otra = tmp_path / "otra.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        db.add_fact("tres", db_path=otra)  # ruta nueva SÍ ejecuta el schema


def test_wal_y_busy_timeout_activos(tmp_path):
    """WAL deja leer mientras se escribe; busy_timeout espera en vez de
    reventar con 'database is locked' (shell + listener sobre el mismo fichero)."""
    from crotolamo.persistence import db

    with db.connect(tmp_path / "wal.sqlite") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 3000
