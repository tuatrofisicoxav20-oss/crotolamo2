from crotolamo.persistence import db
from crotolamo.tools import shortcuts


def test_db_save_get_roundtrip(tmp_path):
    dbp = tmp_path / "t.sqlite"
    db.save_shortcut("mi musica", "url", {"value": "https://open.spotify.com"}, db_path=dbp)
    action = db.get_shortcut("mi musica", db_path=dbp)
    assert action == {"type": "url", "value": "https://open.spotify.com"}


def test_db_upsert_overwrites(tmp_path):
    dbp = tmp_path / "t.sqlite"
    db.save_shortcut("x", "url", {"value": "a"}, db_path=dbp)
    db.save_shortcut("x", "url", {"value": "b"}, db_path=dbp)
    assert db.get_shortcut("x", db_path=dbp)["value"] == "b"
    assert len(db.all_shortcuts(db_path=dbp)) == 1


def test_db_get_missing(tmp_path):
    assert db.get_shortcut("nope", db_path=tmp_path / "t.sqlite") is None


def test_classify_target_site():
    assert shortcuts._classify_target("youtube") == ("url", {"value": "https://www.youtube.com"})


def test_classify_target_folder():
    kind, payload = shortcuts._classify_target("descargas")
    assert kind == "folder" and payload == {"value": "descargas"}


def test_classify_target_free_text_is_search():
    kind, payload = shortcuts._classify_target("latin mafia")
    assert kind == "search" and payload["query"] == "latin mafia"


def test_classify_target_raw_url():
    kind, payload = shortcuts._classify_target("https://ejemplo.com")
    assert kind == "url" and payload["value"] == "https://ejemplo.com"
