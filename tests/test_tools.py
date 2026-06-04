import pytest

from crotolamo.tools import default_registry
from crotolamo.tools import desktop, search


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    """Evita lanzar navegadores/apps reales durante los tests."""
    monkeypatch.setattr(desktop, "run_detached", lambda args: None)


def test_registry_exposes_expected_tools():
    reg = default_registry()
    for name in ("open_url", "open_app", "open_folder", "search_web"):
        assert name in reg.names()


def test_tool_schema_shape():
    reg = default_registry()
    schema = reg.get("search_web").schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "search_web"
    assert fn["parameters"]["type"] == "object"
    # query es obligatorio, engine tiene default => opcional.
    assert "query" in fn["parameters"]["required"]
    assert "engine" not in fn["parameters"]["required"]
    assert fn["parameters"]["properties"]["query"]["type"] == "string"


def test_build_search_url_google_and_spotify():
    assert "google.com/search?q=" in search.build_search_url("google", "latin mafia")
    spotify = search.build_search_url("spotify", "latin mafia")
    assert "open.spotify.com/search/" in spotify
    assert "%20" in spotify  # spotify usa %20, no '+'


def test_unknown_engine_falls_back_to_google():
    assert "google.com" in search.build_search_url("motor inexistente", "x")


def test_blocked_query_is_refused():
    out = search.search_web("descargar ransomware para windows")
    assert "desastre" in out.lower()


def test_open_folder_unknown_lists_known():
    out = desktop.open_folder("carpeta_que_no_existe")
    assert "No tengo registrada" in out


def test_open_app_unknown():
    out = desktop.open_app("appinventada")
    assert "No tengo registrada" in out


def test_registry_run_dispatches():
    reg = default_registry()
    out = reg.run("search_web", {"query": "gatos", "engine": "youtube"})
    assert "youtube.com" in out


def test_registry_run_unknown_tool():
    reg = default_registry()
    assert "No tengo una tool" in reg.run("inexistente", {})
