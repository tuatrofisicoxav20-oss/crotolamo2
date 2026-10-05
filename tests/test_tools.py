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


# --- open_url: el fallback a xdg-open era código muerto -----------------------
# Popen NO falla cuando `flatpak run` no encuentra la app (arranca, escribe el
# error a DEVNULL y sale con 1): hay que preguntar ANTES con `flatpak info`.

class _Lanzador:
    """Fake de run_detached (el wrapper de Popen): guarda los argv sin lanzar nada."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append(list(args))


def _fake_flatpak_info(monkeypatch, returncode, seen=None):
    """Sustituye run_cmd (lo que corre `flatpak info`) por un fake con ese exit."""
    import subprocess

    def fake_run_cmd(args, timeout=10):
        if seen is not None:
            seen.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(desktop, "run_cmd", fake_run_cmd)


def test_open_url_sin_opera_gx_cae_a_xdg_open(monkeypatch):
    monkeypatch.setattr(desktop.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")  # hay flatpak
    seen = []
    _fake_flatpak_info(monkeypatch, returncode=1, seen=seen)  # ...pero Opera GX no está
    lanzador = _Lanzador()
    monkeypatch.setattr(desktop, "run_detached", lanzador)

    out = desktop.open_url("https://example.com")

    assert seen == [["flatpak", "info", "com.opera.opera-gx"]]
    assert lanzador.calls == [["xdg-open", "https://example.com"]]
    assert "navegador por defecto" in out and "patrón" in out


def test_open_url_con_opera_gx_usa_flatpak(monkeypatch):
    monkeypatch.setattr(desktop.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    _fake_flatpak_info(monkeypatch, returncode=0)
    lanzador = _Lanzador()
    monkeypatch.setattr(desktop, "run_detached", lanzador)

    out = desktop.open_url("https://example.com")

    assert lanzador.calls == [["flatpak", "run", "com.opera.opera-gx", "https://example.com"]]
    assert "pestaña" in out


def test_open_url_sin_flatpak_no_consulta_flatpak_info(monkeypatch):
    monkeypatch.setattr(
        desktop.shutil, "which", lambda cmd: None if cmd == "flatpak" else f"/usr/bin/{cmd}"
    )
    seen = []
    _fake_flatpak_info(monkeypatch, returncode=0, seen=seen)
    lanzador = _Lanzador()
    monkeypatch.setattr(desktop, "run_detached", lanzador)

    desktop.open_url("https://example.com")

    assert seen == []
    assert lanzador.calls == [["xdg-open", "https://example.com"]]


def test_build_registry_is_isolated():
    # m4: build_registry() da una copia; mutarla no afecta al GLOBAL_REGISTRY.
    from crotolamo.tools import GLOBAL_REGISTRY, build_registry

    reg = build_registry()
    assert set(reg.names()) == set(GLOBAL_REGISTRY.names())
    before = len(GLOBAL_REGISTRY.names())
    reg._tools.clear()
    assert len(GLOBAL_REGISTRY.names()) == before
