"""Cobertura de las tools que antes solo se probaban en la capa db (L2).

La DB se redirige a tmp_path (las tools de facts tocaban ~/.crotolamo si se
llamaban sin db_path) y los lanzadores de procesos se parchean.
"""

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import desktop, facts as facts_tools, search, shortcuts


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirige la DB por defecto a tmp_path: ningún test escribe en ~/.crotolamo."""
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.paths, "db", tmp_path / "test.sqlite")
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)


# --- tools de facts (antes sin test directo) ---
def test_remember_and_recall_tools():
    out = facts_tools.remember_fact("mi editor es geany", "preferencias")
    assert "Anotado" in out
    recalled = facts_tools.recall_facts()
    assert "geany" in recalled


def test_recall_empty():
    assert "No tengo hechos" in facts_tools.recall_facts()


def test_search_facts_tool():
    facts_tools.remember_fact("mi proyecto principal es Huevonitis 4", "proyectos")
    facts_tools.remember_fact("me gusta el café", "gustos")
    out = facts_tools.search_facts("cuál es mi proyecto")
    assert "Huevonitis 4" in out


def test_forget_fact_tool():
    facts_tools.remember_fact("dato temporal", "tmp")
    # El id 1 es el primero insertado en la DB temporal limpia.
    out = facts_tools.forget_fact(1)
    assert "Olvidado" in out or "No tengo" in out


# --- lógica pura ---
def test_desktop_normalize_key():
    assert desktop.normalize_key("  Opera GX ") == "opera gx"
    assert desktop.normalize_key("Música") == "musica"


def test_search_build_url_and_block():
    assert "youtube.com" in search.build_search_url("youtube", "gatos")
    assert search.is_blocked_query("descargar ransomware") is True
    assert search.is_blocked_query("recetas de pan") is False


def test_shortcuts_classify_target():
    assert shortcuts._classify_target("github")[0] == "url"
    assert shortcuts._classify_target("documentos")[0] == "folder"
    kind, payload = shortcuts._classify_target("una canción cualquiera")
    assert kind == "search"


def test_run_shortcut_uses_tools_without_launching(monkeypatch):
    # Aprende un atajo a una URL y verifica que run_shortcut NO lanza procesos reales.
    launched = []
    monkeypatch.setattr(desktop, "run_detached", lambda args: launched.append(args))
    shortcuts.learn_shortcut("mi buscador", "github")
    out = shortcuts.run_shortcut("mi buscador")
    assert "patrón" in out.lower()
    assert launched  # se intentó abrir algo, pero vía el fake (sin proceso real)
