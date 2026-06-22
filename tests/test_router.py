"""Tests del tool router (determinista, sin LLM)."""

from __future__ import annotations

from crotolamo.core.router import (
    route_schemas,
    select_tool_names,
)
from crotolamo.tools import default_registry


def test_system_query_routes_to_system_tools():
    names = select_tool_names("cuanta RAM estoy usando?")
    assert "ram_usage" in names
    assert "music_now" not in names


def test_music_query_routes_to_media():
    names = select_tool_names("pon la siguiente cancion")
    assert set(names) == {"music_now", "music_control"}


def test_open_app_routes_to_desktop():
    names = select_tool_names("abre la calculadora")
    assert "open_app" in names


def test_greeting_routes_to_no_tools():
    # Saludo/charla: sin tools (turno rápido, sin alucinar). "crotolamo" en el
    # saludo NO debe disparar el grupo de proyectos.
    names = select_tool_names("hola crotolamo, como estas?")
    assert names == []


def test_chat_returns_empty_schemas():
    reg = default_registry()
    assert route_schemas(reg, "jajaja que buena onda") == []


def test_accents_are_ignored():
    # Sin acentos en la consulta debe matchear igual.
    assert select_tool_names("musica") == select_tool_names("música")


def test_cap_is_respected():
    names = select_tool_names("crea una nota y lee un archivo", max_tools=3)
    assert len(names) <= 3


def test_route_schemas_returns_valid_ollama_format():
    reg = default_registry()
    schemas = route_schemas(reg, "cuanta RAM uso?")
    assert schemas, "debe devolver al menos una schema"
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_route_schemas_only_includes_existing_tools():
    reg = default_registry()
    available = set(reg.names())
    schemas = route_schemas(reg, "pon musica y abre spotify")
    for s in schemas:
        assert s["function"]["name"] in available
