"""Tests del guardia que evita filtrar tool-call JSON crudo al usuario."""

from __future__ import annotations

from crotolamo.core.hooks import strip_leaked_tool_json


def test_leaked_tool_call_object_is_replaced():
    leaked = '{"name": "Lista los proyectos", "parameters": {}}'
    out = strip_leaked_tool_json(leaked)
    assert "{" not in out
    assert "patrón" in out


def test_leaked_with_arguments_key():
    leaked = '{"name": "ram_usage", "arguments": {"x": 1}}'
    out = strip_leaked_tool_json(leaked)
    assert out != leaked
    assert "{" not in out


def test_leaked_list_of_tool_calls():
    leaked = '[{"tool": "disk_usage", "args": {}}]'
    out = strip_leaked_tool_json(leaked)
    assert "[" not in out


def test_normal_text_untouched():
    normal = "RAM: 8.8G usados de 15.3G (57%), patrón."
    assert strip_leaked_tool_json(normal) == normal


def test_text_mentioning_braces_untouched():
    # Texto normal que menciona llaves pero NO es JSON puro: no se toca.
    normal = "Usa las llaves { } para los diccionarios en Python, patrón."
    assert strip_leaked_tool_json(normal) == normal


def test_plain_json_data_not_a_tool_call_untouched():
    # JSON válido pero SIN forma de tool-call (sin name+parameters): se respeta.
    data = '{"ram": "8.8G", "libre": "6.7G"}'
    assert strip_leaked_tool_json(data) == data


def test_empty_passthrough():
    assert strip_leaked_tool_json("") == ""
