"""Revisión final, unidad 11: crotolamo/core/router.py (y persona.py).

- `_named_first` no promovía nunca la tool nombrada de un server MCP con nombres
  camelCase (`listIssues`, `searchIssues`): el peso solo convertía "_" en
  espacios, así que en "mcp linear listissues" la keyword "issues" no empezaba
  palabra, el peso era 0 y el tope cortaba justo las tools que la frase nombraba.
"""

from __future__ import annotations

from typing import Any

import pytest

from crotolamo.core import router
from crotolamo.mcp import bridge


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Ningún grupo dinámico sobrevive de un test a otro; el round-robin global
    vuelve a donde estaba."""
    for name in router.dynamic_groups():
        router.unregister_group(name)
    monkeypatch.setattr(router, "_RR_COUNTER", 0)
    yield
    for name in router.dynamic_groups():
        router.unregister_group(name)


# Un server real de ese estilo (13 tools, más que max_tools=8), en el orden de
# tools/list: las que la frase nombra quedan al final.
_LINEAR_TOOLS: list[dict[str, Any]] = [
    {"name": "getIssue", "description": "Get one issue by id"},
    {"name": "createIssue", "description": "Create a new issue"},
    {"name": "updateIssue", "description": "Update an issue"},
    {"name": "getProject", "description": "Get one project"},
    {"name": "listProjects", "description": "List projects"},
    {"name": "getUser", "description": "Get one user"},
    {"name": "listUsers", "description": "List users"},
    {"name": "getTeam", "description": "Get one team"},
    {"name": "listTeams", "description": "List the teams of the workspace"},
    {"name": "listComments", "description": "List the comments of an issue"},
    {"name": "addComment", "description": "Add a comment to an issue"},
    {"name": "searchIssues", "description": "Search issues by text"},
    {"name": "listIssues", "description": "List issues of a team"},
]


def _registrar_linear() -> list[str]:
    """Como lo registra el bridge: nombres saneados (conservan las mayúsculas) y
    keywords derivadas con sus propios helpers."""
    server = bridge.MCPServerConfig(name="linear", command=["x"], prefix="linear")
    taken: set[str] = set()
    names: list[str] = []
    for raw in _LINEAR_TOOLS:
        name = bridge.sanitize_tool_name(server.prefix, raw["name"], taken)
        taken.add(name)
        names.append(name)
    keywords = bridge.default_keywords(server, _LINEAR_TOOLS)
    router.register_group("mcp:linear", names, keywords)
    return names


def test_named_first_promueve_la_tool_camelcase_que_la_frase_nombra():
    names = _registrar_linear()
    assert "mcp_linear_listIssues" in names  # el saneado conserva el camelCase

    issues = router.select_tool_names("lista los issues de linear", 8)
    assert len(issues) == 8
    assert {"mcp_linear_listIssues", "mcp_linear_searchIssues"} <= set(issues[:2])

    busca = router.select_tool_names("busca issues en linear", 8)
    assert {"mcp_linear_listIssues", "mcp_linear_searchIssues"} <= set(busca[:2])

    comments = router.select_tool_names("linear comments", 8)
    assert {"mcp_linear_listComments", "mcp_linear_addComment"} <= set(comments[:2])

    teams = router.select_tool_names("linear teams", 8)
    assert teams[0] == "mcp_linear_listTeams"


def test_named_first_camelcase_sin_palabra_propia_conserva_el_orden():
    """Solo el nombre del server: orden de tools/list, como siempre."""
    names = _registrar_linear()
    assert router.select_tool_names("pregúntale a linear", 8) == names[:8]


def test_named_first_camelcase_nombre_pegado_sigue_casando():
    """Quien teclea el nombre de la tool tal cual ("listIssues" -> keyword derivada
    "listissues") la sigue teniendo primero: el corte camelCase se SUMA a la forma
    de siempre, no la reemplaza."""
    _registrar_linear()
    nombres = router.select_tool_names("linear listIssues", 8)
    assert nombres[0] == "mcp_linear_listIssues"


def test_named_first_camelcase_con_sigla():
    """Una sigla en mayúsculas ("HTTPResponse") también corta antes de la palabra
    que sigue."""
    tools = [f"mcp_web_tool{i}" for i in range(9)] + ["mcp_web_getHTTPResponse"]
    router.register_group("mcp:web", tools, ["web", "response"])
    assert router.select_tool_names("web response", 8)[0] == "mcp_web_getHTTPResponse"
