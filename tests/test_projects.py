"""Tests de projects.py contra un proyecto temporal (fixture fake_project).

No dependen de que existan ~/Documentos/crotolamo2 ni huevonitis 4 en disco (C2).
"""

from crotolamo.tools import projects


def test_read_project_file_ok(fake_project):
    out = projects.read_project_file("crotolamo", "pyproject.toml")
    assert "name" in out


def test_read_project_file_traversal_blocked(fake_project):
    # El proyecto SÍ existe, así que la llamada llega al check de traversal.
    out = projects.read_project_file("crotolamo", "../../../etc/passwd")
    assert "corral" in out.lower() or "se sale" in out.lower()


def test_read_project_file_unknown_project(fake_project):
    assert "No tengo registrado" in projects.read_project_file("inexistente", "x.txt")


def test_analyze_project_lists_python(fake_project):
    out = projects.analyze_project("crotolamo")
    assert "Análisis de" in out
    assert "PY:" in out


def test_list_projects_includes_crotolamo(fake_project):
    out = projects.list_projects()
    assert "crotolamo" in out.lower()


def test_list_project_tree(fake_project):
    out = projects.list_project_tree("crotolamo")
    assert "crotolamo" in out.lower()
