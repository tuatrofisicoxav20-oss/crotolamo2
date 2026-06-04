"""Tests de projects.py contra el propio repo crotolamo2 (existe de verdad)."""

from crotolamo.tools import projects


def test_read_project_file_reads_real_file():
    out = projects.read_project_file("crotolamo", "pyproject.toml")
    assert "crotolamo2" in out


def test_read_project_file_traversal_blocked():
    out = projects.read_project_file("crotolamo", "../../../etc/passwd")
    assert "corral" in out.lower() or "se sale" in out.lower()


def test_read_project_file_unknown_project():
    assert "No tengo registrado" in projects.read_project_file("inexistente", "x.txt")


def test_analyze_project_lists_python():
    out = projects.analyze_project("crotolamo")
    assert "Análisis de" in out
    assert "PY:" in out


def test_list_projects_includes_crotolamo():
    out = projects.list_projects()
    assert "crotolamo" in out.lower()


def test_list_project_tree():
    out = projects.list_project_tree("crotolamo")
    assert "crotolamo" in out.lower()
