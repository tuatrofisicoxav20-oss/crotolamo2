"""Tests de projects.py contra un proyecto temporal (fixture fake_project).

No dependen de que existan ~/Documentos/crotolamo2 ni huevonitis 4 en disco (C2).
"""

import subprocess
from pathlib import Path

from crotolamo.tools import default_registry, projects


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


# --- find_in_project: el patrón lo elige el LLM y no debe leerse como opción ---

def test_find_in_project_patron_con_guion_va_tras_e_y_la_base_tras_doble_guion(
    fake_project, monkeypatch
):
    """Con pattern="-r" y sin `-e`/`--`, grep tomaba "-r" como opción y la ruta
    del proyecto como patrón: buscaba en el CWD del proceso, fuera del corral.
    Se captura el argv sin ejecutar grep de verdad."""
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(projects.subprocess, "run", fake_run)
    out = projects.find_in_project("crotolamo", "-r")

    assert "reventó" not in out and "patrón" in out
    argv = seen[0]
    assert argv[argv.index("-e") + 1] == "-r"  # el patrón, siempre tras -e
    assert argv[-2:] == ["--", str(fake_project)]  # la base, tras -- y como único operando


def test_find_in_project_patron_con_guion_busca_dentro_del_proyecto(fake_project):
    """Con grep de verdad: un patrón que empieza por guion se busca como texto
    dentro del proyecto (antes ni lo encontraba: buscaba otra cosa en otro sitio)."""
    (fake_project / "notas.txt").write_text("usa -r para recursivo\n", encoding="utf-8")
    out = projects.find_in_project("crotolamo", "-r")
    assert "notas.txt" in out


# --- list_project_tree: carpetas sin permisos, en personaje y nunca "reventó" ---

def test_list_project_tree_sin_permisos_responde_en_personaje(fake_project, monkeypatch):
    def sin_permiso(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", sin_permiso)
    out = default_registry().run("list_project_tree", {"name": "crotolamo"})
    assert "reventó" not in out
    assert "No pude leer" in out and "patrón" in out


def test_list_project_tree_subcarpeta_sin_permisos_no_tira_el_arbol(fake_project, monkeypatch):
    real_iterdir = Path.iterdir

    def selectivo(self):
        if self.name == "crotolamo" and self.parent == fake_project:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", selectivo)
    out = projects.list_project_tree("crotolamo")
    assert "no pude leer esta carpeta" in out
    assert "pyproject.toml" in out  # el resto del árbol sigue saliendo
