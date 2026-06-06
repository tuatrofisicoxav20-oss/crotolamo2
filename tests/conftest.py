"""Fixtures compartidos.

fake_project crea un proyecto temporal y hace que get_settings() lo vea, para que
los tests de projects/find_in_project NO dependan de que existan los proyectos
reales en el disco de Emiliano (bug C2 de la auditoría).
"""

import pytest

from crotolamo import settings as settings_mod


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Crea un proyecto de mentiras y registra su ruta en la config (singleton)."""
    proj = tmp_path / "crotolamo"
    (proj / "crotolamo" / "core").mkdir(parents=True)
    (proj / "crotolamo" / "core" / "agent.py").write_text(
        "class ToolAgent:\n    pass\n", encoding="utf-8"
    )
    (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    # get_settings() es singleton perezoso: parcheamos el objeto ya cargado.
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "projects", {"crotolamo": proj})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return proj
