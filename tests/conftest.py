"""Fixtures compartidos.

fake_project crea un proyecto temporal y hace que get_settings() lo vea, para que
los tests de projects/find_in_project NO dependan de que existan los proyectos
reales en el disco de Emiliano (bug C2 de la auditoría).
"""

import pytest

from crotolamo import settings as settings_mod


@pytest.fixture(autouse=True, scope="session")
def _sin_config_local(tmp_path_factory):
    """La suite corre contra la config VERSIONADA, nunca contra
    config/crotolamo.local.toml del dueño: con ese archivo presente, get_settings()
    lo fusionaba y los tests veían otros valores (p. ej. use_oww=false) que en
    cualquier otra máquina. _SETTINGS se vacía por si algo cargó la config al
    importar; los tests del merge apuntan LOCAL_CONFIG a su propio archivo."""
    real_local, real_settings = settings_mod.LOCAL_CONFIG, settings_mod._SETTINGS
    settings_mod.LOCAL_CONFIG = (
        tmp_path_factory.mktemp("sin_config_local") / "crotolamo.local.toml")
    settings_mod._SETTINGS = None
    yield
    settings_mod.LOCAL_CONFIG, settings_mod._SETTINGS = real_local, real_settings


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
