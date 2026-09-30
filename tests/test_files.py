from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import default_registry, files


@pytest.fixture(autouse=True)
def allow_tmp(tmp_path, monkeypatch):
    """Permite operar dentro de tmp_path: las tools de archivo revalidan la
    allowlist (M2), así que sin esto rechazarían /tmp por estar fuera del corral.
    """
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [tmp_path])
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)


def test_write_then_read(tmp_path):
    target = tmp_path / "sub" / "nota.md"
    out = files.write_file(str(target), "hola patrón")
    assert "Escribí" in out
    assert files.read_file(str(target)) == "hola patrón"


def test_read_missing(tmp_path):
    assert "No existe" in files.read_file(str(tmp_path / "nope.txt"))


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "carpeta").mkdir()
    out = files.list_dir(str(tmp_path))
    assert "a.txt" in out and "carpeta" in out


def test_make_dir(tmp_path):
    d = tmp_path / "nueva" / "honda"
    files.make_dir(str(d))
    assert d.is_dir()


def test_move_file(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    dest = tmp_path / "b.txt"
    files.move_file(str(src), str(dest))
    assert dest.exists() and not src.exists()


def test_delete_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    files.delete_file(str(f))
    assert not f.exists()


def test_destructive_tools_marked_unsafe():
    # El guard se apoya en este flag para pedir confirmación.
    assert files.delete_file._crotolamo_tool.safe is False
    assert files.move_file._crotolamo_tool.safe is False
    # M6: sobrescribir archivos también es destructivo -> confirma siempre.
    assert files.write_file._crotolamo_tool.safe is False


# --- defensa en profundidad (M2): la tool rechaza por sí sola, sin guard ---
def test_write_outside_corral_rejected_directly():
    # allow_tmp limita la allowlist a tmp_path; /etc queda fuera del corral.
    out = files.write_file("/etc/se_cuela.txt", "x")
    assert "corral" in out.lower() or "permitidas" in out.lower()
    assert not __import__("pathlib").Path("/etc/se_cuela.txt").exists()


def test_read_outside_corral_rejected_directly():
    out = files.read_file("/etc/passwd")
    assert "corral" in out.lower() or "permitidas" in out.lower()


def test_delete_outside_corral_rejected_directly():
    out = files.delete_file("/etc/hosts")
    assert "corral" in out.lower() or "permitidas" in out.lower()


# --- corral ampliado (M6): la tool acepta también la zona de confirmación ---
def test_write_inside_confirm_roots_accepted(tmp_path, monkeypatch):
    # allowed = libre/, confirm = confirmable/. La confirmación ya ocurrió en el
    # guard: la tool no debe re-bloquear una ruta de la zona de confirmación.
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [tmp_path / "libre"])
    monkeypatch.setattr(real, "confirm_roots", [tmp_path / "confirmable"])
    out = files.write_file(str(tmp_path / "confirmable" / "x.txt"), "hola")
    assert "Escribí" in out


def test_outside_both_zones_still_rejected(tmp_path, monkeypatch):
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [tmp_path / "libre"])
    monkeypatch.setattr(real, "confirm_roots", [tmp_path / "confirmable"])
    out = files.write_file(str(tmp_path / "otra" / "x.txt"), "hola")
    assert "corral" in out.lower() or "permitidas" in out.lower()


# --- errores del sistema de archivos: en personaje, nunca "reventó" ---

def test_list_dir_sin_permisos_responde_en_personaje(tmp_path, monkeypatch):
    def sin_permiso(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", sin_permiso)
    out = default_registry().run("list_dir", {"path": str(tmp_path)})
    assert "reventó" not in out
    assert "No pude listar" in out and "patrón" in out


# --- TOCTOU: las tools operan sobre la MISMA ruta resuelta que pasó el corral ---

def test_resolve_devuelve_la_ruta_canonica(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("x")
    link = tmp_path / "enlace.txt"
    link.symlink_to(real)
    assert files._resolve(str(link)) == real.resolve()


def test_ruta_irresoluble_no_revienta_y_queda_fuera_del_corral():
    # Un '\0' hace que resolve() lance ValueError: vuelve tal cual y el corral la rechaza.
    out = files.read_file("a\0b")
    assert "corral" in out.lower() or "permitidas" in out.lower()


def test_write_a_traves_de_enlace_opera_sobre_la_ruta_canonica(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "enlace"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    out = files.write_file(str(link_dir / "n.txt"), "hola")
    assert (real_dir / "n.txt").read_text() == "hola"
    assert str((real_dir / "n.txt").resolve()) in out  # la tool habla de la ruta canónica
