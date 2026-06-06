import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import files


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
    assert files.write_file._crotolamo_tool.safe is True


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
