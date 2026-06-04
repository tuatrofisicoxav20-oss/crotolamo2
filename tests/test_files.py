from crotolamo.tools import files


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
