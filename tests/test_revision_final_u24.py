"""Revisión final, unidad 24: crotolamo/tools/files.py y crotolamo/tools/home.py.

1. delete_file y move_file actuaban sobre el DESTINO de un enlace simbólico:
   "borra el acceso directo acceso_tesis" (el guard pregunta sin nombrar ruta)
   borraba tesis_final.docx y dejaba el enlace roto, que ya no se podía borrar
   ("No existe"); mover un enlace se llevaba el archivo real. Un enlace a una
   carpeta no se podía borrar ("Es una carpeta").
2. search_files casaba el nombre al pie de la letra: create_note guarda
   "Lista de compras" como Lista_de_compras.md y "busca mi nota lista de
   compras" no la encontraba; "reunión" (con la tilde que pone Whisper) no
   encontraba reunion_lunes.txt, ni "presentacion" la "Presentación final.pptx".
3. read_file devolvía 20 000 caracteres de basura de un PDF, un .docx o una
   sqlite (50-120 KB de JSON con los \\u0000 y los U+FFFD) que viajaban en
   cada petición siguiente.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import default_registry, files


@pytest.fixture
def corral(tmp_path, monkeypatch) -> Path:
    """El corral de las tools de archivo es tmp_path/Documentos (sin zona de
    confirmación); tmp_path/fuera queda fuera."""
    real = settings_mod.get_settings()
    docs = tmp_path.resolve() / "Documentos"
    docs.mkdir()
    (tmp_path / "fuera").mkdir()
    monkeypatch.setattr(real, "allowed_roots", [docs])
    monkeypatch.setattr(real, "confirm_roots", [])
    monkeypatch.setattr(real, "relative_base", None)
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return docs


# ---------------------------------------------------------------------------
# 1. delete_file / move_file sobre un enlace actúan sobre el ENLACE
# ---------------------------------------------------------------------------

def test_borrar_un_acceso_directo_quita_el_enlace_y_no_el_original(corral):
    tesis = corral / "tesis_final.docx"
    tesis.write_text("TRES AÑOS DE TRABAJO")
    acceso = corral / "acceso_tesis"
    acceso.symlink_to(tesis)

    out = default_registry().run("delete_file", {"path": str(acceso)})

    assert out == f"Borrado, patrón: {acceso}"
    assert not os.path.lexists(acceso)
    assert tesis.read_text() == "TRES AÑOS DE TRABAJO"


def test_un_enlace_roto_se_puede_borrar(corral):
    acceso = corral / "acceso_tesis"
    acceso.symlink_to(corral / "ya_no_existe.docx")

    out = files.delete_file(str(acceso))

    assert out == f"Borrado, patrón: {acceso}"
    assert not os.path.lexists(acceso)


def test_borrar_un_enlace_a_una_carpeta_quita_solo_el_enlace(corral):
    fotos = corral / "fotos"
    fotos.mkdir()
    (fotos / "a.jpg").write_text("x")
    atajo = corral / "atajo_fotos"
    atajo.symlink_to(fotos, target_is_directory=True)

    out = files.delete_file(str(atajo))

    assert out == f"Borrado, patrón: {atajo}"
    assert not os.path.lexists(atajo)
    assert (fotos / "a.jpg").read_text() == "x"


def test_una_carpeta_de_verdad_sigue_sin_borrarse(corral):
    (corral / "fotos").mkdir()
    out = files.delete_file(str(corral / "fotos"))
    assert out.startswith("Es una carpeta, patrón.")
    assert (corral / "fotos").is_dir()


def test_una_ruta_que_no_es_enlace_se_valida_como_antes(corral):
    """Sin enlace de por medio, la entrada es la ruta canónica de siempre: la
    propia raíz del corral sigue dentro (y no se borra por ser carpeta)."""
    assert files.delete_file(str(corral)).startswith("Es una carpeta, patrón.")
    (corral / "sub").mkdir()
    assert files.move_file(str(corral / "sub"), str(corral)).startswith("Moví ")


def test_borrar_un_enlace_que_apunta_fuera_solo_quita_el_enlace(corral):
    """Lo que se toca es la entrada de Documentos, no aquello a lo que apunta:
    lo de fuera del corral queda intacto."""
    ajeno = corral.parent / "fuera" / "ajeno.txt"
    ajeno.write_text("ajeno")
    enlace = corral / "enlace_fuera"
    enlace.symlink_to(ajeno)

    assert files.delete_file(str(enlace)) == f"Borrado, patrón: {enlace}"
    assert not os.path.lexists(enlace)
    assert ajeno.read_text() == "ajeno"


def test_un_enlace_que_vive_fuera_del_corral_no_se_toca(corral):
    """Al revés: un enlace FUERA del corral que apunta dentro. Antes se borraba
    el archivo de dentro; ahora la entrada está fuera y se rechaza."""
    mio = corral / "mio.txt"
    mio.write_text("mio")
    enlace = corral.parent / "fuera" / "enlace_dentro"
    enlace.symlink_to(mio)

    out = files.delete_file(str(enlace))
    assert "fuera de las zonas permitidas" in out
    assert os.path.islink(enlace) and mio.read_text() == "mio"

    out = files.move_file(str(enlace), str(corral / "respaldo.txt"))
    assert "fuera de las zonas permitidas" in out
    assert os.path.islink(enlace) and mio.read_text() == "mio"


def test_las_rutas_con_puntos_siguen_pasando_por_el_corral(corral):
    (corral / "sub").mkdir()
    (corral / "nota.txt").write_text("x")
    # '..' dentro de la ruta se resuelve: la entrada es Documentos/nota.txt.
    out = files.delete_file(str(corral / "sub" / ".." / "nota.txt"))
    assert out == f"Borrado, patrón: {corral / 'nota.txt'}"
    # Y lo que sube por encima del corral sigue fuera.
    assert "fuera de las zonas permitidas" in files.delete_file(str(corral / ".."))
    assert "fuera de las zonas permitidas" in files.delete_file(
        str(corral / "sub" / ".." / ".." / "fuera"))


def test_mover_un_enlace_mueve_el_enlace_y_no_el_archivo_real(corral):
    real = corral / "real.txt"
    real.write_text("data")
    enlace = corral / "link.txt"
    enlace.symlink_to(real)
    (corral / "respaldo").mkdir()

    out = files.move_file(str(enlace), str(corral / "respaldo"))

    movido = corral / "respaldo" / "link.txt"
    assert out == f"Moví {enlace} -> {movido}, patrón."
    assert not os.path.lexists(enlace)
    assert movido.is_symlink() and movido.read_text() == "data"
    assert real.read_text() == "data"


def test_un_enlace_roto_se_puede_mover(corral):
    enlace = corral / "roto"
    enlace.symlink_to(corral / "no_existe")
    out = files.move_file(str(enlace), str(corral / "roto_viejo"))
    assert out == f"Moví {enlace} -> {corral / 'roto_viejo'}, patrón."
    assert (corral / "roto_viejo").is_symlink() and not os.path.lexists(enlace)


# ---------------------------------------------------------------------------
# 2. search_files encuentra sin fijarse en tildes, mayúsculas ni '_' / '-'
# ---------------------------------------------------------------------------

@pytest.fixture
def documentos(corral, monkeypatch) -> Path:
    monkeypatch.setitem(settings_mod.get_settings().paths, "documentos", corral)
    monkeypatch.setitem(settings_mod.get_settings().paths, "notas", corral / "crotolamo_notas")
    return corral


@pytest.mark.parametrize("consulta", [
    "lista de compras", "Lista de compras", "LISTA_DE_COMPRAS", "lista-de-compras",
])
def test_encuentra_la_nota_que_creo_por_su_titulo_dicho(documentos, consulta):
    creada = files.create_note("Lista de compras", "leche, huevos")
    nota = documentos / "crotolamo_notas" / "Lista_de_compras.md"
    assert creada == f"Nota creada, patrón: {nota}"
    (documentos / "crotolamo_notas" / "Lista_de_tareas.md").write_text("")
    # Pasa el colador de find (l-i-st-a-d-e-c-o-mpr-a-s en orden), no el filtro.
    (documentos / "crotolamo_notas" / "Lista_de_otras_compras.md").write_text("")

    assert files.search_files(consulta) == f"Encontré esto, patrón:\n{nota}"


def test_encuentra_la_nota_con_tilde_por_su_titulo(documentos):
    files.create_note("Ideas para la reunión")
    nota = documentos / "crotolamo_notas" / "Ideas_para_la_reunión.md"
    assert files.search_files("ideas para la reunión") == f"Encontré esto, patrón:\n{nota}"
    assert files.search_files("ideas para la reunion") == f"Encontré esto, patrón:\n{nota}"


@pytest.mark.parametrize("consulta, nombre", [
    ("reunión", "reunion_lunes.txt"),
    ("reunion", "reunion_lunes.txt"),
    ("presentacion", "Presentación final.pptx"),
    ("PRESENTACIÓN FINAL", "Presentación final.pptx"),
    ("año", "Año_2025.pdf"),
    # Nombre en NFD (la tilde como carácter aparte: zips de macOS, rsync).
    ("presentación", "Presentacio\u0301n_vieja.odp"),
])
def test_las_tildes_y_las_mayusculas_no_deciden(documentos, consulta, nombre):
    (documentos / nombre).write_text("")
    (documentos / "otra_cosa.txt").write_text("")
    assert files.search_files(consulta) == f"Encontré esto, patrón:\n{documentos / nombre}"


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_las_tildes_no_dependen_del_locale_de_find(documentos, monkeypatch, locale):
    """En el locale C, find compara bytes: una clase "[oó]" no casa la 'ó' de
    dos bytes y -iname no pliega 'Ó'. El patrón de find tiene que dejarla pasar
    igual (y el filtro de Python, decidir)."""
    monkeypatch.setenv("LC_ALL", locale)
    nombres = ["PRESENTACIÓN.odp", "Presentación final.pptx", "Presentación_nfd.odp"]
    for nombre in nombres:
        (documentos / nombre).write_text("")
    (documentos / "presente.txt").write_text("")
    out = files.search_files("presentacion")
    assert sorted(out.splitlines()[1:]) == sorted(str(documentos / n) for n in nombres)


def test_un_comodin_del_modelo_sigue_funcionando(documentos):
    (documentos / "informe.pdf").write_text("")
    (documentos / "informe.txt").write_text("")
    assert files.search_files("*.pdf") == f"Encontré esto, patrón:\n{documentos / 'informe.pdf'}"


def test_lo_que_no_casa_sigue_sin_casar(documentos):
    (documentos / "reunion_lunes.txt").write_text("")
    assert files.search_files("reuniones") == "No encontré archivos con «reuniones», patrón."


def test_una_consulta_hecha_solo_de_separadores_no_lista_todo(documentos):
    (documentos / "a_b.txt").write_text("")
    assert files.search_files(" _ ") == "Dame un nombre que buscar, patrón."


# ---------------------------------------------------------------------------
# 3. read_file no devuelve la basura de un archivo binario
# ---------------------------------------------------------------------------

_PDF = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Length 5 /Filter /FlateDecode >>\nstream\n"
        + bytes(range(256)) * 200)


@pytest.mark.parametrize("nombre, contenido", [
    ("informe.pdf", _PDF),
    ("datos.sqlite", b"SQLite format 3\x00" + b"\x00" * 40_000),
    ("foto.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256)) * 100),
])
def test_read_file_no_lee_un_binario(corral, nombre, contenido):
    archivo = corral / nombre
    archivo.write_bytes(contenido)

    out = default_registry().run("read_file", {"path": str(archivo)})

    assert out == f"Eso no es texto, patrón: parece un archivo binario, no lo puedo leer: {archivo}"
    assert "\x00" not in out and "\ufffd" not in out


def test_read_file_sigue_leyendo_un_texto_en_latin1(corral):
    """Un texto en Latin-1 no lleva NUL: se lee como siempre (con U+FFFD en las
    tildes), no se toma por binario."""
    archivo = corral / "carta.txt"
    archivo.write_bytes("Querido patrón: mañana a las ocho.\n".encode("latin-1") * 50)
    assert files.read_file(str(archivo)) == archivo.read_text(encoding="utf-8", errors="replace")


def test_read_file_solo_mira_el_principio(corral):
    """La comprobación mira los primeros 8 KB, como git: un NUL perdido más
    adelante de un texto largo no lo convierte en binario."""
    archivo = corral / "log.txt"
    archivo.write_bytes(b"linea de log\n" * 1000 + b"\x00" + b"fin\n")
    out = files.read_file(str(archivo))
    assert out.startswith("linea de log\n") and out.endswith("fin\n")


def test_read_file_de_un_texto_normal_no_cambia(corral):
    archivo = corral / "nota.md"
    archivo.write_text("# Hola patrón\n\nñandú, acción.\n", encoding="utf-8")
    assert files.read_file(str(archivo)) == "# Hola patrón\n\nñandú, acción.\n"
    (corral / "vacio.txt").write_bytes(b"")
    assert files.read_file(str(corral / "vacio.txt")) == ""
