"""Revisión final, unidad 17: crotolamo/persistence (db.py, facts.py).

1. facts.remember() guardaba copias exactas. Con la libreta SQLite de fábrica
   (pin_facts = true, releída en cada request), "acuérdate que uso Fedora" dos
   veces, o el modelo volviendo a guardar un hecho que ya ve anclado, metía una
   fila más por vez: el mismo hecho viajaba dos o tres veces en cada request,
   ocupaba huecos de la ventana de 30 (echando fuera hechos distintos y
   gastando num_ctx del cerebro local) y "olvida lo de Fedora" borraba UNA
   copia y dejaba otra viva. La memoria mem0 ya lo evitaba con _clave_recuerdo
   (Memoria._ya_guardado); la libreta usa ahora la misma clave. Repetir un
   hecho lo deja en UNA fila y lo pone como el más reciente, como antes lo
   ponía la copia nueva: un hecho viejo que se repite vuelve a la ventana.
"""

from __future__ import annotations

import threading

import pytest

from crotolamo.persistence import db, facts


@pytest.fixture
def dbp(tmp_path):
    return tmp_path / "libreta.sqlite"


def _textos(dbp) -> list[str]:
    return [f["texto"] for f in facts.recall(db_path=dbp)]


def test_recordar_el_mismo_hecho_otra_vez_no_lo_duplica(dbp):
    """El caso del hallazgo: tres veces lo mismo dejaban tres líneas ancladas."""
    facts.remember("uso Fedora", db_path=dbp)
    facts.remember("uso Fedora", db_path=dbp)
    facts.remember("Uso Fedora.", db_path=dbp)
    assert facts.facts_context(db_path=dbp) == "- (general) Uso Fedora."
    assert len(facts.recall(db_path=dbp)) == 1


def test_la_clave_es_la_de_la_memoria_semantica(dbp):
    """Misma regla que mem0 (_clave_recuerdo): mayúsculas, espacios de más y punto
    final no hacen un hecho distinto; otra palabra o un acento distinto, sí."""
    facts.remember("uso Fedora", db_path=dbp)
    facts.remember("  USO   fedora . ", db_path=dbp)
    facts.remember("uso Fedora con Hyprland", db_path=dbp)
    facts.remember("tomo te por las tardes", db_path=dbp)
    facts.remember("tomo té por las tardes", db_path=dbp)
    assert _textos(dbp) == ["USO   fedora .", "uso Fedora con Hyprland",
                            "tomo te por las tardes", "tomo té por las tardes"]


def test_olvidar_un_hecho_repetido_no_deja_una_copia_viva(dbp, monkeypatch):
    """Antes: "olvida lo de Fedora" contestaba "Olvidado" y la otra copia seguía
    anclada en el turno siguiente."""
    from crotolamo.tools import memoria as tools_memoria

    monkeypatch.setattr(db, "_default_db_path", lambda: dbp)
    for texto in ("uso Fedora", "uso Fedora", "Uso Fedora."):
        facts.remember(texto)
    out = tools_memoria._olvidar_sqlite("lo de Fedora")
    assert out.startswith("Olvidado, patrón: «Uso Fedora.»"), out
    assert facts.facts_context() == ""


def test_forget_fact_con_el_id_devuelto_borra_el_hecho_entero(dbp):
    facts.remember("me gusta el pozole", db_path=dbp)
    fid = facts.remember("Me gusta el pozole", db_path=dbp)
    assert facts.forget(fid, db_path=dbp)
    assert facts.recall(db_path=dbp) == []


def test_un_hecho_viejo_que_se_repite_vuelve_a_la_ventana_anclada(dbp):
    """Con más de 30 hechos, repetir uno que ya salió de la ventana lo volvía a
    anclar (con una copia de más). Sin duplicarlo debe seguir pasando: si no,
    Crotolamo diría "Anotado" y el hecho no llegaría al prompt."""
    facts.remember("uso Fedora", db_path=dbp)
    for i in range(35):
        facts.remember(f"hecho {i}", db_path=dbp)
    assert "uso Fedora" not in facts.facts_context(db_path=dbp)

    facts.remember("uso Fedora", "sistema", db_path=dbp)
    lineas = facts.facts_context(db_path=dbp).splitlines()
    assert len(lineas) == 30 and lineas[-1] == "- (sistema) uso Fedora"
    assert _textos(dbp).count("uso Fedora") == 1


def test_repetir_un_hecho_recoge_las_copias_de_una_libreta_vieja(dbp):
    """Las libretas de antes del arreglo pueden traer copias (db.add_fact sigue
    siendo el INSERT crudo): al repetir el hecho quedan en una."""
    db.add_fact("me gusta el café de olla", db_path=dbp)
    db.add_fact("uso Fedora", db_path=dbp)
    db.add_fact("Me gusta el café de olla.", db_path=dbp)
    facts.remember("me gusta el café de olla", "gustos", db_path=dbp)
    assert facts.facts_context(db_path=dbp) == (
        "- (general) uso Fedora\n- (gustos) me gusta el café de olla")


def test_la_tool_sigue_contestando_anotado(dbp, monkeypatch):
    from crotolamo.tools import facts as tools_facts

    monkeypatch.setattr(db, "_default_db_path", lambda: dbp)
    for _ in range(2):
        out = tools_facts.remember_fact("uso Fedora", "sistema")
        assert out == "Anotado, patrón. Lo recordaré: «uso Fedora»."
    assert _textos(dbp) == ["uso Fedora"]


def test_dos_procesos_que_recuerdan_lo_mismo_a_la_vez_dejan_una_fila(dbp, monkeypatch):
    """Shell y listener escriben el mismo .sqlite. Mirar y escribir van en UNA
    transacción con el cerrojo de escritura: si no, los dos miraban, ninguno
    veía el hecho del otro y quedaban dos copias."""
    from crotolamo.core import memoria as memoria_mod

    db.add_fact("otro hecho", db_path=dbp)
    clave_real = memoria_mod._clave_recuerdo
    a_mitad = threading.Event()
    seguir = threading.Event()
    primero = threading.Lock()
    primero.acquire()

    def _clave_que_se_para(texto: str) -> str:
        # La PRIMERA vez que se compara una fila (ya dentro de la consulta) el
        # hilo se queda a medias hasta que el test lo suelte.
        if texto == "otro hecho" and primero.locked():
            primero.release()
            a_mitad.set()
            assert seguir.wait(5)
        return clave_real(texto)

    monkeypatch.setattr(memoria_mod, "_clave_recuerdo", _clave_que_se_para)
    monkeypatch.setattr(facts, "_clave_recuerdo", _clave_que_se_para, raising=False)
    errores: list[BaseException] = []

    def _recordar() -> None:
        try:
            facts.remember("uso Fedora", db_path=dbp)
        except BaseException as error:  # noqa: BLE001 - se re-lanza en el hilo del test
            errores.append(error)

    uno = threading.Thread(target=_recordar)
    uno.start()
    assert a_mitad.wait(5)
    dos = threading.Thread(target=_recordar)
    dos.start()
    dos.join(0.5)  # sin el cerrojo, el segundo termina aquí y escribe su copia
    seguir.set()
    uno.join(5)
    dos.join(5)
    assert not uno.is_alive() and not dos.is_alive()
    assert errores == []
    assert _textos(dbp) == ["otro hecho", "uso Fedora"]
