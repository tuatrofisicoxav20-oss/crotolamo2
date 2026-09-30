"""La confirmación por voz de tools delicadas decide por PALABRA, no por subcadena.

Bug que fija: `contains_any` buscaba la variante como subcadena del texto, así
que "necesito pensarlo" confirmaba (nece-SI-to), "nueva" confirmaba (nue-VA) y
"bueno, dale" cancelaba (bue-NO). Con tools destructivas eso es peligroso.
"""

from crotolamo.voice import wake


# --- contains_any: token exacto ---

def test_subcadena_dentro_de_palabra_no_cuenta():
    assert not wake.contains_any("necesito pensarlo", ["si"])
    assert not wake.contains_any("nueva", ["va"])
    assert not wake.contains_any("bueno", ["no"])


def test_palabra_completa_si_cuenta():
    assert wake.contains_any("si", ["si"])
    assert wake.contains_any("Sí, dale.", ["sí"])  # acento y puntuación normalizados
    assert wake.contains_any("órale, confirmo", ["confirmo"])


def test_variante_de_varias_palabras_exige_secuencia_contigua():
    assert wake.contains_any("mejor no lo hagas", ["no lo hagas"])
    assert not wake.contains_any("no hagas lo que dije", ["no lo hagas"])


def test_texto_vacio_o_solo_puntuacion():
    assert not wake.contains_any("", wake.CONFIRM_VARIANTS)
    assert not wake.contains_any("...", wake.CONFIRM_VARIANTS)


# --- confirmation_from_answer: la regla completa que usa el listener ---

def test_necesito_pensarlo_no_confirma():
    assert wake.confirmation_from_answer("necesito pensarlo") is False


def test_nueva_no_confirma():
    assert wake.confirmation_from_answer("nueva") is False


def test_bueno_dale_confirma_sin_falso_cancel():
    assert wake.confirmation_from_answer("bueno, dale") is True


def test_confirmaciones_claras():
    for answer in ["Sí.", "si", "confirmo", "hazlo ya", "va que va", "correcto"]:
        assert wake.confirmation_from_answer(answer) is True, answer


def test_cancelar_gana_sobre_confirmar():
    assert wake.confirmation_from_answer("sí... no, cancela") is False
    assert wake.confirmation_from_answer("no lo hagas") is False
    assert wake.confirmation_from_answer("nel") is False


def test_sin_respuesta_clara_es_no():
    for answer in ["", "   ", "eh...", "qué dijiste"]:
        assert wake.confirmation_from_answer(answer) is False, answer
