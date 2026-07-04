from crotolamo.voice import wake


def test_exact_wake_word():
    assert wake.is_wake_word("crotolamo")
    score, _ = wake.wake_score("crotolamo")
    assert score > 0.9


def test_fuzzy_variants_from_whisper():
    # Errores típicos de Whisper que deben seguir activando.
    for heard in ["coto y amo", "control amo", "croto lamo abre youtube"]:
        assert wake.is_wake_word(heard), heard


def test_non_wake_word_rejected():
    assert not wake.is_wake_word("abre la carpeta de descargas")


def test_strip_wake_word_leaves_command():
    assert wake.strip_wake_word("crotolamo abre youtube") == "abre youtube"
    assert wake.strip_wake_word("coto y amo busca gatos") == "busca gatos"


def test_strip_without_wake_returns_original():
    assert wake.strip_wake_word("abre youtube") == "abre youtube"


def test_threshold_is_respected():
    # 'crotolama' no es una variante literal: solo pasa por el score difuso.
    # Con el umbral por defecto entra; con 0.99 (casi exacto) no.
    assert wake.is_wake_word("crotolama")
    assert not wake.is_wake_word("crotolama", threshold=0.99)


# --- activación de corrido (wake + orden en una sola frase) ---

def test_split_solo_el_nombre_activa_sin_orden():
    assert wake.split_wake_command("crotolamo") == (True, "")
    assert wake.split_wake_command("Crotolamo.") == (True, "")


def test_split_nombre_mas_orden_activa_directo():
    ok, cmd = wake.split_wake_command("crotolamo pausa la música")
    assert ok and cmd == "pausa la música"
    ok, cmd = wake.split_wake_command("croto lamo ábreme la carpeta de descargas")
    assert ok and cmd == "ábreme la carpeta de descargas"


def test_split_frase_larga_sin_wake_no_activa():
    # el viejo guard rechazaba por longitud; el nuevo exige el wake AL INICIO
    assert wake.split_wake_command("abre la carpeta de descargas") == (False, "")
    assert wake.split_wake_command("yo te voy a amar toda la vida") == (False, "")


def test_split_wake_en_medio_no_activa():
    # anti-alucinación: si "crotolamo" no viene al inicio, no es una llamada
    assert wake.split_wake_command("y entonces le dije que crotolamo era un bot")[0] is False


def test_split_vacio_no_activa():
    assert wake.split_wake_command("") == (False, "")
    assert wake.split_wake_command("   ") == (False, "")
