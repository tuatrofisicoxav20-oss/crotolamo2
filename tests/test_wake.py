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
