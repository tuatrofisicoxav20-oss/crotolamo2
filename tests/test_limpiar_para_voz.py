"""limpiar_para_voz: el TTS no debe leer "asterisco asterisco" ni URLs.

Casos pedidos: negritas, listas, títulos, links, URLs, texto normal intacto,
acentos y ñ intactos. Más: streaming por chunks (un ** partido entre dos
chunks), y que la limpieza ocurre en el ÚNICO punto por el que pasa todo lo
que se habla (TTS.speak), sin tocar el corte (stop).
"""

from __future__ import annotations

import threading
from pathlib import Path

from crotolamo.voice.clean import limpiar_para_voz
from crotolamo.voice.tts import TTS, StreamSpeaker, split_sentences


# --- markdown y formato ---

def test_negritas_y_cursivas():
    out = limpiar_para_voz("**Hola**, patrón. Esto es *importante* y __esto__ también.")
    assert out == "Hola, patrón. Esto es importante y esto también."


def test_listas_con_vinetas():
    out = limpiar_para_voz("Tienes:\n- uno\n- dos\n* tres\n• cuatro\n+ cinco")
    assert out == "Tienes: uno dos tres cuatro cinco"


def test_lista_numerada():
    out = limpiar_para_voz("1. Abre Spotify.\n2. Dale play.")
    assert out == "Abre Spotify. Dale play."


def test_titulos():
    out = limpiar_para_voz("# Título\n## Subtítulo\nTexto normal")
    assert out == "Título Subtítulo Texto normal"


def test_links_dejan_solo_el_texto():
    out = limpiar_para_voz("Mira [la guía](https://ejemplo.com/guia) ahora.")
    assert out == "Mira la guía ahora."


def test_imagen_deja_el_alt():
    assert limpiar_para_voz("![un gato](https://x.com/gato.png) bonito") == "un gato bonito"


def test_urls_sueltas_se_quitan():
    out = limpiar_para_voz("Entra a https://example.com/a?b=1&c=2 o a www.foo.com ya.")
    assert "http" not in out and "www" not in out
    assert out == "Entra a o a ya."


def test_url_al_final_conserva_el_punto_de_la_frase():
    assert limpiar_para_voz("Ve a https://x.com/ruta. Luego avísame.") == "Ve a. Luego avísame."


def test_url_entre_parentesis_no_deja_parentesis_vacios():
    assert limpiar_para_voz("Está en GitHub (https://github.com/x).") == "Está en GitHub."


def test_codigo_inline_y_bloques():
    out = limpiar_para_voz("Usa `ls -la` para ver. ```bash\nrm -rf x\n``` Listo.")
    assert out == "Usa ls -la para ver. Listo."


def test_cerca_sin_cerrar_pierde_solo_el_marcador():
    assert limpiar_para_voz("```python\nprint(1)") == "print(1)"


def test_tabla():
    out = limpiar_para_voz("| Nombre | Uso |\n|---|---|\n| RAM | 40% |")
    assert out == "Nombre Uso RAM 40%"


def test_emojis_fuera():
    assert limpiar_para_voz("Listo 🎉 patrón 😀👍🏽 ✅") == "Listo patrón"


def test_citas_y_restos():
    assert limpiar_para_voz("> cita\n~~tachado~~ | raro > sí") == "cita tachado raro sí"


def test_solo_formato_queda_vacio():
    for texto in ["***", "---", "```", "# ", "", "   ", "| --- | --- |"]:
        assert limpiar_para_voz(texto) == "", repr(texto)


# --- lo que NO debe cambiar ---

def test_texto_normal_intacto():
    texto = "Hola, patrón. ¿Qué tal? Todo bien: 3 archivos, 2 carpetas."
    assert limpiar_para_voz(texto) == texto


def test_acentos_y_enie_intactos():
    texto = "Mañana José irá al Ñandú con Ángela; ¡qué güey! Él dijo: «órale»."
    assert limpiar_para_voz(texto) == texto


def test_numeros_horas_y_decimales_intactos():
    texto = "Son las 3:30 y quedan 40.5 GB libres, el 15% del disco."
    assert limpiar_para_voz(texto) == texto


def test_colapsa_espacios():
    assert limpiar_para_voz("Hola   patrón,\n\n  ¿qué   tal?") == "Hola patrón, ¿qué tal?"


# --- streaming: un ** partido entre chunks no se pronuncia ni se come palabras ---

def _tts_capturando(tmp_path: Path, monkeypatch) -> tuple[TTS, list[str]]:
    """TTS real con el motor sustituido: captura lo que llegaría a sintetizar."""
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    hablado: list[str] = []

    def _fake_engine(text: str) -> bool:
        hablado.append(text)
        return True

    monkeypatch.setattr(tts, "_speak_streaming", _fake_engine)
    return tts, hablado


def test_speak_limpia_en_el_punto_unico(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    tts.speak("**Listo**, patrón 🎉. Mira [esto](https://x.com).")
    assert hablado == ["Listo, patrón. Mira esto."]


def test_speak_no_sintetiza_si_solo_habia_formato(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    tts.speak("***")
    assert hablado == []


def test_speak_sentences_limpia_cada_frase(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    tts.speak_sentences("# Resumen\n**Listo.** Quedan *dos* cosas.")
    assert hablado == ["Resumen Listo.", "Quedan dos cosas."]


def test_streaming_asteriscos_partidos_entre_chunks(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    speaker = StreamSpeaker(tts)
    for chunk in ["**Ho", "la.** ¿Có", "mo es", "tás?", " Bien, ", "*gracias*."]:
        speaker.feed(chunk)
    assert speaker.finish() is True
    assert hablado == ["Hola.", "¿Cómo estás?", "Bien, gracias."]
    assert not any("*" in frase for frase in hablado)


def test_streaming_link_partido_entre_chunks(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    speaker = StreamSpeaker(tts)
    for chunk in ["Mira [la gu", "ía](https://ejem", "plo.com/x) ahora. ", "Ya."]:
        speaker.feed(chunk)
    speaker.finish()
    assert hablado == ["Mira la guía ahora.", "Ya."]


def test_streaming_lista_numerada_no_habla_el_numero_suelto(tmp_path, monkeypatch):
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    speaker = StreamSpeaker(tts)
    for chunk in ["1. Abre Spo", "tify.\n2. Dale pl", "ay."]:
        speaker.feed(chunk)
    speaker.finish()
    assert hablado == ["Abre Spotify.", "Dale play."]


def test_split_sentences_corta_tras_cierre_de_enfasis():
    assert split_sentences("**Listo.** ¿Qué tal?") == ["**Listo.", "¿Qué tal?"]


def test_split_sentences_marcador_de_lista_no_corta_pero_numero_final_si():
    # "1." al inicio de línea es marcador de lista: se queda con su frase.
    assert split_sentences("Pasos:\n1. Abre la app. Luego cierra.") == [
        "Pasos:\n1. Abre la app.", "Luego cierra.",
    ]
    # Un número que cierra frase a media línea sigue cortando.
    assert split_sentences("¿Cuántos? 5. Cinco.") == ["¿Cuántos?", "5.", "Cinco."]


def test_stop_sigue_funcionando_con_texto_con_formato(tmp_path, monkeypatch):
    """La limpieza es texto puro antes de sintetizar: no toca la bandera de corte."""
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    tts.stop()
    assert tts._stop_flag.is_set()
    tts.speak("**hola**")
    assert hablado == ["hola"]
    assert tts._stop_flag.is_set()  # solo el motor real la limpia al arrancar


def test_speak_es_seguro_desde_varios_hilos(tmp_path, monkeypatch):
    """La función es pura: hablar desde StreamSpeaker (hilo) y say() (otro) no choca."""
    tts, hablado = _tts_capturando(tmp_path, monkeypatch)
    hilos = [threading.Thread(target=tts.speak, args=(f"**{i}** ok.",)) for i in range(8)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert sorted(hablado) == sorted(f"{i} ok." for i in range(8))
