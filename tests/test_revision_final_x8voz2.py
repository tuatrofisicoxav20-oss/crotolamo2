"""Revisión final, unidad x8voz2: la voz de punta a punta, segunda pasada.

Cada bloque fija un hallazgo confirmado; todo con fakes, sin micrófono ni audio.

- Una lista numerada que empieza en 1 y cuyo primer elemento empieza por una
  cifra ("1. 2 llamadas al banco", "1. 9:00 reunión con el equipo") no se
  reconocía como lista: la regla del decimal partido ("40. 5 GB") se aplicaba
  antes de anotar que ese "1." la abría, y cada marcador siguiente cuyo elemento
  empieza por letra pasaba por número suelto: se cortaba y se decía, pegado al
  final del elemento anterior ("... 2 llamadas al banco 2.", "Comprar leche 3.").
"""

from __future__ import annotations

import queue
import random
import re
import threading
from types import SimpleNamespace

import pytest

from crotolamo.voice.state import Mode, SharedState
from tests.test_revision_final_x1voz import (
    _frases_en_streaming,
    _oido,
    _oido_de_todas_las_formas,
    _wait,
)

# --- una lista que empieza en 1 con una cifra en su primer elemento ---------------
_PENDIENTES = "Tus pendientes:\n1. 2 llamadas al banco\n2. Comprar leche\n3. Pagar la luz"
_AGENDA = ("Tu agenda de hoy:\n1. 9:00 reunión con el equipo\n2. 11:30 dentista\n"
           "3. Comida con Ana\n4. 16:00 gimnasio")


@pytest.mark.parametrize("texto, esperado", [
    (_PENDIENTES, ["Tus pendientes: 2 llamadas al banco Comprar leche Pagar la luz"]),
    (_AGENDA, ["Tu agenda de hoy: 9:00 reunión con el equipo 11:30 dentista Comida con Ana "
               "16:00 gimnasio"]),
    ("1. 2 tazas de harina\n2. Sal\n3. Azúcar", ["2 tazas de harina Sal Azúcar"]),
    ("1. 500 gramos de harina.\n2. Sal.\n3. Agua tibia.",
     ["500 gramos de harina.", "Sal.", "Agua tibia."]),
    # Con sangría (el salto seguido de sangría corta: una pieza por línea), tras
    # una línea en blanco y con la numeración saltando.
    ("Claro, patrón:\n\n  1. 3 huevos\n  2. Harina\n  12. Leche",
     ["Claro, patrón:", "3 huevos", "Harina", "Leche"]),
    # El elemento que empieza por cifra cierra frase a media línea: la lista sigue.
    ("1. 2 llamadas al banco. Son cortas.\n2. Comprar leche.\n3. Pagar la luz.",
     ["2 llamadas al banco.", "Son cortas.", "Comprar leche.", "Pagar la luz."]),
])
def test_una_lista_cuyo_primer_elemento_empieza_por_cifra_no_dice_sus_numeros(texto, esperado):
    """El "1." abre la lista aunque le siga una cifra: los marcadores siguientes
    son viñetas y no se dicen. Se oía "... 2 llamadas al banco 2.", "Comprar
    leche 3.", "Pagar la luz"."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


_CON_LETRA = ["Comprar leche", "Pagar la luz", "Llamar a Ana", "Abre Spotify", "Sal", "Harina",
              "Comida con Ana", "Reinicia el equipo"]
_CON_CIFRA = ["2 llamadas al banco", "9:00 reunión con el equipo", "500 gramos de harina",
              "3 huevos", "11:30 dentista", "16:00 gimnasio", "200 g de azúcar",
              "2 tazas de harina"]
_ENTRADAS = ["", "Tus pendientes:\n", "Claro, patrón:\n\n", "Tu agenda de hoy:\n", "Va, patrón.\n"]
# Un marcador dicho: una pieza que es, o acaba en, un "N." suelto.
_MARCADOR_DICHO = re.compile(r"(?:^| )\d{1,2}\.$")


def test_ninguna_lista_que_empieza_en_1_dice_sus_numeros_empiece_como_empiece_cada_elemento():
    """La misma regla para todas: que un elemento empiece por cifra o por letra
    (el primero o cualquiera) no cambia que sus números son viñetas. Se oyen los
    elementos, en orden y sin número, igual con la respuesta entera que en
    streaming, llegue como llegue."""
    rng = random.Random(20261120)
    for _ in range(400):
        elementos = [rng.choice(_CON_CIFRA + _CON_LETRA) + rng.choice(["", "", "."])
                     for _ in range(rng.randint(2, 6))]
        entrada = rng.choice(_ENTRADAS)
        texto = entrada + "\n".join(f"{n}. {e}" for n, e in enumerate(elementos, 1))
        palabras = " ".join([*entrada.split(), *elementos])
        for oido in _oido_de_todas_las_formas(texto) + [
                _oido(_frases_en_streaming(texto, rng.randint(2, 9)))]:
            assert " ".join(oido) == palabras, texto
            assert not any(_MARCADOR_DICHO.search(pieza) for pieza in oido), (texto, oido)


@pytest.mark.parametrize("texto, frases", [
    # Sin lista a la vista, un número que no es 1 seguido de una cifra es un
    # decimal partido: ni corta ni abre lista (el "2." que sigue SÍ se dice).
    ("40. 5 GB libres, patrón.\n2. ¿Algo más?", ["40. 5 GB libres, patrón.", "2.", "¿Algo más?"]),
    # A media línea tampoco hay marcador: el decimal sigue de corrido.
    ("¿Cuánto queda? 40. 5 GB libres. Listo.", ["¿Cuánto queda?", "40. 5 GB libres.", "Listo."]),
    ("Listo, patrón. 1. 5 GB libres.\n2. ¿Algo más?",
     ["Listo, patrón.", "1. 5 GB libres.", "2.", "¿Algo más?"]),
])
def test_un_numero_suelto_seguido_de_cifra_sigue_sin_cortar_y_sin_abrir_lista(texto, frases):
    """La otra cara: la regla del decimal partido sigue mandando en un número
    SUELTO (sin lista a la vista, o a media línea), y ese número no abre lista."""
    from crotolamo.voice.tts import split_sentences

    assert split_sentences(texto) == frases
    for paso in (1, 3, 7, len(texto)):
        assert _frases_en_streaming(texto, paso) == frases, paso


def test_la_sonda_de_sentence_no_da_por_cerrada_la_lista_que_abre_con_cifra():
    """stream_release="sentence" usa la misma memoria de lista: con "1. 2 tazas"
    el "2." siguiente pasaba por frase cerrada ("... harina 2.") y la respuesta
    se soltaba a media lista, como si ya hubiera una primera frase dicha."""
    from crotolamo.core.streaming import LiveStreamer

    def soltado_en(texto: str) -> str:
        s = LiveStreamer(lambda t: None, hold_until_done=True, release="sentence")
        for i, c in enumerate(texto):
            if s.feed(c):
                return texto[:i + 1]
        return ""

    assert soltado_en("1. 2 tazas de harina\n2. Sal\n3. Azúcar\n") == ""
    assert soltado_en(_PENDIENTES + "\n") == ""
    # Con punto final el elemento SÍ cierra frase, como en cualquier lista.
    assert soltado_en("1. 2 tazas de harina.\n2. Sal.") == "1. 2 tazas de harina.\n2"


def test_la_lista_que_abre_con_cifra_llega_sin_numeros_a_piper_en_los_dos_loops(tmp_path,
                                                                             monkeypatch):
    """De punta a punta con el TTS de verdad (motor falso): lo que se le pide
    sintetizar a Piper en el loop simple (StreamSpeaker, respuesta soltada de
    golpe o en vivo, y speak_sentences) y en el concurrente (Cerebro + Boca)."""
    from crotolamo.voice.threads import BrainThread, MouthThread
    from crotolamo.voice.tts import TTS, StreamSpeaker

    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    sintetizado: list[str] = []
    monkeypatch.setattr(tts, "_speak_streaming",
                        lambda text, should_abort=None: sintetizado.append(text) or True)
    respuesta = "Va, patrón.\n" + _PENDIENTES
    esperado = ["Va, patrón.", "Tus pendientes: 2 llamadas al banco Comprar leche Pagar la luz"]

    for trozos in ([respuesta], list(respuesta)):
        sintetizado.clear()
        speaker = StreamSpeaker(tts)
        for trozo in trozos:
            speaker.feed(trozo)
        assert speaker.finish() is True
        assert sintetizado == esperado

    sintetizado.clear()
    tts.speak_sentences(respuesta)
    assert sintetizado == esperado

    for stream in (False, True):
        sintetizado.clear()
        state = SharedState()
        cmd_q: queue.Queue = queue.Queue()
        tts_q: queue.Queue = queue.Queue()
        shutdown = threading.Event()

        def handle_turn(command, on_token=None):
            if on_token is not None:
                for c in respuesta:
                    on_token(c)
            return respuesta

        hilos = [BrainThread(SimpleNamespace(handle_turn=handle_turn), cmd_q, tts_q, state,
                             shutdown, stream=stream),
                 MouthThread(tts, tts_q, state, shutdown)]
        for hilo in hilos:
            hilo.start()
        try:
            cmd_q.put(("qué tengo pendiente", state.new_turn(mode=Mode.THINKING)))
            assert _wait(lambda: state.get_mode() is Mode.IDLE and len(sintetizado) >= 2)
        finally:
            shutdown.set()
            for hilo in hilos:
                hilo.join(2)
        assert sintetizado == esperado, stream
