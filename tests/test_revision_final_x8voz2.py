"""Revisión final, unidad x8voz2: la voz de punta a punta, segunda pasada.

Cada bloque fija un hallazgo confirmado; todo con fakes, sin micrófono ni audio.

- Una lista numerada que empieza en 1 y cuyo primer elemento empieza por una
  cifra ("1. 2 llamadas al banco", "1. 9:00 reunión con el equipo") no se
  reconocía como lista: la regla del decimal partido ("40. 5 GB") se aplicaba
  antes de anotar que ese "1." la abría, y cada marcador siguiente cuyo elemento
  empieza por letra pasaba por número suelto: se cortaba y se decía, pegado al
  final del elemento anterior ("... 2 llamadas al banco 2.", "Comprar leche 3.").
- Un número suelto que abre línea (lista retomada en "3.", cuenta atrás) se decía
  pegado al final de la línea anterior ("Te faltan estos: 3.", "Ir al banco a las
  10 5."): el corte caía solo después de él. Ahora es su propia frase.
- Loop concurrente con --barge-in: en pausa los chunks ya no llegan al VAD del
  barge-in y nada lo reseteaba al reanudar; arrancaba con la memoria de antes de
  la pausa (a media voz, puntúa alto sobre un cuarto en silencio).
"""

from __future__ import annotations

import queue
import random
import re
import threading
from types import SimpleNamespace

import pytest

from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import EarThread
from tests.test_revision_final_x1voz import (
    _frases_en_streaming,
    _oido,
    _oido_de_todas_las_formas,
    _TtsContado,
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


# --- un número suelto que abre línea no suena pegado a la línea anterior ----------
_FALTAN = ("Te faltan estos:\n3. Reinicia el equipo\n4. Ir al banco a las 10\n"
           "5. Sube el volumen a 40")
_OPCIONES = "Opciones: 1. Mandar el reporte.\n2. Sube el volumen a 40.\n3. Comprar leche."
_FALTAN_OIDO = ["Te faltan estos:", "3.", "Reinicia el equipo", "4.", "Ir al banco a las 10",
                "5.", "Sube el volumen a 40"]


@pytest.mark.parametrize("texto, esperado", [
    (_FALTAN, _FALTAN_OIDO),
    # El "1." escrito en la línea de entrada no abre lista (va a media línea): los
    # marcadores siguientes son números sueltos. Se oía "Sube el volumen a 40. 3.".
    (_OPCIONES, ["Opciones: 1.", "Mandar el reporte.", "2.", "Sube el volumen a 40.", "3.",
                 "Comprar leche."]),
    # Una cuenta atrás, y la respuesta que ES un número tras su línea de entrada.
    ("Cuenta conmigo:\n3. Tres\n2. Dos", ["Cuenta conmigo:", "3.", "Tres", "2.", "Dos"]),
    ("Son, patrón:\n42. ¿Algo más?", ["Son, patrón:", "42.", "¿Algo más?"]),
    # El salto anterior al número viene tras un punto que no cierra frase (una
    # abreviatura, una cifra) o con espacios y retorno de carro alrededor.
    ("Queda en la Av.\n12. Luego te digo.", ["Queda en la Av.", "12.", "Luego te digo."]),
    ("Ya van 2.\n3. Reinicia el equipo", ["Ya van 2.", "3.", "Reinicia el equipo"]),
    ("Te faltan estos: \r\n3. Reinicia el equipo", ["Te faltan estos:", "3.",
                                                   "Reinicia el equipo"]),
    # Tras un bloque de código (que no se lee) y con el número solo en su línea.
    ("Mira:\n```\nx = 1\n```\n3. Reinicia el equipo", ["Mira:", "3.", "Reinicia el equipo"]),
    ("Faltan estos\n3.\nReinicia el equipo", ["Faltan estos", "3.", "Reinicia el equipo"]),
])
def test_un_numero_suelto_que_abre_linea_es_su_propia_frase(texto, esperado):
    """Sin lista a la vista, el "3." que abre una línea se dice (desde 2259ba2), pero
    el corte caía solo DESPUÉS de él y un salto de línea suelto no corta: sonaba
    pegado al final de la línea anterior ("Te faltan estos: 3.", "Ir al banco a las
    10 5."). Ahora se corta también en ese salto: nada se pierde y nada se pega."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


_ACABAN_EN_CIFRA = ["Ir al banco a las 10", "Sube el volumen a 40", "Comprar 2 kilos de tortilla",
                    "Llamar al 55"]
_ENTRADAS_SUELTAS = ["", "Te faltan estos:\n", "Claro, patrón:\n\n", "Va, patrón.\n",
                     "Quedan 3 de 10\n", "Ya van 2.\n", "Sigue así: \n"]


def test_una_lista_que_no_empieza_en_1_dice_cada_numero_en_frase_aparte():
    """La regla entera sobre listas retomadas, cuentas atrás y numeraciones que
    saltan (ninguna pasa por el 1, que abriría lista): cada número y cada elemento
    es su propia frase, en orden, acabe el elemento anterior en cifra, en punto o
    en nada; con la respuesta entera y en streaming, llegue como llegue."""
    rng = random.Random(20261120)
    for _ in range(400):
        numero, salto = rng.choice([2, 3, 4, 7, 12, 30]), rng.choice([1, 1, 1, 2, 5, -1])
        entrada = rng.choice(_ENTRADAS_SUELTAS)
        lineas: list[str] = []
        esperado = [entrada.strip()] if entrada else []
        for _ in range(rng.randint(1, 6)):
            if not 2 <= numero <= 99:
                break
            elemento = rng.choice(_CON_LETRA + _ACABAN_EN_CIFRA) + rng.choice(["", "", "."])
            lineas.append(f"{numero}. {elemento}")
            esperado += [f"{numero}.", elemento]
            numero += salto
        texto = entrada + "\n".join(lineas)
        for oido in _oido_de_todas_las_formas(texto) + [
                _oido(_frases_en_streaming(texto, rng.randint(2, 9)))]:
            assert oido == esperado, texto


@pytest.mark.parametrize("texto, frases", [
    # El decimal partido al abrir línea no corta (ni antes ni después del número).
    ("Te quedan:\n40. 5 GB libres, patrón.", ["Te quedan:\n40. 5 GB libres, patrón."]),
    # Marcadores "N)" y viñetas: no son candidatos a fin de frase.
    ("Te faltan:\n3) Reinicia el equipo\n4) Ir al banco", ["Te faltan:\n3) Reinicia el equipo\n"
                                                         "4) Ir al banco"]),
    ("Te faltan:\n- Reinicia el equipo\n- Ir al banco", ["Te faltan:\n- Reinicia el equipo\n"
                                                        "- Ir al banco"]),
    # Un número a media línea se queda en su línea, y con las líneas anteriores de
    # su frase; uno de tres cifras nunca fue marcador de lista.
    ("Te faltan 3. Reinicia el equipo\ny ya", ["Te faltan 3.", "Reinicia el equipo\ny ya"]),
    ("Te faltan\nunos 3. Reinicia el equipo", ["Te faltan\nunos 3.", "Reinicia el equipo"]),
    ("Son\n100. Ni uno más", ["Son\n100.", "Ni uno más"]),
    # Dentro de un bloque de código nada corta, tampoco un "3." que abre línea.
    ("Mira:\n```\nuno\n3. dos\n```\nListo.", ["Mira:\n```\nuno\n3. dos\n```\nListo."]),
    # Con lista abierta el marcador no se dice ni corta, salte lo que salte.
    ("Pasos:\n1. Abre la app\n3. Dale play\n7. Listo", ["Pasos:\n1. Abre la app\n3. Dale play\n"
                                                      "7. Listo"]),
])
def test_el_corte_en_el_salto_solo_alcanza_al_numero_suelto(texto, frases):
    """Lo que la regla NO toca corta exactamente igual que antes."""
    from crotolamo.voice.tts import split_sentences

    assert split_sentences(texto) == frases
    for paso in (1, 3, 7, len(texto)):
        assert _frases_en_streaming(texto, paso) == frases, paso


@pytest.mark.parametrize("texto, esperado", [
    ("El total es\n42.", ["El total es 42."]),
    ("El total es\n42. \n", ["El total es 42."]),
    # Con una cerca sin cerrar, al vaciar el buffer el número final SÍ pasa por la
    # segmentación (con sus espacios de cola): tiene que quedar igual que entera.
    ("Mira:\n```\nx = 1\n42. ", ["Mira: x = 1 42."]),
    ("Corre esto:\n```bash\nls\nY ya\n3. \n", ["Corre esto: ls Y ya 3."]),
])
def test_el_numero_que_cierra_la_respuesta_se_dice_con_su_linea_como_siempre(texto, esperado):
    """El número suelto al que no le sigue nada (la última línea de la respuesta es
    solo "42.") nunca pasó por viñeta: se decía, y se dice, con la línea anterior
    ("El total es 42."). La regla del salto no lo alcanza, tampoco en streaming."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


def test_la_primera_frase_no_sale_mas_tarde_por_cortar_antes_del_numero_suelto():
    """El corte del salto se decide con el mismo carácter que el de después del
    número (el que deja ver que no es un decimal): la línea anterior sale en ese
    momento, no más tarde, y ya sin el número pegado. La sonda de "sentence"
    suelta en ese mismo carácter, como antes."""
    from crotolamo.core.streaming import LiveStreamer
    from crotolamo.voice.tts import SentenceFeeder

    for texto, hasta, primeras in [
            (_FALTAN, "Te faltan estos:\n3. R", ["Te faltan estos:", "3."]),
            (_OPCIONES, "Opciones: 1. M", ["Opciones: 1."]),
            ("Ya van 2.\n3. Reinicia el equipo", "Ya van 2.\n3. R", ["Ya van 2.", "3."])]:
        frases: list[str] = []
        feeder = SentenceFeeder(frases.append)
        consumido = ""
        for c in texto:
            consumido += c
            feeder.feed(c)
            if frases:
                break
        assert (consumido, frases) == (hasta, primeras)

        s = LiveStreamer(lambda t: None, hold_until_done=True, release="sentence")
        assert next(texto[:i + 1] for i, c in enumerate(texto) if s.feed(c)) == hasta


def _sintetizado_en_los_dos_loops(tmp_path, monkeypatch, respuesta: str) -> list[list[str]]:
    """Lo que se le pide sintetizar a Piper (TTS de verdad, motor falso) por cada
    camino: StreamSpeaker con la respuesta de golpe y en vivo, speak_sentences y
    el loop concurrente (Cerebro + Boca) sin y con stream."""
    from crotolamo.voice.threads import BrainThread, MouthThread
    from crotolamo.voice.tts import TTS, StreamSpeaker

    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    sintetizado: list[str] = []
    monkeypatch.setattr(tts, "_speak_streaming",
                        lambda text, should_abort=None: sintetizado.append(text) or True)
    caminos: list[list[str]] = []

    for trozos in ([respuesta], list(respuesta)):
        sintetizado.clear()
        speaker = StreamSpeaker(tts)
        for trozo in trozos:
            speaker.feed(trozo)
        assert speaker.finish() is True
        caminos.append(list(sintetizado))

    sintetizado.clear()
    tts.speak_sentences(respuesta)
    caminos.append(list(sintetizado))

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
            cmd_q.put(("qué me falta", state.new_turn(mode=Mode.THINKING)))
            assert _wait(lambda: state.get_mode() is Mode.IDLE and len(sintetizado) >= 2)
        finally:
            shutdown.set()
            for hilo in hilos:
                hilo.join(2)
        caminos.append(list(sintetizado))
    return caminos


def test_el_numero_suelto_llega_a_piper_en_frase_aparte_en_los_dos_loops(tmp_path, monkeypatch):
    """De punta a punta: ninguna frase que se le pide a Piper trae un número suelto
    junto a palabras de la línea anterior, por ningún camino de voz."""
    for sintetizado in _sintetizado_en_los_dos_loops(tmp_path, monkeypatch, _FALTAN):
        assert sintetizado == _FALTAN_OIDO


# --- el VAD del barge-in no llega a la reanudación con lo oído antes de la pausa ---
class _VadConMemoria:
    """VAD falso con memoria, como la RNN de Silero: tras la voz, el silencio sigue
    puntuando alto un rato (baja 0.01 por chunk) hasta un reset(). Anota lo que
    recibe, lo que contesta y los reset()."""

    def __init__(self) -> None:
        self.nivel = 0.0
        self.resets = 0
        self.vistos: list[str] = []
        self.probs: list[float] = []

    def reset(self) -> None:
        self.resets += 1
        self.nivel = 0.0

    def __call__(self, chunk: str) -> float:
        self.vistos.append(chunk)
        self.nivel = 1.0 if chunk.startswith("voz") else max(0.0, self.nivel - 0.01)
        self.probs.append(self.nivel)
        return self.nivel


def _oido_con_vad(vad, *, barge_in: bool = True) -> tuple[EarThread, SharedState, _TtsContado]:
    """El oído con los valores de fábrica del barge-in (umbral 0.8, margen 0.1,
    racha de 5) y sin gracia de arranque, para que cuente desde el primer chunk."""
    state = SharedState()
    tts = _TtsContado()
    ear = EarThread(None, lambda chunk: chunk == "wake", vad, lambda frames: list(frames), tts,
                    queue.Queue(), queue.Queue(), state, threading.Event(),
                    allow_barge_in=barge_in, barge_in_grace_ms=0)
    return ear, state, tts


@pytest.mark.parametrize("modo", [Mode.SPEAKING, Mode.THINKING])
def test_el_vad_del_barge_in_arranca_limpio_al_reanudar_la_escucha(modo):
    """El panel pausa a media voz del cuarto, la voz acaba durante la pausa y la
    escucha se reanuda con el cuarto en silencio. En pausa los chunks no llegan al
    VAD: su memoria se quedaba congelada a media voz y, al reanudar, el silencio
    puntuaba alto. Con este VAD de mentira eso era un barge-in sobre un cuarto
    callado (con Silero de verdad y los umbrales de fábrica no llegó a serlo)."""
    vad = _VadConMemoria()
    ear, state, tts = _oido_con_vad(vad)
    turno = state.new_turn(mode=modo)
    for _ in range(3):
        ear._on_chunk("voz")            # 3 de 5: aún no es barge-in
    state.set_enabled(False)            # el panel pausa a media voz
    for chunk in ["voz"] * 10 + ["sil"] * 10:
        ear._on_chunk(chunk)
    assert vad.vistos == ["voz"] * 3, "en pausa el VAD del barge-in no oye"
    assert vad.resets == 1, "la pausa no limpió el VAD (o lo limpia en cada chunk)"

    state.set_enabled(True)             # reanuda: el cuarto ya está callado
    for _ in range(12):
        ear._on_chunk("sil")
    assert vad.probs[3:] == [0.0] * 12, "el VAD reanudó con la memoria de antes de la pausa"
    assert tts.stops == 0, "barge-in sobre un cuarto en silencio"
    assert state.turn_id == turno and state.get_mode() is modo


def test_el_vad_se_limpia_una_vez_por_pausa_y_el_barge_in_sigue_como_siempre():
    """Un reset por pausa (no uno por chunk: en pausa llegan ~31 por segundo), otro
    en la pausa siguiente, y reanudada la escucha el barge-in dispara igual, con
    una orden que empieza por la racha oída DESPUÉS de reanudar."""
    vad = _VadConMemoria()
    ear, state, tts = _oido_con_vad(vad)
    state.new_turn(mode=Mode.SPEAKING)
    ear._on_chunk("voz antes 1")
    ear._on_chunk("voz antes 2")
    state.set_enabled(False)
    for _ in range(50):
        ear._on_chunk("voz en pausa")
    assert vad.resets == 1
    state.set_enabled(True)
    ear._on_chunk("voz entre pausas 1")
    ear._on_chunk("voz entre pausas 2")
    state.set_enabled(False)
    for _ in range(50):
        ear._on_chunk("voz en pausa")
    assert vad.resets == 2
    assert tts.stops == 0 and state.turn_id == 1

    state.set_enabled(True)
    racha = [f"voz {n}" for n in range(1, 6)]
    for chunk in racha:
        ear._on_chunk(chunk)
    assert tts.stops == 1
    assert state.turn_id == 2 and state.get_mode() is Mode.LISTENING
    assert ear._endpoint.frames == racha
    assert vad.vistos == ["voz antes 1", "voz antes 2", "voz entre pausas 1",
                          "voz entre pausas 2", *racha]


def test_sin_pausa_la_memoria_del_vad_del_barge_in_sigue_de_chunk_en_chunk():
    """Sin pausa no cambia nada: nadie resetea el VAD entre chunks y su memoria
    sigue contando para la racha, como siempre (3 de voz y 2 de su cola: 5)."""
    vad = _VadConMemoria()
    ear, state, tts = _oido_con_vad(vad)
    state.new_turn(mode=Mode.SPEAKING)
    for chunk in ["voz", "voz", "voz", "sil"]:
        ear._on_chunk(chunk)
    assert vad.resets == 0 and tts.stops == 0
    assert vad.probs == [1.0, 1.0, 1.0, 0.99]
    ear._on_chunk("sil")
    assert tts.stops == 1 and state.turn_id == 2
    assert vad.probs == [1.0, 1.0, 1.0, 0.99, 0.98]
    assert vad.resets == 1  # el de siempre: la orden nueva arranca con el VAD limpio


@pytest.mark.parametrize("modo", [Mode.SPEAKING, Mode.THINKING])
def test_sin_barge_in_la_pausa_no_toca_el_vad(modo):
    """Sin --barge-in el oído no usa el VAD mientras Crotolamo piensa o habla: la
    pausa tampoco tiene nada que limpiarle."""
    vad = _VadConMemoria()
    ear, state, tts = _oido_con_vad(vad, barge_in=False)
    turno = state.new_turn(mode=modo)
    for habilitada in (True, False, True, False):
        state.set_enabled(habilitada)
        for chunk in ["voz"] * 8 + ["sil"] * 8:
            ear._on_chunk(chunk)
    assert vad.vistos == [] and vad.resets == 0
    assert tts.stops == 0 and state.turn_id == turno and state.get_mode() is modo
