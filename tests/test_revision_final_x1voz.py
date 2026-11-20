"""Revisión final, unidad x1voz: la voz de punta a punta (oído, cerebro, boca y
los dos loops de interfaces/listener.py).

Cada bloque fija un hallazgo confirmado; todo con fakes, sin micrófono ni audio.

- Loop concurrente con --barge-in: el aviso de una tool delicada va atado al
  turno que el Cerebro está atendiendo, no al vigente cuando se dice. El de una
  orden que el patrón ya interrumpió sonaba sobre la orden NUEVA en cuanto el
  oído la cerraba (turno 2 en THINKING).
- Loop concurrente con --barge-in: con la escucha pausada desde el panel, la voz
  del cuarto durante una respuesta ya no abre (ni contesta) una orden nueva.
- Una frase que empieza por un número de 1-2 cifras y punto ya no pierde el
  número: "42. ¿Algo más, patrón?" se oía sin el 42 (pasaba por viñeta de
  lista), y en streaming le pasaba a cualquier frase tras un corte.
- Con stream_release="sentence", lo que el turno ya soltó se dice ENTERO antes
  de la pregunta de confirmación (loop simple) o del aviso (concurrente): la
  pregunta se colaba a media explicación y el resto sonaba sobre la ventana de
  la respuesta.
- Loop simple: el HUD dice "hablando" mientras suena la pregunta de
  confirmación y "escuchando" mientras se graba la respuesta (decía "pensando").
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import EarThread

from interfaces import listener


def _wait(cond, timeout: float = 5.0) -> bool:
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# --- barge-in con la escucha pausada desde el panel ------------------------------
class _TtsContado:
    """TTS falso: cuenta los stop() y anota lo hablado."""

    def __init__(self) -> None:
        self.stops = 0
        self.dichos: list[str] = []

    def speak(self, text):
        self.dichos.append(text)

    def stop(self):
        self.stops += 1


def _oido_con_barge_in(state: SharedState, tts) -> tuple[EarThread, queue.Queue, queue.Queue]:
    stt_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    ear = EarThread(
        None, lambda chunk: chunk == "wake",
        lambda chunk: 0.95 if chunk == "voz" else 0.0, lambda frames: list(frames),
        tts, stt_q, tts_q, state, threading.Event(),
        allow_barge_in=True, barge_in_min_chunks=5, barge_in_grace_ms=0,
    )
    return ear, stt_q, tts_q


@pytest.mark.parametrize("modo", [Mode.SPEAKING, Mode.THINKING])
def test_barge_in_con_la_escucha_pausada_no_abre_una_orden(modo):
    """El panel pausa la escucha mientras Crotolamo piensa o habla: la voz del
    cuarto ya no corta la respuesta ni abre un turno (era la única forma de
    empezar una orden sin wake estando en pausa). El turno en curso sigue."""
    state = SharedState()
    tts = _TtsContado()
    ear, stt_q, tts_q = _oido_con_barge_in(state, tts)
    turno = state.new_turn(mode=modo)
    tts_q.put("frase pendiente de la respuesta")
    state.set_enabled(False)  # lo que hace VoiceLoop._poll_control con la pausa

    for _ in range(12):
        ear._on_chunk("voz")

    assert tts.stops == 0, "cortó la respuesta con la escucha pausada"
    assert state.turn_id == turno and state.get_mode() is modo
    assert tts_q.qsize() == 1, "tiró las frases de la respuesta en curso"
    assert stt_q.empty()


def test_barge_in_vuelve_al_reanudar_la_escucha_y_no_hereda_la_racha_de_antes():
    """Reanudada la escucha a media respuesta, el barge-in funciona como siempre,
    con una racha NUEVA: los chunks de antes de la pausa no cuentan (4 de antes
    + 1 de después no son 5 seguidos)."""
    state = SharedState()
    tts = _TtsContado()
    ear, _stt_q, _tts_q = _oido_con_barge_in(state, tts)
    state.new_turn(mode=Mode.SPEAKING)
    for _ in range(4):
        ear._on_chunk("voz")
    state.set_enabled(False)
    ear._on_chunk("voz")
    state.set_enabled(True)
    ear._on_chunk("voz")
    assert tts.stops == 0 and state.turn_id == 1, "la racha de antes de la pausa disparó"
    for _ in range(4):
        ear._on_chunk("voz")
    assert tts.stops == 1
    assert state.turn_id == 2 and state.get_mode() is Mode.LISTENING


class _MicDeSala:
    """wake + orden + silencio; después, la voz del cuarto mientras `sala` esté puesta."""

    def __init__(self, sala: threading.Event) -> None:
        self.sala = sala
        self.guion = iter(["wake"] + ["voz"] * 6 + ["sil"] * 6)

    def read(self):
        time.sleep(0.002)
        try:
            return next(self.guion)
        except StopIteration:
            return "voz" if self.sala.is_set() else "sil"


def test_loop_concurrente_pausado_a_media_respuesta_no_atiende_la_voz_del_cuarto(monkeypatch):
    """Con los cuatro hilos de verdad: la respuesta del turno 1 suena, el panel
    pausa y en el cuarto se sigue hablando. Antes: stop(), turno 2, la orden
    transcrita, mandada al LLM y contestada con el panel en pausa."""
    from crotolamo import settings as crotolamo_settings

    monkeypatch.setattr(crotolamo_settings, "get_settings", lambda: SimpleNamespace(voice={
        "vad_threshold": 0.8, "barge_in_grace_ms": 50,
        "barge_in_threshold_margin": 0.1, "barge_in_min_chunks": 5,
    }))
    sala = threading.Event()
    sonando, soltar = threading.Event(), threading.Event()
    ordenes: list[str] = []
    publicado: list[tuple[str, int, bool]] = []

    class _Tts(_TtsContado):
        def speak(self, text):
            self.dichos.append(text)
            sonando.set()
            soltar.wait(5)

    class _Stt:
        def transcribe(self, audio, hotwords=None):
            return f"orden {len(ordenes) + 1}"

    def handle_turn(command):
        ordenes.append(command)
        return "Respuesta larga."

    tts = _Tts()
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=handle_turn), _Stt(), tts, None,
        mic=_MicDeSala(sala), wake_fn=lambda c: c == "wake",
        vad_fn=lambda c: 0.95 if c == "voz" else 0.0, to_wav=lambda frames: "AUDIO",
        silence_ms=64, allow_barge_in=True,
        hud_publisher=lambda d: publicado.append((d["mode"], d["turn_id"], d["enabled"])),
    )
    vl.start()
    try:
        assert sonando.wait(5), "la respuesta del turno 1 no llegó a sonar"
        vl.state.set_enabled(False)
        sala.set()
        time.sleep(0.5)  # pasada la gracia, ~200 chunks con voz: de sobra para la racha
        assert tts.stops == 0
        assert vl.state.turn_id == 1 and vl.state.get_mode() is Mode.SPEAKING
        sala.clear()
        soltar.set()
        assert _wait(lambda: vl.state.get_mode() is Mode.IDLE)
    finally:
        soltar.set()
        vl.stop()
    assert ordenes == ["orden 1"]
    assert tts.dichos == ["Respuesta larga."]
    assert not [p for p in publicado if p[1] > 1], publicado


# --- el aviso de una tool delicada va atado al turno que atiende el Cerebro -------
_AVISO = "Voy a escribir en la lista vieja."
_RESPUESTA_VIEJA = "Cancelado, patrón."
_RESPUESTA_NUEVA = "Respuesta nueva."


def _correr_concurrente(monkeypatch, tmp_path, argv, ordenes, handle_turn, terminado,
                        voz: dict | None = None):
    """run_listen concurrente con el VoiceLoop REAL sobre fakes (el arnés de
    tests/test_auditoria_voice_listener.py, con varias órdenes): `handle_turn(agente,
    comando, on_token)` hace de ToolAgent y recibe el confirm_fn que run_listen le
    pasó a build_agent. Devuelve (tts, publicaciones del HUD, loop, agente)."""
    from crotolamo import settings as crotolamo_settings
    from crotolamo.voice import loop as voice_loop
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _BlockingDetector,
        _FakeSttCls,
        _join_warmers,
        _MicConEco,
        _TtsConEco,
    )

    _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
    if voz is not None:
        monkeypatch.setattr(listener, "get_settings",
                            lambda: SimpleNamespace(voice={"wake_duck": False, **voz}, wake={}))
    monkeypatch.setattr(crotolamo_settings, "get_settings", lambda: SimpleNamespace(voice={
        "vad_threshold": 0.8, "barge_in_grace_ms": 400,
        "barge_in_threshold_margin": 0.1, "barge_in_min_chunks": 5,
    }))
    tts = _TtsConEco()
    patron = threading.Event()
    mic = _MicConEco(tts.sonando, patron)
    monkeypatch.setattr(listener, "TTS", SimpleNamespace(from_settings=lambda s: tts))
    pendientes = list(ordenes)

    class _Stt(_FakeSttCls):
        def transcribe(self, audio, hotwords=None):
            return pendientes.pop(0) if pendientes else ""

    monkeypatch.setattr(listener, "STT", _Stt)
    loops: list = []

    class _Agente:
        def __init__(self, confirm_fn):
            self.confirm_fn = confirm_fn
            self.confirmaciones: list[bool] = []
            self.comandos: list[str] = []
            self.patron = patron
            self.loops = loops

        def handle_turn(self, command, on_token=None):
            self.comandos.append(command)
            return handle_turn(self, command, on_token)

    agentes: list = []

    def _build_agent(confirm_fn=None):
        agentes.append(_Agente(confirm_fn))
        return agentes[-1], None

    monkeypatch.setattr(listener, "build_agent", _build_agent)
    hud: list[tuple[str, int, str]] = []
    real_loop = voice_loop.VoiceLoop

    def _loop(agent, stt, tts_, wake_detector, **kwargs):
        publicar = kwargs["hud_publisher"]

        def _anota(state_dict):
            hud.append((state_dict["mode"], state_dict["turn_id"], state_dict["text"]))
            publicar(state_dict)

        kwargs["hud_publisher"] = _anota
        vl = real_loop(agent, stt, tts_, wake_detector, mic=mic,
                       wake_fn=lambda chunk: chunk == "wake",
                       vad_fn=lambda chunk: {"voz": 0.95, "eco": 0.85}.get(chunk, 0.0),
                       to_wav=lambda frames: frames, **kwargs)
        tts.loop = vl
        loops.append(vl)
        return vl

    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop)
    rc: list[int] = []
    hilo = threading.Thread(target=lambda: rc.append(listener.run_listen(argv)), daemon=True)
    hilo.start()
    try:
        assert _wait(lambda: bool(loops), timeout=10.0), "el loop no arrancó"
        listo = _wait(lambda: terminado(tts, loops[0]), timeout=15.0)
    finally:
        if loops:
            loops[0].shutdown.set()
        hilo.join(timeout=10.0)
        _join_warmers()
    assert listo, f"el turno no terminó: {tts.dichos} / {hud}"
    assert rc == [0]
    return tts, hud, loops[0], agentes[0]


def _modo(state) -> str:
    return state.get_mode().name.lower()


def test_concurrente_aviso_del_turno_interrumpido_no_suena_con_la_orden_nueva_ya_cerrada(
        monkeypatch, tmp_path):
    """--barge-in y [voice].barge_in_cancel=false (de fábrica): el patrón habla
    encima mientras el LLM pensaba, dice otra orden y el oído la cierra (turno 2
    en THINKING, esperando al Cerebro, que sigue dentro del turno 1). El turno
    viejo pide entonces una tool delicada: su aviso sonaba ~8 s sobre el turno
    NUEVO, con el HUD en "hablando" y el texto de la orden nueva."""

    def handle_turn(agente, command, on_token):
        if len(agente.comandos) > 1:
            return _RESPUESTA_NUEVA
        st = agente.loops[0].state
        agente.patron.set()  # el patrón habla encima: barge-in real del oído
        assert _wait(lambda: st.turn_id == 2 and _modo(st) == "listening")
        agente.patron.clear()  # termina su orden: el oído la cierra y el Stt la transcribe
        assert _wait(lambda: st.current_snapshot()["text"] == "mejor dime la hora")
        agente.antes = (_modo(st), st.turn_id)
        agente.confirmaciones.append(agente.confirm_fn(_AVISO))  # ToolAgent._execute_call
        agente.despues = (_modo(st), st.turn_id)
        return _RESPUESTA_VIEJA

    tts, hud, vl, agente = _correr_concurrente(
        monkeypatch, tmp_path, ["--barge-in"],
        ["apunta algo en la lista vieja", "mejor dime la hora"], handle_turn,
        lambda tts, vl: (any(t == _RESPUESTA_NUEVA for t, _, _ in tts.dichos)
                         and _modo(vl.state) == "idle"))
    assert agente.antes == ("thinking", 2)
    assert agente.confirmaciones == [False]  # la tool se sigue denegando
    assert [t for t, _, _ in tts.dichos if _AVISO in t] == [], tts.dichos
    assert agente.despues == ("thinking", 2), "el aviso del turno viejo tocó el modo del nuevo"
    # Lo único que suena tras el saludo (dicho antes de que exista el loop).
    assert [d for d in tts.dichos if d[1] is not None] == [
        (_RESPUESTA_NUEVA, "speaking", "Mouth")]
    # El HUD del turno nuevo: escucha, piensa (con su orden), habla. Sin un
    # "hablando" antes de su respuesta.
    del_nuevo = [modo for modo, turno, _ in hud if turno == 2]
    sin_repetir = [m for i, m in enumerate(del_nuevo) if i == 0 or m != del_nuevo[i - 1]]
    assert sin_repetir == ["listening", "thinking", "speaking", "idle"], hud


def test_aviso_va_atado_al_turno_del_cerebro_y_no_al_vigente():
    """Sin hilos: el turno vigente es el 2; el Cerebro sigue en el 1. Su aviso se
    descarta esté el patrón dictando (LISTENING) o con la orden nueva ya cerrada
    (THINKING), y el de la orden que el Cerebro SÍ atiende suena como siempre."""
    dichos: list[tuple[str, Mode, int]] = []
    state = SharedState()
    tts = SimpleNamespace(speak=lambda text, should_abort=None: dichos.append(
        (text, state.get_mode(), state.turn_id)))
    cerebro = SimpleNamespace(turno_en_curso=state.new_turn(mode=Mode.THINKING),
                              hablar_lo_encolado=lambda: True)
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state, lambda: cerebro)
    state.new_turn(mode=Mode.LISTENING)  # barge-in: el patrón dicta la orden nueva
    decir("aviso del turno 1, el patrón aún dicta")
    state.set_mode(Mode.THINKING)  # el oído cerró la orden nueva
    decir("aviso del turno 1, la orden nueva ya espera")
    assert dichos == []
    assert state.get_mode() is Mode.THINKING
    cerebro.turno_en_curso = 2  # el Cerebro ya atiende la orden nueva
    decir("aviso del turno 2")
    assert dichos == [("aviso del turno 2", Mode.SPEAKING, 2)]
    assert state.get_mode() is Mode.THINKING


def test_el_cerebro_anota_el_turno_que_atiende_solo_mientras_lo_atiende():
    from crotolamo.voice.threads import BrainThread

    vistos: list[int | None] = []
    state = SharedState()
    cmd_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(SimpleNamespace(handle_turn=lambda c: vistos.append(brain.turno_en_curso)
                                        or "Listo."),
                        cmd_q, queue.Queue(), state, shutdown)
    assert brain.turno_en_curso is None
    brain.start()
    try:
        state.new_turn(mode=Mode.THINKING)
        state.new_turn(mode=Mode.THINKING)
        cmd_q.put(("la hora", 2))
        assert _wait(lambda: vistos == [2])
        assert _wait(lambda: brain.turno_en_curso is None)
    finally:
        shutdown.set()
        brain.join(2)


# --- "NN." al principio de una frase: ¿número o viñeta de lista? ------------------
def _frases_en_streaming(texto: str, paso: int = 1) -> list[str]:
    """Las frases que la segmentación en streaming manda a hablar (la de
    StreamSpeaker y del BrainThread), con el texto llegando de `paso` en `paso`."""
    from crotolamo.voice.tts import SentenceFeeder

    frases: list[str] = []
    feeder = SentenceFeeder(frases.append)
    for i in range(0, len(texto), paso):
        feeder.feed(texto[i:i + paso])
    feeder.flush()
    return frases


def _oido(frases: list[str]) -> list[str]:
    """Lo que Piper recibe: cada frase pasa por limpiar_para_voz en TTS.speak."""
    from crotolamo.voice.clean import limpiar_para_voz

    return [limpia for limpia in map(limpiar_para_voz, frases) if limpia]


def _oido_de_todas_las_formas(texto: str) -> list[list[str]]:
    """La respuesta entera (split_sentences), soltada de golpe por on_token (lo
    de fábrica con tools a la vista) y en streaming de 1, 3 y 7 caracteres."""
    from crotolamo.voice.tts import split_sentences

    return [_oido(split_sentences(texto)), _oido(_frases_en_streaming(texto, len(texto)))] + [
        _oido(_frases_en_streaming(texto, paso)) for paso in (1, 3, 7)]


@pytest.mark.parametrize("texto, esperado", [
    ("42. ¿Algo más, patrón?", ["42.", "¿Algo más, patrón?"]),
    ("25. Los cumplió ayer, patrón.", ["25.", "Los cumplió ayer, patrón."]),
    ("  7. Y ya, patrón.", ["7.", "Y ya, patrón."]),
])
def test_una_respuesta_que_empieza_por_su_numero_no_lo_pierde(texto, esperado):
    """El número ES la respuesta: "NN." al principio pasaba por marcador de lista,
    la frase llegaba entera a speak() y limpiar_para_voz le quitaba el "42. " como
    viñeta. Se oía "¿Algo más, patrón?"."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("¿Cuántos? 5. Cinco.", ["¿Cuántos?", "5.", "Cinco."]),
    ("¿Cuánto es seis por siete? 42. Fácil, patrón.",
     ["¿Cuánto es seis por siete?", "42.", "Fácil, patrón."]),
    ("Está en el cap. 12. ¿Lo abro?", ["Está en el cap.", "12.", "¿Lo abro?"]),
    ("Listo, patrón. 1. Abre Spotify.", ["Listo, patrón.", "1.", "Abre Spotify."]),
])
def test_en_streaming_un_numero_tras_un_corte_a_media_linea_no_es_vineta(texto, esperado):
    """SentenceFeeder reinicia el buffer tras cada corte y su posición 0 pasaba por
    principio de línea: cualquier frase que empezara por un número perdía el
    número. Mismas frases que la respuesta entera (lo que ya fijaba
    split_sentences("¿Cuántos? 5. Cinco."))."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("1. Abre Spotify.\n2. Dale play.", ["Abre Spotify.", "Dale play."]),
    ("Claro, patrón.\n1. Abre Spotify.\n2. Busca la lista.\n3. Dale play.",
     ["Claro, patrón.", "Abre Spotify.", "Busca la lista.", "Dale play."]),
    ("Son tres pasos:\n\n1. Abre Spotify\n2. Busca la lista\n3. Dale play",
     ["Son tres pasos:", "Abre Spotify Busca la lista Dale play"]),
    # Una frase suelta entre dos elementos, y una numeración que salta.
    ("Pasos:\n1. Abre la app. Espera a que cargue.\nLuego:\n2. Dale play.\n  12. Otro paso.",
     ["Pasos: Abre la app.", "Espera a que cargue.", "Luego: Dale play.", "Otro paso."]),
    ("".join(f"{n}. Abre la aplicación {n} y revisa.\n" for n in range(1, 13)),
     [f"Abre la aplicación {n} y revisa." for n in range(1, 13)]),
])
def test_los_numeros_de_una_lista_numerada_siguen_sin_hablarse(texto, esperado):
    """La otra cara: con lista a la vista (el "1." que la abre, o una ya abierta
    en la respuesta) el número sigue siendo viñeta y no se dice."""
    for oido in _oido_de_todas_las_formas(texto):
        assert oido == esperado


def test_un_decimal_partido_al_principio_sigue_de_corrido():
    """"40. 5 GB" no es ni lista ni fin de frase: la regla del decimal manda."""
    from crotolamo.voice.tts import split_sentences

    texto = "40. 5 GB libres, patrón. Listo."
    assert split_sentences(texto) == ["40. 5 GB libres, patrón.", "Listo."]
    for paso in (1, 3, len(texto)):
        assert _frases_en_streaming(texto, paso) == ["40. 5 GB libres, patrón.", "Listo."]


_TROZOS_NUM = ["Hola", "patrón", "Dr", "etc", "No", "ya", "1", "2", "3", "5", "12", "42",
               ".", ".", ". ", ". ", " ", " ", "\n", "\n", "\n\n", "!", "? ", "¿", ",", ":",
               "**", "  ", "\t", "á"]


def test_streaming_y_respuesta_entera_cortan_las_mismas_frases():
    """La segmentación en streaming no puede depender de por dónde llegan los
    trozos: se oye lo mismo que con split_sentences sobre la respuesta entera, y
    sin énfasis markdown de por medio las frases son las mismas una a una (un
    "**" tras un salto puede quedar en una frase u otra; se limpia igual). Las
    únicas diferencias eran las de un número al principio de una frase."""
    import random

    from crotolamo.voice.tts import split_sentences

    rng = random.Random(20261120)
    for _ in range(4000):
        texto = "".join(rng.choice(_TROZOS_NUM) for _ in range(rng.randint(1, 30)))
        entera = split_sentences(texto)
        for paso in (1, rng.randint(2, 9), len(texto)):
            frases = _frases_en_streaming(texto, paso)
            assert _oido(frases) == _oido(entera), (texto, paso)
            if "*" not in texto:
                assert frases == entera, (texto, paso)


def test_la_sonda_de_sentence_ve_el_numero_inicial_como_frase():
    """stream_release="sentence" suelta cuando la 1ª frase cerró y le sigue texto,
    con la MISMA segmentación de lo que se habla: "42." ya es una frase; el "1."
    que abre una lista, no."""
    from crotolamo.core.streaming import LiveStreamer

    def soltado_en(texto: str) -> str:
        dichos: list[str] = []
        s = LiveStreamer(dichos.append, hold_until_done=True, release="sentence")
        for i, c in enumerate(texto):
            if s.feed(c):
                return texto[:i + 1]
        return ""

    assert soltado_en("42. ¿Algo más, patrón?") == "42. ¿"
    assert soltado_en("1. Abre Spotify.\n2. Dale play.") == "1. Abre Spotify.\n2"
    assert soltado_en("Mira:\n1. Abre Spotify\n2. Dale play\n") == ""


def test_el_numero_inicial_llega_a_piper_en_los_dos_loops(tmp_path, monkeypatch):
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
    respuesta = "42. ¿Algo más, patrón?"
    esperado = ["42.", "¿Algo más, patrón?"]

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
            cmd_q.put(("cuánto es seis por siete", state.new_turn(mode=Mode.THINKING)))
            assert _wait(lambda: state.get_mode() is Mode.IDLE and len(sintetizado) == 2)
        finally:
            shutdown.set()
            for hilo in hilos:
                hilo.join(2)
        assert sintetizado == esperado, stream


# --- lo ya soltado se dice entero antes de preguntar (y el HUD durante la pregunta) --
_PREAMBULO = ["Claro que sí, patrón.", "Primero reviso que la lista exista.",
              "Después le agrego lo que me pediste.", "Dame un segundo que lo escribo."]
_PREGUNTA = "La acción 'write file' sobre la lista puede ser destructiva, patrón. ¿La confirmas?"
_HECHO = "Hecho, patrón."


class _TtsConTurno:
    """TTS falso con turno de voz, como el real: una frase a la vez, la pida quien
    la pida. Anota el principio y el fin de cada una (y lo que diga `sello`)."""

    def __init__(self, sello=lambda: None, dura_s: float = 0.03) -> None:
        self._voz = threading.Lock()
        self.sello = sello
        self.dura_s = dura_s
        self.eventos: list[tuple] = []

    def available(self):
        return False

    def speak(self, text, should_abort=None):
        with self._voz:
            self.eventos.append(("habla", text, self.sello()))
            time.sleep(self.dura_s)
            self.eventos.append(("calla", text))

    def speak_sentences(self, text):
        self.speak(text)

    def beep(self):
        pass

    def stop(self):
        pass

    def habladas(self) -> list[str]:
        return [e[1] for e in self.eventos if e[0] == "habla"]


def test_stream_speaker_drain_dice_lo_ya_soltado_antes_de_volver(tmp_path, monkeypatch):
    """drain(): la frase que la segmentación retenía (acaba en punto y aún no se
    sabe qué sigue) y las que esperaban en cola se dicen ANTES de volver; el
    speaker sigue sirviendo para el resto de la respuesta. Con el TTS de verdad
    (su turno de voz; el motor, falso): sin drain() la pregunta, dicha desde el
    hilo del turno, se colaba por delante de las frases en cola."""
    from crotolamo.voice.tts import TTS, StreamSpeaker

    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    eventos: list[tuple[str, str]] = []

    def motor(text, should_abort=None):
        eventos.append(("habla", text))
        time.sleep(0.02)
        eventos.append(("calla", text))
        return True

    monkeypatch.setattr(tts, "_speak_streaming", motor)
    speaker = StreamSpeaker(tts)
    for token in " ".join(_PREAMBULO).split(" "):
        speaker.feed(token + " ")
    assert speaker.drain() is True
    assert eventos[-1] == ("calla", _PREAMBULO[-1]), "drain volvió con frases por decir"
    tts.speak(_PREGUNTA)
    speaker.feed("\n" + _HECHO)
    assert speaker.finish() is True
    assert [e[1] for e in eventos if e[0] == "habla"] == [*_PREAMBULO, _PREGUNTA, _HECHO]
    # Sin nada soltado, vuelve en el acto; y tras finish() no se queda esperando.
    vacio = StreamSpeaker(tts)
    assert vacio.drain() is True
    assert vacio.finish() is False
    assert vacio.drain() is False


class _DetectorDeUnWake:
    """openWakeWord falso del loop simple: un wake y, a la siguiente espera, Ctrl-C."""

    threshold = 0.5
    threshold_media = None

    def __init__(self) -> None:
        self.llamadas = 0

    def available(self):
        return True

    def attach_media(self, monitor):
        pass

    def listen_for_wake(self, timeout_s=None):
        self.llamadas += 1
        if self.llamadas == 1:
            return True
        raise KeyboardInterrupt


def _turno_simple_con_confirmacion(monkeypatch, tmp_path, preambulo: list[str]):
    """run_listen --simple de punta a punta sobre fakes: un wake, la orden y un
    agente que (como ToolAgent) suelta `preambulo` por on_token, pide confirmar una
    tool con el confirm_fn que le dio run_listen y termina. Devuelve los eventos
    en orden: ("habla"/"calla", texto[, modo del HUD]), ("escucha"/"oido", modo
    del HUD) de la ventana de la respuesta y ("confirmado", bool, modo del HUD)."""
    import json

    from tests.test_auditoria_voice_listener import _arnes_run_listen, _join_warmers

    hud_path, _control = _arnes_run_listen(monkeypatch, tmp_path, _DetectorDeUnWake())
    monkeypatch.setattr(listener, "get_settings", lambda: SimpleNamespace(
        voice={"wake_duck": False, "followup_s": 0, "ack": "off"}, wake={}))

    def modo_hud() -> str:
        return json.loads(hud_path.read_text(encoding="utf-8"))["mode"]

    tts = _TtsConTurno(sello=lambda: modo_hud() if hud_path.exists() else None)
    eventos = tts.eventos
    monkeypatch.setattr(listener, "TTS", SimpleNamespace(from_settings=lambda s: tts))

    class _Stt:
        model_size = "base"

        def __init__(self, **kwargs):
            pass

        @classmethod
        def from_settings(cls, settings):
            return cls()

        def _get_model(self):
            return None

        def listen_smart(self, **kwargs):  # la orden
            return "apunta leche en la lista"

        def listen_once(self, **kwargs):  # la respuesta a la pregunta
            eventos.append(("escucha", modo_hud()))
            time.sleep(0.15)
            eventos.append(("oido", modo_hud()))
            return "confirmo"

    monkeypatch.setattr(listener, "STT", _Stt)

    class _Agente:
        def __init__(self, confirm_fn):
            self.confirm_fn = confirm_fn

        def handle_turn(self, command, on_token=None):
            for token in " ".join(preambulo).split(" ") if preambulo else []:
                on_token(token + " ")
            ok = self.confirm_fn(_PREGUNTA)  # ToolAgent._execute_call
            eventos.append(("confirmado", ok, modo_hud()))
            if on_token is not None:
                on_token("\n" + _HECHO if preambulo else _HECHO)
            return " ".join([*preambulo, _HECHO])

    monkeypatch.setattr(listener, "build_agent",
                        lambda confirm_fn=None: (_Agente(confirm_fn), None))
    try:
        assert listener.run_listen(["--simple"]) == 0
    finally:
        _join_warmers()
    return eventos


def test_loop_simple_la_pregunta_de_confirmacion_espera_a_lo_ya_soltado(monkeypatch, tmp_path):
    """stream_release="sentence": el modelo suelta un preámbulo de varias frases y
    pide una tool delicada. La pregunta solo esperaba a la frase que sonaba: se
    colaba a media explicación (la última frase, retenida, salía siempre después)
    y el resto del preámbulo sonaba sobre la ventana de la respuesta."""
    eventos = _turno_simple_con_confirmacion(monkeypatch, tmp_path, _PREAMBULO)
    habladas = [e[1] for e in eventos if e[0] == "habla"]
    pregunta = _PREGUNTA + " Di 'confirmo' o 'cancela', patrón."
    # Entre el saludo y la despedida: el preámbulo entero, la pregunta, el final.
    assert habladas[1:-1] == [*_PREAMBULO, pregunta, _HECHO], habladas
    # Con el micrófono de la respuesta abierto, Crotolamo no dice nada.
    abre = eventos.index(next(e for e in eventos if e[0] == "escucha"))
    cierra = eventos.index(next(e for e in eventos if e[0] == "oido"))
    assert eventos[abre - 1] == ("calla", pregunta)
    assert cierra == abre + 1, eventos[abre:cierra + 1]
    assert ("confirmado", True, "speaking") in eventos  # el modo de antes de preguntar


def test_loop_simple_el_hud_acompana_la_pregunta_de_confirmacion(monkeypatch, tmp_path):
    """Lo de fábrica (nada sonó antes de la tool): mientras suena la pregunta el
    HUD dice "hablando", mientras se graba la respuesta "escuchando" y después
    vuelve a "pensando". Decía "pensando" todo el rato, con el patrón esperado
    al micrófono."""
    eventos = _turno_simple_con_confirmacion(monkeypatch, tmp_path, [])
    pregunta = _PREGUNTA + " Di 'confirmo' o 'cancela', patrón."
    assert ("habla", pregunta, "speaking") in eventos
    assert ("escucha", "listening") in eventos and ("oido", "listening") in eventos
    assert ("confirmado", True, "thinking") in eventos
    assert ("habla", _HECHO, "speaking") in eventos


def test_concurrente_el_aviso_espera_a_lo_que_el_turno_ya_mando_a_la_boca():
    """[voice].stream_speak_concurrent con un preámbulo ya soltado: la Boca tiene
    frases de este turno en cola y la segmentación retiene la última. El aviso de
    la tool delicada se dice detrás de todas, no a media explicación."""
    from crotolamo.voice.threads import BrainThread, MouthThread

    state = SharedState()
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    tts = _TtsConTurno(sello=lambda: state.get_mode())
    avisos: list = []

    def handle_turn(command, on_token=None):
        for token in " ".join(_PREAMBULO).split(" "):
            on_token(token + " ")
        avisos[0](_PREGUNTA)
        on_token("\n" + _HECHO)
        return " ".join([*_PREAMBULO, _HECHO])

    brain = BrainThread(SimpleNamespace(handle_turn=handle_turn), cmd_q, tts_q, state,
                        shutdown, stream=True)
    avisos.append(listener._make_notice_say(lambda t: None, tts, lambda: state, lambda: brain))
    hilos = [brain, MouthThread(tts, tts_q, state, shutdown)]
    for hilo in hilos:
        hilo.start()
    try:
        cmd_q.put(("apunta leche en la lista", state.new_turn(mode=Mode.THINKING)))
        assert _wait(lambda: state.get_mode() is Mode.IDLE and _HECHO in tts.habladas())
    finally:
        shutdown.set()
        for hilo in hilos:
            hilo.join(2)
    assert tts.habladas() == [*_PREAMBULO, _PREGUNTA, _HECHO]
    assert ("habla", _PREGUNTA, Mode.SPEAKING) in tts.eventos


def test_el_aviso_no_espera_a_la_boca_de_un_turno_que_ya_no_es_el_vigente():
    """Un barge-in mientras el Cerebro esperaba a la Boca: la espera se suelta (la
    cola se tiró con el turno) y el aviso, que ya es de un turno viejo, no suena."""
    from crotolamo.voice.threads import BrainThread

    state = SharedState()
    tts_q: queue.Queue = queue.Queue()  # sin Boca: nadie atiende la cola
    brain = BrainThread(SimpleNamespace(handle_turn=lambda c, on_token=None: ""),
                        queue.Queue(), tts_q, state, threading.Event(), stream=True)
    dichos: list[str] = []
    tts = SimpleNamespace(speak=lambda text, should_abort=None: dichos.append(text))
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state, lambda: brain)
    hecho = threading.Event()

    def handle_turn(command, on_token=None):
        on_token("Voy a escribirlo. ")
        on_token("Dame un segundo. ")
        decir(_PREGUNTA)
        hecho.set()
        return ""

    brain.agent = SimpleNamespace(handle_turn=handle_turn)
    brain.start()
    try:
        brain.cmd_q.put(("apunta leche", state.new_turn(mode=Mode.THINKING)))
        assert _wait(lambda: tts_q.qsize() == 3)  # las dos frases y la marca de paso
        assert not hecho.is_set()
        state.new_turn(mode=Mode.LISTENING)  # barge-in
        assert hecho.wait(3), "el Cerebro se quedó esperando a la Boca"
    finally:
        brain.shutdown.set()
        brain.join(2)
    assert dichos == []


def test_drain_no_se_cuelga_con_el_tts_trabado():
    """El audio trabado no deja al turno esperando para siempre: pasado el plazo
    por frase, drain() vuelve (False) y quien preguntaba sigue, como antes."""
    from crotolamo.voice.tts import StreamSpeaker

    soltar = threading.Event()
    speaker = StreamSpeaker(SimpleNamespace(speak=lambda text: soltar.wait(5)))
    speaker.feed("Una frase. Y otra.")
    t0 = time.monotonic()
    try:
        assert speaker.drain(timeout_s=0.1) is False
        assert time.monotonic() - t0 < 2.0
    finally:
        soltar.set()
    assert speaker.finish() is True


def test_lo_que_on_token_retenia_tambien_se_dice_antes_de_la_pregunta():
    """El loop simple envuelve on_token en _AvisoEnFraseAparte, que retiene un
    token que empieza por salto de línea hasta ver el siguiente: si era la última
    frase del preámbulo, también va antes de la pregunta."""
    from crotolamo.voice.tts import StreamSpeaker, _AvisoEnFraseAparte

    tts = _TtsConTurno(dura_s=0.01)
    speaker = StreamSpeaker(tts)
    on_token = _AvisoEnFraseAparte(speaker.feed)
    on_token("Voy a apuntarlo.")
    on_token("\nDame un segundo.")
    listener._hablar_pendiente(on_token, speaker)
    assert tts.habladas() == ["Voy a apuntarlo.", "Dame un segundo."]
    on_token("\n" + _HECHO)
    on_token.cerrar("Voy a apuntarlo.\nDame un segundo.\n" + _HECHO)
    assert speaker.finish() is True
    assert tts.habladas() == ["Voy a apuntarlo.", "Dame un segundo.", _HECHO]


def test_tras_un_flush_a_media_respuesta_lo_que_sigue_recuerda_la_linea():
    """flush() a media respuesta (drain) no borra la memoria de línea: lo que
    sigue a una cola cortada a media línea no abre lista con su "1."; lo que
    sigue a un salto de línea, sí."""
    from crotolamo.voice.tts import SentenceFeeder

    frases: list[str] = []
    feeder = SentenceFeeder(frases.append)
    feeder.feed("Te queda")
    feeder.flush()
    feeder.feed(" 1. Uno solo, patrón.")
    feeder.flush()
    assert _oido(frases) == ["Te queda", "1.", "Uno solo, patrón."]

    frases.clear()
    feeder = SentenceFeeder(frases.append)
    feeder.feed("Hola, patrón. Mira")
    feeder.feed(":\n")
    feeder.flush()
    feeder.feed("1. Abre Spotify.\n2. Dale play.")
    feeder.flush()
    assert _oido(frases) == ["Hola, patrón.", "Mira:", "Abre Spotify.", "Dale play."]


def test_el_aviso_no_espera_para_siempre_a_una_boca_trabada(monkeypatch):
    """Con el audio trabado (la Boca no toma otra frase) el aviso se dice igual
    pasado el plazo, como quien espera el turno de voz del TTS."""
    from crotolamo.voice import threads
    from crotolamo.voice.threads import BrainThread

    monkeypatch.setattr(threads, "_ESPERA_BOCA_S", 0.2)
    monkeypatch.setattr(threads, "_ESPERA_BOCA_SONDEO_S", 0.02)
    state = SharedState()
    dichos: list[str] = []
    tts = SimpleNamespace(speak=lambda text, should_abort=None: dichos.append(text))

    def handle_turn(command, on_token=None):
        on_token("Voy a escribirlo. ")
        decir(_PREGUNTA)
        return ""

    brain = BrainThread(SimpleNamespace(handle_turn=handle_turn), queue.Queue(), queue.Queue(),
                        state, threading.Event(), stream=True)  # sin Boca: la cola no avanza
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state, lambda: brain)
    brain.start()
    try:
        brain.cmd_q.put(("apunta leche", state.new_turn(mode=Mode.THINKING)))
        assert _wait(lambda: dichos == [_PREGUNTA], timeout=3.0)
    finally:
        brain.shutdown.set()
        brain.join(2)


def test_run_listen_concurrente_en_streaming_el_aviso_va_detras_del_preambulo(
        monkeypatch, tmp_path):
    """El cableado de punta a punta: run_listen con stream_speak_concurrent arma
    el VoiceLoop de verdad y el aviso (el confirm_fn que recibe el agente) espera
    a lo que el Cerebro ya le mandó a la Boca, la última frase incluida."""
    preambulo = ["Claro que sí, patrón.", "Primero reviso la lista.", "Dame un segundo."]

    def handle_turn(agente, command, on_token):
        for token in " ".join(preambulo).split(" "):
            on_token(token + " ")
        agente.confirmaciones.append(agente.confirm_fn(_AVISO))  # ToolAgent._execute_call
        on_token("\n" + _RESPUESTA_VIEJA)
        return " ".join([*preambulo, _RESPUESTA_VIEJA])

    tts, _hud, _vl, agente = _correr_concurrente(
        monkeypatch, tmp_path, [], ["apunta algo en la lista vieja"], handle_turn,
        lambda tts, vl: (any(t == _RESPUESTA_VIEJA for t, _, _ in tts.dichos)
                         and _modo(vl.state) == "idle"),
        voz={"stream_speak_concurrent": True})
    assert agente.confirmaciones == [False]
    dichos = [(texto, hilo) for texto, modo, hilo in tts.dichos if modo is not None]
    aviso = _AVISO + " Esa está delicada, patrón: dímela por el shell y ahí la confirmo."
    assert dichos == [*[(frase, "Mouth") for frase in preambulo], (aviso, "Brain"),
                      (_RESPUESTA_VIEJA, "Mouth")]
    assert all(modo == "speaking" for _, modo, _ in tts.dichos if modo is not None)


def test_el_aviso_que_esperaba_a_la_boca_no_suena_si_llega_el_apagado():
    """El apagado suelta la espera del Cerebro, y el aviso que esperaba ya no se
    dice: si no, empezaba a sonar con el loop apagándose."""
    from crotolamo.voice.threads import BrainThread

    state = SharedState()
    dichos: list[str] = []
    tts = SimpleNamespace(speak=lambda text, should_abort=None: dichos.append(text))
    hecho = threading.Event()

    def handle_turn(command, on_token=None):
        on_token("Voy a escribirlo. ")
        decir(_PREGUNTA)
        hecho.set()
        return ""

    tts_q: queue.Queue = queue.Queue()  # sin Boca: la cola no avanza
    brain = BrainThread(SimpleNamespace(handle_turn=handle_turn), queue.Queue(), tts_q,
                        state, threading.Event(), stream=True)
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state, lambda: brain)
    brain.start()
    try:
        brain.cmd_q.put(("apunta leche", state.new_turn(mode=Mode.THINKING)))
        assert _wait(lambda: tts_q.qsize() == 2)  # la frase y la marca de paso
    finally:
        brain.shutdown.set()
    assert hecho.wait(3), "el Cerebro se quedó esperando a la Boca"
    brain.join(2)
    assert dichos == []
    assert state.get_mode() is Mode.THINKING
