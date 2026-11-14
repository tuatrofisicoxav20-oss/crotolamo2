"""Causa raíz voice-pipeline-forked-audio-owned-per-call, parte 3 (pasos 5 y 6 del diseño).

(SentenceFeeder también en tests/test_speed.py, el BrainThread en streaming en
tests/test_loop.py y el Endpointer a secas en tests/test_vad_helper.py.)

Paso 5, un solo camino para hablar:
- SentenceFeeder: la segmentación en streaming es UNA (la usan StreamSpeaker y el
  BrainThread); salida idéntica a la de StreamSpeaker antes del paso.
- BrainThread en streaming ([voice].stream_speak_concurrent, apagada de fábrica):
  la primera frase va a la Boca antes de que el LLM termine; las de un turno
  abortado no; el aviso de un turno roto va detrás de lo dicho.
- Turno de voz del TTS: una frase suena a la vez, la pida quien la pida (antes dos
  hilos abrían dos OutputStream a la vez y la segunda deshacía el stop() de la
  primera). Una voz trabada no deja mudo al resto.
- Pre-síntesis ([voice].tts_presynth, apagada de fábrica): la frase siguiente se
  sintetiza mientras suena la actual, con el MISMO audio.

Paso 6, un Endpointer y una tabla de valores de fábrica (refactor sin cambio):
- La grabación por llamada, la de una fuente abierta y la rama LISTENING del oído
  REALES deciden chunk a chunk lo mismo que copias literales del código anterior.
- Los defaults salen de UNA tabla (voice/vad.py DEFAULTS) con los valores de antes;
  los literales de los loops son campos de ListenerConfig.
"""

from __future__ import annotations

import importlib.machinery
import logging
import queue
import random
import sys
import threading
import time
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo.voice import tts as tts_mod
from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.tts import TTS
from tests.test_vad_helper import _parametros, _probs_al_azar, _ref_oido, _ref_record_silero

_REPO = Path(__file__).resolve().parents[1]


def _esperar(cond, timeout: float = 3.0) -> bool:
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_run_listen_concurrente_solo_pasa_stream_speak_con_su_clave(monkeypatch, tmp_path):
    """[voice].stream_speak (true de fábrica) es del loop simple: el concurrente
    sigue respondiendo entero salvo con stream_speak_concurrent = true."""
    from crotolamo.voice import loop as voice_loop
    from interfaces import listener
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _BlockingDetector,
        _join_warmers,
    )

    for clave in (None, False, True):
        _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
        voz: dict = {"wake_duck": False, "stream_speak": True}
        if clave is not None:
            voz["stream_speak_concurrent"] = clave
        monkeypatch.setattr(listener, "get_settings",
                            lambda v=voz: SimpleNamespace(voice=v, wake={}))
        capturado: dict = {}

        class _Loop:
            def __init__(self, agent, stt, tts, wake_detector, **kwargs):
                capturado.update(kwargs)
                self.state = None

            def run(self):
                return 0

        monkeypatch.setattr(voice_loop, "VoiceLoop", _Loop)
        try:
            assert listener.run_listen([]) == 0
        finally:
            _join_warmers()
        if clave:
            assert capturado["stream_speak"] is True
        else:
            assert "stream_speak" not in capturado


def test_aviso_de_tool_delicada_en_streaming_suena_aunque_la_boca_ya_hable():
    """Con stream_speak_concurrent la Boca puede estar diciendo el principio del
    turno cuando el agente pide confirmar una tool delicada: el aviso suena (el
    turno de voz lo pone detrás), no se tira como si fuera de un turno abortado.
    Tras un barge-in (otro turno, LISTENING) sí se tira, como siempre."""
    from interfaces import listener

    dichos: list[str] = []
    tts = SimpleNamespace(speak=lambda text, should_abort=None: dichos.append(text))
    state = SharedState()
    turno = state.new_turn(mode=Mode.THINKING)
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state)
    assert state.set_mode_if_current(turno, Mode.SPEAKING)  # la Boca dice la 1ª frase
    decir("Esa está delicada.")
    assert dichos == ["Esa está delicada."]
    state.new_turn(mode=Mode.LISTENING)  # barge-in
    decir("Otro aviso.")
    assert dichos == ["Esa está delicada."]


# --- paso 5c: turno de voz del TTS ---------------------------------------------

class _Chunk:
    sample_rate = 22050

    def __init__(self, n: int, valor: int = 0) -> None:
        import numpy as np

        self.audio_int16_array = np.full(n, valor, dtype=np.int16)


class _Voz:
    """Voz Piper falsa: 2 chunks de 4096 muestras por frase (2 escrituras cada
    uno). El valor de las muestras es el largo del texto: así se sabe qué frase
    suena en cada escritura. `al_sintetizar(texto)` corre al empezar cada frase."""

    def __init__(self) -> None:
        self.sintetizadas: list[str] = []
        self.al_sintetizar = None
        self.falla: str | None = None

    def synthesize(self, text):
        self.sintetizadas.append(text)
        if self.al_sintetizar is not None:
            self.al_sintetizar(text)
        if text == self.falla:
            raise RuntimeError("piper reventó")
        for _ in range(2):
            yield _Chunk(4096, len(text))


class _Salida:
    """sd.OutputStream falso: anota escrituras (qué frase, por el valor) y cuántos
    streams hay abiertos a la vez. `al_escribir(frase)` corre en cada escritura."""

    abiertos = 0
    max_abiertos = 0
    escrituras: list[int] = []
    al_escribir = None
    _guard = threading.Lock()

    def __init__(self, **kwargs) -> None:
        pass

    def start(self) -> None:
        with _Salida._guard:
            _Salida.abiertos += 1
            _Salida.max_abiertos = max(_Salida.max_abiertos, _Salida.abiertos)

    def write(self, audio) -> None:
        with _Salida._guard:
            _Salida.escrituras.append(int(audio[0]))
        if _Salida.al_escribir is not None:
            _Salida.al_escribir(int(audio[0]))

    def abort(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        with _Salida._guard:
            _Salida.abiertos -= 1


@pytest.fixture
def salida_falsa(tmp_path, monkeypatch):
    """Fábrica de TTS reales con voz y salida de audio falsas (sin Piper ni PortAudio)."""
    pytest.importorskip("numpy")
    sd = types.ModuleType("sounddevice")
    sd.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
    sd.OutputStream = _Salida  # type: ignore[attr-defined]
    sd.stop = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(_Salida, "abiertos", 0)
    monkeypatch.setattr(_Salida, "max_abiertos", 0)
    monkeypatch.setattr(_Salida, "escrituras", [])
    monkeypatch.setattr(_Salida, "al_escribir", None)
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    creados: list[TTS] = []

    def _nuevo(presynth: bool = False) -> tuple[TTS, _Voz]:
        tts = TTS(modelo, presynth=presynth)
        voz = _Voz()
        tts._voice = voz  # type: ignore[assignment]
        creados.append(tts)
        return tts, voz

    yield _nuevo
    _Salida.al_escribir = None
    for tts in creados:
        tts.close()


def _en_hilo(fn, *args, **kwargs) -> threading.Thread:
    hilo = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    hilo.start()
    return hilo


def _esperando_turno(tts: TTS, n: int = 1) -> bool:
    """Hay `n` hilos dormidos esperando el turno de voz (sin sleeps a ciegas)."""
    return _esperar(lambda: len(tts._voz._waiters) == n)  # type: ignore[attr-defined]


def test_dos_hilos_que_hablan_no_suenan_encimados(salida_falsa):
    """La Boca (o StreamSpeaker) dice una frase y otro hilo pide hablar (el aviso
    de una tool delicada, la pregunta de voice_confirm): antes se abrían dos
    OutputStream a la vez. Ahora la segunda espera a que acabe la primera."""
    tts, _voz = salida_falsa()
    primera_suena = threading.Event()
    suelta = threading.Event()

    def _escribir(valor: int) -> None:
        if valor == len("Primera frase."):
            primera_suena.set()
            assert suelta.wait(5)

    _Salida.al_escribir = _escribir
    a = _en_hilo(tts.speak, "Primera frase.")
    assert primera_suena.wait(3)
    b = _en_hilo(tts.speak, "Otra.")
    assert _esperando_turno(tts)
    assert _Salida.abiertos == 1  # la segunda espera su turno
    suelta.set()
    a.join(3)
    b.join(3)
    assert not a.is_alive() and not b.is_alive()
    assert _Salida.max_abiertos == 1
    largo_1, largo_2 = len("Primera frase."), len("Otra.")
    assert _Salida.escrituras == [largo_1] * 4 + [largo_2] * 4


def test_stop_corta_tambien_la_frase_que_esperaba_turno(salida_falsa):
    """Antes, una frase que sonaba encimada con la cortada también se cortaba:
    stop() las alcanzaba a las dos. La que ahora espera turno tampoco suena."""
    tts, voz = salida_falsa()
    suena = threading.Event()

    def _escribir(valor: int) -> None:
        suena.set()
        tts._stop_flag.wait(3)  # suena hasta que la cortan

    _Salida.al_escribir = _escribir
    a = _en_hilo(tts.speak, "Una frase larga del turno viejo.")
    assert suena.wait(3)
    b = _en_hilo(tts.speak, "Detrás.")
    assert _esperando_turno(tts)
    tts.stop()
    a.join(3)
    b.join(3)
    assert len("Detrás.") not in _Salida.escrituras
    assert voz.sintetizadas == ["Una frase larga del turno viejo."]
    # Lo que se pide DESPUÉS del stop() suena, como siempre.
    _Salida.al_escribir = None
    tts.speak("Ya.")
    assert _Salida.escrituras[-4:] == [len("Ya.")] * 4


def test_should_abort_mientras_espera_turno_no_suena(salida_falsa):
    tts, voz = salida_falsa()
    suena, suelta = threading.Event(), threading.Event()

    def _escribir(valor: int) -> None:
        suena.set()
        assert suelta.wait(5)

    _Salida.al_escribir = _escribir
    a = _en_hilo(tts.speak, "Sonando.")
    assert suena.wait(3)
    vigente = {"ok": True}
    b = _en_hilo(tts.speak, "Del turno viejo.", should_abort=lambda: not vigente["ok"])
    assert _esperando_turno(tts)
    vigente["ok"] = False  # barge-in sin stop() (V4)
    suelta.set()
    a.join(3)
    b.join(3)
    assert voz.sintetizadas == ["Sonando."]


def test_una_voz_trabada_no_deja_mudos_a_los_demas(salida_falsa, monkeypatch, caplog):
    """Un sink trabado (PipeWire reiniciado, Bluetooth que se fue) deja una
    escritura colgada con el turno tomado: tras _VOZ_TRABADA_S sin avanzar, quien
    espera habla igual (como antes del turno), con un aviso; y el siguiente ya
    no espera a la trabada."""
    monkeypatch.setattr(tts_mod, "_VOZ_TRABADA_S", 0.3)
    tts, _voz = salida_falsa()
    trabada, suelta = threading.Event(), threading.Event()

    def _escribir(valor: int) -> None:
        if valor == len("Trabada."):
            trabada.set()
            suelta.wait(5)

    _Salida.al_escribir = _escribir
    a = _en_hilo(tts.speak, "Trabada.")
    try:
        assert trabada.wait(3)
        caplog.set_level(logging.WARNING, logger="crotolamo.voice.tts")
        t0 = time.monotonic()
        tts.speak("Hola.")
        assert 0.2 <= time.monotonic() - t0 < 3
        assert caplog.text.count("sin avanzar") == 1
        # El turno ya es de quien habló tras la trabada: el siguiente no la espera.
        tts.speak("Adiós.")
        assert caplog.text.count("sin avanzar") == 1
        assert _Salida.escrituras.count(len("Hola.")) == 4
        assert _Salida.escrituras.count(len("Adiós.")) == 4
    finally:
        suelta.set()
        a.join(3)


def test_sin_nadie_hablando_speak_suena_igual_que_antes(salida_falsa):
    tts, voz = salida_falsa()
    for frase in ("Una.", "Dos frases."):
        tts.speak(frase)
    assert _Salida.escrituras == [4] * 4 + [11] * 4
    assert voz.sintetizadas == ["Una.", "Dos frases."]
    assert tts._voz_dueno is None


def test_el_aviso_solo_devuelve_thinking_si_lo_cambio_el():
    """El aviso de una tool delicada vuelve a THINKING al acabar solo si él pasó
    el modo a SPEAKING (respuesta entera, lo de fábrica). Con la Boca ya hablando
    ESTE turno ([voice].stream_speak_concurrent) el SPEAKING es de la Boca: sus
    frases siguen y el END lo deja en IDLE."""
    from interfaces import listener

    modos: list[Mode] = []
    state = SharedState()
    tts = SimpleNamespace(speak=lambda text, should_abort=None: modos.append(state.get_mode()))
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state)
    state.new_turn(mode=Mode.THINKING)  # respuesta entera: nadie habla aún
    decir("Esa está delicada.")
    assert modos == [Mode.SPEAKING] and state.get_mode() is Mode.THINKING
    turno = state.new_turn(mode=Mode.THINKING)
    assert state.set_mode_if_current(turno, Mode.SPEAKING)  # la Boca dice la 1ª frase
    decir("Otra delicada.")
    assert modos[-1] is Mode.SPEAKING
    assert state.get_mode() is Mode.SPEAKING  # antes: THINKING bajo la frase siguiente


def test_en_streaming_la_frase_que_esperaba_tras_el_aviso_suena_en_speaking(salida_falsa):
    """[voice].stream_speak_concurrent: la Boca dice S1 del turno; el agente pide
    confirmar una tool delicada y el aviso espera turno de voz; S2 (encolada
    antes de la tool) espera también y el aviso se lo gana. Antes el aviso, al
    acabar, ponía THINKING y S2 sonaba con el HUD en "pensando" y, con
    --barge-in, el oído juzgaba su eco con las reglas de THINKING (V9)."""
    from crotolamo.voice.threads import END, STOP, MouthThread, Utterance
    from interfaces import listener

    tts, _voz = salida_falsa()
    state = SharedState()
    s1, s2, aviso = "Voy a borrar ese archivo.", "Dame un segundo, patrón.", "Borrar es delicado."
    s1_suena, suelta_s1, aviso_hecho = threading.Event(), threading.Event(), threading.Event()
    escrituras: list[tuple[int, Mode]] = []

    def _escribir(valor: int) -> None:
        if valor == len(s1):
            s1_suena.set()
            assert suelta_s1.wait(5)
        if valor == len(s2):
            assert aviso_hecho.wait(5)  # S2 suena con el aviso ya de vuelta
        escrituras.append((valor, state.get_mode()))

    _Salida.al_escribir = _escribir

    class _BocaQueLlegaTarde:
        """El tts de la Boca: para S2 ya puso SPEAKING y pide turno de voz cuando el
        aviso ya lo tomó (el orden que el turno de voz no fija)."""

        def speak(self, text, should_abort=None):
            if text == s2:
                assert _esperar(lambda: len(aviso) in _Salida.escrituras)
            tts.speak(text, should_abort=should_abort)

    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    boca = MouthThread(_BocaQueLlegaTarde(), q, state, shutdown)
    boca.start()
    turno = state.new_turn(mode=Mode.THINKING)
    decir = listener._make_notice_say(lambda t: None, tts, lambda: state)
    try:
        q.put(Utterance(s1, turno))
        assert s1_suena.wait(3)

        def _aviso() -> None:
            try:
                decir(aviso)
            finally:
                aviso_hecho.set()

        hilo = _en_hilo(_aviso)
        assert _esperando_turno(tts)  # el aviso espera a que acabe S1
        q.put(Utterance(s2, turno))
        q.put(END)
        suelta_s1.set()
        hilo.join(5)
        assert _esperar(lambda: state.get_mode() is Mode.IDLE)
    finally:
        suelta_s1.set()
        aviso_hecho.set()
        shutdown.set()
        q.put(STOP)
        boca.join(3)
    orden = [v for i, (v, _) in enumerate(escrituras) if i == 0 or v != escrituras[i - 1][0]]
    assert orden == [len(s1), len(aviso), len(s2)]
    assert {m for _, m in escrituras} == {Mode.SPEAKING}, escrituras


# --- paso 5d: pre-síntesis ([voice].tts_presynth) -------------------------------

def test_presintesis_apagada_de_fabrica_no_hace_nada(salida_falsa):
    tts, voz = salida_falsa()
    assert tts.presynth is False
    hilos = {h.name for h in threading.enumerate()}
    tts.prepare("Una.")
    tts.speak_sentences("Una. Dos.")
    assert "TtsPresintesis" not in {h.name for h in threading.enumerate()} - hilos
    assert tts._presint_hilo is None and not tts._presint
    assert voz.sintetizadas == ["Una.", "Dos."]


def test_presintesis_desde_la_config(tmp_path):
    def _settings(voz):
        return SimpleNamespace(paths={"voces": tmp_path}, voice=voz)

    assert TTS.from_settings(_settings({})).presynth is False
    assert TTS.from_settings(_settings({"tts_presynth": True})).presynth is True


@pytest.mark.parametrize("presynth", [False, True])
def test_presintesis_sintetiza_la_siguiente_mientras_suena_la_actual(salida_falsa, presynth):
    """Con la clave, "Dos." se sintetiza MIENTRAS suena "Una." (sin ella, después:
    el silencio entre frases). El audio escrito es el mismo con y sin."""
    tts, voz = salida_falsa(presynth=presynth)
    dos_lista = threading.Event()
    voz.al_sintetizar = lambda texto: dos_lista.set() if texto == "Dos." else None
    durante_una: list[bool] = []

    def _escribir(valor: int) -> None:
        if valor == len("Una.") and not durante_una:
            durante_una.append(dos_lista.wait(1.0 if presynth else 0.05))

    _Salida.al_escribir = _escribir
    tts.speak_sentences("Una. Dos. Tres.")
    assert durante_una == [presynth]
    assert _Salida.escrituras == [4] * 4 + [4] * 4 + [5] * 4
    assert voz.sintetizadas == ["Una.", "Dos.", "Tres."]


def test_presintesis_no_se_adelanta_mas_de_dos_frases(salida_falsa, monkeypatch):
    tts, voz = salida_falsa(presynth=True)
    for i in range(6):
        tts.prepare(f"Frase {i}.")
    assert _esperar(lambda: len(voz.sintetizadas) == 2)
    time.sleep(0.1)
    assert voz.sintetizadas == ["Frase 0.", "Frase 1."]  # CPU que un barge-in tiraría
    for i in range(6):
        tts.speak(f"Frase {i}.")
    assert voz.sintetizadas == [f"Frase {i}." for i in range(6)]
    assert _Salida.escrituras == [len("Frase 0.")] * 24


def test_presintesis_lo_de_un_turno_cortado_se_tira(salida_falsa):
    """stop() (barge-in) tira lo pedido; y si la Boca descarta frases de un turno
    viejo sin hablarlas, la siguiente que sí habla tira las de antes."""
    tts, voz = salida_falsa(presynth=True)
    tts.prepare("Vieja uno.")
    tts.prepare("Vieja dos.")
    tts.stop()
    assert not tts._presint
    tts.prepare("Saltada.")
    tts.prepare("Nueva.")
    tts.speak("Nueva.")
    assert not tts._presint
    assert _Salida.escrituras == [len("Nueva.")] * 4
    tts.speak("Suelta, sin pedir.")  # un say(): se sintetiza en el acto
    assert _Salida.escrituras[-4:] == [len("Suelta, sin pedir.")] * 4


def test_presintesis_que_falla_se_avisa_como_siempre_y_sigue(salida_falsa, caplog):
    tts, voz = salida_falsa(presynth=True)
    voz.falla = "Rota."
    caplog.set_level(logging.WARNING, logger="crotolamo.voice.tts")
    tts.speak_sentences("Rota. Sana.")
    assert "error sintetizando/reproduciendo con Piper: piper reventó" in caplog.text
    assert _Salida.escrituras == [len("Sana.")] * 4


def test_close_para_el_hilo_de_presintesis(salida_falsa):
    tts, _voz = salida_falsa(presynth=True)
    tts.speak("Hola.")
    hilo = tts._presint_hilo
    assert hilo is not None and hilo.is_alive()
    t0 = time.monotonic()
    tts.close()
    assert not hilo.is_alive() and time.monotonic() - t0 < 2.5
    tts.speak("Tras cerrar.")  # sin hilo: se sintetiza en el acto, como sin la clave
    assert _Salida.escrituras[-4:] == [len("Tras cerrar.")] * 4


def test_run_listen_cierra_el_tts_al_salir(monkeypatch, tmp_path):
    from interfaces import listener
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _BlockingDetector,
        _join_warmers,
    )

    _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
    cerrados: list[int] = []
    monkeypatch.setattr(listener, "TTS", SimpleNamespace(from_settings=lambda s: SimpleNamespace(
        available=lambda: False, speak=lambda text: None, close=lambda: cerrados.append(1))))
    monkeypatch.setattr(listener, "build_agent", lambda **k: (_ for _ in ()).throw(
        RuntimeError("sin agente")))
    try:
        assert listener.run_listen([]) == 1
    finally:
        _join_warmers()
    assert cerrados == [1]


def test_la_config_de_fabrica_documenta_las_claves_de_hablar_apagadas():
    texto = (_REPO / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    voz = tomllib.loads(texto)["voice"]
    assert voz["stream_speak_concurrent"] is False
    assert voz["tts_presynth"] is False
    for clave in ("stream_speak_concurrent", "tts_presynth"):
        bloque = texto.split(f"\n{clave} =")[0].rsplit("\n\n", 1)[-1]
        assert "MEDIDO" in bloque, clave


# --- paso 6a: UN Endpointer para las dos rutas de captura -----------------------
# Las referencias (copias LITERALES de las decisiones de antes) y las pruebas del
# Endpointer a secas están en tests/test_vad_helper.py; aquí, los grabadores REALES.

def _ref_silero_de_fuente(probs, respuestas, *, pendientes, threshold, neg, silence_chunks,
                          max_chunks, start_chunks, pre_chunks):
    """El bucle de STT._silero_de_fuente: (frames, first, leídos, on_idle, resets).
    `respuestas`: lo que contesta on_idle, en orden (agotadas: False). La fuente
    se cierra al acabarse `probs`."""
    from collections import deque

    from crotolamo.voice.vad import is_voice

    pre_buffer: deque = deque(maxlen=pre_chunks)
    frames: list = []
    first = 0
    speaking = False
    silent_run = 0
    ventana = pendientes
    leidos = llamadas = resets = 0
    respuestas = list(respuestas)
    i = -1
    while True:
        i += 1
        if i - ventana >= max_chunks:
            break
        if i >= len(probs):
            break
        block = i
        leidos += 1
        prob = probs[i]
        if not speaking:
            pre_buffer.append(block)
            if is_voice(prob, speaking=False, threshold=threshold, neg_threshold=neg):
                speaking = True
                first = i - len(pre_buffer) + 1
                frames.extend(pre_buffer)
                silent_run = 0
            elif i - ventana >= start_chunks:
                llamadas += 1
                if not (respuestas.pop(0) if respuestas else False):
                    break
                resets += 1
                ventana = i + 1
        else:
            frames.append(block)
            if is_voice(prob, speaking=True, threshold=threshold, neg_threshold=neg):
                silent_run = 0
            else:
                silent_run += 1
                if silent_run >= silence_chunks:
                    break
    return frames, first, leidos, llamadas, resets


class _SileroGuion:
    """Silero falso: devuelve las probabilidades del guion en orden."""

    def __init__(self, probs) -> None:
        self.probs = list(probs)
        self.llamadas = 0
        self.resets = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, x, sr) -> float:
        self.llamadas += 1
        return self.probs[self.llamadas - 1]


def _bloque(i: int):
    import numpy as np

    return np.full(512, float(i), dtype=np.float32)


def test_record_silero_real_decide_igual_que_antes(monkeypatch):
    """La grabación REAL por llamada (sounddevice y Silero falsos) frente a la
    copia de antes: mismos frames, mismas lecturas y una sola reset por llamada."""
    pytest.importorskip("numpy")
    from crotolamo.voice import stt as stt_mod

    rng = random.Random(26)
    for _ in range(300):
        p = _parametros(rng)
        probs = _probs_al_azar(rng, p["max_chunks"] + 5)
        silero = _SileroGuion(probs)
        leidos: list[int] = []

        class _Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def read(self, n):
                leidos.append(n)
                return _bloque(len(leidos) - 1).reshape(-1, 1), False

        sd = types.ModuleType("sounddevice")
        sd.InputStream = lambda **kw: _Stream()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sounddevice", sd)
        monkeypatch.setattr(stt_mod, "_silero_vad", silero)
        voz = {"vad_threshold": p["threshold"], "vad_neg_threshold": p["neg"],
               "vad_preactivation_ms": p["pre_chunks"] * 32}
        stt = stt_mod.STT(model_size="fake", initial_prompt=None)
        silence_ms = p["silence_chunks"] * 32
        frames = stt._record_silero(silence_ms, p["max_chunks"] * 0.032,
                                    p["start_chunks"] * 0.032, voz)
        ref_frames, _first, ref_leidos = _ref_record_silero(probs, **p)
        assert [int(f[0]) for f in frames] == ref_frames, (p, probs)
        assert len(leidos) == ref_leidos and silero.resets == 1


def test_silero_de_fuente_real_decide_igual_que_antes(monkeypatch):
    """La grabación REAL sobre una fuente abierta (lo ya capturado alarga los
    plazos; on_idle decide seguir en la misma) frente a la copia de antes."""
    pytest.importorskip("numpy")
    from crotolamo.voice import stt as stt_mod

    rng = random.Random(36)
    for _ in range(300):
        p = _parametros(rng)
        p["max_chunks"] = max(2, p["max_chunks"])
        probs = _probs_al_azar(rng, rng.randint(1, 160))
        respuestas = [rng.random() < 0.7 for _ in range(rng.randint(0, 4))]
        pendientes = rng.randint(0, 6)
        silero = _SileroGuion(probs)
        monkeypatch.setattr(stt_mod, "_silero_vad", silero)

        class _Fuente:
            closed = False

            def __init__(self) -> None:
                self.leidos = 0

            def pending(self, frames):
                return pendientes

            def read(self, timeout=1.0, frames=512):
                if self.leidos >= len(probs):
                    self.closed = True
                    return None
                self.leidos += 1
                return _bloque(self.leidos - 1)

        fuente = _Fuente()
        quedan = list(respuestas)
        llamadas: list[int] = []

        def on_idle() -> bool:
            llamadas.append(1)
            return quedan.pop(0) if quedan else False

        voz = {"vad_threshold": p["threshold"], "vad_neg_threshold": p["neg"],
               "vad_preactivation_ms": p["pre_chunks"] * 32}
        stt = stt_mod.STT(model_size="fake", initial_prompt=None)
        frames = stt._silero_de_fuente(p["silence_chunks"] * 32, p["max_chunks"] * 0.032,
                                       p["start_chunks"] * 0.032, voz, None, fuente, on_idle)
        ref = _ref_silero_de_fuente(probs, respuestas, pendientes=pendientes, **p)
        assert [int(f[0]) for f in frames] == ref[0], (p, pendientes, respuestas, probs)
        assert (fuente.leidos, len(llamadas), silero.resets - 1) == ref[2:]


def test_oido_real_decide_igual_que_antes():
    """El EarThread REAL (rama LISTENING tras un wake o un barge-in) frente a la
    copia de antes: mismos frames a Whisper, mismo chunk final, mismo modo."""
    from crotolamo.voice.threads import EarThread

    rng = random.Random(46)
    for _ in range(300):
        p = _parametros(rng)
        p["start_chunks"] = max(1, p["start_chunks"])
        voiced = rng.random() < 0.3
        preroll = ["b1", "b2", "b3"] if voiced else ["wake"]
        probs = _probs_al_azar(rng, 80)
        guion = {1000 + k: prob for k, prob in enumerate(probs)}
        state = SharedState()
        stt_q: queue.Queue = queue.Queue()
        ear = EarThread(None, lambda c: False, lambda c: guion[c], lambda f: list(f), None,
                        stt_q, queue.Queue(), state, threading.Event(),
                        vad_threshold=p["threshold"], vad_neg_threshold=p["neg"],
                        silence_ms=p["silence_chunks"] * 32,
                        max_command_s=p["max_chunks"] * 0.032,
                        start_timeout_s=p["start_chunks"] * 0.032,
                        preactivation_ms=p["pre_chunks"] * 32)
        assert (ear._silence_chunks, ear._max_chunks, ear._start_chunks, ear._pre_chunks) == (
            p["silence_chunks"], p["max_chunks"], p["start_chunks"], p["pre_chunks"])
        turno = state.new_turn(mode=Mode.LISTENING)
        ear._turn = turno
        ear._start_command(preroll=list(preroll), voiced=voiced)
        consumidos = 0
        for k in range(len(probs)):
            if state.get_mode() is not Mode.LISTENING:
                break
            ear._on_chunk(1000 + k)
            consumidos += 1
        ref_frames, ref_consumidos = _ref_oido(preroll, voiced, probs, **p)
        if ref_frames == "sigue":
            assert state.get_mode() is Mode.LISTENING and consumidos == len(probs)
            continue
        assert consumidos == ref_consumidos, (p, voiced, probs)
        if ref_frames is None:
            assert state.get_mode() is Mode.IDLE and stt_q.empty()
        else:
            assert state.get_mode() is Mode.THINKING
            assert stt_q.get_nowait() == (ref_frames, turno)


# --- paso 6b: UNA tabla de valores de fábrica ------------------------------------

def test_la_tabla_de_defaults_es_la_de_los_literales_de_antes():
    from crotolamo.voice.vad import DEFAULTS

    assert (DEFAULTS.sample_rate, DEFAULTS.vad_backend, DEFAULTS.vad_threshold,
            DEFAULTS.vad_neg_threshold, DEFAULTS.vad_preactivation_ms) == (
        16000, "energy", 0.8, None, 800)
    # Los que difieren entre lectores se guardan los dos (decisión K del dueño).
    assert (DEFAULTS.silence_ms_listener, DEFAULTS.silence_ms) == (800, 640)
    assert DEFAULTS.hotwords_concurrent == "Crotolamo, Tletl, Huevonitis"
    assert (DEFAULTS.barge_in_grace_ms, DEFAULTS.barge_in_threshold_margin,
            DEFAULTS.barge_in_min_chunks) == (400, 0.1, 5)
    assert (DEFAULTS.endpoint_rounds, DEFAULTS.endpoint_continue_s) == (2, 2.5)
    assert (DEFAULTS.command_start_timeout_s, DEFAULTS.max_command_s,
            DEFAULTS.post_speak_grace_s) == (4.0, 12.0, 1.5)


def test_listener_config_tiene_los_literales_de_los_loops_con_su_valor_de_antes():
    from interfaces.listener import ListenerConfig

    cfg = ListenerConfig()
    assert (cfg.command_start_timeout_s, cfg.wake_window_s, cfg.min_audio_s,
            cfg.confirm_max_s, cfg.post_reply_guard_s, cfg.post_speak_grace_s,
            cfg.max_command_s, cfg.silence_ms) == (4.0, 6, 0.3, 5, 0.5, 1.5, 12.0, 800)
    vacia = ListenerConfig.from_settings(SimpleNamespace(voice={}, wake={}))
    assert (vacia.silence_ms, vacia.hotwords, vacia.endpoint_rounds,
            vacia.endpoint_continue_s) == (800, None, 2, 2.5)


def test_los_modulos_toman_sus_defaults_de_la_tabla():
    """Ningún literal suelto: los defaults de firma son los de la tabla."""
    import inspect

    from crotolamo.voice.stt import STT
    from crotolamo.voice.threads import EarThread
    from crotolamo.voice.vad import DEFAULTS

    def defaults(fn) -> dict:
        return {k: v.default for k, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect.Parameter.empty}

    ear = defaults(EarThread.__init__)
    assert (ear["vad_threshold"], ear["silence_ms"], ear["max_command_s"],
            ear["barge_in_grace_ms"], ear["barge_in_threshold_margin"],
            ear["barge_in_min_chunks"], ear["start_timeout_s"], ear["preactivation_ms"],
            ear["post_speak_grace_s"], ear["vad_neg_threshold"]) == (
        DEFAULTS.vad_threshold, DEFAULTS.silence_ms, DEFAULTS.max_command_s,
        DEFAULTS.barge_in_grace_ms, DEFAULTS.barge_in_threshold_margin,
        DEFAULTS.barge_in_min_chunks, DEFAULTS.command_start_timeout_s,
        DEFAULTS.vad_preactivation_ms, DEFAULTS.post_speak_grace_s,
        DEFAULTS.vad_neg_threshold)
    vl = defaults(VoiceLoop.__init__)
    assert (vl["silence_ms"], vl["start_timeout_s"], vl["max_command_s"],
            vl["post_speak_grace_s"]) == (
        DEFAULTS.silence_ms, DEFAULTS.command_start_timeout_s, DEFAULTS.max_command_s,
        DEFAULTS.post_speak_grace_s)
    for fn in (STT.record_until_silence, STT.record_audio_until_silence, STT.listen_smart):
        d = defaults(fn)
        assert (d["max_seconds"], d["start_timeout_s"]) == (
            DEFAULTS.max_command_s, DEFAULTS.command_start_timeout_s), fn
    smart = defaults(STT.listen_smart)
    assert (smart["rounds"], smart["continue_timeout_s"]) == (
        DEFAULTS.endpoint_rounds, DEFAULTS.endpoint_continue_s)
    assert defaults(STT.__init__)["sample_rate"] == DEFAULTS.sample_rate


def test_sin_claves_cada_loop_conserva_su_default_de_hoy(monkeypatch):
    """voice-runtime#27: los dos loops leían defaults distintos sin la clave. Siguen
    siendo los de hoy (unificarlos es decisión del dueño), pero salen de la tabla."""
    from crotolamo import settings as crotolamo_settings

    class _Mic:
        def read(self):
            return None

    monkeypatch.setattr(crotolamo_settings, "get_settings",
                        lambda: SimpleNamespace(voice={}, wake={}))
    vl = VoiceLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), None, mic=_Mic(),
                   wake_fn=lambda c: False, vad_fn=lambda c: 0.0, to_wav=lambda f: f)
    ear, stt_thread = vl.threads[0], vl.threads[1]
    assert stt_thread.hotwords == "Crotolamo, Tletl, Huevonitis"
    assert (ear._silence_chunks, ear.vad_threshold, ear._pre_chunks) == (20, 0.8, 25)
    assert (ear._start_chunks, ear._max_chunks, ear._post_speak_grace_s) == (125, 375, 1.5)


def test_con_la_config_de_fabrica_ningun_default_de_la_tabla_se_alcanza():
    texto = (_REPO / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    voz = tomllib.loads(texto)["voice"]
    for clave in ("sample_rate", "vad_backend", "vad_threshold", "vad_neg_threshold",
                  "vad_preactivation_ms", "vad_silence_ms", "hotwords", "barge_in_grace_ms",
                  "barge_in_threshold_margin", "barge_in_min_chunks", "endpoint_max_rounds",
                  "endpoint_continue_s"):
        assert clave in voz, clave


def test_run_listen_concurrente_pasa_los_literales_de_listener_config(monkeypatch, tmp_path):
    from crotolamo.voice import loop as voice_loop
    from interfaces import listener
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _BlockingDetector,
        _join_warmers,
    )

    _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
    capturado: dict = {}

    class _Loop:
        def __init__(self, agent, stt, tts, wake_detector, **kwargs):
            capturado.update(kwargs)
            self.state = None

        def run(self):
            return 0

    monkeypatch.setattr(voice_loop, "VoiceLoop", _Loop)
    try:
        assert listener.run_listen([]) == 0
    finally:
        _join_warmers()
    assert (capturado["start_timeout_s"], capturado["max_command_s"],
            capturado["post_speak_grace_s"], capturado["silence_ms"]) == (4.0, 12.0, 1.5, 800)


def test_el_loop_simple_usa_los_literales_de_listener_config(monkeypatch):
    """Cada literal del loop simple es ahora un campo: cambiarlo es una línea (las
    decisiones del dueño), y con los valores de fábrica las llamadas son las de
    siempre (lo fijan los tests de test_auditoria_voice_listener)."""
    from dataclasses import replace

    from interfaces import listener
    from tests.test_auditoria_voice_listener import _run_simple

    llamadas: list[tuple[str, dict]] = []
    oidos = iter(["crotolamo", ""])

    class _Wake:
        def listen_once(self, **kw):
            llamadas.append(("wake", kw))
            texto = next(oidos, None)
            if texto is None:
                raise KeyboardInterrupt
            return texto

    class _Orden:
        def listen_smart(self, **kw):
            llamadas.append(("orden", kw))
            return "pausa la música" if len([n for n, _ in llamadas if n == "orden"]) == 1 else ""

    pausas: list[float] = []
    cfg = replace(listener.ListenerConfig(threshold=0.5), wake_window_s=5, min_audio_s=0.2,
                  command_start_timeout_s=3.0, post_reply_guard_s=0.25, followup_s=2.0,
                  stream_speak=False)
    _run_simple(monkeypatch, cfg=cfg, wake_stt=_Wake(), stt=_Orden(), sleep=pausas.append)
    wake_kw = llamadas[0][1]
    assert (wake_kw["start_timeout_s"], wake_kw["min_audio_s"]) == (5, 0.2)
    orden_kw = [kw for n, kw in llamadas if n == "orden"]
    assert (orden_kw[0]["start_timeout_s"], orden_kw[0]["min_audio_s"]) == (3.0, 0.2)
    assert orden_kw[1]["start_timeout_s"] == 2.0  # el followup, como siempre
    assert 0.25 in pausas
