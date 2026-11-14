"""Revisión final, unidad 38v: crotolamo/voice/tts.py.

1. Un stop() que cortaba una frase trabada lo deshacía el speak() siguiente. El
   sink de audio se cuelga dentro de stream.write (PipeWire reiniciado,
   Bluetooth que se fue), StreamSpeaker.finish() se rinde y _cancelar() llama a
   tts.stop(), que solo levantaba la bandera COMPARTIDA _stop_flag. La frase
   siguiente habla igual tras _VOZ_TRABADA_S (_tomar_voz) y, al empezar,
   limpiaba esa bandera: cuando el sink se destrababa, la frase cortada escribía
   el resto en su propio OutputStream A LA VEZ que la respuesta nueva. Ahora un
   stop() corta toda frase que ya tenía el turno, aunque otra haya limpiado la
   bandera (se cuentan los stop(): TTS._cortes). Lo mismo con un stop() que caía
   entre tomar el turno y limpiar la bandera (incluida la carga en frío de la
   voz), cuando quien llama no pasa should_abort (StreamSpeaker, say()).

2. TTS._play_interruptible era código muerto (speak() suena por
   _speak_streaming) que se saltaba el turno de voz, limpiaba la bandera
   compartida y usaba el stream global de sd.play, el mismo del bip: fuera.

3. TTS.warm no tenía quien la llamara: el listener calienta Piper por su
   cargador (tts._get_voice) a propósito, para que una voz rota llegue al
   WARNING de _Warmer; warm() la bajaba a DEBUG. Una sola entrada: el cargador.

Sin audio ni Piper reales: voz Piper y sd.OutputStream falsos.
"""

from __future__ import annotations

import importlib.machinery
import logging
import sys
import threading
import types

import pytest

from crotolamo.voice import tts as tts_mod
from crotolamo.voice.tts import TTS, StreamSpeaker

_REBANADA = 2048  # muestras por escritura de _speak_streaming


class _Chunk:
    sample_rate = 22050

    def __init__(self, n: int) -> None:
        import numpy as np

        self.audio_int16_array = np.zeros(n, dtype=np.int16)


class _Voz:
    """Voz Piper falsa: 3 chunks de 2 rebanadas cada uno (6 escrituras por frase)."""

    def __init__(self) -> None:
        self.sintetizadas: list[str] = []

    def synthesize(self, text):
        self.sintetizadas.append(text)
        for _ in range(3):
            yield _Chunk(2 * _REBANADA)


class _Salida:
    """sd.OutputStream falso: cada stream anota sus escrituras y llamadas;
    `al_escribir(stream, n)` corre en la escritura n (1, 2, ...) de cada uno."""

    creados: list[_Salida] = []
    al_escribir = None
    abiertos = 0
    max_abiertos = 0
    _guard = threading.Lock()

    def __init__(self, **kwargs) -> None:
        self.escrituras = 0
        self.llamadas: list[str] = []
        with _Salida._guard:
            _Salida.creados.append(self)

    def start(self) -> None:
        self.llamadas.append("start")
        with _Salida._guard:
            _Salida.abiertos += 1
            _Salida.max_abiertos = max(_Salida.max_abiertos, _Salida.abiertos)

    def write(self, audio) -> None:
        self.escrituras += 1
        if _Salida.al_escribir is not None:
            _Salida.al_escribir(self, self.escrituras)

    def abort(self) -> None:
        self.llamadas.append("abort")

    def stop(self) -> None:
        self.llamadas.append("stop")

    def close(self) -> None:
        self.llamadas.append("close")
        with _Salida._guard:
            _Salida.abiertos -= 1


@pytest.fixture
def salida(tmp_path, monkeypatch):
    """TTS real con voz y salida de audio falsas (sin Piper ni PortAudio)."""
    pytest.importorskip("numpy")
    sd = types.ModuleType("sounddevice")
    sd.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
    sd.OutputStream = _Salida  # type: ignore[attr-defined]
    sd.stop = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(_Salida, "creados", [])
    monkeypatch.setattr(_Salida, "al_escribir", None)
    monkeypatch.setattr(_Salida, "abiertos", 0)
    monkeypatch.setattr(_Salida, "max_abiertos", 0)
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    voz = _Voz()
    tts._voice = voz  # type: ignore[assignment]
    yield tts, voz
    _Salida.al_escribir = None
    tts.close()


# --- 1. un stop() corta la frase que ya sonaba aunque otra limpie la bandera ---


def test_el_stop_de_una_frase_trabada_no_lo_deshace_la_respuesta_siguiente(
        salida, monkeypatch, caplog):
    """El flujo del loop simple (un StreamSpeaker por turno): el sink se traba en
    la 2ª rebanada de la frase vieja, finish() se rinde y corta (stop()), y el
    turno siguiente habla igual tras _VOZ_TRABADA_S. Cuando el sink se destraba,
    la frase vieja no escribe ni una rebanada más (antes: las 4 que le quedaban,
    en su propio stream, encima de la respuesta nueva) y se aborta su stream."""
    tts, voz = salida
    monkeypatch.setattr(tts_mod, "_VOZ_TRABADA_S", 0.3)
    trabada, suelta = threading.Event(), threading.Event()

    def _escribir(stream: _Salida, n: int) -> None:
        if stream is _Salida.creados[0] and n == 2:
            trabada.set()
            suelta.wait(10)  # el sink colgado: ni stop() lo desbloquea

    _Salida.al_escribir = _escribir
    sp = StreamSpeaker(tts)
    try:
        sp.feed("¡Frase vieja que se traba! ")  # cerrada: el hilo ya la habla
        assert trabada.wait(5)  # el plazo de finish() corre ya con el sink colgado
        assert sp.finish(timeout_s=0.3) is True  # se rinde: _cancelar() -> stop()
        assert sp._thread.is_alive()
        caplog.set_level(logging.WARNING, logger="crotolamo.voice.tts")
        tts.speak("Respuesta nueva.")  # otro turno: habla igual tras la trabada
        assert caplog.text.count("sin avanzar") == 1
    finally:
        suelta.set()  # el sink se destraba
        sp._thread.join(5)
    assert not sp._thread.is_alive()
    vieja, nueva = _Salida.creados
    assert vieja.escrituras == 2  # la que estaba en vuelo, y ni una más
    assert vieja.llamadas == ["start", "abort", "stop", "close"]
    assert nueva.escrituras == 6 and "abort" not in nueva.llamadas
    assert voz.sintetizadas == ["¡Frase vieja que se traba!", "Respuesta nueva."]
    # Lo que se pide DESPUÉS de todo eso suena entero, como siempre.
    tts.speak("Ya.")
    assert _Salida.creados[-1].escrituras == 6


def test_un_stop_tras_tomar_el_turno_corta_la_frase_sin_should_abort(salida, monkeypatch):
    """StreamSpeaker y say() no pasan should_abort: un stop() que caía entre tomar
    el turno de voz y limpiar la bandera se perdía y la frase sonaba entera."""
    tts, voz = salida
    tomar = tts._tomar_voz

    def _tomar_y_cortan(should_abort):
        turno = tomar(should_abort)
        tts.stop()  # el barge-in llega justo ahí
        return turno

    monkeypatch.setattr(tts, "_tomar_voz", _tomar_y_cortan)
    tts.speak("Frase que cortaron al empezar.")
    assert _Salida.creados == []  # ni se abrió el audio...
    assert voz.sintetizadas == []  # ...ni se gastó CPU sintetizando
    assert tts._voz_dueno is None  # y el turno quedó libre


def test_un_stop_durante_la_carga_en_frio_de_la_voz_corta_la_frase(salida, monkeypatch):
    """La primera frase carga Piper (segundos en frío) DESPUÉS de tomar el turno:
    un stop() durante esa carga se perdía (la bandera se limpiaba al terminarla)."""
    tts, voz = salida
    tts._voice = None
    cargar = threading.Event()

    def _cargar_voz():
        cargar.set()
        tts.stop()  # el patrón corta mientras la voz carga
        return voz

    monkeypatch.setattr(tts, "_get_voice", _cargar_voz)
    tts.speak("Primera frase del día.")
    assert cargar.is_set()
    assert all(s.escrituras == 0 for s in _Salida.creados)


def test_un_stop_anterior_a_la_frase_no_la_corta(salida):
    """Lo que se pide DESPUÉS de un stop() suena entero: el corte es de lo que ya
    tenía el turno, no de lo que viene (por eso la bandera se limpia al entrar)."""
    tts, _voz = salida
    tts.stop()
    tts.speak("Después del corte.")
    assert tts._speak_streaming("Directa.") is True
    assert [s.escrituras for s in _Salida.creados] == [6, 6]
    assert all("abort" not in s.llamadas for s in _Salida.creados)


@pytest.mark.parametrize("trabada_en", [2, 6])
def test_el_corte_de_speak_streaming_no_depende_de_la_bandera_compartida(salida, trabada_en):
    """_speak_streaming por sí solo (sin turno de speak()): trabada en una
    escritura, le llega un stop() y otra frase limpia la bandera al empezar. Al
    destrabarse no escribe ni una rebanada más y aborta el stream (tira lo que
    queda en el búfer), también si la trabada era la ÚLTIMA escritura."""
    tts, _voz = salida
    trabada, suelta = threading.Event(), threading.Event()

    def _escribir(stream: _Salida, n: int) -> None:
        if stream is _Salida.creados[0] and n == trabada_en:
            trabada.set()
            suelta.wait(10)

    _Salida.al_escribir = _escribir
    resultado: list[bool] = []
    hilo = threading.Thread(
        target=lambda: resultado.append(tts._speak_streaming("Vieja.")), daemon=True)
    hilo.start()
    try:
        assert trabada.wait(5)
        tts.stop()
        tts._stop_flag.clear()  # lo que hace la frase siguiente al empezar
    finally:
        suelta.set()
        hilo.join(5)
    assert not hilo.is_alive()
    vieja = _Salida.creados[0]
    assert vieja.escrituras == trabada_en
    assert vieja.llamadas == ["start", "abort", "stop", "close"]
    assert resultado == [trabada_en == 6]  # cortada a media frase: False


# --- 2 y 3. código sin quien lo llame ---------------------------------------


def test_no_queda_un_segundo_camino_de_reproduccion_que_se_salte_el_turno():
    """speak() suena por _speak_streaming; _play_interruptible no lo llamaba nadie
    y, si alguien lo hiciera, sonaría sin turno de voz (dos voces a la vez),
    limpiaría la bandera de stop de otra frase y usaría el stream global de
    sd.play, el del bip."""
    assert not hasattr(TTS, "_play_interruptible")


def test_piper_se_calienta_solo_por_su_cargador():
    """El listener calienta Piper con tts._get_voice (un .onnx roto llega al
    WARNING de _Warmer); TTS.warm, sin quien la llamara, lo bajaba a DEBUG."""
    assert not hasattr(TTS, "warm")
