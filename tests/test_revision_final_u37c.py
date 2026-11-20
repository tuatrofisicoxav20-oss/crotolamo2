"""Revisión final u37c: la Boca (MouthThread) ata su speak() al turno (V4).

TTS.speak acepta `should_abort` desde el arreglo V4 porque un stop() SOLO no
basta: corta lo que ya tenía el turno de voz, no lo que llega detrás. La Boca,
que es quien sabe de turnos, nunca lo pasaba. Con --barge-in, si el barge-in
del oído (tts.stop(), vaciar la cola, new_turn(LISTENING)) caía entre el
set_mode_if_current(turno, SPEAKING) de la Boca y la toma del turno de voz en
speak(), ese stop() se perdía: la frase del turno abortado sonaba ENTERA encima
de la orden nueva y el oído, ya en LISTENING, la grababa dentro de esa orden.

Sin audio ni Piper reales: el TTS es el real con la voz y el OutputStream de
mentira (como en test_auditoria_voice_tts.py) y el barge-in es el del EarThread
real, disparado justo después de que la Boca compruebe su turno.
"""

from __future__ import annotations

import importlib.machinery
import queue
import sys
import threading
import types

import pytest

from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import END, STOP, EarThread, MouthThread, Utterance
from crotolamo.voice.tts import TTS


class _Chunk:
    sample_rate = 22050

    def __init__(self, n: int) -> None:
        import numpy as np

        self.audio_int16_array = np.zeros(n, dtype=np.int16)


class _FakeVoice:
    """Voz Piper falsa: 3 chunks de 4096 muestras; anota lo que sintetiza."""

    def __init__(self) -> None:
        self.sintetizado: list[str] = []

    def synthesize(self, text):
        self.sintetizado.append(text)
        for _ in range(3):
            yield _Chunk(4096)


class _FakeOutputStream:
    instances: list[_FakeOutputStream] = []

    def __init__(self, **kwargs) -> None:
        self.writes: list[int] = []
        _FakeOutputStream.instances.append(self)

    def start(self) -> None:
        pass

    def write(self, audio) -> None:
        self.writes.append(len(audio))

    def abort(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def tts_falso(tmp_path, monkeypatch):
    """TTS real con voz y salida de audio falsas (sin PortAudio ni Piper)."""
    pytest.importorskip("numpy")
    sd = types.ModuleType("sounddevice")
    sd.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
    sd.OutputStream = _FakeOutputStream  # type: ignore[attr-defined]
    sd.stop = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    _FakeOutputStream.instances = []
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    voz = _FakeVoice()
    tts._voice = voz  # type: ignore[assignment]
    return tts, voz


def _oido_con_barge_in(tts, tts_q: queue.Queue, state: SharedState) -> EarThread:
    """El oído real con barge-in: un solo chunk con voz basta y sin gracia."""
    return EarThread(None, lambda _c: False, lambda _c: 1.0, lambda frames: frames, tts,
                     queue.Queue(), tts_q, state, threading.Event(), allow_barge_in=True,
                     barge_in_grace_ms=0.0, barge_in_min_chunks=1)


def test_barge_in_tras_el_check_de_la_boca_no_deja_sonar_la_frase_vieja(tts_falso):
    """El barge-in del oído cae justo DESPUÉS de que la Boca puso SPEAKING para su
    turno (todavía vigente) y ANTES de que speak() tome el turno de voz. Antes la
    frase del turno abortado se sintetizaba y sonaba entera (6 escrituras) con el
    loop ya escuchando la orden nueva; ahora ni se sintetiza ni abre audio."""
    import numpy as np

    tts, voz = tts_falso
    tts_q: queue.Queue = queue.Queue()

    class _BargeInTrasElCheck(SharedState):
        oido: EarThread | None = None
        disparado = False

        def set_mode_if_current(self, turn: int, mode: Mode) -> bool:
            ok = super().set_mode_if_current(turn, mode)
            if ok and mode is Mode.SPEAKING and not self.disparado:
                self.disparado = True
                assert self.oido is not None
                # El oído oye al patrón con la Boca en SPEAKING: su barge-in real.
                self.oido._on_chunk(np.zeros(512, dtype=np.float32))
            return ok

    state = _BargeInTrasElCheck()
    state.oido = _oido_con_barge_in(tts, tts_q, state)
    turno = state.new_turn(mode=Mode.THINKING)
    tts_q.put(Utterance("Frase del turno viejo.", turno))
    tts_q.put(END)
    tts_q.put(STOP)
    MouthThread(tts, tts_q, state, threading.Event()).run()

    assert state.disparado  # el barge-in de verdad cayó en el hueco
    assert state.get_mode() is Mode.LISTENING
    assert not state.is_current(turno)
    assert voz.sintetizado == []
    assert sum(sum(s.writes) for s in _FakeOutputStream.instances) == 0


def test_la_frase_del_turno_vigente_suena_entera(tts_falso):
    """Sin barge-in, atar el speak() al turno no cambia nada: suena entera y el
    END la deja en reposo."""
    tts, voz = tts_falso
    tts_q: queue.Queue = queue.Queue()
    state = SharedState()
    turno = state.new_turn(mode=Mode.THINKING)
    tts_q.put(Utterance("Hola, patrón.", turno))
    tts_q.put(END)
    tts_q.put(STOP)
    MouthThread(tts, tts_q, state, threading.Event()).run()

    assert voz.sintetizado == ["Hola, patrón."]
    assert sum(sum(s.writes) for s in _FakeOutputStream.instances) == 3 * 4096
    assert state.get_mode() is Mode.IDLE


def test_should_abort_de_la_boca_sigue_al_turno_de_su_frase():
    """El should_abort que recibe un TTS que lo acepta es el de ESA frase: falso
    mientras su turno es el vigente, verdadero en cuanto otro turno lo releva."""
    state = SharedState()
    turno = state.new_turn(mode=Mode.THINKING)
    vistos: list[tuple[str, bool, bool]] = []

    class _TTSConCorte:
        def speak(self, text, *, should_abort=None):
            antes = should_abort()
            state.new_turn(mode=Mode.LISTENING)  # un barge-in mientras suena
            vistos.append((text, antes, should_abort()))

    tts_q: queue.Queue = queue.Queue()
    tts_q.put(Utterance("Una frase.", turno))
    tts_q.put(STOP)
    MouthThread(_TTSConCorte(), tts_q, state, threading.Event()).run()

    assert vistos == [("Una frase.", False, True)]


def test_un_tts_que_solo_acepta_el_texto_recibe_la_llamada_de_siempre():
    """Los TTS de mentira y el protocolo TextToSpeech: speak(text), sin más."""
    state = SharedState()
    turno = state.new_turn(mode=Mode.THINKING)
    dichas: list[str] = []

    class _TTSSoloTexto:
        def speak(self, text):
            dichas.append(text)

    tts_q: queue.Queue = queue.Queue()
    tts_q.put(Utterance("Una frase.", turno))
    tts_q.put(STOP)
    MouthThread(_TTSSoloTexto(), tts_q, state, threading.Event()).run()

    assert dichas == ["Una frase."]
