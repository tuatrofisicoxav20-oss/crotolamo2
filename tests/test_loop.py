"""Tests del loop concurrente con fakes (sin micrófono ni audio real).

Todos los tests de threads usan fakes y terminan con shutdown.set() + join(timeout).
Ningún test toca audio real ni debe colgar.
"""

import queue
import threading
import time

from pathlib import Path

from crotolamo.voice.loop import (
    END,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
)
from crotolamo.voice.state import Mode, SharedState


# --- fakes compartidos ---
class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# --- M3.3: MouthThread ---
def test_mouth_discards_old_turn():
    tts = FakeTts()
    state = SharedState()
    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    mouth = MouthThread(tts, q, state, shutdown)
    mouth.start()
    try:
        # Avanzamos a turno 2 ANTES de encolar, para que el test sea determinista.
        state.new_turn()
        state.new_turn()  # turno actual = 2
        q.put(Utterance("vieja", 1))   # de un turno abortado -> debe descartarse
        q.put(Utterance("nueva", 2))   # vigente -> debe hablarse
        assert _wait(lambda: tts.spoken == ["nueva"] and q.empty())
    finally:
        shutdown.set()
        mouth.join(timeout=2.0)
    assert not mouth.is_alive()
    assert tts.spoken == ["nueva"]


def test_mouth_end_sentinel_returns_to_idle():
    tts = FakeTts()
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    mouth = MouthThread(tts, q, state, shutdown)
    mouth.start()
    try:
        q.put(END)
        assert _wait(lambda: state.get_mode() is Mode.IDLE)
    finally:
        shutdown.set()
        mouth.join(timeout=2.0)


# --- M3.4: BrainThread ---
class FakeAgent:
    def __init__(self, reply):
        self.reply = reply

    def handle_turn(self, command):
        return self.reply


class GatedAgent:
    """Agente que se bloquea hasta que el test lo libera (para forzar el race)."""

    def __init__(self, reply):
        self.reply = reply
        self.started = threading.Event()
        self.release = threading.Event()

    def handle_turn(self, command):
        self.started.set()
        self.release.wait(timeout=2.0)
        return self.reply


def test_brain_enqueues_sentences_when_turn_unchanged():
    state = SharedState()
    state.new_turn()  # actual = 1
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(FakeAgent("Hola. Mundo."), cmd_q, tts_q, state, shutdown)
    brain.start()
    try:
        cmd_q.put(("saluda", 1))
        assert _wait(lambda: tts_q.qsize() >= 3)
        got = [tts_q.get() for _ in range(3)]
        assert [u.text for u in got[:2]] == ["Hola.", "Mundo."]
        assert got[-1] is END
    finally:
        shutdown.set()
        brain.join(timeout=2.0)


def test_brain_drops_reply_if_turn_changed():
    state = SharedState()
    state.new_turn()  # actual = 1
    agent = GatedAgent("Hola. Mundo.")
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(agent, cmd_q, tts_q, state, shutdown)
    brain.start()
    try:
        cmd_q.put(("saluda", 1))
        assert agent.started.wait(timeout=2.0)  # el brain ya está "pensando"
        state.new_turn()                         # barge-in: actual = 2, invalida turno 1
        agent.release.set()                      # dejamos terminar la inferencia vieja
        time.sleep(0.1)
        assert tts_q.empty()                     # su respuesta NO se encoló
    finally:
        shutdown.set()
        brain.join(timeout=2.0)


# --- M3.5: SttThread ---
class FakeStt:
    def __init__(self, text="abre youtube"):
        self.text = text
        self.transcribed = 0

    def transcribe(self, path):
        self.transcribed += 1
        return self.text

    def listen_once(self, **kwargs):
        return ""


_NO_WAV = Path("/tmp/crotolamo_no_existe.wav")


def test_stt_passes_text_for_current_turn():
    state = SharedState()
    state.new_turn()  # actual = 1
    stt = FakeStt("abre youtube")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    th = SttThread(stt, in_q, out_q, state, shutdown)
    th.start()
    try:
        in_q.put((_NO_WAV, 1))
        assert _wait(lambda: not out_q.empty())
        text, turn = out_q.get()
        assert text == "abre youtube" and turn == 1
    finally:
        shutdown.set()
        th.join(timeout=2.0)


def test_stt_drops_if_turn_changed():
    state = SharedState()
    state.new_turn()
    state.new_turn()  # actual = 2; el audio será de turno 1 (abortado)
    stt = FakeStt("lo que sea")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    th = SttThread(stt, in_q, out_q, state, shutdown)
    th.start()
    try:
        in_q.put((_NO_WAV, 1))
        time.sleep(0.15)
        assert out_q.empty()
        assert stt.transcribed == 0  # ni siquiera se transcribió
    finally:
        shutdown.set()
        th.join(timeout=2.0)


# --- M3.6: EarThread (con micrófono fake) ---
class FakeMic:
    """Entrega una secuencia fija de 'chunks' (etiquetas) y luego None."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def read(self):
        try:
            return next(self._it)
        except StopIteration:
            return None


def _ear(mic, state, *, allow_barge_in=False, silence_ms=64, tts=None,
         wake_label="wake", voice_label="voice"):
    stt_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    ear = EarThread(
        mic,
        wake_fn=lambda c: c == wake_label,
        vad_fn=lambda c: 0.9 if c == voice_label else 0.0,
        to_wav=lambda frames: _NO_WAV,
        tts=tts or FakeTts(),
        stt_queue=stt_q, tts_queue=tts_q, state=state, shutdown=shutdown,
        silence_ms=silence_ms, allow_barge_in=allow_barge_in,
    )
    return ear, stt_q, tts_q, shutdown


def test_ear_wake_in_idle_starts_listening():
    state = SharedState()  # IDLE, turno 0
    ear, _stt_q, _tts_q, shutdown = _ear(FakeMic(["wake"]), state)
    ear.start()
    try:
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        assert state.turn_id == 1
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_silence_ends_command_and_enqueues():
    state = SharedState()
    # silence_ms=64 con chunk 32ms -> 2 chunks de silencio cierran el comando.
    mic = FakeMic(["wake", "voice", "sil", "sil"])
    ear, stt_q, _tts_q, shutdown = _ear(mic, state, silence_ms=64)
    ear.start()
    try:
        assert _wait(lambda: not stt_q.empty())
        wav, turn = stt_q.get()
        assert turn == 1
        assert _wait(lambda: state.get_mode() is Mode.THINKING)
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_barge_in_during_speaking():
    state = SharedState()
    state.new_turn()              # turno 1
    state.set_mode(Mode.SPEAKING)  # Crotolamo está hablando
    tts = FakeTts()
    ear, _stt_q, tts_q, shutdown = _ear(
        FakeMic(["voice"]), state, allow_barge_in=True, tts=tts
    )
    tts_q.put(Utterance("frase en curso", 1))  # algo en la cola de habla
    ear.start()
    try:
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        assert tts.stopped >= 1          # cortó el audio
        assert tts_q.empty()             # vació la cola de habla
        assert state.turn_id == 2        # turno nuevo invalida lo viejo
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_no_barge_in_when_disabled():
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    tts = FakeTts()
    ear, _stt_q, _tts_q, shutdown = _ear(
        FakeMic(["voice", "voice", "voice"]), state, allow_barge_in=False, tts=tts
    )
    ear.start()
    try:
        time.sleep(0.15)
        assert state.get_mode() is Mode.SPEAKING  # half-duplex: ignora la voz
        assert tts.stopped == 0
    finally:
        shutdown.set()
        ear.join(timeout=2.0)
