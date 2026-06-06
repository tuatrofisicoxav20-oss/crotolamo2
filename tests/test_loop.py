"""Tests del loop concurrente con fakes (sin micrófono ni audio real).

Todos los tests de threads usan fakes y terminan con shutdown.set() + join(timeout).
Ningún test toca audio real ni debe colgar.
"""

import queue
import threading
import time

from crotolamo.voice.loop import END, BrainThread, MouthThread, Utterance
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
