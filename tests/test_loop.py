"""Tests del loop concurrente con fakes (sin micrófono ni audio real)."""

import time
import types

from crotolamo.voice.loop import VoiceLoop


class FakeTTS:
    def __init__(self):
        self.stopped = 0
        self.spoken = []

    def speak(self, sentence):
        self.spoken.append(sentence)

    def stop(self):
        self.stopped += 1


class FakeAgent:
    def __init__(self, reply):
        self.reply = reply
        self.conversation = types.SimpleNamespace(add_user=lambda _t: None)

    def handle_turn(self, command):
        return self.reply


class FakeWake:
    def __init__(self, wake=False):
        self._wake = wake

    def available(self):
        return True

    def listen_for_wake(self, timeout_s=None):
        time.sleep(0.05)  # evita busy-spin del thread de escucha en el test
        return self._wake


class FakeSTT:
    def listen_once(self, **kwargs):
        return ""


def _make(reply="Hola, patrón. ¿Qué tal?"):
    tts = FakeTTS()
    loop = VoiceLoop(FakeAgent(reply), FakeSTT(), tts, FakeWake())
    return loop, tts


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# --- barge-in (M3.2) ---
def test_barge_in_stops_tts_and_clears_queue():
    loop, tts = _make()
    loop.tts_queue.put("frase 1")
    loop.tts_queue.put("frase 2")
    loop.speaking.set()  # simula reproducción en curso

    assert loop.barge_in() is True
    assert tts.stopped == 1
    assert loop.tts_queue.empty()


def test_barge_in_noop_when_idle():
    loop, tts = _make()
    assert loop.barge_in() is False
    assert tts.stopped == 0


def test_barge_in_marks_interruption_in_history():
    marks = []
    loop, _ = _make()
    loop.agent.conversation = types.SimpleNamespace(add_user=marks.append)
    loop.tts_queue.put("algo")
    loop.speaking.set()
    loop.barge_in()
    assert marks and "interrump" in marks[0].lower()


# --- pipeline proceso -> habla (M3.1) ---
def test_command_flows_to_speech():
    loop, tts = _make(reply="Hola, patrón. Todo bien.")
    loop.start()
    try:
        loop.command_queue.put("saluda")
        # El thread de proceso parte la respuesta en frases y el de habla las dice.
        assert _wait(lambda: len(tts.spoken) >= 2)
        assert "Hola, patrón." in tts.spoken
    finally:
        loop.stop()


def test_stop_terminates_threads():
    loop, _ = _make()
    loop.start()
    loop.stop()
    assert all(not t.is_alive() for t in loop._threads)
