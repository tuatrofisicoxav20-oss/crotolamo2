"""TTS interrumpible (M3.1), sin audio real: se parchea sounddevice."""

from pathlib import Path

import sounddevice as sd

from crotolamo.voice.tts import TTS


class _FakeStream:
    def __init__(self, active: bool) -> None:
        self._active = active

    @property
    def active(self) -> bool:
        return self._active


def test_stop_sets_flag(monkeypatch):
    monkeypatch.setattr(sd, "stop", lambda: None)
    tts = TTS(Path("x.onnx"))
    tts.stop()
    assert tts._stop_flag.is_set()


def test_play_interruptible_returns_false_when_stopped(monkeypatch):
    tts = TTS(Path("x.onnx"))
    calls = {"n": 0}

    def fake_sleep(_ms):
        calls["n"] += 1
        if calls["n"] >= 2:
            tts.stop()  # simula barge-in externo a mitad de frase

    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(sd, "get_stream", lambda: _FakeStream(True))  # nunca termina solo
    monkeypatch.setattr(sd, "stop", lambda: None)
    monkeypatch.setattr(sd, "sleep", fake_sleep)

    assert tts._play_interruptible([1, 2, 3], 22050) is False


def test_play_interruptible_completes_returns_true(monkeypatch):
    tts = TTS(Path("x.onnx"))
    actives = [True, True, False]
    state = {"i": 0}

    def fake_get_stream():
        return _FakeStream(actives[min(state["i"], len(actives) - 1)])

    def fake_sleep(_ms):
        state["i"] += 1

    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(sd, "get_stream", fake_get_stream)
    monkeypatch.setattr(sd, "sleep", fake_sleep)
    monkeypatch.setattr(sd, "stop", lambda: None)

    assert tts._play_interruptible([1], 22050) is True
