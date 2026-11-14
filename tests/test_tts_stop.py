"""TTS interrumpible (M3.1), sin audio real: se parchea sounddevice.

El corte del camino que de verdad suena (_speak_streaming) está en
tests/test_auditoria_voice_tts.py; _play_interruptible, código muerto que se
saltaba el turno de voz, ya no existe (tests/test_revision_final_u38v.py).
"""

from pathlib import Path

import sounddevice as sd

from crotolamo.voice.tts import TTS


def test_stop_sets_flag(monkeypatch):
    monkeypatch.setattr(sd, "stop", lambda: None)
    tts = TTS(Path("x.onnx"))
    tts.stop()
    assert tts._stop_flag.is_set()
