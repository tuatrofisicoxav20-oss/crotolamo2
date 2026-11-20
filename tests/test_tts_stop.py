"""TTS interrumpible (M3.1), sin audio real: sounddevice es un doble.

El corte del camino que de verdad suena (_speak_streaming) está en
tests/test_auditoria_voice_tts.py; _play_interruptible, código muerto que se
saltaba el turno de voz, ya no existe (tests/test_revision_final_u38v.py).

El doble entra por sys.modules y no parcheando el sounddevice de verdad: este
archivo lo importaba arriba del todo, y sounddevice inicializa PortAudio al
importarse. En una máquina sin servidor de audio (el runner ubuntu-26.04 de
GitHub, cuyo PortAudio trae la API de PulseAudio) eso es un PortAudioError: el
archivo no se podía ni recoger y pytest paraba ahí, sin correr un solo test.
"""

import sys
import types
from pathlib import Path

from crotolamo.voice.tts import TTS


def test_stop_sets_flag(monkeypatch):
    cortes: list[None] = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(stop=lambda: cortes.append(None)))
    tts = TTS(Path("x.onnx"))
    tts.stop()
    assert tts._stop_flag.is_set()
    assert len(cortes) == 1   # y corta lo que suena, no solo apunta la bandera


def test_stop_sin_audio_en_la_maquina_no_revienta(monkeypatch):
    """Sin PortAudio que inicialice no hay nada sonando que cortar: stop() apunta
    el corte y sigue (el barge-in y el apagado lo llaman sin mirar)."""

    class _SinAudio:
        def find_spec(self, nombre, path=None, target=None):
            if nombre == "sounddevice":
                raise OSError("PortAudio library not found")
            return None

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_SinAudio(), *sys.meta_path])
    tts = TTS(Path("x.onnx"))
    tts.stop()
    assert tts._stop_flag.is_set()
