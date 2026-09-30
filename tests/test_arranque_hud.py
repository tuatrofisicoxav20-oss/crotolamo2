"""Arranque y apagado del listener vistos desde el HUD/panel.

Tres arreglos de la revisión:
- El loop publica el estado inicial SIEMPRE (un hud_state.json rancio de un
  proceso que murió a medio turno dejaba el HUD en "PENSANDO" para siempre).
- El idle final que escribe el apagado lleva `enabled`, como el resto de
  publicaciones (sin él, el panel mostraba la escucha "activa" aunque estuviera
  pausada).
- `WakeWordDetector.available()` trata la falta de libportaudio (OSError al
  importar sounddevice) como "no disponible" y no como traceback.
"""

from __future__ import annotations

import builtins
import json
import threading
import time
from types import SimpleNamespace

from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.wakeword import WakeWordDetector

from interfaces import listener


class _FakeStt:
    def transcribe(self, audio, hotwords=None):
        return ""

    def _frames_to_audio(self, frames):
        return b""


class _FakeTts:
    def speak(self, text):
        pass

    def stop(self):
        pass


class _FakeMic:
    def read(self):
        return None


def test_voice_loop_publica_el_estado_inicial_al_arrancar(tmp_path):
    publicado: list[dict] = []
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _FakeStt(), _FakeTts(), None,
        mic=_FakeMic(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda f: b"", hud_publisher=publicado.append,
        control_path=tmp_path / "control.json",  # ausente => escucha activa
    )
    hilo = threading.Thread(target=vl.run, daemon=True)
    hilo.start()
    fin = time.monotonic() + 2.0
    while not publicado and time.monotonic() < fin:
        time.sleep(0.01)
    vl.shutdown.set()
    hilo.join(timeout=3.0)
    assert publicado, "el loop no publicó nada al arrancar"
    assert publicado[0]["mode"] == "idle"
    assert publicado[0]["enabled"] is True


def test_idle_final_lleva_enabled_del_canal_de_control(tmp_path, monkeypatch):
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: False)
    destino = tmp_path / "hud_state.json"
    listener._write_idle_hud(destino)
    estado = json.loads(destino.read_text(encoding="utf-8"))
    assert estado["mode"] == "idle"
    assert estado["enabled"] is False
    assert {"mode", "turn_id", "text", "enabled", "ts", "pid"} <= set(estado)


def test_available_es_false_si_falta_libportaudio(monkeypatch):
    real_import = builtins.__import__

    def _sin_portaudio(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _sin_portaudio)
    assert WakeWordDetector().available() is False
