"""Tests de huecos críticos del loop de voz (con fakes, sin audio real).

1. Cooldown anti-eco del EarThread: al volver a IDLE tras un turno, el wake se
   IGNORA durante _post_speak_grace_s (el eco del TTS Piper dispara falsos wakes
   porque el modelo se entrenó con voz Piper); pasado el período, se acepta.
2. Fallback Silero -> energía en _SileroVad: si torch/numpy o _get_silero_vad
   fallan, _fallback se activa y el VAD de energía distingue voz de silencio.
"""

from __future__ import annotations

import queue
import threading
import time

import crotolamo.voice.adapters as adapters
from crotolamo.voice.adapters import _SileroVad
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import EarThread


class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


class FeedMic:
    """Micrófono controlado por el test: entrega chunks encolados o None."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def feed(self, chunk, times=1):
        for _ in range(times):
            self.q.put(chunk)

    def read(self):
        try:
            return self.q.get(timeout=0.02)
        except queue.Empty:
            return None


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _ear_con_mic(mic, state, grace_s):
    """EarThread con fakes, siguiendo el patrón de test_loop.py."""
    stt_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    ear = EarThread(
        mic,
        wake_fn=lambda c: c == "wake",
        vad_fn=lambda c: 0.9 if c == "voice" else 0.0,
        to_wav=lambda frames: "WAV",
        tts=FakeTts(),
        stt_queue=stt_q, tts_queue=tts_q, state=state, shutdown=shutdown,
    )
    ear._post_speak_grace_s = grace_s
    return ear, shutdown


# --- cooldown anti-eco (loop de "habla -> falso wake -> habla...") ---
def test_cooldown_ignora_el_wake_durante_la_gracia_post_habla():
    """En IDLE recién salido de SPEAKING, el wake DEBE ignorarse (eco del TTS)."""
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    mic = FeedMic()
    ear, shutdown = _ear_con_mic(mic, state, grace_s=30.0)  # gracia enorme
    ear.start()
    try:
        # 1) Un chunk en SPEAKING para que _prev_mode registre el habla.
        mic.feed("sil")
        time.sleep(0.1)
        # 2) Fin del turno: a IDLE. El siguiente chunk arranca el cooldown.
        state.set_mode(Mode.IDLE)
        # 3) Eco del TTS: chunks que disparan la wake word dentro de la gracia.
        mic.feed("wake", times=10)
        time.sleep(0.3)
        assert state.get_mode() is Mode.IDLE, "el wake en gracia NO debe despertar"
        assert state.turn_id == 0, "no debe abrirse ningún turno durante la gracia"
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_cooldown_acepta_el_wake_pasada_la_gracia():
    """Pasado _post_speak_grace_s, el mismo wake SÍ debe despertar al asistente."""
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    mic = FeedMic()
    ear, shutdown = _ear_con_mic(mic, state, grace_s=0.2)  # gracia corta y real
    ear.start()
    try:
        mic.feed("sil")
        time.sleep(0.1)
        state.set_mode(Mode.IDLE)
        # Dentro de la gracia: ignorado (y de paso arranca el cooldown).
        mic.feed("wake")
        time.sleep(0.05)
        assert state.get_mode() is Mode.IDLE
        # Esperar a que la gracia expire y volver a intentar.
        time.sleep(0.3)
        mic.feed("wake", times=3)
        assert _wait(lambda: state.get_mode() is Mode.LISTENING), \
            "pasada la gracia, el wake DEBE despertar"
        assert state.turn_id == 1
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


# --- fallback Silero -> energía en _SileroVad ---
_VOZ = [0.5] * 512       # chunk de alta energía (RMS >> _ENERGY_FLOOR)
_SILENCIO = [0.0] * 512  # chunk de silencio total


def test_silero_vad_cae_a_energia_si_falta_la_dependencia(monkeypatch):
    """ImportError al cargar Silero => _fallback y VAD de energía funcional."""
    def _sin_silero():
        raise ImportError("no hay silero-vad, patrón")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _sin_silero)
    vad = _SileroVad()
    # El primer chunk dispara el intento de carga, falla y ya responde por energía.
    assert vad(_VOZ) == 1.0
    assert vad._fallback is True
    assert vad._model is None
    # Ya en fallback: alta energía = voz, silencio = no voz.
    assert vad(_VOZ) == 1.0
    assert vad(_SILENCIO) == 0.0


def test_silero_vad_cae_a_energia_si_la_carga_revienta(monkeypatch):
    """Error genérico (carga/inferencia) => fallback permanente sin modelo."""
    def _revienta():
        raise RuntimeError("ONNX corrupto")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _revienta)
    vad = _SileroVad()
    assert vad(_SILENCIO) == 0.0
    assert vad._fallback is True
    assert vad._model is None
    assert vad(_VOZ) == 1.0


def test_silero_vad_cae_a_energia_si_falla_el_import_de_torch(monkeypatch):
    """Si torch/numpy no importan dentro de __call__, cae a energía igual."""
    import builtins

    real_import = builtins.__import__

    def _sin_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _sin_torch)
    vad = _SileroVad()
    assert vad(_VOZ) == 1.0
    assert vad._fallback is True
    assert vad(_SILENCIO) == 0.0


def test_silero_vad_no_reintenta_el_import_tras_el_fallback(monkeypatch):
    """Una vez en fallback, __call__ no vuelve a intentar cargar Silero."""
    llamadas = {"n": 0}

    def _cuenta():
        llamadas["n"] += 1
        raise ImportError("sin silero")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _cuenta)
    vad = _SileroVad()
    for _ in range(5):
        vad(_SILENCIO)
    assert llamadas["n"] == 1, "el import fallido debe intentarse UNA sola vez"


def test_adapters_reexportados_desde_loop():
    """Compatibilidad: los nombres históricos siguen importables desde loop."""
    from crotolamo.voice import loop

    assert loop._SileroVad is adapters._SileroVad
    assert loop.EarThread is EarThread
