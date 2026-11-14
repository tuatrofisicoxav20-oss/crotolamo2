"""Revisión final, unidad 36v: crotolamo/voice/stt.py.

1. Un pre-filtro del wake ([wake].pregate='oww') roto se rearmaba en CADA
   ventana reiniciada de una grabación sobre la sesión de audio. Con on_idle
   pidiendo seguir (la escucha activa en una sala en silencio) la grabación no
   vuelve al listener, que es quien deja de pasar un _PreFiltro `roto`: cada
   reinicio (Silero: SIN_VOZ; energía: espera agotada) construía un _GateRun
   nuevo con el gate ORIGINAL, que volvía a fallar y a avisar con WARNING cada
   ~6 s, para siempre (~600 por hora). Roto una vez, sigue fuera el resto de la
   grabación, como en el camino por llamada: un solo aviso.
2. El filtro anti-alucinación de transcribe (no_speech_prob > 0.6 o
   avg_logprob < -1.2, None cuenta como 0) no lo fijaba ningún test: todos los
   Whisper falsos de la suite devuelven 0.0/-0.1, y quitarlo entero no rompía
   nada.
"""

from __future__ import annotations

import logging
import types

import pytest

np = pytest.importorskip("numpy")

from crotolamo.voice import stt as stt_mod  # noqa: E402
from tests.test_capture import (  # noqa: E402
    ENERGIA,
    SALA,
    SILERO,
    VOICE,
    VOZ,
    _FuenteGuion,
    _linea,
    _SileroFalso,
)

SR = 16000


@pytest.fixture
def stt(monkeypatch):
    """STT real con Silero falso (inyectado por la caché _silero_vad)."""
    monkeypatch.setattr(stt_mod, "_silero_vad", _SileroFalso())
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    voice = dict(VOICE)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: voice)
    return types.SimpleNamespace(stt=stt_mod.STT(model_size="fake", initial_prompt=None),
                                 voice=voice)


# --- 1. un pre-filtro roto no se rearma en cada ventana -----------------------

class _GateRoto:
    """Como listener._PreFiltro con un modelo que carga pero no predice (dims
    incompatibles): rearm() solo llega al detector tras un hueco (aquí nunca
    falla) y feed() lanza siempre."""

    def __init__(self) -> None:
        self.rearmes = 0
        self.feeds = 0
        self.roto = False

    def rearm(self) -> None:
        self.rearmes += 1

    def feed(self, _bloque) -> bool:
        self.feeds += 1
        self.roto = True
        raise RuntimeError("predict: dims incompatibles")


class _GateSano:
    """Pre-filtro que nunca dispara: cuenta rearmes y chunks oídos."""

    def __init__(self) -> None:
        self.rearmes = 0
        self.feeds = 0

    def rearm(self) -> None:
        self.rearmes += 1

    def feed(self, _bloque) -> bool:
        self.feeds += 1
        return False


def _on_idle(ventanas: int):
    """on_idle que pide seguir `ventanas - 1` veces (la escucha activa en una
    sala en silencio) y luego suelta la grabación."""
    llamadas = [0]

    def on_idle() -> bool:
        llamadas[0] += 1
        return llamadas[0] < ventanas

    return on_idle, llamadas


def _grabar(stt, backend: str, gate, audio, on_idle):
    stt.voice["vad_backend"] = backend
    return stt.stt._record_frames(400, 8.0, 6.0, gate=gate, source=_FuenteGuion(audio),
                                  on_idle=on_idle)


def _chunk(backend: str) -> int:
    return SILERO if backend == "silero" else ENERGIA


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_gate_roto_avisa_una_vez_en_toda_la_espera(stt, backend, caplog):
    chunk = _chunk(backend)
    audio = _linea((SALA, 60 * SR // chunk), chunk=chunk)
    gate = _GateRoto()
    on_idle, llamadas = _on_idle(10)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        frames = _grabar(stt, backend, gate, audio, on_idle)
    assert llamadas[0] == 10  # diez ventanas de espera sobre la misma fuente
    avisos = [r for r in caplog.records if "pre-filtro del wake" in r.getMessage()]
    assert len(avisos) == 1, [r.getMessage() for r in avisos]
    assert gate.feeds == 1
    assert frames == []  # sala en silencio: nada para Whisper


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_gate_roto_deja_la_voz_de_una_ventana_posterior_entera(stt, backend, caplog):
    """Roto en la primera ventana, la voz que llega en otra posterior va entera
    a Whisper, como sin pre-filtro (no se queda sorda la escucha)."""
    chunk = _chunk(backend)
    espera = 8 * SR // chunk  # más que start_timeout: al menos un reinicio antes de la voz
    amp = VOZ if backend == "silero" else 0.1
    audio = _linea((SALA, espera), (amp, 30), (SALA, 200), chunk=chunk)
    gate = _GateRoto()
    on_idle, llamadas = _on_idle(5)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        frames = _grabar(stt, backend, gate, audio, on_idle)
    assert llamadas[0] >= 1
    voz = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    assert int(np.sum(np.isclose(voz, amp))) == 30 * chunk
    avisos = [r for r in caplog.records if "pre-filtro del wake" in r.getMessage()]
    assert len(avisos) == 1
    assert gate.feeds == 1


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_gate_sano_se_rearma_en_cada_ventana(stt, backend):
    """Lo sano no cambia: cada ventana reiniciada vuelve a pedir rearm() (el
    _PreFiltro decide si llega al detector) y el gate oye cada chunk."""
    chunk = _chunk(backend)
    audio = _linea((SALA, 60 * SR // chunk), chunk=chunk)
    gate = _GateSano()
    on_idle, llamadas = _on_idle(4)
    frames = _grabar(stt, backend, gate, audio, on_idle)
    assert frames == []
    assert gate.rearmes == llamadas[0] == 4
    assert gate.feeds > 4 * 6 * SR // chunk


# --- 2. filtro anti-alucinación de transcribe ---------------------------------

class _WhisperSegmentos:
    """Whisper falso que devuelve segmentos con las métricas dadas."""

    def __init__(self, segmentos) -> None:
        self.segmentos = segmentos

    def transcribe(self, _audio, **_kwargs):
        segs = [types.SimpleNamespace(text=texto, no_speech_prob=ns, avg_logprob=lp)
                for texto, ns, lp in self.segmentos]
        return iter(segs), None


def _transcribir(monkeypatch, segmentos) -> str:
    monkeypatch.setitem(stt_mod._models, "_fake_alucina_", _WhisperSegmentos(segmentos))
    stt = stt_mod.STT(model_size="_fake_alucina_", initial_prompt=None)
    return stt.transcribe(np.zeros(SR, dtype=np.float32))


def test_transcribe_descarta_segmentos_alucinados(monkeypatch):
    texto = _transcribir(monkeypatch, [
        (" Hola", 0.1, -0.3),                       # voz con confianza: se queda
        (" yo te voy a amar", 0.9, -0.3),           # muy probablemente no-voz
        (" la la la", 0.1, -1.5),                   # confianza baja
        ("  Mundo  ", None, None),                  # sin métricas: cuentan como 0
        ("   ", 0.0, -0.1),                         # vacío: no aporta un espacio
    ])
    assert texto == "hola mundo"


@pytest.mark.parametrize(("no_speech", "avg_logprob", "queda"), [
    (0.6, -0.3, True),     # el umbral de no-voz es estricto (> 0.6)
    (0.61, -0.3, False),
    (0.1, -1.2, True),     # el de confianza también (< -1.2)
    (0.1, -1.21, False),
    (0.0, 0.0, True),
    (None, -0.5, True),
    (0.2, None, True),
    (None, -1.3, False),
    (0.7, None, False),
])
def test_transcribe_umbrales_del_filtro(monkeypatch, no_speech, avg_logprob, queda):
    texto = _transcribir(monkeypatch, [(" Enciende la luz", no_speech, avg_logprob)])
    assert texto == ("enciende la luz" if queda else "")
