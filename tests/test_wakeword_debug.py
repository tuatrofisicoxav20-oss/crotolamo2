"""Tests del modo debug del wake word (validación de mic).

El flag debug de WakeWordDetector imprime cada score por stdout para calibrar
el umbral con el micrófono real. Aquí se prueba con un modelo falso inyectado:
no requiere openwakeword ni micrófono, solo numpy (que ya exige la suite).
"""

import numpy as np
import pytest

from crotolamo.voice.wakeword import WakeWordDetector


class _FakeModel:
    """Sustituto de openwakeword.Model: devuelve un score fijo."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, arr):
        return {"crotolamo": self.value}


def _detector(score: float, threshold: float = 0.5, debug: bool = False) -> WakeWordDetector:
    det = WakeWordDetector(model_name="fake", threshold=threshold, debug=debug)
    det._model = _FakeModel(score)  # inyectado: evita cargar openwakeword real
    return det


_CHUNK = np.zeros(1280, dtype=np.int16)


def test_debug_prints_score_in_realtime(capsys):
    det = _detector(score=0.83, threshold=0.5, debug=True)
    assert det.feed(_CHUNK) is True
    out = capsys.readouterr().out
    assert "score=0.830" in out
    assert "umbral=0.50" in out
    assert "DISPARA" in out


def test_debug_prints_below_threshold_without_fire(capsys):
    det = _detector(score=0.12, threshold=0.5, debug=True)
    assert det.feed(_CHUNK) is False
    out = capsys.readouterr().out
    assert "score=0.120" in out
    assert "DISPARA" not in out


def test_no_debug_prints_nothing(capsys):
    det = _detector(score=0.83, threshold=0.5, debug=False)
    assert det.feed(_CHUNK) is True
    assert capsys.readouterr().out == ""


def test_score_converts_float_chunks():
    det = _detector(score=0.3)
    chunk = np.zeros(1280, dtype=np.float32)
    assert det.score(chunk) == pytest.approx(0.3)


def test_from_settings_reads_debug_env(monkeypatch):
    class _Settings:
        wake = {"oww_model": "fake", "oww_threshold": 0.6}

    monkeypatch.delenv("CROTOLAMO_WAKE_DEBUG", raising=False)
    assert WakeWordDetector.from_settings(_Settings()).debug is False

    monkeypatch.setenv("CROTOLAMO_WAKE_DEBUG", "1")
    det = WakeWordDetector.from_settings(_Settings())
    assert det.debug is True
    assert det.threshold == 0.6

    # "0" explícito lo apaga (coherente con systemd Environment=...=0).
    monkeypatch.setenv("CROTOLAMO_WAKE_DEBUG", "0")
    assert WakeWordDetector.from_settings(_Settings()).debug is False
