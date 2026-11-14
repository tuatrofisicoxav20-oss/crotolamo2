"""Revisión final, unidad 50: scripts/smoke_voz.py (smoke circular TTS -> STT).

circular() con TTS/STT falsos (sin Piper ni Whisper reales):

- Transcribía a través de un WAV temporal: en Python >= 3.12 pip resuelve av 19,
  y faster-whisper 1.2.1 lo abre con av.open(..., metadata_errors=...), que av 19
  rechaza con TypeError. TypeError no estaba entre las excepciones atrapadas: el
  smoke (`./launch.sh smoke`, menú 6) soltaba un traceback y decía «voz rota»
  por un camino que el daemon no usa (él pasa un ndarray float32 EN MEMORIA).
- El resampleo 22050 -> 16000 sobrepasa el rango int16 en los picos y el
  .astype(int16) volteaba esas muestras (+40000 -> -25498) antes del WAV.
"""

from __future__ import annotations

import types
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly


class _TTSFalso:
    voice_model = "voz.onnx"

    def __init__(self, pcm: np.ndarray, rate: int = 22050) -> None:
        self.pcm = pcm
        self.rate = rate

    def available(self) -> bool:
        return True

    def synthesize_pcm(self, text: str):
        return self.pcm, self.rate


class _STTQueAnota:
    """Anota lo que recibe transcribe(); si le llega una ruta, lee el WAV como
    lo haría faster-whisper (int16 / 32768) para poder comparar el audio."""

    def __init__(self, texto: str = "crotolamo prueba uno dos tres",
                 error: Exception | None = None) -> None:
        self.texto = texto
        self.error = error
        self.recibido: object = None
        self.audio: np.ndarray | None = None

    def transcribe(self, audio, hotwords=None) -> str:
        self.recibido = audio
        if isinstance(audio, (str, Path)):
            with wave.open(str(audio), "rb") as w:
                crudo = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            self.audio = crudo.astype(np.float32) / 32768.0
        else:
            self.audio = np.asarray(audio)
        if self.error is not None:
            raise self.error
        return self.texto


def _smoke_con(monkeypatch, tts, stt):
    from scripts import smoke_voz

    monkeypatch.setattr(smoke_voz, "get_settings", lambda: object())
    monkeypatch.setattr(smoke_voz, "TTS", types.SimpleNamespace(from_settings=lambda s: tts))
    monkeypatch.setattr(smoke_voz, "STT", types.SimpleNamespace(from_settings=lambda s: stt))
    return smoke_voz


def test_smoke_circular_transcribe_en_memoria_como_el_daemon(monkeypatch, tmp_path, capsys):
    """El STT recibe un ndarray float32 1-D a 16 kHz normalizado a [-1, 1] (lo
    que da record_audio_until_silence), no la ruta de un WAV temporal."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stt = _STTQueAnota()
    smoke = _smoke_con(monkeypatch, _TTSFalso(np.full(2205, 1000, dtype=np.int16)), stt)

    assert smoke.circular() == 0
    assert "OK: la cadena TTS->STT funciona." in capsys.readouterr().out
    assert isinstance(stt.recibido, np.ndarray), type(stt.recibido)
    assert stt.recibido.dtype == np.float32 and stt.recibido.ndim == 1
    assert stt.recibido.size == 1600  # 0,1 s de Piper (22050 Hz) -> 16 kHz
    # Lejos de los bordes del filtro, la meseta de 1000 vale 1000/32768.
    assert stt.recibido[400:1200] == pytest.approx(1000 / 32768.0, abs=1e-3)
    assert list(tmp_path.iterdir()) == []  # ni WAV ni nada en /tmp


def test_smoke_circular_no_voltea_los_picos_al_resamplear(monkeypatch):
    """Una voz a tope (cuadrada ±32767) sobrepasa el rango int16 al resamplear:
    con .astype(int16) esos picos salían con el signo cambiado."""
    pcm = np.repeat(np.array([32767, -32768] * 20, dtype=np.int16), 110)
    stt = _STTQueAnota()
    smoke = _smoke_con(monkeypatch, _TTSFalso(pcm), stt)

    assert smoke.circular() == 0
    esperado = resample_poly(pcm.astype(np.float32), 16000, 22050) / 32768.0
    assert stt.audio is not None and stt.audio.shape == esperado.shape
    picos = np.abs(esperado) > 0.5
    assert np.array_equal(np.sign(stt.audio[picos]), np.sign(esperado[picos]))
    assert stt.audio == pytest.approx(esperado, abs=1e-4)


def test_smoke_circular_error_inesperado_dice_fallo_y_no_traceback(monkeypatch, capsys):
    """Es el diagnóstico de arriba del todo: cualquier error de la cadena (p. ej.
    el TypeError de av 19 al abrir un archivo) es «FALLO: ...» y sale 1."""
    error = TypeError("open() got an unexpected keyword argument 'metadata_errors'")
    smoke = _smoke_con(monkeypatch, _TTSFalso(np.full(2205, 1000, dtype=np.int16)),
                       _STTQueAnota(error=error))

    assert smoke.circular() == 1
    out = capsys.readouterr().out
    assert "FALLO: TypeError: open() got an unexpected keyword argument" in out
    assert "OK:" not in out
