"""Revisión final, unidad 40v: crotolamo/voice/wakeword.py.

1. feed()/score() pasaban el audio float a int16 sin recortar: una muestra
   fuera de [-1, 1] daba la vuelta (1.2 -> -26216, 2.0 -> -2). El oído del loop
   concurrente (_RealMic, stream float32) y el pre-filtro alimentan feed() con
   esos chunks tal cual, así que con la ganancia de la fuente por encima del
   100 % el modelo puntuaba audio corrompido justo cuando el dueño habla más
   fuerte (medido con la voz sintetizada y el modelo real: wakes perdidos o
   tardíos). El modo simple con sesión (_wake_de_fuente) ya recortaba los
   mismos float32, así que los dos modos decidían distinto sobre el mismo
   audio. Ahora _as_int16 satura igual que _wake_de_fuente; lo que está en
   rango no cambia ni una muestra.
"""

from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

from crotolamo.voice import wakeword as ww  # noqa: E402

_FRAME = 1280
# Muestras del caso del informe: dentro de rango, en el borde y fuera.
_PICOS = [0.5, 0.99, 1.0, 1.2, 1.5, -1.2, -1.5, 2.0]


class _Grabadora:
    """openWakeWord falso: guarda cada array que recibe predict y puntúa con su
    media (una muestra que da la vuelta cambia de signo y mueve el score)."""

    def __init__(self) -> None:
        self.vistos: list = []
        self.preprocessor = types.SimpleNamespace()  # sin búfer crudo: 0.4

    def predict(self, arr) -> dict:
        self.vistos.append(np.array(arr, copy=True))
        return {"wake": float(np.mean(np.asarray(arr, dtype=np.float64))) / 32768.0}


def _detector(**kwargs) -> tuple[ww.WakeWordDetector, _Grabadora]:
    det = ww.WakeWordDetector(model_name="fake", threshold=kwargs.pop("threshold", 2.0),
                              **kwargs)
    modelo = _Grabadora()
    det._model = modelo  # type: ignore[assignment]
    det._np = np
    det._blank_state = {}
    return det, modelo


def _saturado(n: int = _FRAME, dtype=np.float32):
    """Voz fuerte con ganancia > 100 %: un seno de pico 1.6 más los picos del
    informe al principio."""
    t = np.arange(n, dtype=np.float64)
    audio = 1.6 * np.sin(2 * np.pi * 440.0 * t / 16000.0)
    audio[:len(_PICOS)] = _PICOS
    return audio.astype(dtype)


def _recortado_int16(audio):
    # La conversión de _wake_de_fuente (modo simple con sesión), la de referencia;
    # sin forzar float32 para que un float64 se compare con su propio producto.
    return np.clip(np.asarray(audio) * 32767, -32768, 32767).astype(np.int16)


# --- 1. el float fuera de rango satura, no da la vuelta ------------------------

def test_as_int16_satura_las_muestras_fuera_de_rango():
    det, _ = _detector()
    arr = det._as_int16(np.array(_PICOS, dtype=np.float32))
    assert arr.dtype == np.int16
    # Antes: [16383, 32439, 32767, -26216, -16386, 26216, 16386, -2].
    assert arr.tolist() == [16383, 32439, 32767, 32767, 32767, -32768, -32768, 32767]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_feed_de_un_chunk_saturado_puntua_como_el_mismo_chunk_recortado(dtype):
    det, modelo = _detector()
    ref, modelo_ref = _detector()
    audio = _saturado(dtype=dtype)
    assert np.max(np.abs(audio)) > 1.0
    assert det.feed(audio) is False
    # El mismo audio ya saturado en int16 (lo que da el stream int16 del modo
    # simple): feed() lo pasa tal cual. Recortar en float a [-1, 1] daría -32767
    # en vez de -32768 en los negativos: 1 LSB, no la vuelta.
    assert ref.feed(_recortado_int16(audio)) is False
    assert len(modelo.vistos) == len(modelo_ref.vistos) == 1
    assert np.array_equal(modelo.vistos[0], modelo_ref.vistos[0])
    assert np.array_equal(modelo.vistos[0], _recortado_int16(audio))
    assert det._cached == ref._cached


def test_feed_en_chunks_de_512_del_oido_concurrente_tampoco_da_la_vuelta():
    """El camino real del oído: chunks float32 de 512 (Silero) re-encuadrados
    por _stream_score; lo que llega a predict es el audio recortado."""
    det, modelo = _detector()
    ref, modelo_ref = _detector()
    audio = _saturado(n=512 * 15)
    recortado = _recortado_int16(audio)
    for i in range(0, len(audio), 512):
        assert det.stream_score(audio[i:i + 512]) == ref.stream_score(recortado[i:i + 512])
    assert len(modelo.vistos) == len(modelo_ref.vistos) > 0
    for visto, esperado in zip(modelo.vistos, modelo_ref.vistos, strict=True):
        assert np.array_equal(visto, esperado)
    # El calentamiento va chunk a chunk: el primer predict es el primer chunk.
    assert np.array_equal(modelo.vistos[0], _recortado_int16(audio[:512]))


def test_score_crudo_tambien_satura():
    det, modelo = _detector()
    audio = _saturado()
    det.score(audio)
    assert np.array_equal(modelo.vistos[0], _recortado_int16(audio))


# --- lo que está en rango no cambia -------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_audio_en_rango_se_convierte_exactamente_como_antes(dtype):
    det, _ = _detector()
    rng = np.random.default_rng(40)
    audio = rng.uniform(-1.0, 1.0, 4096).astype(dtype)
    audio[:4] = [1.0, -1.0, 0.0, 0.999]
    antes = (audio * 32767).astype(np.int16)  # la conversión de siempre, sin clip
    assert np.array_equal(det._as_int16(audio), antes)


def test_int16_pasa_tal_cual():
    det, _ = _detector()
    audio = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    arr = det._as_int16(audio)
    assert arr is audio  # sin copia: el hot path del int16 no paga nada


# --- los dos modos ven el mismo int16 del mismo audio --------------------------

class _Fuente:
    """capture.MicSession falsa: entrega los bloques y luego queda cerrada."""

    def __init__(self, bloques) -> None:
        self._bloques = list(bloques)
        self.closed = False

    def read(self, timeout=None, frames=None):
        if self._bloques:
            return self._bloques.pop(0)
        self.closed = True
        return None


def test_feed_y_listen_for_wake_con_sesion_puntuan_el_mismo_int16(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    bloques = [_saturado(), _saturado()[::-1].copy()]
    simple, modelo_simple = _detector()
    assert simple.listen_for_wake(timeout_s=None, source=_Fuente(bloques)) is False
    oido, modelo_oido = _detector()
    for bloque in bloques:
        assert oido.feed(bloque) is False
    assert len(modelo_simple.vistos) == len(modelo_oido.vistos) == 2
    for a, b in zip(modelo_simple.vistos, modelo_oido.vistos, strict=True):
        assert np.array_equal(a, b)
