"""Rendimiento: el detector de wake no estaba hecho a la medida del bucle real.

Paso 1: feed() es dueño de su encuadre. El micrófono entrega chunks de 512
muestras (el frame de Silero) y openWakeWord solo calcula features nuevas cada
1280 muestras acumuladas: en 0.4 cada predict() de un chunk que no completa
frame vuelve a correr el clasificador sobre las MISMAS features (2 de cada 3
predicciones, 1875 por minuto alimentado), y en 0.5+ devuelve la predicción
anterior. feed() junta los chunks y llama a predict una sola vez cuando el
paquete calcularía algo nuevo; entre medias devuelve el score guardado, que es
el que el paquete habría devuelto. Mismo score en CADA chunk (se prueba contra
el feed de antes, con el modelo real en 0.4 y en 0.6), un predict por frame (en
0.4, un tercio de las llamadas). [wake].oww_exact_frames (false) encuadra en
frames exactos de 1280, los de listen_for_wake y wake_debug.py: decisión del
dueño, cambia scores en 0.4 y en 0.5+ no cambia nada.

Paso 2: el estado de streaming no sobrevive a un hueco. Tras un turno ya se
resetea al disparar (voice-capture#1); lo que quedaba es la escucha PAUSADA
desde el panel en el loop concurrente: el oído deja de alimentar al detector y,
al reanudar, la ventana de features (16 embeddings, 1.28 s) todavía tenía el
audio de antes de la pausa. rearm() la vacía solo si se alimentó algo desde el
último reset (tras un turno no hace nada: el disparo ya la vació).

Paso 3: el modelo se valida antes de elegir el loop. Cargarlo al arrancar ya lo
hacía el listener (integración, voice-capture#18) llamando a un método privado;
ahora es WakeWordDetector.can_load() (la misma carga única, sin lanzar, con la
causa en load_error) y la decisión de wake y loop vive en listener._choose_wake.

Paso 4: el diagnóstico es proporcional a los eventos. feed() escribe a INFO el
inicio de un episodio de señal, cada cambio de decisión y su fin con el pico (los
scores de en medio, a DEBUG); el wake difuso ya no vuelca al journal lo que se
habla en el cuarto: a INFO solo activaciones y casi-wakes, con las 3 primeras
palabras (todo, como antes, con CROTOLAMO_WAKE_DEBUG=1).

Paso 5: compuertas que no cambian lo reconocido. _record_energy ya devolvía []
sin voz (V7a); lo nuevo es transcribe_if: una pausa del panel que llega a media
ventana del wake difuso ya no paga Whisper por un texto que se tiraba.

Paso 6: la calibración y el doctor miden lo de producción. wake_debug.py
--como-produccion lee chunks de 512 y puntúa por el re-encuadre de feed()
(stream_score); el doctor prueba la MISMA carga del modelo (can_load, sin
descargas) en vez de decir "importable" con un oww_model que no carga.

Paso 7: [wake].pregate (apagado de fábrica): openWakeWord como compuerta delante
del wake difuso. Sin disparo la ventana no llega a Whisper; con disparo se
recorta a ~1.28 s antes de él. Sin la clave, todo es lo de siempre.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo.voice import runtime
from crotolamo.voice import threads as threads_mod
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import EarThread
from crotolamo.voice.wakeword import WakeWordDetector

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]


def _run_py(code: str, timeout: float = 240.0) -> dict:
    """Corre `code` en un intérprete NUEVO y devuelve el JSON de su última línea."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["CROTOLAMO_LOG"] = "WARNING"
    env["PYTHONWARNINGS"] = "ignore"
    env.pop("CROTOLAMO_WAKE_DEBUG", None)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# =============================================================================
# Paso 1: feed() encuadra como el paquete y no repite predicciones
# =============================================================================

_CALENTAMIENTO = 5  # predicciones que openWakeWord anula tras cargar o Model.reset()


class _Oww04:
    """openwakeword 0.4 en miniatura: las features avanzan cuando el
    preprocessor acumula >= 1280 muestras (y las absorbe todas); el clasificador
    corre en CADA predict sobre las últimas features; las 5 primeras
    predicciones tras cargar o reset() valen 0."""

    def __init__(self) -> None:
        self.preprocessor = SimpleNamespace(accumulated_samples=0)
        self.prediction_buffer: list[float] = []
        self.frescas = 0
        self.llamadas: list[int] = []

    def reset(self) -> None:
        self.prediction_buffer = []

    def _clasificador(self) -> float:
        return round(0.1 + 0.01 * self.frescas, 4)

    def predict(self, x) -> dict:
        self.llamadas.append(len(x))
        pre = self.preprocessor
        pre.accumulated_samples += len(x)
        if pre.accumulated_samples >= 1280:
            self.frescas += pre.accumulated_samples // 1280
            pre.accumulated_samples = 0
        s = self._clasificador()
        if len(self.prediction_buffer) < _CALENTAMIENTO:
            s = 0.0
        self.prediction_buffer.append(s)
        return {"wake": s}


class _Oww06(_Oww04):
    """openwakeword 0.5+/0.6 en miniatura: el preprocessor guarda el resto
    (raw_data_remainder) y procesa frames exactos de 1280; un predict que no
    completa frame devuelve la predicción anterior (prediction_buffer[-1])."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = SimpleNamespace(accumulated_samples=0,
                                            raw_data_remainder=np.empty(0))

    def reset(self) -> None:
        self.prediction_buffer = []
        self.preprocessor.accumulated_samples = 0

    def predict(self, x) -> dict:
        self.llamadas.append(len(x))
        pre = self.preprocessor
        total = pre.accumulated_samples + len(x)
        nuevas = total // 1280
        pre.accumulated_samples = total % 1280
        if nuevas:
            self.frescas += nuevas
            s = self._clasificador()
        else:
            s = self.prediction_buffer[-1] if self.prediction_buffer else 0.0
        if len(self.prediction_buffer) < _CALENTAMIENTO:
            s = 0.0
        self.prediction_buffer.append(s)
        return {"wake": s}


def _detector(model, **kwargs) -> WakeWordDetector:
    det = WakeWordDetector(model_name="fake", threshold=kwargs.pop("threshold", 0.99), **kwargs)
    det._model = model  # inyectado: sin openwakeword real
    det._np = np
    # Como _load_model: copia (vacía) del estado en blanco, para que reset()
    # ponga a cero el acumulado del preprocessor como hace con el real.
    det._blank_state = {}
    det._carry = hasattr(model.preprocessor, "raw_data_remainder")
    return det


def _chunks(n: int, size: int) -> list:
    rng = np.random.default_rng(size)
    return [rng.integers(-3000, 3000, size, dtype=np.int16) for _ in range(n)]


@pytest.mark.parametrize("modelo", [_Oww04, _Oww06], ids=["oww0.4", "oww0.6"])
@pytest.mark.parametrize("size", [512, 1280, 1000, 400, 3000])
def test_feed_da_el_mismo_score_por_chunk_que_un_predict_por_chunk(modelo, size):
    """La referencia es el feed() de antes: un predict por chunk. Con un reset
    a mitad (lo que hace un disparo) el calentamiento del paquete vuelve a
    empezar y también tiene que coincidir chunk a chunk."""
    chunks = _chunks(60, size)
    ref, nuevo = modelo(), modelo()
    det_ref, det = _detector(ref), _detector(nuevo)
    viejos, nuevos = [], []
    for i, chunk in enumerate(chunks):
        if i == 31:
            det_ref.reset()
            det.reset()
        viejos.append(det_ref.score(chunk))
        nuevos.append(det._stream_score(chunk))
    assert nuevos == viejos
    assert nuevo.frescas == ref.frescas  # mismas features calculadas
    if size == 512:
        # Fuera del calentamiento, un predict cada frame en vez de uno por chunk.
        assert len(ref.llamadas) == 60
        assert len(nuevo.llamadas) <= 60 // 2.5 + 2 * (_CALENTAMIENTO + 1)


def test_tres_chunks_de_512_son_un_predict_de_1536_con_el_score_guardado_entre_medias():
    modelo = _Oww04()
    det = _detector(modelo)
    for chunk in _chunks(_CALENTAMIENTO + 1, 512):  # el calentamiento va chunk a chunk
        det._stream_score(chunk)
    assert modelo.llamadas == [512] * (_CALENTAMIENTO + 1)
    modelo.llamadas.clear()

    guardado = det._stream_score(_chunks(1, 512)[0])
    assert det._stream_score(_chunks(1, 512)[0]) == guardado
    assert modelo.llamadas == []  # dos chunks sin frame: ningún predict
    fresco = det._stream_score(_chunks(1, 512)[0])
    assert modelo.llamadas == [1536]
    assert fresco != guardado


def test_chunks_de_1280_siguen_siendo_un_predict_por_chunk():
    modelo = _Oww04()
    det = _detector(modelo)
    for chunk in _chunks(12, 1280):
        det._stream_score(chunk)
    assert modelo.llamadas == [1280] * 12


def test_frames_exactos_pasan_1280_y_guardan_el_resto():
    """[wake].oww_exact_frames en 0.4: frames de 1280 exactos (los de
    listen_for_wake y wake_debug.py); el resto espera al chunk siguiente."""
    m = _Oww04()
    det = _detector(m, exact_frames=True)
    for chunk in _chunks(3, 512):
        det._stream_score(chunk)
    assert m.llamadas == [1280]
    for chunk in _chunks(2, 512):
        det._stream_score(chunk)
    assert m.llamadas == [1280, 1280]  # 256 de resto + 1024
    assert sum(m.llamadas) == 5 * 512


def test_frames_exactos_en_0_6_no_cambian_nada():
    """En 0.5+ el preprocessor ya encuadra en frames exactos y guarda el resto:
    la clave no cambia ni un score."""
    chunks = _chunks(40, 512)
    normal, exacto = _detector(_Oww06()), _detector(_Oww06(), exact_frames=True)
    assert ([normal._stream_score(c) for c in chunks]
            == [exacto._stream_score(c) for c in chunks])


def test_reset_tira_lo_pendiente_y_reinicia_el_calentamiento():
    """Un disparo resetea el modelo: lo que el re-encuadre tenía guardado es
    audio de ANTES del disparo y no puede colarse en el frame siguiente."""
    m = _Oww04()
    det = _detector(m, exact_frames=True)
    for chunk in _chunks(3, 512):
        det._stream_score(chunk)        # 1280 al modelo y 256 guardadas
    det.reset()
    m.llamadas.clear()
    for chunk in _chunks(3, 512):
        det._stream_score(chunk)
    assert m.llamadas == [1280]           # sin las 256 de antes del reset

    m4 = _Oww04()
    det = _detector(m4)
    for chunk in _chunks(8, 512):
        det._stream_score(chunk)        # calentamiento y luego guarda 2 chunks
    det.reset()
    m4.llamadas.clear()
    for chunk in _chunks(_CALENTAMIENTO + 1, 512):
        det._stream_score(chunk)
    assert m4.llamadas == [512] * (_CALENTAMIENTO + 1)  # otra vez chunk a chunk


def test_feed_dispara_por_el_score_reencuadrado_y_resetea():
    m = _Oww04()
    det = _detector(m, threshold=0.105)
    disparos = [det.feed(chunk) for chunk in _chunks(12, 512)]
    # El 6º predict es el primero vivo (0.1 + 0.01 * 2 frames) y dispara; el
    # reset del disparo vuelve a abrir las 5 predicciones anuladas del paquete.
    unos = [False] * _CALENTAMIENTO + [True]
    assert disparos == unos + unos


def test_from_settings_lee_oww_exact_frames():
    class _S:
        wake = {"oww_model": "fake"}

    assert WakeWordDetector.from_settings(_S()).exact_frames is False

    class _Exacto:
        wake = {"oww_model": "fake", "oww_exact_frames": True}

    assert WakeWordDetector.from_settings(_Exacto()).exact_frames is True


def test_la_config_de_fabrica_documenta_oww_exact_frames_apagado():
    import tomllib

    texto = (REPO / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    assert tomllib.loads(texto)["wake"].get("oww_exact_frames", False) is False
    assert "oww_exact_frames" in texto


# --- con el openWakeWord REAL (0.4 en 3.12, 0.6 en 3.11) ---------------------

_OWW_IGUALES = """
    import json, sys, types
    import numpy as np
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")
    import openwakeword.model as owm
    from crotolamo.voice.wakeword import WakeWordDetector

    llamadas = {"n": 0}
    _predict = owm.Model.predict
    def _contar(self, x, *a, **k):
        llamadas["n"] += 1
        return _predict(self, x, *a, **k)
    owm.Model.predict = _contar

    rng = np.random.default_rng(11)
    t = np.arange(16000 * 8) / 16000
    audio = (rng.standard_normal(t.size) * 0.02 + 0.2 * np.sin(2 * np.pi * 300 * t)
             * (np.sin(2 * np.pi * 0.7 * t) > 0)).astype(np.float32)
    chunks = [audio[i:i + 512] for i in range(0, audio.size - 511, 512)]

    def correr(nuevo, umbral):
        # openWakeWord 0.5+ ceba sus features con np.random al cargar y en
        # reset(): misma semilla en las dos corridas, mismos números.
        np.random.seed(1234)
        det = WakeWordDetector(model_name="hey_jarvis", threshold=umbral)
        det._get_model()
        scores = []
        crudo = det._stream_score if nuevo else det.score  # antes: un predict por chunk
        def _anota(chunk):
            scores.append(crudo(chunk))
            return scores[-1]
        det._stream_score = _anota
        llamadas["n"] = 0
        disparos = [i for i, c in enumerate(chunks) if det.feed(c)]
        return scores, disparos, llamadas["n"]

    # Un umbral que dispare varias veces: así el reset y su calentamiento entran.
    sondeo, _, _ = correr(False, 2.0)
    vivos = sorted(s for s in sondeo if s > 0)
    umbral = vivos[len(vivos) * 3 // 4] if vivos else 2.0
    viejos, disparos_v, n_v = correr(False, umbral)
    nuevos, disparos_n, n_n = correr(True, umbral)
    import openwakeword
    print(json.dumps({"iguales": nuevos == viejos, "n": len(nuevos),
                      "disparos_viejos": disparos_v, "disparos_nuevos": disparos_n,
                      "predict_viejos": n_v, "predict_nuevos": n_n,
                      "version": getattr(openwakeword, "__version__", "?")}))
"""


def test_con_el_openwakeword_real_cada_chunk_puntua_igual_con_un_tercio_de_predicts():
    pytest.importorskip("onnxruntime")
    if not runtime.module_available("openwakeword"):
        pytest.skip("openwakeword no está instalado")
    out = _run_py(_OWW_IGUALES)
    assert out["n"] == 250
    assert out["iguales"] is True, out
    assert out["disparos_nuevos"] == out["disparos_viejos"]
    assert len(out["disparos_viejos"]) >= 2, out  # hubo reset y calentamiento de nuevo
    assert out["predict_viejos"] == 250
    # Un predict por frame de 1280 (2.5 chunks) más el calentamiento tras cada reset.
    assert out["predict_nuevos"] <= 250 / 2.5 + 6 * (1 + len(out["disparos_viejos"])), out


# --- revisión: el búfer crudo de 10 s que el paquete copia entero en cada frame ---

class _CrudoDelPaquete:
    """Lo que hace el preprocessor de openwakeword (0.4 y 0.6) con el audio
    crudo: un deque de 10 s y, en cada frame, list(raw)[-acumulado-480:]. El
    score depende de esa cola EXACTA; anota cuánto copia list() y el máximo del
    búfer."""

    def __init__(self) -> None:
        self.preprocessor = SimpleNamespace(raw_data_buffer=deque(maxlen=16000 * 10),
                                            accumulated_samples=0)
        self.ultimo = 0.0
        self.copiados = 0
        self.maximo = 0

    def predict(self, x) -> dict:
        pre = self.preprocessor
        pre.raw_data_buffer.extend(np.asarray(x).tolist())
        pre.accumulated_samples += len(x)
        self.maximo = max(self.maximo, len(pre.raw_data_buffer))
        if pre.accumulated_samples >= 1280:
            self.copiados += len(pre.raw_data_buffer)
            cola = list(pre.raw_data_buffer)[-pre.accumulated_samples - 160 * 3:]
            self.ultimo = (sum(cola) % 9973) / 9973 + len(cola)
            pre.accumulated_samples = 0
        return {"w": self.ultimo}


def _audio_int16(segundos: float, semilla: int = 3) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.integers(-3000, 3000, int(16000 * segundos)).astype(np.int16)


def test_feed_no_copia_los_10_s_de_audio_crudo_en_cada_frame():
    """Sin resets (el pre-filtro ya no vacía cada ventana; el oído del loop
    concurrente nunca lo hace) el deque llega a 160000 muestras y list() las
    copia todas cada 80 ms: ~0.6-0.8 CPU-s por minuto alimentado. Solo se lee
    la cola, así que el detector recorta lo demás: mismos scores."""
    audio = _audio_int16(30)
    chunks = [audio[i:i + 512] for i in range(0, audio.size - 511, 512)]
    crudo = _CrudoDelPaquete()
    antes = [crudo.predict(c)["w"] for c in chunks]          # un predict por chunk, sin recorte
    modelo = _CrudoDelPaquete()
    det = WakeWordDetector(model_name="fake", threshold=99.0)
    det._model = modelo  # type: ignore[assignment]
    det._np = np
    assert [det.stream_score(c)[0] for c in chunks] == antes
    assert crudo.maximo == 160000
    assert modelo.maximo <= 16000                             # el búfer ya no pasa de 1 s
    assert modelo.copiados * 10 < crudo.copiados              # recurso: muestras copiadas


def test_listen_for_wake_tampoco_arrastra_los_10_s(monkeypatch):
    from crotolamo.voice import wakeword as ww

    audio = _audio_int16(20)
    lecturas = iter(range(0, audio.size - 1279, 1280))

    class _Stream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            pass

        def read(self, frames: int):
            i = next(lecturas)
            return audio[i:i + frames].reshape(-1, 1), False

    reloj = iter(range(10_000))
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_Stream))
    monkeypatch.setattr(ww, "time", SimpleNamespace(monotonic=lambda: next(reloj) * 0.08))
    modelo = _CrudoDelPaquete()
    det = WakeWordDetector(model_name="fake", threshold=1e9)  # el score falso pasa de 1000
    det._model = modelo  # type: ignore[assignment]
    det._np = np
    assert det.listen_for_wake(timeout_s=19.0) is False
    assert modelo.maximo <= 16000


_OWW_SIN_ARRASTRE = """
    import json, sys, types
    import numpy as np
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")
    import openwakeword
    from crotolamo.voice.wakeword import WakeWordDetector

    rng = np.random.default_rng(5)
    t = np.arange(16000 * 14) / 16000
    audio = (rng.standard_normal(t.size) * 0.02 + 0.2 * np.sin(2 * np.pi * 300 * t)
             * (np.sin(2 * np.pi * 0.7 * t) > 0)).astype(np.float32)
    chunks = [audio[i:i + 512] for i in range(0, audio.size - 511, 512)]

    def nuevo_modelo():
        np.random.seed(77)  # 0.5+ ceba sus features con np.random al cargar
        det = WakeWordDetector(model_name="hey_jarvis", threshold=2.0)
        det._get_model()
        return det

    # Antes: el paquete tal cual, un predict por chunk y sin tocar su búfer.
    crudo = nuevo_modelo()._model
    antes, max_antes = [], 0
    for c in chunks:
        antes.append(float(max(crudo.predict((c * 32767).astype(np.int16)).values())))
        max_antes = max(max_antes, len(crudo.preprocessor.raw_data_buffer))
    det = nuevo_modelo()
    nuevos, max_nuevo = [], 0
    for c in chunks:
        nuevos.append(det.stream_score(c)[0])
        max_nuevo = max(max_nuevo, len(det._model.preprocessor.raw_data_buffer))
    print(json.dumps({"iguales": nuevos == antes, "n": len(nuevos), "max_antes": max_antes,
                      "max_nuevo": max_nuevo, "vivos": sum(1 for s in antes if s > 0),
                      "version": getattr(openwakeword, "__version__", "?")}))
"""


def test_con_el_openwakeword_real_el_bufer_crudo_queda_corto_y_los_scores_iguales():
    pytest.importorskip("onnxruntime")
    if not runtime.module_available("openwakeword"):
        pytest.skip("openwakeword no está instalado")
    out = _run_py(_OWW_SIN_ARRASTRE)
    assert out["n"] == 437 and out["vivos"] > 400, out
    assert out["iguales"] is True, out
    assert out["max_antes"] == 160000, out
    assert out["max_nuevo"] <= 16000, out


# =============================================================================
# Paso 2: tras un hueco sin disparo, la ventana de features no es la de antes
# =============================================================================

_SIL = np.zeros(1280, dtype=np.int16)
_PAL = np.full(1280, 20000, dtype=np.int16)  # un trozo de la palabra de activación


class _PalabraQueTerminaEnSilencio:
    """Modelo falso con lo que se midió del real: puntúa las últimas 16
    features y dispara al TERMINAR la palabra (el primer frame de silencio tras
    ella), no mientras suena. 0.4 sin reset() del preprocessor: lo restaura el
    detector desde la copia en blanco."""

    def __init__(self) -> None:
        self.preprocessor = SimpleNamespace(accumulated_samples=0,
                                            feature_buffer=np.zeros((16, 1)))
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict(self, x) -> dict:
        pre = self.preprocessor
        fila = 1.0 if np.abs(x).max() > 1000 else 0.0
        pre.feature_buffer = np.vstack((pre.feature_buffer, [[fila]]))[-120:]
        ventana = pre.feature_buffer[-16:, 0]
        termino = fila == 0.0 and ventana.sum() >= 4
        return {"wake": 0.9 if termino else 0.0}


def _detector_real(**kwargs) -> tuple[WakeWordDetector, _PalabraQueTerminaEnSilencio]:
    modelo = _PalabraQueTerminaEnSilencio()
    det = WakeWordDetector(model_name="fake", threshold=0.5, **kwargs)
    det._model = modelo  # type: ignore[assignment]
    det._np = np
    det._remember_blank_state(modelo)  # lo que hace _load_model con 0.4
    return det, modelo


def test_rearm_vacia_la_ventana_solo_si_se_alimento_desde_el_ultimo_reset():
    det, modelo = _detector_real()
    det.rearm()                                  # recién cargado: nada que olvidar
    assert modelo.resets == 0
    for _ in range(8):
        det.feed(_SIL)
    det.rearm()                                  # hubo audio: se vacía
    assert modelo.resets == 1
    det.rearm()                                  # sin audio desde entonces: nada
    assert modelo.resets == 1

    for chunk in [_PAL] * 4:
        assert det.feed(chunk) is False
    assert det.feed(_SIL) is True                # el disparo ya resetea (voice-capture#1)
    assert modelo.resets == 2
    det.rearm()                                  # tras un turno: no hace NADA
    assert modelo.resets == 2


def test_rearm_sin_modelo_no_lo_carga_y_apagado_no_hace_nada():
    det = WakeWordDetector(model_name="fake")
    det.rearm()
    assert det._model is None

    det, modelo = _detector_real(rearm_after_gaps=False)
    for _ in range(8):
        det.feed(_SIL)
    det.rearm()
    assert modelo.resets == 0


def test_from_settings_lee_oww_rearm():
    class _S:
        wake = {"oww_model": "fake"}

    assert WakeWordDetector.from_settings(_S()).rearm_after_gaps is True

    class _Apagado:
        wake = {"oww_model": "fake", "oww_rearm": False}

    assert WakeWordDetector.from_settings(_Apagado()).rearm_after_gaps is False


def test_la_config_de_fabrica_documenta_oww_rearm_encendido():
    import tomllib

    texto = (REPO / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    assert tomllib.loads(texto)["wake"].get("oww_rearm") is True
    assert "oww_rearm" in texto


class _Reloj:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, _s) -> None:
        pass


def _oido(monkeypatch, state, wake_fn, wake_rearm, grace_s=1.5):
    reloj = _Reloj()
    monkeypatch.setattr(threads_mod, "time", reloj)
    ear = EarThread(
        None, wake_fn=wake_fn, vad_fn=lambda c: 0.0, to_wav=lambda frames: "WAV",
        tts=None, stt_queue=queue.Queue(), tts_queue=queue.Queue(), state=state,
        shutdown=threading.Event(), wake_rearm=wake_rearm,
    )
    ear._post_speak_grace_s = grace_s
    return ear, reloj


def test_el_oido_rearma_una_vez_tras_el_hueco_y_antes_del_primer_wake(monkeypatch):
    eventos: list[str] = []
    state = SharedState()
    ear, reloj = _oido(monkeypatch, state, lambda c: eventos.append("wake") or False,
                       lambda: eventos.append("rearm"))

    for _ in range(5):                       # IDLE seguido: ni un rearm
        ear._on_chunk(_SIL)
    assert eventos == ["wake"] * 5

    eventos.clear()
    state.set_mode(Mode.SPEAKING)            # un turno: el oído no alimenta al wake
    for _ in range(3):
        ear._on_chunk(_SIL)
    state.set_mode(Mode.IDLE)
    for _ in range(3):                       # gracia post-habla: tampoco
        ear._on_chunk(_SIL)
        reloj.t += 0.1
    assert eventos == []
    reloj.t += 2.0
    for _ in range(3):
        ear._on_chunk(_SIL)
    assert eventos == ["rearm", "wake", "wake", "wake"]

    eventos.clear()
    state.set_enabled(False)                 # escucha pausada desde el panel
    for _ in range(4):
        ear._on_chunk(_SIL)
    state.set_enabled(True)
    ear._on_chunk(_SIL)
    ear._on_chunk(_SIL)
    assert eventos == ["rearm", "wake", "wake"]


def test_un_rearm_que_revienta_no_tumba_al_oido(monkeypatch, caplog):
    import logging

    llamadas: list[str] = []

    def _revienta() -> None:
        raise RuntimeError("modelo raro")

    state = SharedState()
    ear, _reloj = _oido(monkeypatch, state, lambda c: llamadas.append("wake") or False,
                        _revienta)
    state.set_enabled(False)
    ear._on_chunk(_SIL)
    state.set_enabled(True)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.loop"):
        ear._on_chunk(_SIL)
    assert llamadas == ["wake"]
    assert any("modelo raro" in r.getMessage() for r in caplog.records)


def test_sin_wake_rearm_el_oido_es_el_de_siempre(monkeypatch):
    state = SharedState()
    llamadas: list[str] = []
    ear, _reloj = _oido(monkeypatch, state, lambda c: llamadas.append("wake") or False, None)
    state.set_enabled(False)
    ear._on_chunk(_SIL)
    state.set_enabled(True)
    ear._on_chunk(_SIL)
    assert llamadas == ["wake"]


def test_una_palabra_cortada_por_la_pausa_ya_no_dispara_al_reanudar(monkeypatch):
    """Lo medido con openWakeWord real (0.4 y 0.6, 15 de 15 casos): un 'Hey
    Jarvis' cortado por la pausa del panel entre 1 y 6 chunks antes de su
    disparo, disparaba un turno fantasma a los 0.03-0.3 s de reanudar, durase
    lo que durase la pausa. Aquí con el modelo falso que dispara al terminar la
    palabra, por el camino real: EarThread -> rearm() -> feed()."""

    def correr(rearm: bool) -> list[int]:
        state = SharedState()
        det, _modelo = _detector_real(rearm_after_gaps=rearm)
        ear, _reloj = _oido(monkeypatch, state, det.feed, det.rearm)
        for _ in range(8):
            ear._on_chunk(_SIL)
        for _ in range(4):
            ear._on_chunk(_PAL)              # la palabra suena...
        state.set_enabled(False)             # ...y el patrón pausa la escucha
        for _ in range(30):
            ear._on_chunk(_SIL)
        state.set_enabled(True)
        turnos = []
        for i in range(10):
            ear._on_chunk(_SIL)
            if state.get_mode() is Mode.LISTENING:
                turnos.append(i)
                break
        return turnos

    assert correr(rearm=False) == [0], "sin rearm debería reproducir el fantasma"
    assert correr(rearm=True) == []


def test_un_wake_dicho_al_reanudar_dispara_igual_que_tras_un_turno(monkeypatch):
    """Tras la pausa, el detector queda como tras un disparo (el mismo reset()):
    un wake dicho justo al reanudar dispara en el mismo chunk que uno dicho al
    volver de un turno."""

    def primer_turno(tras_pausa: bool) -> int | None:
        state = SharedState()
        det, _modelo = _detector_real()
        ear, reloj = _oido(monkeypatch, state, det.feed, det.rearm)
        for _ in range(8):
            ear._on_chunk(_SIL)
        if tras_pausa:
            state.set_enabled(False)
            ear._on_chunk(_SIL)
            state.set_enabled(True)
        else:
            for chunk in [_PAL] * 4 + [_SIL]:  # un wake real y su turno
                ear._on_chunk(chunk)
            assert state.get_mode() is Mode.LISTENING
            state.set_mode(Mode.IDLE)
            ear._on_chunk(_SIL)
            reloj.t += 2.0                    # pasada la gracia post-habla
        for i, chunk in enumerate([_PAL] * 4 + [_SIL] * 3):
            ear._on_chunk(chunk)
            if state.get_mode() is Mode.LISTENING:
                return i
        return None

    assert primer_turno(tras_pausa=True) == primer_turno(tras_pausa=False) == 4


def test_voice_loop_engancha_rearm_solo_con_el_detector_de_verdad():
    from crotolamo.voice.loop import VoiceLoop

    class _Det:
        def feed(self, chunk) -> bool:
            return False

        def rearm(self) -> None:
            pass

    det = _Det()
    vl = VoiceLoop(SimpleNamespace(handle_turn=lambda c: "ok"), SimpleNamespace(), None, det,
                   mic=SimpleNamespace(read=lambda: None), vad_fn=lambda c: 0.0,
                   to_wav=lambda frames: "WAV")
    assert vl._ear.wake_rearm == det.rearm
    vl = VoiceLoop(SimpleNamespace(handle_turn=lambda c: "ok"), SimpleNamespace(), None, det,
                   mic=SimpleNamespace(read=lambda: None), wake_fn=lambda c: False,
                   vad_fn=lambda c: 0.0, to_wav=lambda frames: "WAV")
    assert vl._ear.wake_rearm is None  # wake_fn inyectado (tests): el oído de siempre


def test_listen_for_wake_rearma_antes_de_leer_el_primer_frame(monkeypatch):
    """Cada espera abre un stream nuevo: lo que el modelo oyó en la anterior es
    de antes del hueco. En producción (timeout None) toda espera acaba en un
    disparo que ya resetea, y el rearm no hace nada; con timeout sí."""
    eventos: list[str] = []

    class _Stream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            pass

        def read(self, frames: int):
            eventos.append("read")
            return np.zeros((frames, 1), dtype=np.int16), False

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_Stream))
    det, modelo = _detector_real()
    real_rearm = det.rearm

    def _rearm() -> None:
        eventos.append("rearm")
        real_rearm()

    det.rearm = _rearm  # type: ignore[method-assign]
    assert det.listen_for_wake(timeout_s=0.02) is False
    assert eventos[:2] == ["rearm", "read"]
    assert modelo.resets == 0                    # recién cargado: nada que olvidar
    eventos.clear()
    assert det.listen_for_wake(timeout_s=0.02) is False
    assert eventos[:2] == ["rearm", "read"]
    assert modelo.resets == 1                    # la espera anterior lo alimentó


# =============================================================================
# Paso 3: el modelo se valida antes de elegir el loop (can_load) y la decisión
# vive en un solo sitio (listener._choose_wake)
# =============================================================================

def _detector_con_carga(monkeypatch, *errores: str) -> tuple[WakeWordDetector, list[int]]:
    """WakeWordDetector real con _load_model falso: cada carga se cuenta y las
    primeras fallan con los VoiceUnavailable de `errores`."""
    from crotolamo.voice.stt import VoiceUnavailable

    cargas: list[int] = []
    pendientes = list(errores)

    def _load_model(self):
        cargas.append(1)
        if pendientes:
            raise VoiceUnavailable(pendientes.pop(0))
        return SimpleNamespace(predict=lambda x: {"wake": 0.0})

    monkeypatch.setattr(WakeWordDetector, "_load_model", _load_model)
    return WakeWordDetector(model_name="fake"), cargas


def test_can_load_carga_una_vez_y_el_oido_usa_esa_misma_carga(monkeypatch):
    det, cargas = _detector_con_carga(monkeypatch)
    assert det.can_load() is True
    assert det.load_error is None
    modelo = det._model
    assert modelo is not None
    det.feed(np.zeros(512, dtype=np.int16))       # el primer chunk del oído
    assert det.can_load() is True
    assert det._model is modelo and cargas == [1], "se cargó dos veces"


def test_can_load_no_lanza_y_deja_la_causa(monkeypatch):
    causa = ("No pude cargar el wake word 'crotolamo.onnx', patrón: External data "
             "path does not exist: crotolamo.onnx.data")
    det, cargas = _detector_con_carga(monkeypatch, causa)
    assert det.can_load() is False
    assert det.load_error == causa
    assert det._model is None
    # No se memoriza el fallo: como _get_model, el siguiente intento vuelve a cargar.
    assert det.can_load() is True and det.load_error is None
    assert cargas == [1, 1]


def test_can_load_con_el_modelo_propio_sin_sus_pesos_da_la_causa_real():
    """El crotolamo.onnx del repo referencia un crotolamo.onnx.data que no está:
    available() decía que sí (el paquete se importa) y el oído moría en su
    primer chunk. can_load() lo dice ANTES, con la causa."""
    import importlib.util

    modelo = REPO / "wakeword_training" / "my_custom_model" / "crotolamo.onnx"
    if importlib.util.find_spec("openwakeword") is None or not modelo.is_file():
        pytest.skip("sin openwakeword o sin el modelo propio del repo")
    if (modelo.parent / "crotolamo.onnx.data").exists():
        pytest.skip("el modelo propio ya trae sus pesos")
    det = WakeWordDetector(model_name=str(modelo))
    assert det.can_load() is False
    assert "crotolamo.onnx.data" in (det.load_error or "")
    assert det._model is None


class _DetectorElegible:
    """Lo que _choose_wake mira de un detector: available() y can_load()."""

    def __init__(self, disponible: bool = True, carga: bool = True,
                 causa: str = "No pude cargar el wake word 'x', patrón: rota") -> None:
        self.disponible = disponible
        self.carga = carga
        self.causa = causa
        self.llamadas: list[str] = []
        self.load_error: str | None = None

    def available(self) -> bool:
        self.llamadas.append("available")
        return self.disponible

    def can_load(self) -> bool:
        self.llamadas.append("can_load")
        if not self.carga:
            self.load_error = self.causa
        return self.carga


def _ajustes(**wake) -> SimpleNamespace:
    return SimpleNamespace(wake=wake, voice={})


@pytest.mark.parametrize("simple", [False, True])
def test_choose_wake_con_use_oww_false_no_mira_el_detector(simple):
    from interfaces import listener

    det = _DetectorElegible()
    assert listener._choose_wake(_ajustes(use_oww=False), det, simple) == (False, False)
    assert det.llamadas == [], "con el difuso pedido no se importa ni se carga nada"


def test_choose_wake_sin_openwakeword_no_intenta_cargar():
    from interfaces import listener

    det = _DetectorElegible(disponible=False)
    assert listener._choose_wake(_ajustes(), det, False) == (False, False)
    assert det.llamadas == ["available"]


@pytest.mark.parametrize("simple, esperado", [(False, (True, True)), (True, (True, False))])
def test_choose_wake_con_el_modelo_que_carga_es_el_de_siempre(simple, esperado):
    from interfaces import listener

    det = _DetectorElegible()
    assert listener._choose_wake(_ajustes(use_oww=True), det, simple) == esperado
    assert det.llamadas == ["available", "can_load"]


def test_choose_wake_con_un_modelo_que_no_carga_cae_al_difuso_y_avisa(capsys, caplog):
    import logging

    from interfaces import listener

    det = _DetectorElegible(carga=False)
    with caplog.at_level(logging.WARNING, logger="crotolamo.listener"):
        assert listener._choose_wake(_ajustes(), det, False) == (False, False)
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert avisos == [f"openWakeWord no carga ({det.causa}); uso el wake difuso"]
    assert det.causa in capsys.readouterr().out  # lo mismo que imprimía run_listen


def test_choose_wake_con_un_detector_sin_can_load_lo_usa_tal_cual():
    """Un detector que solo trae la interfaz de wake (available, feed,
    listen_for_wake) no tiene modelo que validar: se usa tal cual, como antes."""
    from interfaces import listener

    det = SimpleNamespace(available=lambda: True)
    assert listener._choose_wake(_ajustes(), det, False) == (True, True)


# =============================================================================
# Paso 4: diagnóstico proporcional a los eventos, no al audio
# =============================================================================

_WAKEWORD_LOG = "crotolamo.voice.wakeword"
_LISTENER_LOG = "crotolamo.listener"


class _Guion:
    """Modelo con un score por predict; con chunks de 1280 cada chunk es uno."""

    def __init__(self, scores) -> None:
        self.scores = list(scores)

    def predict(self, x) -> dict:
        return {"wake": self.scores.pop(0) if self.scores else 0.0}


def _det_guion(scores, **kwargs) -> WakeWordDetector:
    det = WakeWordDetector(model_name="fake", threshold=kwargs.pop("threshold", 0.5), **kwargs)
    det._model = _Guion(scores)  # type: ignore[assignment]
    det._np = np
    return det


def _registros(caplog, logger: str, nivel: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno == nivel]


def test_un_episodio_sin_disparo_da_su_inicio_y_su_fin_con_el_pico(caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger=_WAKEWORD_LOG)
    det = _det_guion([0.0, 0.2, 0.3, 0.25, 0.0, 0.0])
    assert [det.feed(_SIL) for _ in range(6)] == [False] * 6
    assert _registros(caplog, _WAKEWORD_LOG, logging.INFO) == [
        "wake score=0.200 (umbral=0.50) -> no",
        "wake pico=0.300 (umbral=0.50): fin de la señal",
    ]
    # Cada score intermedio sigue a mano, en DEBUG.
    assert _registros(caplog, _WAKEWORD_LOG, logging.DEBUG) == [
        "wake score=0.300 (umbral=0.50) -> no",
        "wake score=0.250 (umbral=0.50) -> no",
    ]


def test_un_chispazo_de_un_solo_score_es_una_sola_linea(caplog):
    """Si el pico ya salió a INFO (el episodio fue un solo score), el fin no
    añade nada: no se escribe. Antes también era una línea."""
    import logging

    caplog.set_level(logging.INFO, logger=_WAKEWORD_LOG)
    det = _det_guion([0.0, 0.085, 0.0, 0.3, 0.2, 0.0])
    assert not any(det.feed(_SIL) for _ in range(6))
    assert _registros(caplog, _WAKEWORD_LOG, logging.INFO) == [
        "wake score=0.085 (umbral=0.50) -> no",
        "wake score=0.300 (umbral=0.50) -> no",   # bajó después: su pico es el inicio
    ]


def test_un_ruido_largo_cerca_del_wake_no_inunda_el_journal(caplog):
    """Música o voz que ronda el modelo: un score fresco distinto cada 80 ms.
    Antes era una línea INFO por cada uno (50 aquí, ~12 por segundo de ruido);
    ahora dos por episodio."""
    import logging

    caplog.set_level(logging.DEBUG, logger=_WAKEWORD_LOG)
    ruido = [round(0.06 + 0.007 * i, 3) for i in range(50)]  # 0.06 .. 0.403, sin disparo
    det = _det_guion(ruido + [0.0])
    assert not any(det.feed(_SIL) for _ in range(51))
    info = _registros(caplog, _WAKEWORD_LOG, logging.INFO)
    assert info == ["wake score=0.060 (umbral=0.50) -> no",
                    "wake pico=0.403 (umbral=0.50): fin de la señal"]
    assert len(_registros(caplog, _WAKEWORD_LOG, logging.DEBUG)) == 49


def test_la_subida_y_el_disparo_salen_y_el_disparo_cierra_el_episodio(caplog):
    import logging

    caplog.set_level(logging.INFO, logger=_WAKEWORD_LOG)
    det = _det_guion([0.2, 0.4, 0.9, 0.0, 0.0])
    assert [det.feed(_SIL) for _ in range(5)] == [False, False, True, False, False]
    assert _registros(caplog, _WAKEWORD_LOG, logging.INFO) == [
        "wake score=0.200 (umbral=0.50) -> no",
        "wake score=0.900 (umbral=0.50) -> DISPARA",
    ]


def test_un_cambio_de_decision_por_el_umbral_de_la_musica_se_loguea(caplog):
    """El mismo score que con música no dispara y sin ella sí: es un cambio de
    decisión y sale a INFO aunque el score no se mueva."""
    import logging

    class _Media:
        playing = True

        def is_playing(self) -> bool:
            return self.playing

    caplog.set_level(logging.INFO, logger=_WAKEWORD_LOG)
    media = _Media()
    det = _det_guion([0.6] * 4, threshold_media=0.7, media=media)  # type: ignore[arg-type]
    assert det.feed(_SIL) is False
    assert det.feed(_SIL) is False
    media.playing = False
    assert det.feed(_SIL) is True
    assert _registros(caplog, _WAKEWORD_LOG, logging.INFO) == [
        "wake score=0.600 (umbral=0.70) -> no",
        "wake score=0.600 (umbral=0.50) -> DISPARA",
    ]


def test_un_rearm_a_mitad_de_episodio_lo_cierra_con_su_pico(caplog):
    import logging

    caplog.set_level(logging.INFO, logger=_WAKEWORD_LOG)
    det = _det_guion([0.2, 0.3, 0.1])
    det._blank_state = {}
    det.feed(_SIL)
    det.feed(_SIL)
    det.rearm()          # pausa desde el panel con la señal a medias
    det.feed(_SIL)       # al reanudar es otro episodio
    assert _registros(caplog, _WAKEWORD_LOG, logging.INFO) == [
        "wake score=0.200 (umbral=0.50) -> no",
        "wake pico=0.300 (umbral=0.50): fin de la señal",
        "wake score=0.100 (umbral=0.50) -> no",
    ]


def test_el_modo_debug_sigue_imprimiendo_cada_chunk(capsys, caplog):
    """CROTOLAMO_WAKE_DEBUG (validación del micro): un print por chunk, como
    siempre, aunque el journal ya no repita."""
    import logging

    caplog.set_level(logging.INFO, logger=_WAKEWORD_LOG)
    det = _det_guion([0.2, 0.3, 0.3], debug=True)
    for _ in range(3):
        det.feed(_SIL)
    assert capsys.readouterr().out.splitlines() == [
        "[wake] score=0.200 umbral=0.50", "[wake] score=0.300 umbral=0.50",
        "[wake] score=0.300 umbral=0.50",
    ]


_VARIANTES = ["crotolamo", "croto lamo", "gotolamo", "grotolamo", "crotoamo",
              "coto lamo", "coto y amo", "cottoy amo", "croto el amo", "crotolo amo"]


def test_lo_que_se_habla_en_casa_no_va_al_journal(caplog, monkeypatch):
    import logging

    from interfaces import listener

    monkeypatch.delenv("CROTOLAMO_WAKE_DEBUG", raising=False)
    caplog.set_level(logging.DEBUG, logger=_LISTENER_LOG)
    texto = "buenas noches, estas son las noticias de hoy en todo el país."
    listener._log_heard(texto, 0.72, _VARIANTES, activated=False)
    assert _registros(caplog, _LISTENER_LOG, logging.INFO) == []
    assert _registros(caplog, _LISTENER_LOG, logging.DEBUG) == [f"wake difuso oyó: {texto!r}"]


def test_un_casi_wake_sale_con_su_score_y_solo_su_cabeza(caplog, monkeypatch):
    """Lo que sirve para afinar [wake].variants: cómo transcribe Whisper el
    "crotolamo" del patrón cuando NO llega al umbral (score >= umbral - 0.17)."""
    import logging

    from interfaces import listener

    monkeypatch.delenv("CROTOLAMO_WAKE_DEBUG", raising=False)
    caplog.set_level(logging.DEBUG, logger=_LISTENER_LOG)
    listener._log_heard("con todo el amor de mi vida", 0.72, _VARIANTES, activated=False)
    info = _registros(caplog, _LISTENER_LOG, logging.INFO)
    assert info == ["wake difuso: score=0.63 (umbral=0.72) cabeza='con todo el' -> no"]
    assert "amor" not in info[0]
    assert _registros(caplog, _LISTENER_LOG, logging.DEBUG) == [
        "wake difuso oyó: 'con todo el amor de mi vida'"]


def test_una_activacion_sale_a_info(caplog, monkeypatch):
    import logging

    from interfaces import listener

    monkeypatch.delenv("CROTOLAMO_WAKE_DEBUG", raising=False)
    caplog.set_level(logging.INFO, logger=_LISTENER_LOG)
    listener._log_heard("crotolamo pausa la música", 0.72, _VARIANTES, activated=True)
    assert _registros(caplog, _LISTENER_LOG, logging.INFO) == [
        "wake difuso: score=1.00 (umbral=0.72) cabeza='crotolamo pausa la' -> ACTIVA"]


def test_con_crotolamo_wake_debug_el_journal_es_el_de_siempre(caplog, monkeypatch):
    import logging

    from interfaces import listener

    monkeypatch.setenv("CROTOLAMO_WAKE_DEBUG", "1")
    caplog.set_level(logging.DEBUG, logger=_LISTENER_LOG)
    for texto, activo in [("buenas noches, estas son las noticias", False),
                          ("con todo el amor", False), ("crotolamo pausa la música", True)]:
        listener._log_heard(texto, 0.72, _VARIANTES, activated=activo)
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == _LISTENER_LOG] == [
        (logging.INFO, "wake difuso oyó: 'buenas noches, estas son las noticias'"),
        (logging.INFO, "wake difuso oyó: 'con todo el amor'"),
        (logging.INFO, "wake difuso oyó: 'crotolamo pausa la música'"),
    ]


def test_el_loop_simple_con_la_tele_no_escribe_lo_que_oye(caplog, monkeypatch):
    """De punta a punta por _run_simple_loop: tres ventanas de tele y un
    'crotolamo' con orden pegada. Antes, cuatro transcripciones al journal; ahora
    solo la activación, con su cabeza."""
    import logging
    import time as _time

    from interfaces import listener

    oido = ["buenas noches, estas son las noticias de hoy.",
            "anunció un nuevo programa de apoyo para las familias.",
            "y ahora vamos con el reporte del tráfico en la ciudad",
            "crotolamo pausa la música"]

    class _Wake:
        def listen_once(self, **_kwargs):
            if oido:
                return oido.pop(0)
            raise KeyboardInterrupt

    class _Agente:
        def __init__(self) -> None:
            self.ordenes: list[str] = []

        def handle_turn(self, command, on_token=None):
            self.ordenes.append(command)
            return "listo, patrón."

    tts = SimpleNamespace(available=lambda: False, speak=lambda t: None,
                          speak_sentences=lambda t: None, beep=lambda: None)
    monkeypatch.delenv("CROTOLAMO_WAKE_DEBUG", raising=False)
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=_time.time))
    caplog.set_level(logging.DEBUG, logger=_LISTENER_LOG)
    agente = _Agente()
    cfg = listener.ListenerConfig(threshold=0.72, variants=_VARIANTES, followup_s=0, ack="off",
                                  stream_speak=False, smart_endpoint=False)
    rc = listener._run_simple_loop(
        agent=agente, stt=SimpleNamespace(), wake_stt=_Wake(), tts=tts, wake_detector=None,
        use_oww=False, cfg=cfg, hud_state=SharedState(), say=lambda t: None)
    assert rc == 0 and agente.ordenes == ["pausa la música"]
    assert _registros(caplog, _LISTENER_LOG, logging.INFO) == [
        "wake difuso: score=1.00 (umbral=0.72) cabeza='crotolamo pausa la' -> ACTIVA"]
    assert len(_registros(caplog, _LISTENER_LOG, logging.DEBUG)) == 4


# =============================================================================
# Paso 5: compuertas que no pueden cambiar lo reconocido
# =============================================================================
#
# _record_energy ya devuelve [] sin voz (V7a, tests/test_auditoria_voice_stt.py)
# y el loop simple ya no abre el micro en pausa (voice-runtime#12). Lo que queda:
# una pausa que llega A MEDIA ventana del wake difuso. La grabación ya está hecha,
# pero su transcripción se tiraba igual (tras el wake se vuelve a mirar el flag):
# transcribe_if la corta antes de Whisper.

class _SttGrabado:
    """STT real con la grabación y Whisper sustituidos: graba `segundos` de audio
    (llamando a `al_grabar` mientras "graba") y anota lo que llega a Whisper."""

    def __init__(self, texto: str = "", segundos: float = 2.0, al_grabar=None) -> None:
        from crotolamo.voice.stt import STT

        self.stt = STT(model_size="fake", initial_prompt=None)
        self.texto = texto
        self.audio = np.full(int(16000 * segundos), 0.1, dtype=np.float32)
        self.al_grabar = al_grabar
        self.grabaciones: list[dict] = []
        self.whisper: list = []
        self.stt.record_audio_until_silence = self._grabar
        self.stt.transcribe = self._transcribir

    def _grabar(self, **kwargs):
        self.grabaciones.append(kwargs)
        if self.al_grabar is not None:
            self.al_grabar()
        return self.audio

    def _transcribir(self, audio, hotwords=None):
        self.whisper.append(audio)
        return self.texto


@pytest.mark.parametrize("compuerta, esperado", [
    (None, "hola"), (lambda: True, "hola"), (lambda: False, "")])
def test_transcribe_if_decide_si_la_grabacion_llega_a_whisper(compuerta, esperado):
    falso = _SttGrabado(texto="hola")
    kwargs = {} if compuerta is None else {"transcribe_if": compuerta}
    assert falso.stt.listen_once(silence_ms=400, min_audio_s=0.3, **kwargs) == esperado
    # La grabación es la misma (la compuerta no llega al grabador) y Whisper recibe
    # el MISMO audio o nada.
    assert falso.grabaciones == [{"silence_ms": 400}]
    assert [a is falso.audio for a in falso.whisper] == ([True] if esperado else [])


def test_transcribe_if_no_se_consulta_si_min_audio_ya_corto():
    """min_audio_s va primero: una ventana sin voz no lee el flag del panel."""
    falso = _SttGrabado(texto="hola", segundos=0.0)
    consultas: list[int] = []
    assert falso.stt.listen_once(min_audio_s=0.3,
                                 transcribe_if=lambda: consultas.append(1) or True) == ""
    assert consultas == [] and falso.whisper == []


def _loop_difuso(monkeypatch, wake_stt, control: dict, *, hud=None, agente=None):
    """_run_simple_loop con el wake difuso; el flag del panel es control['on'] y
    la vuelta en pausa (o la siguiente grabación) termina el loop."""
    import time as _time

    from interfaces import listener

    vueltas = {"n": 0}
    grabar = wake_stt.listen_once

    def _listen_once(**kwargs):
        vueltas["n"] += 1
        if vueltas["n"] > 1:
            raise KeyboardInterrupt
        return grabar(**kwargs)

    wake_stt.listen_once = _listen_once

    def _dormir(_s):
        raise KeyboardInterrupt

    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: control["on"])
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=_dormir, time=_time.time))
    tts = SimpleNamespace(available=lambda: False, speak=lambda t: None,
                          speak_sentences=lambda t: None, beep=lambda: None)
    cfg = listener.ListenerConfig(threshold=0.72, variants=_VARIANTES, followup_s=0, ack="off",
                                  stream_speak=False, smart_endpoint=False)
    agente = agente or SimpleNamespace(handle_turn=lambda c, on_token=None: "listo.")
    return listener._run_simple_loop(
        agent=agente, stt=SimpleNamespace(), wake_stt=wake_stt, tts=tts, wake_detector=None,
        use_oww=False, cfg=cfg, hud_state=hud or SharedState(), say=lambda t: None)


def test_una_pausa_a_media_ventana_no_pasa_lo_grabado_por_whisper(monkeypatch, caplog):
    """Antes: el patrón pausa la escucha mientras la tele habla; la ventana en
    curso (hasta 6 s + 8 s) se transcribía igual, su texto se escribía al
    journal y el resultado se tiraba. Ahora no llega a Whisper y el panel ve la
    pausa en el acto, la misma publicación que hacía la comprobación tras el wake."""
    import logging

    control = {"on": True}
    falso = _SttGrabado(texto="crotolamo pausa la música",
                        al_grabar=lambda: control.update(on=False))
    publicado: list[dict] = []
    hud = SharedState(publisher=publicado.append)
    hud.set_enabled(True)
    caplog.set_level(logging.DEBUG, logger=_LISTENER_LOG)
    agente = SimpleNamespace(ordenes=[])
    agente.handle_turn = lambda c, on_token=None: agente.ordenes.append(c) or "listo."
    assert _loop_difuso(monkeypatch, falso.stt, control, hud=hud, agente=agente) == 0
    assert len(falso.grabaciones) == 1
    assert falso.whisper == [], "Whisper transcribió una ventana con la escucha en pausa"
    assert agente.ordenes == []
    assert hud.is_enabled() is False and publicado[-1]["enabled"] is False
    assert [r for r in caplog.records if r.name == _LISTENER_LOG] == []


def test_con_la_escucha_activa_whisper_recibe_lo_mismo_y_la_orden_es_la_misma(monkeypatch):
    control = {"on": True}
    falso = _SttGrabado(texto="crotolamo pausa la música")
    agente = SimpleNamespace(ordenes=[])
    agente.handle_turn = lambda c, on_token=None: agente.ordenes.append(c) or "listo."
    assert _loop_difuso(monkeypatch, falso.stt, control, agente=agente) == 0
    assert [a is falso.audio for a in falso.whisper] == [True]
    assert agente.ordenes == ["pausa la música"]
    # La grabación del wake es la de siempre: la compuerta no llega al grabador.
    assert falso.grabaciones == [{"silence_ms": 400, "max_seconds": 8.0,
                                  "start_timeout_s": 6}]


# =============================================================================
# Paso 6: la calibración y el doctor miden el camino de producción
# =============================================================================

_OWW_IMPORTABLE = ("voz-oww", True, "openwakeword importable (modo concurrente OK)", "")
_OWW_NO_IMPORTA = ("voz-oww", False,
                   "falta openwakeword (sin él, solo modo simple/difuso está disponible)",
                   "pip install openwakeword  (o pip install -e '.[voice]')")
_SIN_PESOS = ("No pude cargar el wake word 'wakeword_training/my_custom_model/crotolamo.onnx', "
              "patrón: [ONNXRuntimeError] : 1 : FAIL : External data path does not exist: "
              "crotolamo.onnx.data")


class _DetectorDelDoctor:
    """WakeWordDetector falso para el doctor: cuenta from_settings y can_load;
    `pendientes` es lo que la descarga del paquete aún tiene que bajar y `aviso`,
    un warning que la carga emite (el de CUDA de onnxruntime en 0.4)."""

    creados: list = []
    carga = True
    pendientes: list = []
    aviso: tuple | None = None

    def __init__(self, settings) -> None:
        self.settings = settings
        self.cargas: list[dict] = []
        self.load_error: str | None = None

    @classmethod
    def from_settings(cls, settings):
        det = cls(settings)
        cls.creados.append(det)
        return det

    def pending_downloads(self) -> list:
        return list(self.pendientes)

    def can_load(self, **kwargs) -> bool:
        import warnings

        self.cargas.append(kwargs)
        if self.aviso is not None:
            warnings.warn(self.aviso[0], self.aviso[1], stacklevel=1)
        self.load_error = None if self.carga else _SIN_PESOS
        return self.carga


def _ctx_doctor(**wake):
    return SimpleNamespace(settings=SimpleNamespace(wake=wake, voice={}))


@pytest.fixture
def doctor_oww(monkeypatch):
    """El doctor con openwakeword 'importable' y el detector falso."""
    from crotolamo.voice import wakeword as wakeword_mod
    from scripts import crotolamo_doctor as doctor

    _DetectorDelDoctor.creados = []
    _DetectorDelDoctor.carga = True
    _DetectorDelDoctor.pendientes = []
    _DetectorDelDoctor.aviso = None
    monkeypatch.setattr(runtime, "import_openwakeword", lambda: types.ModuleType("openwakeword"))
    monkeypatch.setattr(wakeword_mod, "WakeWordDetector", _DetectorDelDoctor)
    return doctor


def _resumen_check(checks) -> tuple:
    (check,) = checks
    return (check.name, check.ok, check.detail, check.fix)


def test_doctor_voz_oww_con_un_modelo_que_no_carga_lo_dice_con_la_causa(doctor_oww):
    """Antes: 'openwakeword importable (modo concurrente OK)' en verde con el
    crotolamo.onnx del repo, al que le faltan sus pesos: el servicio arrancaba
    sordo (hoy cae al wake difuso). Ahora el doctor prueba la MISMA carga."""
    _DetectorDelDoctor.carga = False
    (check,) = doctor_oww._check_oww(_ctx_doctor(use_oww=True))
    assert check.ok is False and check.name == "voz-oww"
    assert _SIN_PESOS in check.detail and "wake difuso" in check.detail
    assert ".onnx.data" in check.fix and "--como-produccion" in check.fix
    (det,) = _DetectorDelDoctor.creados
    # Sin descargas desde el doctor (su regla: nada de red sin timeout corto).
    assert det.cargas == [{"download": False}]


def test_doctor_voz_oww_sigue_siendo_opcional(doctor_oww):
    _DetectorDelDoctor.carga = False
    spec = next(s for s in doctor_oww._CHECKS if s.name == "voz-oww")
    (check,) = doctor_oww._guarded(spec, _ctx_doctor(use_oww=True))
    assert check.ok is False and check.required is False


def test_doctor_voz_oww_con_el_modelo_que_carga_dice_lo_de_siempre(doctor_oww):
    assert _resumen_check(doctor_oww._check_oww(_ctx_doctor(use_oww=True))) == _OWW_IMPORTABLE
    assert _resumen_check(doctor_oww._check_oww(_ctx_doctor())) == _OWW_IMPORTABLE  # default
    assert len(_DetectorDelDoctor.creados) == 2


def test_doctor_voz_oww_con_el_wake_difuso_no_carga_nada(doctor_oww):
    """use_oww=false (el modo del patrón): el check de siempre, sin cargar el modelo."""
    _DetectorDelDoctor.carga = False
    assert _resumen_check(doctor_oww._check_oww(_ctx_doctor(use_oww=False))) == _OWW_IMPORTABLE
    assert _resumen_check(doctor_oww._check_oww(None)) == _OWW_IMPORTABLE
    assert _DetectorDelDoctor.creados == []


def test_doctor_voz_oww_sin_openwakeword_no_intenta_cargar(doctor_oww, monkeypatch):
    def _sin_paquete():
        raise ImportError("No module named 'openwakeword'")

    monkeypatch.setattr(runtime, "import_openwakeword", _sin_paquete)
    assert _resumen_check(doctor_oww._check_oww(_ctx_doctor(use_oww=True))) == _OWW_NO_IMPORTA
    assert _DetectorDelDoctor.creados == []


def _openwakeword_falso(monkeypatch, descargas: list) -> None:
    """openwakeword mínimo en sys.modules: Model de la firma 0.4 y download_models
    que anota lo que se le pide."""

    class Model:
        def __init__(self, wakeword_model_paths=None, **kwargs) -> None:
            self.preprocessor = None

        def predict(self, x):
            return {"wake": 0.0}

    paquete = types.ModuleType("openwakeword")
    modelo = types.ModuleType("openwakeword.model")
    modelo.Model = Model  # type: ignore[attr-defined]
    utils = types.ModuleType("openwakeword.utils")
    utils.download_models = descargas.append  # type: ignore[attr-defined]
    for nombre, modulo in [("openwakeword", paquete), ("openwakeword.model", modelo),
                           ("openwakeword.utils", utils)]:
        monkeypatch.setitem(sys.modules, nombre, modulo)
    monkeypatch.setattr(runtime, "import_openwakeword", lambda: paquete)


def test_can_load_sin_descarga_no_llama_a_download_models(monkeypatch):
    descargas: list = []
    _openwakeword_falso(monkeypatch, descargas)
    det = WakeWordDetector(model_name="hey_jarvis")
    assert det.can_load(download=False) is True
    assert descargas == []
    # La carga por defecto (la del listener) es la de siempre: descarga si falta.
    otro = WakeWordDetector(model_name="hey_jarvis")
    assert otro.can_load() is True
    assert descargas == [["hey_jarvis"]]


def test_doctor_voz_oww_real_con_el_modelo_propio_sin_pesos_lo_dice():
    """Con el openwakeword instalado (0.4 en 3.12, 0.6 en 3.11) y el
    crotolamo.onnx del repo."""
    import importlib.util

    from scripts import crotolamo_doctor as doctor

    modelo = REPO / "wakeword_training" / "my_custom_model" / "crotolamo.onnx"
    if importlib.util.find_spec("openwakeword") is None or not modelo.is_file():
        pytest.skip("sin openwakeword o sin el modelo propio del repo")
    if (modelo.parent / "crotolamo.onnx.data").exists():
        pytest.skip("el modelo propio ya trae sus pesos")
    (check,) = doctor._check_oww(_ctx_doctor(use_oww=True, oww_model=str(modelo)))
    assert check.ok is False
    assert "crotolamo.onnx.data" in check.detail


# --- Revisión: el doctor con un openwakeword recién instalado y sin ruido ------

_PENDIENTES = ["embedding_model.onnx", "melspectrogram.onnx", "hey_jarvis_v0.1.onnx"]
_AVISO_CUDA = ("Specified provider 'CUDAExecutionProvider' is not in available provider "
               "names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'")


def test_doctor_voz_oww_con_el_modelo_sin_descargar_avisa_sin_probar_la_carga(doctor_oww):
    """openwakeword 0.5+ recién instalado (el de CI en 3.11): ni hey_jarvis ni los
    modelos de features están en disco hasta que el servicio los descarga al
    arrancar. Antes: ❌ 'el servicio usará el wake difuso', cuando el servicio,
    que sí descarga, cargaba bien y usaba el loop concurrente."""
    _DetectorDelDoctor.carga = False
    _DetectorDelDoctor.pendientes = list(_PENDIENTES)
    (check,) = doctor_oww._check_oww(_ctx_doctor(use_oww=True))
    assert check.name == "voz-oww" and check.ok is True and check.warn is True
    assert all(nombre in check.detail for nombre in _PENDIENTES)
    assert "descarga" in check.detail and "wake difuso" in check.detail
    assert "doctor" in check.fix
    (det,) = _DetectorDelDoctor.creados
    # Sin red no hay carga que probar: fallaría por lo que aún no bajó.
    assert det.cargas == []


def test_doctor_voz_oww_no_saca_el_aviso_de_cuda_de_onnxruntime(doctor_oww):
    """openwakeword 0.4 crea sus sesiones pidiendo CUDA y CPU: en un portátil sin
    GPU onnxruntime avisa por stderr. El doctor de antes no cargaba el modelo y su
    salida no lo traía; la de ahora tampoco. Solo se calla ESE aviso."""
    import warnings

    _DetectorDelDoctor.aviso = (_AVISO_CUDA, UserWarning)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        check = _resumen_check(doctor_oww._check_oww(_ctx_doctor(use_oww=True)))
        _DetectorDelDoctor.aviso = ("otro aviso de la carga", UserWarning)
        doctor_oww._check_oww(_ctx_doctor(use_oww=True))
    assert check == _OWW_IMPORTABLE
    assert [str(aviso.message) for aviso in avisos] == ["otro aviso de la carga"]


def _openwakeword_que_descarga(monkeypatch, destino: Path, *, descarga: bool = True) -> None:
    """openwakeword 0.5+ mínimo: FEATURE_MODELS, VAD_MODELS y MODELS con sus URLs
    y un download_models cuyo destino por defecto es `destino` (el resources/models
    del paquete). Sin `descarga`, el 0.4: su utils no tiene download_models."""
    url = "https://example.invalid/v0.5.1/"
    paquete = types.ModuleType("openwakeword")
    paquete.FEATURE_MODELS = {  # type: ignore[attr-defined]
        "embedding": {"download_url": url + "embedding_model.tflite"},
        "melspectrogram": {"download_url": url + "melspectrogram.tflite"},
    }
    paquete.VAD_MODELS = {  # type: ignore[attr-defined]
        "silero_vad": {"download_url": url + "silero_vad.onnx"},
    }
    paquete.MODELS = {  # type: ignore[attr-defined]
        "alexa": {"download_url": url + "alexa_v0.1.tflite"},
        "hey_jarvis": {"download_url": url + "hey_jarvis_v0.1.tflite"},
    }
    utils = types.ModuleType("openwakeword.utils")
    if descarga:
        def download_models(model_names=[], target_directory=str(destino)):  # noqa: B006
            raise AssertionError("pending_downloads no descarga nada")

        utils.download_models = download_models  # type: ignore[attr-defined]
    for nombre, modulo in [("openwakeword", paquete), ("openwakeword.utils", utils)]:
        monkeypatch.setitem(sys.modules, nombre, modulo)
    monkeypatch.setattr(runtime, "import_openwakeword", lambda: paquete)


def test_pending_downloads_dice_lo_que_la_descarga_del_arranque_bajara(monkeypatch, tmp_path):
    _openwakeword_que_descarga(monkeypatch, tmp_path)
    det = WakeWordDetector(model_name="hey_jarvis")
    # El VAD también se descarga, pero la carga del wake no lo usa: no cuenta.
    assert det.pending_downloads() == _PENDIENTES
    # Un .onnx en disco ya no está pendiente. Si solo falta el .onnx (su .tflite
    # está), la descarga, que mira el .tflite, no lo bajaría: tampoco cuenta y la
    # carga dirá su causa de verdad.
    (tmp_path / "embedding_model.onnx").write_bytes(b"")
    (tmp_path / "melspectrogram.tflite").write_bytes(b"")
    assert det.pending_downloads() == ["hey_jarvis_v0.1.onnx"]
    (tmp_path / "hey_jarvis_v0.1.onnx").write_bytes(b"")
    assert det.pending_downloads() == []


def test_pending_downloads_de_un_modelo_propio_o_mal_escrito_solo_cuenta_features(
        monkeypatch, tmp_path):
    _openwakeword_que_descarga(monkeypatch, tmp_path)
    for nombre in ("wakeword_training/my_custom_model/crotolamo.onnx", "hey_jarvs"):
        assert WakeWordDetector(model_name=nombre).pending_downloads() == _PENDIENTES[:2]


def test_pending_downloads_vacio_si_el_paquete_no_descarga_o_no_lo_entiende(
        monkeypatch, tmp_path):
    _openwakeword_que_descarga(monkeypatch, tmp_path, descarga=False)
    assert WakeWordDetector(model_name="hey_jarvis").pending_downloads() == []  # 0.4
    # Un download_models sin target_directory (otra versión): no se adivina.
    sys.modules["openwakeword.utils"].download_models = lambda names: None  # type: ignore
    assert WakeWordDetector(model_name="hey_jarvis").pending_downloads() == []

    def _sin_paquete():
        raise ImportError("No module named 'openwakeword'")

    monkeypatch.setattr(runtime, "import_openwakeword", _sin_paquete)
    assert WakeWordDetector(model_name="hey_jarvis").pending_downloads() == []


def _correr(code: str, *, delante: tuple[str, ...] = (), timeout: float = 240.0):
    """Como _run_py pero con stderr y los warnings tal cual (sin PYTHONWARNINGS)
    y con `delante` antes del repo en PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*delante, str(REPO)])
    env["CROTOLAMO_LOG"] = "WARNING"
    env.pop("PYTHONWARNINGS", None)
    env.pop("CROTOLAMO_WAKE_DEBUG", None)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


_CTX_FABRICA = """
    from types import SimpleNamespace
    ctx = SimpleNamespace(settings=SimpleNamespace(
        wake={"use_oww": True, "oww_model": "hey_jarvis"}, voice={}))
"""


def test_doctor_real_con_un_openwakeword_recien_instalado(tmp_path):
    """El paquete instalado, copiado SIN sus modelos: como recién instalado con
    pip. 0.5+ (3.11): el doctor avisa sin cargar, y lo que dice que falta es lo
    que la descarga del listener baja; tras ella, el ✅ de siempre. 0.4 (3.12) no
    descarga: sin sus modelos, ❌ con la causa (el servicio tampoco cargaría)."""
    import importlib.util
    import shutil

    spec = importlib.util.find_spec("openwakeword")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("sin openwakeword")
    origen = Path(list(spec.submodule_search_locations)[0])
    copia = tmp_path / "pkg" / "openwakeword"
    shutil.copytree(origen, copia,
                    ignore=shutil.ignore_patterns("__pycache__", "*.onnx", "*.tflite"))
    assert (copia / "resources" / "models").is_dir()
    proc = _correr(_CTX_FABRICA + f"""
    import json, os, shutil
    from crotolamo.voice import runtime
    from crotolamo.voice.wakeword import WakeWordDetector
    from scripts import crotolamo_doctor as doctor

    oww = runtime.import_openwakeword()
    assert oww.__file__.startswith({str(copia)!r}), oww.__file__

    def resumen():
        (c,) = doctor._check_oww(ctx)
        return [c.ok, c.warn, c.detail]

    out = {{"pendientes": WakeWordDetector.from_settings(ctx.settings).pending_downloads(),
            "antes": resumen()}}
    import openwakeword.utils as owu
    out["descarga"] = hasattr(owu, "download_models")
    if out["descarga"]:
        # La descarga del listener, sin red: copia del paquete instalado.
        bajados = []

        def download_file(url, target_directory, file_size=None):
            nombre = url.split("/")[-1]
            bajados.append(nombre)
            fuente = os.path.join({str(origen / "resources" / "models")!r}, nombre)
            if os.path.exists(fuente):
                shutil.copy(fuente, target_directory)

        owu.download_file = download_file
        det = WakeWordDetector.from_settings(ctx.settings)
        out["listener_carga"] = det.can_load()
        out["bajados"] = bajados
        out["pendientes_despues"] = det.pending_downloads()
        out["despues"] = resumen()
    print(json.dumps(out))
    """, delante=(str(copia.parent),))
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not out["descarga"]:
        assert out["pendientes"] == []
        assert out["antes"][0] is False and "no carga" in out["antes"][2]
        return
    assert {"hey_jarvis_v0.1.onnx", "melspectrogram.onnx"} <= set(out["pendientes"])
    ok, warn, detalle = out["antes"]
    assert ok is True and warn is True and "descarga" in detalle
    assert out["listener_carga"] is True
    assert set(out["pendientes"]) <= {n.replace(".tflite", ".onnx") for n in out["bajados"]}
    assert out["pendientes_despues"] == []
    assert out["despues"] == [True, False, _OWW_IMPORTABLE[2]]
    # Sin intentos de carga condenados: ni sus avisos de firma vieja por stderr.
    assert "DEPRECATION" not in proc.stderr


def test_doctor_real_no_saca_por_stderr_el_aviso_de_cuda():
    """Con el openwakeword instalado y hey_jarvis: misma línea y stderr limpio.
    En 0.4 la carga directa SÍ avisa de CUDA (se comprueba en el mismo proceso):
    el doctor lo calla; el listener, que no pasa por aquí, sigue igual."""
    import importlib.util

    if importlib.util.find_spec("openwakeword") is None:
        pytest.skip("sin openwakeword")
    proc = _correr(_CTX_FABRICA + """
    import json, sys, warnings
    from crotolamo.voice.wakeword import WakeWordDetector
    from scripts import crotolamo_doctor as doctor

    (c,) = doctor._check_oww(ctx)
    sys.stderr.flush()
    print("FIN-DOCTOR", file=sys.stderr, flush=True)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        WakeWordDetector.from_settings(ctx.settings).can_load(download=False)
    cuda = any("CUDAExecutionProvider" in str(a.message) for a in avisos)
    import onnxruntime
    from importlib.metadata import version
    print(json.dumps({"check": [c.ok, c.warn, c.detail], "carga_directa_avisa": cuda,
                      "oww": version("openwakeword"),
                      "ort_cuda": "CUDAExecutionProvider" in onnxruntime.get_available_providers()}))
    """)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["check"] == [True, False, _OWW_IMPORTABLE[2]]
    if out["oww"].startswith("0.4") and not out["ort_cuda"]:
        assert out["carga_directa_avisa"] is True  # el aviso existe y el doctor lo calla
    del_doctor = proc.stderr.split("FIN-DOCTOR")[0]
    assert "CUDAExecutionProvider" not in del_doctor, del_doctor


# --- scripts/wake_debug.py --como-produccion ----------------------------------

class _MicGrabado:
    """sounddevice falso: sirve los chunks de `audio` del tamaño pedido y anota
    cómo se abrió el stream y qué se leyó; al acabarse, Ctrl+C (el resumen)."""

    def __init__(self, segundos: float = 3.0) -> None:
        rng = np.random.default_rng(7)
        self.audio = (rng.standard_normal(int(16000 * segundos)) * 0.05).astype(np.float32)
        self.pos = 0
        self.dtypes: list[str] = []
        self.lecturas: list[int] = []
        modulo = types.ModuleType("sounddevice")
        modulo.PortAudioError = type("PortAudioError", (Exception,), {})  # type: ignore
        modulo.InputStream = self._abrir  # type: ignore[attr-defined]
        self.modulo = modulo

    def _abrir(self, *, samplerate, channels, dtype, device=None):
        self.dtypes.append(dtype)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        if self.pos + n > self.audio.size:
            raise KeyboardInterrupt
        bloque = self.audio[self.pos:self.pos + n]
        self.pos += n
        self.lecturas.append(n)
        if self.dtypes[-1] == "int16":
            bloque = (bloque * 32767).astype(np.int16)
        return bloque.reshape(-1, 1), False


def _wake_debug_con(monkeypatch, modelo, segundos: float = 3.0):
    from scripts import wake_debug

    mic = _MicGrabado(segundos)
    det = _detector(modelo, threshold=0.5)
    monkeypatch.setitem(sys.modules, "sounddevice", mic.modulo)
    monkeypatch.setattr(wake_debug, "get_settings",
                        lambda: SimpleNamespace(voice={}, wake={}))
    monkeypatch.setattr(wake_debug.WakeWordDetector, "from_settings",
                        classmethod(lambda cls, s: det))
    return wake_debug, mic


@pytest.mark.parametrize("argv, esperado", [
    ([], (None, False, False)),
    (["--como-produccion", "--verbose"], (None, True, True)),
    (["--threshold", "0.6", "--como-produccion"], (0.6, False, True)),
])
def test_wake_debug_lee_como_produccion(monkeypatch, argv, esperado):
    from scripts import wake_debug

    pedido: list = []
    monkeypatch.setattr(sys, "argv", ["wake_debug.py", *argv])
    monkeypatch.setattr(wake_debug, "run",
                        lambda t, v, como_produccion=False: pedido.append((t, v, como_produccion))
                        or 0)
    assert wake_debug.main() == 0
    assert pedido == [esperado]


def test_wake_debug_como_produccion_lee_512_y_mide_por_el_reencuadre(monkeypatch, capsys):
    """Lo que entrega _RealMic (512 float32) y por el mismo re-encuadre que
    feed(): un predict por frame del paquete, y solo esos scores se imprimen."""
    modelo = _Oww04()
    wake_debug, mic = _wake_debug_con(monkeypatch, modelo)
    assert wake_debug.run(None, True, como_produccion=True) == 0
    out = capsys.readouterr().out
    assert mic.dtypes == ["float32"] and set(mic.lecturas) == {512}
    chunks = len(mic.lecturas)
    # Calentamiento chunk a chunk y después uno de 1536 cada 3 chunks.
    assert modelo.llamadas[:_CALENTAMIENTO + 1] == [512] * (_CALENTAMIENTO + 1)
    assert set(modelo.llamadas[_CALENTAMIENTO + 1:]) == {1536}
    assert len(modelo.llamadas) < chunks / 2
    lineas = [ln for ln in out.splitlines() if "] score=" in ln]
    assert len(lineas) == len(modelo.llamadas) - _CALENTAMIENTO  # 0.0 no pasa el piso
    assert f"frames: {len(modelo.llamadas)}" in out
    assert "como producción" in out


def test_wake_debug_como_produccion_da_los_mismos_scores_que_el_oido(monkeypatch, capsys):
    """Los scores impresos son los de feed() en el loop concurrente, uno por
    predicción nueva (los chunks que repiten el guardado no se imprimen)."""
    wake_debug, mic = _wake_debug_con(monkeypatch, _Oww04())
    assert wake_debug.run(None, True, como_produccion=True) == 0
    impresos = [float(ln.rsplit("score=", 1)[1]) for ln in capsys.readouterr().out.splitlines()
                if "] score=" in ln]
    oido = _detector(_Oww04(), threshold=0.99)
    vistos: list[float] = []
    for i in range(0, mic.pos, 512):
        nuevo, fresco = oido.stream_score(mic.audio[i:i + 512])
        if fresco and nuevo >= 0.05:
            vistos.append(round(nuevo, 3))
    assert impresos == vistos and len(vistos) > 5


def test_wake_debug_sin_el_flag_es_el_de_siempre(monkeypatch, capsys):
    modelo = _Oww04()
    wake_debug, mic = _wake_debug_con(monkeypatch, modelo)
    assert wake_debug.run(None, True) == 0
    out = capsys.readouterr().out
    assert mic.dtypes == ["int16"] and set(mic.lecturas) == {1280}
    assert modelo.llamadas == [1280] * len(mic.lecturas)  # score() crudo por frame
    assert "como producción" not in out


def test_stream_score_es_el_score_de_feed_y_dice_si_es_nuevo():
    """feed() y stream_score() pasan por el mismo re-encuadre: mismo score y la
    misma marca de 'predicción nueva' que usa el log de feed()."""
    a, b = _Oww04(), _Oww04()
    det_feed, det_ss = _detector(a), _detector(b)
    for chunk in _chunks(30, 512):
        antes = det_feed._calls
        det_feed.feed(chunk)
        score, nuevo = det_ss.stream_score(chunk)
        assert score == det_feed._cached and nuevo == (det_feed._calls != antes)
    assert a.llamadas == b.llamadas


# =============================================================================
# Paso 7: hueco para un pre-filtro barato delante del wake difuso ([wake].pregate,
# apagado de fábrica)
# =============================================================================

class _MicMarcado:
    """sounddevice falso para los grabadores de stt: el chunk i lleva su índice en
    la primera muestra (i * 1e-6, no mueve la energía) y la amplitud del guion."""

    def __init__(self, amplitud) -> None:
        self.amplitud = amplitud  # i -> amplitud del chunk i
        self.i = 0
        modulo = types.ModuleType("sounddevice")
        modulo.InputStream = lambda **kwargs: self  # type: ignore[attr-defined]
        self.modulo = modulo

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        bloque = np.full(n, self.amplitud(self.i), dtype=np.float32)
        bloque[0] = self.i * 1e-6
        self.i += 1
        return bloque.reshape(-1, 1), False


def _indices(frames) -> list[int]:
    return [int(round(float(f[0]) * 1e6)) for f in frames]


class _Compuerta:
    """Pre-filtro falso: dispara en el chunk `en` (None = nunca) y anota lo que le pasa."""

    def __init__(self, en: int | None = None, revienta_en: int | None = None) -> None:
        self.en = en
        self.revienta_en = revienta_en
        self.eventos: list = []
        self.n = 0

    def rearm(self) -> None:
        self.eventos.append("rearm")

    def feed(self, chunk) -> bool:
        i = self.n
        self.n += 1
        self.eventos.append(i)
        if i == self.revienta_en:
            raise RuntimeError("onnx roto")
        return i == self.en


_VOZ = range(30, 121)  # chunks con voz en los guiones de abajo


@pytest.fixture
def grabador(monkeypatch):
    """STT real con micrófono y Silero falsos: voz en los chunks de _VOZ."""
    from crotolamo.voice import stt as stt_mod

    mic = _MicMarcado(lambda i: 0.3 if i in _VOZ else 0.001)
    monkeypatch.setitem(sys.modules, "sounddevice", mic.modulo)

    class _Silero:
        def reset_states(self) -> None:
            mic.i = 0

        def __call__(self, bloque, sr) -> float:
            return 0.95 if mic.i - 1 in _VOZ else 0.1

    monkeypatch.setattr(stt_mod, "_get_silero_vad", lambda **k: _Silero())
    voz = {"vad_threshold": 0.8, "vad_preactivation_ms": 800}
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: voz)
    return stt_mod.STT(model_size="fake", initial_prompt=None), mic, voz


def _grabar(stt, mic, backend: str, **kwargs):
    mic.i = 0
    if backend == "silero":
        return stt._record_silero(400, 8.0, 6, {"vad_threshold": 0.8,
                                                "vad_preactivation_ms": 800}, **kwargs)
    return stt._record_energy(400, 8.0, 6, **kwargs)


@pytest.mark.parametrize("backend, primero", [("silero", 6), ("energy", 0)])
def test_sin_compuerta_los_grabadores_son_los_de_siempre(grabador, backend, primero):
    stt, mic, _ = grabador
    sin = _indices(_grabar(stt, mic, backend))
    assert sin[0] == primero and sin[-1] >= _VOZ[-1]
    assert _indices(_grabar(stt, mic, backend, gate=None)) == sin


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_una_compuerta_que_no_dispara_no_deja_nada_para_whisper(grabador, backend):
    stt, mic, _ = grabador
    compuerta = _Compuerta(en=None)
    assert _grabar(stt, mic, backend, gate=compuerta) == []
    # Rearmada al abrir el stream y alimentada con CADA chunk leído (el
    # pre-roll de antes de la voz incluido).
    assert compuerta.eventos == ["rearm", *range(mic.i)]


@pytest.mark.parametrize("backend, atras", [("silero", 40), ("energy", 43)])
@pytest.mark.parametrize("disparo", [80, 35, 31])
def test_una_compuerta_que_dispara_recorta_a_su_ventana(grabador, backend, atras, disparo):
    """La palabra encabeza lo que oye Whisper: se tira lo de más de ~1.28 s antes
    del disparo (40 chunks de 32 ms con Silero, 43 de 30 ms con energía)."""
    stt, mic, _ = grabador
    entera = _indices(_grabar(stt, mic, backend))
    compuerta = _Compuerta(en=disparo)
    recortada = _indices(_grabar(stt, mic, backend, gate=compuerta))
    assert recortada[0] == max(entera[0], disparo - atras)
    assert recortada == entera[entera.index(recortada[0]):]
    # Tras el primer disparo ya no se le da audio (feed() resetea al disparar).
    assert compuerta.eventos == ["rearm", *range(disparo + 1)]


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_una_compuerta_que_revienta_deja_la_ventana_entera(grabador, backend, caplog):
    import logging

    stt, mic, _ = grabador
    entera = _indices(_grabar(stt, mic, backend))
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        frames = _grabar(stt, mic, backend, gate=_Compuerta(en=None, revienta_en=12))
    assert _indices(frames) == entera
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1 and "onnx roto" in avisos[0]


def test_la_compuerta_llega_al_grabador_por_listen_once_y_sin_disparo_no_hay_whisper(
        grabador, monkeypatch):
    stt, mic, voz = grabador
    voz["vad_backend"] = "silero"
    whisper: list = []
    monkeypatch.setattr(stt, "transcribe", lambda audio, hotwords=None: whisper.append(audio)
                        or "crotolamo")
    compuerta = _Compuerta(en=None)
    assert stt.listen_once(silence_ms=400, max_seconds=8.0, start_timeout_s=6,
                           min_audio_s=0.3, gate=compuerta) == ""
    assert whisper == [] and compuerta.eventos[0] == "rearm"
    compuerta = _Compuerta(en=70)
    assert stt.listen_once(silence_ms=400, max_seconds=8.0, start_timeout_s=6,
                           min_audio_s=0.3, gate=compuerta) == "crotolamo"
    assert len(whisper) == 1


# --- el listener: [wake].pregate ---------------------------------------------

class _DetectorPregate(_DetectorElegible):
    threshold = 0.5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alimentado: list = []

    def rearm(self) -> None:
        self.llamadas.append("rearm")

    def feed(self, chunk) -> bool:
        self.alimentado.append(chunk)
        return False


@pytest.mark.parametrize("wake", [{}, {"pregate": ""}, {"pregate": "oww", "use_oww": True}])
def test_sin_pregate_no_hay_pre_filtro_ni_se_mira_el_detector(wake):
    from interfaces import listener

    det = _DetectorPregate()
    use_oww = bool(wake.get("use_oww", False))
    assert listener._choose_pregate(_ajustes(**wake), det, use_oww) is None
    assert det.llamadas == []


def test_pregate_oww_con_un_modelo_que_carga_pone_el_pre_filtro():
    from interfaces import listener

    det = _DetectorPregate()
    pre = listener._choose_pregate(_ajustes(pregate="oww", use_oww=False), det, False)
    assert pre is not None and det.llamadas == ["available", "can_load"]
    pre.rearm()
    assert pre.feed(np.zeros(512, dtype=np.float32)) is False
    assert det.llamadas[-1] == "rearm" and len(det.alimentado) == 1


@pytest.mark.parametrize("det, aviso", [
    (_DetectorPregate(carga=False), "rota"),
    (_DetectorPregate(disponible=False), "openwakeword"),
])
def test_pregate_oww_que_no_carga_avisa_y_el_difuso_sigue_sin_el(det, aviso, caplog):
    import logging

    from interfaces import listener

    with caplog.at_level(logging.WARNING, logger="crotolamo.listener"):
        assert listener._choose_pregate(_ajustes(pregate="oww"), det, False) is None
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1 and aviso in avisos[0]


def test_pregate_desconocido_avisa_y_no_filtra(caplog):
    import logging

    from interfaces import listener

    det = _DetectorPregate()
    with caplog.at_level(logging.WARNING, logger="crotolamo.listener"):
        assert listener._choose_pregate(_ajustes(pregate="silero"), det, False) is None
    assert det.llamadas == []
    assert any("pregate" in r.getMessage() for r in caplog.records)


class _WakeQueAnota:
    def __init__(self) -> None:
        self.pedidos: list[dict] = []

    def listen_once(self, **kwargs):
        self.pedidos.append(kwargs)
        if len(self.pedidos) > 2:
            raise KeyboardInterrupt
        return ""


def _loop_con_pregate(monkeypatch, pregate):
    import time as _time

    from interfaces import listener

    wake_stt = _WakeQueAnota()
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=_time.time))
    tts = SimpleNamespace(available=lambda: False, speak=lambda t: None,
                          speak_sentences=lambda t: None, beep=lambda: None)
    cfg = listener.ListenerConfig(threshold=0.72, variants=_VARIANTES, followup_s=0, ack="off",
                                  stream_speak=False, smart_endpoint=False)
    kwargs = {} if pregate is None else {"pregate": pregate}
    rc = listener._run_simple_loop(
        agent=SimpleNamespace(), stt=SimpleNamespace(), wake_stt=wake_stt, tts=tts,
        wake_detector=None, use_oww=False, cfg=cfg, hud_state=SharedState(),
        say=lambda t: None, **kwargs)
    assert rc == 0
    return wake_stt.pedidos


def test_sin_pregate_el_wake_difuso_pide_lo_de_siempre(monkeypatch):
    for pedido in _loop_con_pregate(monkeypatch, None):
        assert set(pedido) == {"silence_ms", "max_seconds", "start_timeout_s", "min_audio_s",
                               "transcribe_if"}


def test_con_pregate_el_wake_difuso_pasa_la_compuerta_y_si_se_rompe_la_quita(monkeypatch):
    from interfaces import listener

    pre = listener._PreFiltro(_DetectorPregate())
    pedidos = _loop_con_pregate(monkeypatch, pre)
    assert [p.get("gate") for p in pedidos] == [pre, pre, pre]

    roto = listener._PreFiltro(SimpleNamespace(rearm=lambda: None, feed=_revienta))
    with pytest.raises(RuntimeError):
        roto.feed(np.zeros(512, dtype=np.float32))
    assert roto.roto is True
    assert all("gate" not in p for p in _loop_con_pregate(monkeypatch, roto))


def _revienta(chunk=None):
    raise RuntimeError("sesión ONNX rota")


def test_un_pre_filtro_que_no_se_rearma_tambien_se_retira():
    """Si rearm() revienta, el grabador avisa y esa ventana va entera; el loop
    deja de pasarlo para no avisar en cada ventana."""
    from interfaces import listener

    roto = listener._PreFiltro(SimpleNamespace(rearm=_revienta, feed=lambda c: False))
    with pytest.raises(RuntimeError):
        roto.rearm()
    assert roto.roto is True


def test_el_pre_filtro_solo_vacia_la_ventana_tras_un_hueco_del_loop():
    """Revisión: _GateRun pide rearm() al abrir CADA ventana y en openwakeword
    0.6 un reset recalcula 4 s de embeddings (~66 ms de CPU): 13 de 14 ventanas
    por minuto de tele, sin hueco entre ellas (sin disparo no hubo Whisper y la
    siguiente abre en milisegundos). Solo se vacía tras un hueco que marca el
    loop (pausa del panel, error); la primera ventana cuenta como tal."""
    from interfaces import listener

    det, modelo = _detector_real()
    pre = listener._PreFiltro(det)
    pre.rearm()                       # primera ventana: recién cargado, nada que vaciar
    for _ in range(3):                # ventanas seguidas sin disparo
        for _ in range(8):
            assert pre.feed(_SIL) is False
        pre.rearm()
    assert modelo.resets == 0
    pre.hueco()                       # el loop dejó de escuchar
    pre.rearm()
    assert modelo.resets == 1
    pre.rearm()                       # otra ventana sin hueco: nada
    assert modelo.resets == 1


class _WakeConCompuerta:
    """wake_stt falso que hace con la compuerta lo que _GateRun: rearm() al abrir
    la ventana y feed() de cada chunk. Cada ventana del guion es una lista de
    chunks o una excepción (el stream que no abre); sin guion, se acaba."""

    def __init__(self, ventanas) -> None:
        self.ventanas = list(ventanas)
        self.disparos: list[list[bool]] = []

    def listen_once(self, *, gate=None, **kwargs):
        if not self.ventanas:
            raise KeyboardInterrupt
        ventana = self.ventanas.pop(0)
        if isinstance(ventana, Exception):
            raise ventana
        gate.rearm()
        self.disparos.append([gate.feed(chunk) for chunk in ventana])
        return ""


def _loop_con_huecos(monkeypatch, ventanas, escucha):
    """El loop simple con [wake].pregate y la escucha del panel según `escucha`
    (un bool por vuelta; True cuando se acaba)."""
    import time as _time

    from interfaces import listener

    det, modelo = _detector_real()
    pre = listener._PreFiltro(det)
    wake_stt = _WakeConCompuerta(ventanas)
    guion = list(escucha)
    monkeypatch.setattr(listener, "read_control_enabled",
                        lambda *a, **k: guion.pop(0) if guion else True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=_time.time))
    tts = SimpleNamespace(available=lambda: False, speak=lambda t: None,
                          speak_sentences=lambda t: None, beep=lambda: None)
    cfg = listener.ListenerConfig(threshold=0.72, variants=_VARIANTES, followup_s=0, ack="off",
                                  stream_speak=False, smart_endpoint=False)
    rc = listener._run_simple_loop(
        agent=SimpleNamespace(), stt=SimpleNamespace(), wake_stt=wake_stt, tts=tts,
        wake_detector=None, use_oww=False, cfg=cfg, hud_state=SharedState(),
        say=lambda t: None, pregate=pre)
    assert rc == 0
    return wake_stt.disparos, modelo


def test_el_loop_simple_vacia_el_pre_filtro_solo_tras_pausa_o_error(monkeypatch):
    """Recurso que baja: resets del detector = huecos (2), no ventanas (6)."""
    ventanas = [[_SIL] * 8, [_SIL] * 8, [_SIL] * 8,   # seguidas: ni un reset
                [_SIL] * 2,                           # tras la pausa
                RuntimeError("PortAudio"),            # error: el loop duerme 1 s
                [_SIL], [_SIL]]
    escucha = [True, True, True, False, False]        # pausa antes de la 4ª ventana
    disparos, modelo = _loop_con_huecos(monkeypatch, ventanas, escucha)
    assert len(disparos) == 6 and not any(any(d) for d in disparos)
    assert modelo.resets == 2


def test_una_palabra_cortada_por_la_pausa_no_pasa_el_pre_filtro_al_reanudar(monkeypatch):
    """La protección que importa sigue: lo oído antes de la pausa no dispara la
    compuerta al reanudar (el modelo falso dispara en el silencio tras la palabra)."""
    disparos, modelo = _loop_con_huecos(monkeypatch, [[_PAL] * 4, [_SIL] * 3],
                                        escucha=[True, False])
    assert disparos == [[False] * 4, [False] * 3]
    assert modelo.resets == 1


def test_run_listen_cablea_pregate_hasta_el_loop_simple(monkeypatch, tmp_path):
    """De punta a punta por run_listen: [wake].pregate='oww' con el difuso y un
    modelo que carga llega como compuerta al loop simple; sin la clave, None."""
    from interfaces import listener
    from tests.test_auditoria_voice_listener import _arnes_run_listen, _join_warmers

    for wake, espera in [({"use_oww": False}, False),
                         ({"use_oww": False, "pregate": "oww"}, True)]:
        det = _DetectorPregate()
        det.attach_media = lambda monitor: None
        det.threshold_media = None
        _arnes_run_listen(monkeypatch, tmp_path, det)
        monkeypatch.setattr(listener, "get_settings",
                            lambda w=wake: SimpleNamespace(voice={"wake_duck": False}, wake=w))
        visto: list = []
        monkeypatch.setattr(listener, "_run_simple_loop",
                            lambda **kw: visto.append(kw.get("pregate")) or 0)
        try:
            assert listener.run_listen([]) == 0
        finally:
            _join_warmers()
        assert (visto[0] is not None) is espera, wake
        if espera:
            assert visto[0]._detector is det


def test_la_config_de_fabrica_documenta_pregate_apagado():
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert tomllib.loads(texto)["wake"]["pregate"] == ""
    lineas = texto.splitlines()
    i = next(n for n, linea in enumerate(lineas) if linea.startswith("pregate"))
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[i - 14:i])
    assert '"oww"' in comentario and "--como-produccion" in comentario, comentario


def test_con_musica_el_pre_filtro_decide_con_oww_threshold_media_y_la_config_lo_dice():
    """Revisión: el comentario de pregate decía que la compuerta usa
    oww_threshold, pero run_listen le engancha el MediaMonitor al mismo detector
    y con música en Playing decide con oww_threshold_media. El dueño que baja
    solo oww_threshold (y calibra con wake_debug --como-produccion, sin música)
    seguía exigiendo 0.7 con un reproductor sonando."""
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG
    from interfaces import listener

    class _Medio:
        def predict(self, x) -> dict:
            return {"w": 0.5}

    ajustes = _ajustes(pregate="oww", use_oww=False, oww_threshold=0.2)
    det = WakeWordDetector.from_settings(ajustes)
    det._model = _Medio()  # type: ignore[assignment]
    det._np = np
    det.available = lambda: True  # type: ignore[method-assign]
    pre = listener._choose_pregate(ajustes, det, False)
    assert pre is not None
    suena = {"v": True}
    det.attach_media(SimpleNamespace(is_playing=lambda: suena["v"]))  # como run_listen
    chunk = np.zeros(1280, dtype=np.int16)
    assert pre.feed(chunk) is False          # con música rige oww_threshold_media (0.7)
    suena["v"] = False
    assert pre.feed(chunk) is True           # sin música, oww_threshold (0.2)

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    lineas = texto.splitlines()
    i = next(n for n, linea in enumerate(lineas) if linea.startswith("pregate"))
    j = next(n for n, linea in enumerate(lineas[:i]) if linea.startswith("oww_rearm"))
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[j + 1:i])
    assert "oww_threshold_media" in comentario, comentario
    assert tomllib.loads(texto)["wake"]["pregate"] == ""
