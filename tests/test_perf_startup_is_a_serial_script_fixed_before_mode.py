"""Rendimiento: el arranque era un guion en serie (startup-is-a-serial-script).

Paso 1 (voz): los cuatro cargadores perezosos (Whisper, Piper, Silero y el modelo
de openWakeWord) construyen UNA sola vez aunque los pidan N hilos a la vez, mezclando
el calentamiento con el primer uso real; y cada uno tiene su entrada de
calentamiento (stt.warm_vad, STT.warm, TTS.warm, WakeWordDetector.warm) que nunca
lanza ni se queda con la decisión del fallback: el primer uso real avisa y decide
exactamente como sin calentar. Se fija con cuentas (construcciones, hilos que
chocaron en el cargador), no con relojes.

Paso 2 (listener): primero se decide el modo y luego se calienta, en el hilo
WarmVoice, lo que ese loop usa y en el orden en que lo usa: Piper primero (el
saludo, a la vez que la decisión del modo), Silero, y el Whisper de la orden; el
'tiny' del wake solo con el wake difuso. Se fija con construcciones por hilo y en
orden, de punta a punta por run_listen con los cargadores reales. El calentador
tiene dueño: al salir run_listen no carga más y su espera es acotada.

Paso 3 (listener): el warm-up del LLM solo llama a chat() si el primario es un
modelo local (needs_warmup, o LLMClient si el cliente no lo declara): con GLM, 0
llamadas a la nube por arranque; con Ollama (o glm sin key), 1, como antes.

Paso 4 (MCP): los handshakes de los servers corren A LA VEZ y el registro sigue
en serie, en el hilo del llamador y en el orden de la config. Se fija con
CUENTAS, no con relojes: handshakes en vuelo a la vez, lanzamientos (Popen) por
server, hilos que quedan vivos. Y que lo que el resto del agente ve no cambia:
mismos nombres (con el sufijo _2 de un choque en el mismo server que en serie),
mismo orden, mismos grupos del router. Más las garantías de la auditoría MCP
con varios handshakes en vuelo: el apagado por señal (terminate_all) y el
Ctrl+C no dejan ningún server vivo (mcp#14), y un server que muere durante el
handshake de otro se retira igual que en serie (mcp#11).

Todo contra el fake server (proceso real, sin red).
"""

from __future__ import annotations

import builtins
import io
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from crotolamo.core import router
from crotolamo.mcp import bridge
from crotolamo.mcp.client import StdioMCPClient
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry
from crotolamo.voice import runtime
from crotolamo.voice import stt as stt_mod
from crotolamo.voice.adapters import _SileroVad
from crotolamo.voice.stt import STT
from crotolamo.voice.tts import TTS
from crotolamo.voice.wakeword import WakeWordDetector

FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]

# Wrapper que tarda `segundos` en arrancar y luego ES el fake (execv: mismo pid,
# como un npx que descarga antes de lanzar el server de verdad).
_LENTO = (
    "import os, sys, time; time.sleep(float(sys.argv[1])); "
    "os.execv(sys.executable, [sys.executable, sys.argv[2]])"
)


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Ningún server ni grupo dinámico sobrevive de un test a otro."""
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)
    monkeypatch.setattr(router, "_RR_COUNTER", 0)
    yield
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)


def mk_settings(servers: dict, **mcp: Any) -> Settings:
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 8, "servers": servers}
    raw.update(mcp)
    return Settings(raw={"mcp": raw}, user="test", home=Path("/tmp"))


def lento(segundos: float) -> list[str]:
    return [sys.executable, "-c", _LENTO, str(segundos), str(FAKE)]


def esperar(cond: Callable[[], bool], plazo: float = 5.0) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def vivo(pid: int) -> bool:
    """True si el proceso existe y no es un zombi."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


def hilos_de_arranque() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("mcp-start")]


class Contador:
    """Cuenta lanzamientos (start) y handshakes (initialize) por server, cuántos
    initialize hay en vuelo a la vez y en qué hilo corre cada uno."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch,
                 antes_de_initialize: Callable[[StdioMCPClient], None] | None = None) -> None:
        self.lock = threading.Lock()
        self.starts: list[str] = []
        self.inits: list[str] = []
        self.hilos: dict[str, str] = {}
        self.clientes: dict[str, StdioMCPClient] = {}
        self.todos: list[StdioMCPClient] = []
        self.listados: dict[str, threading.Event] = {}
        self.en_vuelo = 0
        self.max_en_vuelo = 0
        real_start = StdioMCPClient.start
        real_init = StdioMCPClient.initialize
        real_list = StdioMCPClient.list_tools
        contador = self

        def start(client: StdioMCPClient) -> None:
            with contador.lock:
                contador.starts.append(client.name)
                contador.clientes[client.name] = client
                contador.todos.append(client)
                contador.listados.setdefault(client.name, threading.Event())
            real_start(client)

        def initialize(client: StdioMCPClient, timeout: float = 15.0) -> dict:
            with contador.lock:
                contador.inits.append(client.name)
                contador.hilos[client.name] = threading.current_thread().name
                contador.en_vuelo += 1
                contador.max_en_vuelo = max(contador.max_en_vuelo, contador.en_vuelo)
            try:
                if antes_de_initialize is not None:
                    antes_de_initialize(client)
                return real_init(client, timeout)
            finally:
                with contador.lock:
                    contador.en_vuelo -= 1

        def list_tools(client: StdioMCPClient, timeout: float = 15.0) -> list:
            try:
                return real_list(client, timeout)
            finally:
                contador.listados.setdefault(client.name, threading.Event()).set()

        monkeypatch.setattr(StdioMCPClient, "start", start)
        monkeypatch.setattr(StdioMCPClient, "initialize", initialize)
        monkeypatch.setattr(StdioMCPClient, "list_tools", list_tools)


def foto(registry: Registry, names: list[str]) -> dict[str, Any]:
    """Lo que el resto del agente ve del registro MCP."""
    return {
        "names": list(names),
        "registry": registry.names(),
        "servers": {
            n: registry.get(n).description.rsplit("[MCP: ", 1)[1].rstrip("]")  # type: ignore[union-attr]
            for n in registry.names()
        },
        "groups": {g: list(v["tools"]) for g, v in router.dynamic_groups().items()},
        "connected": bridge.connected_servers(),
    }


# ---------------------------------------------------------------------------
# Paso 1: cargadores de una sola construcción y entradas de calentamiento
# ---------------------------------------------------------------------------

N_HILOS = 8
_TAMANO = "_perf_arranque_"  # tamaño de Whisper que solo usan estos tests


class LockContado:
    """Lock de verdad que cuenta cuántos hilos intentaron tomarlo y abre `todos`
    cuando ya lo intentaron los N. La construcción falsa espera a `todos`: así el
    test PRUEBA que los N hilos chocaron en el cargador mientras se construía (sin
    relojes) y que aun así hubo una sola construcción."""

    def __init__(self, n: int) -> None:
        self._lock = threading.Lock()
        self._cuenta = threading.Lock()
        self.n = n
        self.intentos = 0
        self.todos = threading.Event()

    def __enter__(self) -> LockContado:
        with self._cuenta:
            self.intentos += 1
            if self.intentos >= self.n:
                self.todos.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


class Construcciones:
    """Cuenta las construcciones de modelo (etiqueta, hilo) de todos los vendors
    falsos. Cada una lanza `falla` si la hay, o espera a `compuerta` (hasta 5 s)."""

    def __init__(self) -> None:
        self.hechas: list[tuple[str, str]] = []
        self.compuerta: threading.Event | None = None
        self.falla: Exception | None = None
        self._guard = threading.Lock()

    def __call__(self, etiqueta: str) -> None:
        with self._guard:
            self.hechas.append((etiqueta, threading.current_thread().name))
        if self.falla is not None:
            raise self.falla
        if self.compuerta is not None:
            self.compuerta.wait(5)

    def etiquetas(self) -> list[str]:
        return [etiqueta for etiqueta, _ in self.hechas]


class _ModeloFalso:
    """Lo que construyen los vendors falsos: un objeto nuevo por construcción, con
    lo mínimo que cada consumidor real le pide."""

    def __init__(self, etiqueta: str) -> None:
        self.etiqueta = etiqueta

    def reset_states(self) -> None:  # Silero
        pass

    def __call__(self, x: Any, sr: int) -> float:  # Silero
        return 0.25

    def predict(self, x: Any) -> dict[str, float]:  # openWakeWord
        return {"hey_jarvis": 0.0}

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list, None]:  # Whisper
        return [], None


@pytest.fixture
def cargas(monkeypatch):
    """faster_whisper, piper, openwakeword y el cargador de Silero falsos; todos
    construyen a través del mismo contador. Cachés y memoria de fallo limpias."""
    pytest.importorskip("numpy")
    contador = Construcciones()

    class WhisperModel(_ModeloFalso):
        def __init__(self, size: str, **kwargs: Any) -> None:
            contador(f"whisper:{size}")
            super().__init__(f"whisper:{size}")

    class PiperVoice:
        @staticmethod
        def load(path: str) -> _ModeloFalso:
            contador("piper")
            return _ModeloFalso("piper")

    class Model(_ModeloFalso):
        def __init__(self, wakeword_model_paths: tuple = ()) -> None:
            contador("oww")
            super().__init__("oww")

    def load_silero() -> _ModeloFalso:
        contador("silero")
        return _ModeloFalso("silero")

    whisper = types.ModuleType("faster_whisper")
    whisper.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    piper = types.ModuleType("piper")
    piper.PiperVoice = PiperVoice  # type: ignore[attr-defined]
    oww = types.ModuleType("openwakeword")
    oww.__file__ = "/nonexistent/openwakeword/__init__.py"
    oww_model = types.ModuleType("openwakeword.model")
    oww_model.Model = Model  # type: ignore[attr-defined]
    oww_utils = types.ModuleType("openwakeword.utils")
    oww_utils.download_models = lambda names: None  # type: ignore[attr-defined]
    for name, module in (("faster_whisper", whisper), ("piper", piper), ("openwakeword", oww),
                         ("openwakeword.model", oww_model),
                         ("openwakeword.utils", oww_utils)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runtime, "load_silero", load_silero)
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {})
    yield contador
    if contador.compuerta is not None:
        contador.compuerta.set()  # nunca dejar un hilo colgado si el test falló
    stt_mod._models.pop(_TAMANO, None)
    stt_mod._model_locks.pop(_TAMANO, None)


def _cargador(nombre: str, monkeypatch, tmp_path: Path):
    """(calentar, primer_uso_real, poner_lock, en_cache, etiqueta) de un cargador."""
    if nombre == "whisper":
        orden = STT(model_size=_TAMANO)
        return (orden.warm, orden._get_model,
                lambda lock: stt_mod._model_locks.__setitem__(_TAMANO, lock),
                lambda: stt_mod._models.get(_TAMANO), f"whisper:{_TAMANO}")
    if nombre == "piper":
        onnx = tmp_path / "voz.onnx"
        onnx.touch()
        tts = TTS(onnx)
        return (tts.warm, tts._get_voice, lambda lock: setattr(tts, "_voice_lock", lock),
                lambda: tts._voice, "piper")
    if nombre == "silero":
        return (stt_mod.warm_vad, stt_mod._get_silero_vad,
                lambda lock: monkeypatch.setattr(stt_mod, "_silero_lock", lock),
                lambda: stt_mod._silero_vad, "silero")
    det = WakeWordDetector(model_name="hey_jarvis")
    return (det.warm, det._get_model, lambda lock: setattr(det, "_load_lock", lock),
            lambda: det._model, "oww")


@pytest.mark.parametrize("nombre", ["whisper", "piper", "silero", "oww"])
def test_n_hilos_a_la_vez_construyen_cada_modelo_una_sola_vez(nombre, cargas, monkeypatch,
                                                              tmp_path):
    """La mitad de los hilos calienta y la otra mitad es el primer uso real, todos
    detrás de una barrera. El modelo se construye cuando los N ya chocaron en el
    cargador: una sola construcción, y todos ven la misma instancia."""
    calentar, usar, poner_lock, en_cache, etiqueta = _cargador(nombre, monkeypatch, tmp_path)
    lock = LockContado(N_HILOS)
    poner_lock(lock)
    cargas.compuerta = lock.todos
    barrera = threading.Barrier(N_HILOS, timeout=5)
    resultados: list[Any] = [None] * N_HILOS
    errores: list[BaseException] = []

    def hilo(i: int) -> None:
        try:
            barrera.wait()
            resultados[i] = calentar() if i % 2 else usar()
        except BaseException as error:  # noqa: BLE001 - se comprueba abajo
            errores.append(error)

    hilos = [threading.Thread(target=hilo, args=(i,), name=f"carga-{i}", daemon=True)
             for i in range(N_HILOS)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(10)
    assert not any(h.is_alive() for h in hilos)
    assert errores == []
    assert lock.todos.is_set() and lock.intentos == N_HILOS  # los N chocaron...
    assert cargas.etiquetas() == [etiqueta]                  # ...y UNA construcción
    modelo = en_cache()
    assert isinstance(modelo, _ModeloFalso)
    assert all(r is modelo for r in resultados[0::2])  # el uso real: la misma instancia
    assert resultados[1::2] == [True] * (N_HILOS // 2)  # el calentamiento: cargado


@pytest.mark.parametrize("nombre", ["whisper", "piper", "silero", "oww"])
def test_calentar_lo_ya_cargado_no_construye_otra_vez(nombre, cargas, monkeypatch,
                                                      tmp_path):
    calentar, usar, _poner, en_cache, etiqueta = _cargador(nombre, monkeypatch, tmp_path)
    assert calentar() is True
    modelo = en_cache()
    assert calentar() is True and usar() is modelo
    assert cargas.etiquetas() == [etiqueta]


def test_tras_calentar_el_primer_uso_real_no_construye_nada(cargas, tmp_path):
    """El camino caliente de cada consumidor (transcribe, sintetizar, el primer
    frame del VAD del oído, el primer score del wake) ya no carga: 0 construcciones
    después del calentamiento, y el mismo resultado que sin calentar."""
    np = pytest.importorskip("numpy")
    onnx = tmp_path / "voz.onnx"
    onnx.touch()
    orden, tts = STT(model_size=_TAMANO), TTS(onnx)
    det = WakeWordDetector(model_name="hey_jarvis")
    assert [orden.warm(), tts.warm(), stt_mod.warm_vad(), det.warm()] == [True] * 4
    calentadas = len(cargas.hechas)
    assert sorted(cargas.etiquetas()) == sorted(
        [f"whisper:{_TAMANO}", "piper", "silero", "oww"])
    assert all(hilo == threading.current_thread().name for _, hilo in cargas.hechas)

    assert orden.transcribe(np.zeros(1600, dtype=np.float32)) == ""
    assert tts._get_voice() is tts._voice
    assert _SileroVad()(np.zeros(512, dtype=np.float32)) == 0.25
    assert det.score(np.zeros(1280, dtype=np.int16)) == 0.0
    assert len(cargas.hechas) == calentadas


# --- el calentamiento nunca lanza ni decide el fallback -------------------------


class _MicMudo:
    """InputStream falso de sounddevice: silencio digital."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _MicMudo:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, n: int) -> tuple[Any, bool]:
        import numpy as np

        return np.zeros((n, 1), dtype=np.float32), False


def _primer_uso_vad_sin_silero(monkeypatch, caplog, calentar: bool) -> dict[str, Any]:
    """El primer uso real del VAD (el oído del loop concurrente y una grabación del
    modo simple) con silero_vad sin instalar, con o sin calentamiento previo."""
    np = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "silero_vad", None)  # el paquete falta
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_MicMudo))
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"vad_backend": "silero"})
    salida: dict[str, Any] = {}
    for camino in ("oido", "grabacion"):
        monkeypatch.setattr(stt_mod, "_silero_vad", None)
        monkeypatch.setattr(stt_mod, "_silero_load_error", None)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="crotolamo"):
            if calentar:
                assert stt_mod.warm_vad() is False
                # El calentamiento no memoriza su fallo: decide el primer uso real.
                assert stt_mod._silero_load_error is None
            if camino == "oido":
                vad = _SileroVad()
                valores: Any = [vad(np.full(512, 0.5, dtype=np.float32)),
                                vad(np.zeros(512, dtype=np.float32)), vad._fallback]
            else:
                valores = [STT(model_size=_TAMANO)._record_frames(None, 0.4, 0.2)
                           for _ in range(2)]
        salida[camino] = {
            "valores": valores,
            "avisos": [(r.name, r.levelno, r.getMessage()) for r in caplog.records
                       if r.levelno >= logging.WARNING],
            "fallo_memorizado": stt_mod._silero_load_error,
        }
    return salida


def test_warm_vad_sin_silero_no_lanza_y_el_primer_uso_avisa_igual_que_sin_calentar(
        monkeypatch, caplog):
    sin_calentar = _primer_uso_vad_sin_silero(monkeypatch, caplog, calentar=False)
    con_calentar = _primer_uso_vad_sin_silero(monkeypatch, caplog, calentar=True)
    assert con_calentar == sin_calentar
    # Y lo que hace hoy el primer uso: un aviso, energía, fallo memorizado.
    assert sin_calentar["oido"]["valores"] == [1.0, 0.0, True]
    assert len(sin_calentar["oido"]["avisos"]) == 1
    assert len(sin_calentar["grabacion"]["avisos"]) == 1
    assert sin_calentar["grabacion"]["fallo_memorizado"].startswith("ModuleNotFoundError")


def test_warm_vad_que_falla_con_el_oido_esperando_deja_que_el_oido_decida(monkeypatch,
                                                                         caplog):
    """El calentamiento falla MIENTRAS el primer frame del oído espera en el lock:
    el oído no hereda ese fallo, carga él mismo (una vez), falla, avisa una vez y
    cae a energía, y el fallo memorizado es el suyo: lo mismo que sin calentar."""
    np = pytest.importorskip("numpy")
    lock = LockContado(2)
    cargas: list[str] = []

    def load_silero():
        cargas.append(threading.current_thread().name)
        if len(cargas) == 1:
            lock.todos.wait(5)  # el calentamiento falla con el oído ya en el lock
        raise ModuleNotFoundError("No module named 'silero_vad'")

    monkeypatch.setattr(runtime, "load_silero", load_silero)
    monkeypatch.setattr(stt_mod, "_silero_lock", lock)
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    calentado: list[bool] = []
    oido: list[Any] = []
    vad = _SileroVad()
    with caplog.at_level(logging.DEBUG, logger="crotolamo"):
        w = threading.Thread(target=lambda: calentado.append(stt_mod.warm_vad()),
                             name="WarmVoice", daemon=True)
        w.start()
        assert esperar(lambda: len(cargas) == 1)
        e = threading.Thread(target=lambda: oido.append(vad(np.zeros(512, np.float32))),
                             name="Ear", daemon=True)
        e.start()
        w.join(10)
        e.join(10)
    assert not w.is_alive() and not e.is_alive()
    assert calentado == [False] and oido == [0.0] and vad._fallback is True
    assert cargas == ["WarmVoice", "Ear"]
    assert stt_mod._silero_load_error == "ModuleNotFoundError: No module named 'silero_vad'"
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and avisos[0].startswith("silero-vad no disponible")


def test_warm_vad_carga_el_silero_real_y_el_primer_frame_del_oido_ya_no_carga(monkeypatch):
    """Con el silero_vad y el onnxruntime instalados: misma probabilidad que sin
    calentar, y el primer frame del oído hace 0 cargas (antes, 1)."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("silero_vad")
    real = runtime.load_silero
    cargas: list[str] = []

    def contada():
        cargas.append(threading.current_thread().name)
        return real()

    monkeypatch.setattr(runtime, "load_silero", contada)
    chunk = (0.3 * np.sin(np.arange(512) / 5.0)).astype(np.float32)
    resultados = {}
    for calentar in (False, True):
        monkeypatch.setattr(stt_mod, "_silero_vad", None)
        monkeypatch.setattr(stt_mod, "_silero_load_error", None)
        del cargas[:]
        if calentar:
            assert stt_mod.warm_vad() is True
            assert len(cargas) == 1
        vad = _SileroVad()
        prob = vad(chunk)
        resultados[calentar] = (prob, vad._fallback, len(cargas))
    assert resultados[True][:2] == resultados[False][:2]
    assert resultados[False][2] == 1 and resultados[True][2] == 1  # 0 en el primer frame


def _error(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - se compara el error tal cual
        return f"{type(error).__name__}: {error}"
    raise AssertionError("no lanzó")


def test_stt_warm_sin_faster_whisper_no_lanza_y_el_uso_da_el_error_de_siempre(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    sin_calentar = _error(STT(model_size=_TAMANO)._get_model)
    orden = STT(model_size=_TAMANO)
    assert orden.warm() is False
    assert _error(orden._get_model) == sin_calentar
    assert sin_calentar.startswith("VoiceUnavailable: Falta 'faster_whisper'")
    assert _TAMANO not in stt_mod._models


def test_stt_warm_fallido_no_envenena_la_carga_siguiente(cargas):
    cargas.falla = RuntimeError("modelo corrupto")
    orden = STT(model_size=_TAMANO)
    assert orden.warm() is False
    assert _error(orden._get_model) == "RuntimeError: modelo corrupto"
    cargas.falla = None
    assert orden.warm() is True and orden._get_model() is stt_mod._models[_TAMANO]
    # local_files_only y la llamada de siempre por cada intento fallido, y una buena.
    assert cargas.etiquetas() == [f"whisper:{_TAMANO}"] * 5


def test_tts_warm_sin_onnx_no_carga_nada(cargas, tmp_path):
    tts = TTS(tmp_path / "no_existe.onnx")
    assert tts.warm() is False
    assert cargas.hechas == [] and tts._voice is None


def test_tts_warm_fallido_no_lanza_y_el_uso_da_el_error_de_siempre(cargas, tmp_path):
    onnx = tmp_path / "voz.onnx"
    onnx.touch()
    cargas.falla = RuntimeError("onnx corrupto")
    sin_calentar = _error(TTS(onnx)._get_voice)
    tts = TTS(onnx)
    assert tts.warm() is False
    assert _error(tts._get_voice) == sin_calentar == "RuntimeError: onnx corrupto"
    cargas.falla = None
    assert tts.warm() is True and tts._get_voice() is tts._voice
    assert cargas.etiquetas() == ["piper"] * 4


def test_wake_warm_que_no_carga_no_lanza_y_el_uso_da_el_error_de_siempre(cargas):
    """Un oww_model malo: el calentamiento no decide nada; el primer uso real (el
    listener al elegir el modo) recibe el mismo VoiceUnavailable que sin calentar."""
    cargas.falla = FileNotFoundError("NoSuchFile: hey_jarvis.onnx")
    sin_calentar = _error(WakeWordDetector(model_name="hey_jarvis")._get_model)
    det = WakeWordDetector(model_name="hey_jarvis")
    assert det.warm() is False and det._model is None
    assert _error(det._get_model) == sin_calentar
    assert sin_calentar.startswith("VoiceUnavailable: No pude cargar el wake word")
    cargas.falla = None
    assert det.warm() is True and det._get_model() is det._model


def test_wake_warm_sin_openwakeword_no_lanza(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)
    sin_calentar = _error(WakeWordDetector()._get_model)
    det = WakeWordDetector()
    assert det.warm() is False
    assert _error(det._get_model) == sin_calentar
    assert sin_calentar.startswith("VoiceUnavailable: Falta openwakeword")


# ---------------------------------------------------------------------------
# Paso 2: primero el modo, luego el calentamiento en el orden del primer uso
# ---------------------------------------------------------------------------

_ORDEN = "_perf_orden_"  # tamaños de Whisper que solo usan estos tests
_WAKE = "_perf_wake_"


class _SttPlan:
    """Lo mínimo de STT que mira _warm_plan: el tamaño y su cargador (_get_model)."""

    def __init__(self, model_size: str, llamadas: list[str]) -> None:
        self.model_size = model_size
        self._llamadas = llamadas

    def _get_model(self) -> object:
        self._llamadas.append(f"whisper:{self.model_size}")
        return object()


@pytest.mark.parametrize("modo, concurrente, use_oww, vad_silero, esperado", [
    # El oído del concurrente usa Silero SIEMPRE (_SileroVad no mira vad_backend).
    ("concurrente", True, True, True, ["silero", "whisper:base"]),
    ("concurrente", True, True, False, ["silero", "whisper:base"]),
    # simple + openWakeWord: grabación de la orden (Silero según vad_backend) y su
    # Whisper; la confirmación por voz también usa el de la orden (C5).
    ("simple_oww", False, True, True, ["silero", "whisper:base"]),
    ("simple_oww", False, True, False, ["whisper:base"]),
    # difuso: grabación del wake y los dos Whisper; el de la orden primero, como
    # antes (con 'tiny' primero el proceso queda con ~40-50 MB más de RSS).
    ("difuso", False, False, True, ["silero", "whisper:base", "whisper:tiny"]),
    ("difuso", False, False, False, ["whisper:base", "whisper:tiny"]),
])
def test_plan_de_calentamiento_por_modo(monkeypatch, modo, concurrente, use_oww, vad_silero,
                                         esperado):
    """El orden es el del primer uso del loop elegido, y el 'tiny' del wake solo
    entra con el wake difuso: el concurrente y el simple+oww nunca lo usan."""
    from interfaces import listener

    llamadas: list[str] = []
    monkeypatch.setattr(listener, "warm_vad", lambda: llamadas.append("silero") or True)
    plan = listener._warm_plan(concurrente, use_oww, vad_silero,
                               _SttPlan("base", llamadas), _SttPlan("tiny", llamadas))
    assert [nombre for nombre, _ in plan] == esperado
    assert ("whisper:tiny" in esperado) is (modo == "difuso")
    # Cada nombre llama al cargador de SU objeto (el del wake no carga la orden).
    for _nombre, cargar in plan:
        assert cargar() is True
    assert llamadas == esperado


def test_plan_difuso_con_el_mismo_tamano_carga_whisper_una_vez(monkeypatch):
    from interfaces import listener

    monkeypatch.setattr(listener, "warm_vad", lambda: True)
    plan = listener._warm_plan(False, False, True, _SttPlan("base", []), _SttPlan("base", []))
    assert [nombre for nombre, _ in plan] == ["silero", "whisper:base"]


def _settings_voz(tmp_path: Path, *, use_oww: bool, vad: str) -> types.SimpleNamespace:
    """Config mínima para run_listen con los STT/TTS/WakeWordDetector REALES: sin
    ducking ni umbrales con música (no arranca el monitor de playerctl)."""
    (tmp_path / "voz.onnx").touch()
    return types.SimpleNamespace(
        voice={"whisper_model": _ORDEN, "wake_whisper_model": _WAKE, "vad_backend": vad,
               "piper_voice": "voz.onnx", "wake_duck": False, "wake_threshold_media": 0.67},
        wake={"use_oww": use_oww, "oww_model": "hey_jarvis", "oww_threshold_media": 0.5},
        paths={"voces": tmp_path},
    )


def _arnes_arranque(monkeypatch, tmp_path: Path, settings: Any) -> dict[str, Any]:
    """run_listen con las clases de voz reales sobre los vendors falsos de `cargas`;
    todo lo demás (agente, LLM, HUD, señales, el cuerpo de los loops) es falso. El
    loop falso espera a que WarmVoice termine y anota con qué lo armaron."""
    from crotolamo.core import engine
    from crotolamo.voice import loop as voice_loop
    from crotolamo.voice import state as voice_state
    from interfaces import listener

    visto: dict[str, Any] = {"dichos": []}
    real_read = voice_state.read_control_enabled
    control = tmp_path / "control.json"
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.setattr(stt_mod, "_models", {})
    monkeypatch.setattr(stt_mod, "_model_locks", {})
    # El saludo no sintetiza: así Piper solo lo construye quien lo precalienta.
    monkeypatch.setattr(TTS, "speak", lambda self, text, **kw: visto["dichos"].append(text))
    monkeypatch.setattr(listener, "get_settings", lambda: settings)
    monkeypatch.setattr(listener, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(listener, "build_agent",
                        lambda confirm_fn=None: (types.SimpleNamespace(), None))
    monkeypatch.setattr(engine, "build_primary_llm",
                        lambda s: types.SimpleNamespace(needs_warmup=False))
    monkeypatch.setattr(listener, "_HUD_STATE_PATH", tmp_path / "hud_state.json")
    monkeypatch.setattr(listener, "CONTROL_PATH", control)
    monkeypatch.setattr(listener, "read_control_enabled", lambda path=control: real_read(path))
    for nombre in ("_ACTIVE_DUCKER", "_ACTIVE_MEDIA", "_ACTIVE_HUD_GATE"):
        monkeypatch.setattr(listener, nombre, None)

    def esperar_calentamiento() -> None:
        visto["calentador_vivo_en_el_loop"] = [
            t.name for t in threading.enumerate() if t.name == "WarmVoice"]
        for hilo in threading.enumerate():
            if hilo.name == "WarmVoice":
                hilo.join(5)

    class _Loop:
        def __init__(self, agent, stt, tts, wake_detector, **kwargs: Any) -> None:
            visto["loop"] = ("concurrente", stt.model_size)
            self.state = None

        def run(self) -> int:
            if not visto.get("no_esperar"):
                esperar_calentamiento()
            return 0

    def _simple(*, stt, wake_stt, use_oww, **kwargs: Any) -> int:
        visto["loop"] = ("simple", stt.model_size, wake_stt.model_size, use_oww)
        esperar_calentamiento()
        return 0

    monkeypatch.setattr(voice_loop, "VoiceLoop", _Loop)
    monkeypatch.setattr(listener, "_run_simple_loop", _simple)
    return visto


def _por_hilo(cargas: Construcciones) -> dict[str, list[str]]:
    hilos: dict[str, list[str]] = {}
    for etiqueta, hilo in cargas.hechas:
        hilos.setdefault(hilo, []).append(etiqueta)
    return hilos


@pytest.mark.parametrize("argv, use_oww, vad, loop, principal, calentador", [
    ([], True, "silero", "concurrente", ["oww"], ["piper", "silero", f"whisper:{_ORDEN}"]),
    # _SileroVad del oído no mira vad_backend: con 'energy' el concurrente sigue con Silero.
    ([], True, "energy", "concurrente", ["oww"], ["piper", "silero", f"whisper:{_ORDEN}"]),
    (["--simple"], True, "silero", "simple", ["oww"],
     ["piper", "silero", f"whisper:{_ORDEN}"]),
    (["--simple"], True, "energy", "simple", ["oww"], ["piper", f"whisper:{_ORDEN}"]),
    (["--simple"], False, "silero", "simple", [],
     ["piper", "silero", f"whisper:{_ORDEN}", f"whisper:{_WAKE}"]),
    (["--simple"], False, "energy", "simple", [],
     ["piper", f"whisper:{_ORDEN}", f"whisper:{_WAKE}"]),
    # use_oww=false sin --simple: el wake difuso, igual que con --simple.
    ([], False, "silero", "simple", [],
     ["piper", "silero", f"whisper:{_ORDEN}", f"whisper:{_WAKE}"]),
])
def test_run_listen_calienta_lo_que_usa_el_modo_en_su_orden(
        cargas, monkeypatch, tmp_path, argv, use_oww, vad, loop, principal, calentador):
    """De punta a punta con los cargadores reales: WarmVoice construye Piper
    primero (el saludo) y luego lo que el loop elegido usa, en orden de primer
    uso; el modelo de openWakeWord lo sigue cargando el hilo principal (decide el
    modo). El 'tiny' del wake: 0 construcciones fuera del difuso (antes, 1 en
    todos los modos: +64 MB que el concurrente jamás usa). Nada se construye dos
    veces ni en otro hilo."""
    from interfaces import listener

    visto = _arnes_arranque(monkeypatch, tmp_path, _settings_voz(tmp_path, use_oww=use_oww,
                                                                 vad=vad))
    assert listener.run_listen(argv) == 0
    assert _por_hilo(cargas) == ({"MainThread": principal, "WarmVoice": calentador}
                                 if principal else {"WarmVoice": calentador})
    assert visto["loop"][0] == loop and visto["loop"][1] == _ORDEN
    if loop == "simple":
        # El loop simple sigue recibiendo el STT del wake tal cual (difuso o no).
        assert visto["loop"][2:] == (_WAKE, use_oww)
    assert visto["dichos"][0].startswith("Crotolamo escuchando, patrón.")
    assert not [t for t in threading.enumerate() if t.name == "WarmVoice"]


def test_piper_carga_mientras_se_decide_el_modo(cargas, monkeypatch, tmp_path):
    """Piper se encola ANTES de la decisión del modo: su carga va a la vez que la
    de openWakeWord en el hilo principal (antes: detrás de dos Whisper). Se prueba
    con esperas acotadas, no con relojes: available() ve a Piper ya cargando."""
    from interfaces import listener

    piper_empezo = threading.Event()
    real_load = sys.modules["piper"].PiperVoice.load

    def load(path: str) -> Any:
        piper_empezo.set()
        return real_load(path)

    monkeypatch.setattr(sys.modules["piper"].PiperVoice, "load", staticmethod(load))
    real_available = WakeWordDetector.available
    vio_piper: list[bool] = []

    def available(self: WakeWordDetector) -> bool:
        vio_piper.append(piper_empezo.wait(5))
        return real_available(self)

    monkeypatch.setattr(WakeWordDetector, "available", available)
    _arnes_arranque(monkeypatch, tmp_path, _settings_voz(tmp_path, use_oww=True, vad="silero"))
    assert listener.run_listen([]) == 0
    assert vio_piper == [True]
    assert cargas.hechas[0] == ("piper", "WarmVoice")


def test_el_calentador_tiene_dueno_al_salir_espera_acotada_y_no_carga_mas(
        cargas, monkeypatch, tmp_path):
    """Si el loop termina con el calentamiento a medias (el oído se rindió al
    arrancar), run_listen no espera más de _WARM_STOP_S a la carga en curso, y lo
    que quedaba en cola ya no se carga: el hilo termina en cuanto suelta esa carga."""
    from interfaces import listener

    soltar = threading.Event()
    piper_empezo = threading.Event()
    real_load = sys.modules["piper"].PiperVoice.load

    def load(path: str) -> Any:
        piper_empezo.set()
        soltar.wait(10)
        return real_load(path)

    monkeypatch.setattr(sys.modules["piper"].PiperVoice, "load", staticmethod(load))
    monkeypatch.setattr(listener, "_WARM_STOP_S", 0.2)
    real_plan = listener._warm_plan

    def plan(*args: Any) -> Any:
        # Determinista: Piper ya está cargando (en vuelo) cuando el loop termina. Si
        # WarmVoice arrancara tarde (máquina cargada), stop() llegaría antes que su
        # primer item y, bien hecho, no cargaría nada: no es lo que se prueba aquí.
        assert piper_empezo.wait(5)
        return real_plan(*args)

    monkeypatch.setattr(listener, "_warm_plan", plan)
    visto = _arnes_arranque(monkeypatch, tmp_path,
                            _settings_voz(tmp_path, use_oww=True, vad="silero"))
    visto["no_esperar"] = True
    try:
        assert listener.run_listen([]) == 0
        calentador = [t for t in threading.enumerate() if t.name == "WarmVoice"]
        assert len(calentador) == 1, "run_listen esperó sin límite a la carga en curso"
    finally:
        soltar.set()
    calentador[0].join(5)
    assert not calentador[0].is_alive()
    assert _por_hilo(cargas) == {"MainThread": ["oww"], "WarmVoice": ["piper"]}


def test_un_arranque_que_revienta_al_decidir_el_modo_no_deja_al_calentador_esperando(
        cargas, monkeypatch, tmp_path):
    """El error sube como siempre, y el hilo WarmVoice (que ya tenía a Piper) no
    se queda colgado esperando el resto del plan."""
    from interfaces import listener

    def available(self: WakeWordDetector) -> bool:
        raise RuntimeError("openwakeword roto")

    monkeypatch.setattr(WakeWordDetector, "available", available)
    _arnes_arranque(monkeypatch, tmp_path, _settings_voz(tmp_path, use_oww=True, vad="silero"))
    with pytest.raises(RuntimeError, match="openwakeword roto"):
        listener.run_listen([])
    assert esperar(lambda: not [t for t in threading.enumerate() if t.name == "WarmVoice"])
    assert set(cargas.etiquetas()) <= {"piper"}


def test_el_calentador_sobrevive_a_un_cargador_que_revienta(monkeypatch, caplog):
    """Frontera de excepciones: un cargador que lanza (en vez de devolver False) no
    mata al hilo; se avisa y se sigue con el resto, en orden."""
    from interfaces import listener

    hechos: list[str] = []

    def revienta() -> bool:
        raise RuntimeError("cargador roto")

    caplog.set_level(logging.INFO, logger="crotolamo")
    calentador = listener._Warmer()
    calentador.add("piper", lambda: hechos.append("piper") or True)
    calentador.add("silero", revienta)
    calentador.add("whisper:base", lambda: hechos.append("whisper:base") or True)
    calentador.add("whisper:tiny", lambda: False)  # no cargó: lo decide su primer uso
    calentador.close()
    calentador.join(5)
    assert hechos == ["piper", "whisper:base"]
    mensajes = [(r.levelno, r.getMessage()) for r in caplog.records
                if r.name.startswith("crotolamo.listener")]
    assert (logging.WARNING, "no pude precalentar la voz: cargador roto (silero)") in mensajes
    assert (logging.INFO, "voz caliente (piper, whisper:base); sin precalentar (lo decide "
            "su primer uso): silero, whisper:tiny") in mensajes


# ---------------------------------------------------------------------------
# Revisión del paso 2: el aviso de una voz rota y el dueño en todas las salidas
# ---------------------------------------------------------------------------

def _causa(fn: Callable[[], Any]) -> str:
    """El texto del error tal cual (lo que el aviso de antes ponía tras los dos puntos)."""
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - se compara el texto tal cual
        return str(error)
    raise AssertionError("no lanzó")


@pytest.mark.parametrize("argv, use_oww, rompe, cargados, sin_cargar", [
    # Concurrente sin el extra [voice] completo: falta faster_whisper.
    ([], True, "faster_whisper", ["piper", "silero"], [f"whisper:{_ORDEN}"]),
    # Difuso (el modo del patrón): los dos Whisper fallan por LA MISMA causa; antes
    # el hilo se cortaba en el primero y había un solo aviso, y sigue habiéndolo.
    (["--simple"], False, "faster_whisper", ["piper", "silero"],
     [f"whisper:{_ORDEN}", f"whisper:{_WAKE}"]),
    # Un .onnx de Piper que no abre.
    ([], True, "piper", ["silero", f"whisper:{_ORDEN}"], ["piper"]),
])
def test_una_voz_rota_se_avisa_al_arrancar_como_antes(
        cargas, monkeypatch, tmp_path, caplog, argv, use_oww, rompe, cargados, sin_cargar):
    """Antes (HEAD), una carga de Whisper o Piper que fallaba al precalentar dejaba
    en el journal, con el nivel INFO de fábrica, 'no pude precalentar la voz: <el
    error>' (WARNING): sin faster_whisper, con la instrucción para instalarlo. Con
    los warm() ese aviso bajó a DEBUG y la causa solo aparecía tras el primer wake.
    Vuelve a ser un WARNING al arrancar, con el texto de antes como comienzo, uno
    por causa; y nada queda cacheado: el primer uso real da el error de siempre."""
    from interfaces import listener

    if rompe == "faster_whisper":
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        causa = _causa(STT(model_size=_ORDEN)._get_model)
        assert "pip install -e '.[voice]'" in causa
    else:
        def load(path: str) -> Any:
            raise RuntimeError("onnx corrupto")

        monkeypatch.setattr(sys.modules["piper"].PiperVoice, "load", staticmethod(load))
        causa = "onnx corrupto"
    _arnes_arranque(monkeypatch, tmp_path, _settings_voz(tmp_path, use_oww=use_oww,
                                                         vad="silero"))
    caplog.set_level(logging.DEBUG, logger="crotolamo")
    assert listener.run_listen(argv) == 0
    avisos = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.WARNING and r.threadName == "WarmVoice"]
    assert len(avisos) == 1, avisos
    assert avisos[0].startswith(f"no pude precalentar la voz: {causa}")
    assert (f"voz caliente ({', '.join(cargados)}); sin precalentar (lo decide su primer "
            f"uso): {', '.join(sin_cargar)}") in [r.getMessage() for r in caplog.records]
    if rompe == "faster_whisper":
        assert _ORDEN not in stt_mod._models
        assert _causa(STT(model_size=_ORDEN)._get_model) == causa


@pytest.mark.parametrize("argv, donde, error", [
    # Ctrl+C sin el manejador de señales (run_listen fuera del hilo principal) con un
    # handshake MCP lento, o durante el saludo: BaseException, no Exception.
    ([], "build_agent", KeyboardInterrupt),
    ([], "saludo", KeyboardInterrupt),
    # Ya cerrado el plan: el monitor de música, el import del loop concurrente, o
    # el prólogo del loop simple revientan.
    ([], "media", RuntimeError),
    ([], "import_loop", ImportError),
    (["--simple"], "vigia", RuntimeError),
])
def test_el_calentador_se_para_salga_por_donde_salga_run_listen(
        cargas, monkeypatch, tmp_path, argv, donde, error):
    """UN dueño para WarmVoice: salga run_listen como salga, lo que quedaba en cola
    ya no se carga y la carga en curso se espera (acotado). Antes solo tres salidas
    lo paraban: con un error entre el cierre del plan y los loops, el intérprete se
    apagaba con la sesión ONNX de Piper o Silero a medio crear y abortaba (SIGABRT,
    rc 134, medido 4/6 con los modelos reales) en vez de salir con su código."""
    from interfaces import listener

    soltar = threading.Event()
    piper_empezo = threading.Event()
    real_load = sys.modules["piper"].PiperVoice.load

    def load(path: str) -> Any:
        piper_empezo.set()
        soltar.wait(10)
        return real_load(path)

    monkeypatch.setattr(sys.modules["piper"].PiperVoice, "load", staticmethod(load))
    monkeypatch.setattr(listener, "_WARM_STOP_S", 0.2)
    real_plan = listener._warm_plan

    def plan(*args: Any) -> Any:
        # Determinista: Piper ya está cargando (en vuelo) cuando llega cada fallo.
        assert piper_empezo.wait(5)
        return real_plan(*args)

    monkeypatch.setattr(listener, "_warm_plan", plan)
    _arnes_arranque(monkeypatch, tmp_path, _settings_voz(tmp_path, use_oww=True, vad="silero"))

    def revienta(*args: Any, **kwargs: Any) -> Any:
        raise error(donde)

    if donde == "build_agent":
        monkeypatch.setattr(listener, "build_agent", revienta)
    elif donde == "saludo":
        monkeypatch.setattr(TTS, "speak", revienta)
    elif donde == "media":
        monkeypatch.setattr(listener, "MediaMonitor", revienta)
    elif donde == "import_loop":
        monkeypatch.setitem(sys.modules, "crotolamo.voice.loop", None)
    else:
        monkeypatch.setattr(listener, "_start_control_watcher", revienta)
    try:
        with pytest.raises(error):
            listener.run_listen(argv)
    finally:
        soltar.set()
    hilos = [t for t in threading.enumerate() if t.name == "WarmVoice"]
    for hilo in hilos:
        hilo.join(5)
    assert not [t for t in hilos if t.is_alive()]
    # Silero y el Whisper de la orden seguían en cola: stop() llegó antes de soltarlos.
    assert _por_hilo(cargas) == {"MainThread": ["oww"], "WarmVoice": ["piper"]}


# ---------------------------------------------------------------------------
# Paso 3: el LLM se calienta solo si es un modelo local
# ---------------------------------------------------------------------------

_OK = [{"role": "user", "content": "di solo: ok"}]


def _cliente_que_cuenta(monkeypatch, cls: type, llamadas: list) -> None:
    """chat() de la CLASE anotando (clase, mensajes): sin red."""
    monkeypatch.setattr(cls, "chat", lambda self, messages, tools=None: llamadas.append(
        (cls.__name__, messages)))


def test_warm_llm_con_cerebro_en_la_nube_no_manda_nada(monkeypatch, caplog):
    from crotolamo.core import engine
    from crotolamo.core.glm import GLMClient
    from interfaces import listener

    llamadas: list = []
    _cliente_que_cuenta(monkeypatch, GLMClient, llamadas)
    monkeypatch.setattr(engine, "build_primary_llm", lambda s: GLMClient(api_key="k"))
    caplog.set_level(logging.INFO, logger="crotolamo")
    listener._warm_llm(object())
    assert llamadas == []
    assert any("no hay LLM que calentar" in r.getMessage() for r in caplog.records)


def test_warm_llm_con_ollama_calienta_una_vez_como_siempre(monkeypatch):
    from crotolamo.core import engine
    from crotolamo.core.llm import LLMClient
    from interfaces import listener

    llamadas: list = []
    _cliente_que_cuenta(monkeypatch, LLMClient, llamadas)
    monkeypatch.setattr(engine, "build_primary_llm", lambda s: LLMClient())
    listener._warm_llm(object())
    assert llamadas == [("LLMClient", _OK)]


@pytest.mark.parametrize("clase, calienta", [("LLMClient", True), ("GLMClient", False)])
def test_warm_llm_sin_needs_warmup_decide_por_el_tipo(monkeypatch, clase, calienta):
    """Si el cliente no declara needs_warmup (el cambio del cerebro aún no está),
    la regla de siempre: solo el cliente local se calienta."""
    from crotolamo.core import engine
    from crotolamo.core.glm import GLMClient
    from crotolamo.core.llm import LLMClient
    from interfaces import listener

    cls = {"LLMClient": LLMClient, "GLMClient": GLMClient}[clase]
    monkeypatch.delattr(cls, "needs_warmup", raising=False)
    llamadas: list = []
    _cliente_que_cuenta(monkeypatch, cls, llamadas)
    cliente = LLMClient() if cls is LLMClient else GLMClient(api_key="k")
    assert not hasattr(cliente, "needs_warmup")
    monkeypatch.setattr(engine, "build_primary_llm", lambda s: cliente)
    listener._warm_llm(object())
    assert llamadas == ([(clase, _OK)] if calienta else [])


@pytest.mark.parametrize("backend, key, esperado", [
    ("glm", "secreta", []),                       # la nube: 0 llamadas (antes, 1)
    ("glm", None, [("LLMClient", _OK)]),          # glm sin key cae a Ollama: se calienta
    ("ollama", "secreta", [("LLMClient", _OK)]),  # Ollama primario: igual que antes
])
def test_warm_llm_con_el_cableado_real_de_build_primary_llm(monkeypatch, backend, key, esperado):
    from crotolamo.core.glm import API_KEY_ENVS, GLMClient
    from crotolamo.core.llm import LLMClient
    from crotolamo.settings import get_settings
    from interfaces import listener

    for env in API_KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    if key is not None:
        monkeypatch.setenv(API_KEY_ENVS[0], key)
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", backend)
    llamadas: list = []
    _cliente_que_cuenta(monkeypatch, GLMClient, llamadas)
    _cliente_que_cuenta(monkeypatch, LLMClient, llamadas)
    listener._warm_llm(get_settings())
    assert llamadas == esperado


def test_warm_llm_que_falla_solo_avisa(monkeypatch, caplog):
    from crotolamo.core import engine
    from interfaces import listener

    def revienta(settings: Any) -> Any:
        raise RuntimeError("sin config")

    monkeypatch.setattr(engine, "build_primary_llm", revienta)
    caplog.set_level(logging.WARNING, logger="crotolamo")
    listener._warm_llm(object())  # no lanza
    assert [r.getMessage() for r in caplog.records] == ["no pude calentar el LLM: sin config"]


# ---------------------------------------------------------------------------
# Paso 4: handshakes a la vez, registro en serie y en el orden de la config
# ---------------------------------------------------------------------------

def test_los_handshakes_de_varios_servers_corren_a_la_vez(monkeypatch):
    """Tres servers que tardan en arrancar: los tres initialize en vuelo a la
    vez (en serie había uno). La barrera solo se pasa si los tres llegan a la
    vez; en serie el primero la espera hasta su plazo y todos fallan."""
    barrera = threading.Barrier(3, timeout=5)
    contador = Contador(monkeypatch, antes_de_initialize=lambda _c: barrera.wait())
    servers = {f"s{i}": {"command": lento(0.3), "prefix": f"s{i}"} for i in range(3)}
    registry = Registry()
    names = bridge.register_mcp_tools(registry, mk_settings(servers))

    assert contador.max_en_vuelo == 3
    assert bridge.connected_servers() == ["s0", "s1", "s2"]
    assert {n.split("_")[1] for n in names} == {"s0", "s1", "s2"}
    # Mismos recursos que en serie: un lanzamiento y un handshake por server.
    assert sorted(contador.starts) == ["s0", "s1", "s2"]
    assert sorted(contador.inits) == ["s0", "s1", "s2"]
    # Los handshakes van en hilos ayudantes, y ninguno queda vivo al volver.
    assert all(h.startswith("mcp-start") for h in contador.hilos.values())
    assert hilos_de_arranque() == []
    assert bridge._STARTING == set()


def test_registro_identico_al_de_serie_aunque_el_segundo_conteste_antes(monkeypatch):
    """Dos servers con el MISMO prefijo: el sufijo _2 cae en el segundo de la
    config, como en serie, aunque el segundo termine su handshake antes. Y un
    server roto en medio no cambia nada. Se compara contra _connect en serie."""
    servers = {
        "uno": {"command": lento(0.6), "prefix": "x"},
        "roto": {"command": ["/no/existe/binario"]},
        "dos": {"command": lento(0.05), "prefix": "x"},
        "tres": {"command": lento(0.2)},
    }
    settings = mk_settings(servers)
    contador = Contador(monkeypatch)
    registry = Registry()
    names = bridge.register_mcp_tools(registry, settings)
    paralelo = foto(registry, names)
    assert contador.max_en_vuelo >= 2  # de verdad hubo solape
    assert "mcp_x_echo" in names and "mcp_x_echo_2" in names
    assert paralelo["servers"]["mcp_x_echo"] == "uno"
    assert paralelo["servers"]["mcp_x_echo_2"] == "dos"

    # Idempotente: no relanza los conectados ni cambia los nombres (el que
    # nunca conectó se reintenta, como en serie: es un server sin _STATE).
    lanzados = len(contador.starts)
    assert bridge.register_mcp_tools(registry, settings) == names
    assert contador.starts[lanzados:] == ["roto"]
    assert hilos_de_arranque() == []

    # La referencia: el registro en serie de siempre, server a server.
    bridge.close_all()
    serie_registry = Registry()
    serie_names: list[str] = []
    for server in bridge.load_mcp_config(settings).servers:
        serie_names.extend(bridge._connect(serie_registry, server))
    assert foto(serie_registry, serie_names) == paralelo


def test_un_solo_server_sigue_en_el_hilo_del_llamador(monkeypatch):
    contador = Contador(monkeypatch)
    names = bridge.register_mcp_tools(Registry(), mk_settings({"fake": {"command": FAKE_CMD}}))
    assert "mcp_fake_echo" in names
    assert contador.hilos == {"fake": threading.current_thread().name}
    assert hilos_de_arranque() == []


def test_un_server_muerto_en_state_se_relanza_en_serie_y_con_los_mismos_nombres(monkeypatch):
    """El proceso viejo se retira ANTES de lanzar el nuevo (dos procesos del
    mismo server a la vez podían pisarse un lockfile), como en serie."""
    settings = mk_settings({"a": {"command": FAKE_CMD}, "b": {"command": FAKE_CMD}})
    registry = Registry()
    names = bridge.register_mcp_tools(registry, settings)
    viejo = bridge._STATE["a"].client
    viejo._on_dead = None  # que siga en _STATE muerto, sin el aviso que lo retira
    assert viejo._proc is not None
    os.killpg(viejo._proc.pid, signal.SIGKILL)
    assert esperar(lambda: not viejo.alive)

    contador = Contador(monkeypatch)
    again = bridge.register_mcp_tools(registry, settings)
    assert again == names
    assert contador.starts == ["a"]  # solo el muerto; "b" sigue con su proceso
    assert contador.hilos == {"a": threading.current_thread().name}
    assert bridge._STATE["a"].client is not viejo
    assert bridge.connected_servers() == ["b", "a"]


# ---------------------------------------------------------------------------
# mcp#14 con varios handshakes en vuelo: el apagado no deja huérfanos
# ---------------------------------------------------------------------------

def _mudos(tmp_path: Path, n: int) -> tuple[dict, list[Path]]:
    """n servers que nunca contestan ni leen stdin (ignoran el EOF); cada uno
    apunta su pid."""
    code = (
        "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    pidfiles = [tmp_path / f"pid{i}" for i in range(n)]
    servers = {f"mudo{i}": {"command": [sys.executable, "-c", code, str(p)]}
               for i, p in enumerate(pidfiles)}
    return servers, pidfiles


def _pids(pidfiles: list[Path]) -> list[int]:
    return [int(p.read_text()) for p in pidfiles if p.exists() and p.read_text().strip()]


def _matar_restos(pidfiles: list[Path]) -> None:
    for pid in _pids(pidfiles):
        if vivo(pid):
            os.kill(pid, signal.SIGKILL)


def _senal_cuando_arranquen(pidfiles: list[Path], visto: list[bool]) -> threading.Thread:
    """Manda SIGUSR2 al hilo principal cuando TODOS los servers están lanzados
    a la vez (o a los 3 s, para que la versión en serie falle acotada)."""
    principal = threading.main_thread().ident
    assert principal is not None

    def run() -> None:
        visto.append(esperar(lambda: len(_pids(pidfiles)) == len(pidfiles), plazo=3.0))
        signal.pthread_kill(principal, signal.SIGUSR2)

    hilo = threading.Thread(target=run, name="test-senal", daemon=True)
    hilo.start()
    return hilo


def test_terminate_all_por_senal_con_varios_handshakes_en_vuelo_no_deja_vivo_ninguno(tmp_path):
    """El apagado del listener: la señal se atiende en el hilo principal (el
    que arranca los servers) y llama terminate_all() antes de os._exit. Con
    los tres handshakes en vuelo, los tres servers mueren y el arranque aborta
    al momento en vez de agotar el presupuesto."""
    servers, pidfiles = _mudos(tmp_path, 3)
    llamadas: list[int] = []

    def manejador(signum, _frame) -> None:
        llamadas.append(signum)
        bridge.terminate_all()

    previo = signal.signal(signal.SIGUSR2, manejador)
    visto: list[bool] = []
    try:
        _senal_cuando_arranquen(pidfiles, visto)
        t0 = time.monotonic()
        names = bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=6))
        transcurrido = time.monotonic() - t0
        assert visto == [True], "los tres handshakes no estaban en vuelo a la vez"
        assert llamadas == [signal.SIGUSR2]
        assert names == []
        pids = _pids(pidfiles)
        assert len(pids) == 3
        assert esperar(lambda: not any(vivo(p) for p in pids), plazo=3.0), "quedó un huérfano"
        assert transcurrido < 5.0  # abortó; no esperó el plazo de 6 s
        assert bridge._STARTING == set() and bridge.connected_servers() == []
        assert hilos_de_arranque() == []
    finally:
        signal.signal(signal.SIGUSR2, previo)
        _matar_restos(pidfiles)


def test_ctrl_c_con_varios_handshakes_en_vuelo_cierra_todos_y_a_la_vez(
    tmp_path, monkeypatch, caplog,
):
    """Ctrl+C en el shell (KeyboardInterrupt en el hilo principal, que espera
    un handshake): se propaga, ningún server queda vivo ni en _STARTING, y los
    TRES se cierran a la vez, también el que el hilo principal esperaba. Cada
    close() de un server que ignora el EOF cuesta ~1 s; el arranque en serie,
    con un solo server en vuelo, pagaba uno. Cerrar primero el esperado y luego
    los otros dos pagaba dos; esperar a los handshakes antes de cerrarlos,
    todo el presupuesto (15 s). Y el log no dice que unos servers que el
    propio Ctrl+C cerró "no arrancaron" (en serie no decía nada)."""
    caplog.set_level(logging.DEBUG, logger="crotolamo.mcp.bridge")
    servers, pidfiles = _mudos(tmp_path, 3)
    cierres = {"en_vuelo": 0, "max": 0}
    lock = threading.Lock()
    real_close = StdioMCPClient.close

    def close(client: StdioMCPClient) -> None:
        # Solo los cierres de verdad: el del hilo del handshake que aborta
        # porque ya lo cerraron es un no-op y no cuenta.
        if client._closed:
            return real_close(client)
        with lock:
            cierres["en_vuelo"] += 1
            cierres["max"] = max(cierres["max"], cierres["en_vuelo"])
        try:
            real_close(client)
        finally:
            with lock:
                cierres["en_vuelo"] -= 1

    monkeypatch.setattr(StdioMCPClient, "close", close)

    senal: list[float] = []

    def manejador(_signum, _frame) -> None:
        senal.append(time.monotonic())
        raise KeyboardInterrupt

    previo = signal.signal(signal.SIGUSR2, manejador)
    visto: list[bool] = []
    try:
        _senal_cuando_arranquen(pidfiles, visto)
        with pytest.raises(KeyboardInterrupt):
            bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=15))
        abortar = time.monotonic() - senal[0]
        assert visto == [True]
        pids = _pids(pidfiles)
        assert len(pids) == 3
        assert not any(vivo(p) for p in pids), "quedó un server vivo tras el Ctrl+C"
        assert bridge._STARTING == set() and bridge.connected_servers() == []
        assert hilos_de_arranque() == []
        assert cierres["max"] == 3  # los tres a la vez: una ronda de close(), no dos
        # ~1 s (un close()); con los handshakes esperados antes de cerrar, ~15 s.
        assert abortar < 4.0, f"el Ctrl+C tardó {abortar:.2f}s en propagarse"
        avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("no arrancó" in m for m in avisos), avisos
        assert caplog.text.count("handshake abortado") == 3
    finally:
        signal.signal(signal.SIGUSR2, previo)
        _matar_restos(pidfiles)


def test_close_all_desde_otro_hilo_alcanza_a_todos_los_handshakes_en_vuelo(tmp_path):
    servers, pidfiles = _mudos(tmp_path, 3)
    out: list = []
    hilo = threading.Thread(
        target=lambda: out.append(bridge.register_mcp_tools(
            Registry(), mk_settings(servers, startup_timeout_s=6),
        )),
        daemon=True,
    )
    try:
        hilo.start()
        assert esperar(lambda: len(_pids(pidfiles)) == 3, plazo=3.0)
        bridge.close_all()
        pids = _pids(pidfiles)
        assert esperar(lambda: not any(vivo(p) for p in pids), plazo=3.0)
        hilo.join(timeout=5)
        assert not hilo.is_alive()
        assert out == [[]]
        assert bridge._STARTING == set()
    finally:
        _matar_restos(pidfiles)


def test_ctrl_c_dentro_de_un_handshake_en_paralelo_cierra_los_no_registrados(monkeypatch):
    """KeyboardInterrupt dentro del handshake de "b" (en un hilo ayudante): se
    propaga al llamador cuando llega su turno. "a", ya registrado, sigue como
    en serie (lo cierra close_all/atexit); "b" y "c", lanzado a la vez, quedan
    cerrados y fuera de _STARTING."""

    def antes(client: StdioMCPClient) -> None:
        if client.name == "b":
            raise KeyboardInterrupt

    contador = Contador(monkeypatch, antes_de_initialize=antes)
    servers = {n: {"command": FAKE_CMD} for n in ("a", "b", "c")}
    with pytest.raises(KeyboardInterrupt):
        bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert sorted(contador.starts) == ["a", "b", "c"]
    assert bridge.connected_servers() == ["a"]
    for name in ("b", "c"):
        proc = contador.clientes[name]._proc
        assert proc is not None and proc.poll() is not None, f"'{name}' siguió vivo"
    assert bridge._STARTING == set()
    assert hilos_de_arranque() == []


def test_ctrl_c_mientras_se_lanzan_cierra_los_ya_lanzados(monkeypatch):
    """Ctrl+C entre un Popen y el siguiente (se lanzan en el hilo del llamador):
    el ya lanzado, con su handshake en vuelo, se cierra; el siguiente no llega
    a lanzarse."""
    lanzados: list[StdioMCPClient] = []
    real_start = StdioMCPClient.start

    def start(client: StdioMCPClient) -> None:
        if client.name == "b":
            raise KeyboardInterrupt
        real_start(client)
        lanzados.append(client)

    monkeypatch.setattr(StdioMCPClient, "start", start)
    servers = {n: {"command": FAKE_CMD} for n in ("a", "b", "c")}
    with pytest.raises(KeyboardInterrupt):
        bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert [c.name for c in lanzados] == ["a"]
    proc = lanzados[0]._proc
    assert proc is not None and proc.poll() is not None, "'a' siguió vivo tras el Ctrl+C"
    assert bridge._STARTING == set() and bridge.connected_servers() == []
    assert hilos_de_arranque() == []


def test_server_que_otro_hilo_conecto_entretanto_se_cierra_sin_registrar(monkeypatch):
    """Si mientras corren los handshakes otro hilo conecta el mismo server, el
    lanzado aquí no se usa: se cierra (sin huérfano) y se reponen las tools
    del conectado, como en el camino idempotente de siempre."""
    otro = Registry()

    def antes(client: StdioMCPClient) -> None:
        if client.name == "a":
            bridge.register_mcp_tools(otro, mk_settings({"b": {"command": FAKE_CMD}}))

    contador = Contador(monkeypatch, antes_de_initialize=antes)
    registry = Registry()
    names = bridge.register_mcp_tools(
        registry, mk_settings({"a": {"command": FAKE_CMD}, "b": {"command": FAKE_CMD}}),
    )
    clientes_b = [c for c in contador.todos if c.name == "b"]
    assert len(clientes_b) == 2
    conectado = bridge._STATE["b"].client
    sobrante = next(c for c in clientes_b if c is not conectado)
    assert conectado.alive
    assert sobrante._proc is not None and sobrante._proc.poll() is not None
    assert "mcp_b_echo" in names and "mcp_b_echo" in registry.names()
    assert bridge.connected_servers() == ["b", "a"]
    assert bridge._STARTING == set()
    assert hilos_de_arranque() == []


# ---------------------------------------------------------------------------
# mcp#11 con varios handshakes en vuelo: un server que muere se retira
# ---------------------------------------------------------------------------

def _matar_y_esperar(client: StdioMCPClient) -> None:
    assert client._proc is not None
    os.killpg(client._proc.pid, signal.SIGKILL)
    assert esperar(lambda: not client.alive)


def test_server_registrado_que_muere_durante_el_handshake_de_otro_se_retira(monkeypatch, caplog):
    """"a" (primero en la config) ya está registrado cuando "b" sigue en su
    handshake; "a" muere entonces y su aviso (on_dead) lo retira, igual que en
    serie (donde "b" arrancaba después)."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    hecho = threading.Event()

    def antes(client: StdioMCPClient) -> None:
        if client.name != "b":
            return
        assert esperar(lambda: "a" in bridge.connected_servers())
        _matar_y_esperar(bridge._STATE["a"].client)
        assert esperar(lambda: "a" not in bridge.connected_servers())
        hecho.set()

    Contador(monkeypatch, antes_de_initialize=antes)
    registry = Registry()
    bridge.register_mcp_tools(
        registry, mk_settings({"a": {"command": FAKE_CMD}, "b": {"command": lento(0.3)}}),
    )
    assert hecho.is_set()
    assert bridge.connected_servers() == ["b"]
    assert not any(n.startswith("mcp_a_") for n in registry.names())
    assert any(n.startswith("mcp_b_") for n in registry.names())
    assert "mcp:a" not in router.dynamic_groups()
    assert "transporte roto" in caplog.text
    assert bridge._STARTING == set()


def test_server_que_muere_esperando_su_registro_no_se_registra(monkeypatch, caplog):
    """"a" (segundo en la config) termina su handshake mientras "b" sigue en el
    suyo y muere esperando su turno de registro: no se registra, su proceso
    queda cerrado y el log dice por qué (como el que muere tras su handshake)."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    contador: Contador

    def antes(client: StdioMCPClient) -> None:
        if client.name != "b":
            return
        assert esperar(lambda: "a" in contador.listados and contador.listados["a"].is_set())
        _matar_y_esperar(contador.clientes["a"])

    contador = Contador(monkeypatch, antes_de_initialize=antes)
    registry = Registry()
    names = bridge.register_mcp_tools(
        registry, mk_settings({"b": {"command": lento(0.3)}, "a": {"command": FAKE_CMD}}),
    )
    assert not any(n.startswith("mcp_a_") for n in names + registry.names())
    assert any(n.startswith("mcp_b_") for n in names)
    assert bridge.connected_servers() == ["b"]
    assert "mcp:a" not in router.dynamic_groups()
    assert "se cayó tras el handshake" in caplog.text
    proc = contador.clientes["a"]._proc
    assert proc is not None and proc.poll() is not None
    assert bridge._STARTING == set()


# ---------------------------------------------------------------------------
# Revisión del paso 4: ventanas que abría el arranque en paralelo
# ---------------------------------------------------------------------------

def test_un_server_que_no_arranca_en_paralelo_sigue_avisando(caplog):
    """Lo que baja a DEBUG es solo el handshake que aborta el propio registro
    (Ctrl+C, apagado): un server que de verdad no arranca sigue avisando."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    servers = {"ok": {"command": FAKE_CMD}, "muere": {"command": [sys.executable, "-c", "pass"]}}
    names = bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert any(n.startswith("mcp_ok_") for n in names)
    assert bridge.connected_servers() == ["ok"]
    assert "MCP 'muere': no arrancó, lo salto" in caplog.text


def test_server_con_el_handshake_hecho_sigue_al_alcance_del_apagado_mientras_espera(monkeypatch):
    """mcp#14 en la ventana que abre el paralelo: "rapido" ya terminó su
    handshake y espera su turno de registro detrás de "lento". Si el apagado
    por señal llega entonces, terminate_all() tiene que alcanzarlo: sigue en
    _STARTING hasta registrarse o cerrarse. Si no, el listener sale con
    os._exit y "rapido" (tarda 30 s en salir tras el EOF) queda huérfano."""
    real_handshake = bridge._handshake
    hecho = threading.Event()
    pids: list[int] = []

    def handshake(server, client, deadline, *args):
        tools = real_handshake(server, client, deadline, *args)
        if server.name == "rapido":
            assert client._proc is not None
            pids.append(client._proc.pid)
            hecho.set()
        return tools

    monkeypatch.setattr(bridge, "_handshake", handshake)
    llamadas: list[int] = []

    def manejador(signum, _frame) -> None:
        llamadas.append(signum)
        bridge.terminate_all()

    principal = threading.main_thread().ident
    assert principal is not None

    def senal() -> None:
        if hecho.wait(5):
            signal.pthread_kill(principal, signal.SIGUSR2)

    previo = signal.signal(signal.SIGUSR2, manejador)
    hilo = threading.Thread(target=senal, name="test-senal", daemon=True)
    try:
        hilo.start()
        servers = {
            "lento": {"command": lento(2.0)},
            "rapido": {"command": FAKE_CMD, "env": {"FAKE_MCP_EOF_DELAY": "30"}},
        }
        names = bridge.register_mcp_tools(Registry(), mk_settings(servers))
        hilo.join(5)
        assert llamadas == [signal.SIGUSR2]
        assert len(pids) == 1
        assert esperar(lambda: not vivo(pids[0]), plazo=3.0), "'rapido' quedó huérfano"
        assert names == [] and bridge.connected_servers() == []
        assert bridge._STARTING == set()
    finally:
        signal.signal(signal.SIGUSR2, previo)
        for pid in pids:
            if vivo(pid):
                os.kill(pid, signal.SIGKILL)


def _hijo_que_no_es_server(env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    """Un hijo directo y líder de su sesión que NO es un server MCP, como las
    apps que abre run_detached (Blender, el terminal...): parar Crotolamo no
    debe matarlo."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True, env=env,
    )


def _popen_que_corta(monkeypatch, cuando: int,
                     accion: Callable[[], None]) -> list[subprocess.Popen[bytes]]:
    """Envuelve el Popen de los clientes MCP: en el lanzamiento número `cuando`,
    con el hijo ya creado y antes de que Popen devuelva (su cliente aún no sabe
    el pid), corre `accion`, lo que haría una señal atendida justo ahí."""
    real = subprocess.Popen
    hijos: list[subprocess.Popen[bytes]] = []

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc = real(*args, **kwargs)
        hijos.append(proc)
        if len(hijos) == cuando:
            accion()
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    return hijos


def _recoger(*procs: subprocess.Popen[bytes]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(5)


def test_terminate_all_en_pleno_popen_alcanza_al_server_sin_pid_y_solo_a_el(tmp_path, monkeypatch):
    """La señal de apagado puede atenderse justo tras el fork de un server, antes
    de que su Popen devuelva: el proceso existe pero su cliente aún no sabe el
    pid, y terminate_now() no lo alcanzaba (huérfano; con los servers lanzados
    seguidos, esas ventanas caen todas al principio del arranque). Y SOLO a él:
    las apps que abre run_detached también son hijos directos y líderes de su
    sesión, y la marca heredada de otro Crotolamo no es la de este."""
    app = _hijo_que_no_es_server()
    ajeno = _hijo_que_no_es_server({**os.environ, "CROTOLAMO_MCP_LANZAMIENTO": "1.1"})
    servers, _pidfiles = _mudos(tmp_path, 2)
    hijos = _popen_que_corta(monkeypatch, 2, bridge.terminate_all)
    try:
        t0 = time.monotonic()
        names = bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=6))
        assert time.monotonic() - t0 < 5.0  # abortó; no agotó el presupuesto
        assert names == [] and len(hijos) == 2
        assert esperar(lambda: not any(vivo(p.pid) for p in hijos), plazo=3.0), "quedó huérfano"
        assert vivo(app.pid) and vivo(ajeno.pid)
        assert bridge._STARTING == set() and bridge._SPAWNING == set()
        assert hilos_de_arranque() == []
    finally:
        _recoger(*hijos, app, ajeno)


def test_ctrl_c_en_pleno_popen_no_deja_huerfano(tmp_path, monkeypatch):
    """Lo mismo con Ctrl+C en el shell: el KeyboardInterrupt sale de dentro del
    Popen con el hijo ya creado. Cerrar el cliente no lo alcanzaba (no sabe su
    pid) y un server que ignora el EOF sobrevivía a Crotolamo."""
    app = _hijo_que_no_es_server()
    servers, _pidfiles = _mudos(tmp_path, 3)

    def ctrl_c() -> None:
        raise KeyboardInterrupt

    hijos = _popen_que_corta(monkeypatch, 2, ctrl_c)
    try:
        with pytest.raises(KeyboardInterrupt):
            bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=6))
        assert len(hijos) == 2  # el tercero ni se lanzó
        assert esperar(lambda: not any(vivo(p.pid) for p in hijos), plazo=3.0), "quedó huérfano"
        assert vivo(app.pid)
        assert bridge._STARTING == set() and bridge._SPAWNING == set()
        assert hilos_de_arranque() == []
    finally:
        _recoger(*hijos, app)


def _lecturas_de_environ(monkeypatch, vacias: Callable[[int], int]) -> dict[int, int]:
    """Cuenta, por pid, las lecturas de /proc/<pid>/environ del barrido. Las
    `vacias(pid)` primeras salen vacías: lo que lee mientras el exec del hijo
    aún no publicó su entorno. Con vfork, Popen vuelve con el hijo DENTRO de su
    exec y esa ventana dura de µs a ms (medido: 178 de 300 lecturas vacías
    justo tras volver Popen entero), así que el barrido real cae en ella a veces;
    aquí, siempre."""
    lecturas: dict[int, int] = {}

    def open_(path: Any, *args: Any, **kwargs: Any) -> Any:
        partes = str(path).split("/")
        if len(partes) == 4 and partes[1] == "proc" and partes[3] == "environ":
            pid = int(partes[2])
            lecturas[pid] = lecturas.get(pid, 0) + 1
            if lecturas[pid] <= vacias(pid):
                return io.BytesIO(b"")
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(bridge, "open", open_, raising=False)
    return lecturas


def test_barrido_alcanza_al_server_cuyo_exec_aun_no_publico_su_entorno(monkeypatch):
    """Revisión: el barrido leía el environ del hijo UNA vez y, vacío (exec en
    curso), lo daba por ajeno: el server cortado en pleno Popen quedaba
    huérfano (1-3 de cada 200 apagados en el instante de la señal; el test de
    Ctrl+C en pleno Popen fallaba a veces bajo carga). Se relee, acotado."""
    marca = {bridge._MARCA_ENV: f"{os.getpid()}.exec-en-curso"}
    server = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True, env={**os.environ, **marca},
    )
    lecturas = _lecturas_de_environ(monkeypatch, lambda pid: 1 if pid == server.pid else 0)
    try:
        bridge._kill_unspawned([types.SimpleNamespace(env=marca, name="x")])  # type: ignore[list-item]
        assert esperar(lambda: not vivo(server.pid), plazo=3.0), "quedó huérfano"
        # La vacía y, tras releer (más de una vez si el exec real aún sigue), la buena.
        assert lecturas[server.pid] >= 2
    finally:
        _recoger(server)


def test_terminate_all_al_volver_el_fork_alcanza_al_server_aun_en_su_exec(tmp_path, monkeypatch):
    """Lo mismo en el instante exacto de la señal del apagado: al volver
    _fork_exec (`self.pid = _fork_exec(...)`), con el hijo el primero en /proc
    (lo que más deprisa lo lee el barrido) y su environ aún vacío. Y la espera
    no mata a quien no es nuestro: una app de run_detached lanzada con el
    entorno vacío lo lee vacío siempre y sigue viva."""
    app = _hijo_que_no_es_server({})
    servers, _pidfiles = _mudos(tmp_path, 2)
    real_fork_exec = subprocess._fork_exec  # type: ignore[attr-defined]
    real_listdir = os.listdir
    hijos: list[int] = []

    def fork_exec(*args: Any, **kwargs: Any) -> int:
        pid = real_fork_exec(*args, **kwargs)
        hijos.append(pid)
        if len(hijos) == 2:
            bridge.terminate_all()
        return pid

    def listdir(path: Any = ".") -> list[str]:
        entradas = real_listdir(path)
        if path == "/proc" and hijos:
            primero = str(hijos[-1])
            entradas = [primero] + [e for e in entradas if e != primero]
        return entradas

    monkeypatch.setattr(subprocess, "_fork_exec", fork_exec)
    monkeypatch.setattr(os, "listdir", listdir)
    lecturas = _lecturas_de_environ(monkeypatch, lambda pid: 1 if pid in hijos[1:2] else 0)
    try:
        names = bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=6))
        assert names == [] and len(hijos) == 2
        assert esperar(lambda: not any(vivo(p) for p in hijos), plazo=3.0), "quedó huérfano"
        assert lecturas[hijos[1]] >= 2  # releyó tras la vacía
        assert vivo(app.pid) and lecturas[app.pid] > 1  # esperó acotado y no la mató
        assert bridge._STARTING == set() and bridge._SPAWNING == set()
    finally:
        for pid in hijos:
            if vivo(pid):
                os.kill(pid, signal.SIGKILL)
        _recoger(app)


def test_barrido_no_espera_a_un_hijo_zombi(monkeypatch):
    """Un zombi lee su environ vacío para siempre: no es un exec en curso y el
    barrido no lo relee (ni lo lee), así un apagado no paga la espera por él."""
    marca = {bridge._MARCA_ENV: f"{os.getpid()}.zombi"}
    zombi = subprocess.Popen(
        [sys.executable, "-c", "pass"], start_new_session=True, env={**os.environ, **marca},
    )

    def estado() -> str:
        with open(f"/proc/{zombi.pid}/stat", encoding="ascii") as f:
            return f.read().rsplit(")", 1)[1].split()[0]

    try:
        assert esperar(lambda: estado() == "Z")
        lecturas = _lecturas_de_environ(monkeypatch, lambda pid: 0)
        bridge._kill_unspawned([types.SimpleNamespace(env=marca, name="x")])  # type: ignore[list-item]
        assert lecturas.get(zombi.pid, 0) == 0
    finally:
        _recoger(zombi)


def _start_de_hilos_de_arranque(monkeypatch, cual: int, error: BaseException,
                                tras_arrancar: bool) -> None:
    """Thread.start() del hilo de handshake número `cual` lanza `error`: antes
    de arrancarlo ("can't start new thread") o ya arrancado (Ctrl+C mientras
    start() espera a que el hilo diga que corre)."""
    real_start = threading.Thread.start
    vistos: list[str] = []

    def start(hilo: threading.Thread) -> None:
        if not hilo.name.startswith("mcp-start"):
            return real_start(hilo)
        vistos.append(hilo.name)
        if len(vistos) == cual and not tras_arrancar:
            raise error
        real_start(hilo)
        if len(vistos) == cual:
            raise error

    monkeypatch.setattr(threading.Thread, "start", start)


def test_sin_hilo_para_un_handshake_ese_server_se_conecta_en_serie(monkeypatch):
    """Sin hilo para el handshake de un server, ese server lo hace en el hilo
    del llamador, como en serie: no se queda lanzado sin dueño, no tumba el
    registro de los demás y no se relanza (un proceso por server)."""
    _start_de_hilos_de_arranque(monkeypatch, 2, RuntimeError("can't start new thread"), False)
    contador = Contador(monkeypatch)
    servers = {n: {"command": FAKE_CMD} for n in ("a", "b", "c")}
    names = bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert bridge.connected_servers() == ["a", "b", "c"]
    assert {n.split("_")[1] for n in names} == {"a", "b", "c"}
    assert sorted(contador.starts) == ["a", "b", "c"]
    assert contador.hilos["b"] == threading.current_thread().name
    assert contador.hilos["a"].startswith("mcp-start")
    assert contador.hilos["c"].startswith("mcp-start")
    assert bridge._STARTING == set() and hilos_de_arranque() == []


def test_ctrl_c_al_arrancar_el_hilo_de_un_handshake_cierra_su_server_y_espera_su_hilo(
    monkeypatch,
):
    """Ctrl+C dentro de Thread.start() del hilo de un handshake: el hilo ya
    corre y su server ya está lanzado. Los dos tienen dueño: el server se cierra
    y el hilo se espera, como los demás."""
    _start_de_hilos_de_arranque(monkeypatch, 2, KeyboardInterrupt(), True)
    contador = Contador(monkeypatch)
    servers = {n: {"command": FAKE_CMD} for n in ("a", "b", "c")}
    with pytest.raises(KeyboardInterrupt):
        bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert sorted(contador.starts) == ["a", "b"]
    for name in ("a", "b"):
        proc = contador.clientes[name]._proc
        assert proc is not None and proc.poll() is not None, f"'{name}' siguió vivo"
    assert bridge._STARTING == set() and bridge.connected_servers() == []
    assert hilos_de_arranque() == []


def _sin_hilos_de_cierre(monkeypatch, desde: int) -> list[str]:
    """Thread.start() de los hilos que cierran lo no registrado lanza "can't
    start new thread" a partir del número `desde`."""
    real_start = threading.Thread.start
    vistos: list[str] = []

    def start(hilo: threading.Thread) -> None:
        if hilo.name.endswith("-cierre"):
            vistos.append(hilo.name)
            if len(vistos) >= desde:
                raise RuntimeError("can't start new thread")
        return real_start(hilo)

    monkeypatch.setattr(threading.Thread, "start", start)
    return vistos


def test_ctrl_c_sin_hilo_para_cerrar_cierra_en_serie_y_el_ctrl_c_se_propaga(tmp_path, monkeypatch):
    """Revisión: sin hilo para cerrar a los que no se registraron, el Ctrl+C
    se volvía RuntimeError (el shell lo tragaba con "no pude registrar las
    tools MCP" y seguía arrancando), dos servers quedaban vivos y sus hilos de
    handshake sin esperar. Como con el hilo de un handshake, el que no tiene
    hilo se cierra aquí, en serie."""
    servers, pidfiles = _mudos(tmp_path, 3)
    vistos = _sin_hilos_de_cierre(monkeypatch, desde=2)

    def manejador(_signum, _frame) -> None:
        raise KeyboardInterrupt

    previo = signal.signal(signal.SIGUSR2, manejador)
    visto: list[bool] = []
    try:
        _senal_cuando_arranquen(pidfiles, visto)
        with pytest.raises(KeyboardInterrupt):
            bridge.register_mcp_tools(Registry(), mk_settings(servers, startup_timeout_s=15))
        assert visto == [True]
        assert len(vistos) == 3  # uno con hilo, dos sin él
        pids = _pids(pidfiles)
        assert len(pids) == 3
        assert not any(vivo(p) for p in pids), "quedó un server vivo tras el Ctrl+C"
        assert bridge._STARTING == set() and bridge.connected_servers() == []
        assert hilos_de_arranque() == []
    finally:
        signal.signal(signal.SIGUSR2, previo)
        _matar_restos(pidfiles)


def test_sin_hilo_para_cerrar_un_sobrante_el_registro_vuelve_normal(monkeypatch, caplog):
    """El mismo cierre en el camino normal (el server que otro hilo conectó
    entretanto): sin hilo, register_mcp_tools no lanza ni avisa; devuelve lo
    registrado y el sobrante queda cerrado."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    otro = Registry()

    def antes(client: StdioMCPClient) -> None:
        if client.name == "a":
            bridge.register_mcp_tools(otro, mk_settings({"b": {"command": FAKE_CMD}}))

    contador = Contador(monkeypatch, antes_de_initialize=antes)
    vistos = _sin_hilos_de_cierre(monkeypatch, desde=1)
    registry = Registry()
    names = bridge.register_mcp_tools(
        registry, mk_settings({"a": {"command": FAKE_CMD}, "b": {"command": FAKE_CMD}}),
    )
    assert vistos == ["mcp-b-cierre"]
    assert "mcp_a_echo" in names and "mcp_b_echo" in names
    assert bridge.connected_servers() == ["b", "a"]
    conectado = bridge._STATE["b"].client
    sobrante = next(c for c in contador.todos if c.name == "b" and c is not conectado)
    assert sobrante._proc is not None and sobrante._proc.poll() is not None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert bridge._STARTING == set() and hilos_de_arranque() == []
