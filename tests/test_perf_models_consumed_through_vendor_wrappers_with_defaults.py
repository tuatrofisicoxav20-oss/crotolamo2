"""Modelos de voz cargados por crotolamo/voice/runtime.py, no por los wrappers del vendor.

Causa raíz (medida con los cargadores reales, proceso nuevo por escenario): cada
modelo entraba por el paquete de conveniencia de su vendor, con sus imports y sus
defaults pensados para scripts cortos o para entrenar:

- Silero: `silero_vad` importa torch (+218 MB, ~950 módulos, en el hilo del oído)
  solo como contenedor de tensores; el mismo ONNX con numpy + onnxruntime da las
  MISMAS probabilidades, bit a bit.
- openWakeWord: `import openwakeword` arrastra sklearn y scipy (+87 MB, +3 hilos)
  por un módulo de entrenamiento, y el bucle de firmas construía 9 sesiones ONNX
  para quedarse con 3.
- Whisper: cada arranque consultaba el Hub (1 conexión por modelo) aunque el modelo
  ya estuviera en la caché, y tqdm dejaba un hilo monitor vivo para siempre.
- Piper: el arena de ONNX Runtime retenía, de por vida del servicio, el pico de la
  frase más larga dicha (+409 MB tras una de 23 s).
- El doctor importaba torch, sklearn y scipy (~280 MB) para escribir "importable".

Lo que fijan estos tests: mismas salidas que el camino del vendor (probabilidades,
scores), menos recursos (módulos pesados, sesiones construidas) y el respaldo
automático al camino de siempre cuando la versión instalada no encaja. Los que
necesitan los paquetes reales se saltan si no están; el resto usa falsos.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

from crotolamo.voice import runtime
from crotolamo.voice import stt as stt_mod

REPO = Path(__file__).resolve().parents[1]


def _run_py(code: str, *extra_path: str, timeout: float = 180.0) -> dict:
    """Corre `code` en un intérprete NUEVO (sys.modules limpio) y devuelve el
    JSON que imprime en su última línea."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*extra_path, str(REPO)])
    env["CROTOLAMO_LOG"] = "WARNING"
    env["PYTHONWARNINGS"] = "ignore"
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# =============================================================================
# Paso 1: crotolamo/voice/runtime.py
# =============================================================================


def test_importar_runtime_no_importa_nada_pesado():
    """Lo importan stt, adapters y wakeword: debe costar lo que la stdlib."""
    out = _run_py("""
        import json, sys
        import crotolamo.voice.runtime
        pesados = ("numpy", "onnxruntime", "torch", "sklearn", "scipy", "silero_vad",
                   "openwakeword", "faster_whisper", "piper")
        print(json.dumps([m for m in pesados if m in sys.modules]))
    """)
    assert out == []


def _paquete_espia(base: Path, nombre: str, datos: dict[str, bytes]) -> Path:
    """Paquete en disco cuyo __init__ REVIENTA: si alguien lo ejecuta, se nota."""
    pkg = base / nombre
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('se ejecutó el __init__')\n")
    for rel, contenido in datos.items():
        destino = pkg / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
    return pkg


def test_module_available_no_importa_y_respeta_los_idiomas_de_los_tests(
        monkeypatch, tmp_path):
    _paquete_espia(tmp_path, "paquete_espia_runtime", {})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert runtime.module_available("paquete_espia_runtime") is True
    assert "paquete_espia_runtime" not in sys.modules
    assert runtime.module_available("no_existe_este_modulo_xyz") is False
    monkeypatch.setitem(sys.modules, "falta_a_proposito", None)
    assert runtime.module_available("falta_a_proposito") is False
    monkeypatch.setitem(sys.modules, "falso_sin_spec", types.ModuleType("falso_sin_spec"))
    assert runtime.module_available("falso_sin_spec") is True


def test_package_file_localiza_datos_sin_ejecutar_el_init(monkeypatch, tmp_path):
    pkg = _paquete_espia(tmp_path, "paquete_datos_runtime",
                         {"data/modelo.onnx": b"onnx", "resources/models/a.onnx": b"x"})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert runtime.package_file("paquete_datos_runtime", "data", "modelo.onnx") == str(
        pkg / "data" / "modelo.onnx")
    # También carpetas (openWakeWord busca en resources/models).
    assert runtime.package_file("paquete_datos_runtime", "resources", "models") == str(
        pkg / "resources" / "models")
    assert runtime.package_file("paquete_datos_runtime", "data", "otro.onnx") is None
    assert "paquete_datos_runtime" not in sys.modules
    assert runtime.package_file("no_existe_este_paquete_xyz", "data") is None
    # sys.modules manda: None = falta; un falso con __file__ se resuelve por él.
    monkeypatch.setitem(sys.modules, "paquete_datos_runtime", None)
    assert runtime.package_file("paquete_datos_runtime", "data", "modelo.onnx") is None
    falso = types.ModuleType("paquete_datos_runtime")
    falso.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "paquete_datos_runtime", falso)
    assert runtime.package_file("paquete_datos_runtime", "data", "modelo.onnx") == str(
        pkg / "data" / "modelo.onnx")
    monkeypatch.setitem(sys.modules, "paquete_datos_runtime",
                        types.ModuleType("paquete_datos_runtime"))
    assert runtime.package_file("paquete_datos_runtime", "data", "modelo.onnx") is None


class _SesionFalsa:
    """InferenceSession falsa: anota con qué run_options se llamó cada run."""

    def __init__(self) -> None:
        self.llamadas: list = []
        self.algo = "delegado"

    def run(self, output_names, input_feed, run_options=None):
        self.llamadas.append(run_options)
        return ["salida"]

    def get_inputs(self):
        return ["entrada"]


def _encoge(run_options) -> bool:
    return (run_options is not None
            and run_options.get_run_config_entry(runtime._SHRINK_KEY) == "cpu:0")


def test_arena_shrinking_session_solo_encoge_los_runs_largos():
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    sesion = _SesionFalsa()
    proxy = runtime.ArenaShrinkingSession(sesion, keep_len=256)

    assert proxy.run(None, {"input": np.zeros((1, 256))}) == ["salida"]
    assert sesion.llamadas == [None], "un run corto no debe pagar el encogido"
    proxy.run(None, {"input": np.zeros((1, 257))})
    assert _encoge(sesion.llamadas[-1])
    mias = object()
    proxy.run(None, {"input": np.zeros((1, 999))}, mias)
    assert sesion.llamadas[-1] is mias, "las run_options del llamador no se pisan"
    proxy.run(None, {"otra": np.zeros(5)})
    assert sesion.llamadas[-1] is None, "sin la entrada principal: como hoy"
    # Todo lo demás se delega en la sesión real.
    assert proxy.algo == "delegado"
    assert proxy.get_inputs() == ["entrada"]

    siempre = runtime.ArenaShrinkingSession(_SesionFalsa(), keep_len=None)
    siempre.run(None, {"input": np.zeros((1, 3))})
    assert _encoge(siempre._session.llamadas[-1])


def test_wrap_piper_session_envuelve_y_tolera_voces_sin_sesion():
    pytest.importorskip("onnxruntime")
    voz = types.SimpleNamespace(session=_SesionFalsa(), config="cfg")
    assert runtime.wrap_piper_session(voz) is voz
    assert isinstance(voz.session, runtime.ArenaShrinkingSession)
    assert voz.session._keep_len == runtime.PIPER_KEEP_PHONEMES
    envuelta = voz.session
    runtime.wrap_piper_session(voz)
    assert voz.session is envuelta, "no se envuelve dos veces"
    sin_sesion = types.SimpleNamespace(config="cfg")
    assert runtime.wrap_piper_session(sin_sesion) is sin_sesion
    assert not hasattr(sin_sesion, "session")


_RECHAZO_ORT = ("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Did not find an arena based "
                "allocator registered for device-id combination in the memory arena shrink "
                "list: cpu:0")


class _SesionSinArena(_SesionFalsa):
    """Sesión SIN arena de CPU (un ORT con mimalloc, o un Piper que lo apague): ORT
    rechaza la lista de encogido antes de calcular, con este mismo error.
    `arena_declarada` es lo que dice get_session_options()."""

    def __init__(self, arena_declarada: bool = False) -> None:
        super().__init__()
        self.arena_declarada = arena_declarada

    def run(self, output_names, input_feed, run_options=None):
        self.llamadas.append(run_options)
        if run_options is not None:
            from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

            raise InvalidArgument(_RECHAZO_ORT)
        return ["salida"]

    def get_session_options(self):
        return types.SimpleNamespace(enable_cpu_mem_arena=self.arena_declarada)


def test_arena_shrinking_session_si_ort_rechaza_encoger_repite_el_run_de_siempre(
        monkeypatch, caplog):
    """Sin arena, ORT rechaza el encogido: la frase larga debe sonar igual (el run
    de siempre, sin run_options), dejar de intentarlo y avisar UNA vez."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(runtime, "_warned", set())
    sesion = _SesionSinArena(arena_declarada=True)
    proxy = runtime.ArenaShrinkingSession(sesion, keep_len=256)

    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        assert proxy.run(None, {"input": np.zeros((1, 256))}) == ["salida"]
        assert sesion.llamadas == [None]
        assert proxy.run(None, {"input": np.zeros((1, 300))}) == ["salida"]
        assert _encoge(sesion.llamadas[1]) and sesion.llamadas[2:] == [None], sesion.llamadas
        # Las siguientes largas van directas al run de siempre: ni un rechazo más.
        assert proxy.run(None, {"input": np.zeros((1, 800))}) == ["salida"]
        assert sesion.llamadas[3:] == [None]
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and "arena" in avisos[0] and _RECHAZO_ORT in avisos[0], avisos
    # Unas run_options del llamador siguen yendo tal cual, y su error sube tal cual.
    mias = object()
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    with pytest.raises(InvalidArgument):
        proxy.run(None, {"input": np.zeros((1, 999))}, mias)
    assert sesion.llamadas[-1] is mias and len(sesion.llamadas) == 5


def test_arena_shrinking_session_si_tambien_falla_sin_encoger_sube_el_error_de_siempre(
        monkeypatch, caplog):
    """Si el run de siempre TAMBIÉN falla, sube SU error (el mismo que daría la voz de
    fábrica), sin encadenar el rechazo, y el encogido no se apaga por eso."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(runtime, "_warned", set())

    class _Rota(_SesionFalsa):
        def run(self, output_names, input_feed, run_options=None):
            self.llamadas.append(run_options)
            raise RuntimeError("con encogido" if run_options is not None else "de siempre")

    sesion = _Rota()
    proxy = runtime.ArenaShrinkingSession(sesion, keep_len=256)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        with pytest.raises(RuntimeError, match="^de siempre$") as error:
            proxy.run(None, {"input": np.zeros((1, 300))})
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert _encoge(sesion.llamadas[0]) and sesion.llamadas[1:] == [None]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    with pytest.raises(RuntimeError):
        proxy.run(None, {"input": np.zeros((1, 300))})
    assert _encoge(sesion.llamadas[2]), "un fallo ajeno al encogido no lo apaga"


def test_wrap_piper_session_no_envuelve_una_sesion_sin_arena():
    """Sin arena de CPU no hay pico que devolver (y ORT rechazaría el encogido): la
    sesión se queda tal cual, la de fábrica."""
    pytest.importorskip("onnxruntime")
    sin_arena = types.SimpleNamespace(session=_SesionSinArena(arena_declarada=False))
    cruda = sin_arena.session
    assert runtime.wrap_piper_session(sin_arena) is sin_arena
    assert sin_arena.session is cruda
    con_arena = types.SimpleNamespace(session=_SesionSinArena(arena_declarada=True))
    runtime.wrap_piper_session(con_arena)
    assert isinstance(con_arena.session, runtime.ArenaShrinkingSession)

    class _NoLoDice(_SesionFalsa):
        def get_session_options(self):
            raise RuntimeError("API distinta")

    # Si la sesión no sabe decirlo, se envuelve: run() cae solo si ORT rechaza.
    muda = types.SimpleNamespace(session=_NoLoDice())
    runtime.wrap_piper_session(muda)
    assert isinstance(muda.session, runtime.ArenaShrinkingSession)


def test_load_whisper_prueba_la_cache_local_y_cae_a_la_llamada_de_siempre(monkeypatch):
    llamadas: list = []

    class _Whisper:
        def __init__(self, size, **kwargs) -> None:
            llamadas.append((size, kwargs))
            if kwargs.get("local_files_only"):
                raise RuntimeError("LocalEntryNotFoundError: no está en la caché")

    fw = types.SimpleNamespace(WhisperModel=_Whisper)
    tqdm_falso = types.ModuleType("tqdm")
    tqdm_falso.tqdm = type("tqdm", (), {"monitor_interval": 10})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_falso)

    modelo = runtime.load_whisper(fw, "base")
    assert isinstance(modelo, _Whisper)
    assert llamadas == [
        ("base", {"device": "cpu", "compute_type": "int8", "local_files_only": True}),
        # EXACTAMENTE la llamada de hoy: la primera vez descarga.
        ("base", {"device": "cpu", "compute_type": "int8"}),
    ]
    assert tqdm_falso.tqdm.monitor_interval == 0, "el monitor de tqdm sigue activo"

    class _Cacheado(_Whisper):
        def __init__(self, size, **kwargs) -> None:
            llamadas.append((size, kwargs))

    llamadas.clear()
    runtime.load_whisper(types.SimpleNamespace(WhisperModel=_Cacheado), "tiny", cpu_threads=2)
    assert llamadas == [("tiny", {"device": "cpu", "compute_type": "int8",
                                  "local_files_only": True, "cpu_threads": 2})]


# --- Silero: SileroOnnx frente al wrapper del vendor ------------------------------


def _vocal(np, segundos: float, f0: float, formantes, amp: float = 0.3):
    """Vocal sintética determinista (tren de pulsos + resonadores): Silero la
    puntúa como voz (>0.9), así que la prueba cubre las dos decisiones."""
    n = int(16000 * segundos)
    t = np.arange(n) / 16000
    fase = 2 * np.pi * np.cumsum(f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / 16000
    pulsos = (np.diff(np.floor(fase / (2 * np.pi)), prepend=0) > 0).astype(np.float64)
    ir_t = np.arange(int(0.03 * 16000)) / 16000
    ir = sum(np.exp(-np.pi * bw * ir_t) * np.sin(2 * np.pi * f * ir_t) for f, bw in formantes)
    y = np.convolve(pulsos, ir)[:n]
    env = np.clip(np.sin(np.pi * np.arange(n) / n) * 1.5, 0, 1)
    return (y / (np.abs(y).max() + 1e-9) * amp * env).astype(np.float32)


def _enunciados(np) -> list:
    """Tres 'enunciados' (voz, silencio digital, ruido, tono) en chunks de 512."""
    rng = np.random.default_rng(3)

    def ruido(s, a):
        return (rng.standard_normal(int(16000 * s)) * a).astype(np.float32)

    t = np.arange(8000) / 16000
    tono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    clips = [
        np.concatenate([ruido(0.5, 0.003), _vocal(np, 0.6, 120, ((700, 80), (1220, 90))),
                        _vocal(np, 0.4, 150, ((300, 60), (2300, 100))), ruido(0.5, 0.003)]),
        np.concatenate([np.zeros(8000, np.float32), ruido(0.6, 0.05), tono,
                        _vocal(np, 0.8, 200, ((500, 70), (900, 80), (2500, 110)))]),
        np.concatenate([ruido(0.3, 0.2), _vocal(np, 0.5, 110, ((650, 80), (1100, 90))),
                        np.zeros(4000, np.float32)]),
    ]
    return [[c[i:i + 512] for i in range(0, c.size - 511, 512)] for c in clips]


@pytest.fixture
def silero_real():
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    silero_vad = pytest.importorskip("silero_vad")
    path = runtime.package_file("silero_vad", "data", "silero_vad.onnx")
    if path is None:
        pytest.skip("este silero_vad no trae data/silero_vad.onnx")
    return np, torch, silero_vad, path


def _probs_vendor(silero_vad, torch, enunciados) -> list[float]:
    viejo = silero_vad.load_silero_vad(onnx=True)
    probs: list[float] = []
    for chunks in enunciados:
        viejo.reset_states()  # como _SileroVad.reset() entre turnos
        probs += [float(viejo(torch.from_numpy(c.copy()), 16000)) for c in chunks]
    return probs


def test_silero_onnx_da_exactamente_las_probabilidades_del_vendor(silero_real):
    np, torch, silero_vad, path = silero_real
    enunciados = _enunciados(np)
    esperado = _probs_vendor(silero_vad, torch, enunciados)

    nuevo = runtime.SileroOnnx(path)
    obtenido: list[float] = []
    for chunks in enunciados:
        nuevo.reset_states()
        obtenido += [nuevo(c, 16000) for c in chunks]

    assert len(obtenido) == len(esperado) > 100
    assert max(abs(a - b) for a, b in zip(obtenido, esperado)) == 0.0
    # Las tramas recorren las dos decisiones (si no, la igualdad no probaría nada).
    assert min(esperado) < 0.05 and max(esperado) > 0.9
    # Un reset_states() devuelve el estado a cero: repetir da lo mismo.
    nuevo.reset_states()
    assert [nuevo(c, 16000) for c in enunciados[0]] == obtenido[:len(enunciados[0])]


def test_silero_onnx_valida_la_entrada_como_el_vendor(silero_real):
    np, _torch, _silero_vad, path = silero_real
    modelo = runtime.SileroOnnx(path)
    with pytest.raises(ValueError, match="number of samples"):
        modelo(np.zeros(1024, np.float32), 16000)
    with pytest.raises(ValueError, match="too short"):
        modelo(np.zeros(100, np.float32), 16000)
    with pytest.raises(ValueError, match="sampling rates"):
        modelo(np.zeros(512, np.float32), 22050)
    # Lista de Python (np.asarray) y 8 kHz con 256 muestras: aceptados.
    assert 0.0 <= modelo([0.0] * 512, 16000) <= 1.0
    assert 0.0 <= modelo(np.zeros(256), 8000) <= 1.0


@pytest.fixture
def silero_vendor_falso(monkeypatch):
    """silero_vad falso (sin data/) cuyo OnnxWrapper anota lo que recibe, y torch
    falso con from_numpy; memoria de avisos limpia."""
    pytest.importorskip("numpy")
    estado: dict = {"cargas": 0, "recibido": [], "resets": 0}

    class _OnnxWrapper:
        def reset_states(self) -> None:
            estado["resets"] += 1

        def __call__(self, x, sr):
            estado["recibido"].append((x, sr))
            return 0.75  # el tensor (1, 1) real: float() lo acepta

    def _load(onnx=False):
        assert onnx is True
        estado["cargas"] += 1
        return _OnnxWrapper()

    falso = types.ModuleType("silero_vad")
    falso.load_silero_vad = _load  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda a: ("tensor", a)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", falso)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runtime, "_warned", set())
    return estado


def test_load_silero_sin_el_onnx_empaquetado_usa_el_cargador_de_siempre(
        silero_vendor_falso, caplog):
    np = pytest.importorskip("numpy")
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        modelo = runtime.load_silero()
        runtime.load_silero()
    assert silero_vendor_falso["cargas"] == 2
    assert modelo(np.zeros(512, dtype=np.float64), 16000) == 0.75
    tensor, sr = silero_vendor_falso["recibido"][-1]
    assert tensor[0] == "tensor" and sr == 16000
    assert tensor[1].dtype == np.float32 and tensor[1].flags["C_CONTIGUOUS"]
    modelo.reset_states()
    assert silero_vendor_falso["resets"] == 1
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and "silero" in avisos[0].lower(), avisos


def test_load_silero_cae_al_vendor_si_el_onnx_no_abre(silero_vendor_falso, monkeypatch,
                                                      tmp_path):
    onnx = tmp_path / "silero_vad.onnx"
    onnx.write_bytes(b"no es un onnx")
    monkeypatch.setattr(runtime, "package_file", lambda *a: str(onnx))

    def _cambio_de_api(path):
        raise RuntimeError("Invalid input name: state")

    monkeypatch.setattr(runtime, "SileroOnnx", _cambio_de_api)
    assert isinstance(runtime.load_silero(), runtime._VendorSileroAdapter)
    assert silero_vendor_falso["cargas"] == 1


def test_load_silero_cae_al_vendor_si_el_onnx_trae_otras_entradas(silero_vendor_falso,
                                                                   monkeypatch, tmp_path, caplog):
    """Un silero_vad futuro con otro ONNX (p. ej. las entradas h/c de la v4): se
    detecta AL CARGAR y va el cargador de siempre; si no, el primer chunk
    reventaría y el VAD quedaría en energía para todo el proceso."""
    pytest.importorskip("numpy")
    onnx = tmp_path / "silero_vad.onnx"
    onnx.write_bytes(b"onnx")
    monkeypatch.setattr(runtime, "package_file", lambda *a: str(onnx))
    entradas = [types.SimpleNamespace(name=n) for n in ("input", "sr", "h", "c")]
    monkeypatch.setattr(runtime, "ort_session", lambda path, threads=1: types.SimpleNamespace(
        get_inputs=lambda: entradas))
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        modelo = runtime.load_silero()
    assert isinstance(modelo, runtime._VendorSileroAdapter)
    assert silero_vendor_falso["cargas"] == 1
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and "entradas inesperadas" in avisos[0], avisos


def test_load_silero_sin_onnxruntime_no_cae_al_vendor(silero_vendor_falso, monkeypatch,
                                                      tmp_path):
    onnx = tmp_path / "silero_vad.onnx"
    onnx.write_bytes(b"onnx")
    monkeypatch.setattr(runtime, "package_file", lambda *a: str(onnx))

    def _sin_ort(path):
        raise ModuleNotFoundError("No module named 'onnxruntime'")

    monkeypatch.setattr(runtime, "SileroOnnx", _sin_ort)
    with pytest.raises(ImportError, match="onnxruntime"):
        runtime.load_silero()
    assert silero_vendor_falso["cargas"] == 0, "el vendor también necesita onnxruntime"


def test_load_silero_sin_silero_vad_da_el_import_error_de_siempre(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    monkeypatch.setattr(runtime, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        with pytest.raises(ImportError):
            runtime.load_silero()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- openWakeWord: import sin el verificador, kwargs por firma, arena -------------


def _oww_falso_en_disco(base: Path, init_extra: str = "") -> None:
    """Paquete 'openwakeword' de mentira con el MISMO __init__ que 0.4/0.6."""
    pkg = base / "openwakeword"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from openwakeword.model import Model\n"
        f"from openwakeword.custom_verifier_model import train_custom_verifier{init_extra}\n")
    (pkg / "model.py").write_text("class Model:\n    pass\n")
    (pkg / "custom_verifier_model.py").write_text(
        "import os\n"
        "os.environ['CROTO_VERIFICADOR_EJECUTADO'] = '1'  # aquí iría sklearn + scipy\n"
        "OTRA = 'otra'\n"
        "def train_custom_verifier(*args, **kwargs):\n"
        "    return ['real', list(args), kwargs]\n")


def test_import_openwakeword_no_ejecuta_el_verificador_hasta_que_se_entrena(tmp_path):
    _oww_falso_en_disco(tmp_path)
    out = _run_py("""
        import json, os
        from crotolamo.voice import runtime
        oww = runtime.import_openwakeword()
        antes = os.environ.get("CROTO_VERIFICADOR_EJECUTADO")
        primera = oww.train_custom_verifier(1, x=2)
        despues = os.environ.get("CROTO_VERIFICADOR_EJECUTADO")
        segunda = oww.train_custom_verifier(3)
        print(json.dumps([antes, primera, despues, segunda]))
    """, str(tmp_path))
    assert out == [None, ["real", [1], {"x": 2}], "1", ["real", [3], {}]]


def test_import_openwakeword_reintenta_sin_el_stub_si_no_encaja(tmp_path):
    """Una versión futura que importe OTRA cosa del verificador: sin stub, como hoy."""
    _oww_falso_en_disco(tmp_path, init_extra=", OTRA")
    out = _run_py("""
        import json, os
        from crotolamo.voice import runtime
        oww = runtime.import_openwakeword()
        print(json.dumps([os.environ.get("CROTO_VERIFICADOR_EJECUTADO"),
                          oww.OTRA, oww.train_custom_verifier(5)]))
    """, str(tmp_path))
    assert out == ["1", "otra", ["real", [5], {}]]


def test_import_openwakeword_respeta_lo_que_ya_hay_en_sys_modules(monkeypatch):
    falso = types.ModuleType("openwakeword")
    monkeypatch.setitem(sys.modules, "openwakeword", falso)
    monkeypatch.delitem(sys.modules, runtime._OWW_VERIFIER, raising=False)
    assert runtime.import_openwakeword() is falso
    assert runtime._OWW_VERIFIER not in sys.modules, "con el paquete ya cargado no hay stub"
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    with pytest.raises(ImportError):
        runtime.import_openwakeword()


class _Model04:
    """Firma de openwakeword 0.4.0: rutas explícitas y **kwargs hacia AudioFeatures."""

    def __init__(self, wakeword_model_paths=(), class_mapping_dicts=(), **kwargs) -> None:
        pass


class _Model06:
    """Firma de openwakeword 0.5/0.6: nombres y framework explícitos."""

    def __init__(self, wakeword_models=(), inference_framework="tflite", **kwargs) -> None:
        pass


class _ModelSoloKwargs:
    def __init__(self, **kwargs) -> None:
        pass


def test_oww_constructor_kwargs_prueba_primero_la_firma_instalada():
    legado = [
        {"wakeword_models": ["hey_jarvis"], "inference_framework": "onnx"},
        {"wakeword_model_paths": ["/m/hey_jarvis_v0.1.onnx"], "inference_framework": "onnx"},
        {"wakeword_model_paths": ["/m/hey_jarvis_v0.1.onnx"]},
        {"wakeword_models": ["/m/hey_jarvis_v0.1.onnx"]},
    ]
    k04 = runtime.oww_constructor_kwargs(_Model04, "hey_jarvis", "/m/hey_jarvis_v0.1.onnx")
    assert k04[0] == {"wakeword_model_paths": ["/m/hey_jarvis_v0.1.onnx"]}
    assert k04[1:] == [legado[0], legado[1], legado[3]], "el respaldo no repite la elegida"
    k06 = runtime.oww_constructor_kwargs(_Model06, "hey_jarvis", "/m/hey_jarvis_v0.1.onnx")
    assert k06 == legado
    assert runtime.oww_constructor_kwargs(
        _ModelSoloKwargs, "hey_jarvis", "/m/hey_jarvis_v0.1.onnx") == legado


class _SesionFeatures(_SesionFalsa):
    def __init__(self, nombre: str, shape: list, revienta: bool = False) -> None:
        super().__init__()
        self._entrada = types.SimpleNamespace(name=nombre, shape=shape)
        self.revienta = revienta
        self.feeds: list = []

    def get_inputs(self):
        return [self._entrada]

    def run(self, output_names, input_feed, run_options=None):
        self.feeds.append({k: v.shape for k, v in input_feed.items()})
        if self.revienta:
            raise RuntimeError("sesión rota")
        return super().run(output_names, input_feed, run_options)


def test_release_oww_warmup_arena_un_run_minimo_con_encogido_por_sesion():
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    mel = _SesionFeatures("input", ["batch_size", "samples"])
    emb = _SesionFeatures("input_1", ["unk__316", 76, 32, 1])
    modelo = types.SimpleNamespace(
        preprocessor=types.SimpleNamespace(melspec_model=mel, embedding_model=emb))
    runtime.release_oww_warmup_arena(modelo)
    assert mel.feeds == [{"input": (1, 1280)}]
    assert emb.feeds == [{"input_1": (1, 76, 32, 1)}]
    assert _encoge(mel.llamadas[0]) and _encoge(emb.llamadas[0])

    # Nunca lanza: sesión rota, preprocessor sin sesiones, modelo sin preprocessor.
    rota = types.SimpleNamespace(preprocessor=types.SimpleNamespace(
        melspec_model=_SesionFeatures("input", [1, 1], revienta=True), embedding_model=None))
    runtime.release_oww_warmup_arena(rota)
    runtime.release_oww_warmup_arena(types.SimpleNamespace(preprocessor=object()))
    runtime.release_oww_warmup_arena(object())


# =============================================================================
# Paso 2: Silero por el runtime (stt._get_silero_vad, _record_silero, _SileroVad)
# =============================================================================

_SIN_TORCH = """
    import json, sys, types
    import numpy as np

    class _Stream:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def read(self, n):
            return np.zeros((n, 1), dtype=np.float32), False

    sys.modules["sounddevice"] = types.SimpleNamespace(InputStream=_Stream)
    from crotolamo.voice import adapters, stt

    vad = adapters._SileroVad()
    probs = [vad(np.zeros(512, dtype=np.float32)) for _ in range(3)]
    vad.reset()
    frames = stt.STT(model_size="x")._record_silero(600, 0.5, 0.3, {})
    print(json.dumps({
        "fallback": vad._fallback,
        "modelo": type(stt._get_silero_vad()).__name__,
        "misma_instancia": vad._model is stt._get_silero_vad(),
        "probs_en_rango": all(0.0 <= p < 0.5 for p in probs),
        "frames": len(frames),
        "pesados": [m for m in ("torch", "silero_vad", "sklearn", "scipy") if m in sys.modules],
    }))
"""


def test_el_vad_silero_ya_no_importa_torch():
    """En un proceso nuevo: oído concurrente + grabación del modo simple, sin torch
    (antes: +218 MB y ~950 módulos importados en el hilo del oído)."""
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    if runtime.package_file("silero_vad", "data", "silero_vad.onnx") is None:
        pytest.skip("silero_vad (con su data/silero_vad.onnx) no está instalado")
    out = _run_py(_SIN_TORCH)
    assert out == {"fallback": False, "modelo": "SileroOnnx", "misma_instancia": True,
                   "probs_en_rango": True, "frames": 0, "pesados": []}


def test_silero_vad_del_oido_da_las_mismas_probabilidades_que_antes(silero_real, monkeypatch):
    """adapters._SileroVad (numpy -> SileroOnnx) contra el wrapper torch de antes,
    con reset() entre enunciados como hace el oído entre turnos."""
    from crotolamo.voice.adapters import _SileroVad

    np, torch, silero_vad, _path = silero_real
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    enunciados = _enunciados(np)
    esperado = _probs_vendor(silero_vad, torch, enunciados)

    vad = _SileroVad()
    obtenido: list[float] = []
    for chunks in enunciados:
        vad.reset()
        obtenido += [vad(c) for c in chunks]
    assert vad._fallback is False
    assert isinstance(stt_mod._silero_vad, runtime.SileroOnnx)
    assert max(abs(a - b) for a, b in zip(obtenido, esperado)) == 0.0


def test_sin_el_onnx_empaquetado_el_oido_usa_el_cargador_de_siempre(silero_real,
                                                                    monkeypatch, caplog):
    """Respaldo real: si el archivo no se encuentra, load_silero_vad(onnx=True) y
    torch, con las mismas probabilidades y UN aviso."""
    from crotolamo.voice.adapters import _SileroVad

    np, torch, silero_vad, _path = silero_real
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    monkeypatch.setattr(runtime, "_warned", set())
    monkeypatch.setattr(runtime, "package_file", lambda *a: None)
    enunciados = _enunciados(np)[:1]
    esperado = _probs_vendor(silero_vad, torch, enunciados)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.runtime"):
        vad = _SileroVad()
        obtenido = [vad(c) for c in enunciados[0]]
    assert isinstance(stt_mod._silero_vad, runtime._VendorSileroAdapter)
    assert obtenido == esperado
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


class _ModeloAnotador:
    """Silero falso: anota el tipo de cada chunk y dice 'voz' en los primeros."""

    def __init__(self, voz: int = 3) -> None:
        self.recibido: list = []
        self.voz = voz
        self.resets = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, x, sr):
        self.recibido.append((type(x), str(getattr(x, "dtype", None)),
                              getattr(getattr(x, "flags", None), "c_contiguous", None), sr))
        return 0.9 if len(self.recibido) <= self.voz else 0.0


class _MicCeros:
    def __init__(self, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        import numpy as np

        return np.zeros((n, 1), dtype=np.float32), False


def _bloquear_torch(monkeypatch) -> None:
    real_import = builtins.__import__

    def _sin_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _sin_torch)
    monkeypatch.setitem(sys.modules, "torch", None)


def test_record_silero_ya_no_exige_torch(monkeypatch):
    """Antes `_require('torch')` lanzaba VoiceUnavailable sin torch: _record_frames
    la relanzaba, el loop simple salía con 1 y systemd lo reiniciaba cada 3 s."""
    np = pytest.importorskip("numpy")
    _bloquear_torch(monkeypatch)
    modelo = _ModeloAnotador(voz=3)
    monkeypatch.setattr(stt_mod, "_get_silero_vad", lambda: modelo)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_MicCeros))
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"vad_backend": "silero",
                                                        "vad_silence_ms": 64})
    frames = stt_mod.STT(model_size="x")._record_frames(None, 1.0, 0.5)
    assert len(frames) == 5, "3 chunks de voz + 2 de silencio (silence_ms=64)"
    assert modelo.resets == 1
    assert set(modelo.recibido) == {(np.ndarray, "float32", True, 16000)}


def test_silero_vad_del_oido_pasa_numpy_y_no_importa_torch(monkeypatch):
    np = pytest.importorskip("numpy")
    _bloquear_torch(monkeypatch)
    from crotolamo.voice.adapters import _SileroVad

    modelo = _ModeloAnotador(voz=1)
    monkeypatch.setattr(stt_mod, "_get_silero_vad", lambda: modelo)
    vad = _SileroVad()
    assert vad([0.1] * 512) == 0.9
    assert vad(np.zeros(512, dtype=np.float64)) == 0.0
    assert vad._fallback is False
    assert modelo.recibido == [(np.ndarray, "float32", True, 16000)] * 2


def test_get_silero_vad_carga_una_sola_vez_aunque_lo_pidan_dos_hilos(monkeypatch):
    """WarmVoice y el primer uso real pueden llegar a la vez: sin lock cada uno
    construía su propia sesión y una se tiraba."""
    cargas = {"n": 0}
    compuerta = threading.Event()

    def _lento():
        cargas["n"] += 1
        compuerta.wait(5)
        return object()

    monkeypatch.setattr(runtime, "load_silero", _lento)
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    vistos: list = []
    hilos = [threading.Thread(target=lambda: vistos.append(stt_mod._get_silero_vad()))
             for _ in range(3)]
    for h in hilos:
        h.start()
    time.sleep(0.2)
    compuerta.set()
    for h in hilos:
        h.join(5)
    assert cargas["n"] == 1
    assert len(vistos) == 3 and all(v is vistos[0] for v in vistos)


# =============================================================================
# Paso 3: openWakeWord por el runtime (available, _get_model, _resolve_model_path)
# =============================================================================

_OWW_REAL = """
    import json, os, sys, types
    import numpy as np
    import onnxruntime as ort

    creadas = []
    _init = ort.InferenceSession.__init__
    def _contar(self, path, *a, **k):
        creadas.append(os.path.basename(path) if isinstance(path, str) else "<bytes>")
        _init(self, path, *a, **k)
    ort.InferenceSession.__init__ = _contar
    # Sin PortAudio en CI: available() solo necesita que sounddevice importe.
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

    from crotolamo.voice.wakeword import WakeWordDetector

    det = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)
    disponible = det.available()
    tras_available = [m for m in ("sklearn", "scipy", "torch") if m in sys.modules]
    # openWakeWord 0.5+ ceba su búfer de features con np.random al construir el
    # Model (utils.AudioFeatures): sin fijar la semilla, dos Model de fábrica dan
    # puntuaciones distintas sobre el MISMO audio (medido: hasta 0.005 en 0.6). La
    # misma semilla antes de cada construcción hace la comparación exacta.
    np.random.seed(1234)
    det._get_model()
    sesiones = list(creadas)

    rng = np.random.default_rng(11)
    t = np.arange(16000 * 4) / 16000
    audio = (rng.standard_normal(t.size) * 800 + 6000 * np.sin(2 * np.pi * 300 * t)
             * (np.sin(2 * np.pi * 0.7 * t) > 0)).astype(np.int16)
    chunks = [audio[i:i + 512] for i in range(0, audio.size - 511, 512)]
    nuevos = [det.score(c) for c in chunks]

    # Referencia: el Model del vendor construido como lo hacía el bucle de antes,
    # con SUS mismos intentos y en el mismo orden (la firma cambia entre versiones:
    # 0.4 acepta wakeword_model_paths; 0.6 solo wakeword_models e inference_framework,
    # y sin él elige tflite). Fijar la firma de 0.4 rompía en 0.6 (CI en 3.11).
    from openwakeword.model import Model
    ruta = det._resolve_model_path()
    ref = None
    for kwargs in (
        {"wakeword_models": [det.model_name], "inference_framework": "onnx"},
        {"wakeword_model_paths": [ruta], "inference_framework": "onnx"},
        {"wakeword_model_paths": [ruta]},
        {"wakeword_models": [ruta]},
    ):
        try:
            np.random.seed(1234)
            ref = Model(**kwargs)
            break
        except Exception:
            continue
    assert ref is not None, "ningún intento del bucle de antes construyó el Model"
    viejos = [float(max(ref.predict(c).values())) for c in chunks]
    tras_puntuar = [m for m in ("sklearn", "scipy", "torch") if m in sys.modules]

    # Quien ENTRENE sigue llegando a la función real (que exige sus argumentos).
    import openwakeword
    try:
        openwakeword.train_custom_verifier()
        entrenar = "sin error"
    except TypeError as error:
        entrenar = str(error)
    real = sys.modules["openwakeword.custom_verifier_model"].train_custom_verifier
    print(json.dumps({
        "disponible": disponible,
        "pesados_tras_available": tras_available,
        "pesados_tras_puntuar": tras_puntuar,
        "sesiones": sorted(sesiones),
        "iguales": nuevos == viejos,
        "n": len(nuevos),
        "max": max(nuevos),
        "entrenar": entrenar,
        "entrenar_es_la_real": openwakeword.train_custom_verifier is real,
        "sklearn_al_entrenar": "sklearn" in sys.modules,
    }))
"""


def test_openwakeword_carga_sin_sklearn_con_3_sesiones_y_mismos_scores():
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    if not runtime.module_available("openwakeword"):
        pytest.skip("openwakeword no está instalado")
    out = _run_py(_OWW_REAL)
    assert out["disponible"] is True
    assert out["pesados_tras_available"] == []
    assert out["pesados_tras_puntuar"] == []
    # Antes 9 (los 5 pre-entrenados + 1 en intentos que reventaban después).
    assert out["sesiones"] == ["embedding_model.onnx", "hey_jarvis_v0.1.onnx",
                               "melspectrogram.onnx"]
    assert out["iguales"] is True and out["n"] == 125
    assert "train_custom_verifier() missing" in out["entrenar"], out["entrenar"]
    assert out["entrenar_es_la_real"] is True and out["sklearn_al_entrenar"] is True


def _oww_en_sys_modules(monkeypatch, model_cls) -> None:
    paquete = types.ModuleType("openwakeword")
    paquete.__file__ = "/nonexistent/openwakeword/__init__.py"
    modelo = types.ModuleType("openwakeword.model")
    modelo.Model = model_cls  # type: ignore[attr-defined]
    utils = types.ModuleType("openwakeword.utils")
    utils.download_models = lambda names: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", paquete)
    monkeypatch.setitem(sys.modules, "openwakeword.model", modelo)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils)


def test_get_model_en_0_4_construye_el_modelo_una_sola_vez(monkeypatch):
    """La firma de 0.4 va primero: antes los intentos 1 y 2 construían modelos
    (6 sesiones ONNX) que reventaban con TypeError y se tiraban."""
    pytest.importorskip("numpy")
    from crotolamo.voice.wakeword import WakeWordDetector

    construidos: list = []

    class _Oww04:
        def __init__(self, wakeword_model_paths=(), **kwargs) -> None:
            construidos.append((list(wakeword_model_paths), kwargs))
            if kwargs:  # AudioFeatures(**kwargs), DESPUÉS de cargar los modelos
                raise TypeError(f"unexpected keyword argument '{next(iter(kwargs))}'")

    _oww_en_sys_modules(monkeypatch, _Oww04)
    det = WakeWordDetector(model_name="hey_jarvis")
    assert isinstance(det._get_model(), _Oww04)
    assert construidos == [(["hey_jarvis"], {})]


def test_get_model_suelta_el_arena_del_calentamiento_y_guarda_el_estado_en_blanco(
        monkeypatch):
    np = pytest.importorskip("numpy")
    from crotolamo.voice import wakeword as wakeword_mod

    soltados: list = []

    class _Pre:
        def __init__(self) -> None:
            self.melspectrogram_buffer = np.ones((76, 32))
            self.feature_buffer = np.zeros((16, 1))
            self.raw_data_remainder = np.empty(0)

    class _Oww:
        def __init__(self, wakeword_model_paths=()) -> None:
            self.preprocessor = _Pre()

    def _soltar(modelo):
        # Al soltar el arena el modelo aún no está publicado (nadie lo usa) y
        # el estado en blanco se copia del preprocessor recién cargado.
        soltados.append((modelo, det._model))

    _oww_en_sys_modules(monkeypatch, _Oww)
    monkeypatch.setattr(wakeword_mod.runtime, "release_oww_warmup_arena", _soltar)
    det = wakeword_mod.WakeWordDetector(model_name="hey_jarvis")
    modelo = det._get_model()
    det._get_model()
    assert soltados == [(modelo, None)]
    assert set(det._blank_state or {}) == {"melspectrogram_buffer", "feature_buffer",
                                           "raw_data_remainder"}


def test_get_model_carga_una_sola_vez_aunque_lo_pidan_dos_hilos(monkeypatch):
    pytest.importorskip("numpy")
    from crotolamo.voice.wakeword import WakeWordDetector

    compuerta = threading.Event()
    construidos: list = []

    class _Lento:
        def __init__(self, wakeword_model_paths=()) -> None:
            construidos.append(self)
            compuerta.wait(5)

    _oww_en_sys_modules(monkeypatch, _Lento)
    det = WakeWordDetector(model_name="hey_jarvis")
    vistos: list = []
    hilos = [threading.Thread(target=lambda: vistos.append(det._get_model()))
             for _ in range(3)]
    for h in hilos:
        h.start()
    time.sleep(0.2)
    compuerta.set()
    for h in hilos:
        h.join(5)
    assert len(construidos) == 1
    assert len(vistos) == 3 and all(v is construidos[0] for v in vistos)


def test_resolve_model_path_no_ejecuta_el_init_de_openwakeword(monkeypatch, tmp_path):
    from crotolamo.voice.wakeword import WakeWordDetector

    pkg = _paquete_espia(tmp_path, "openwakeword",
                         {"resources/models/hey_jarvis_v0.1.onnx": b"onnx",
                          "resources/models/alexa_v0.1.onnx": b"onnx"})
    monkeypatch.delitem(sys.modules, "openwakeword", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    ruta = WakeWordDetector(model_name="hey_jarvis")._resolve_model_path()
    assert ruta == str(pkg / "resources" / "models" / "hey_jarvis_v0.1.onnx")
    assert "openwakeword" not in sys.modules
    assert WakeWordDetector(model_name="no_hay")._resolve_model_path() == "no_hay"


def test_available_es_false_sin_openwakeword(monkeypatch):
    from crotolamo.voice.wakeword import WakeWordDetector

    monkeypatch.setitem(sys.modules, "openwakeword", None)
    assert WakeWordDetector().available() is False


# =============================================================================
# Paso 4: el arena de Piper acotado sin frenar las frases típicas (TTS._get_voice)
# =============================================================================


class _VozPiperFalsa:
    """PiperVoice falso: cuenta cargas y trae una sesión que anota sus run_options."""

    cargas: list[str] = []

    def __init__(self) -> None:
        self.session = _SesionFalsa()
        self.config = types.SimpleNamespace(sample_rate=22050)

    @classmethod
    def load(cls, path):
        cls.cargas.append(path)
        return cls()


@pytest.fixture
def piper_falso(monkeypatch):
    fake = types.ModuleType("piper")
    fake.PiperVoice = _VozPiperFalsa  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake)
    _VozPiperFalsa.cargas = []
    return _VozPiperFalsa


def _ids(n: int) -> dict:
    """La entrada que Piper le da a su sesión para una frase de `n` phoneme ids."""
    np = pytest.importorskip("numpy")
    return {"input": np.zeros((1, n), dtype=np.int64),
            "input_lengths": np.array([n], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}


def test_get_voice_envuelve_la_sesion_de_piper_con_el_umbral(piper_falso):
    pytest.importorskip("onnxruntime")
    from crotolamo.voice.tts import TTS

    tts = TTS(Path("voz.onnx"))
    voz = tts._get_voice()
    assert isinstance(voz.session, runtime.ArenaShrinkingSession)
    assert voz.session._keep_len == runtime.PIPER_KEEP_PHONEMES == 256
    cruda = voz.session._session
    # Frase típica (<= 256 ids, ~115 caracteres): la MISMA llamada a ORT que hoy,
    # sin run_options, así que el arena se queda y la velocidad no cambia.
    assert voz.session.run(None, _ids(256)) == ["salida"]
    assert cruda.llamadas == [None]
    # Frase larga: devuelve su pico al sistema al terminar.
    voz.session.run(None, _ids(257))
    assert _encoge(cruda.llamadas[-1])
    # Lo demás de la voz sigue igual, y la caché también: ni se recarga ni se
    # vuelve a envolver.
    assert voz.config.sample_rate == 22050
    assert tts._get_voice() is voz and voz.session._session is cruda
    assert piper_falso.cargas == ["voz.onnx"]


def test_get_voice_deja_en_paz_una_voz_sin_sesion(monkeypatch):
    from crotolamo.voice.tts import TTS

    voz = types.SimpleNamespace(config="cfg")
    fake = types.ModuleType("piper")
    fake.PiperVoice = types.SimpleNamespace(load=lambda _ruta: voz)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake)
    assert TTS(Path("voz.onnx"))._get_voice() is voz
    assert not hasattr(voz, "session")


def test_get_voice_no_publica_la_voz_antes_de_envolverla(piper_falso, monkeypatch):
    """El camino caliente de _get_voice lee self._voice SIN lock: si la voz se
    publicara antes de envolverla, otro hilo podría sintetizar con la sesión sin
    acotar (o mientras se le cambia la sesión)."""
    from crotolamo.voice import tts as tts_mod

    vista: list = []
    envolver = runtime.wrap_piper_session

    def _espia(voz, keep_phonemes=runtime.PIPER_KEEP_PHONEMES):
        vista.append(tts._voice)
        return envolver(voz, keep_phonemes)

    monkeypatch.setattr(runtime, "wrap_piper_session", _espia)
    tts = tts_mod.TTS(Path("voz.onnx"))
    voz = tts._get_voice()
    assert vista == [None]
    assert tts._voice is voz


def test_get_voice_con_una_sesion_sin_arena_dice_las_frases_largas(monkeypatch, caplog):
    """Un Piper (u ORT) cuya sesión no tiene arena de CPU: sin acotar nada, la voz
    queda como la de fábrica y las frases de más de 256 ids siguen sonando."""
    pytest.importorskip("onnxruntime")
    from crotolamo.voice.tts import TTS

    voz = types.SimpleNamespace(session=_SesionSinArena(arena_declarada=False))
    cruda = voz.session
    fake = types.ModuleType("piper")
    fake.PiperVoice = types.SimpleNamespace(load=lambda _ruta: voz)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake)
    with caplog.at_level(logging.WARNING):
        assert TTS(Path("voz.onnx"))._get_voice() is voz
        assert voz.session is cruda
        assert voz.session.run(None, _ids(803)) == ["salida"]
    assert cruda.llamadas == [None]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _settings_falsos(tmp_path, **voice):
    return types.SimpleNamespace(paths={"voces": tmp_path},
                                 voice={"piper_voice": "v.onnx", **voice})


def test_piper_arena_keep_phonemes_de_la_config(piper_falso, tmp_path, caplog):
    from crotolamo.voice.tts import TTS

    def _umbral(**voice) -> int | None:
        tts = TTS.from_settings(_settings_falsos(tmp_path, **voice))
        assert tts.voice_model == tmp_path / "v.onnx"
        return tts._get_voice().session._keep_len

    assert _umbral() == 256, "sin la clave: el default de fábrica"
    assert _umbral(piper_arena_keep_phonemes=400) == 400
    assert _umbral(piper_arena_keep_phonemes=0) == 0, "0 = encoger tras cada frase"
    # Lo que documenta la config: 0 encoge tras cada frase y un número enorme nunca.
    pytest.importorskip("onnxruntime")
    for keep, frase, encoge in ((0, 1, True), (100000, 5000, False)):
        voz = TTS.from_settings(
            _settings_falsos(tmp_path, piper_arena_keep_phonemes=keep))._get_voice()
        voz.session.run(None, _ids(frase))
        assert _encoge(voz.session._session.llamadas[-1]) is encoge, keep
    # Un valor inválido no impide hablar: el default y UN aviso por valor.
    for malo in ("mucho", -1, True, 2.5):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="crotolamo.voice.tts"):
            assert _umbral(piper_arena_keep_phonemes=malo) == 256
        avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(avisos) == 1 and "piper_arena_keep_phonemes" in avisos[0], avisos


def test_la_config_de_fabrica_documenta_piper_arena_keep_phonemes():
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    voice = tomllib.loads(texto)["voice"]
    assert voice["piper_arena_keep_phonemes"] == runtime.PIPER_KEEP_PHONEMES == 256
    lineas = texto.splitlines()
    i = next(n for n, linea in enumerate(lineas)
             if linea.startswith("piper_arena_keep_phonemes"))
    # Junto a las demás claves de Piper, con su explicación encima.
    assert any(linea.startswith("piper_voice") for linea in lineas[i - 15:i])
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[i - 12:i]
                          if linea.startswith("#"))
    assert "arena" in comentario and "0 =" in comentario, comentario


def _voz_piper_real() -> str:
    """El .onnx de Piper para los tests con el modelo real: CROTOLAMO_TEST_PIPER_VOICE
    o el de la config versionada ([paths].voces + [voice].piper_voice)."""
    if not (runtime.module_available("piper") and runtime.module_available("onnxruntime")):
        pytest.skip("piper-tts/onnxruntime no están instalados")
    ruta = os.environ.get("CROTOLAMO_TEST_PIPER_VOICE", "")
    if not ruta:
        try:
            from crotolamo.settings import get_settings
            from crotolamo.voice.tts import TTS

            ruta = str(TTS.from_settings(get_settings()).voice_model)
        except Exception as error:  # noqa: BLE001
            pytest.skip(f"sin config para localizar la voz: {error}")
    if not (os.path.isfile(ruta) and os.path.isfile(ruta + ".json")):
        pytest.skip(f"no hay voz Piper en {ruta} (CROTOLAMO_TEST_PIPER_VOICE la indica)")
    return ruta


_CORTA = "Listo, patrón."
_RESPUESTA = ("Son las tres de la tarde, patrón, y ya quedó anotado en tu lista de pendientes "
              "para mañana temprano.")
_LARGA = ("Mira patrón, revisé los tres proyectos que me pediste y encontré que en el primero "
          "hay doce archivos modificados desde el lunes, en el segundo no hay cambios "
          "pendientes, y en el tercero quedaron dos pruebas fallando que tienen que ver con "
          "el módulo de audio")


def test_piper_real_mismo_audio_y_el_arena_no_se_queda_el_pico_de_la_frase_larga():
    """Con la voz real, en un proceso nuevo: el audio sale idéntico bit a bit al de
    PiperVoice.load de siempre (sin ruido, para que sea determinista), solo los runs
    de más de 256 ids pasan run_options, y tras una frase larga la RSS ya no se
    queda con su pico (antes ~+330 MB para esta frase, de por vida del servicio)."""
    ruta = _voz_piper_real()
    out = _run_py(f"""
        import gc, hashlib, json
        import numpy as np
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        from crotolamo.voice import runtime
        from crotolamo.voice.tts import TTS

        def rss():
            with open("/proc/self/status") as f:
                for linea in f:
                    if linea.startswith("VmRSS"):
                        return int(linea.split()[1]) // 1024
            return 0

        voz = TTS({ruta!r})._get_voice()
        proxy = voz.session
        cruda = proxy._session
        runs = []
        run_real = cruda.run

        def run(names, feed, *opciones):
            runs.append([int(feed["input"].shape[-1]), bool(opciones)])
            return run_real(names, feed, *opciones)

        cruda.run = run
        cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

        def md5(v, texto):
            audio = np.concatenate([c.audio_int16_array
                                    for c in v.synthesize(texto, syn_config=cfg)])
            gc.collect()
            return hashlib.md5(audio.tobytes()).hexdigest()

        nuevo = [md5(voz, {_CORTA!r})]
        r_corta = rss()
        nuevo += [md5(voz, {_RESPUESTA!r}), md5(voz, {_LARGA!r})]
        r_larga = rss()
        md5(voz, {_CORTA!r})
        r_corta_otra_vez = rss()
        stock = PiperVoice.load({ruta!r})
        viejo = [md5(stock, t) for t in ({_CORTA!r}, {_RESPUESTA!r}, {_LARGA!r})]
        print(json.dumps({{
            "proxy": type(proxy).__name__, "keep": proxy._keep_len,
            "iguales": nuevo == viejo, "runs": runs,
            "tras_larga_mb": r_larga - r_corta,
            "trinquete_mb": r_corta_otra_vez - r_corta,
        }}))
    """)
    assert out["proxy"] == "ArenaShrinkingSession" and out["keep"] == 256
    assert out["iguales"] is True, "el audio cambió"
    largos = [n for n, _ in out["runs"]]
    assert min(largos) <= 256 < max(largos), out["runs"]
    assert all(encoge == (n > 256) for n, encoge in out["runs"]), out["runs"]
    # Sin el arreglo: trinquete ~+330 MB con esta frase. Con él, unos pocos MB.
    assert out["tras_larga_mb"] < 100, out
    assert out["trinquete_mb"] < 100, out


def test_piper_real_sin_arena_las_frases_largas_suenan_como_las_de_fabrica():
    """Con la voz real y un ORT SIN arena de CPU (lo que da un build con mimalloc o un
    Piper que lo apague), ORT rechaza el encogido. Antes del arreglo toda frase de más
    de 256 ids reventaba y TTS.speak la callaba. Ahora: si la sesión dice que no
    tiene arena no se envuelve, y si lo oculta, el run se repite como el de fábrica,
    deja de encoger y avisa una vez. En los dos casos, el mismo audio bit a bit."""
    ruta = _voz_piper_real()
    out = _run_py(f"""
        import hashlib, json, logging
        import numpy as np
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        from crotolamo.voice import runtime
        from crotolamo.voice.tts import TTS

        _opciones = ort.SessionOptions

        def _sin_arena():
            so = _opciones()
            so.enable_cpu_mem_arena = False
            return so

        ort.SessionOptions = _sin_arena  # PiperVoice.load crea su sesión con esto
        avisos = []

        class _Anota(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    avisos.append(record.getMessage())

        logging.getLogger("crotolamo.voice.runtime").addHandler(_Anota())
        cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

        def md5(v, texto):
            audio = np.concatenate([c.audio_int16_array
                                    for c in v.synthesize(texto, syn_config=cfg)])
            return hashlib.md5(audio.tobytes()).hexdigest()

        def anotar(sesion):
            runs = []
            run_real = sesion.run

            def run(names, feed, *opciones):
                runs.append([int(feed["input"].shape[-1]), bool(opciones)])
                return run_real(names, feed, *opciones)

            sesion.run = run
            return runs

        stock = PiperVoice.load({ruta!r})
        fabrica = [md5(stock, t) for t in ({_CORTA!r}, {_LARGA!r})]

        declarada = TTS({ruta!r})._get_voice()
        tipo_declarada = type(declarada.session).__name__
        audio_declarada = [md5(declarada, t) for t in ({_CORTA!r}, {_LARGA!r})]

        runtime._has_cpu_arena = lambda sesion: True  # el ORT no lo dice
        oculta = TTS({ruta!r})._get_voice()
        tipo_oculta = type(oculta.session).__name__
        runs = anotar(oculta.session._session)
        audio_oculta = [md5(oculta, t) for t in ({_CORTA!r}, {_LARGA!r}, {_LARGA!r})]
        print(json.dumps({{
            "tipo_declarada": tipo_declarada, "tipo_oculta": tipo_oculta,
            "fabrica": fabrica, "declarada": audio_declarada, "oculta": audio_oculta,
            "runs": runs, "avisos": avisos,
        }}))
    """)
    assert out["tipo_declarada"] == "InferenceSession", out
    assert out["declarada"] == out["fabrica"], "sin arena declarado: otro audio"
    assert out["tipo_oculta"] == "ArenaShrinkingSession", out
    assert out["oculta"] == out["fabrica"] + [out["fabrica"][1]], "con el respaldo: otro audio"
    largos = [n for n, _ in out["runs"] if n > 256]
    assert largos, out["runs"]
    # Solo el PRIMER run largo intenta encoger (y ORT lo rechaza); el resto, de fábrica.
    intentos = [r for r in out["runs"] if r[1]]
    assert len(intentos) == 1 and intentos[0][0] > 256, out["runs"]
    assert len(out["avisos"]) == 1 and "arena" in out["avisos"][0], out["avisos"]


# =============================================================================
# Paso 5: Whisper por el runtime (STT._get_model)
# =============================================================================


class _WhisperAnotador:
    """WhisperModel falso: anota (tamaño, kwargs) de cada construcción. Con
    local_files_only revienta salvo que el tamaño esté en `en_cache`, como
    LocalEntryNotFoundError de huggingface_hub."""

    llamadas: list = []
    en_cache: set = set()

    def __init__(self, size, **kwargs) -> None:
        _WhisperAnotador.llamadas.append((size, kwargs))
        if kwargs.get("local_files_only") and size not in _WhisperAnotador.en_cache:
            raise RuntimeError("LocalEntryNotFoundError: no está en la caché local")


@pytest.fixture
def whisper_falso(monkeypatch):
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _WhisperAnotador  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    tqdm_falso = types.ModuleType("tqdm")
    tqdm_falso.tqdm = type("tqdm", (), {"monitor_interval": 10})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_falso)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {})
    _WhisperAnotador.llamadas = []
    _WhisperAnotador.en_cache = set()
    yield _WhisperAnotador, tqdm_falso
    for size in ("_perf_a_", "_perf_b_"):
        stt_mod._models.pop(size, None)


_HOY = {"device": "cpu", "compute_type": "int8"}  # la llamada de siempre


def test_get_model_carga_whisper_de_la_cache_local_sin_preguntar_al_hub(whisper_falso):
    cls, tqdm_falso = whisper_falso
    cls.en_cache = {"_perf_a_"}
    modelo = stt_mod.STT(model_size="_perf_a_")._get_model()
    assert isinstance(modelo, cls) and stt_mod._models["_perf_a_"] is modelo
    assert cls.llamadas == [("_perf_a_", {**_HOY, "local_files_only": True})]
    assert tqdm_falso.tqdm.monitor_interval == 0, "el monitor de tqdm sigue activo"
    # La caché por tamaño sigue siendo la de siempre: no se construye otra vez.
    assert stt_mod.STT(model_size="_perf_a_")._get_model() is modelo
    assert len(cls.llamadas) == 1


def test_get_model_sin_cache_local_hace_la_llamada_de_siempre(whisper_falso):
    """Primera vez (no está en la caché): EXACTAMENTE la llamada de hoy, que lo
    descarga del Hub."""
    cls, _tqdm = whisper_falso
    modelo = stt_mod.STT(model_size="_perf_b_")._get_model()
    assert stt_mod._models["_perf_b_"] is modelo
    assert cls.llamadas == [("_perf_b_", {**_HOY, "local_files_only": True}),
                            ("_perf_b_", _HOY)]


def test_whisper_cpu_threads_de_la_config(whisper_falso, monkeypatch, caplog):
    cls, _tqdm = whisper_falso
    cls.en_cache = {"_perf_a_"}
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"whisper_cpu_threads": 2})
    stt_mod.STT(model_size="_perf_a_")._get_model()
    assert cls.llamadas == [("_perf_a_", {**_HOY, "local_files_only": True, "cpu_threads": 2})]

    # Un valor inválido no tumba la carga: el default de la librería y UN aviso.
    stt_mod._models.pop("_perf_a_")
    cls.llamadas.clear()
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"whisper_cpu_threads": "muchos"})
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        stt_mod.STT(model_size="_perf_a_")._get_model()
    assert cls.llamadas == [("_perf_a_", {**_HOY, "local_files_only": True})]
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and "whisper_cpu_threads" in avisos[0], avisos


_RED_BLOQUEADA = """
    import json, os, socket, sys
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    {extra}
    conexiones = []
    _connect = socket.socket.connect

    def _sin_red(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            conexiones.append(str(addr)[:60])
            raise ConnectionRefusedError("test: red bloqueada")
        return _connect(self, addr)

    socket.socket.connect = _sin_red
    import numpy as np
    import faster_whisper
    from crotolamo.voice import stt
    stt._voice_cfg = lambda: {{}}
"""


def test_whisper_cacheado_no_pregunta_al_hub_ni_deja_el_monitor_de_tqdm():
    """Con la red bloqueada y el modelo ya descargado: 0 conexiones al cargar (antes
    1 por modelo en cada arranque) y sin el hilo TMonitor tras transcribir."""
    if not runtime.module_available("faster_whisper"):
        pytest.skip("faster-whisper no está instalado")
    out = _run_py(textwrap.dedent(_RED_BLOQUEADA).format(extra="") + textwrap.dedent("""
        import threading
        from faster_whisper.utils import download_model
        try:  # ¿está 'tiny' en la caché? (lo mismo que mira local_files_only)
            download_model("tiny", local_files_only=True)
        except Exception as error:
            print(json.dumps({"skip": f"{type(error).__name__}: {error}"}))
            raise SystemExit(0)
        orden = stt.STT(model_size="tiny", initial_prompt=None)
        modelo = orden._get_model()
        al_cargar = len(conexiones)
        orden.transcribe(np.zeros(16000, dtype=np.float32))
        tras_transcribir = len(conexiones)
        monitores = [t for t in threading.enumerate() if type(t).__name__ == "TMonitor"]
        # Referencia: la llamada de antes SÍ pregunta al Hub (el contador mide).
        faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
        print(json.dumps({
            "modelo": type(modelo).__name__,
            "al_cargar": al_cargar,
            "tras_transcribir": tras_transcribir,
            "monitores_tqdm": len(monitores),
            "la_llamada_de_antes": len(conexiones) - tras_transcribir,
        }))
    """))
    if "skip" in out:
        pytest.skip(f"faster-whisper 'tiny' no está en la caché local: {out['skip']}")
    assert out["modelo"] == "WhisperModel"
    assert out["al_cargar"] == 0 and out["tras_transcribir"] == 0
    assert out["monitores_tqdm"] == 0
    assert out["la_llamada_de_antes"] >= 1


def test_whisper_sin_cache_sigue_yendo_al_hub_como_antes(tmp_path):
    """Primera vez (caché vacía): el camino nuevo va al Hub igual que el de antes
    (ahí descargaría) y, con la red caída, falla con EL MISMO error."""
    if not runtime.module_available("faster_whisper"):
        pytest.skip("faster-whisper no está instalado")
    vacia = tmp_path / "hf"
    vacia.mkdir()
    extra = f"os.environ['HF_HUB_CACHE'] = {str(vacia)!r}; os.environ['HF_HOME'] = {str(vacia)!r}"
    out = _run_py(textwrap.dedent(_RED_BLOQUEADA).format(extra=extra) + textwrap.dedent("""
        def _error(fn):
            antes = len(conexiones)
            try:
                fn()
            except Exception as error:
                return [type(error).__name__, str(error), len(conexiones) - antes]
            return ["sin error", "", len(conexiones) - antes]

        viejo = _error(lambda: faster_whisper.WhisperModel("tiny", device="cpu",
                                                           compute_type="int8"))
        nuevo = _error(lambda: stt.STT(model_size="tiny")._get_model())
        print(json.dumps({"viejo": viejo, "nuevo": nuevo, "cacheado": "tiny" in stt._models}))
    """))
    assert out["viejo"][0] != "sin error", out
    assert out["nuevo"][:2] == out["viejo"][:2], out
    assert out["nuevo"][2] >= 1, "el camino nuevo no intentó descargar"
    assert out["cacheado"] is False


def test_la_config_de_fabrica_documenta_whisper_cpu_threads_con_el_default_de_hoy():
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert tomllib.loads(texto)["voice"]["whisper_cpu_threads"] == 0
    lineas = texto.splitlines()
    i = next(n for n, linea in enumerate(lineas) if linea.startswith("whisper_cpu_threads"))
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[i - 4:i])
    assert "0 = el default de" in comentario and "faster-whisper" in comentario, comentario


# =============================================================================
# Paso 7: el doctor comprueba lo que usa el demonio, sin los imports pesados
# =============================================================================

_OWW_OK = ("voz-oww", True, "openwakeword importable (modo concurrente OK)", "")
_OWW_FALTA = ("voz-oww", False,
              "falta openwakeword (sin él, solo modo simple/difuso está disponible)",
              "pip install openwakeword  (o pip install -e '.[voice]')")
_SILERO_OK = ("voz-silero", True, "silero-vad importable (VAD neuronal activo)", "")
_SILERO_FALTA = ("voz-silero", False,
                 "falta silero-vad (el modo concurrente usará VAD por energía como fallback)",
                 "pip install silero-vad --extra-index-url https://download.pytorch.org/whl/cpu"
                 "  (opcional, mejora la detección de voz)")


def _resumen(checks) -> tuple:
    (check,) = checks
    return (check.name, check.ok, check.detail, check.fix)


def test_doctor_voz_oww_y_voz_silero_sin_los_paquetes_dicen_falta(monkeypatch):
    """El idioma de tests/test_doctor.py (sys.modules[...] = None): mismos textos."""
    from scripts import crotolamo_doctor as doctor

    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    assert _resumen(doctor._check_oww(None)) == _OWW_FALTA
    assert _resumen(doctor._check_silero(None)) == _SILERO_FALTA


def test_doctor_voz_silero_abre_el_onnx_sin_ejecutar_el_paquete(monkeypatch, tmp_path):
    """Con data/silero_vad.onnx basta que abra con onnxruntime: el __init__ del
    paquete (que importa torch) ni se ejecuta. Sin el archivo, el chequeo de
    siempre (importar el paquete): uno que no importa dice 'falta'."""
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    real = runtime.package_file("silero_vad", "data", "silero_vad.onnx")
    if real is None:
        pytest.skip("silero_vad (con su data/silero_vad.onnx) no está instalado")
    from scripts import crotolamo_doctor as doctor

    con = tmp_path / "con"
    _paquete_espia(con, "silero_vad", {"data/silero_vad.onnx": Path(real).read_bytes()})
    monkeypatch.delitem(sys.modules, "silero_vad", raising=False)
    monkeypatch.syspath_prepend(str(con))
    assert _resumen(doctor._check_silero(None)) == _SILERO_OK
    assert "silero_vad" not in sys.modules

    sin = tmp_path / "sin"
    (sin / "silero_vad").mkdir(parents=True)
    (sin / "silero_vad" / "__init__.py").write_text(
        "raise ImportError(\"No module named 'torch'\")\n")
    monkeypatch.syspath_prepend(str(sin))
    assert _resumen(doctor._check_silero(None)) == _SILERO_FALTA


def test_doctor_voz_silero_decide_lo_mismo_que_load_silero(monkeypatch, tmp_path):
    """Con el ONNX empaquetado, el doctor sigue la decisión del demonio: sin
    numpy/onnxruntime el VAD neuronal no arranca (falta, aunque el paquete
    importe); si el ONNX no abre por otra causa, el demonio usa el cargador de
    silero_vad y el doctor comprueba ese import, como siempre."""
    from scripts import crotolamo_doctor as doctor

    onnx = tmp_path / "silero_vad.onnx"
    onnx.write_bytes(b"onnx")
    monkeypatch.setattr(runtime, "package_file", lambda *a: str(onnx))
    # silero_vad 'importable' (falso): si el doctor cayera al import, diría OK.
    monkeypatch.setitem(sys.modules, "silero_vad", types.ModuleType("silero_vad"))

    def _sin_ort(path):
        raise ModuleNotFoundError("No module named 'onnxruntime'")

    monkeypatch.setattr(runtime, "SileroOnnx", _sin_ort)
    assert _resumen(doctor._check_silero(None)) == _SILERO_FALTA

    def _otra_version(path):
        raise ValueError(f"entradas inesperadas en {path}")

    monkeypatch.setattr(runtime, "SileroOnnx", _otra_version)
    assert _resumen(doctor._check_silero(None)) == _SILERO_OK
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    assert _resumen(doctor._check_silero(None)) == _SILERO_FALTA


def test_doctor_voz_oww_dice_falta_si_el_import_revienta_con_otra_cosa(monkeypatch):
    """Como _try_import: una .so de onnxruntime rota (OSError) también es 'falta'."""
    from scripts import crotolamo_doctor as doctor

    def _so_rota():
        raise OSError("libonnxruntime.so: cannot open shared object file")

    monkeypatch.setattr(runtime, "import_openwakeword", _so_rota)
    assert _resumen(doctor._check_oww(None)) == _OWW_FALTA


_DOCTOR_VOZ = """
    import json, sys
    from scripts import crotolamo_doctor as doctor

    def r(checks):
        (c,) = checks
        return [c.name, c.ok, c.detail]

    print(json.dumps({
        "oww": r(doctor._check_oww(None)),
        "silero": r(doctor._check_silero(None)),
        "pesados": [m for m in ("torch", "sklearn", "scipy", "silero_vad")
                    if m in sys.modules],
    }))
"""


def test_doctor_voz_no_importa_torch_ni_sklearn():
    """Antes, decir 'importable' costaba torch + sklearn + scipy (~280 MB, segundos)."""
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    if (not runtime.module_available("openwakeword")
            or runtime.package_file("silero_vad", "data", "silero_vad.onnx") is None):
        pytest.skip("openwakeword o silero_vad no están instalados")
    out = _run_py(_DOCTOR_VOZ)
    assert out == {"oww": list(_OWW_OK[:3]), "silero": list(_SILERO_OK[:3]), "pesados": []}


# =============================================================================
# Paso 8: la extra [voice] declara lo que runtime.py importa directo
# =============================================================================


def test_la_extra_voice_declara_onnxruntime():
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    voz = data["project"]["optional-dependencies"]["voice"]
    assert "onnxruntime" in voz
