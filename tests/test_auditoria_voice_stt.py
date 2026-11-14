"""Arreglos de la auditoría en la captura de voz (STT, endpointing, normalización).

Cada bloque fija un hallazgo confirmado; sin micrófono ni Whisper reales: el
`faster_whisper` que ve `stt._require` es un módulo falso en `sys.modules`.

- La carga perezosa de Whisper se hace UNA vez aunque dos hilos la pidan a la
  vez (WarmVoice + el primer transcribe): antes cada hilo construía su copia
  (~2 s de CPU y ~137 MB de pico tirados). El lock es POR tamaño: cargar 'base'
  no deja sordo al wake, que usa 'tiny'.
- El endpointing no confunde la forma con tilde que CIERRA la frase con el
  conector átono que la deja abierta: "sí" no es "si", "está" no es "esta",
  "él/mí/tú" no son "el/mi/tu". Un "sí" a secas ya no reabre la escucha.
- Las listas del endpointing se comparan ya normalizadas (sin acentos): una
  entrada escrita con tilde/eñe no puede coincidir nunca.
- Las correcciones de Whisper se aplican de la clave más larga a la más corta:
  "yutub" es prefijo de "yutube" y, aplicada antes, dejaba "youtubee".
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types

import pytest

from crotolamo.voice import endpoint as endpoint_mod
from crotolamo.voice import normalize as normalize_mod
from crotolamo.voice import stt as stt_mod
from crotolamo.voice.endpoint import seems_incomplete
from crotolamo.voice.normalize import normalize_text

# --- carga perezosa de Whisper: una sola construcción por tamaño --------------


class _SlowWhisper:
    """WhisperModel falso: cuenta construcciones y se queda dentro del
    constructor hasta que el test abre la compuerta de su tamaño (así la
    ventana de la carrera dura lo que el test quiera, no lo que tarde el CPU)."""

    built: list[str] = []
    gates: dict[str, threading.Event] = {}
    fail_next: bool = False
    _guard = threading.Lock()

    def __init__(self, model_size, **kwargs):
        with _SlowWhisper._guard:
            _SlowWhisper.built.append(model_size)
            fail = _SlowWhisper.fail_next
            _SlowWhisper.fail_next = False
        if fail:
            raise RuntimeError("descarga cortada")
        gate = _SlowWhisper.gates.get(model_size)
        if gate is not None:
            gate.wait(5)


@pytest.fixture
def slow_whisper(monkeypatch):
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _SlowWhisper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    _SlowWhisper.built = []
    _SlowWhisper.gates = {}
    _SlowWhisper.fail_next = False
    sizes = ("_aud_a_", "_aud_b_")
    yield _SlowWhisper
    for gate in _SlowWhisper.gates.values():
        gate.set()  # nunca dejar un hilo colgado si el test falló a medias
    for size in sizes:
        stt_mod._models.pop(size, None)


def _wait_until(cond, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_dos_hilos_cargan_whisper_una_sola_vez(slow_whisper):
    """WarmVoice y el primer transcribe() piden el mismo modelo a la vez."""
    slow_whisper.gates["_aud_a_"] = threading.Event()
    start = threading.Barrier(2)
    got: list[object] = []

    def _load():
        start.wait(5)
        got.append(stt_mod.STT(model_size="_aud_a_")._get_model())

    threads = [threading.Thread(target=_load, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    # Con la carrera, el segundo hilo entra al constructor mientras el primero
    # sigue dentro; con el lock, espera fuera. (Si la máquina va tan cargada que
    # el segundo hilo ni arranca en este plazo, encuentra el modelo ya cacheado:
    # el test nunca falla en falso.)
    _wait_until(lambda: len(slow_whisper.built) >= 2, 0.4)
    slow_whisper.gates["_aud_a_"].set()
    for t in threads:
        t.join(5)
        assert not t.is_alive()

    assert slow_whisper.built == ["_aud_a_"]
    assert len(got) == 2 and got[0] is got[1]
    assert stt_mod._models["_aud_a_"] is got[0]


def test_cargar_un_tamano_no_bloquea_otro(slow_whisper):
    """El lock es por tamaño: mientras 'base' (comando) carga o descarga, el
    wake ('tiny') carga el suyo sin esperar. Un lock global lo dejaría sordo."""
    slow_whisper.gates["_aud_a_"] = threading.Event()  # 'a' se queda cargando
    lento = threading.Thread(
        target=lambda: stt_mod.STT(model_size="_aud_a_")._get_model(), daemon=True)
    lento.start()
    assert _wait_until(lambda: "_aud_a_" in slow_whisper.built, 5)

    rapido = threading.Thread(
        target=lambda: stt_mod.STT(model_size="_aud_b_")._get_model(), daemon=True)
    rapido.start()
    rapido.join(5)
    assert not rapido.is_alive(), "cargar otro tamaño esperó al que seguía cargando"
    assert "_aud_b_" in stt_mod._models
    assert lento.is_alive()  # 'a' seguía dentro del constructor todo el rato

    slow_whisper.gates["_aud_a_"].set()
    lento.join(5)
    assert sorted(slow_whisper.built) == ["_aud_a_", "_aud_b_"]


def test_carga_fallida_no_envenena_la_siguiente(slow_whisper):
    """Si la carga revienta (descarga cortada), no queda nada cacheado ni el
    lock tomado: el siguiente intento vuelve a construir y funciona."""
    slow_whisper.fail_next = True
    stt = stt_mod.STT(model_size="_aud_a_")
    with pytest.raises(RuntimeError):
        stt._get_model()
    assert "_aud_a_" not in stt_mod._models

    done: list[object] = []
    t = threading.Thread(target=lambda: done.append(stt._get_model()), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive()
    assert done and stt_mod._models["_aud_a_"] is done[0]
    assert slow_whisper.built == ["_aud_a_", "_aud_a_"]


def test_modelo_ya_cacheado_no_toca_faster_whisper(monkeypatch):
    """Camino caliente intacto: con el modelo en caché no se importa ni se
    construye nada (los tests que inyectan un modelo falso dependen de esto)."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import -> ImportError
    fake = object()
    stt_mod._models["_aud_a_"] = fake
    try:
        assert stt_mod.STT(model_size="_aud_a_")._get_model() is fake
    finally:
        stt_mod._models.pop("_aud_a_", None)


# --- endpointing: la tilde distingue la palabra que cierra del conector -------


def test_formas_tonicas_cierran_la_frase():
    for frase in [
        "sí",
        "Sí.",
        "SÍ",
        "creo que sí",
        "dile que sí",
        "¿dónde está?",
        "dime dónde está",
        "llámale a él",
        "mándamelo a mí",
        "igual que tú",
        "sí",  # "sí" en NFD (tilde combinante): misma palabra
    ]:
        assert not seems_incomplete(frase), frase


def test_los_conectores_atonos_siguen_siendo_pausa_de_pensar():
    for frase in [
        "avísame si",
        "abre esta",
        "pásame el",
        "busca en mi",
        "sube tu",
        "Avísame SI",
    ]:
        assert seems_incomplete(frase), frase
    # La coma de trailing-off manda aunque la palabra cierre: "sí, ..." seguía.
    assert seems_incomplete("sí,")


class _ScriptedSTT(stt_mod.STT):
    """STT con _listen_transcribe guionizado (sin mic ni Whisper reales)."""

    def __init__(self, script):
        super().__init__(model_size="fake")
        self.script = list(script)
        self.calls: list[dict] = []

    def _listen_transcribe(self, hotwords=None, **vad_kwargs):
        self.calls.append({"hotwords": hotwords, **vad_kwargs})
        return self.script.pop(0) if self.script else ""


@pytest.mark.parametrize("respuesta", ["sí", "¿dónde está?", "llámale a él"])
def test_listen_smart_no_reabre_con_una_frase_que_cierra_en_tonica(respuesta):
    """Antes cada una costaba una segunda grabación de endpoint_continue_s
    (2.5 s de espera) y lo que sonara en esa ventana se pegaba a la orden."""
    stt = _ScriptedSTT([respuesta, "ruido de la tele"])
    assert stt.listen_smart(rounds=2) == respuesta
    assert len(stt.calls) == 1


def test_listas_del_endpoint_estan_en_forma_normalizada():
    """seems_incomplete compara contra _norm(palabra): una entrada con tilde o
    eñe ("enseñame") no puede coincidir jamás, es dato muerto."""
    for lista in (endpoint_mod._CONNECTORS, endpoint_mod._DANGLING_VERBS):
        muertas = sorted(w for w in lista if endpoint_mod._norm(w) != w)
        assert muertas == []
    # y la forma viva sigue reconociéndose
    assert seems_incomplete("enséñame")
    assert seems_incomplete("ensename")


# --- normalización: ninguna regla queda tapada por otra más corta -------------


def test_yutube_no_queda_en_youtubee():
    assert normalize_text("abre yutube") == "abre youtube"
    assert normalize_text("abre yutub") == "abre youtube"
    assert normalize_text("Pon Yutube en la tele") == "pon youtube en la tele"


def test_cada_regla_de_reemplazo_produce_su_forma_correcta():
    """Toda clave de _REPLACEMENTS, sola, debe dar exactamente su valor: si una
    regla más corta le gana el turno, la larga es código muerto y el texto sale
    mal corregido."""
    for wrong, right in normalize_mod._REPLACEMENTS.items():
        assert normalize_text(wrong) == right, wrong


# --- V7(c): un Silero que no carga se intenta UNA vez y se avisa UNA vez -------
#
# Con [voice].vad_backend = "silero" y silero-vad ausente (u ONNX corrupto), cada
# grabación del modo simple reintentaba el import/carga y escribía un WARNING:
# uno cada <=6 s, para siempre, en el bucle del wake difuso. El resultado era el
# mismo siempre (cae al VAD por energía): trabajo y ruido en el log tirados.


class _MicSilencio:
    """sounddevice.InputStream falso: entrega silencio digital, sin hardware."""

    abiertos = 0

    def __init__(self, **kwargs):
        type(self).abiertos += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        import numpy as np

        return np.zeros((n, 1), dtype=np.float32), False


@pytest.fixture
def silero_falso(monkeypatch):
    """silero_vad falso cuyo load_silero_vad hace lo que diga `carga` y cuenta
    intentos; torch y sounddevice falsos; caché y memoria de fallo limpias."""
    pytest.importorskip("numpy")
    estado: dict = {"intentos": 0, "carga": None}

    def _load(onnx=True):
        estado["intentos"] += 1
        return estado["carga"]()

    fake = types.ModuleType("silero_vad")
    fake.load_silero_vad = _load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", fake)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_MicSilencio))
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    # raising=False: el test debe poder correr (y fallar por la razón buena)
    # contra el código sin la memoria de fallo.
    monkeypatch.setattr(stt_mod, "_silero_load_error", None, raising=False)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"vad_backend": "silero"})
    _MicSilencio.abiertos = 0
    return estado


def _onnx_corrupto():
    raise RuntimeError("ONNX corrupto")


def test_silero_que_no_carga_se_intenta_y_se_avisa_una_sola_vez(silero_falso, caplog):
    silero_falso["carga"] = _onnx_corrupto
    stt = stt_mod.STT(model_size="fake")
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.stt"):
        for _ in range(5):
            stt._record_frames(None, 0.6, 0.4)
    assert silero_falso["intentos"] == 1, "cada grabación reintentaba cargar Silero"
    avisos = [r for r in caplog.records
              if r.levelno >= logging.WARNING and "silero" in r.getMessage().lower()]
    assert len(avisos) == 1, [r.getMessage() for r in avisos]
    # El fallback sigue grabando de verdad en cada vuelta (por energía).
    assert _MicSilencio.abiertos == 5


def test_get_silero_vad_no_reintenta_una_carga_fallida(silero_falso):
    """También el loop concurrente (adapters._SileroVad) pasa por aquí: una
    instancia nueva no vuelve a pagar el import/carga que ya falló."""
    silero_falso["carga"] = _onnx_corrupto
    for _ in range(3):
        with pytest.raises(Exception, match="ONNX corrupto"):
            stt_mod._get_silero_vad()
    assert silero_falso["intentos"] == 1
    assert stt_mod._silero_vad is None


def test_get_silero_vad_sigue_cacheando_la_carga_buena(silero_falso):
    modelo = object()
    silero_falso["carga"] = lambda: modelo
    assert stt_mod._get_silero_vad() is modelo
    assert stt_mod._get_silero_vad() is modelo
    assert silero_falso["intentos"] == 1


def test_un_fallo_al_grabar_con_silero_no_lo_apaga_para_siempre(silero_falso):
    """Solo se memoriza el fallo de CARGA (permanente en el proceso). Un fallo
    con el modelo ya cargado (estado, stream) se reintenta en la siguiente
    grabación, como antes."""
    resets = {"n": 0}

    class _ModeloQueFalla:
        def reset_states(self):
            resets["n"] += 1
            raise RuntimeError("estado roto")

    silero_falso["carga"] = _ModeloQueFalla
    stt = stt_mod.STT(model_size="fake")
    stt._record_frames(None, 0.6, 0.4)
    stt._record_frames(None, 0.6, 0.4)
    assert resets["n"] == 2
    assert silero_falso["intentos"] == 1  # el modelo cargado sí queda en caché


# --- V7(a): el VAD por energía no manda a Whisper una ventana sin voz ---------
#
# Con [voice].vad_backend = "energy" (o Silero sin cargar), una ventana en la que
# nadie habló devolvía TODOS los frames (6 s en el wake difuso): min_audio_s=0.3
# no la filtraba y Whisper transcribía ruido normalizado a 0.9 en cada ciclo,
# con el micrófono cerrado (la "ventana sorda"). Silero ya devolvía [] ahí.


class _MicGuion:
    """sounddevice.InputStream falso que entrega un guion de audio (sin hardware)."""

    audio = None
    pos = 0

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        import numpy as np

        cls = type(self)
        block = cls.audio[cls.pos:cls.pos + n]
        cls.pos += n
        if len(block) < n:
            block = np.concatenate([block, np.zeros(n - len(block), dtype=np.float32)])
        return block.reshape(-1, 1).astype(np.float32), False


def _ruido(segundos: float, amp: float, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * segundos)) * amp).astype(np.float32)


def _voz(segundos: float, amp: float = 0.1):
    import numpy as np

    t = np.arange(int(16000 * segundos)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp).astype(np.float32)


@pytest.fixture
def mic_energia(monkeypatch):
    """STT con backend de energía, mic guionizado y transcribe() espía."""
    np = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_MicGuion))
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"vad_backend": "energy"})
    stt = stt_mod.STT(model_size="fake", initial_prompt=None)
    llamadas: list[float] = []

    def _transcribe(audio, hotwords=None):
        llamadas.append(audio.size / 16000)
        return "algo"

    monkeypatch.setattr(stt, "transcribe", _transcribe)

    def _poner(*partes):
        _MicGuion.audio = np.concatenate(partes).astype(np.float32)
        _MicGuion.pos = 0

    return stt, _poner, llamadas


@pytest.mark.parametrize("fondo", ["sala_baja", "sala_alta", "ceros", "ceros_y_sala"])
def test_energia_sin_voz_no_llega_a_whisper(mic_energia, fondo):
    """Ruido de fondo estable (bajo, o por encima del piso absoluto de 0.01/3),
    silencio digital (mic silenciado) o un primer chunk de ceros al abrir el
    stream: el wake difuso no transcribe nada y la ventana de espera dura lo
    mismo que antes."""
    import numpy as np

    stt, poner, llamadas = mic_energia
    ceros = np.zeros(16000 * 10, dtype=np.float32)
    poner({"sala_baja": _ruido(10, 0.0005), "sala_alta": _ruido(10, 0.004),
           "ceros": ceros, "ceros_y_sala": np.concatenate([ceros[:480], _ruido(10, 0.002)]),
           }[fondo])
    oido = stt.listen_once(silence_ms=400, max_seconds=8.0, start_timeout_s=6,
                           min_audio_s=0.3)
    assert oido == ""
    assert llamadas == [], "Whisper transcribió una ventana sin voz"
    # Se esperó toda la ventana de arranque (6 s), ni más ni menos que antes.
    assert abs(_MicGuion.pos / 16000 - 6.0) < 0.1

    _MicGuion.pos = 0
    assert stt._record_energy(400, 8.0, 6) == []


@pytest.mark.parametrize("guion", [
    "voz_desde_t0",      # habla al instante tras el bip: la calibración ve la voz
    "si_corto_en_t0",    # un "sí" de 0.24 s dentro de los 10 chunks de calibración
    "voz_tras_silencio",  # el caso normal
    "voz_bajita",        # mic con poca ganancia: RMS < 0.01, nunca enciende `speaking`
])
def test_energia_con_voz_no_pierde_la_orden(mic_energia, guion):
    """El "no hubo voz" no se decide con el umbral calibrado (que sube si la voz
    cae en la calibración) ni con el piso absoluto de 0.01 (una voz bajita pero
    muy por encima del ruido no lo cruza): lo que antes llegaba a Whisper con
    voz dentro sigue llegando."""
    stt, poner, llamadas = mic_energia
    if guion == "voz_desde_t0":
        poner(_voz(2.0), _ruido(10, 0.0005))
    elif guion == "si_corto_en_t0":
        poner(_ruido(0.03, 0.0005), _voz(0.24), _ruido(10, 0.0005))
    elif guion == "voz_bajita":
        poner(_ruido(0.5, 0.0005), _voz(2.0, amp=0.01), _ruido(10, 0.0005))
    else:
        poner(_ruido(0.5, 0.0005), _voz(2.0), _ruido(10, 0.0005))
    assert stt.listen_once(silence_ms=600, max_seconds=12.0, start_timeout_s=4.0,
                           min_audio_s=0.3) == "algo"
    assert len(llamadas) == 1 and llamadas[0] >= 0.3
