"""Arreglos de la auditoría en el oído de la voz (voice/adapters.py, interfaces.py, wake.py).

Cada bloque fija un hallazgo confirmado; todo con fakes (un `sounddevice` falso
en sys.modules), sin micrófono, PortAudio ni modelos.

- V17: _RealMic.close() es TERMINAL. VoiceLoop.stop() cierra el micrófono desde
  el hilo principal mientras el oído sigue en su bucle: un read() posterior no
  puede reabrir un InputStream que ya nadie cierra, un stream que se estaba
  abriendo durante el close no puede quedar huérfano, y el error de la lectura
  a la que le cerraron el stream bajo los pies no puede tumbar el hilo.
- V14: el Protocol SpeechToText declara la firma que el loop usa de verdad
  (transcribe(audio, hotwords=...)), y las clases reales se comparan por FIRMA,
  no solo por nombre de método como hace isinstance().
- voice-capture#5: la confirmación por voz de una tool delicada no acepta una
  pregunta ni el "va" de futuro ("¿se va a borrar todo?") como un sí.
- voice-capture#12 (V16): wake_score lee [wake].target de la config.
- voice-capture#2: strip_wake_word no se come palabras de la orden cuando el
  wake entra por coincidencia difusa ni mutila un followup sin wake.
- voice-capture#1: openWakeWord se resetea al disparar (sin turnos fantasma).
- voice-capture#18: [wake].oww_model admite ~, $VARS y rutas relativas al repo,
  y el error de carga muestra la causa real.
- voice-capture#11 / x-concurrency-lifecycle#25 (V10): feed() no repite en el
  log INFO el score que openWakeWord no recalculó (2-3 líneas iguales seguidas).
"""

from __future__ import annotations

import inspect
import queue
import sys
import threading
import types

import pytest

from crotolamo.voice import wake
from crotolamo.voice.adapters import _RealMic
from crotolamo.voice.interfaces import SpeechToText, TextToSpeech
from crotolamo.voice.normalize import normalize_text
from crotolamo.voice.state import SharedState
from crotolamo.voice.threads import EarThread

np = pytest.importorskip("numpy")


# --- V17: _RealMic.close() terminal -------------------------------------
class _FakeStream:
    """InputStream de mentira. Modela lo que hace PortAudio con un stream ya
    parado: read() lanza ("Stream is stopped"), no devuelve audio."""

    def __init__(self, streams: list, **_kwargs) -> None:
        self.started = False
        self.closed = False
        streams.append(self)

    def start(self) -> None:
        self.started = True

    def read(self, frames: int):
        if self.closed:
            raise RuntimeError("Stream is stopped [PaErrorCode -9983]")
        return np.zeros((frames, 1), dtype=np.float32), False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _fake_sounddevice(monkeypatch, factory) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=factory))


def _ear(mic, shutdown) -> EarThread:
    return EarThread(mic, lambda chunk: False, lambda chunk: 0.0, lambda frames: frames,
                     None, queue.Queue(), queue.Queue(), SharedState(), shutdown)


def test_realmic_read_tras_close_no_reabre_el_microfono(monkeypatch):
    streams: list[_FakeStream] = []
    _fake_sounddevice(monkeypatch, lambda **kw: _FakeStream(streams, **kw))
    mic = _RealMic()

    assert mic.read().shape == (512,)
    mic.close()

    assert mic.read() is None, "tras close() no hay audio: el oído ve None y sale"
    assert len(streams) == 1, "read() tras close() abrió OTRO InputStream"
    assert streams[0].closed


def test_stop_entre_el_chequeo_y_el_read_no_deja_un_stream_abierto(monkeypatch):
    """La carrera real de VoiceLoop.stop(): el oído ya pasó el `while not
    shutdown.is_set()` cuando el hilo principal hace shutdown.set() +
    mic.close(). Se fuerza ESE intercalado de forma determinista con un Event
    cuyo is_set() devuelve False y, acto seguido, corre el stop()."""
    streams: list[_FakeStream] = []
    _fake_sounddevice(monkeypatch, lambda **kw: _FakeStream(streams, **kw))
    mic = _RealMic()

    class _StopTrasElChequeo:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            if self.checks == 1:
                return False         # 1ª vuelta normal: el oído abre y lee
            if self.checks == 2:
                mic.close()          # stop() justo DESPUÉS del chequeo
                return False
            return True

    _ear(mic, _StopTrasElChequeo()).run()   # en este hilo: sin sleeps ni carreras

    assert len(streams) == 1, "el oído reabrió el micrófono después del stop()"
    assert all(s.closed for s in streams), "quedó un InputStream que nadie cierra"


def test_close_durante_la_lectura_bloqueante_no_tumba_el_oido(monkeypatch):
    """stop() cierra el stream mientras el oído está DENTRO de read(): la
    lectura revienta (así responde PortAudio a un stream parado). Con el mic
    ya cerrado eso es el apagado, no un fallo: el hilo debe salir limpio, sin
    excepción no manejada."""
    en_lectura = threading.Event()
    soltar = threading.Event()

    class _StreamBloqueante(_FakeStream):
        def read(self, frames: int):
            en_lectura.set()
            soltar.wait(5.0)         # bloqueado en PortAudio hasta el close()
            return super().read(frames)

        def close(self) -> None:
            super().close()
            soltar.set()

    streams: list[_FakeStream] = []
    _fake_sounddevice(monkeypatch, lambda **kw: _StreamBloqueante(streams, **kw))
    reventones: list = []
    monkeypatch.setattr(threading, "excepthook", reventones.append)
    mic = _RealMic()
    shutdown = threading.Event()
    ear = _ear(mic, shutdown)
    ear.start()
    try:
        assert en_lectura.wait(2.0)
        shutdown.set()               # lo mismo que VoiceLoop.stop()
        mic.close()
        ear.join(timeout=2.0)
    finally:
        shutdown.set()
        soltar.set()
    assert not ear.is_alive()
    assert not reventones, f"el oído murió con: {reventones[0].exc_value!r}"


def test_un_error_de_lectura_sin_close_se_sigue_propagando(monkeypatch):
    """El arreglo no esconde fallos reales del micrófono: solo se calla el
    error de la lectura que el propio close() provocó."""
    class _StreamRoto(_FakeStream):
        def read(self, frames: int):
            raise RuntimeError("Input overflowed de mentira")

    streams: list[_FakeStream] = []
    _fake_sounddevice(monkeypatch, lambda **kw: _StreamRoto(streams, **kw))
    with pytest.raises(RuntimeError, match="de mentira"):
        _RealMic().read()


def test_close_mientras_se_abre_el_stream_no_lo_deja_huerfano(monkeypatch):
    """close() llega cuando el primer read() está a medio abrir el dispositivo:
    todavía no hay stream publicado que close() pueda cerrar, así que quien lo
    abrió es quien debe cerrarlo."""
    streams: list[_FakeStream] = []
    mic = _RealMic()

    def _abre_y_le_cierran(**kwargs):
        stream = _FakeStream(streams, **kwargs)
        mic.close()                  # el stop() del otro hilo, justo aquí
        return stream

    _fake_sounddevice(monkeypatch, _abre_y_le_cierran)

    assert mic.read() is None
    assert len(streams) == 1
    assert streams[0].closed, "el stream abierto durante el close() quedó huérfano"


# --- V14: los Protocols de voz declaran la firma real --------------------
def _metodos(proto) -> list[str]:
    return [name for name, value in vars(proto).items()
            if inspect.isfunction(value) and not name.startswith("_")]


def test_protocolo_stt_admite_la_llamada_del_stt_thread():
    """SttThread llama transcribe(<ndarray o Path>, hotwords=...). Un backend
    escrito contra el Protocol debe poder recibir exactamente eso."""
    firma = inspect.signature(SpeechToText.transcribe)
    firma.bind(object(), object(), hotwords="Crotolamo")   # TypeError = costura rota


def _parametros(funcion) -> dict[str, inspect.Parameter]:
    params = dict(inspect.signature(funcion).parameters)
    params.pop("self", None)
    return params


@pytest.mark.parametrize("proto_name", ["SpeechToText", "TextToSpeech"])
def test_las_clases_reales_cumplen_la_firma_del_protocolo(proto_name, tmp_path):
    """isinstance() con runtime_checkable solo mira que el método EXISTA. Aquí
    se compara la firma: mismo nombre de parámetros (el loop pasa hotwords por
    nombre) y ningún parámetro obligatorio que el Protocol no declare."""
    from crotolamo.voice.stt import STT
    from crotolamo.voice.tts import TTS

    proto, real = {"SpeechToText": (SpeechToText, STT()),
                   "TextToSpeech": (TextToSpeech, TTS(tmp_path / "v.onnx"))}[proto_name]
    assert isinstance(real, proto)
    metodos = _metodos(proto)
    assert metodos, "el Protocol no declara métodos: el test no probaría nada"
    for name in metodos:
        declarado = _parametros(getattr(proto, name))
        concreto = _parametros(getattr(type(real), name))
        for pname, param in declarado.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                assert any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in concreto.values()), f"{name}: falta **kwargs"
                continue
            assert pname in concreto, (
                f"{proto_name}.{name} declara '{pname}' pero {type(real).__name__}."
                f"{name} recibe {list(concreto)}"
            )
            if param.default is not inspect.Parameter.empty:
                assert concreto[pname].default is not inspect.Parameter.empty, (
                    f"{name}: '{pname}' es opcional en el Protocol y obligatorio en la clase"
                )
        for pname, param in concreto.items():
            if pname in declarado or param.kind in (
                inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL
            ):
                continue
            assert param.default is not inspect.Parameter.empty, (
                f"{type(real).__name__}.{name} exige '{pname}', que el Protocol no declara"
            )


# --- voice-capture#5: una pregunta o el "va a" de futuro no confirman ------
# voice_confirm decide con confirmation_from_answer si corre una tool DELICADA
# (borrar, mover, sobrescribir). Su contrato: "cualquier respuesta que no
# contenga una confirmación clara se trata como no". Una pregunta no es un sí,
# y el "va" de "se va a borrar" es el verbo (ir a + infinitivo), no el "va" de
# "va que va". Las respuestas pasan por normalize_text, como en STT.transcribe.
@pytest.mark.parametrize("answer", [
    "¿Qué va a borrar?",
    "¿Se va a borrar todo?",
    "Espera, ¿cuál archivo va a mover?",
    "¿Y si mejor lo dejamos?",
    "¿Seguro que sí?",
    "¿Correcto?",
])
def test_una_pregunta_nunca_confirma(answer):
    assert wake.confirmation_from_answer(normalize_text(answer)) is False, answer


@pytest.mark.parametrize("answer", [
    # El modelo tiny del wake (el que transcribe la confirmación) a veces se
    # come los signos de interrogación: el "va a" de futuro sigue sin contar.
    "qué va a borrar",
    "se va a borrar todo",
    "cuál archivo va a mover",
])
def test_el_va_de_futuro_no_es_un_si(answer):
    assert wake.confirmation_from_answer(normalize_text(answer)) is False, answer


@pytest.mark.parametrize("answer", [
    "Va.", "va que va", "Va, hazlo.", "Sí, va a quedar bien.", "¡Sí, confirmo!",
    "Sí.", "bueno, dale", "correcto",
])
def test_las_confirmaciones_claras_siguen_confirmando(answer):
    assert wake.confirmation_from_answer(normalize_text(answer)) is True, answer


# --- voice-capture#12 (V16): [wake].target se lee de la config ----------
# El docstring del módulo promete "leer target/threshold/variants desde
# config", pero wake_score usaba siempre la constante WAKE_TARGET: la clave
# [wake].target de crotolamo.toml no hacía nada.
def _wake_config(monkeypatch, section: dict) -> None:
    from crotolamo import settings as settings_mod

    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "wake", section)
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)


def test_wake_score_usa_el_target_de_la_config(monkeypatch):
    _wake_config(monkeypatch, {"target": "jarvis", "variants": ["croto lamo"]})
    score, variant = wake.wake_score("jarvis")
    assert (score, variant) == (1.0, "jarvis")
    assert wake.is_wake_word("oye jarvis abre youtube", threshold=0.95)


def test_sin_target_en_la_config_se_usa_crotolamo(monkeypatch):
    _wake_config(monkeypatch, {"variants": ["jarvis"]})
    assert wake.wake_score("crotolamo") == (1.0, "crotolamo")
    _wake_config(monkeypatch, {"target": "", "variants": ["jarvis"]})
    assert wake.wake_score("crotolamo") == (1.0, "crotolamo")


# --- voice-capture#2: strip_wake_word no se come palabras de la orden ------
# El wake por coincidencia DIFUSA (Whisper lo transcribió de una forma que no
# está literal en [wake].variants) probaba los prefijos de 3, 2 y 1 palabras
# con is_wake_word, que puntúa por VENTANAS: cualquier prefijo de 3 palabras
# que CONTUVIERA algo parecido a "crotolamo" pasaba y se cortaban 3 palabras.
# Variantes y umbral como los de config/crotolamo.toml, fijados aquí para que
# el test pruebe el algoritmo aunque la config cambie.
_VARIANTES = [
    "crotolamo", "croto lamo", "gotolamo", "grotolamo", "crotoamo",
    "coto lamo", "coto y amo", "cottoy amo", "croto el amo", "crotolo amo",
]


@pytest.mark.parametrize("heard, command", [
    ("protolamo pon música", "pon música"),
    ("crotolama pausa la música", "pausa la música"),
    ("krotolamo abre la carpeta de descargas", "abre la carpeta de descargas"),
    ("Crotolano, apaga la luz del cuarto", "apaga la luz del cuarto"),
    ("autolamo pausa a la música", "pausa a la música"),
    ("proto lamo, abren la capita de descargas.", "abren la capita de descargas"),
    ("oye crotolamo abre youtube", "abre youtube"),
    ("hola crotolamo pon música", "pon música"),
])
def test_wake_difuso_deja_la_orden_entera(heard, command):
    assert wake.split_wake_command(heard, 0.72, _VARIANTES) == (True, command)


def test_wake_mal_oido_en_dos_palabras_pide_la_orden():
    """Solo el wake, partido en dos tokens: no hay orden. Antes la frase entera
    nunca contaba como wake (`len(words) <= n: continue`) y la orden era "llamo"."""
    assert wake.split_wake_command("caroto llamo", 0.72, _VARIANTES) == (True, "")


def test_la_cuenta_de_palabras_del_wake_literal_se_hace_sobre_el_original():
    """"Croto-lamo," es UNA palabra del original y DOS normalizadas ("croto
    lamo"): contar sobre lo normalizado y cortar el original se comía "pon"."""
    assert wake.split_wake_command("Croto-lamo, pon música", 0.72, _VARIANTES) == (
        True, "pon música")
    assert wake.strip_wake_word("- crotolamo pon música", _VARIANTES) == "pon música"


@pytest.mark.parametrize("followup", [
    "quiero otro lamento boliviano",
    "pero tómalo con calma",
    "pon otra canción",
])
def test_el_followup_sin_wake_no_se_mutila(followup):
    """En la ventana de followup (listener) strip_wake_word se aplica a frases
    SIN wake: una orden normal debe llegar intacta al agente."""
    assert wake.strip_wake_word(followup, _VARIANTES) == followup


# --- voice-capture#1: openWakeWord se resetea al disparar -----------------
# El modelo de openWakeWord guarda en su preprocessor las últimas features
# (~1.3 s de audio). Tras disparar, el oído deja de alimentarlo (LISTENING,
# THINKING, SPEAKING, gracia) y al volver a IDLE el primer chunk de SILENCIO
# todavía puntúa con el wake dentro: turno fantasma, y otro, y otro (con el
# modelo real, un solo "alexa" abría 29 turnos en 60 s de ruido de fondo).
class _PreprocSinReset:
    """Preprocessor de openwakeword 0.4/0.5: buffers de streaming, sin reset()."""

    def __init__(self) -> None:
        from collections import deque

        self.raw_data_buffer: deque = deque(maxlen=160000)
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.feature_buffer = np.zeros((16, 1))   # "embeddings" del audio en blanco
        self.resets = 0

    def __call__(self, x) -> None:
        self.raw_data_buffer.extend(x.tolist())
        self.accumulated_samples += len(x)
        fila = 1.0 if np.abs(x).max() > 1000 else 0.0   # 1.0 = "suena el wake"
        self.feature_buffer = np.vstack((self.feature_buffer, [[fila]]))[-120:]


class _PreprocConReset(_PreprocSinReset):
    """Preprocessor de openwakeword 0.6+: trae reset() y Model.reset() lo llama."""

    def reset(self) -> None:
        self.resets += 1
        self.raw_data_buffer.clear()
        self.accumulated_samples = 0
        self.feature_buffer = np.zeros((16, 1))


def _fake_oww(monkeypatch, preproc_cls, modelos: list) -> None:
    """openwakeword falso en sys.modules: _get_model() corre su camino real."""

    class _Model:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.preprocessor = preproc_cls()
            self.prediction_buffer: dict = {}
            modelos.append(self)

        def reset(self) -> None:
            self.prediction_buffer = {}
            if hasattr(self.preprocessor, "reset"):   # 0.6+
                self.preprocessor.reset()

        def predict(self, x) -> dict:
            self.preprocessor(x)
            # Como el modelo real: puntúa las ÚLTIMAS 16 features, no el chunk.
            return {"wake": float(self.preprocessor.feature_buffer[-16:].max())}

    paquete = types.ModuleType("openwakeword")
    paquete.__file__ = "/nonexistent/openwakeword/__init__.py"
    modelo = types.ModuleType("openwakeword.model")
    modelo.Model = _Model  # type: ignore[attr-defined]
    utils = types.ModuleType("openwakeword.utils")
    utils.download_models = lambda names: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", paquete)
    monkeypatch.setitem(sys.modules, "openwakeword.model", modelo)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils)


_WAKE = np.full(512, 20000, dtype=np.int16)
_SILENCIO = np.zeros(512, dtype=np.int16)


@pytest.mark.parametrize("preproc_cls", [_PreprocSinReset, _PreprocConReset])
def test_tras_disparar_el_silencio_no_vuelve_a_disparar(monkeypatch, preproc_cls):
    from crotolamo.voice.wakeword import WakeWordDetector

    modelos: list = []
    _fake_oww(monkeypatch, preproc_cls, modelos)
    det = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)

    assert det.feed(_SILENCIO) is False
    assert det.feed(_WAKE) is True
    # El oído vuelve a IDLE y solo le llega silencio: ningún disparo más.
    assert [det.feed(_SILENCIO) for _ in range(20)] == [False] * 20
    # Un segundo wake REAL sigue disparando.
    assert det.feed(_WAKE) is True
    assert len(modelos) == 1, "el reset no debe recargar el modelo"
    if preproc_cls is _PreprocConReset:
        assert modelos[0].preprocessor.resets == 2, "un reset por disparo, no dos"


def test_reset_deja_el_preprocessor_04_como_recien_cargado(monkeypatch):
    """openwakeword 0.4/0.5: Model.reset() solo vacía prediction_buffer. El
    detector restaura además los buffers del preprocessor desde una copia del
    estado en blanco (recalcularlo cuesta ~200 ms de embeddings)."""
    from crotolamo.voice.wakeword import WakeWordDetector

    modelos: list = []
    _fake_oww(monkeypatch, _PreprocSinReset, modelos)
    det = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)
    det._get_model()
    pre = modelos[0].preprocessor
    en_blanco = pre.feature_buffer.copy()

    det.score(_WAKE)
    pre.melspectrogram_buffer = np.zeros((80, 32))
    det.reset()

    assert len(pre.raw_data_buffer) == 0
    assert pre.accumulated_samples == 0
    assert np.array_equal(pre.melspectrogram_buffer, np.ones((76, 32)))
    assert np.array_equal(pre.feature_buffer, en_blanco)
    pre.feature_buffer[0, 0] = 9.0          # la copia guardada no se comparte
    det.reset()
    assert np.array_equal(pre.feature_buffer, en_blanco)


def test_reset_sin_modelo_cargado_ni_metodo_no_falla():
    """Los fakes de los tests (y un modelo aún sin cargar) no traen reset()."""
    from crotolamo.voice.wakeword import WakeWordDetector

    det = WakeWordDetector(model_name="fake")
    det.reset()                              # sin modelo: no hace nada

    class _SinReset:
        def predict(self, arr):
            return {"wake": 0.9}

    det._model = _SinReset()
    assert det.feed(_WAKE) is True           # dispara y el reset tolera el fake


def test_listen_for_wake_no_vuelve_a_disparar_con_el_wake_anterior(monkeypatch):
    """Modo simple con openWakeWord: tras un wake, la siguiente espera abre un
    stream nuevo; con el modelo sin resetear, devolvía True en el PRIMER frame
    de silencio (y el patrón oía "No te escuché claro" una y otra vez)."""
    from crotolamo.voice.wakeword import _FRAME, WakeWordDetector

    modelos: list = []
    _fake_oww(monkeypatch, _PreprocSinReset, modelos)
    audio = [np.full((_FRAME, 1), 20000, dtype=np.int16)]   # el wake real

    class _Stream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            pass

        def read(self, frames: int):
            block = audio.pop(0) if audio else np.zeros((frames, 1), dtype=np.int16)
            return block, False

    _fake_sounddevice(monkeypatch, _Stream)
    det = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)

    assert det.listen_for_wake(timeout_s=1.0) is True
    assert det.listen_for_wake(timeout_s=0.05) is False, "disparó con el wake anterior"


# --- voice-capture#18: [wake].oww_model con ~, $VARS o relativa al repo ----
# El resto de rutas de la config se expanden en settings.py; oww_model se usaba
# crudo: "~/modelos/x.onnx" no existía para os.path.isfile y una ruta relativa
# como la documentada (wakeword_training/my_custom_model/crotolamo.onnx) solo
# cargaba lanzando el proceso desde la raíz del repo.
def _modelo_propio(base, nombre: str = "mi_wake.onnx"):
    ruta = base / "modelos" / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(b"onnx")
    return ruta


def test_oww_model_expande_la_virgulilla_y_las_variables(monkeypatch, tmp_path):
    from crotolamo.voice.wakeword import WakeWordDetector

    ruta = _modelo_propio(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CROTO_MODELOS", str(tmp_path / "modelos"))
    monkeypatch.chdir(tmp_path.parent)
    for nombre in ("~/modelos/mi_wake.onnx", "$CROTO_MODELOS/mi_wake.onnx"):
        assert WakeWordDetector(model_name=nombre)._resolve_model_path() == str(ruta)


def test_oww_model_relativo_se_busca_tambien_desde_la_raiz_del_repo(monkeypatch, tmp_path):
    from crotolamo import settings as settings_mod
    from crotolamo.voice.wakeword import WakeWordDetector

    repo = tmp_path / "repo"
    ruta = _modelo_propio(repo)
    monkeypatch.setattr(settings_mod, "REPO_ROOT", repo)
    otra = tmp_path / "otra"
    otra.mkdir()
    monkeypatch.chdir(otra)                  # lanzado desde fuera del repo
    det = WakeWordDetector(model_name="modelos/mi_wake.onnx")
    assert det._resolve_model_path() == str(ruta)

    # Si la ruta existe desde el CWD, gana el CWD (lo que ya funcionaba no cambia).
    local = _modelo_propio(otra)
    assert det._resolve_model_path() == "modelos/mi_wake.onnx"
    assert local.is_file()


def test_oww_0_6_carga_el_modelo_propio_con_la_ruta_expandida(monkeypatch, tmp_path):
    """openwakeword 0.6+ solo acepta wakeword_models (nombre o ruta EXISTENTE):
    el primer intento le pasaba "~/..." crudo y no lo encontraba."""
    from crotolamo.voice.wakeword import WakeWordDetector

    ruta = _modelo_propio(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    cargados: list = []

    class _Model06:
        def __init__(self, wakeword_models=(), inference_framework="tflite") -> None:
            import os

            if not all(os.path.exists(m) for m in wakeword_models):
                raise ValueError(f"no existe el modelo {wakeword_models}")
            cargados.append(list(wakeword_models))

    _fake_oww(monkeypatch, _PreprocSinReset, [])
    monkeypatch.setattr(sys.modules["openwakeword.model"], "Model", _Model06)
    WakeWordDetector(model_name="~/modelos/mi_wake.onnx")._get_model()
    assert cargados == [[str(ruta)]]


def test_el_error_de_carga_reporta_la_causa_real_y_no_el_de_firma(monkeypatch):
    """Las 4 firmas se prueban en orden; las que no son de esta versión fallan
    con TypeError. Se mostraba el ÚLTIMO error (un TypeError de firma, que
    apunta a incompatibilidad de versión) y se escondía la causa real."""
    from crotolamo.voice.stt import VoiceUnavailable
    from crotolamo.voice.wakeword import WakeWordDetector

    class _Model04:
        def __init__(self, wakeword_model_paths=(), **kwargs) -> None:
            if kwargs:
                raise TypeError("AudioFeatures.__init__() got an unexpected keyword "
                                f"argument '{next(iter(kwargs))}'")
            raise RuntimeError("[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model "
                               f"from {wakeword_model_paths[0]} failed. File doesn't exist")

    _fake_oww(monkeypatch, _PreprocSinReset, [])
    monkeypatch.setattr(sys.modules["openwakeword.model"], "Model", _Model04)
    with pytest.raises(VoiceUnavailable) as info:
        WakeWordDetector(model_name="no_existe.onnx")._get_model()
    assert "NO_SUCHFILE" in str(info.value)
    assert "unexpected keyword" not in str(info.value)


# --- voice-capture#11 / x-concurrency-lifecycle#25 (V10): log del score ----
# Con los chunks de 512 del micrófono, openWakeWord solo calcula una predicción
# nueva cada 1280 muestras; entre medias devuelve el MISMO score (0.4 repite la
# inferencia sobre las mismas features, 0.6 devuelve prediction_buffer[-1]) y
# feed() escribía la misma línea INFO 2-3 veces seguidas en el journal.
class _ModeloCada1280:
    """Como openwakeword con chunks cortos: el score cambia cada 1280 muestras."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.acumuladas = 0
        self.i = 0

    def predict(self, x) -> dict:
        self.acumuladas += len(x)
        if self.acumuladas >= 1280:
            self.acumuladas = 0
            self.i = min(self.i + 1, len(self.scores) - 1)
        return {"wake": self.scores[self.i]}


def _lineas_score(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "crotolamo.voice.wakeword"
            and r.getMessage().startswith("wake score=")]


def test_el_mismo_score_repetido_se_loguea_una_sola_vez(caplog):
    import logging

    from crotolamo.voice.wakeword import WakeWordDetector

    caplog.set_level(logging.INFO, logger="crotolamo.voice.wakeword")
    det = WakeWordDetector(model_name="fake", threshold=0.5)
    det._model = _ModeloCada1280([0.2, 0.3, 0.02, 0.3])  # type: ignore[assignment]

    assert [det.feed(_SILENCIO) for _ in range(12)] == [False] * 12
    assert _lineas_score(caplog) == [
        "wake score=0.200 (umbral=0.50) -> no",
        "wake score=0.300 (umbral=0.50) -> no",
        # Predicción nueva tras una sin señal: vuelve a salir aunque repita valor.
        "wake score=0.300 (umbral=0.50) -> no",
    ]


def test_cada_disparo_y_cada_cambio_de_umbral_se_loguean(caplog):
    """Un disparo es un turno nuevo: su línea sale siempre. Y el mismo score con
    otro umbral (arrancó la música) es información nueva: también sale."""
    import logging

    from crotolamo.voice.wakeword import WakeWordDetector

    class _Media:
        playing = False

        def is_playing(self) -> bool:
            return self.playing

    caplog.set_level(logging.INFO, logger="crotolamo.voice.wakeword")
    media = _Media()
    det = WakeWordDetector(model_name="fake", threshold=0.5, threshold_media=0.7,
                           media=media)  # type: ignore[arg-type]
    det._model = _ModeloCada1280([0.6])  # type: ignore[assignment]  # sin reset()

    assert det.feed(_SILENCIO) is True
    assert det.feed(_SILENCIO) is True
    media.playing = True
    assert det.feed(_SILENCIO) is False
    assert det.feed(_SILENCIO) is False
    assert _lineas_score(caplog) == [
        "wake score=0.600 (umbral=0.50) -> DISPARA",
        "wake score=0.600 (umbral=0.50) -> DISPARA",
        "wake score=0.600 (umbral=0.70) -> no",
    ]
