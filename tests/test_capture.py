"""Captura de audio en la frontera del dispositivo (sin hardware).

Paso 0 (caracterización): qué hacen HOY los grabadores que abren su propio
sd.InputStream por llamada — STT._record_silero, STT._record_energy,
STT.listen_smart y WakeWordDetector.listen_for_wake — contra un sounddevice
falso con audio guionizado y un contador de aperturas. Ningún test de la suite
abría un dispositivo (ni falso): la forma "un stream por grabación" y sus
huecos solo se oían en el portátil. Estos tests describen el camino por llamada,
que se queda tal cual como el de siempre (source=None), y deben pasar sin
cambios tras los pasos siguientes.

Paso 1: capture.MicSession, el único dueño del micrófono en una sesión de
escucha (cola acotada, re-encuadre, backoff, rendición, cierre sin cerrar bajo
un read()). Paso 2: los mismos grabadores leyendo de una fuente ya abierta
(source=...), comparados con el camino por llamada sobre el mismo audio.
Paso 3: el loop simple dueño de UNA sesión ([voice].audio_session, apagada de
fábrica) y cómo la arma run_listen.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types

import pytest

np = pytest.importorskip("numpy")

from crotolamo.voice import capture  # noqa: E402
from crotolamo.voice import stt as stt_mod  # noqa: E402
from crotolamo.voice import wakeword as ww  # noqa: E402

SR = 16000
SILERO = 512   # chunk de Silero (32 ms)
ENERGIA = 480  # chunk del VAD por energía (30 ms)
OWW = 1280     # frame de openWakeWord (80 ms)

VOZ = 0.3        # amplitud de "voz" (Silero falso: media |x| > 0.02)
SALA = 0.001     # ruido de sala estable
WAKE = 0.5       # amplitud que el openWakeWord falso reconoce como el wake


# --- mundo guionizado ---------------------------------------------------------

class _Mundo:
    """sounddevice falso: UNA línea de tiempo de audio que los streams leen en
    orden (lo que un stream no lee no lo lee nadie). Cuenta aperturas, cierres
    y lecturas, y guarda con qué argumentos se abrió cada stream."""

    def __init__(self, audio) -> None:
        self.audio = np.asarray(audio, dtype=np.float32)
        self.pos = 0
        self.abiertos: list[dict] = []
        self.cerrados = 0
        self.lecturas: list[int] = []
        self.modulo = types.ModuleType("sounddevice")
        self.modulo.InputStream = self._abrir  # type: ignore[attr-defined]

    def _abrir(self, **kwargs):
        self.abiertos.append(kwargs)
        return _Stream(self, kwargs.get("dtype", "float32"))

    def tomar(self, n: int):
        bloque = self.audio[self.pos:self.pos + n]
        self.pos += n
        if len(bloque) < n:
            bloque = np.concatenate([bloque, np.zeros(n - len(bloque), dtype=np.float32)])
        self.lecturas.append(n)
        return bloque


class _Stream:
    def __init__(self, mundo: _Mundo, dtype: str) -> None:
        self.mundo = mundo
        self.dtype = dtype
        self.cerrado = False

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.cerrado = True
        self.mundo.cerrados += 1

    def read(self, n: int):
        assert not self.cerrado, "read() sobre un stream cerrado"
        bloque = self.mundo.tomar(n)
        if self.dtype == "int16":
            bloque = (bloque * 32767).astype(np.int16)
        return bloque.reshape(-1, 1), False


def _linea(*partes: tuple[float, int], chunk: int):
    """Línea de tiempo de `chunk`s de amplitud constante: (amplitud, nº chunks)."""
    return np.concatenate([np.full(n * chunk, amp, dtype=np.float32) for amp, n in partes])


class _SileroFalso:
    """Silero falso con la misma exigencia que el real: 512 muestras por chunk."""

    def __init__(self) -> None:
        self.resets = 0
        self.llamadas = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, x, sr) -> float:
        x = np.asarray(x)
        assert x.shape == (SILERO,) and sr == SR, x.shape
        self.llamadas += 1
        return 0.97 if float(np.mean(np.abs(x))) > 0.02 else 0.01


class _WhisperFalso:
    """Whisper falso: devuelve los textos del guion en orden y anota cuánto
    audio recibió en cada llamada."""

    def __init__(self, textos=()) -> None:
        self.textos = list(textos)
        self.audios: list[int] = []

    def transcribe(self, audio, **_kwargs):
        self.audios.append(len(audio))
        texto = self.textos.pop(0) if self.textos else ""
        segs = [types.SimpleNamespace(text=texto, no_speech_prob=0.0, avg_logprob=-0.1)]
        return (segs if texto else []), None


VOICE = {"vad_backend": "silero", "vad_threshold": 0.8, "vad_preactivation_ms": 800,
         "input_device": "mic_x", "vad_silence_ms": 400}


@pytest.fixture
def grabador(monkeypatch):
    """STT real con sounddevice, Silero y Whisper falsos (inyectados por las
    cachés que ya tiene stt: _silero_vad y _models)."""
    silero = _SileroFalso()
    whisper = _WhisperFalso()
    monkeypatch.setattr(stt_mod, "_silero_vad", silero)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    monkeypatch.setitem(stt_mod._models, "fake", whisper)
    voice = dict(VOICE)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: voice)

    def _con(audio) -> _Mundo:
        mundo = _Mundo(audio)
        monkeypatch.setitem(sys.modules, "sounddevice", mundo.modulo)
        return mundo

    stt = stt_mod.STT(model_size="fake", initial_prompt=None)
    return types.SimpleNamespace(stt=stt, silero=silero, whisper=whisper, voice=voice,
                                 con=_con)


def _junto(frames):
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


# --- paso 0: _record_silero por llamada ---------------------------------------
# silence_ms=400, max 8 s, start_timeout 6 s con chunks de 32 ms:
# 12 chunks de silencio cierran, 250 de tope, 187 de espera, 25 de pre-activación.

def test_silero_antepone_la_preactivacion_y_cierra_tras_el_silencio(grabador):
    audio = _linea((SALA, 40), (VOZ, 30), (SALA, 100), chunk=SILERO)
    mundo = grabador.con(audio)
    frames = grabador.stt._record_silero(400, 8.0, 6, grabador.voice)
    # La voz empieza en el chunk 40: los 25 de pre-activación (16..40) van
    # delante; tras la voz (hasta el 69) 12 de silencio la cierran (70..81).
    assert np.array_equal(_junto(frames), audio[16 * SILERO:82 * SILERO])
    assert all(len(f) == SILERO for f in frames)
    assert len(mundo.lecturas) == 82 and set(mundo.lecturas) == {SILERO}
    assert grabador.silero.resets == 1


def test_silero_sin_voz_devuelve_vacio_al_agotar_la_espera(grabador):
    mundo = grabador.con(_linea((SALA, 400), chunk=SILERO))
    assert grabador.stt._record_silero(400, 8.0, 6, grabador.voice) == []
    assert len(mundo.lecturas) == 188  # chunks 0..187: corta al 187 sin voz


def test_silero_corta_en_el_tope_de_grabacion(grabador):
    audio = _linea((VOZ, 400), chunk=SILERO)
    mundo = grabador.con(audio)
    frames = grabador.stt._record_silero(400, 8.0, 6, grabador.voice)
    assert len(mundo.lecturas) == 250
    assert np.array_equal(_junto(frames), audio[:250 * SILERO])


def test_silero_abre_y_cierra_un_stream_por_llamada(grabador):
    mundo = grabador.con(_linea((SALA, 10), (VOZ, 10), (SALA, 2000), chunk=SILERO))
    for _ in range(3):
        grabador.stt._record_silero(400, 8.0, 6, grabador.voice)
    assert mundo.abiertos == [{"samplerate": SR, "channels": 1, "dtype": "float32",
                               "device": "mic_x"}] * 3
    assert mundo.cerrados == 3
    assert grabador.silero.resets == 3


# --- paso 0: _record_energy por llamada ---------------------------------------
# Chunks de 30 ms: 13 de silencio cierran, 266 de tope, 200 de espera, 10 de
# calibración. Sin pre-activación: devuelve todo desde el primer chunk.

def test_energia_devuelve_desde_el_principio_hasta_el_silencio(grabador):
    audio = _linea((SALA, 40), (0.1, 30), (SALA, 100), chunk=ENERGIA)
    mundo = grabador.con(audio)
    frames = grabador.stt._record_energy(400, 8.0, 6)
    assert np.array_equal(_junto(frames), audio[:83 * ENERGIA])
    assert len(mundo.lecturas) == 83 and set(mundo.lecturas) == {ENERGIA}
    assert mundo.abiertos == [{"samplerate": SR, "channels": 1, "dtype": "float32",
                               "device": "mic_x"}]
    assert mundo.cerrados == 1


def test_energia_sin_voz_devuelve_vacio(grabador):
    mundo = grabador.con(_linea((SALA, 400), chunk=ENERGIA))
    assert grabador.stt._record_energy(400, 8.0, 6) == []
    assert len(mundo.lecturas) == 201


def test_record_frames_elige_backend_por_config(grabador):
    mundo = grabador.con(_linea((SALA, 400), chunk=SILERO))
    grabador.stt._record_frames(None, 8.0, 6)
    grabador.voice["vad_backend"] = "energy"
    grabador.stt._record_frames(None, 8.0, 6)
    assert mundo.lecturas[0] == SILERO and mundo.lecturas[-1] == ENERGIA
    assert len(mundo.abiertos) == 2 and mundo.cerrados == 2


# --- paso 0: listen_smart ------------------------------------------------------

def test_listen_smart_abre_un_stream_mas_por_ronda(grabador):
    audio = _linea((SALA, 10), (VOZ, 20), (SALA, 20), (VOZ, 20), (SALA, 200),
                   chunk=SILERO)
    mundo = grabador.con(audio)
    grabador.whisper.textos = ["mueve el archivo de", "huevonitis a descargas"]
    texto = grabador.stt.listen_smart(silence_ms=400, start_timeout_s=4.0, rounds=2,
                                      continue_timeout_s=2.5, min_audio_s=0.3)
    assert texto == "mueve el archivo de huevonitis a descargas"
    assert len(mundo.abiertos) == 2 and mundo.cerrados == 2
    assert len(grabador.whisper.audios) == 2


def test_listen_smart_con_frase_completa_abre_uno(grabador):
    mundo = grabador.con(_linea((SALA, 10), (VOZ, 20), (SALA, 200), chunk=SILERO))
    grabador.whisper.textos = ["abre youtube"]
    assert grabador.stt.listen_smart(silence_ms=400, rounds=2) == "abre youtube"
    assert len(mundo.abiertos) == 1


def test_listen_once_sin_voz_no_llama_a_whisper(grabador):
    mundo = grabador.con(_linea((SALA, 400), chunk=SILERO))
    assert grabador.stt.listen_once(silence_ms=400, start_timeout_s=4.0,
                                    min_audio_s=0.3) == ""
    assert grabador.whisper.audios == [] and len(mundo.abiertos) == 1


# --- paso 0: listen_for_wake por llamada --------------------------------------

class _OwwFalso:
    """openWakeWord falso: dispara en el frame que trae la amplitud del wake.
    Guarda cada frame que recibe (tipo y tamaño incluidos)."""

    def __init__(self) -> None:
        self.frames: list = []

    def predict(self, arr):
        self.frames.append(np.array(arr, copy=True))
        return {"wake": 0.9 if int(np.max(np.abs(arr))) >= int(WAKE * 32767) - 1 else 0.0}


@pytest.fixture
def detector(monkeypatch):
    """WakeWordDetector real con modelo falso; time.monotonic avanza con el
    audio leído (80 ms por frame), así el timeout es determinista."""
    from crotolamo import settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_settings",
                        lambda: types.SimpleNamespace(voice={"input_device": "mic_x"}))
    det = ww.WakeWordDetector(model_name="fake", threshold=0.5)
    modelo = _OwwFalso()
    det._model = modelo  # type: ignore[assignment]
    det._np = np

    def _con(audio) -> _Mundo:
        mundo = _Mundo(audio)
        monkeypatch.setitem(sys.modules, "sounddevice", mundo.modulo)
        monkeypatch.setattr(ww, "time", types.SimpleNamespace(
            monotonic=lambda: 1000.0 + sum(mundo.lecturas) / SR))
        return mundo

    return types.SimpleNamespace(det=det, modelo=modelo, con=_con)


def test_listen_for_wake_dispara_en_el_frame_del_wake(detector):
    audio = _linea((SALA, 7), (WAKE, 1), (SALA, 20), chunk=OWW)
    mundo = detector.con(audio)
    assert detector.det.listen_for_wake(timeout_s=None) is True
    # Frames de 1280 en int16, uno por lectura, hasta el que dispara (el 8º).
    assert len(detector.modelo.frames) == 8 and mundo.lecturas == [OWW] * 8
    for i, frame in enumerate(detector.modelo.frames):
        assert frame.dtype == np.int16 and frame.shape == (OWW,)
        esperado = (audio[i * OWW:(i + 1) * OWW] * 32767).astype(np.int16)
        assert np.array_equal(frame, esperado)
    assert mundo.abiertos == [{"samplerate": SR, "channels": 1, "dtype": "int16",
                               "device": "mic_x"}]
    assert mundo.cerrados == 1


def test_listen_for_wake_devuelve_false_al_agotar_el_timeout(detector):
    mundo = detector.con(_linea((SALA, 100), chunk=OWW))
    assert detector.det.listen_for_wake(timeout_s=1.0) is False
    # Mira el reloj antes de cada frame: 0.00, 0.08 ... 0.96 s -> 13 frames.
    assert len(detector.modelo.frames) == 13
    assert len(mundo.abiertos) == 1 and mundo.cerrados == 1


# --- paso 1: MicSession, el único dueño del micrófono -------------------------
# Un sounddevice falso "vivo": entrega el guion tal cual y, al acabarse, o sigue
# con silencio (como una sala callada) o se CUELGA dentro de read() hasta que el
# test lo suelte (un driver atascado: el límite B1 de PortAudio).


class _PortAudioError(Exception):
    pass


class _Micro:
    def __init__(self, audio=None, *, tras: str = "silencio", fallos_lectura=(),
                 fallos_apertura=()) -> None:
        self.audio = np.zeros(0, np.float32) if audio is None else np.asarray(
            audio, dtype=np.float32)
        self.pos = 0
        self.tras = tras
        self.soltar = threading.Event()
        # Guion de las próximas lecturas / aperturas: un mensaje = esa falla con
        # él; None = esa va bien. Agotado, todas van bien.
        self.fallos_lectura = list(fallos_lectura)
        self.fallos_apertura = list(fallos_apertura)
        self.abiertos: list[dict] = []
        self.cierres = 0
        self.leyendo = 0
        self.cerrado_leyendo = False
        self.lock = threading.Lock()
        self.modulo = types.ModuleType("sounddevice")
        self.modulo.InputStream = self._abrir  # type: ignore[attr-defined]

    def _abrir(self, **kwargs):
        # Cada apertura consume una entrada: un mensaje la hace fallar, None no.
        motivo = self.fallos_apertura.pop(0) if self.fallos_apertura else None
        if motivo is not None:
            raise _PortAudioError(motivo)
        self.abiertos.append(kwargs)
        return _StreamVivo(self)


class _StreamVivo:
    def __init__(self, micro: _Micro) -> None:
        self.micro = micro
        self.iniciado = False
        self.cerrado = False

    def start(self) -> None:
        self.iniciado = True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        m = self.micro
        with m.lock:
            if m.leyendo:
                m.cerrado_leyendo = True
            if not self.cerrado:
                self.cerrado = True
                m.cierres += 1

    def read(self, n: int):
        m = self.micro
        assert self.iniciado and not self.cerrado
        with m.lock:
            m.leyendo += 1
        try:
            motivo = m.fallos_lectura.pop(0) if m.fallos_lectura else None
            if motivo is not None:
                raise _PortAudioError(motivo)
            if m.pos < len(m.audio):
                bloque = m.audio[m.pos:m.pos + n]
                m.pos += n
                if len(bloque) < n:
                    bloque = np.concatenate([bloque, np.zeros(n - len(bloque), np.float32)])
            elif m.tras == "cuelga":
                m.soltar.wait(10)
                bloque = np.zeros(n, np.float32)
            else:
                time.sleep(0.004)  # silencio de sala, algo más rápido que el real
                bloque = np.zeros(n, np.float32)
            return bloque.reshape(-1, 1).copy(), False
        finally:
            with m.lock:
                m.leyendo -= 1


@pytest.fixture
def micro(monkeypatch):
    """Instala un _Micro como sounddevice; cierra las sesiones al acabar."""
    sesiones: list = []
    micros: list[_Micro] = []

    def _con(audio=None, **kwargs) -> _Micro:
        m = _Micro(audio, **kwargs)
        micros.append(m)
        monkeypatch.setitem(sys.modules, "sounddevice", m.modulo)
        return m

    def _sesion(**kwargs) -> capture.MicSession:
        s = capture.MicSession(SR, "mic_x", **kwargs)
        sesiones.append(s)
        return s

    yield types.SimpleNamespace(con=_con, sesion=_sesion)
    for m in micros:
        m.soltar.set()
    for s in sesiones:
        s.close()


def _leer_todo(sesion, n: int, frames: int | None = None):
    bloques = []
    for _ in range(n):
        bloque = sesion.read(timeout=2.0, frames=frames)
        assert bloque is not None, "la sesión dejó de entregar audio"
        bloques.append(bloque)
    return bloques


def test_sesion_abre_una_vez_y_entrega_en_orden(micro):
    audio = np.arange(40 * SILERO, dtype=np.float32) / (40 * SILERO)
    m = micro.con(audio, tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    bloques = _leer_todo(sesion, 40)
    assert all(b.dtype == np.float32 and b.shape == (SILERO,) for b in bloques)
    assert np.array_equal(np.concatenate(bloques), audio)
    assert m.abiertos == [{"samplerate": SR, "channels": 1, "dtype": "float32",
                           "device": "mic_x"}]
    assert sesion.opens == 1


def test_sesion_reencuadra_a_cualquier_tamano(micro):
    """Los grabadores piden su propio bloque (480 la energía, 1280 el wake): la
    sesión parte y junta sus chunks de 512 sin perder ni repetir muestras."""
    audio = np.arange(50 * SILERO, dtype=np.float32)
    micro.con(audio, tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    a = _leer_todo(sesion, 3, frames=OWW)
    b = _leer_todo(sesion, 7, frames=ENERGIA)
    c = _leer_todo(sesion, 4)
    leido = np.concatenate(a + b + c)
    assert [len(x) for x in a + b + c] == [OWW] * 3 + [ENERGIA] * 7 + [SILERO] * 4
    assert np.array_equal(leido, audio[:len(leido)])


@pytest.mark.parametrize("frames", [SILERO, OWW])
def test_quien_espera_despierta_una_vez_por_bloque(micro, frames):
    """El lector solo despierta a quien espera cuando ya tiene su bloque entero.
    Antes avisaba en cada chunk de 512: pidiendo 1280 (openWakeWord en --simple
    con la sesión) eran 2,5 despertares por bloque, 1,5 de ellos en balde, ~19
    de más por segundo en reposo."""
    micro.con()  # silencio de sala a ritmo (casi) real: un chunk cada 4 ms
    sesion = micro.sesion()
    esperas = [0]
    esperar = sesion._cond.wait

    def _contar(timeout=None):
        esperas[0] += 1
        return esperar(timeout)

    sesion._cond.wait = _contar  # type: ignore[method-assign]
    sesion.start()
    sesion.read(timeout=2.0, frames=frames)  # arranque: el primer bloque no cuenta
    esperas[0] = 0
    bloques = 40
    _leer_todo(sesion, bloques, frames=frames)
    assert esperas[0] <= bloques * 1.2, f"{esperas[0] / bloques:.2f} esperas por bloque"


def test_pending_cuenta_lo_capturado_sin_consumir(micro):
    micro.con(np.ones(10 * SILERO, np.float32), tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    _esperar(lambda: sesion.pending() == 10)
    assert sesion.pending(ENERGIA) == 10 * SILERO // ENERGIA
    sesion.read(frames=100)
    assert sesion.pending() == (10 * SILERO - 100) // SILERO


def test_read_devuelve_none_si_no_llega_audio(micro):
    micro.con(np.ones(2 * SILERO, np.float32), tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    _leer_todo(sesion, 2)
    t0 = time.monotonic()
    assert sesion.read(timeout=0.05) is None
    assert time.monotonic() - t0 < 1.0


def test_la_cola_es_acotada_y_tira_lo_mas_viejo(micro):
    audio = np.arange(30 * SILERO, dtype=np.float32)
    micro.con(audio, tras="cuelga")
    sesion = micro.sesion(backlog_s=10 * SILERO / SR)  # caben 10 chunks
    sesion.start()
    _esperar(lambda: sesion.dropped == 20)
    assert sesion.pending() == 10
    assert np.array_equal(np.concatenate(_leer_todo(sesion, 10)), audio[20 * SILERO:])


def test_flush_tira_lo_pendiente(micro):
    micro.con(np.ones(10 * SILERO, np.float32), tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    _esperar(lambda: sesion.pending() == 10)
    assert sesion.flush() == 10 * SILERO
    assert sesion.pending() == 0 and sesion.read(timeout=0.02) is None


def test_start_que_no_abre_lanza_y_no_deja_hilo(micro):
    micro.con(fallos_apertura=["Error opening InputStream: device unavailable"])
    sesion = micro.sesion()
    antes = {t.name for t in threading.enumerate()}
    with pytest.raises(_PortAudioError):
        sesion.start()
    assert "MicSession" not in {t.name for t in threading.enumerate()} - antes
    sesion.close()  # cerrar una sesión que no arrancó no hace nada


def test_close_espera_al_lector_antes_de_cerrar_el_stream(micro):
    """B1: cerrar el stream con otro hilo dentro de read() es lo que hacía
    VoiceLoop.stop(). Aquí lo cierra el lector al salir de read()."""
    m = micro.con(np.ones(5 * SILERO, np.float32))  # luego silencio vivo
    sesion = micro.sesion()
    sesion.start()
    _leer_todo(sesion, 5)
    sesion.close()
    assert not m.cerrado_leyendo
    assert m.cierres == 1
    assert sesion._hilo is not None and not sesion._hilo.is_alive()
    assert sesion.read(timeout=0.01) is None


def test_close_con_el_driver_colgado_no_cierra_bajo_el_read(micro):
    m = micro.con(np.ones(SILERO, np.float32), tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    _leer_todo(sesion, 1)
    _esperar(lambda: m.leyendo == 1)
    t0 = time.monotonic()
    sesion.close()
    assert time.monotonic() - t0 < 2.0       # espera acotada (0.5 s)
    assert m.cierres == 0                    # no lo cerró con el read dentro
    m.soltar.set()                           # el driver vuelve...
    _esperar(lambda: m.cierres == 1)         # ...y el lector lo cierra al salir
    assert not m.cerrado_leyendo


def test_close_despierta_a_quien_espera_audio(micro):
    micro.con(tras="cuelga")
    sesion = micro.sesion()
    sesion.start()
    resultado: list = []
    lector = threading.Thread(target=lambda: resultado.append(sesion.read(timeout=5.0)))
    lector.start()
    time.sleep(0.05)
    sesion.close()  # con el driver todavía colgado
    lector.join(2.0)
    assert not lector.is_alive() and resultado == [None]


def _sin_esperas(sesion, reloj=None):
    """Sustituye la espera del backoff: la anota (y avanza el reloj falso)."""
    esperas: list[float] = []

    def _pausa(segundos: float) -> bool:
        esperas.append(segundos)
        if reloj is not None:
            reloj[0] += segundos
        return sesion._stop.is_set()

    sesion._pausa = _pausa  # type: ignore[method-assign]
    return esperas


def test_tras_un_fallo_reabre_con_backoff_y_un_aviso_por_mensaje(micro, caplog):
    audio = np.arange(8 * SILERO, dtype=np.float32)
    # La 1ª lectura falla; las 3 reaperturas siguientes también ("device lost").
    m = micro.con(audio, tras="cuelga", fallos_lectura=["Input overflowed"],
                  fallos_apertura=[None] + ["device lost"] * 3)
    sesion = micro.sesion()
    esperas = _sin_esperas(sesion)
    with caplog.at_level(logging.INFO, logger="crotolamo.voice.capture"):
        sesion.start()
        bloques = _leer_todo(sesion, 8)
    assert np.array_equal(np.concatenate(bloques), audio)   # nada se perdió del guion
    assert esperas == [1.0, 2.0, 4.0, 8.0]
    assert len(m.abiertos) == 2                             # la de start + la que volvió
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 2, avisos                         # uno por mensaje distinto
    assert any("Input overflowed" in a for a in avisos)
    assert any("device lost" in a for a in avisos)
    assert any("de vuelta" in r.getMessage() for r in caplog.records
               if r.levelno == logging.INFO)
    assert not sesion.failed


def test_el_backoff_se_topa_en_30_s_y_se_reinicia_al_volver(micro):
    micro.con(np.ones(4 * SILERO, np.float32), tras="cuelga",
              fallos_lectura=["x", None, None, "z"],
              fallos_apertura=[None] + ["y"] * 7)
    sesion = micro.sesion()
    esperas = _sin_esperas(sesion)
    sesion.start()
    _leer_todo(sesion, 4)
    assert esperas == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 1.0]
    # (la última: el primer fallo de la racha nueva, "z", vuelve a esperar 1 s)


def test_se_rinde_tras_give_up_s_de_fallos_seguidos(micro, monkeypatch, caplog):
    micro.con(np.ones(SILERO, np.float32), tras="cuelga", fallos_lectura=["device lost"],
              fallos_apertura=[None] + ["device unavailable"] * 100)
    reloj = [0.0]
    monkeypatch.setattr(capture, "time", types.SimpleNamespace(
        monotonic=lambda: reloj[0]))
    sesion = micro.sesion(give_up_s=90.0)
    esperas = _sin_esperas(sesion, reloj)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.capture"):
        sesion.start()
        _esperar(lambda: sesion.failed)
        with pytest.raises(capture.MicFailed, match="device unavailable"):
            sesion.read(timeout=0.01)
    # 1+2+4+8+16+30+30 = 91 s >= 90: el 8º fallo se rinde, sin otra espera.
    assert esperas == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert [r.levelno for r in caplog.records].count(logging.ERROR) == 1
    _esperar(lambda: sesion._hilo is not None and not sesion._hilo.is_alive())


class _ActivaQueAvisa(threading.Event):
    """El _activa de la sesión, que avisa cuando el lector se queda esperando en él."""

    def __init__(self) -> None:
        super().__init__()
        self.set()
        self.esperando = threading.Event()

    def wait(self, timeout=None):
        if not self.is_set():
            self.esperando.set()
        return super().wait(timeout)


def test_la_pausa_del_panel_no_cuenta_para_rendirse(micro, monkeypatch, caplog):
    """give_up_s cuenta fallos con la escucha ACTIVA. El micro falla, el patrón
    pausa desde el panel (en plena espera del backoff) y reanuda 1000 s después
    con el micro aún caído: antes la pausa contaba como fallo y la sesión se
    rendía al primer reintento tras reanudar (el listener salía con 1 sin los
    90 s de reintentos que promete la clave). Ahora cuentan los 3 s de fallos de
    antes de la pausa más los de después, sin el tramo de backoff ya en pausa."""
    micro.con(np.ones(SILERO, np.float32), tras="cuelga", fallos_lectura=["device lost"],
              fallos_apertura=[None] + ["device unavailable"] * 100)
    reloj = [0.0]
    monkeypatch.setattr(capture, "time", types.SimpleNamespace(
        monotonic=lambda: reloj[0]))
    sesion = micro.sesion(give_up_s=90.0)
    activa = _ActivaQueAvisa()
    sesion._activa = activa
    esperas: list[float] = []

    def _pausa(segundos: float) -> bool:
        esperas.append(segundos)
        if len(esperas) == 3:
            sesion.suspend()  # el patrón pausa al empezar la 3ª espera (t = 3 s)
        reloj[0] += segundos
        return sesion._stop.is_set()

    sesion._pausa = _pausa  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.capture"):
        sesion.start()
        assert activa.esperando.wait(3.0), "el lector no se quedó en pausa"
        reloj[0] += 1000.0  # en pausa mucho más que give_up_s
        sesion.resume()     # y el micro sigue caído
        _esperar(lambda: sesion.failed)
    # Fallando y activa: 3 s antes de la pausa + 0, 8, 24, 54, 84, 114 tras ella
    # -> se rinde con 117 s >= 90. Antes se rendía nada más reanudar: [1, 2, 4].
    assert esperas == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    assert [r.levelno for r in caplog.records].count(logging.ERROR) == 1
    # Un aviso por mensaje distinto en todo el incidente, pausa incluida.
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 2


def test_la_pausa_con_suspend_en_cada_vuelta_tampoco_cuenta_para_rendirse(micro, monkeypatch,
                                                                         caplog):
    """Lo mismo, pero como lo hace el loop simple: mic.suspend() en CADA vuelta
    de la pausa (una cada 0,3 s), no una vez. Antes cada suspend() movía el
    inicio de la pausa: al reanudar solo se descontaban los últimos 0,3 s y la
    sesión se rendía al primer reintento ([1, 2, 4], 'me rindo' a los 1000 s)."""
    micro.con(np.ones(SILERO, np.float32), tras="cuelga", fallos_lectura=["device lost"],
              fallos_apertura=[None] + ["device unavailable"] * 100)
    reloj = [0.0]
    monkeypatch.setattr(capture, "time", types.SimpleNamespace(
        monotonic=lambda: reloj[0]))
    sesion = micro.sesion(give_up_s=90.0)
    activa = _ActivaQueAvisa()
    sesion._activa = activa
    esperas: list[float] = []

    def _pausa(segundos: float) -> bool:
        esperas.append(segundos)
        if len(esperas) == 3:
            sesion.suspend()  # primera vuelta de la pausa, en la 3ª espera (t = 3 s)
        reloj[0] += segundos
        return sesion._stop.is_set()

    sesion._pausa = _pausa  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.capture"):
        sesion.start()
        assert activa.esperando.wait(3.0), "el lector no se quedó en pausa"
        while reloj[0] < 1000.0:  # el resto de la pausa: una vuelta del loop cada 0,3 s
            reloj[0] += 0.3
            sesion.suspend()
        sesion.resume()     # y el micro sigue caído
        _esperar(lambda: sesion.failed)
    assert esperas == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    assert [r.levelno for r in caplog.records].count(logging.ERROR) == 1


def test_el_loop_simple_marca_la_pausa_desde_su_primera_vuelta(loop_simple, micro,
                                                               monkeypatch):
    """El loop simple de verdad sobre una sesión de verdad: una pausa del panel de
    10 vueltas (0,3 s cada una) llama 10 veces a suspend(), y la pausa que la
    sesión descuenta de give_up_s empieza en la PRIMERA (antes, en la última)."""
    m = micro.con()
    sesion = micro.sesion()
    sesion.start()
    _habla_tras_cada_flush(sesion, m, [None, _wake_y_orden()])
    reloj = [100.0]
    monkeypatch.setattr(capture, "time", types.SimpleNamespace(monotonic=lambda: reloj[0]))
    llamadas: list[float] = []
    suspender = sesion.suspend

    def _suspend() -> None:
        llamadas.append(reloj[0])
        suspender()

    sesion.suspend = _suspend  # type: ignore[method-assign]
    loop_simple.panel["on"] = False

    def _en_pausa(segundos: float) -> None:
        reloj[0] += segundos
        if len(llamadas) >= 10:
            loop_simple.panel["on"] = True

    loop_simple.panel["al_dormir"] = _en_pausa
    rc, agente = loop_simple.correr(tiny=_WhisperQueGuarda(["crotolamo"]),
                                    base=_WhisperQueGuarda(["pausa la música"]), mic=sesion)
    sesion.close()
    assert rc == 0 and agente.commands == ["pausa la música"]
    assert len(llamadas) == 10
    assert sesion._pausada_en == llamadas[0] == 100.0


def test_give_up_cero_no_se_rinde_nunca(micro):
    micro.con(np.ones(SILERO, np.float32), tras="cuelga", fallos_lectura=["x"],
              fallos_apertura=[None] + ["y"] * 40)
    sesion = micro.sesion(give_up_s=0.0)
    esperas = _sin_esperas(sesion)
    sesion.start()
    _leer_todo(sesion, 1)
    assert len(esperas) == 41 and not sesion.failed


def test_suspend_cierra_el_dispositivo_y_resume_lo_reabre(micro):
    m = micro.con(np.ones(4 * SILERO, np.float32))
    sesion = micro.sesion()
    sesion.start()
    _leer_todo(sesion, 2)
    sesion.suspend()
    _esperar(lambda: m.cierres == 1)
    assert sesion.pending() == 0 and sesion.read(timeout=0.05) is None
    sesion.resume()
    _leer_todo(sesion, 2)
    assert len(m.abiertos) == 2


def _esperar(condicion, limite: float = 3.0) -> None:
    fin = time.monotonic() + limite
    while not condicion():
        assert time.monotonic() < fin, "la condición no se cumplió a tiempo"
        time.sleep(0.002)


def test_el_lector_no_muere_en_silencio(micro, monkeypatch, caplog):
    """Algo que revienta FUERA de la lectura del dispositivo (un bug) no deja la
    sesión muda para siempre: se registra y read() lo dice con MicFailed."""
    micro.con(np.ones(4 * SILERO, np.float32), tras="cuelga")
    sesion = micro.sesion()

    def _roto(bloque):
        raise ValueError("bug")

    monkeypatch.setattr(sesion, "_guardar", _roto)
    with caplog.at_level(logging.ERROR, logger="crotolamo.voice.capture"):
        sesion.start()
        _esperar(lambda: sesion.failed)
    with pytest.raises(capture.MicFailed, match="bug"):
        sesion.read(timeout=0.01)
    assert any("se cayó" in r.getMessage() for r in caplog.records)


# --- paso 2: los grabadores leen de una fuente ya abierta ---------------------
# Con source=None todo es lo de arriba (paso 0). Con una fuente, MISMOS frames
# para el mismo audio: se compara el camino por llamada con el de la fuente
# sobre la misma línea de tiempo, llamada a llamada. La fuente de la comparación
# entrega al paso de quien lee (nada capturado de antemano, como un stream que
# se abre en la llamada); que MicSession entrega exactamente esa secuencia lo
# fijan los tests de arriba, y lo que cambia con audio ya capturado
# (los plazos) tiene su propio test.

class _FuenteGuion:
    """Fuente determinista sin hilos (el contrato de capture.MicSession):
    `pendiente` muestras ya capturadas al empezar; las lecturas cuyo índice
    está en `huecos` no traen nada (None, como un segundo sin audio); agotado el
    guion, silencio digital (como el _Mundo). Puede lanzar en una lectura dada
    (`rompe`)."""

    def __init__(self, audio, pendiente: int = 0, huecos=(), rompe=None) -> None:
        self.audio = np.asarray(audio, dtype=np.float32)
        self.pos = 0
        self.capturado = pendiente
        self.huecos = set(huecos)
        self.rompe = rompe or {}
        self.lecturas: list[int] = []

    def pending(self, frames: int = SILERO) -> int:
        return max(0, self.capturado - self.pos) // frames

    def read(self, timeout: float = 1.0, frames: int = SILERO):
        k = len(self.lecturas)
        self.lecturas.append(frames)
        if k in self.rompe:
            raise self.rompe[k]
        if k in self.huecos:
            return None
        bloque = self.audio[self.pos:self.pos + frames]
        if len(bloque) < frames:
            bloque = np.concatenate([bloque, np.zeros(frames - len(bloque), np.float32)])
        self.pos += frames
        self.capturado = max(self.capturado, self.pos)
        return bloque


def _sesion_sobre(micro, audio) -> capture.MicSession:
    micro.con(audio)  # tras el guion, silencio vivo (como el stream propio)
    sesion = micro.sesion(backlog_s=120.0)
    sesion.start()
    return sesion


_SILERO_GUIONES = {
    "orden": ((SALA, 40), (VOZ, 30), (SALA, 100)),
    "dos_ordenes": ((SALA, 30), (VOZ, 20), (SALA, 15), (VOZ, 25), (SALA, 200)),
    "sin_voz": ((SALA, 400),),
    "tope": ((VOZ, 300), (SALA, 100)),
    "voz_al_abrir": ((VOZ, 10), (SALA, 50)),
    "micropausas": ((SALA, 5), (VOZ, 8), (SALA, 6), (VOZ, 8), (SALA, 11), (VOZ, 4),
                    (SALA, 100)),
}


@pytest.mark.parametrize("guion", sorted(_SILERO_GUIONES))
def test_silero_con_fuente_da_los_mismos_frames(grabador, guion):
    audio = _linea(*_SILERO_GUIONES[guion], chunk=SILERO)
    mundo = grabador.con(audio)
    por_llamada = [grabador.stt._record_silero(400, 8.0, 6, grabador.voice)
                   for _ in range(3)]
    fuente = _FuenteGuion(audio)
    con_fuente = [grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente)
                  for _ in range(3)]
    for a, b in zip(por_llamada, con_fuente):
        assert len(a) == len(b)
        assert np.array_equal(_junto(a), _junto(b))
    assert fuente.lecturas == mundo.lecturas   # mismas lecturas, una a una


_ENERGIA_GUIONES = {
    "orden": ((SALA, 40), (0.1, 30), (SALA, 100)),
    "sin_voz": ((SALA, 400),),
    "voz_en_calibracion": ((0.1, 8), (SALA, 300)),
    "voz_bajita": ((0.0005, 40), (0.004, 20), (0.0005, 300)),
}


@pytest.mark.parametrize("guion", sorted(_ENERGIA_GUIONES))
def test_energia_con_fuente_da_los_mismos_frames(grabador, guion):
    audio = _linea(*_ENERGIA_GUIONES[guion], chunk=ENERGIA)
    mundo = grabador.con(audio)
    por_llamada = [grabador.stt._record_energy(400, 8.0, 6) for _ in range(2)]
    fuente = _FuenteGuion(audio)
    con_fuente = [grabador.stt._record_energy(400, 8.0, 6, source=fuente) for _ in range(2)]
    for a, b in zip(por_llamada, con_fuente):
        assert [len(f) for f in a] == [len(f) for f in b]
        assert np.array_equal(_junto(a), _junto(b))
    assert fuente.lecturas == mundo.lecturas


def test_varias_grabaciones_sobre_una_sesion_abren_un_solo_dispositivo(grabador, micro):
    audio = _linea((SALA, 10), (VOZ, 10), (SALA, 30), (VOZ, 10), (SALA, 300), chunk=SILERO)
    sesion = _sesion_sobre(micro, audio)
    primera = grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=sesion)
    segunda = grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=sesion)
    # Cada una con su voz y su pre-activación; la segunda sigue donde acabó la
    # primera (chunk 32), sin hueco entre ambas.
    assert np.array_equal(_junto(primera), audio[:32 * SILERO])
    assert np.array_equal(_junto(segunda), audio[32 * SILERO:72 * SILERO])
    assert sesion.opens == 1


class _Compuerta:
    """Pre-filtro del wake falso (rearm/feed): dispara en el chunk `en`."""

    def __init__(self, en: int) -> None:
        self.en = en
        self.vistos = 0
        self.rearmes = 0

    def rearm(self) -> None:
        self.rearmes += 1

    def feed(self, _bloque) -> bool:
        self.vistos += 1
        return self.vistos - 1 == self.en


@pytest.mark.parametrize("backend", ["silero", "energy"])
def test_con_fuente_la_compuerta_del_wake_recorta_igual(grabador, backend):
    chunk = SILERO if backend == "silero" else ENERGIA
    audio = _linea((SALA, 30), (VOZ, 60), (SALA, 100), chunk=chunk)

    def _grabar(**kwargs):
        if backend == "silero":
            return grabador.stt._record_silero(400, 8.0, 6, grabador.voice,
                                               gate=_Compuerta(70), **kwargs)
        return grabador.stt._record_energy(400, 8.0, 6, gate=_Compuerta(70), **kwargs)

    grabador.con(audio)
    a = _grabar()
    b = _grabar(source=_FuenteGuion(audio))
    assert a and np.array_equal(_junto(a), _junto(b))


def test_listen_for_wake_con_fuente_da_los_mismos_frames(detector, micro):
    audio = _linea((SALA, 7), (WAKE, 1), (SALA, 20), chunk=OWW)
    detector.con(audio)
    assert detector.det.listen_for_wake(timeout_s=None) is True
    por_llamada = list(detector.modelo.frames)
    detector.modelo.frames.clear()

    sesion = _sesion_sobre(micro, audio)
    assert detector.det.listen_for_wake(source=sesion) is True
    assert len(detector.modelo.frames) == len(por_llamada) == 8
    for a, b in zip(por_llamada, detector.modelo.frames):
        assert b.dtype == np.int16 and np.array_equal(a, b)
    # Lo que sigue al frame que disparó se queda en la sesión para la orden.
    siguiente = sesion.read(timeout=2.0)
    assert np.array_equal(siguiente, audio[8 * OWW:8 * OWW + SILERO])


def test_listen_for_wake_con_fuente_mira_should_stop_cada_segundo(detector):
    fuente = _FuenteGuion(_linea((SALA, 100), chunk=OWW), huecos={3})
    detector.con(np.zeros(0, np.float32))   # reloj: avanza con lo que lee la fuente
    consultas: list[int] = []

    def _parar() -> bool:
        consultas.append(len(fuente.lecturas))
        return len(fuente.lecturas) >= 4

    assert detector.det.listen_for_wake(source=fuente, should_stop=_parar) is False
    # Un hueco (1 s sin audio) es un tick: se consulta y para ahí.
    assert consultas == [4] and len(detector.modelo.frames) == 3


def test_listen_for_wake_con_fuente_should_stop_con_audio_continuo(detector, monkeypatch):
    """Con el micrófono vivo read() nunca devuelve None: el tick es de 1 s de
    reloj, no de huecos."""
    fuente = _FuenteGuion(_linea((SALA, 100), chunk=OWW))
    reloj = [0.0]

    def _monotonic() -> float:
        reloj[0] += 0.08   # cada consulta del reloj, un frame de tiempo
        return reloj[0]

    monkeypatch.setattr(ww, "time", types.SimpleNamespace(monotonic=_monotonic))
    llamadas = [0]

    def _parar() -> bool:
        llamadas[0] += 1
        return llamadas[0] >= 2

    assert detector.det.listen_for_wake(source=fuente, should_stop=_parar) is False
    # 0.08 s de reloj por frame: un tick cada ~12.5 frames; el 2º dice parar.
    assert llamadas[0] == 2
    assert len(detector.modelo.frames) == 25


def test_listen_for_wake_con_fuente_respeta_el_timeout(detector):
    fuente = _FuenteGuion(_linea((SALA, 100), chunk=OWW))
    mundo = detector.con(np.zeros(0, np.float32))
    mundo.lecturas = fuente.lecturas  # el reloj del fixture avanza con la fuente
    assert detector.det.listen_for_wake(timeout_s=1.0, source=fuente) is False
    assert len(detector.modelo.frames) == 13   # como con el stream propio


def test_listen_for_wake_con_fuente_propaga_el_fallo_del_micro(detector):
    fuente = _FuenteGuion(_linea((SALA, 100), chunk=OWW),
                          rompe={2: capture.MicFailed("no volvió")})
    detector.con(np.zeros(0, np.float32))
    with pytest.raises(capture.MicFailed):
        detector.det.listen_for_wake(source=fuente)


def test_los_plazos_cuentan_desde_la_llamada(grabador):
    """Lo ya capturado (p. ej. mientras Whisper decodificaba) se lee primero y
    no se come la espera ni el tope: como si el stream se abriera ahora."""
    audio = _linea((SALA, 1000), chunk=SILERO)
    fuente = _FuenteGuion(audio, pendiente=50 * SILERO)
    assert grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente) == []
    assert len(fuente.lecturas) == 50 + 188

    fuente = _FuenteGuion(_linea((VOZ, 1000), chunk=SILERO), pendiente=50 * SILERO)
    frames = grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente)
    assert len(frames) == 50 + 250

    fuente = _FuenteGuion(_linea((SALA, 1000), chunk=ENERGIA), pendiente=40 * ENERGIA)
    assert grabador.stt._record_energy(400, 8.0, 6, source=fuente) == []
    assert len(fuente.lecturas) == 40 + 201


def test_on_idle_true_sigue_esperando_con_la_preactivacion(grabador):
    """Agotada la espera sin voz, on_idle()=True sigue en la misma fuente como
    una llamada nueva (Silero en blanco, plazos desde ahí) SIN perder la
    pre-activación: la voz que empieza justo tras el reinicio trae sus 800 ms."""
    # Espera de 1 s (31 chunks): la voz empieza 3 chunks después del reinicio.
    audio = _linea((SALA, 35), (VOZ, 20), (SALA, 50), chunk=SILERO)
    fuente = _FuenteGuion(audio)
    vueltas: list[int] = []

    def _seguir() -> bool:
        vueltas.append(len(fuente.lecturas))
        return True

    frames = grabador.stt._record_silero(400, 8.0, 1.0, grabador.voice, source=fuente,
                                         on_idle=_seguir)
    assert vueltas == [32]                           # una espera agotada (chunks 0..31)
    assert grabador.silero.resets == 2               # la de la llamada + la del reinicio
    assert np.array_equal(_junto(frames), audio[11 * SILERO:(55 + 12) * SILERO])


@pytest.mark.parametrize("on_idle", [None, lambda: False])
def test_sin_on_idle_la_espera_agotada_devuelve_vacio(grabador, on_idle):
    fuente = _FuenteGuion(_linea((SALA, 35), (VOZ, 20), (SALA, 50), chunk=SILERO))
    assert grabador.stt._record_silero(400, 8.0, 1.0, grabador.voice, source=fuente,
                                       on_idle=on_idle) == []
    assert len(fuente.lecturas) == 32


def test_energia_on_idle_true_recalibra_como_llamada_nueva(grabador):
    audio = _linea((SALA, 40), (0.1, 20), (SALA, 50), chunk=ENERGIA)
    fuente = _FuenteGuion(audio)
    frames = grabador.stt._record_energy(400, 8.0, 0.6, source=fuente, on_idle=lambda: True)
    # Espera de 20 chunks: corta en el 20 y reinicia en el 21 (frames desde ahí).
    assert np.array_equal(_junto(frames), audio[21 * ENERGIA:(60 + 13) * ENERGIA])


def test_un_segundo_sin_audio_pregunta_a_on_idle(grabador):
    audio = _linea((SALA, 5), (VOZ, 20), (SALA, 50), chunk=SILERO)
    preguntas: list[int] = []

    def _seguir() -> bool:
        preguntas.append(len(fuente.lecturas))
        return True

    fuente = _FuenteGuion(audio, huecos={3, 4})
    frames = grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente,
                                         on_idle=_seguir)
    assert preguntas == [4, 5]      # un hueco = una pregunta; con True se sigue
    assert np.array_equal(_junto(frames), audio[:(25 + 12) * SILERO])


def test_sin_audio_y_sin_on_idle_la_grabacion_acaba_ahi(grabador):
    # A media frase: se queda con lo grabado (el dispositivo se calló).
    audio = _linea((SALA, 5), (VOZ, 20), (SALA, 50), chunk=SILERO)
    fuente = _FuenteGuion(audio, huecos={15})
    frames = grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente)
    assert np.array_equal(_junto(frames), audio[:15 * SILERO])
    # Sin voz todavía: [] como al agotar la espera.
    fuente = _FuenteGuion(audio, huecos={2})
    assert grabador.stt._record_silero(400, 8.0, 6, grabador.voice, source=fuente) == []


def test_con_fuente_un_fallo_del_micro_no_es_un_fallo_del_vad(grabador, caplog):
    """Antes un error del DISPOSITIVO dentro de la grabación con Silero se
    registraba como 'VAD silero falló' y se reabría el micrófono con el VAD por
    energía. Con fuente, el fallo del micrófono sube tal cual, sin reintento."""
    audio = _linea((SALA, 5), (VOZ, 20), (SALA, 50), chunk=SILERO)
    fuente = _FuenteGuion(audio, rompe={7: capture.MicFailed("el micrófono no volvió")})
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.stt"):
        with pytest.raises(capture.MicFailed):
            grabador.stt._record_frames(None, 8.0, 6, source=fuente)
    assert not [r for r in caplog.records if "VAD" in r.getMessage()]
    assert fuente.lecturas == [SILERO] * 8      # nada más: no se reintentó con energía


def test_con_fuente_un_fallo_del_modelo_sigue_cayendo_a_energia(grabador, caplog):
    """Un fallo DEL VAD (el modelo) cae a energía como siempre, ahora sobre la
    misma fuente (480 por bloque): no hay dispositivo que reabrir."""
    def _roto(*_a):
        raise RuntimeError("estado roto")

    grabador.silero.reset_states = _roto  # type: ignore[method-assign]
    audio = _linea((SALA, 40), (0.1, 30), (SALA, 100), chunk=ENERGIA)
    fuente = _FuenteGuion(audio)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        frames = grabador.stt._record_frames(400, 8.0, 6, source=fuente)
    assert np.array_equal(_junto(frames), audio[:83 * ENERGIA])
    assert set(fuente.lecturas) == {ENERGIA}
    assert [r.getMessage() for r in caplog.records] == [
        "VAD silero falló (estado roto); uso energía"]


def test_sin_fuente_el_fallo_del_micro_sigue_como_siempre(grabador, monkeypatch, caplog):
    """source=None no cambia: el error al abrir cae a energía con su aviso de
    siempre (el arreglo de ese aviso llega con la sesión, no aquí)."""
    def _no_abre(**_kw):
        raise OSError("device unavailable")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_no_abre))
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.stt"):
        with pytest.raises(OSError):
            grabador.stt._record_frames(None, 8.0, 6)
    assert [r.getMessage() for r in caplog.records] == [
        "VAD silero falló (device unavailable); uso energía"]


def test_listen_smart_con_fuente_no_pierde_el_arranque_de_la_continuacion(grabador, micro):
    """La continuación de smart_endpoint lee de la MISMA fuente: lo que se dijo
    mientras Whisper decodificaba la primera parte no se pierde y no se abre
    ningún stream más (por llamada eran dos, con el hueco del decode en medio)."""
    audio = _linea((SALA, 10), (VOZ, 20), (SALA, 20), (VOZ, 20), (SALA, 200),
                   chunk=SILERO)
    grabador.whisper.textos = ["mueve el archivo de", "huevonitis a descargas"]
    sesion = _sesion_sobre(micro, audio)
    texto = grabador.stt.listen_smart(silence_ms=400, start_timeout_s=4.0, rounds=2,
                                      continue_timeout_s=2.5, min_audio_s=0.3,
                                      source=sesion)
    assert texto == "mueve el archivo de huevonitis a descargas"
    assert sesion.opens == 1
    # 1ª ronda: chunks 0..41 (pre-activación de 10, voz, 12 de silencio). La 2ª
    # sigue en el 42, sin hueco: 8 de silencio + voz + 12 de silencio.
    assert grabador.whisper.audios == [(10 + 20 + 12) * SILERO, (8 + 20 + 12) * SILERO]


def test_listen_once_pasa_la_fuente(grabador):
    fuente = _FuenteGuion(_linea((SALA, 10), (VOZ, 20), (SALA, 200), chunk=SILERO))
    grabador.whisper.textos = ["hola"]
    assert grabador.stt.listen_once(silence_ms=400, source=fuente, min_audio_s=0.3) == "hola"
    assert fuente.lecturas and set(fuente.lecturas) == {SILERO}


def test_una_sesion_cerrada_no_hace_girar_a_quien_espera(grabador, detector, micro):
    """Cerrada, read() vuelve al instante con None: on_idle=True o un wake sin
    timeout girarían en vacío. Se acaba la grabación (o la espera) ahí."""
    sesion = _sesion_sobre(micro, _linea((SALA, 4), chunk=SILERO))
    sesion.close()
    preguntas: list[int] = []

    def _seguir() -> bool:
        preguntas.append(1)
        return len(preguntas) < 1000

    assert grabador.stt._record_frames(None, 8.0, 6, source=sesion, on_idle=_seguir) == []
    assert preguntas == []
    assert detector.det.listen_for_wake(source=sesion) is False


# --- paso 3: el loop simple es dueño de UNA sesión ----------------------------
# _run_simple_loop con los STT reales (Silero y Whisper falsos) sobre el
# micrófono falso. mic=None es el loop de siempre (un stream por grabación);
# con mic=MicSession todas las grabaciones leen de la misma sesión, y lo que
# antes no oía nadie (Whisper decodificando el wake) llega a la orden.

ORDEN = 0.25   # amplitud de la orden (voz para el Silero falso, distinta del wake)
_DECODE = 8    # chunks que "dura" decodificar el wake: el stream por llamada no los oye


class _WhisperQueGuarda:
    """Whisper falso que devuelve los textos del guion y guarda cada clip.
    al_transcribir(): lo que pasa en el mundo mientras decodifica."""

    def __init__(self, textos=(), al_transcribir=None) -> None:
        self.textos = list(textos)
        self.clips: list = []
        self.al_transcribir = al_transcribir

    def transcribe(self, audio, **_kwargs):
        self.clips.append(np.array(audio, copy=True))
        if self.al_transcribir is not None:
            self.al_transcribir()
        texto = self.textos.pop(0) if self.textos else ""
        segs = [types.SimpleNamespace(text=texto, no_speech_prob=0.0, avg_logprob=-0.1)]
        return (segs if texto else []), None


def _con_voz(clip) -> int:
    """Muestras con voz de un clip ya normalizado (pico 0.9; la sala queda en ~0.004)."""
    return int(np.sum(np.abs(clip) > 0.1))


class _TtsDelLoop:
    def __init__(self, al_hablar=None) -> None:
        self.dichos: list[str] = []
        self.al_hablar = al_hablar

    def speak(self, text, **_kwargs) -> None:
        self.dichos.append(text)
        if self.al_hablar is not None:
            self.al_hablar(text)

    def speak_sentences(self, text) -> None:
        self.speak(text)

    def beep(self) -> None:
        self.dichos.append("<bip>")

    def stop(self) -> None:
        pass


class _AgenteDelLoop:
    """Atiende las órdenes; tras `turnos` el siguiente vistazo al panel es el Ctrl-C."""

    def __init__(self, panel: dict, turnos: int = 1) -> None:
        self.panel = panel
        self.turnos = turnos
        self.commands: list[str] = []

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        if len(self.commands) >= self.turnos:
            self.panel["fin"] = True
        return "hecho, patrón."


@pytest.fixture
def loop_simple(monkeypatch):
    """Lo que necesita _run_simple_loop con STT reales: Silero falso, un Whisper
    falso por modelo ('tiny' del wake, 'base' de la orden), la config de voz, el
    flag del panel (panel['on']; con panel['fin'] el siguiente vistazo es un
    Ctrl-C) y time.sleep sin esperas."""
    from crotolamo.voice.state import SharedState
    from interfaces import listener

    silero = _SileroFalso()
    monkeypatch.setattr(stt_mod, "_silero_vad", silero)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    voice = dict(VOICE)
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: voice)
    panel: dict = {"on": True, "fin": False, "vistazos": 0}

    def _panel(*_a, **_k) -> bool:
        panel["vistazos"] += 1
        # Tope de seguridad: un test que falla no se queda esperando para siempre.
        if panel["fin"] or panel["vistazos"] > 60:
            raise KeyboardInterrupt
        return panel["on"]

    dormidas: list[float] = []

    def _dormir(s: float) -> None:
        dormidas.append(s)
        al_dormir = panel.get("al_dormir")
        if al_dormir is not None:
            al_dormir(s)

    monkeypatch.setattr(listener, "read_control_enabled", _panel)
    monkeypatch.setattr(listener, "time", types.SimpleNamespace(sleep=_dormir, time=time.time))

    def _correr(*, tiny, base, mic=None, tts=None, turnos=1, say=None, **cfg_kw):
        monkeypatch.setitem(stt_mod._models, "tiny_loop", tiny)
        monkeypatch.setitem(stt_mod._models, "base_loop", base)
        cfg = dict(threshold=0.72, variants=["crotolamo"], followup_s=0, ack="off",
                   stream_speak=False, smart_endpoint=False, silence_ms=400,
                   wake_silence_ms=400, wake_max_s=8.0)
        cfg.update(cfg_kw)
        agente = _AgenteDelLoop(panel, turnos)
        extra = {} if mic is None else {"mic": mic}
        rc = listener._run_simple_loop(
            agent=agente, stt=stt_mod.STT(model_size="base_loop"),
            wake_stt=stt_mod.STT(model_size="tiny_loop", initial_prompt=None),
            tts=tts or _TtsDelLoop(), wake_detector=None, use_oww=False,
            cfg=listener.ListenerConfig(**cfg), hud_state=SharedState(),
            say=say or (lambda text: None), **extra,
        )
        return rc, agente

    return types.SimpleNamespace(correr=_correr, panel=panel, silero=silero,
                                 dormidas=dormidas)


def _wake_y_orden(tv: int = 0):
    """Sala, `tv` frases que no despiertan, el wake (que cierra con 12 chunks de
    silencio, wake_silence_ms=400) y la orden JUSTO detrás: 20 chunks de voz."""
    partes = [(SALA, 20)]
    for _ in range(tv):
        partes += [(VOZ, 10), (SALA, 14)]
    partes += [(VOZ, 15), (SALA, 12), (ORDEN, 20), (SALA, 60)]
    return _linea(*partes, chunk=SILERO)


def test_loop_simple_por_llamada_pierde_lo_que_suena_mientras_decodifica(loop_simple,
                                                                         monkeypatch):
    """Caracterización del camino de siempre (mic=None): el wake cierra su
    stream, Whisper decodifica y la orden abre otro; lo dicho entre medias no lo
    oye nadie (bench: 'la música.' en vez de 'pausa la música')."""
    mundo = _Mundo(_wake_y_orden())
    monkeypatch.setitem(sys.modules, "sounddevice", mundo.modulo)

    def _decodificando() -> None:
        mundo.pos += _DECODE * SILERO

    tiny = _WhisperQueGuarda(["crotolamo"], al_transcribir=_decodificando)
    base = _WhisperQueGuarda(["la música"])
    rc, agente = loop_simple.correr(tiny=tiny, base=base)
    assert rc == 0 and agente.commands == ["la música"]
    assert _con_voz(base.clips[0]) == (20 - _DECODE) * SILERO
    assert len(mundo.abiertos) == 2 and mundo.cerrados == 2   # wake + orden


def _habla_tras_cada_flush(sesion, m: _Micro, guiones: list) -> None:
    """Lo que dice el patrón, en orden, justo después de cada flush() de la sesión
    (None = nada). Así el guion no depende de cuándo arranca el loop: lo que la
    sesión capturó antes de una espera nueva se tira, como debe."""
    tirar = sesion.flush

    def _flush() -> int:
        tiradas = tirar()
        guion = guiones.pop(0) if guiones else None
        if guion is not None:
            m.audio = np.concatenate([m.audio, guion])
        return tiradas

    sesion.flush = _flush  # type: ignore[method-assign]


def test_loop_simple_con_sesion_oye_la_orden_entera_con_un_solo_stream(loop_simple, micro):
    """Con la sesión, la orden empezada mientras Whisper decodificaba el wake
    llega entera, y las ventanas del wake (dos frases de la tele y el wake) y la
    orden leen del MISMO dispositivo: una apertura en todo el ciclo."""
    m = micro.con()
    sesion = micro.sesion()
    sesion.start()
    _habla_tras_cada_flush(sesion, m, [_wake_y_orden(tv=2)])
    tiny = _WhisperQueGuarda(["hola qué tal", "pues sí", "crotolamo"])
    base = _WhisperQueGuarda(["pausa la música"])
    rc, agente = loop_simple.correr(tiny=tiny, base=base, mic=sesion)
    assert rc == 0 and agente.commands == ["pausa la música"]
    assert len(tiny.clips) == 3
    assert _con_voz(base.clips[0]) == 20 * SILERO       # ni un chunk perdido
    assert len(m.abiertos) == 1 and sesion.opens == 1 and m.cierres == 0


def test_loop_simple_con_sesion_tira_lo_que_sono_mientras_respondia(loop_simple, micro):
    """El micrófono ya no se cierra mientras Crotolamo habla: lo que capturó en
    ese rato (su propia voz) y en la guarda de 0.5 s se tira antes de la ventana
    de followup, como cuando el stream estaba cerrado. Sin eso, el followup
    transcribía el eco de la respuesta."""
    m = micro.con()
    sesion = micro.sesion()
    sesion.start()
    _habla_tras_cada_flush(sesion, m, [_wake_y_orden()])

    def _eco(_texto: str) -> None:
        # Lo que oye el micrófono mientras suena la respuesta; la sala detrás
        # deja todo el eco en la cola al leerse el último chunk.
        m.audio = np.concatenate([m.audio, _linea((VOZ, 30), (SALA, 4), chunk=SILERO)])
        _esperar(lambda: m.pos >= len(m.audio))

    tiny = _WhisperQueGuarda(["crotolamo"])
    base = _WhisperQueGuarda(["pausa la música", "eco de la respuesta"])
    rc, agente = loop_simple.correr(tiny=tiny, base=base, mic=sesion,
                                    tts=_TtsDelLoop(al_hablar=_eco), followup_s=2.0)
    assert rc == 0 and agente.commands == ["pausa la música"]
    assert len(base.clips) == 1, "el followup transcribió el eco de la respuesta"
    assert 0.5 in loop_simple.dormidas                  # la guarda de siempre sigue


def test_loop_simple_con_sesion_cierra_el_micro_en_pausa(loop_simple, micro):
    """Pausa del panel con sesión: el dispositivo se cierra mientras dura (como
    sin sesión, donde en pausa no se abría nada) y se reabre al reanudar."""
    m = micro.con()
    sesion = micro.sesion()
    sesion.start()
    # El flush de suspend() no trae nada; tras reanudar, el wake y la orden.
    _habla_tras_cada_flush(sesion, m, [None, _wake_y_orden()])
    loop_simple.panel["on"] = False

    def _en_pausa(_s: float) -> None:
        _esperar(lambda: m.cierres == 1)       # la sesión soltó el dispositivo
        loop_simple.panel["on"] = True

    loop_simple.panel["al_dormir"] = _en_pausa
    tiny = _WhisperQueGuarda(["crotolamo"])
    base = _WhisperQueGuarda(["pausa la música"])
    rc, agente = loop_simple.correr(tiny=tiny, base=base, mic=sesion)
    assert rc == 0 and agente.commands == ["pausa la música"]
    assert sesion.opens == 2 and len(m.abiertos) == 2


def test_loop_simple_con_sesion_y_acuse_de_voz_no_se_oye_a_si_mismo(loop_simple, micro):
    """ack='voz': "Te escucho, patrón" suena antes de grabar la orden. Con el
    stream por llamada eso no se grababa; con la sesión se tira al terminar."""
    m = micro.con()
    sesion = micro.sesion()
    sesion.start()
    # Tras el flush de la espera, el wake; tras el del acuse, la orden.
    _habla_tras_cada_flush(sesion, m, [
        _linea((SALA, 20), (VOZ, 15), (SALA, 52), chunk=SILERO),
        _linea((ORDEN, 20), (SALA, 60), chunk=SILERO),
    ])

    def _say(texto: str) -> None:
        if texto == "Te escucho, patrón.":
            # Lo que oye el micrófono mientras suena el acuse.
            m.audio = np.concatenate([m.audio, _linea((VOZ, 25), (SALA, 4), chunk=SILERO)])
            _esperar(lambda: m.pos >= len(m.audio))

    tiny = _WhisperQueGuarda(["crotolamo"])
    base = _WhisperQueGuarda(["pausa la música", "te escucho patrón"])
    rc, agente = loop_simple.correr(tiny=tiny, base=base, mic=sesion, ack="voz", say=_say)
    assert rc == 0 and agente.commands == ["pausa la música"]
    assert _con_voz(base.clips[0]) == 20 * SILERO


class _DetectorQueAnota:
    """openWakeWord falso para el loop simple: anota con qué se le pide esperar
    y dispara a la primera (o devuelve lo que diga should_stop)."""

    threshold = 0.5
    threshold_media = None

    def __init__(self) -> None:
        self.pedidos: list[dict] = []

    def listen_for_wake(self, **kwargs) -> bool:
        self.pedidos.append(kwargs)
        should_stop = kwargs.get("should_stop")
        return not (should_stop is not None and should_stop())


@pytest.mark.parametrize("con_sesion", [False, True])
def test_loop_simple_oww_espera_el_wake_en_la_sesion(loop_simple, micro, con_sesion):
    """openWakeWord en el loop simple: sin sesión, la llamada de siempre
    (timeout_s=None y nada más). Con sesión, lee de ella y mira el panel en su
    tick (should_stop): una pausa ya no espera a que alguien diga el wake."""
    from crotolamo.voice.state import SharedState
    from interfaces import listener

    detector = _DetectorQueAnota()
    stt = types.SimpleNamespace(
        listen_once=lambda **kw: "pausa la música", listen_smart=lambda **kw: "")
    extra: dict = {}
    if con_sesion:
        micro.con()
        extra["mic"] = micro.sesion()
        extra["mic"].start()
    agente = _AgenteDelLoop(loop_simple.panel)
    rc = listener._run_simple_loop(
        agent=agente, stt=stt, wake_stt=None, tts=_TtsDelLoop(), wake_detector=detector,
        use_oww=True, cfg=listener.ListenerConfig(followup_s=0, ack="off",
                                                  stream_speak=False, smart_endpoint=False),
        hud_state=SharedState(), say=lambda text: None, **extra)
    assert rc == 0 and agente.commands == ["pausa la música"]
    if not con_sesion:
        assert detector.pedidos == [{"timeout_s": None}]
        return
    pedido = detector.pedidos[0]
    assert pedido["source"] is extra["mic"] and pedido["timeout_s"] is None
    loop_simple.panel["fin"] = False
    assert pedido["should_stop"]() is False
    loop_simple.panel["on"] = False
    assert pedido["should_stop"]() is True


# --- paso 3: run_listen arma (o no) la sesión ---------------------------------

class _SesionFalsa:
    """MicSession falsa para el cableado de run_listen."""

    creadas: list = []
    falla: Exception | None = None

    def __init__(self, sample_rate, device, *, give_up_s=0.0, **_kw) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.give_up_s = give_up_s
        self.arrancada = False
        self.flushes = 0
        self.cierres = 0
        type(self).creadas.append(self)

    def start(self) -> None:
        if type(self).falla is not None:
            raise type(self).falla
        self.arrancada = True

    def flush(self) -> int:
        self.flushes += 1
        return 0

    def close(self) -> None:
        self.cierres += 1


@pytest.fixture
def sesion_falsa(monkeypatch):
    from interfaces import listener

    monkeypatch.setattr(_SesionFalsa, "creadas", [])
    monkeypatch.setattr(_SesionFalsa, "falla", None)
    monkeypatch.setattr(listener, "MicSession", _SesionFalsa)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    return _SesionFalsa


def _settings(**voice):
    return types.SimpleNamespace(voice={"sample_rate": SR, "input_device": "mic_x", **voice},
                                 wake={})


def test_sin_audio_session_no_se_abre_ninguna_sesion(sesion_falsa):
    from interfaces import listener

    cfg = listener.ListenerConfig.from_settings(_settings())
    assert cfg.audio_session is False                   # de fábrica: lo de siempre
    assert listener._abrir_sesion_mic(_settings(), cfg) is None
    assert sesion_falsa.creadas == []


@pytest.mark.parametrize("servicio, rendirse", [(False, 0.0), (True, 45.0)])
def test_la_sesion_solo_se_rinde_bajo_systemd(sesion_falsa, monkeypatch, servicio, rendirse):
    """mic_give_up_s solo cuenta como servicio (INVOCATION_ID): ahí systemd
    reinicia al listener que sale con 1. En una terminal se reintenta siempre."""
    from interfaces import listener

    if servicio:
        monkeypatch.setenv("INVOCATION_ID", "abc123")
    settings = _settings(audio_session=True, mic_give_up_s=45)
    sesion = listener._abrir_sesion_mic(settings, listener.ListenerConfig.from_settings(settings))
    assert sesion is sesion_falsa.creadas[0] and sesion.arrancada
    assert (sesion.sample_rate, sesion.device, sesion.give_up_s) == (SR, "mic_x", rendirse)


def test_una_sesion_que_no_abre_cae_al_stream_por_grabacion(sesion_falsa, caplog):
    from interfaces import listener

    sesion_falsa.falla = OSError("device unavailable")
    settings = _settings(audio_session=True)
    with caplog.at_level(logging.WARNING, logger="crotolamo.listener"):
        assert listener._abrir_sesion_mic(
            settings, listener.ListenerConfig.from_settings(settings)) is None
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1 and "device unavailable" in avisos[0]
    assert sesion_falsa.creadas[0].cierres == 1


def test_run_listen_simple_con_audio_session_pasa_la_sesion_y_la_cierra(
        sesion_falsa, monkeypatch, tmp_path):
    """De punta a punta (--simple): el loop recibe la sesión, la confirmación por
    voz escucha en ella tras tirar lo capturado mientras sonaba la pregunta, y
    al salir se cierra. Con la clave apagada el loop no recibe `mic`."""
    from interfaces import listener
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _clase_stt_por_rol,
        _FakeAgent,
        _join_warmers,
    )

    for activa in (False, True):
        detector = _DetectorQueAnota()
        detector.available = lambda: True  # type: ignore[attr-defined]
        detector.attach_media = lambda monitor: None  # type: ignore[attr-defined]
        _arnes_run_listen(monkeypatch, tmp_path, detector)
        monkeypatch.setattr(listener, "get_settings", lambda a=activa: types.SimpleNamespace(
            voice={"wake_duck": False, "audio_session": a}, wake={}))
        instancias: list = []
        monkeypatch.setattr(listener, "STT", _clase_stt_por_rol(instancias, "sí, confirmo"))
        confirms: list = []
        monkeypatch.setattr(listener, "build_agent", lambda confirm_fn=None: (
            confirms.append(confirm_fn) or (_FakeAgent(), None)))
        visto: dict = {}

        def _loop(**kw) -> int:
            visto["mic"] = kw.get("mic", "sin mic")
            visto["confirmado"] = confirms[0]("Voy a borrar el archivo.")
            return 0

        monkeypatch.setattr(listener, "_run_simple_loop", _loop)
        try:
            assert listener.run_listen(["--simple"]) == 0
        finally:
            _join_warmers()
        orden = next(s for s in instancias if s.rol == "orden")
        assert visto["confirmado"] is True
        if not activa:
            assert visto["mic"] == "sin mic" and sesion_falsa.creadas == []
            assert orden.escuchas == [{"silence_ms": 400, "max_seconds": 5,
                                       "min_audio_s": 0.3}]
            continue
        sesion = sesion_falsa.creadas[0]
        assert visto["mic"] is sesion
        assert orden.escuchas == [{"silence_ms": 400, "max_seconds": 5, "min_audio_s": 0.3,
                                   "source": sesion}]
        assert sesion.flushes == 1 and sesion.cierres == 1


def test_la_config_de_fabrica_documenta_audio_session_apagada():
    """audio_session cambia cómo el loop del patrón es dueño del micrófono y no
    se puede validar sin su hardware: va apagada de fábrica (el camino de
    siempre), documentada con lo medido, y con mic_give_up_s al lado."""
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG
    from interfaces import listener

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    voice = tomllib.loads(texto)["voice"]
    assert voice["audio_session"] is False and voice["mic_give_up_s"] == 90
    cfg = listener.ListenerConfig.from_settings(types.SimpleNamespace(voice=voice, wake={}))
    assert cfg.audio_session is False and cfg.mic_give_up_s == 90
    lineas = texto.splitlines()
    i = lineas.index("audio_session = false")
    inicio = max(n for n in range(i) if lineas[n].startswith("# Quién es dueño del micrófono"))
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[inicio:i])
    assert "MEDIDO" in comentario and "false (de fábrica)" in comentario, comentario


def test_lo_medido_con_el_micro_caido_cuenta_todo_lo_que_va_al_journal():
    """Con false, el micro caído no solo imprime: cada reintento deja un WARNING
    "VAD silero falló" (a stderr, que bajo systemd también va al journal). Lo
    documentado para decidir la clave debe contar los dos canales: el bench
    micfail_fuzzy da 2 prints + 6 avisos el primer minuto y luego ~2 avisos por
    minuto mientras siga caído; con true, 2 avisos en todo el incidente."""
    from crotolamo.settings import DEFAULT_CONFIG

    lineas = DEFAULT_CONFIG.read_text(encoding="utf-8").splitlines()
    i = lineas.index("audio_session = false")
    inicio = max(n for n in range(i) if lineas[n].startswith("# Quién es dueño del micrófono"))
    comentario = " ".join(linea.lstrip("# ") for linea in lineas[inicio:i])
    assert "2 líneas al journal -> 0" not in comentario  # solo contaba stdout
    assert "por reintento" in comentario and "VAD silero falló" in comentario, comentario
    assert "2.880 al día" in comentario, comentario
