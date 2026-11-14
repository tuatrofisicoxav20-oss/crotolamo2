"""Tests de huecos críticos del loop de voz (con fakes, sin audio real).

1. Cooldown anti-eco del EarThread: al volver a IDLE tras un turno, el wake se
   IGNORA durante _post_speak_grace_s (el eco del TTS Piper dispara falsos wakes
   porque el modelo se entrenó con voz Piper); pasado el período, se acepta.
2. Fallback Silero -> energía en _SileroVad: si onnxruntime/numpy o _get_silero_vad
   fallan, _fallback se activa y el VAD de energía distingue voz de silencio.
3. Supervisión: el loop simple ante un error que se repite (un aviso por
   mensaje, espera creciente, código 1 si la sesión de captura se rinde) y el
   oído concurrente leyendo de una MicSession ([voice].audio_session).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from types import SimpleNamespace

import crotolamo.voice.adapters as adapters
from crotolamo.voice.adapters import _SileroVad
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import EarThread


class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


class FeedMic:
    """Micrófono controlado por el test: entrega chunks encolados o None."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._fed = 0         # chunks encolados por el test
        self._served = 0      # chunks entregados al oído
        self._idle_after = 0  # entregados cuando el oído volvió a leer y no había más

    def feed(self, chunk, times=1):
        for _ in range(times):
            with self._lock:
                self._fed += 1
            self.q.put(chunk)

    def read(self):
        try:
            chunk = self.q.get(timeout=0.02)
        except queue.Empty:
            with self._lock:
                self._idle_after = self._served
            return None
        with self._lock:
            self._served += 1
        return chunk

    def wait_consumed(self, timeout=2.0) -> bool:
        """True cuando el oído PROCESÓ todo lo encolado: volvió a leer tras el
        último chunk (lee y procesa de a uno). Los tests que comprueban que algo
        NO pasó esperan esto en vez de un sleep, que pasaba aunque no corriera."""
        def consumido():
            with self._lock:
                return self._idle_after == self._fed
        return _wait(consumido, timeout)


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _ear_con_mic(mic, state, grace_s):
    """EarThread con fakes, siguiendo el patrón de test_loop.py."""
    stt_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    ear = EarThread(
        mic,
        wake_fn=lambda c: c == "wake",
        vad_fn=lambda c: 0.9 if c == "voice" else 0.0,
        to_wav=lambda frames: "WAV",
        tts=FakeTts(),
        stt_queue=stt_q, tts_queue=tts_q, state=state, shutdown=shutdown,
    )
    ear._post_speak_grace_s = grace_s
    return ear, shutdown


# --- cooldown anti-eco (loop de "habla -> falso wake -> habla...") ---
def test_cooldown_ignora_el_wake_durante_la_gracia_post_habla():
    """En IDLE recién salido de SPEAKING, el wake DEBE ignorarse (eco del TTS)."""
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    mic = FeedMic()
    ear, shutdown = _ear_con_mic(mic, state, grace_s=30.0)  # gracia enorme
    ear.start()
    try:
        # 1) Un chunk en SPEAKING para que _prev_mode registre el habla.
        mic.feed("sil")
        assert mic.wait_consumed(), "el oído no procesó el chunk en SPEAKING"
        # 2) Fin del turno: a IDLE. El siguiente chunk arranca el cooldown.
        state.set_mode(Mode.IDLE)
        # 3) Eco del TTS: chunks que disparan la wake word dentro de la gracia.
        mic.feed("wake", times=10)
        assert mic.wait_consumed(), "el oído no procesó los wakes"
        assert state.get_mode() is Mode.IDLE, "el wake en gracia NO debe despertar"
        assert state.turn_id == 0, "no debe abrirse ningún turno durante la gracia"
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_cooldown_acepta_el_wake_pasada_la_gracia():
    """Pasado _post_speak_grace_s, el mismo wake SÍ debe despertar al asistente."""
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    mic = FeedMic()
    ear, shutdown = _ear_con_mic(mic, state, grace_s=0.2)  # gracia corta y real
    ear.start()
    try:
        mic.feed("sil")
        time.sleep(0.1)
        state.set_mode(Mode.IDLE)
        # Dentro de la gracia: ignorado (y de paso arranca el cooldown).
        mic.feed("wake")
        time.sleep(0.05)
        assert state.get_mode() is Mode.IDLE
        # Esperar a que la gracia expire y volver a intentar.
        time.sleep(0.3)
        mic.feed("wake", times=3)
        assert _wait(lambda: state.get_mode() is Mode.LISTENING), \
            "pasada la gracia, el wake DEBE despertar"
        assert state.turn_id == 1
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


# --- fallback Silero -> energía en _SileroVad ---
_VOZ = [0.5] * 512       # chunk de alta energía (RMS >> _ENERGY_FLOOR)
_SILENCIO = [0.0] * 512  # chunk de silencio total


def test_silero_vad_cae_a_energia_si_falta_la_dependencia(monkeypatch):
    """ImportError al cargar Silero => _fallback y VAD de energía funcional."""
    def _sin_silero():
        raise ImportError("no hay silero-vad, patrón")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _sin_silero)
    vad = _SileroVad()
    # El primer chunk dispara el intento de carga, falla y ya responde por energía.
    assert vad(_VOZ) == 1.0
    assert vad._fallback is True
    assert vad._model is None
    # Ya en fallback: alta energía = voz, silencio = no voz.
    assert vad(_VOZ) == 1.0
    assert vad(_SILENCIO) == 0.0


def test_silero_vad_cae_a_energia_si_la_carga_revienta(monkeypatch):
    """Error genérico (carga/inferencia) => fallback permanente sin modelo."""
    def _revienta():
        raise RuntimeError("ONNX corrupto")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _revienta)
    vad = _SileroVad()
    assert vad(_SILENCIO) == 0.0
    assert vad._fallback is True
    assert vad._model is None
    assert vad(_VOZ) == 1.0


def test_silero_vad_cae_a_energia_si_falla_el_import_de_torch(monkeypatch):
    """Si las dependencias del modelo no importan al cargarlo, cae a energía igual.

    El VAD ya no usa torch (voice/runtime.py corre el ONNX con numpy +
    onnxruntime): se bloquean los dos y se vacía la caché de stt, para que la
    carga ocurra DE VERDAD dentro de __call__ y falle en el import."""
    import builtins

    import crotolamo.voice.stt as stt_mod

    real_import = builtins.__import__

    def _sin_torch(name, *args, **kwargs):
        if name in ("torch", "onnxruntime"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _sin_torch)
    monkeypatch.setattr(stt_mod, "_silero_vad", None)
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    vad = _SileroVad()
    assert vad(_VOZ) == 1.0
    assert vad._fallback is True
    assert vad(_SILENCIO) == 0.0


def test_silero_vad_no_reintenta_el_import_tras_el_fallback(monkeypatch):
    """Una vez en fallback, __call__ no vuelve a intentar cargar Silero."""
    llamadas = {"n": 0}

    def _cuenta():
        llamadas["n"] += 1
        raise ImportError("sin silero")

    monkeypatch.setattr("crotolamo.voice.stt._get_silero_vad", _cuenta)
    vad = _SileroVad()
    for _ in range(5):
        vad(_SILENCIO)
    assert llamadas["n"] == 1, "el import fallido debe intentarse UNA sola vez"


def test_adapters_reexportados_desde_loop():
    """Compatibilidad: los nombres históricos siguen importables desde loop."""
    from crotolamo.voice import loop

    assert loop._SileroVad is adapters._SileroVad
    assert loop.EarThread is EarThread


# --- supervisión: el loop simple ante un error que se repite -----------------
# Antes: print + sleep(1) por vuelta. Con el micrófono caído eran 119 líneas a
# stdout por minuto (171.360 al día en el journal) y 119 aperturas.

def _wake_que_falla(guion):
    """wake_stt del loop simple: cada entrada del guion es una excepción que
    lanza o el texto que oye; agotado, Ctrl-C."""
    from tests.test_auditoria_voice_listener import _ScriptedWake

    textos = [x for x in guion if isinstance(x, str)]
    wake = _ScriptedWake(textos)
    real = wake.listen_once
    pendientes = list(guion)

    def listen_once(**kwargs):
        if pendientes and isinstance(pendientes[0], BaseException):
            wake.calls.append(kwargs)
            raise pendientes.pop(0)
        if pendientes:
            pendientes.pop(0)
        return real(**kwargs)

    wake.listen_once = listen_once
    return wake


def test_loop_simple_un_error_que_se_repite_se_imprime_una_vez_y_espera_mas(
        monkeypatch, capsys):
    from interfaces import listener
    from tests.test_auditoria_voice_listener import _run_simple

    caido = OSError("Error opening InputStream: device unavailable")
    perdido = OSError("Input overflowed / device lost")
    # 7 fallos de un mensaje, 2 de otro, una ventana que va bien y el primero otra vez.
    wake_stt = _wake_que_falla([caido] * 7 + [perdido] * 2 + [""] + [caido])
    dormidas: list[float] = []
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt, sleep=dormidas.append)
    assert rc == 0 and agent.commands == []
    # 1, 2, 4 ... hasta 30 s; la ventana buena reinicia la racha (el último 1).
    assert dormidas == [1, 2, 4, 8, 16, 30, 30, 30, 30, 1]
    salida = capsys.readouterr().out
    # Un aviso por mensaje distinto y racha (antes: uno por vuelta, 10 en total).
    assert salida.count(f"Error en el listener: {caido}") == 2
    assert salida.count(f"Error en el listener: {perdido}") == 1
    # La espera se reanuncia tras el primer fallo de cada racha, no tras cada uno.
    assert salida.count("Esperando palabra de activación...") == 3
    assert listener._Reintentos().min_s == 1.0


def test_reintentos_un_fallo_tras_un_rato_bueno_es_un_incidente_nuevo():
    """openWakeWord por llamada solo vuelve con un wake: si el reintento anduvo
    más que la espera más larga, el fallo siguiente es otro incidente (se avisa
    y se espera 1 s), no la cola de la racha vieja."""
    from interfaces import listener

    reloj = [0.0]
    r = listener._Reintentos(reloj=lambda: reloj[0])
    assert r.fallo(OSError("x")) == (True, True, 1.0)
    reloj[0] = 1.0                       # el reintento falló al instante
    assert r.fallo(OSError("x")) == (False, False, 2.0)
    reloj[0] = 3.0 + 29.0                # anduvo 29 s (< 30): sigue la racha
    assert r.fallo(OSError("x")) == (False, False, 4.0)
    reloj[0] = 36.0 + 30.0               # anduvo 30 s: incidente nuevo
    assert r.fallo(OSError("x")) == (True, True, 1.0)
    r.ok()
    assert r.fallo(OSError("x")) == (True, True, 1.0)


def test_loop_simple_sale_con_1_si_el_microfono_no_vuelve(monkeypatch, capsys):
    """MicFailed (la sesión de captura se rindió, solo bajo systemd): código 1
    para que Restart=on-failure reinicie el servicio, sin reintentar más."""
    from crotolamo.voice.capture import MicFailed
    from tests.test_auditoria_voice_listener import _run_simple

    wake_stt = _wake_que_falla([MicFailed("el micrófono no volvió: device lost")])
    dormidas: list[float] = []
    rc, _agent = _run_simple(monkeypatch, wake_stt=wake_stt, sleep=dormidas.append)
    assert rc == 1 and dormidas == []
    assert "device lost" in capsys.readouterr().out


# --- supervisión: el oído concurrente leyendo de una MicSession ---------------
# Con [voice].audio_session el loop concurrente lee de capture.MicSession: un
# fallo del dispositivo se reintenta DENTRO del proceso (cierra, un aviso por
# mensaje, espera creciente, reabre) en vez de rendirse en ~1 s y esperar a que
# systemd reinicie el servicio. Sin la clave, _RealMic y la regla de siempre.

def _loop_con_sesion(micro_falso, **sesion_kw):
    import numpy as np

    from crotolamo.voice import capture
    from crotolamo.voice.loop import VoiceLoop

    sesion = capture.MicSession(16000, None, **sesion_kw)
    sesion._pausa = lambda s: sesion._stop.wait(0.001)  # type: ignore[method-assign]
    sesion.start()
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"),
        SimpleNamespace(transcribe=lambda audio, hotwords=None: ""),
        FakeTts(), None, mic=sesion,
        wake_fn=lambda c: float(np.max(np.abs(c))) > 0.4, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    vl._ear._PAUSA_TRAS_FALLO_S = 0.001
    return vl, sesion


def _correr(vl, timeout: float = 5.0) -> tuple[list, threading.Thread]:
    resultado: list = []
    hilo = threading.Thread(target=lambda: resultado.append(vl.run()), daemon=True)
    hilo.start()
    return resultado, hilo


def test_oido_con_sesion_se_recupera_de_un_microfono_que_falla(monkeypatch, caplog):
    import logging

    import numpy as np

    from tests.test_capture import _Micro

    # 20 lecturas fallan (antes: 10 seguidas y el oído se rendía) y la
    # reapertura también falla 3 veces; luego el patrón dice el wake.
    audio = np.concatenate([np.zeros(512 * 4, np.float32), np.full(512, 0.5, np.float32)])
    micro = _Micro(audio, fallos_lectura=["device lost"] * 20,
                   fallos_apertura=[None] + ["device unavailable"] * 3 + [None] * 20)
    monkeypatch.setitem(sys.modules, "sounddevice", micro.modulo)
    with caplog.at_level(logging.WARNING):
        vl, sesion = _loop_con_sesion(micro)
        resultado, hilo = _correr(vl)
        try:
            assert _wait(lambda: vl.state.turn_id == 1, 5.0), "el oído no volvió a oír"
            assert vl.state.get_mode() is Mode.LISTENING
            assert all(t.is_alive() for t in vl.threads), "murió un hilo del loop"
        finally:
            vl.shutdown.set()
            hilo.join(5.0)
            sesion.close()
    assert resultado == [0]
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len([a for a in avisos if "device lost" in a]) == 1
    assert len([a for a in avisos if "device unavailable" in a]) == 1
    assert not [a for a in avisos if a.startswith("oído:")], avisos


def test_oido_con_sesion_que_se_rinde_apaga_el_loop_con_1(monkeypatch):
    """Bajo systemd la sesión se rinde tras mic_give_up_s: el oído recibe
    MicFailed, se rinde como ante cualquier fallo seguido y run() devuelve 1."""
    import numpy as np

    from tests.test_capture import _Micro

    micro = _Micro(np.zeros(512, np.float32), fallos_lectura=["device lost"],
                   fallos_apertura=[None] + ["device unavailable"] * 100000)
    monkeypatch.setitem(sys.modules, "sounddevice", micro.modulo)
    vl, sesion = _loop_con_sesion(micro, give_up_s=0.05)
    resultado, hilo = _correr(vl)
    try:
        hilo.join(5.0)
    finally:
        vl.shutdown.set()
        hilo.join(5.0)
        sesion.close()
    assert resultado == [1] and sesion.failed


def test_run_listen_concurrente_con_audio_session_le_da_la_sesion_al_oido(
        monkeypatch, tmp_path):
    """Cableado: con [voice].audio_session el VoiceLoop recibe la sesión como
    micrófono y se cierra al salir; sin la clave (de fábrica), ni se crea y el
    VoiceLoop no recibe `mic` (su _RealMic de siempre)."""
    from crotolamo.voice import loop as voice_loop
    from interfaces import listener
    from tests.test_auditoria_voice_listener import (
        _arnes_run_listen,
        _BlockingDetector,
        _join_warmers,
    )
    from tests.test_capture import _SesionFalsa

    monkeypatch.setattr(_SesionFalsa, "creadas", [])
    monkeypatch.setattr(_SesionFalsa, "falla", None)
    monkeypatch.setattr(listener, "MicSession", _SesionFalsa)
    for activa in (False, True):
        _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
        monkeypatch.setattr(listener, "get_settings", lambda a=activa: SimpleNamespace(
            voice={"wake_duck": False, "audio_session": a}, wake={}))
        capturado: dict = {}

        class _Loop:
            def __init__(self, agent, stt, tts, wake_detector, **kwargs):
                capturado.update(kwargs)
                self.state = None

            def run(self):
                return 0

        monkeypatch.setattr(voice_loop, "VoiceLoop", _Loop)
        try:
            assert listener.run_listen([]) == 0
        finally:
            _join_warmers()
        if not activa:
            assert "mic" not in capturado and _SesionFalsa.creadas == []
            continue
        sesion = _SesionFalsa.creadas[0]
        assert capturado["mic"] is sesion and sesion.arrancada and sesion.cierres == 1


# Con la sesión, mic.read() devuelve None cada segundo mientras el micrófono
# reintenta, y los plazos de la orden (start_timeout_s, max_command_s) cuentan
# CHUNKS: un micro que moría tras el wake dejaba al loop en LISTENING todo el
# corte (HUD 'escuchando', música atenuada; en una terminal, para siempre).

class _MicQueCalla(FeedMic):
    """FeedMic que anota cuándo entregó su último chunk."""

    def __init__(self):
        super().__init__()
        self.ultimo_chunk = 0.0

    def read(self):
        chunk = super().read()
        if chunk is not None:
            self.ultimo_chunk = time.monotonic()
        return chunk


def _oido_que_anota(mic, state):
    ear, shutdown = _ear_con_mic(mic, state, grace_s=0.0)
    enviados: list[float] = []

    def _to_wav(frames):
        enviados.append(time.monotonic())
        return "WAV"

    ear.to_wav = _to_wav
    return ear, shutdown, enviados


def test_oido_sin_audio_en_plena_orden_la_cierra_con_lo_oido():
    """Un segundo sin audio en plena orden la cierra como el fin de la frase: lo
    oído va a Whisper (como la grabación del modo simple con fuente, que acaba
    tras 1 s sin audio). Un None suelto no la cierra."""
    state = SharedState()
    mic = _MicQueCalla()
    ear, shutdown, enviados = _oido_que_anota(mic, state)
    ear.start()
    try:
        mic.feed("wake")
        mic.feed("voice", times=3)
        assert mic.wait_consumed()
        assert state.get_mode() is Mode.LISTENING, "un None suelto no cierra la orden"
        assert _wait(lambda: state.get_mode() is Mode.THINKING, 5.0), "la orden no se cerró"
        assert ear.stt_q.get_nowait() == ("WAV", 1)
        assert enviados[0] - mic.ultimo_chunk >= ear._SIN_AUDIO_S
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_oido_sin_audio_y_sin_voz_vuelve_a_reposo_sin_whisper():
    state = SharedState()
    mic = _MicQueCalla()
    ear, shutdown, enviados = _oido_que_anota(mic, state)
    ear.start()
    try:
        mic.feed("wake")
        mic.feed("sil", times=3)
        assert mic.wait_consumed()
        assert state.get_mode() is Mode.LISTENING
        assert _wait(lambda: state.get_mode() is Mode.IDLE, 5.0), "se quedó escuchando"
        assert ear.stt_q.empty() and enviados == []
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_oido_sin_audio_fuera_de_la_orden_no_toca_el_turno():
    """Sin audio pensando o hablando no hay orden que cerrar: el turno sigue."""
    state = SharedState()
    mic = _MicQueCalla()
    ear, shutdown, enviados = _oido_que_anota(mic, state)
    ear._SIN_AUDIO_S = 0.05
    state.new_turn(mode=Mode.THINKING)
    ear.start()
    try:
        mic.feed("sil")
        assert mic.wait_consumed()
        time.sleep(0.3)
        assert state.get_mode() is Mode.THINKING
        assert ear.stt_q.empty() and enviados == []
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_loop_con_sesion_y_micro_caido_tras_el_wake_vuelve_a_reposo(monkeypatch):
    """El VoiceLoop de verdad sobre una MicSession de verdad: el dispositivo se
    pierde justo después del wake y no vuelve. Antes: LISTENING mientras durase
    el corte (el plazo de 4 s nunca vencía). Ahora: a reposo en ~1 s, nada a
    Whisper y los cuatro hilos vivos (la sesión sigue reintentando)."""
    import numpy as np

    from crotolamo.voice import capture
    from crotolamo.voice.loop import VoiceLoop
    from tests.test_capture import _Micro

    sala, wake = np.full(512, 0.001, np.float32), np.full(512, 0.5, np.float32)
    micro = _Micro(np.concatenate([sala] * 5 + [wake] + [sala] * 3),
                   fallos_lectura=[None] * 9 + ["device lost"],
                   fallos_apertura=[None] + ["device unavailable"] * 50)
    monkeypatch.setitem(sys.modules, "sounddevice", micro.modulo)
    sesion = capture.MicSession(16000, None)
    sesion.start()
    transcritos: list = []
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"),
        SimpleNamespace(transcribe=lambda audio, hotwords=None: transcritos.append(audio) or ""),
        FakeTts(), None, mic=sesion,
        wake_fn=lambda c: float(np.max(np.abs(c))) > 0.4, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    resultado, hilo = _correr(vl)
    try:
        assert _wait(lambda: vl.state.turn_id == 1, 5.0), "no oyó el wake"
        assert _wait(lambda: vl.state.get_mode() is Mode.IDLE, 5.0), "se quedó escuchando"
        assert transcritos == [] and all(t.is_alive() for t in vl.threads)
    finally:
        vl.shutdown.set()
        hilo.join(5.0)
        sesion.close()
    assert resultado == [0]
