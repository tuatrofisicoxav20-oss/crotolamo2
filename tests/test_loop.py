"""Tests del loop concurrente con fakes (sin micrófono ni audio real).

Todos los tests de threads usan fakes y terminan con shutdown.set() + join(timeout).
Ningún test toca audio real ni debe colgar.
"""

import queue
import threading
import time

from pathlib import Path
from types import SimpleNamespace

from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.threads import (
    END,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
)
from crotolamo.voice.state import Mode, SharedState


# --- fakes compartidos ---
class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# --- M3.3: MouthThread ---
def test_mouth_discards_old_turn():
    tts = FakeTts()
    state = SharedState()
    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    mouth = MouthThread(tts, q, state, shutdown)
    mouth.start()
    try:
        # Avanzamos a turno 2 ANTES de encolar, para que el test sea determinista.
        state.new_turn()
        state.new_turn()  # turno actual = 2
        q.put(Utterance("vieja", 1))   # de un turno abortado -> debe descartarse
        q.put(Utterance("nueva", 2))   # vigente -> debe hablarse
        assert _wait(lambda: tts.spoken == ["nueva"] and q.empty())
    finally:
        shutdown.set()
        mouth.join(timeout=2.0)
    assert not mouth.is_alive()
    assert tts.spoken == ["nueva"]


def test_mouth_end_sentinel_returns_to_idle():
    tts = FakeTts()
    state = SharedState()
    state.set_mode(Mode.SPEAKING)
    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    mouth = MouthThread(tts, q, state, shutdown)
    mouth.start()
    try:
        q.put(END)
        assert _wait(lambda: state.get_mode() is Mode.IDLE)
    finally:
        shutdown.set()
        mouth.join(timeout=2.0)


# --- M3.4: BrainThread ---
class FakeAgent:
    def __init__(self, reply):
        self.reply = reply

    def handle_turn(self, command):
        return self.reply


class GatedAgent:
    """Agente que se bloquea hasta que el test lo libera (para forzar el race)."""

    def __init__(self, reply):
        self.reply = reply
        self.started = threading.Event()
        self.release = threading.Event()

    def handle_turn(self, command):
        self.started.set()
        self.release.wait(timeout=2.0)
        return self.reply


def test_brain_enqueues_sentences_when_turn_unchanged():
    state = SharedState()
    state.new_turn()  # actual = 1
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(FakeAgent("Hola. Mundo."), cmd_q, tts_q, state, shutdown)
    brain.start()
    try:
        cmd_q.put(("saluda", 1))
        assert _wait(lambda: tts_q.qsize() >= 3)
        got = [tts_q.get() for _ in range(3)]
        assert [u.text for u in got[:2]] == ["Hola.", "Mundo."]
        assert got[-1] is END
    finally:
        shutdown.set()
        brain.join(timeout=2.0)


def test_brain_drops_reply_if_turn_changed():
    state = SharedState()
    state.new_turn()  # actual = 1
    agent = GatedAgent("Hola. Mundo.")
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(agent, cmd_q, tts_q, state, shutdown)
    brain.start()
    try:
        cmd_q.put(("saluda", 1))
        assert agent.started.wait(timeout=2.0)  # el brain ya está "pensando"
        state.new_turn()                         # barge-in: actual = 2, invalida turno 1
        agent.release.set()                      # dejamos terminar la inferencia vieja
        # El comando del turno vigente sale DESPUÉS del viejo (FIFO): cuando su END
        # llega, el viejo ya se procesó (sin esto, un sleep pasaba sin el Brain).
        cmd_q.put(("otra", 2))
        assert _wait(lambda: not tts_q.empty() and tts_q.queue[-1] is END)
        got = list(tts_q.queue)
        assert [u.turn_id for u in got[:-1]] == [2, 2]  # la respuesta vieja NO se encoló
    finally:
        shutdown.set()
        brain.join(timeout=2.0)


# --- M3.5: SttThread ---
class FakeStt:
    def __init__(self, text="abre youtube"):
        self.text = text
        self.transcribed = 0
        self.hotwords_seen: list = []  # hotwords recibidos por llamada

    def transcribe(self, path, hotwords=None):
        self.transcribed += 1
        self.hotwords_seen.append(hotwords)
        return self.text

    def listen_once(self, **kwargs):
        return ""


_NO_WAV = Path("/tmp/crotolamo_no_existe.wav")


def test_stt_passes_text_for_current_turn():
    state = SharedState()
    state.new_turn()  # actual = 1
    stt = FakeStt("abre youtube")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    th = SttThread(stt, in_q, out_q, state, shutdown)
    th.start()
    try:
        in_q.put((_NO_WAV, 1))
        assert _wait(lambda: not out_q.empty())
        text, turn = out_q.get()
        assert text == "abre youtube" and turn == 1
    finally:
        shutdown.set()
        th.join(timeout=2.0)


def test_stt_thread_propaga_hotwords_de_comandos():
    """La ruta de COMANDOS (SttThread, siempre post-wake) transcribe con las
    hotwords configuradas; sin configurar, transcribe con None (sin sesgo)."""
    state = SharedState()
    state.new_turn()
    stt = FakeStt("abre tletl")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    th = SttThread(stt, in_q, out_q, state, shutdown,
                   hotwords="Crotolamo, Tletl, Huevonitis")
    th.start()
    try:
        in_q.put((_NO_WAV, 1))
        assert _wait(lambda: not out_q.empty())
        assert stt.hotwords_seen == ["Crotolamo, Tletl, Huevonitis"]
    finally:
        shutdown.set()
        th.join(timeout=2.0)


def test_stt_drops_if_turn_changed():
    state = SharedState()
    state.new_turn()
    state.new_turn()  # actual = 2; el audio será de turno 1 (abortado)
    stt = FakeStt("lo que sea")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    th = SttThread(stt, in_q, out_q, state, shutdown)
    th.start()
    try:
        in_q.put((_NO_WAV, 1))
        # Detrás va un audio del turno vigente (FIFO): cuando sale, el abortado ya
        # se procesó (sin esto, un sleep pasaba aunque el Stt no corriera).
        in_q.put((_NO_WAV, 2))
        assert _wait(lambda: not out_q.empty())
        assert out_q.get() == ("lo que sea", 2) and out_q.empty()
        assert stt.transcribed == 1  # el abortado ni siquiera se transcribió
    finally:
        shutdown.set()
        th.join(timeout=2.0)


# --- M3.6: EarThread (con micrófono fake) ---
class FakeMic:
    """Entrega una secuencia fija de 'chunks' (etiquetas) y luego None.

    `agotado` se enciende cuando el oído vuelve a leer tras el ÚLTIMO chunk, es
    decir, cuando ya procesó todo el guion: los tests que comprueban que algo NO
    pasó esperan eso en vez de un sleep (que pasaba aunque el oído no corriera).
    """

    def __init__(self, chunks):
        self._it = iter(chunks)
        self.agotado = threading.Event()

    def read(self):
        try:
            return next(self._it)
        except StopIteration:
            self.agotado.set()
            return None


def _ear(mic, state, *, allow_barge_in=False, silence_ms=64, tts=None,
         wake_label="wake", voice_label="voice",
         grace_ms=0, margin=0.0, min_chunks=1):
    # Por defecto, gracia=0/min=1: barge-in de un solo chunk (conducta base M3.6).
    stt_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    ear = EarThread(
        mic,
        wake_fn=lambda c: c == wake_label,
        vad_fn=lambda c: 0.9 if c == voice_label else 0.0,
        to_wav=lambda frames: _NO_WAV,
        tts=tts or FakeTts(),
        stt_queue=stt_q, tts_queue=tts_q, state=state, shutdown=shutdown,
        silence_ms=silence_ms, allow_barge_in=allow_barge_in,
        barge_in_grace_ms=grace_ms, barge_in_threshold_margin=margin,
        barge_in_min_chunks=min_chunks,
    )
    return ear, stt_q, tts_q, shutdown


def test_ear_wake_in_idle_starts_listening():
    state = SharedState()  # IDLE, turno 0
    ear, _stt_q, _tts_q, shutdown = _ear(FakeMic(["wake"]), state)
    ear.start()
    try:
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        assert state.turn_id == 1
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_silence_ends_command_and_enqueues():
    state = SharedState()
    # silence_ms=64 con chunk 32ms -> 2 chunks de silencio cierran el comando.
    mic = FakeMic(["wake", "voice", "sil", "sil"])
    ear, stt_q, _tts_q, shutdown = _ear(mic, state, silence_ms=64)
    ear.start()
    try:
        assert _wait(lambda: not stt_q.empty())
        wav, turn = stt_q.get()
        assert turn == 1
        assert _wait(lambda: state.get_mode() is Mode.THINKING)
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_barge_in_during_speaking():
    state = SharedState()
    state.new_turn()              # turno 1
    state.set_mode(Mode.SPEAKING)  # Crotolamo está hablando
    tts = FakeTts()
    ear, _stt_q, tts_q, shutdown = _ear(
        FakeMic(["voice"]), state, allow_barge_in=True, tts=tts
    )
    tts_q.put(Utterance("frase en curso", 1))  # algo en la cola de habla
    ear.start()
    try:
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        assert tts.stopped >= 1          # cortó el audio
        assert tts_q.empty()             # vació la cola de habla
        assert state.turn_id == 2        # turno nuevo invalida lo viejo
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


def test_ear_no_barge_in_when_disabled():
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    tts = FakeTts()
    mic = FakeMic(["voice", "voice", "voice"])
    ear, _stt_q, _tts_q, shutdown = _ear(mic, state, allow_barge_in=False, tts=tts)
    ear.start()
    try:
        assert mic.agotado.wait(2.0), "el oído no procesó el guion"
        assert state.get_mode() is Mode.SPEAKING  # half-duplex: ignora la voz
        assert tts.stopped == 0
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


# --- M3.7: orquestador VoiceLoop (todos los componentes fake) ---
def test_loop_starts_and_stops():
    vl = VoiceLoop(
        FakeAgent("hola, patrón."), FakeStt(), FakeTts(), None,
        mic=FakeMic([]), wake_fn=lambda c: False,
        vad_fn=lambda c: 0.0, to_wav=lambda f: _NO_WAV,
    )
    vl.start()
    time.sleep(0.2)
    vl.stop()
    # Si algún thread no respeta shutdown, esto los caza (deadlock de apagado).
    assert all(not t.is_alive() for t in vl.threads)


# --- M3.8: mitigación de eco (gracia + persistencia) ---
def _speaking_state():
    s = SharedState()
    s.new_turn()
    s.set_mode(Mode.SPEAKING)
    return s


def test_barge_in_persistence_requires_multiple_chunks():
    # Un pico suelto NO dispara (min_chunks=3); una racha sostenida SÍ.
    s1 = _speaking_state()
    mic1 = FakeMic(["voice"])
    ear1, _q, _t, sd1 = _ear(mic1, s1, allow_barge_in=True, grace_ms=0, min_chunks=3)
    ear1.start()
    try:
        assert mic1.agotado.wait(2.0), "el oído no procesó el guion"
        assert s1.get_mode() is Mode.SPEAKING  # un pico no interrumpe
    finally:
        sd1.set()
        ear1.join(timeout=2.0)

    s2 = _speaking_state()
    ear2, _q2, _t2, sd2 = _ear(FakeMic(["voice", "voice", "voice"]), s2,
                               allow_barge_in=True, grace_ms=0, min_chunks=3)
    ear2.start()
    try:
        assert _wait(lambda: s2.get_mode() is Mode.LISTENING)  # racha sí interrumpe
        assert s2.turn_id == 2
    finally:
        sd2.set()
        ear2.join(timeout=2.0)


def test_barge_in_grace_ignores_early_voice():
    # Con gracia amplia, la voz temprana (eco del arranque) se ignora.
    state = _speaking_state()
    mic = FakeMic(["voice"] * 6)
    ear, _q, _t, shutdown = _ear(mic, state, allow_barge_in=True,
                                 grace_ms=5000, min_chunks=1)
    ear.start()
    try:
        assert mic.agotado.wait(2.0), "el oído no procesó el guion"
        assert state.get_mode() is Mode.SPEAKING  # gracia bloquea el barge-in
    finally:
        shutdown.set()
        ear.join(timeout=2.0)


# --- B6 (M3.8): las TRES capas de mitigación de eco, fijadas contra regresión ---
# Si se tunean los números (gracia/margen/persistencia), este test define el
# comportamiento CONJUNTO esperado de _is_patron_voice.

def _ear_para_eco(min_chunks=3, grace_ms=400, margin=0.1):
    """EarThread sin arrancar, con vad_fn que interpreta el chunk como su prob."""
    ear = EarThread(
        FakeMic([]), wake_fn=lambda c: False, vad_fn=lambda c: float(c),
        to_wav=lambda f: _NO_WAV, tts=FakeTts(),
        stt_queue=queue.Queue(), tts_queue=queue.Queue(),
        state=SharedState(), shutdown=threading.Event(),
        allow_barge_in=True, barge_in_grace_ms=grace_ms,
        barge_in_threshold_margin=margin, barge_in_min_chunks=min_chunks,
    )
    return ear


def test_m38_tres_capas_de_mitigacion_de_eco(monkeypatch):
    reloj = {"t": 100.0}
    monkeypatch.setattr("crotolamo.voice.threads.time.monotonic", lambda: reloj["t"])
    ear = _ear_para_eco(min_chunks=3, grace_ms=400, margin=0.1)
    ear._speaking_since = 100.0  # el TTS acaba de arrancar

    # Capa 1 (gracia): dentro de los 400ms ni voz clarísima cuenta.
    for _ in range(10):
        assert ear._is_patron_voice(0.95, Mode.SPEAKING) is False

    reloj["t"] += 0.5  # pasó la gracia

    # Capa 2 (margen): 0.85 pasa el umbral base (0.8) pero NO el elevado
    # de SPEAKING (0.8 + 0.1): el eco inflado no dispara.
    for _ in range(10):
        assert ear._is_patron_voice(0.85, Mode.SPEAKING) is False

    # Capa 3 (persistencia): voz real sostenida; 2 chunks no bastan, el 3º sí.
    assert ear._is_patron_voice(0.95, Mode.SPEAKING) is False
    assert ear._is_patron_voice(0.95, Mode.SPEAKING) is False
    assert ear._is_patron_voice(0.95, Mode.SPEAKING) is True

    # Un pico suelto con silencio en medio NO acumula: la racha se reinicia.
    assert ear._is_patron_voice(0.95, Mode.SPEAKING) is False
    assert ear._is_patron_voice(0.0, Mode.SPEAKING) is False   # rompe la racha
    assert ear._is_patron_voice(0.95, Mode.SPEAKING) is False  # racha vuelve a 1


def test_m38_en_thinking_no_aplica_margen_ni_gracia(monkeypatch):
    """El margen y la gracia son anti-eco del TTS: solo aplican en SPEAKING.
    En THINKING (sin audio saliendo) la voz del patrón entra con el umbral base."""
    reloj = {"t": 100.0}
    monkeypatch.setattr("crotolamo.voice.threads.time.monotonic", lambda: reloj["t"])
    ear = _ear_para_eco(min_chunks=2, grace_ms=400, margin=0.1)
    ear._speaking_since = 100.0  # irrelevante en THINKING

    assert ear._is_patron_voice(0.85, Mode.THINKING) is False  # racha 1/2
    assert ear._is_patron_voice(0.85, Mode.THINKING) is True   # 2/2: dispara


# --- BrainThread en streaming ([voice].stream_speak_concurrent, apagada) -----------
# Causa raíz voice-pipeline-forked-audio-owned-per-call, paso 5 del diseño: la
# primera frase va a la Boca antes de que el LLM termine; las de un turno abortado
# no; el aviso de un turno roto va detrás de lo dicho.
_TURNO_ROTO = "Se me trabaron los cables, patrón. Vuelve a intentarlo."


class _AgenteQueHabla:
    """LLM falso: emite `tokens` por on_token y devuelve `reply`. Se para tras
    `parar_tras` tokens hasta que el test lo suelta; `revienta` lanza al final."""

    def __init__(self, tokens, reply, parar_tras=None, revienta=False) -> None:
        self.tokens = tokens
        self.reply = reply
        self.parar_tras = parar_tras
        self.revienta = revienta
        self.parado = threading.Event()
        self.suelta = threading.Event()
        self.con_on_token: list[bool] = []

    def handle_turn(self, command, on_token=None):
        self.con_on_token.append(on_token is not None)
        for i, token in enumerate(self.tokens):
            if i == self.parar_tras:
                self.parado.set()
                assert self.suelta.wait(5)
            if on_token is not None:
                on_token(token)
        if self.revienta:
            raise RuntimeError("se cayó el hook")
        return self.reply


def _brain(agente, *, stream=True, tts=None):
    state = SharedState()
    turno = state.new_turn()
    cmd_q: queue.Queue = queue.Queue()
    tts_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    brain = BrainThread(agente, cmd_q, tts_q, state, shutdown, stream=stream, tts=tts)
    brain.start()
    cmd_q.put(("saluda", turno))
    return brain, state, tts_q, shutdown


def _vaciar(tts_q: queue.Queue) -> list:
    return [x.text if isinstance(x, Utterance) else x for x in list(tts_q.queue)]


def _parar(brain, shutdown) -> None:
    shutdown.set()
    brain.join(timeout=3.0)
    assert not brain.is_alive()


def test_brain_en_streaming_la_primera_frase_sale_antes_de_que_acabe_el_llm():
    # Se para con "Ya " ya dado: el '.' de "patrón." solo cierra frase al ver
    # lo que le sigue (¿decimal? ¿abreviatura?).
    agente = _AgenteQueHabla(["Hola, ", "patrón. ", "Ya ", "quedó."],
                             "Hola, patrón. Ya quedó.", parar_tras=3)
    brain, state, tts_q, shutdown = _brain(agente)
    try:
        assert agente.parado.wait(3)
        # El LLM sigue "generando" y la primera frase ya está en la cola de la Boca.
        assert _wait(lambda: _vaciar(tts_q) == ["Hola, patrón."])
        assert state.get_mode() is Mode.THINKING
        agente.suelta.set()
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        assert _vaciar(tts_q) == ["Hola, patrón.", "Ya quedó.", END]
        assert all(u.turn_id == 1 for u in list(tts_q.queue) if isinstance(u, Utterance))
        assert state._text == "Hola, patrón. Ya quedó."
        assert agente.con_on_token == [True]
    finally:
        agente.suelta.set()
        _parar(brain, shutdown)


def test_brain_en_streaming_un_turno_abortado_no_encola_mas():
    agente = _AgenteQueHabla(["Hola. ", "Esto ", "ya ", "no."], "Hola. Esto ya no.",
                             parar_tras=2)
    brain, state, tts_q, shutdown = _brain(agente)
    try:
        assert agente.parado.wait(3)
        assert _wait(lambda: _vaciar(tts_q) == ["Hola."])
        state.new_turn()  # barge-in: el turno 1 ya no es el vigente
        agente.suelta.set()
        # La orden del turno vigente sale DESPUÉS de la vieja (FIFO): cuando su
        # END llega, el turno 1 ya terminó de procesarse.
        brain.cmd_q.put(("otra", 2))
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        por_turno = [(u.turn_id, u.text) for u in list(tts_q.queue) if isinstance(u, Utterance)]
        # Del turno 1 solo lo que se encoló antes del barge-in; ni su END.
        assert por_turno == [(1, "Hola."), (2, "Hola."), (2, "Esto ya no.")]
        assert list(tts_q.queue).count(END) == 1
    finally:
        agente.suelta.set()
        _parar(brain, shutdown)


def test_brain_en_streaming_sin_tokens_habla_la_respuesta_entera():
    """Un atajo que no pasa por el LLM (fastpath) no llama a on_token: se habla
    la respuesta entera, como sin streaming."""
    agente = _AgenteQueHabla([], "Listo. Ya.")
    brain, _state, tts_q, shutdown = _brain(agente)
    try:
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        assert _vaciar(tts_q) == ["Listo.", "Ya.", END]
    finally:
        _parar(brain, shutdown)


def test_brain_en_streaming_si_el_turno_revienta_el_aviso_va_detras_de_lo_dicho():
    """Igual que el loop simple: lo que alcanzó a decir (también la cola sin
    cerrar) y detrás el aviso de turno roto."""
    agente = _AgenteQueHabla(["Hola. ", "Ya casi"], "", revienta=True)
    brain, state, tts_q, shutdown = _brain(agente)
    try:
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        assert _vaciar(tts_q) == ["Hola.", "Ya casi", "Se me trabaron los cables, patrón.",
                                  "Vuelve a intentarlo.", END]
        assert state._text == _TURNO_ROTO
    finally:
        _parar(brain, shutdown)


def test_brain_en_streaming_el_aviso_del_llm_caido_va_en_frase_aparte():
    aviso = "GLM no respondió a tiempo."
    agente = _AgenteQueHabla(["La capital ", "de Francia es", "\n" + aviso], aviso)
    brain, _state, tts_q, shutdown = _brain(agente)
    try:
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        assert _vaciar(tts_q) == ["La capital de Francia es", aviso, END]
    finally:
        _parar(brain, shutdown)


def test_brain_en_streaming_respuesta_vacia_sin_nada_dicho_vuelve_a_reposo():
    agente = _AgenteQueHabla([], "")
    brain, state, tts_q, shutdown = _brain(agente)
    try:
        assert _wait(lambda: state.get_mode() is Mode.IDLE)
        assert tts_q.empty()
    finally:
        _parar(brain, shutdown)


def test_brain_pide_la_presintesis_de_cada_frase_en_el_orden_de_hablar():
    from tests.test_speed import _TtsQueAnota

    tts = _TtsQueAnota()
    for stream in (False, True):
        tts.eventos.clear()
        agente = _AgenteQueHabla(["Una. ", "Dos. ", "Tres."], "Una. Dos. Tres.")
        brain, _state, tts_q, shutdown = _brain(agente, stream=stream, tts=tts)
        try:
            assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
            assert tts.eventos == [("pide", "Una."), ("pide", "Dos."), ("pide", "Tres.")]
            assert _vaciar(tts_q) == ["Una.", "Dos.", "Tres.", END]
        finally:
            _parar(brain, shutdown)


def test_brain_por_defecto_no_pasa_on_token_y_el_voiceloop_tampoco():
    agente = _AgenteQueHabla(["Hola. "], "Hola.")
    brain, _state, tts_q, shutdown = _brain(agente, stream=False)
    try:
        assert _wait(lambda: _vaciar(tts_q)[-1:] == [END])
        assert agente.con_on_token == [False]
    finally:
        _parar(brain, shutdown)

    class _Mic:
        def read(self):
            return None

    comunes = dict(mic=_Mic(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
                   to_wav=lambda f: f)
    vl = VoiceLoop(agente, SimpleNamespace(), SimpleNamespace(), None, **comunes)
    assert vl.threads[2].stream is False
    vl = VoiceLoop(agente, SimpleNamespace(), SimpleNamespace(), None, stream_speak=True,
                   **comunes)
    assert vl.threads[2].stream is True
