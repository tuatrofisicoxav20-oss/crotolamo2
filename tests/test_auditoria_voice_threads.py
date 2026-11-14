"""Arreglos de la auditoría en el loop de voz concurrente (voice/state.py,
voice/threads.py y voice/loop.py).

Cada bloque fija un hallazgo confirmado; todo con fakes, sin micrófono ni audio.

- set_mode() sin cambio de modo NO publica: el Brain repite THINKING y el Mouth
  pone SPEAKING antes de CADA frase, y cada publicación es un temporal +
  os.replace + un despertar del HUD por inotify para contar el mismo estado.
  publish_now() sigue siendo el camino forzado.
- Las publicaciones salen en el orden de los cambios (C8): la foto del estado
  se tomaba bajo el lock pero se escribía fuera, así que la foto VIEJA podía
  escribirse la última y dejar al HUD (y al ducker) con un modo rancio hasta el
  siguiente cambio.
- VoiceLoop.stop() cierra el micrófono DESPUÉS de que el oído salga de read()
  (V17): cerrarlo con otra lectura en curso es indefinido en PortAudio. Si la
  lectura está colgada, el cierre desde fuera sigue como último recurso.
- Un barge-in que cae entre la comprobación del turno y el cambio de modo de
  Mouth/Brain/Stt ya no queda pisado (V5): el loop se quedaba pegado en
  "hablando"/"pensando" o perdía la orden nueva. Se prueba el barge-in tras
  CADA operación de estado de cada hilo.
- La orden nueva arranca con la RNN del VAD limpia (V11): con barge-in el VAD
  se alimenta del eco del TTS y ese contexto se colaba en la orden siguiente.
- Una captura sin NINGÚN chunk con voz no va a Whisper (V6): cada falso wake
  costaba ~2 s de pared y ~3 s de CPU transcribiendo ruido normalizado, con el
  loop en "pensando" y sordo al wake.
- Un fallo de audio no mata al oído ni a la boca en silencio (V2): el oído
  aguanta fallos sueltos; tras varios seguidos se rinde y VoiceLoop.run(),
  que vigila a sus hilos, se apaga y devuelve 1 en vez de seguir sordo.
- Una pausa tras el wake no cierra la orden (V1): el silencio no cuenta hasta
  la primera voz (umbral de entrada), la espera tiene su propio límite (4 s,
  como el modo simple) y solo antepone la pre-activación.
- El barge-in manda a Whisper toda la racha del VAD que lo disparó, no solo su
  último chunk (se perdía la primera sílaba de la orden).
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import (
    END,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
    _drain_queue,
)


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class _ScriptMic:
    """Entrega los chunks del guion y luego None (sin audio)."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def read(self):
        try:
            return next(self._it)
        except StopIteration:
            return None


class _SlowStt:
    """Transcribe con una demora, como el STT real: cuando termina, el oído ya
    dejó publicado THINKING (el orden del guion queda determinista)."""

    def transcribe(self, audio, hotwords=None):
        time.sleep(0.05)
        return "abre youtube"


class _Tts:
    def speak(self, text):
        pass

    def stop(self):
        pass


# --- set_mode() sin cambio de modo no publica --------------------------------
def test_set_mode_sin_cambio_no_publica():
    publicado: list[dict] = []
    state = SharedState(publisher=publicado.append)

    state.set_mode(Mode.IDLE)  # ya estaba en IDLE: nada nuevo que contar
    assert publicado == []

    for _ in range(6):  # el Mouth: una llamada por frase de la respuesta
        state.set_mode(Mode.SPEAKING)
    assert [p["mode"] for p in publicado] == ["speaking"]
    assert state.get_mode() is Mode.SPEAKING

    # El camino forzado sigue ahí: publish_now publica aunque nada cambie.
    state.publish_now()
    assert [p["mode"] for p in publicado] == ["speaking", "speaking"]


def test_un_turno_del_loop_no_republica_estados_identicos():
    """Un turno completo (wake, orden, respuesta de 6 frases) publica cada
    cambio UNA vez: antes eran 13 escrituras, 6 de ellas idénticas a la previa
    (el THINKING repetido del Brain y 5 SPEAKING del Mouth)."""
    respuesta = "Uno. Dos. Tres. Cuatro. Cinco. Seis."
    publicado: list[tuple] = []
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: respuesta), _SlowStt(), _Tts(), None,
        mic=_ScriptMic(["wake", "voice", "sil", "sil"]),
        wake_fn=lambda c: c == "wake",
        vad_fn=lambda c: 0.9 if c == "voice" else 0.0,
        to_wav=lambda frames: "WAV", silence_ms=64,
        hud_publisher=lambda d: publicado.append((d["mode"], d["turn_id"], d["text"])),
    )
    vl.start()
    try:
        # El turno termina cuando el Mouth vuelve a IDLE tras la última frase.
        assert _wait(lambda: len(publicado) > 1 and publicado[-1][0] == "idle")
        time.sleep(0.1)  # por si algo más publicara después del cierre
    finally:
        vl.stop()
    assert publicado == [
        ("idle", 1, ""),                 # wake: turno nuevo
        ("listening", 1, ""),
        ("thinking", 1, ""),             # fin de la orden
        ("thinking", 1, "abre youtube"),  # frase reconocida
        ("thinking", 1, respuesta),      # respuesta que se va a hablar
        ("speaking", 1, respuesta),      # UNA vez, no una por frase
        ("idle", 1, respuesta),
    ]


# --- C8: el orden de publicación sigue al orden de los cambios ----------------
def test_dos_cambios_casi_simultaneos_dejan_publicado_el_ultimo():
    """El hilo A cambia a SPEAKING y se queda a medias de escribir; el hilo B
    cambia a IDLE. Lo último que ve el HUD tiene que ser IDLE (el estado real),
    no la foto vieja de A escrita encima."""
    publicado: list[str] = []
    dentro = threading.Event()
    suelta = threading.Event()

    def publisher(d: dict) -> None:
        if d["mode"] == "speaking" and not dentro.is_set():
            dentro.set()
            suelta.wait(5)  # A desalojado entre tomar su foto y escribirla
        publicado.append(d["mode"])

    state = SharedState(publisher=publisher)
    a = threading.Thread(target=state.set_mode, args=(Mode.SPEAKING,), daemon=True)
    a.start()
    assert dentro.wait(2)
    b = threading.Thread(target=state.set_mode, args=(Mode.IDLE,), daemon=True)
    b.start()
    b.join(timeout=0.3)
    # El CAMBIO de B ya es visible para los demás hilos aunque A siga
    # escribiendo: solo la publicación espera su turno, nunca el estado (el
    # oído consulta el modo ~31 veces por segundo).
    assert state.get_mode() is Mode.IDLE
    suelta.set()
    a.join(timeout=2)
    b.join(timeout=2)
    assert not a.is_alive() and not b.is_alive()
    assert state.get_mode() is Mode.IDLE
    assert publicado[-1] == "idle", f"el HUD se quedó con un modo viejo: {publicado}"


def test_el_orden_de_publicacion_sigue_al_de_los_cambios():
    """Cuatro hilos abren turnos a la vez con un publisher que suelta el GIL a
    media publicación (como la escritura a disco): lo publicado nunca retrocede
    y lo último publicado es el turno real."""
    publicado: list[int] = []

    def publisher(d: dict) -> None:
        time.sleep(0.0002)
        publicado.append(d["turn_id"])

    state = SharedState(publisher=publisher)

    def hammer() -> None:
        for _ in range(50):
            state.new_turn()

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.turn_id == 200
    assert publicado == sorted(publicado), "una publicación vieja llegó después de una nueva"
    assert publicado[-1] == 200


def test_un_publisher_que_vuelve_a_entrar_al_estado_no_se_bloquea():
    """El publisher corre sin el lock de estado tomado: puede consultar el
    estado y hasta cambiarlo sin dejar colgado al hilo que publica."""
    visto: list[tuple] = []

    def publisher(d: dict) -> None:
        visto.append((d["mode"], d["text"], state.get_mode().name.lower()))
        if d["mode"] == "listening" and not d["text"]:
            state.set_text("eco")  # reentrada desde el propio publisher

    state = SharedState(publisher=publisher)
    hilo = threading.Thread(target=state.set_mode, args=(Mode.LISTENING,), daemon=True)
    hilo.start()
    hilo.join(timeout=2)
    assert not hilo.is_alive(), "el publisher reentrante dejó colgado a set_mode"
    assert visto == [("listening", "", "listening"), ("listening", "eco", "listening")]
    assert state.current_snapshot()["text"] == "eco"


# --- V17: stop() no cierra el micrófono con el oído DENTRO de read() -----------
class _FrameMic:
    """Micro falso con la forma del real: read() bloquea hasta que llega audio
    (aquí: hasta el apagado) y tarda el resto del frame en volver. Registra si
    close() llegó con una lectura EN CURSO, que con PortAudio es cerrar el
    stream bajo los pies del hilo que está leyendo (comportamiento indefinido)."""

    def __init__(self, tail_s: float = 0.2) -> None:
        self.shutdown = threading.Event()  # el del loop, al armarlo
        self._tail_s = tail_s
        self.reading = False
        self.inside = threading.Event()
        self.closed = threading.Event()
        self.closed_while_reading = False

    def read(self):
        self.reading = True
        self.inside.set()
        try:
            if self.closed.is_set():
                return None
            self.shutdown.wait(5.0)
            time.sleep(self._tail_s)  # el frame en curso no se corta solo
            return None
        finally:
            self.reading = False

    def close(self) -> None:
        if self.reading:
            self.closed_while_reading = True
        self.closed.set()


def _loop_con_mic(mic):
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _SlowStt(), _Tts(), None,
        mic=mic, wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    mic.shutdown = vl.shutdown
    return vl


def test_stop_espera_a_que_el_oido_salga_de_read_antes_de_cerrar_el_mic():
    """El apagado ordenado cerraba el InputStream desde el hilo principal con
    el oído bloqueado dentro de read(): PortAudio no admite cerrar un stream
    mientras otro hilo lee de él. El oído sale de read() en un frame y ve
    shutdown; el mic se cierra después, ya sin nadie leyendo."""
    mic = _FrameMic()
    vl = _loop_con_mic(mic)
    vl.start()
    assert mic.inside.wait(2), "el oído nunca llegó a leer"
    vl.stop()
    assert mic.closed.is_set(), "stop() tiene que seguir cerrando el micrófono"
    assert not mic.closed_while_reading, "close() llegó con el oído dentro de read()"
    assert all(not t.is_alive() for t in vl.threads)


class _HungMic(_FrameMic):
    """Lectura colgada (B1): read() solo vuelve cuando alguien cierra el mic."""

    def read(self):
        self.reading = True
        self.inside.set()
        try:
            self.closed.wait(10.0)
            return None
        finally:
            self.reading = False


def test_stop_sigue_cerrando_el_mic_si_el_oido_no_sale_de_una_lectura_colgada():
    """Último recurso intacto: si el oído no sale de read() dentro del plazo
    del join (driver atorado), stop() cierra el mic desde fuera igual que
    antes, que es lo único que puede desbloquear la lectura."""
    mic = _HungMic()
    vl = _loop_con_mic(mic)
    vl.start()
    assert mic.inside.wait(2)
    vl.stop()
    assert mic.closed.is_set()
    ear = vl.threads[0]
    ear.join(timeout=2.0)
    assert not ear.is_alive(), "cerrar el mic tenía que desbloquear la lectura colgada"


# --- V5: comprobación del turno y cambio de modo, atómicos ---------------------
# Operaciones de estado que cuentan como "un paso" del hilo bajo prueba. Las que
# no existan en SharedState simplemente nunca se llaman.
_OPS_DE_ESTADO = (
    "get_mode", "set_mode", "set_text", "is_current",
    "set_mode_if_current", "set_text_if_current", "compare_and_set_mode",
)


class _Desalojado(SharedState):
    """SharedState que mete el barge-in del oído JUSTO DESPUÉS de la k-ésima
    operación de estado del hilo bajo prueba: es el planificador desalojándolo
    en ese punto exacto. El barge-in corre en OTRO hilo (como el EarThread) y
    se espera a que termine antes de devolverle la CPU al hilo bajo prueba."""

    def armar(self, k: int, barge_in) -> None:
        self._k = k
        self._ops = 0
        self._barge_in = barge_in
        self._dueno = threading.get_ident()
        self.disparado = False

    def _tras_op(self) -> None:
        if getattr(self, "_dueno", None) != threading.get_ident() or self.disparado:
            return
        self._ops += 1
        if self._ops == self._k:
            self.disparar()

    def disparar(self) -> None:
        self.disparado = True
        hilo = threading.Thread(target=self._barge_in, name="Ear")
        hilo.start()
        hilo.join(timeout=2.0)


def _envolver(nombre):
    def op(self, *args):
        resultado = getattr(SharedState, nombre)(self, *args)
        self._tras_op()
        return resultado
    return op


for _nombre in _OPS_DE_ESTADO:
    setattr(_Desalojado, _nombre, _envolver(_nombre))


class _ColaDeUnaPasada(queue.Queue):
    """Al vaciarse apaga al hilo que la consume: run() procesa lo encolado y
    vuelve, todo en el hilo del test y sin esperas por tiempo."""

    def __init__(self, shutdown: threading.Event, items) -> None:
        super().__init__()
        self._shutdown = shutdown
        for item in items:
            self.put(item)

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        if self.empty():
            self._shutdown.set()
        return item


class _TtsQueAnota:
    def __init__(self) -> None:
        self.dicho: list[str] = []
        self.cortes = 0

    def speak(self, text) -> None:
        self.dicho.append(text)

    def stop(self) -> None:
        self.cortes += 1


def _barge_in_del_oido(state, tts, tts_q):
    """Lo que hace EarThread.run al detectar la voz del patrón en THINKING o
    SPEAKING (corta el audio, tira las frases, turno nuevo y a escuchar)."""
    def barge_in() -> None:
        tts.stop()
        _drain_queue(tts_q)
        state.new_turn()
        state.set_mode(Mode.LISTENING)
    return barge_in


def _tras_el_barge_in(state) -> None:
    """Pase lo que pase con el turno viejo, manda el del oído: escuchando la
    orden nueva y sin el texto de la interacción anterior en el HUD."""
    assert state.disparado
    assert state.turn_id == 2
    assert state.get_mode() is Mode.LISTENING, (
        f"el turno viejo pisó el LISTENING del barge-in: {state.get_mode()}")
    assert state.current_snapshot()["text"] == ""


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_barge_in_en_cualquier_paso_del_mouth_deja_escuchando(k):
    """Antes: tras el is_current() de una frase, el barge-in ponía LISTENING y
    el Mouth lo pisaba con SPEAKING; el END ya estaba drenado y nadie volvía a
    IDLE (HUD en "hablando", música bajada, wake ignorado). O el END miraba
    SPEAKING, llegaba el barge-in y el IDLE del Mouth tiraba la orden nueva."""
    shutdown = threading.Event()
    tts = _TtsQueAnota()
    state = _Desalojado()
    state.new_turn()
    state.set_mode(Mode.THINKING)
    state.set_text("Primera frase. Segunda.")
    tts_q = _ColaDeUnaPasada(shutdown, [Utterance("Primera frase.", 1), END])
    state.armar(k, _barge_in_del_oido(state, tts, tts_q))

    MouthThread(tts, tts_q, state, shutdown).run()
    if not state.disparado:  # el Mouth ya terminó: el barge-in llega después
        state.disparar()
    _tras_el_barge_in(state)


@pytest.mark.parametrize("respuesta", ["Respuesta del turno viejo.", "  "])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_barge_in_en_cualquier_paso_del_brain_deja_escuchando(k, respuesta):
    """Antes: tras el is_current() del comando, el barge-in ponía LISTENING y el
    Brain lo pisaba con THINKING; el oído abandonaba la orden nueva y, al ver
    el turno cambiado, el Brain salía sin tocar el modo: "pensando" para
    siempre. Con respuesta vacía, su IDLE tiraba la orden; con texto, la
    respuesta vieja volvía a pintarse en el HUD bajo "escuchando"."""
    shutdown = threading.Event()
    tts = _TtsQueAnota()
    state = _Desalojado()
    state.new_turn()
    state.set_mode(Mode.THINKING)  # como lo deja el oído al cerrar la orden
    cmd_q = _ColaDeUnaPasada(shutdown, [("orden vieja", 1)])
    tts_q: queue.Queue = queue.Queue()
    state.armar(k, _barge_in_del_oido(state, tts, tts_q))

    agente = SimpleNamespace(handle_turn=lambda c: respuesta)
    BrainThread(agente, cmd_q, tts_q, state, shutdown).run()
    if not state.disparado:
        state.disparar()
    _tras_el_barge_in(state)


@pytest.mark.parametrize("texto", ["abre youtube", ""])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_barge_in_en_cualquier_paso_del_stt_deja_escuchando(k, texto):
    """Antes: con transcripción vacía, el is_current() pasaba, llegaba el
    barge-in y el IDLE del Stt tiraba la orden nueva; con texto, la frase
    vieja volvía a pintarse en el HUD bajo "escuchando"."""
    shutdown = threading.Event()
    tts = _TtsQueAnota()
    state = _Desalojado()
    state.new_turn()
    state.set_mode(Mode.THINKING)
    stt_q = _ColaDeUnaPasada(shutdown, [("AUDIO", 1)])
    cmd_q: queue.Queue = queue.Queue()
    state.armar(k, _barge_in_del_oido(state, tts, queue.Queue()))

    stt = SimpleNamespace(transcribe=lambda audio, hotwords=None: texto)
    SttThread(stt, stt_q, cmd_q, state, shutdown).run()
    if not state.disparado:
        state.disparar()
    _tras_el_barge_in(state)


def test_set_mode_if_current_comprueba_y_cambia_bajo_el_mismo_lock():
    publicado: list[str] = []
    state = SharedState(publisher=lambda d: publicado.append(d["mode"]))
    state.new_turn()
    state.new_turn()  # vigente: 2
    publicado.clear()

    assert state.set_mode_if_current(1, Mode.SPEAKING) is False  # turno viejo
    assert state.get_mode() is Mode.IDLE and publicado == []

    assert state.set_mode_if_current(2, Mode.SPEAKING) is True
    assert state.set_mode_if_current(2, Mode.SPEAKING) is True  # sin cambio
    assert state.get_mode() is Mode.SPEAKING and publicado == ["speaking"]


def test_set_text_if_current_no_repinta_el_texto_de_un_turno_abortado():
    publicado: list[str] = []
    state = SharedState(publisher=lambda d: publicado.append(d["text"]))
    state.new_turn()
    state.new_turn()
    publicado.clear()

    assert state.set_text_if_current(1, "respuesta vieja") is False
    assert state.current_snapshot()["text"] == "" and publicado == []
    assert state.set_text_if_current(2, "abre youtube") is True
    assert publicado == ["abre youtube"]


def test_compare_and_set_mode_no_pisa_un_modo_distinto_del_esperado():
    publicado: list[str] = []
    state = SharedState(publisher=lambda d: publicado.append(d["mode"]))
    state.set_mode(Mode.LISTENING)  # el barge-in llegó antes que el END
    publicado.clear()

    assert state.compare_and_set_mode(Mode.SPEAKING, Mode.IDLE) is False
    assert state.get_mode() is Mode.LISTENING and publicado == []

    state.set_mode(Mode.SPEAKING)
    assert state.compare_and_set_mode(Mode.SPEAKING, Mode.IDLE) is True
    assert state.get_mode() is Mode.IDLE and publicado == ["speaking", "idle"]


# --- V11: la orden nueva arranca con la RNN del VAD limpia ---------------------
class _VadConMemoria:
    """VAD falso con reset(), como _SileroVad: anota cada inferencia y cada
    limpieza de la RNN, en orden."""

    def __init__(self) -> None:
        self.eventos: list[str] = []

    def __call__(self, chunk) -> float:
        self.eventos.append(f"vad:{chunk}")
        return 0.95 if chunk in ("eco", "voz") else 0.0

    def reset(self) -> None:
        self.eventos.append("reset")


class _MicDeCola:
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()

    def read(self):
        try:
            return self.q.get(timeout=0.02)
        except queue.Empty:
            return None


def _oido_con_vad(vad, state, mic, *, min_chunks):
    shutdown = threading.Event()
    ear = EarThread(
        mic, wake_fn=lambda c: c == "wake", vad_fn=vad, to_wav=lambda f: "WAV",
        tts=_TtsQueAnota(), stt_queue=queue.Queue(), tts_queue=queue.Queue(),
        state=state, shutdown=shutdown, allow_barge_in=True,
        barge_in_grace_ms=0, barge_in_threshold_margin=0.0,
        barge_in_min_chunks=min_chunks,
    )
    ear._post_speak_grace_s = 0.0
    return ear, shutdown


def _primera_inferencia_de_la_orden(vad, chunk_de_la_orden):
    eventos = vad.eventos
    return eventos.index(f"vad:{chunk_de_la_orden}")


def test_barge_in_arranca_la_orden_con_el_vad_limpio():
    """Con barge-in el VAD se alimenta del eco del TTS mientras habla; la orden
    que abre el barge-in tiene que clasificarse sin ese contexto."""
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    vad = _VadConMemoria()
    mic = _MicDeCola()
    ear, shutdown = _oido_con_vad(vad, state, mic, min_chunks=2)
    ear.start()
    try:
        for chunk in ("eco", "eco"):  # el patrón interrumpe (racha de 2)
            mic.q.put(chunk)
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        mic.q.put("voz")  # primer chunk de la orden nueva
        assert _wait(lambda: "vad:voz" in vad.eventos)
    finally:
        shutdown.set()
        ear.join(timeout=2.0)
    i = _primera_inferencia_de_la_orden(vad, "voz")
    assert "reset" in vad.eventos[:i], f"la orden heredó la RNN del eco: {vad.eventos}"
    assert vad.eventos.index("reset") > vad.eventos.index("vad:eco")


def test_el_wake_siguiente_no_hereda_la_rnn_del_eco_del_turno_anterior():
    """El VAD solo se limpiaba al CERRAR la orden; con barge-in luego se sigue
    alimentando con el eco del TTS hasta volver a IDLE, y ese estado llegaba
    intacto a la orden del wake siguiente."""
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    vad = _VadConMemoria()
    mic = _MicDeCola()
    ear, shutdown = _oido_con_vad(vad, state, mic, min_chunks=1000)  # sin barge-in
    ear.start()
    try:
        for _ in range(3):
            mic.q.put("eco")  # Crotolamo habla; el micro capta su voz
        assert _wait(lambda: vad.eventos.count("vad:eco") == 3)
        state.set_mode(Mode.IDLE)  # fin de la respuesta (END del Mouth)
        mic.q.put("sil")  # el oído ve IDLE
        mic.q.put("wake")
        assert _wait(lambda: state.get_mode() is Mode.LISTENING)
        mic.q.put("voz")
        assert _wait(lambda: "vad:voz" in vad.eventos)
    finally:
        shutdown.set()
        ear.join(timeout=2.0)
    i = _primera_inferencia_de_la_orden(vad, "voz")
    ultimo_eco = max(j for j, e in enumerate(vad.eventos) if e == "vad:eco")
    assert "reset" in vad.eventos[ultimo_eco:i], (
        f"la orden del wake heredó la RNN del eco: {vad.eventos}")


# --- V6: una captura sin voz no va a Whisper -----------------------------------
class _MicDeUnaPasada:
    """Entrega el guion y, agotado, apaga al oído: run() corre entero en el
    hilo del test, sin esperas por tiempo. Un elemento Exception se LANZA en
    esa lectura (PortAudioError al desenchufar o suspender el micro)."""

    def __init__(self, shutdown: threading.Event, guion) -> None:
        self._shutdown = shutdown
        self._guion = list(guion)
        self.lecturas = 0

    def read(self):
        self.lecturas += 1
        if not self._guion:
            self._shutdown.set()
            return None
        item = self._guion.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _ToWavQueCuenta:
    def __init__(self) -> None:
        self.llamadas = 0

    def __call__(self, frames):
        self.llamadas += 1
        return "WAV"


def _oido_de_una_pasada(guion, state, *, vad=None, wake_fn=None, allow_barge_in=False,
                        **kw):
    shutdown = threading.Event()
    mic = _MicDeUnaPasada(shutdown, guion)
    stt_q: queue.Queue = queue.Queue()
    to_wav = _ToWavQueCuenta()
    ear = EarThread(
        mic, wake_fn=wake_fn or (lambda c: c == "wake"), vad_fn=vad or _VadConMemoria(),
        to_wav=to_wav, tts=_TtsQueAnota(), stt_queue=stt_q, tts_queue=queue.Queue(),
        state=state, shutdown=shutdown, silence_ms=64, allow_barge_in=allow_barge_in,
        barge_in_grace_ms=0, barge_in_threshold_margin=0.0, barge_in_min_chunks=1, **kw,
    )
    return ear, mic, stt_q, to_wav


def test_un_falso_wake_sin_voz_no_manda_audio_a_whisper():
    """Wake y luego solo ruido de la habitación: antes el oído armaba el audio,
    lo encolaba para Whisper (que normaliza ese ruido a pico 0.9 y alucina
    sobre él, ~2 s de pared) y dejaba el loop en "pensando", sordo al wake.
    Sin ningún chunk con voz no hay orden: directo a reposo."""
    publicado: list[str] = []
    state = SharedState(publisher=lambda d: publicado.append(d["mode"]))
    vad = _VadConMemoria()
    # Sin voz tras el wake la captura se cierra al expirar la espera de la orden
    # (V1): 0.096 s = los 3 chunks de silencio del guion.
    ear, _mic, stt_q, to_wav = _oido_de_una_pasada(["wake", "sil", "sil", "sil"], state,
                                                  vad=vad, start_timeout_s=0.096)
    ear.run()

    assert to_wav.llamadas == 0, "armó el audio de una captura sin voz"
    assert stt_q.empty(), "mandó a Whisper una captura sin ningún chunk con voz"
    assert state.get_mode() is Mode.IDLE
    assert publicado == ["idle", "listening", "idle"]  # turno nuevo, escucha, reposo
    # Cerrar la captura sigue limpiando la RNN del VAD, como al cerrar una orden.
    assert vad.eventos[-1] == "reset", vad.eventos


def test_la_orden_con_voz_sigue_yendo_a_whisper():
    state = SharedState()
    ear, _mic, stt_q, to_wav = _oido_de_una_pasada(["wake", "voz", "sil", "sil"], state)
    ear.run()

    assert to_wav.llamadas == 1
    assert stt_q.get_nowait() == ("WAV", 1)
    assert state.get_mode() is Mode.THINKING


def test_un_barge_in_va_a_whisper_aunque_el_patron_calle_despues():
    """El chunk que dispara el barge-in YA es voz del patrón (la racha del VAD):
    esa captura sí tiene voz y se transcribe, como siempre."""
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    ear, _mic, stt_q, to_wav = _oido_de_una_pasada(["voz", "sil", "sil"], state,
                                                  allow_barge_in=True)
    ear.run()

    assert to_wav.llamadas == 1
    assert stt_q.get_nowait() == ("WAV", 2)
    assert state.get_mode() is Mode.THINKING


# --- V2: un fallo de audio no deja al loop sordo ni mudo -----------------------
_FALLO_MIC = OSError("PortAudioError: Device unavailable [PaErrorCode -9985]")


def test_un_fallo_suelto_del_micro_no_deja_sordo_al_oido():
    """Antes la excepción de mic.read() salía de EarThread.run y el hilo moría:
    el proceso seguía vivo, el HUD en reposo y nadie volvía a oír el wake."""
    state = SharedState()
    ear, mic, _stt_q, _to_wav = _oido_de_una_pasada([_FALLO_MIC, "wake"], state)
    ear._PAUSA_TRAS_FALLO_S = 0.0
    ear.run()  # antes: OSError hasta aquí

    assert state.get_mode() is Mode.LISTENING, "el oído no siguió escuchando tras el fallo"
    assert state.turn_id == 1
    assert mic.lecturas == 3  # fallo, wake y el fin del guion


def test_un_fallo_suelto_del_detector_de_wake_no_deja_sordo_al_oido():
    """openWakeWord se carga perezoso DENTRO del oído, en su primer chunk: un
    VoiceUnavailable o un error de onnxruntime ahí también mataba el hilo."""
    def wake_fn(chunk) -> bool:
        if chunk == "boom":
            raise RuntimeError("onnxruntime reventó")
        return chunk == "wake"

    state = SharedState()
    ear, _mic, _stt_q, _to_wav = _oido_de_una_pasada(["boom", "wake"], state,
                                                    wake_fn=wake_fn)
    ear._PAUSA_TRAS_FALLO_S = 0.0
    ear.run()

    assert state.get_mode() is Mode.LISTENING
    assert state.turn_id == 1


def test_fallos_seguidos_hacen_que_el_oido_se_rinda_sin_quemar_cpu():
    """Un stream muerto revienta en CADA lectura y al instante: el oído espera
    entre intentos y, tras un número fijo de fallos SEGUIDOS, se rinde (el
    hilo termina) para que el loop lo note y el servicio se reinicie."""
    n = EarThread._MAX_FALLOS_SEGUIDOS
    state = SharedState()
    ear, mic, _stt_q, _to_wav = _oido_de_una_pasada([_FALLO_MIC] * (n + 50), state)
    ear._PAUSA_TRAS_FALLO_S = 0.01
    inicio = time.monotonic()
    ear.run()  # vuelve solo: nadie puso shutdown
    transcurrido = time.monotonic() - inicio

    assert mic.lecturas == n, f"siguió leyendo tras rendirse: {mic.lecturas} lecturas"
    assert transcurrido >= (n - 1) * 0.01, "reintentó sin pausa (busy-spin)"
    assert not ear.shutdown.is_set(), "rendirse no es apagar: eso lo decide VoiceLoop.run()"


def test_un_chunk_bueno_entre_fallos_reinicia_la_cuenta():
    """Solo cuentan los fallos SEGUIDOS: errores esporádicos separados por
    audio bueno no tumban al oído."""
    n = EarThread._MAX_FALLOS_SEGUIDOS
    state = SharedState()
    guion = [_FALLO_MIC] * (n - 1) + ["sil"] + [_FALLO_MIC] * (n - 1) + ["wake"]
    ear, _mic, _stt_q, _to_wav = _oido_de_una_pasada(guion, state)
    ear._PAUSA_TRAS_FALLO_S = 0.0
    ear.run()

    assert state.get_mode() is Mode.LISTENING
    assert state.turn_id == 1


class _MicMuerto:
    def __init__(self) -> None:
        self.lecturas = 0

    def read(self):
        self.lecturas += 1
        raise OSError("PortAudioError: Device unavailable [PaErrorCode -9985]")


def _correr_loop(vl, timeout: float = 5.0) -> list:
    resultado: list = []
    hilo = threading.Thread(target=lambda: resultado.append(vl.run()), daemon=True)
    hilo.start()
    hilo.join(timeout=timeout)
    if hilo.is_alive():  # limpieza: que el test no deje hilos colgados
        vl.shutdown.set()
        hilo.join(timeout=3.0)
    return resultado


def test_voice_loop_se_apaga_con_codigo_1_si_el_oido_muere():
    """Antes VoiceLoop.run() solo miraba shutdown: con el oído muerto seguía
    corriendo para siempre, sordo, y systemd veía el servicio sano. Ahora
    vigila a sus hilos y, si uno muere, apaga todo y devuelve 1 (el listener lo
    usa como código de salida: Restart=on-failure reinicia el servicio)."""
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _SlowStt(), _Tts(), None,
        mic=_MicMuerto(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    vl._ear._PAUSA_TRAS_FALLO_S = 0.01
    resultado = _correr_loop(vl)

    assert resultado == [1], "el loop siguió corriendo sordo con el oído muerto"
    assert vl.shutdown.is_set()
    assert all(not t.is_alive() for t in vl.threads)


def test_voice_loop_apagado_por_las_buenas_devuelve_0():
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _SlowStt(), _Tts(), None,
        mic=_ScriptMic([]), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    threading.Timer(0.2, vl.shutdown.set).start()
    assert _correr_loop(vl) == [0]
    assert all(not t.is_alive() for t in vl.threads)


class _TtsQueRevienta(_TtsQueAnota):
    def speak(self, text) -> None:
        if text == "Rota.":
            raise RuntimeError("piper reventó")
        super().speak(text)


def test_una_frase_que_revienta_no_mata_a_la_boca():
    """Sin guarda alrededor de tts.speak, una frase rota mataba al Mouth: el
    resto de la respuesta no sonaba y el loop se quedaba en "hablando"."""
    shutdown = threading.Event()
    tts = _TtsQueRevienta()
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.THINKING)
    tts_q = _ColaDeUnaPasada(shutdown, [Utterance("Rota.", 1), Utterance("Sana.", 1), END])

    MouthThread(tts, tts_q, state, shutdown).run()  # antes: RuntimeError hasta aquí

    assert tts.dicho == ["Sana."]
    assert state.get_mode() is Mode.IDLE


# --- V1: una pausa tras el wake no cierra la orden ----------------------------
# Tras "crotolamo" la captura ESPERA la orden: el silencio no cuenta hasta el
# primer chunk con voz (umbral de ENTRADA, sin histéresis) y la espera tiene su
# propio límite, como el start_timeout_s de stt._record_silero en el modo simple.
class _ToWavQueGuarda:
    def __init__(self) -> None:
        self.capturas: list[list] = []

    def __call__(self, frames):
        self.capturas.append(list(frames))
        return "WAV"


def _vad_por_etiqueta(chunk) -> float:
    # "voz" clarísima, "ruido" entre el umbral de mantenerse (0.35) y el de
    # entrar (0.8), y cualquier otra cosa (silencio de la habitación) en 0.
    if str(chunk).startswith("voz"):
        return 0.95
    return 0.5 if str(chunk).startswith("ruido") else 0.0


def _oido_v1(guion, state, **kw):
    """Oído de una pasada (corre entero en el hilo del test) con el VAD por
    etiquetas y la histéresis de la config real (0.8 / 0.35)."""
    shutdown = threading.Event()
    mic = _MicDeUnaPasada(shutdown, guion)
    stt_q: queue.Queue = queue.Queue()
    to_wav = _ToWavQueGuarda()
    ear = EarThread(
        mic, wake_fn=lambda c: c == "wake", vad_fn=_vad_por_etiqueta, to_wav=to_wav,
        tts=_TtsQueAnota(), stt_queue=stt_q, tts_queue=queue.Queue(), state=state,
        shutdown=shutdown, vad_threshold=0.8, vad_neg_threshold=0.35, **kw,
    )
    return ear, stt_q, to_wav


def test_una_pausa_tras_el_wake_no_cierra_la_orden():
    """El patrón dice "crotolamo", respira 0.7 s y da la orden. Antes el
    silencio contaba desde el chunk del wake: con vad_silence_ms=600 la captura
    se cerraba VACÍA a los ~0.6 s y la orden, dicha justo después, se perdía
    (el oído ya no estaba en LISTENING y luego venía la gracia anti-eco)."""
    state = SharedState()
    pausa = [f"sil{i}" for i in range(22)]  # 0.7 s de silencio tras el wake
    orden = [f"voz{i}" for i in range(5)]
    cola = [f"fin{i}" for i in range(19)]  # 0.6 s de silencio: fin de la orden
    ear, stt_q, to_wav = _oido_v1(["wake", *pausa, *orden, *cola], state,
                                  silence_ms=600)
    ear.run()

    assert stt_q.get_nowait() == ("WAV", 1), "la orden dicha tras la pausa se perdió"
    assert state.get_mode() is Mode.THINKING
    (frames,) = to_wav.capturas
    assert frames[-len(orden) - 18:-18] == orden, frames


def test_sin_orden_tras_el_wake_la_espera_expira_y_vuelve_a_reposo():
    """Si tras el wake nadie habla, la espera tiene su propio límite y vuelve a
    reposo sin mandar nada a Whisper."""
    state = SharedState()
    # start_timeout_s=0.32 con chunks de 32 ms -> 10 chunks de espera.
    ear, stt_q, to_wav = _oido_v1(["wake"] + ["sil"] * 9, state, start_timeout_s=0.32)
    ear.run()
    assert state.get_mode() is Mode.LISTENING, "cerró la espera antes de su límite"

    state2 = SharedState()
    ear2, stt_q2, to_wav2 = _oido_v1(["wake"] + ["sil"] * 10, state2, start_timeout_s=0.32)
    ear2.run()
    assert state2.get_mode() is Mode.IDLE, "la espera sin voz no expiró"
    assert stt_q2.empty() and to_wav2.capturas == []


def test_el_ruido_bajo_el_umbral_de_entrada_no_abre_la_orden():
    """Con la histéresis (0.35) cualquier ruido >= 0.35 contaba como voz desde
    el primer chunk y mantenía abierta la captura hasta los 12 s. Para EMPEZAR
    la orden hace falta el umbral de entrada (0.8), como en el modo simple."""
    state = SharedState()
    # Límite de espera por defecto: 4 s (125 chunks), el del modo simple.
    ear, stt_q, to_wav = _oido_v1(["wake"] + ["ruido"] * 125, state)
    ear.run()

    assert state.get_mode() is Mode.IDLE, "el ruido mantuvo abierta la espera"
    assert stt_q.empty() and to_wav.capturas == []


def test_una_espera_larga_solo_manda_la_preactivacion_a_whisper():
    """Lo esperado en silencio no se acumula entero: como el buffer de
    pre-activación del modo simple, se anteponen solo los últimos
    preactivation_ms a la orden (no 4 s de ruido de la habitación)."""
    state = SharedState()
    pausa = [f"sil{i}" for i in range(40)]
    # preactivation_ms=96 -> 3 chunks de pre-activación.
    ear, stt_q, to_wav = _oido_v1(["wake", *pausa, "voz0", "voz1", "fin0", "fin1"], state,
                                  silence_ms=64, preactivation_ms=96)
    ear.run()

    assert stt_q.get_nowait() == ("WAV", 1)
    assert to_wav.capturas == [["sil37", "sil38", "sil39", "voz0", "voz1", "fin0", "fin1"]]


def test_la_orden_sin_pausa_conserva_todo_desde_el_wake():
    """El caso de siempre (orden de corrido tras el wake) manda a Whisper
    exactamente lo mismo que antes: el chunk del wake y todo lo que sigue."""
    state = SharedState()
    ear, stt_q, to_wav = _oido_v1(["wake", "sil0", "voz0", "voz1", "fin0", "fin1"], state,
                                  silence_ms=64)
    ear.run()

    assert stt_q.get_nowait() == ("WAV", 1)
    assert to_wav.capturas == [["wake", "sil0", "voz0", "voz1", "fin0", "fin1"]]


def test_tras_la_primera_voz_el_silencio_cuenta_como_siempre():
    state = SharedState()
    ear, stt_q, to_wav = _oido_v1(["wake", "voz0", "ruido", "sil0", "sil1"], state,
                                  silence_ms=64)
    ear.run()
    # "ruido" (0.5 >= 0.35) mantiene la voz ya empezada; dos silencios la cierran.
    assert stt_q.get_nowait() == ("WAV", 1)
    assert to_wav.capturas == [["wake", "voz0", "ruido", "sil0", "sil1"]]


def test_voice_loop_pasa_la_preactivacion_de_la_config(monkeypatch):
    from crotolamo import settings as settings_mod

    voz = {"vad_preactivation_ms": 320}
    monkeypatch.setattr(settings_mod, "get_settings", lambda: SimpleNamespace(voice=voz))
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _SlowStt(), _Tts(), None,
        mic=_ScriptMic([]), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda frames: "WAV",
    )
    assert vl._ear._pre_chunks == 10  # 320 ms / 32 ms


# --- barge-in: la racha del VAD también es la orden ---------------------------
def test_el_barge_in_conserva_toda_la_racha_que_lo_disparo():
    """El barge-in exige barge_in_min_chunks chunks de voz SEGUIDOS, pero solo
    el último llegaba a Whisper: los primeros 128 ms de la orden (con 5) se
    tiraban y la primera sílaba salía cortada."""
    state = SharedState()
    state.new_turn()
    state.set_mode(Mode.SPEAKING)
    shutdown = threading.Event()
    racha = [f"voz{i}" for i in range(5)]
    mic = _MicDeUnaPasada(shutdown, ["eco0", "voz-suelta", "sil", *racha, "voz5", "fin0",
                                     "fin1"])
    stt_q: queue.Queue = queue.Queue()
    to_wav = _ToWavQueGuarda()
    ear = EarThread(
        mic, wake_fn=lambda c: False, vad_fn=_vad_por_etiqueta, to_wav=to_wav,
        tts=_TtsQueAnota(), stt_queue=stt_q, tts_queue=queue.Queue(), state=state,
        shutdown=shutdown, silence_ms=64, allow_barge_in=True, barge_in_grace_ms=0,
        barge_in_threshold_margin=0.0, barge_in_min_chunks=5,
    )
    ear.run()

    assert stt_q.get_nowait() == ("WAV", 2)
    # Ni el pico suelto (la racha se rompió con "sil") ni el eco previo.
    assert to_wav.capturas == [[*racha, "voz5", "fin0", "fin1"]]
