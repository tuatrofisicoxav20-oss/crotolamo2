"""Threads del loop de voz concurrente (extraídos de loop.py, de GLaDOS, redo M3).

Arquitectura race-free: cuatro threads (Ear/Stt/Brain/Mouth) coordinados por colas
y un SharedState con turn_id monótono. Cada frase nace con su turn_id; el MouthThread
descarta las de turnos abortados. La interrupción es correcta POR CONSTRUCCIÓN: nada
de Events de "interrupción" que se limpian (races); toda invalidación pasa por
state.new_turn() + state.is_current().

Se construye por sub-fases (M3.3 Mouth, M3.4 Brain, M3.5 Stt, M3.6 Ear).
El orquestador VoiceLoop vive en loop.py; los adapters reales en adapters.py.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial

from crotolamo.logging_setup import get_logger
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.vad import DEFAULTS, FIN, SIN_VOZ, Endpointer, resolve_neg_threshold
from crotolamo.voice.tts import SentenceFeeder, _AvisoEnFraseAparte, split_sentences

log = get_logger("voice.loop")

END = object()  # sentinel: fin de la respuesta de un turno
STOP = object()  # sentinel: apagado (lo mete en cada cola el Wakeable de VoiceLoop)


def _get_timeout(shutdown) -> float | None:
    """Timeout de la espera en cola de Mouth/Brain/Stt.

    Con el apagado de VoiceLoop (notify.Wakeable) la cola se espera SIN
    timeout: set() mete un STOP en ella y el hilo sale al instante. Antes la
    espera expiraba cada 0.2 s solo para releer el flag (900 despertares por
    minuto entre los tres, en reposo). Con un threading.Event normal (hilos
    armados sueltos, como en los tests) nadie mete el STOP: se conserva el
    sondeo y su contrato de salir en menos de 2 s.
    """
    return None if getattr(shutdown, "wakes_waiters", False) else 0.2


@dataclass
class Utterance:
    text: str
    turn_id: int


class MouthThread(threading.Thread):
    """Habla las frases de la cola, descartando las de turnos viejos (M3.3)."""

    def __init__(self, tts, tts_queue: queue.Queue, state: SharedState,
                 shutdown: threading.Event) -> None:
        super().__init__(name="Mouth", daemon=True)
        self.tts = tts
        self.q = tts_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        timeout = _get_timeout(self.shutdown)
        while not self.shutdown.is_set():
            try:
                item = self.q.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is STOP:
                return
            if item is END:
                # Terminó la respuesta de un turno; si nadie habló nuevo, a IDLE.
                # Comprobar y cambiar de una vez (V5): mirar SPEAKING y luego
                # poner IDLE pisaba el LISTENING de un barge-in recién llegado.
                self.state.compare_and_set_mode(Mode.SPEAKING, Mode.IDLE)
                continue
            assert isinstance(item, Utterance)
            # Descarta frases de turnos abortados (barge-in o comando nuevo).
            # Turno y SPEAKING de una vez (V5): un barge-in entre los dos
            # quedaba pisado y, con su END ya drenado, el loop se quedaba en
            # "hablando".
            if not self.state.set_mode_if_current(item.turn_id, Mode.SPEAKING):
                continue
            try:
                self.tts.speak(item.text)  # interrumpible (M3.1)
            except Exception as error:  # noqa: BLE001 - una frase rota no mata la boca
                # Sin guarda, la excepción mataba al Mouth (V2): el resto de la
                # respuesta no sonaba y el loop se quedaba en "hablando".
                log.warning("mouth: %s", error)


class BrainThread(threading.Thread):
    """Saca comandos, llama al agente y encola las frases con su turn_id (M3.4).

    Nota honesta: agent.handle_turn llama al LLM por HTTP de forma BLOQUEANTE; no se
    puede cancelar a media inferencia. Si hay barge-in mientras el LLM piensa, la
    inferencia vieja termina en background pero su salida se DESCARTA por el check del
    turno (set_text_if_current); el audio para al instante. Cancelar el LLM en vuelo es mejora
    futura (streaming con check de turno).

    stream ([voice].stream_speak_concurrent, false de fábrica): como el loop simple
    con stream_speak, cada frase va a la Boca EN CUANTO el LLM la cierra (la
    misma SentenceFeeder), en vez de esperar la respuesta entera; las de un
    turno que ya no es el vigente no se encolan. tts: solo para pedir la
    pre-síntesis de cada frase encolada ([voice].tts_presynth); None = nada.
    """

    def __init__(self, agent, command_queue: queue.Queue, tts_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event, *,
                 stream: bool = False, tts=None) -> None:
        super().__init__(name="Brain", daemon=True)
        self.agent = agent
        self.cmd_q = command_queue
        self.tts_q = tts_queue
        self.state = state
        self.shutdown = shutdown
        self.stream = stream
        self._prepare = getattr(tts, "prepare", None)

    def _encolar(self, sentence: str, turn: int) -> None:
        if self._prepare is not None:
            self._prepare(sentence)
        self.tts_q.put(Utterance(sentence, turn))

    def _encolar_si_vigente(self, turn: int, sentence: str) -> None:
        """emit del streaming: una frase de un turno abortado ya no va a la Boca."""
        if self.state.is_current(turn):
            self._encolar(sentence, turn)

    def run(self) -> None:
        timeout = _get_timeout(self.shutdown)
        while not self.shutdown.is_set():
            try:
                item = self.cmd_q.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is STOP:
                return
            command, turn = item
            # El turno ya cambió (barge-in mientras esperaba en cola): ignora.
            # Turno y THINKING de una vez (V5): un barge-in entre los dos
            # quedaba pisado y el loop se quedaba en "pensando" para siempre.
            if not self.state.set_mode_if_current(turn, Mode.THINKING):
                continue
            feeder: SentenceFeeder | None = None
            on_token: _AvisoEnFraseAparte | None = None
            roto = False
            try:
                if self.stream:
                    feeder = SentenceFeeder(partial(self._encolar_si_vigente, turn))
                    # El aviso de un LLM que se cae a media respuesta, en frase
                    # aparte (lo mismo que el loop simple).
                    on_token = _AvisoEnFraseAparte(feeder.feed)
                    reply = self.agent.handle_turn(command, on_token=on_token)
                else:
                    reply = self.agent.handle_turn(command)
            except Exception as error:  # noqa: BLE001 - un turno roto no mata el loop
                log.warning("brain: %s", error)
                # CLAVE: si el LLM falla, hablamos un error y volvemos a reposo.
                # Antes hacía `continue` sin tocar el modo -> el HUD se quedaba
                # PEGADO en "pensando" para siempre.
                reply = "Se me trabaron los cables, patrón. Vuelve a intentarlo."
                roto = True
            hablado = False
            if feeder is not None and on_token is not None:
                # Fin del streaming: lo retenido y la cola que no cerró frase.
                on_token.cerrar(reply or "")
                feeder.flush()
                hablado = feeder.fed > 0
            # Si interrumpieron mientras pensaba, el turno nuevo ya maneja el
            # estado: los *_if_current no tocan nada de un turno abortado (V5).
            # Respuesta vacía: volver a reposo en vez de dejar el HUD en "pensando".
            if not (reply and reply.strip()):
                if not hablado:
                    self.state.set_mode_if_current(turn, Mode.IDLE)
                    continue
            # HUD: respuesta que se va a hablar.
            elif not self.state.set_text_if_current(turn, reply):
                continue
            # En streaming ya se encoló lo dicho; la respuesta entera solo si no
            # llegó nada por on_token (un atajo que no pasa por el LLM) o si el
            # turno reventó: el aviso va detrás de lo que alcanzó a decir.
            if roto or not hablado:
                for sentence in split_sentences(reply):
                    self._encolar(sentence, turn)
            self.tts_q.put(END)


class SttThread(threading.Thread):
    """Transcribe el audio de comandos, solo si el turno sigue vigente (M3.5).

    hotwords: sesgo del decoder Whisper hacia nombres propios, SOLO para
    comandos (esta ruta corre siempre POST-wake). La detección de wake no pasa
    por aquí y queda sin sesgo, para no inflar falsos despertares.
    """

    def __init__(self, stt, stt_queue: queue.Queue, command_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event,
                 hotwords: str | None = None) -> None:
        super().__init__(name="Stt", daemon=True)
        self.stt = stt
        self.in_q = stt_queue
        self.out_q = command_queue
        self.state = state
        self.shutdown = shutdown
        self.hotwords = hotwords

    def run(self) -> None:
        timeout = _get_timeout(self.shutdown)
        while not self.shutdown.is_set():
            try:
                item = self.in_q.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is STOP:
                return
            audio, turn = item
            # `audio` puede ser un ndarray EN MEMORIA (camino caliente, sin
            # disco) o un Path a WAV (compat/tests). El WAV se borra pase lo
            # que pase: si el turno se abortó (barge-in) entre el encolado y
            # aquí, saltarse el unlink dejaba el temporal huérfano en /tmp.
            is_path = hasattr(audio, "unlink")
            try:
                if not self.state.is_current(turn):
                    continue  # comando abortado antes de transcribir
                try:
                    text = self.stt.transcribe(audio, hotwords=self.hotwords)
                except Exception as error:  # noqa: BLE001
                    log.warning("stt: %s", error)
                    text = ""
            finally:
                if is_path:
                    try:
                        audio.unlink(missing_ok=True)
                    except OSError as error:
                        log.debug("stt: no pude borrar el WAV temporal %s: %s",
                                  audio, error)
            # Comprobar el turno y tocar el estado de una vez (V5): un barge-in
            # entre los dos veía su LISTENING pisado por el IDLE de aquí, o la
            # frase vieja repintada en el HUD.
            if text.strip():
                # HUD: frase reconocida del usuario.
                if self.state.set_text_if_current(turn, text):
                    self.out_q.put((text, turn))
            else:
                # Transcripción VACÍA (comando corto/silencio: el VAD cortó pronto o
                # el patrón solo dijo "crotolamo" sin orden). No hay nada que pensar:
                # volver a reposo. Antes se quedaba PEGADO en "pensando" para siempre
                # porque el EarThread ya había puesto el modo en THINKING.
                self.state.set_mode_if_current(turn, Mode.IDLE)


def _drain_queue(q: queue.Queue) -> None:
    """Saca todo de la cola sin bloquear.

    El STOP del apagado NO se tira: si un barge-in que coincide con el apagado
    lo tirara, la Boca, que espera sin timeout, se quedaría bloqueada para
    siempre en una cola ya vacía.
    """
    stop = False
    try:
        while True:
            if q.get_nowait() is STOP:
                stop = True
    except queue.Empty:
        pass
    if stop:
        q.put(STOP)


class EarThread(threading.Thread):
    """El oído: único thread que toca el micrófono y decide transiciones por audio
    (M3.6). Reúne wake word (M1), Silero VAD (M2) y barge-in. Dependencias inyectadas
    (mic/wake_fn/vad_fn/to_wav) para ser testeable con fakes sin micrófono real.

    Eco/barge-in: con auriculares es confiable. Con altavoces y SIN AEC, las capas de
    M3.8 reducen los auto-disparos pero no los eliminan; si molesta, usa --no-barge-in.
    AEC real (webrtc-audio-processing) es una mejora futura.
    """

    # Fallos de audio (V2): mic.read() o wake_fn() revientan (micro desenchufado
    # o suspendido, PipeWire reiniciado, openWakeWord que no carga). Uno suelto
    # se aguanta; tras estos SEGUIDOS (~1 s con la pausa) el oído se rinde y
    # termina, VoiceLoop.run() lo nota y apaga el loop con código de fallo.
    _MAX_FALLOS_SEGUIDOS = 10
    _PAUSA_TRAS_FALLO_S = 0.1
    # Sin audio este rato en plena orden, se cierra con lo oído (ver _sin_audio).
    # El mismo segundo tras el que acaba la grabación del modo simple con fuente
    # (stt._leer_fuente). Con _RealMic no llega a pasar: read() bloquea hasta
    # el chunk siguiente y solo da None ya cerrado, en el apagado.
    _SIN_AUDIO_S = 1.0

    def __init__(self, mic, wake_fn, vad_fn, to_wav, tts,
                 stt_queue: queue.Queue, tts_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event, *,
                 vad_threshold: float = DEFAULTS.vad_threshold,
                 silence_ms: int = DEFAULTS.silence_ms,
                 max_command_s: float = DEFAULTS.max_command_s, allow_barge_in: bool = False,
                 chunk_ms: float = 32.0,
                 barge_in_grace_ms: float = DEFAULTS.barge_in_grace_ms,
                 barge_in_threshold_margin: float = DEFAULTS.barge_in_threshold_margin,
                 barge_in_min_chunks: int = DEFAULTS.barge_in_min_chunks,
                 vad_neg_threshold: float | None = DEFAULTS.vad_neg_threshold,
                 start_timeout_s: float = DEFAULTS.command_start_timeout_s,
                 preactivation_ms: float = DEFAULTS.vad_preactivation_ms,
                 wake_rearm=None,
                 post_speak_grace_s: float = DEFAULTS.post_speak_grace_s) -> None:
        super().__init__(name="Ear", daemon=True)
        self.mic = mic
        self.wake_fn = wake_fn
        # Olvido del detector tras un HUECO (WakeWordDetector.rearm): un chunk
        # que no llega a wake_fn (turno, gracia post-habla, escucha pausada)
        # abre el hueco y el primer wake_fn tras él lo cierra. None = nada.
        self.wake_rearm = wake_rearm
        self._wake_gap = False
        self.vad_fn = vad_fn
        self.to_wav = to_wav
        self.tts = tts
        self.stt_q = stt_queue
        self.tts_q = tts_queue
        self.state = state
        self.shutdown = shutdown
        self.vad_threshold = vad_threshold
        # Histéresis: umbral para MANTENERSE en voz, más bajo que el de entrar.
        # None = sin histéresis (idéntico al comportamiento previo). Ver voice/vad.py.
        self.vad_neg_threshold = resolve_neg_threshold(vad_threshold, vad_neg_threshold)
        self.allow_barge_in = allow_barge_in
        self._silence_chunks = max(1, int(silence_ms / chunk_ms))
        self._max_chunks = int(max_command_s * 1000 / chunk_ms)
        # V1: tras el wake la captura ESPERA la orden, con su propio límite (el
        # start_timeout_s del modo simple) y guardando solo la pre-activación.
        self._start_chunks = max(1, int(start_timeout_s * 1000 / chunk_ms))
        self._pre_chunks = max(1, int(preactivation_ms / chunk_ms))
        # M3.8: mitigación de eco (gracia + margen de umbral + persistencia).
        self._grace_s = barge_in_grace_ms / 1000.0
        self._margin = barge_in_threshold_margin
        self._min_chunks = max(1, barge_in_min_chunks)
        # Los últimos chunks vistos por _is_patron_voice: al disparar, son
        # justo la racha de voz del patrón, el arranque de su orden.
        self._barge_frames: deque = deque(maxlen=self._min_chunks)
        self._turn = 0
        # La captura de la orden en curso (voice/vad.py): sus frames y si tuvo AL
        # MENOS un chunk con voz (`speaking`, V6). La rehace _start_command.
        self._endpoint = self._endpointer()
        self._prev_mode = Mode.IDLE
        self._speaking_since = 0.0
        self._barge_run = 0
        # Cooldown anti-eco tras un turno: el modelo de wake se entrenó con voz
        # Piper sintética y el TTS de Crotolamo TAMBIÉN es Piper, así que al hablar
        # su respuesta el micro la capta y dispara FALSOS wakes -> bucle infinito
        # (habla -> wake -> comando vacío -> "Lo siento" -> habla...). Al volver a
        # IDLE ignoramos el wake un momento para que el eco del TTS no lo dispare.
        self._idle_grace_until = 0.0
        self._post_speak_grace_s = post_speak_grace_s

    def _endpointer(self, preroll: list | None = None, *, voiced: bool = False) -> Endpointer:
        """El Endpointer de una orden con las reglas del oído. Guarda pre_chunks
        chunks ANTES del primero con voz (+ 1: el anillo del Endpointer incluye
        el actual) y su plazo de espera cuenta el chunk actual: los dos, tal
        cual eran aquí (la grabación del modo simple guarda uno menos y espera
        uno más; ver STT._endpointer)."""
        return Endpointer(threshold=self.vad_threshold, neg_threshold=self.vad_neg_threshold,
                          silence_chunks=self._silence_chunks, max_chunks=self._max_chunks,
                          start_chunks=self._start_chunks, pre_chunks=self._pre_chunks + 1,
                          armed=not voiced, preroll=preroll or [])

    def _start_command(self, preroll: list | None = None, *, voiced: bool = False) -> None:
        # M2: arrancar con el buffer de pre-activación para no perder el inicio.
        # voiced=True: el preroll ya es voz del patrón (la racha del barge-in);
        # el del wake es la palabra de activación, no la orden.
        self._endpoint = self._endpointer(preroll, voiced=voiced)
        # La orden arranca con la RNN del VAD limpia (V11): con barge-in el VAD
        # se alimenta del eco del TTS en THINKING/SPEAKING y ese contexto
        # llegaba a la orden nueva (la del barge-in o la del wake siguiente).
        self._reset_vad()

    def _rearm_wake(self) -> None:
        """Primer chunk para el wake tras un hueco: que el detector olvide lo
        que oyó antes (ver WakeWordDetector.rearm). Un fallo aquí no puede
        dejar sordo al oído: se anota y el wake sigue como estaba."""
        self._wake_gap = False
        if self.wake_rearm is None:
            return
        try:
            self.wake_rearm()
        except Exception as error:  # noqa: BLE001 - el rearm no tumba al oído
            log.debug("oído: no pude rearmar el wake tras el hueco: %s", error)

    def _reset_vad(self) -> None:
        # Limpia el estado de la RNN del VAD (si el vad_fn lo soporta; en tests
        # suele ser un lambda sin reset).
        reset = getattr(self.vad_fn, "reset", None)
        if callable(reset):
            reset()

    def _is_patron_voice(self, chunk, mode) -> bool:
        """Detecta voz del patrón con mitigación de eco (M3.8), 3 capas:
        1) ventana de gracia tras empezar a hablar, 2) umbral elevado en SPEAKING,
        3) persistencia (N chunks consecutivos), no un pico suelto.
        """
        # Capa 2: umbral más alto mientras suena el TTS (el eco infla el VAD).
        thr = self.vad_threshold + (self._margin if mode is Mode.SPEAKING else 0.0)
        is_voice = self.vad_fn(chunk) >= thr
        self._barge_frames.append(chunk)
        # Capa 1: gracia tras arrancar el habla (el eco del arranque es el peor).
        in_grace = (mode is Mode.SPEAKING
                    and (time.monotonic() - self._speaking_since) < self._grace_s)
        if is_voice and not in_grace:
            self._barge_run += 1
        else:
            self._barge_run = 0
        # Capa 3: exige racha sostenida.
        if self._barge_run >= self._min_chunks:
            self._barge_run = 0
            return True
        return False

    def _finish_command(self) -> None:
        if self._endpoint.speaking:
            wav = self.to_wav(self._endpoint.frames)
            self.stt_q.put((wav, self._turn))
            self.state.set_mode(Mode.THINKING)
        else:
            # Captura SIN un solo chunk con voz (falso wake, o el wake y nada
            # más): no hay orden que transcribir (V6). Mandarla a Whisper costaba
            # ~2 s de pared y ~3 s de CPU por falso wake (el STT normaliza ese
            # ruido a pico 0.9 y el decoder alucina sobre él), con el loop en
            # THINKING y sordo al wake. Directo a reposo, como haría el Stt con
            # su transcripción vacía.
            self.state.set_mode(Mode.IDLE)
        # Cerrar el turno también limpia la RNN: la detección de barge-in en
        # THINKING/SPEAKING no arranca con el contexto de la orden.
        self._reset_vad()

    def _sin_audio(self) -> None:
        """El micrófono lleva _SIN_AUDIO_S sin entregar nada. Con la sesión de
        captura ([voice].audio_session) read() da None cada segundo mientras el
        micrófono reintenta, y los plazos de la orden cuentan CHUNKS: sin esto,
        un micro caído tras el wake dejaba el loop en LISTENING todo el corte
        (HUD 'escuchando' y la música atenuada). La orden se cierra como al
        acabar la frase: con voz, lo oído va a Whisper; sin ella, a reposo."""
        if self.state.get_mode() is not Mode.LISTENING:
            return
        log.info("oído: %.0f s sin audio del micrófono en plena orden; la cierro con lo oído",
                 self._SIN_AUDIO_S)
        self._finish_command()

    def run(self) -> None:
        fallos = 0
        ultimo_audio = time.monotonic()
        while not self.shutdown.is_set():
            try:
                chunk = self.mic.read()
                if chunk is not None:
                    ultimo_audio = time.monotonic()
                    self._on_chunk(chunk)
                elif time.monotonic() - ultimo_audio >= self._SIN_AUDIO_S:
                    self._sin_audio()
            except Exception as error:  # noqa: BLE001 - un fallo de audio no deja sordo al loop
                # Sin esto la excepción mataba al hilo (V2) y el proceso seguía
                # vivo, sordo y con el HUD en reposo: nadie lo notaba.
                fallos += 1
                if fallos >= self._MAX_FALLOS_SEGUIDOS:
                    log.error("oído: %d fallos seguidos, me rindo (el loop se apaga): %s",
                              fallos, error, exc_info=error)
                    return
                log.warning("oído: %s (fallo %d de %d seguidos)", error, fallos,
                            self._MAX_FALLOS_SEGUIDOS, exc_info=fallos == 1)
                # Un stream muerto revienta al instante en cada lectura: sin
                # pausa, esto sería un busy-spin. wait() atiende al apagado.
                self.shutdown.wait(self._PAUSA_TRAS_FALLO_S)
                continue
            fallos = 0
            if chunk is None:
                time.sleep(0.005)  # sin audio ahora mismo: no hacer busy-spin

    def _on_chunk(self, chunk) -> None:
        """Decide las transiciones por audio con UN chunk del micro (era el
        cuerpo del bucle de run(); vive aparte para que run() aguante sus fallos)."""
        mode = self.state.get_mode()
        # M3.8: marca el inicio del habla para la ventana de gracia.
        if mode is Mode.SPEAKING and self._prev_mode is not Mode.SPEAKING:
            self._speaking_since = time.monotonic()
            self._barge_run = 0
        # Al ENTRAR a IDLE desde un turno (habló/pensó/escuchó), arranca el
        # cooldown anti-eco para que el TTS y su eco no disparen un falso wake.
        if mode is Mode.IDLE and self._prev_mode in (
            Mode.SPEAKING, Mode.THINKING, Mode.LISTENING
        ):
            self._idle_grace_until = time.monotonic() + self._post_speak_grace_s
        self._prev_mode = mode
        if mode is not Mode.IDLE:
            self._wake_gap = True  # este chunk no llega al detector de wake

        if mode is Mode.IDLE:
            # Escucha pausada desde el panel: ignora la wake word por completo.
            # (Un turno en curso NO se corta; solo dejamos de despertar de nuevo.)
            if not self.state.is_enabled():
                self._wake_gap = True
                return
            # Cooldown anti-eco: tras un turno, no escuches el wake un momento.
            if time.monotonic() < self._idle_grace_until:
                self._wake_gap = True
                return
            if self._wake_gap:
                self._rearm_wake()
            if self.wake_fn(chunk):
                # Turno nuevo y a escuchar: UNA publicación (ver new_turn).
                self._turn = self.state.new_turn(mode=Mode.LISTENING)
                self._start_command(preroll=[chunk])

        elif mode is Mode.LISTENING:
            # Para EMPEZAR la orden hace falta el umbral de entrada (no arrancar
            # con ruido); ya DENTRO del habla basta el bajo (histéresis), así una
            # micro-pausa no cierra el comando. Aún sin voz tras el wake, el
            # silencio NO cuenta (V1): antes una pausa de respirar tras
            # "crotolamo" cerraba la captura vacía y la orden, dicha justo
            # después, se perdía. La espera tiene su propio límite, como
            # start_timeout_s en el modo simple, y solo guarda la pre-activación:
            # segundos de ruido de la habitación no van a Whisper. Sin voz al
            # final, a reposo y nada a Whisper (V6). Ver voice/vad.py.
            if self._endpoint.feed(chunk, self.vad_fn(chunk)) in (FIN, SIN_VOZ):
                self._finish_command()

        elif mode in (Mode.THINKING, Mode.SPEAKING):
            if self.allow_barge_in and self._is_patron_voice(chunk, mode):
                self.tts.stop()            # corta el audio YA
                _drain_queue(self.tts_q)   # tira las frases viejas
                # Invalida los turnos viejos y pasa a escuchar: UNA publicación.
                self._turn = self.state.new_turn(mode=Mode.LISTENING)
                # La racha del VAD ya es voz del patrón y el arranque de su orden:
                # va ENTERA (antes solo su último chunk y la primera sílaba, los
                # barge_in_min_chunks - 1 chunks previos, se perdía).
                self._start_command(preroll=list(self._barge_frames), voiced=True)
