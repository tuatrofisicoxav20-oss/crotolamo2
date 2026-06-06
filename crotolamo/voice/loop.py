"""Loop de voz concurrente con interrupción por turn_id (de GLaDOS, redo M3).

Arquitectura race-free: cuatro threads (Ear/Stt/Brain/Mouth) coordinados por colas
y un SharedState con turn_id monótono. Cada frase nace con su turn_id; el MouthThread
descarta las de turnos abortados. La interrupción es correcta POR CONSTRUCCIÓN: nada
de Events de "interrupción" que se limpian (races); toda invalidación pasa por
state.new_turn() + state.is_current().

Se construye por sub-fases (M3.3 Mouth, M3.4 Brain, M3.5 Stt, M3.6 Ear, M3.7 loop).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from crotolamo.logging_setup import get_logger
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.tts import split_sentences

log = get_logger("voice.loop")

END = object()  # sentinel: fin de la respuesta de un turno


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
        while not self.shutdown.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is END:
                # Terminó la respuesta de un turno; si nadie habló nuevo, a IDLE.
                if self.state.get_mode() == Mode.SPEAKING:
                    self.state.set_mode(Mode.IDLE)
                continue
            assert isinstance(item, Utterance)
            # Descarta frases de turnos abortados (barge-in o comando nuevo).
            if not self.state.is_current(item.turn_id):
                continue
            self.state.set_mode(Mode.SPEAKING)
            self.tts.speak(item.text)  # interrumpible (M3.1)


class BrainThread(threading.Thread):
    """Saca comandos, llama al agente y encola las frases con su turn_id (M3.4).

    Nota honesta: agent.handle_turn llama al LLM por HTTP de forma BLOQUEANTE; no se
    puede cancelar a media inferencia. Si hay barge-in mientras el LLM piensa, la
    inferencia vieja termina en background pero su salida se DESCARTA por el check de
    is_current(turn); el audio para al instante. Cancelar el LLM en vuelo es mejora
    futura (streaming con check de turno).
    """

    def __init__(self, agent, command_queue: queue.Queue, tts_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event) -> None:
        super().__init__(name="Brain", daemon=True)
        self.agent = agent
        self.cmd_q = command_queue
        self.tts_q = tts_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                command, turn = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # El turno ya cambió (barge-in mientras esperaba en cola): ignora.
            if not self.state.is_current(turn):
                continue
            self.state.set_mode(Mode.THINKING)
            try:
                reply = self.agent.handle_turn(command)
            except Exception as error:  # noqa: BLE001 - un turno roto no mata el loop
                log.warning("brain: %s", error)
                continue
            # Si interrumpieron mientras pensaba, NO encolar la respuesta vieja.
            if not self.state.is_current(turn):
                continue
            for sentence in split_sentences(reply):
                self.tts_q.put(Utterance(sentence, turn))
            self.tts_q.put(END)


class SttThread(threading.Thread):
    """Transcribe el audio de comandos, solo si el turno sigue vigente (M3.5)."""

    def __init__(self, stt, stt_queue: queue.Queue, command_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event) -> None:
        super().__init__(name="Stt", daemon=True)
        self.stt = stt
        self.in_q = stt_queue
        self.out_q = command_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                audio_path, turn = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.state.is_current(turn):
                continue  # comando abortado antes de transcribir
            try:
                text = self.stt.transcribe(audio_path)
            except Exception as error:  # noqa: BLE001
                log.warning("stt: %s", error)
                text = ""
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            if text.strip() and self.state.is_current(turn):
                self.out_q.put((text, turn))
