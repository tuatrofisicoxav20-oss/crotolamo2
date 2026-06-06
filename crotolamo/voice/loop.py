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
