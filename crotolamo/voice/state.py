"""Estado compartido del loop de voz. Thread-safe por lock.

El turn_id monótono es lo que hace CORRECTA la interrupción: cada comando nuevo
(wake o barge-in) incrementa el turno; las frases nacen con su turn_id y el
MouthThread descarta las de turnos viejos. Sin Events de "interrupción" que se
limpian (eso da races); toda invalidación pasa por new_turn() + is_current().
"""

from __future__ import annotations

import threading
from enum import Enum, auto


class Mode(Enum):
    IDLE = auto()       # esperando wake word
    LISTENING = auto()  # capturando el comando
    THINKING = auto()   # el LLM/STT procesa
    SPEAKING = auto()   # reproduciendo respuesta


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = Mode.IDLE
        self._turn_id = 0

    def get_mode(self) -> Mode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            self._mode = mode

    @property
    def turn_id(self) -> int:
        with self._lock:
            return self._turn_id

    def new_turn(self) -> int:
        """Arranca un turno nuevo (wake o barge-in). Devuelve el id nuevo."""
        with self._lock:
            self._turn_id += 1
            return self._turn_id

    def is_current(self, turn: int) -> bool:
        with self._lock:
            return turn == self._turn_id
