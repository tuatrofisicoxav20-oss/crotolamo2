"""Estado compartido del loop de voz. Thread-safe por lock.

El turn_id monótono es lo que hace CORRECTA la interrupción: cada comando nuevo
(wake o barge-in) incrementa el turno; las frases nacen con su turn_id y el
MouthThread descarta las de turnos viejos. Sin Events de "interrupción" que se
limpian (eso da races); toda invalidación pasa por new_turn() + is_current().

Publicador de estado para el HUD (opt-in):
    SharedState acepta un ``publisher`` opcional: un callable(dict) -> None que
    se invoca cada vez que cambia el modo, el turno o el texto. Si no se pasa
    (None), el publicador es un no-op y no se escribe ningún archivo.

    La función de utilidad ``make_file_publisher(path)`` devuelve un publicador
    que escribe JSON de forma ATÓMICA (tmp + os.replace) en ``path``.  Cualquier
    excepción de E/S queda atrapada y solo loguea (NUNCA mata el loop de voz).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable

log = logging.getLogger("voice.state")


class Mode(Enum):
    IDLE = auto()       # esperando wake word
    LISTENING = auto()  # capturando el comando
    THINKING = auto()   # el LLM/STT procesa
    SPEAKING = auto()   # reproduciendo respuesta


def make_file_publisher(path: Path) -> Callable[[dict], None]:
    """Devuelve un callable que escribe el dict como JSON de forma atómica en ``path``.

    Escribe a un archivo temporal en el mismo directorio y usa os.replace para
    garantizar que los lectores nunca vean un archivo a medias.
    Cualquier error de E/S solo loguea, NUNCA propaga (no mata el loop de voz).
    """
    path = Path(path)

    def _publish(state_dict: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Escritura atómica: tmp en el mismo directorio -> os.replace
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state_dict, f)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            os.replace(tmp_name, path)
        except Exception as exc:  # noqa: BLE001 — nunca matar el loop
            log.warning("hud_state: no pude escribir %s: %s", path, exc)

    return _publish


class SharedState:
    """Estado compartido del loop de voz.

    Args:
        publisher: callable(dict) -> None invocado en cada cambio de estado.
                   Si es None (default) es un no-op completo (sin HUD).
    """

    def __init__(self, publisher: Callable[[dict], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._mode = Mode.IDLE
        self._turn_id = 0
        self._text = ""
        self._publisher = publisher

    # ------------------------------------------------------------------
    # Publicación interna: siempre se llama SIN el lock tomado para evitar
    # deadlock (el lock en SharedState no es reentrant).
    # ------------------------------------------------------------------
    def _publish(self, mode: Mode, turn_id: int, text: str) -> None:
        if self._publisher is None:
            return
        state_dict = {
            "mode": mode.name.lower(),
            "turn_id": turn_id,
            "text": text,
            "ts": time.time(),
            "pid": os.getpid(),
        }
        try:
            self._publisher(state_dict)
        except Exception as exc:  # noqa: BLE001
            log.warning("hud publisher excepción: %s", exc)

    def get_mode(self) -> Mode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            self._mode = mode
            _turn = self._turn_id
            _text = self._text
        # Publicar FUERA del lock
        self._publish(mode, _turn, _text)

    def set_text(self, text: str) -> None:
        """Actualiza el texto visible (última frase reconocida o respuesta)."""
        with self._lock:
            self._text = text
            _mode = self._mode
            _turn = self._turn_id
        self._publish(_mode, _turn, text)

    @property
    def turn_id(self) -> int:
        with self._lock:
            return self._turn_id

    def new_turn(self) -> int:
        """Arranca un turno nuevo (wake o barge-in). Devuelve el id nuevo."""
        with self._lock:
            self._turn_id += 1
            _turn = self._turn_id
            _mode = self._mode
            _text = self._text
        # Publicar FUERA del lock
        self._publish(_mode, _turn, _text)
        return _turn

    def is_current(self, turn: int) -> bool:
        with self._lock:
            return turn == self._turn_id

    def current_snapshot(self) -> dict:
        """Devuelve un snapshot del estado actual (para publicar el idle final)."""
        with self._lock:
            return {
                "mode": self._mode.name.lower(),
                "turn_id": self._turn_id,
                "text": self._text,
                "ts": time.time(),
                "pid": os.getpid(),
            }
