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


# Canal de control INVERSO (panel -> loop): el panel escribe este archivo para
# pausar/reanudar la escucha por voz SIN matar el proceso. El loop lo sondea.
CONTROL_PATH = Path.home() / ".crotolamo" / "control.json"

# Canal de estado (loop -> HUD y panel): el archivo que publica SharedState vía
# make_file_publisher. Vivía solo en interfaces/listener.py; aquí queda junto a
# CONTROL_PATH, una sola definición por lado del contrato (el del escritorio es
# desktop/common/ipc.py, y los tests cruzan las dos).
HUD_STATE_PATH = Path.home() / ".crotolamo" / "hud_state.json"


def read_control_enabled(path: Path = CONTROL_PATH) -> bool:
    """Lee el flag de escucha del canal de control. Default-SEGURO.

    Contrato: el archivo es ``{"listening_enabled": bool}``. Devuelve False SOLO
    si el flag es explícitamente False; ausente, vacío, corrupto o cualquier otro
    error de E/S => True. NUNCA arrancar mudo por un archivo que falta.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "listening_enabled" in data:
            return bool(data["listening_enabled"])
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        pass
    return True


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
        # Turno de PUBLICAR (C8), aparte del lock de estado: las consultas y los
        # cambios (el oído mira el modo ~31 veces por segundo) nunca esperan a
        # que otro hilo termine de escribir el archivo del HUD; solo la
        # publicación espera su turno. Reentrante: un publisher puede volver a
        # entrar al estado sin colgarse.
        self._pub_lock = threading.RLock()
        self._mode = Mode.IDLE
        self._turn_id = 0
        self._text = ""
        # Escucha por voz habilitada (default True). El botón del panel la pausa
        # sin matar el proceso: cuando es False, el EarThread ignora la wake word.
        self._enabled = True
        self._publisher = publisher

    # ------------------------------------------------------------------
    # Publicación interna: siempre se llama SIN el lock de estado tomado para
    # evitar deadlock (ese lock no es reentrant y el publisher hace E/S).
    #
    # La foto del estado se toma AQUÍ, ya con el turno de publicar en la mano,
    # y no al cambiar (C8): cuando cada hilo publicaba la foto que tomó bajo el
    # lock, dos cambios casi simultáneos podían escribirse al revés y la foto
    # VIEJA quedaba la última; el HUD (y el ducker) se quedaban con un modo
    # rancio hasta el siguiente cambio. Así lo último publicado es siempre el
    # estado real.
    # ------------------------------------------------------------------
    def _publish(self) -> None:
        if self._publisher is None:
            return
        with self._pub_lock:
            state_dict = self.current_snapshot()
            try:
                self._publisher(state_dict)
            except Exception as exc:  # noqa: BLE001
                log.warning("hud publisher excepción: %s", exc)

    def get_mode(self) -> Mode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: Mode) -> None:
        """Cambia el modo. Publica SOLO si cambia (como set_enabled).

        El Brain repite THINKING (el oído ya lo puso al cerrar la orden) y el
        Mouth pone SPEAKING antes de CADA frase: sin la guarda, la mitad de las
        escrituras de un turno (temporal + os.replace + un despertar del HUD
        por inotify) contaban el mismo estado. Para forzar: publish_now().
        """
        with self._lock:
            if mode is self._mode:
                return
            self._mode = mode
        # Publicar FUERA del lock
        self._publish()

    def set_mode_if_current(self, turn: int, mode: Mode) -> bool:
        """set_mode SOLO si ``turn`` sigue vigente; devuelve si lo era (V5).

        is_current() y LUEGO set_mode() dejaba un hueco: un barge-in en medio
        (new_turn + LISTENING) quedaba pisado por el THINKING/SPEAKING/IDLE del
        turno viejo, y el oído, que obedece al modo, abandonaba la orden nueva:
        el loop se quedaba en "pensando"/"hablando" (música bajada, wake
        ignorado) hasta otro barge-in. Aquí comprobar y cambiar van bajo el
        MISMO lock. Como set_mode, publica solo si el modo cambia.
        """
        with self._lock:
            if turn != self._turn_id:
                return False
            if mode is self._mode:
                return True
            self._mode = mode
        self._publish()
        return True

    def compare_and_set_mode(self, expected: Mode, mode: Mode) -> bool:
        """Pasa a ``mode`` SOLO si el modo actual es ``expected`` (V5).

        Para el fin de respuesta del Mouth: get_mode() == SPEAKING y luego
        set_mode(IDLE) pisaba el LISTENING que un barge-in acababa de poner, y
        la orden nueva se perdía.
        """
        with self._lock:
            if self._mode is not expected:
                return False
            if mode is self._mode:
                return True
            self._mode = mode
        self._publish()
        return True

    def set_text(self, text: str) -> None:
        """Actualiza el texto visible (última frase reconocida o respuesta)."""
        with self._lock:
            self._text = text
        self._publish()

    def set_text_if_current(self, turn: int, text: str) -> bool:
        """set_text SOLO si ``turn`` sigue vigente; devuelve si lo era (V5).

        Mismo hueco que set_mode_if_current: la frase o la respuesta de un
        turno abortado volvía a pintarse en el HUD bajo "escuchando" después
        de que new_turn() la limpiara.
        """
        with self._lock:
            if turn != self._turn_id:
                return False
            self._text = text
        self._publish()
        return True

    @property
    def turn_id(self) -> int:
        with self._lock:
            return self._turn_id

    def new_turn(self) -> int:
        """Arranca un turno nuevo (wake o barge-in). Devuelve el id nuevo.

        El texto visible se limpia: es de la interacción anterior (la última
        respuesta hablada) y el HUD lo pintaba bajo "ESCUCHANDO" hasta que el
        STT publicaba la frase nueva.
        """
        with self._lock:
            self._turn_id += 1
            self._text = ""
            _turn = self._turn_id
        # Publicar FUERA del lock
        self._publish()
        return _turn

    def is_current(self, turn: int) -> bool:
        with self._lock:
            return turn == self._turn_id

    # ------------------------------------------------------------------
    # Escucha por voz habilitada/pausada (canal de control del panel).
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Habilita/pausa la escucha. Publica SOLO si cambia el valor.

        Sin la guarda old != new, el poll del canal de control (varias veces por
        segundo) reescribiría hud_state.json en cada tick aunque nada cambie.
        """
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
        # Publicar FUERA del lock (para que el panel vea el estado real).
        self._publish()

    def publish_now(self) -> None:
        """Publica el estado actual aunque nada haya cambiado.

        Se llama al ARRANCAR el loop: si el proceso anterior murió sin pasar
        por el apagado limpio (kill -9, OOM, segfault de audio), hud_state.json
        se quedó en "thinking"/"listening" y el HUD lo mostraba para siempre,
        porque set_mode/set_enabled solo publican al cambiar.
        """
        self._publish()

    def current_snapshot(self) -> dict:
        """Devuelve un snapshot del estado actual: es el dict que publica
        _publish (el esquema de hud_state.json; el idle final del listener lo
        repite y un test de la auditoría los mantiene iguales)."""
        with self._lock:
            return {
                "mode": self._mode.name.lower(),
                "turn_id": self._turn_id,
                "text": self._text,
                "enabled": self._enabled,
                "ts": time.time(),
                "pid": os.getpid(),
            }
