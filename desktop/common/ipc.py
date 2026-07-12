"""IPC por archivos entre el loop de voz, el HUD y el panel.

Única fuente de verdad para las rutas del contrato de integración (antes
duplicadas en crotolamo_hud.py y crotolamo_panel.py) y para la lectura
defensiva de `hud_state.json`.

Importable sin display y sin gi: solo stdlib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("crotolamo.desktop.ipc")

# Estado publicado por el loop de voz (solo lectura para HUD y panel).
HUD_STATE_FILE = Path.home() / ".crotolamo" / "hud_state.json"

# Canal de control INVERSO (panel -> loop): pausar/reanudar la escucha por voz
# sin apagar el servicio. El loop lo sondea; se escribe de forma atómica.
CONTROL_FILE = Path.home() / ".crotolamo" / "control.json"


def parse_hud_state(raw: str) -> dict[str, Any]:
    """Parsea el JSON del contrato. Nunca lanza; devuelve {} si hay error."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, ValueError):
        return {}


def read_hud_state(path: Path | None = None) -> dict[str, Any]:
    """Lee y parsea hud_state.json de forma defensiva.

    Con `path=None` usa la ruta real del contrato (HUD_STATE_FILE, resuelta en
    el momento de la llamada para que los tests puedan parchearla).
    """
    if path is None:
        path = HUD_STATE_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        log.debug("No se pudo leer %s: %s", path, exc)
        return {}
    return parse_hud_state(raw)
