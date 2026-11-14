"""Lógica pura del HUD (sin GTK, sin gi) — testeable headless.

El parseo defensivo del JSON vive en common.ipc (compartido con el panel);
aquí queda la interpretación de los campos del contrato.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from common.ipc import HUD_STATE_FILE, parse_hud_state, read_hud_state

__all__ = [
    "HUD_STATE_FILE",
    "VALID_MODES",
    "parse_hud_state",
    "read_hud_file",
    "extract_mode",
    "extract_text",
    "publisher_alive",
]

# Modos reconocidos (igual que los valores JSON del contrato de integración)
VALID_MODES = frozenset({"idle", "listening", "thinking", "speaking"})


def read_hud_file(path: Path) -> dict[str, Any]:
    """Lee y parsea hud_state.json de forma defensiva."""
    return read_hud_state(path)


def extract_mode(data: dict[str, Any]) -> str:
    """Extrae y valida el campo 'mode'; devuelve 'idle' si no es válido."""
    mode = data.get("mode", "idle")
    # isinstance ANTES de mirar el frozenset: una lista o un objeto JSON no son
    # hashables y el `in` lanzaba TypeError dentro del callback de GLib (que
    # descarta para siempre un timer cuyo callback lanza: el poll se moría).
    if not isinstance(mode, str) or mode not in VALID_MODES:
        return "idle"
    return mode


def extract_text(data: dict[str, Any]) -> str:
    """Extrae el campo 'text' de forma segura, en UN solo párrafo.

    El loop publica la respuesta entera del LLM, con sus saltos de línea, y el
    tope de líneas de la etiqueta del HUD lo aplica Pango POR PÁRRAFO: con
    saltos de línea no habría tope real.
    """
    text = data.get("text", "")
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def publisher_alive(data: dict[str, Any]) -> bool:
    """¿Sigue vivo el proceso que publicó este estado? (campo 'pid' del contrato)

    El loop escribe su pid en cada publicación. Si muere sin su apagado limpio
    (SIGKILL, OOM, segfault de audio, terminal cerrada, apagón) el archivo se
    queda en "speaking"/"thinking" y nadie lo corrige hasta que otro listener
    cargue modelos y publique: el HUD, siempre encima, seguía diciendo
    "HABLANDO". Sin un pid válido (esquema viejo) no se puede saber: se confía.
    """
    pid = data.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    try:
        os.kill(pid, 0)  # señal 0: solo pregunta si el proceso existe
    except ProcessLookupError:
        return False
    except (OSError, OverflowError):
        # EPERM: existe (de otro usuario o bajo otra política) y, ante la duda,
        # no se esconde un estado que puede ser real. OverflowError: un número
        # que no cabe en un pid_t es basura, como no tener pid. Nunca lanzar:
        # esto corre en un callback de GLib, que descarta el timer si lanza.
        return True
    return True
