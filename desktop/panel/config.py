"""Configuración persistente del panel (env file + canal de pausa) — sin gi.

Las lecturas se cachean por (mtime, tamaño): el panel refresca cada 1.5 s y
antes releía el disco entero en cada tick aunque nada hubiera cambiado.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from common import ipc

log = logging.getLogger("crotolamo.desktop.panel.config")

ENV_FILE = Path.home() / ".config" / "crotolamo" / "listener.env"

# Modo de escucha -> argumentos que recibe `python -m crotolamo listen`.
# El servicio los lee de ENV_FILE (variable CROTOLAMO_LISTEN_ARGS).
MODES: list[tuple[str, str, str]] = [
    # (clave, etiqueta visible, args)
    ("half", "Half-duplex (altavoces)", "--no-barge-in"),
    ("barge", "Barge-in (auriculares)", "--barge-in"),
    ("simple", "Simple (a prueba de fallos)", "--simple"),
]
ARGS_BY_KEY = {k: a for k, _, a in MODES}
KEY_BY_ARGS = {a: k for k, _, a in MODES}

# --- caché por mtime: releer solo si el archivo cambió -------------------------

# path -> (firma stat o None, valor parseado)
_MTIME_CACHE: dict[Path, tuple[tuple[int, int] | None, Any]] = {}


def _stat_sig(path: Path) -> tuple[int, int] | None:
    """Firma barata del archivo (mtime_ns, tamaño); None si no existe/error."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _read_cached(path: Path, reader) -> Any:
    """Devuelve el valor cacheado si la firma stat no cambió; si no, relee."""
    sig = _stat_sig(path)
    entry = _MTIME_CACHE.get(path)
    if entry is not None and entry[0] == sig:
        return entry[1]
    value = reader()
    _MTIME_CACHE[path] = (sig, value)
    return value


# --- helpers del env file (modo de voz) ---------------------------------------

def _parse_mode_args(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("CROTOLAMO_LISTEN_ARGS="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo leer %s: %s", path, exc)
    return "--no-barge-in"


def read_mode_args() -> str:
    """Args de escucha guardados en el env file (cacheado por mtime)."""
    path = ENV_FILE
    return _read_cached(path, lambda: _parse_mode_args(path))


def write_mode_args(args: str) -> None:
    try:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text(f"CROTOLAMO_LISTEN_ARGS={args}\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo escribir %s: %s", ENV_FILE, exc)


# --- helpers del canal de pausa de escucha (panel <-> loop) -------------------

def _parse_listening_enabled(path: Path) -> bool:
    data = ipc.read_hud_state(path)
    if "enabled" in data:
        return bool(data["enabled"])
    return True


def read_listening_enabled() -> bool:
    """Si la escucha por voz está activa, según el estado que publica el loop.

    Default-SEGURO: ausente/corrupto/sin campo => True. El panel refleja la
    realidad leyendo aquí (cacheado por mtime), no lo que él cree haber escrito.
    """
    path = ipc.HUD_STATE_FILE
    return _read_cached(path, lambda: _parse_listening_enabled(path))


def write_listening_enabled(enabled: bool) -> None:
    """Escribe el flag de control de forma ATÓMICA (tmp + os.replace).

    El loop nunca debe leer un archivo a medias; cualquier error de E/S solo se
    registra (no revienta la UI).
    """
    control = ipc.CONTROL_FILE
    try:
        control.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=control.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"listening_enabled": enabled}, f)
        except Exception:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError as exc:
                log.debug("No se pudo borrar el temporal %s: %s", tmp, exc)
            raise
        os.replace(tmp, control)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo escribir %s: %s", control, exc)
