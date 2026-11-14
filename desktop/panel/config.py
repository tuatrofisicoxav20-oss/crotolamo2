"""Configuración persistente del panel (env file + canal de pausa) — sin gi.

Las lecturas se cachean por (mtime, tamaño): el panel refresca cada 1.5 s y
antes releía el disco entero en cada tick aunque nada hubiera cambiado.
"""

from __future__ import annotations

import json
import logging
import os
import stat
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

# Prefijo de la línea del modo en ENV_FILE (lectura y escritura usan el mismo).
_MODE_KEY = "CROTOLAMO_LISTEN_ARGS="


def _is_mode_line(line: str) -> bool:
    return line.strip().startswith(_MODE_KEY)


def _replace_atomically(path: Path, data: bytes) -> None:
    """Escribe `data` en `path` de forma ATÓMICA (temporal del mismo dir + os.replace).

    Quien lea (el loop, o systemd al arrancar el servicio) ve el archivo viejo o
    el nuevo entero, nunca uno vacío o a medias. Si `path` ya existía conserva sus
    permisos (mkstemp crea con 0600). Si algo falla no queda temporal huérfano y
    el error se propaga: quien llama lo registra.
    """
    if path.is_symlink():
        # dotfiles con stow/chezmoi: se reemplaza el DESTINO; os.replace sobre el
        # enlace lo cambiaría por un archivo suelto y rompería el enlace
        path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        except FileNotFoundError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError as exc:
            log.debug("No se pudo borrar el temporal %s: %s", tmp, exc)
        raise


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
    args = "--no-barge-in"
    try:
        # bytes + surrogateescape, como write_mode_args: un comentario en latin-1
        # (que el escritor conserva) hacía fallar read_text() y el panel mostraba
        # el modo por defecto mientras el servicio corría en otro
        text = path.read_bytes().decode("utf-8", "surrogateescape")
        for line in text.splitlines():
            if _is_mode_line(line):
                # systemd (EnvironmentFile=) aplica la ÚLTIMA asignación: el panel
                # muestra esa, no la primera de un archivo editado a mano
                args = line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo leer %s: %s", path, exc)
    return args


def read_mode_args() -> str:
    """Args de escucha guardados en el env file (cacheado por mtime)."""
    path = ENV_FILE
    return _read_cached(path, lambda: _parse_mode_args(path))


def write_mode_args(args: str) -> bool:
    """Guarda el modo en ENV_FILE tocando SOLO la línea CROTOLAMO_LISTEN_ARGS.

    El servicio carga el archivo ENTERO (EnvironmentFile=): cualquier otra línea
    (CROTOLAMO_LOG=DEBUG, un comentario) es del usuario y antes se perdía al
    elegir modo. Sin la línea, se añade al final. Si el archivo existe pero no se
    puede leer, no se pisa a ciegas: solo se registra.

    Devuelve si quedó guardado (como write_listening_enabled): reiniciar el
    servicio para aplicar un modo que no está en disco recarga los modelos de voz
    y lo deja sordo unos segundos para nada.
    """
    path = ENV_FILE
    try:
        try:
            # bytes + surrogateescape: lo que no sea UTF-8 vuelve al disco tal cual
            old = path.read_bytes().decode("utf-8", "surrogateescape")
        except FileNotFoundError:
            old = ""
        if old and not old.endswith("\n"):
            old += "\n"
        lines = old.split("\n")[:-1]
        new_line = f"{_MODE_KEY}{args}"
        found = False
        for i, line in enumerate(lines):
            if _is_mode_line(line):
                # systemd se queda con la ÚLTIMA asignación (y el panel lee esa):
                # todas llevan el valor nuevo para que ninguna diga otra cosa
                lines[i] = new_line + ("\r" if line.endswith("\r") else "")
                found = True
        if not found:
            lines.append(new_line)
        data = "".join(f"{line}\n" for line in lines)
        _replace_atomically(path, data.encode("utf-8", "surrogateescape"))
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo escribir %s: %s", path, exc)
        return False
    return True


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


def write_listening_enabled(enabled: bool) -> bool:
    """Escribe el flag de control de forma ATÓMICA (tmp + os.replace).

    El loop nunca debe leer un archivo a medias; cualquier error de E/S solo se
    registra (no revienta la UI). Devuelve si quedó escrito: el panel solo
    sostiene el switch en lo pedido si el loop de verdad lo va a leer.
    """
    control = ipc.CONTROL_FILE
    try:
        data = json.dumps({"listening_enabled": enabled}).encode("utf-8")
        _replace_atomically(control, data)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo escribir %s: %s", control, exc)
        return False
    return True
