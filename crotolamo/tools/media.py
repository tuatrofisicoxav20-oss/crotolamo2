"""Control de música/reproductores vía playerctl (MPRIS).

Funciona con Spotify, navegadores (YouTube), VLC y cualquier reproductor MPRIS.
Todo read-only salvo las acciones de transporte (play/pause/next/prev), que son
inofensivas y reversibles, así que van como `safe=True`. No genera bash: cada
acción es un comando fijo de playerctl.
"""

from __future__ import annotations

import shutil
import subprocess

from crotolamo.tools.base import tool

# Acción -> argumentos de playerctl. 'status'/'now' se tratan aparte (leen metadata).
_ACTIONS: dict[str, list[str]] = {
    "play": ["play"],
    "pause": ["pause"],
    "toggle": ["play-pause"],
    "play-pause": ["play-pause"],
    "next": ["next"],
    "siguiente": ["next"],
    "previous": ["previous"],
    "anterior": ["previous"],
    "stop": ["stop"],
    "parar": ["stop"],
}


def _player_arg(player: str = "") -> list[str]:
    """Si se pide un reproductor concreto (p.ej. spotify), restringe a él."""
    p = player.strip().lower()
    return ["-p", p] if p else []


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["playerctl", *args], text=True, capture_output=True, timeout=8
    )


@tool
def music_now() -> str:
    """Dice qué canción/contenido suena ahora mismo (Spotify, YouTube, etc.)."""
    if not shutil.which("playerctl"):
        return "No tengo playerctl, patrón. Instálalo con `sudo dnf install playerctl`."
    try:
        meta = _run(["metadata", "--format", "{{artist}} — {{title}}"])
        status = _run(["status"])
    except (OSError, subprocess.TimeoutExpired):
        return "No pude leer el reproductor, patrón."
    if status.returncode != 0:
        return "No hay nada sonando, patrón. El silencio también es música."
    estado = (status.stdout or "").strip().lower()
    cancion = (meta.stdout or "").strip(" —")
    if not cancion:
        return f"Reproductor en estado «{estado}», patrón, pero sin metadatos."
    etiqueta = {"playing": "Sonando", "paused": "En pausa"}.get(estado, estado.capitalize())
    return f"{etiqueta}, patrón: {cancion}"


@tool
def music_control(action: str, player: str = "") -> str:
    """Controla la música: play, pause, toggle, next/siguiente, previous/anterior, stop.

    Sirve para Spotify y cualquier reproductor. Usa 'toggle' para alternar
    play/pausa cuando el patrón solo dice "pausa la música" o "ponle play".

    Args:
        action: la acción (play, pause, toggle, next, previous, stop).
        player: opcional, reproductor concreto (p.ej. 'spotify'); vacío = el activo.
    """
    if not shutil.which("playerctl"):
        return "No tengo playerctl, patrón. Instálalo con `sudo dnf install playerctl`."
    key = action.strip().lower()
    if key not in _ACTIONS:
        return (
            f"No conozco la acción '{action}', patrón. Tengo: "
            "play, pause, toggle, next, previous, stop."
        )
    try:
        result = _run([*_player_arg(player), *_ACTIONS[key]])
    except (OSError, subprocess.TimeoutExpired):
        return "No pude controlar la música, patrón."
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        if "No players found" in err or not err:
            return "No hay ningún reproductor abierto, patrón. Abre Spotify primero."
        return f"playerctl se quejó, patrón: {err}"
    # Tras la acción, decir qué quedó sonando (mejor feedback para el patrón).
    bonito = {
        "play": "Dale play",
        "pause": "Pausado",
        "toggle": "Listo",
        "play-pause": "Listo",
        "next": "Siguiente",
        "siguiente": "Siguiente",
        "previous": "Anterior",
        "anterior": "Anterior",
        "stop": "Parado",
        "parar": "Parado",
    }.get(key, "Hecho")
    return f"{bonito}, patrón."
