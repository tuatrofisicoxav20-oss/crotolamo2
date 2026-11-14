"""Control de música/reproductores vía playerctl (MPRIS).

Funciona con Spotify, navegadores (YouTube), VLC y cualquier reproductor MPRIS.
Todo read-only salvo las acciones de transporte (play/pause/next/prev), que son
inofensivas y reversibles, así que van como `safe=True`. No genera bash: cada
acción es un comando fijo de playerctl.
"""

from __future__ import annotations

import shutil
import subprocess

from crotolamo.tools.base import run_cmd, tool

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

# Órdenes de playerctl que van para LO QUE SUENA. Sin `-p`, playerctl se la
# manda al primer reproductor que la admita, suene o no: con una pestaña en
# pausa delante de Spotify, `pause` "pausaba" la pestaña, Spotify seguía y la
# tool decía "Pausado" (y `play-pause` arrancaba la pestaña encima). `play` no
# está: es para lo que NO suena, y ahí decide playerctl como siempre.
_SOBRE_LO_QUE_SUENA = frozenset({"pause", "play-pause", "next", "previous", "stop"})

# Una línea por reproductor: nombre, estado y pista. Separa el tabulador, que
# ni el nombre ni el estado llevan; la pista va al final por si lo trae.
_LISTADO = "{{playerName}}\t{{status}}\t{{artist}} — {{title}}"


def _player_arg(player: str | None = "") -> list[str]:
    """Si se pide un reproductor concreto (p.ej. spotify), restringe a él."""
    # `or ""`: un `"player": null` del modelo es "sin reproductor", el activo;
    # antes reventaba en .strip() y volvía como "Argumentos inválidos".
    p = (player or "").strip().lower()
    return ["-p", p] if p else []


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return run_cmd(["playerctl", *args], timeout=8)


def _players() -> list[tuple[str, str, str]]:
    """(nombre, estado en minúsculas, "artista — título") de cada reproductor
    con pista, en el orden de playerctl. Lanza lo mismo que `_run`.

    Vale lo que haya en stdout aunque la salida no sea 0: playerctl sale con 1
    si un reproductor falla aunque los demás ya imprimieron su línea. Los que
    no tienen pista no salen (`metadata` se los salta).
    """
    result = _run(["-a", "metadata", "--format", _LISTADO])
    players: list[tuple[str, str, str]] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].strip():
            players.append((parts[0].strip(), parts[1].strip().lower(), parts[2]))
    return players


def _playing(players: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """El primer reproductor que está sonando, o None si no suena ninguno."""
    return next((p for p in players if p[1] == "playing"), None)


@tool(parallel_safe=True)
def music_now() -> str:
    """Dice qué canción/contenido suena ahora mismo (Spotify, YouTube, etc.)."""
    if not shutil.which("playerctl"):
        return "No tengo playerctl, patrón. Instálalo con `sudo dnf install playerctl`."
    try:
        # Un solo playerctl con el estado Y la pista de cada reproductor. Antes
        # eran dos (`metadata` y `status`), cada uno sobre "el primero que
        # pueda": con una pestaña en pausa delante de Spotify se describía la
        # pestaña, y podían contestar reproductores distintos (el estado de uno
        # con la canción de otro).
        players = _players()
        # Ninguno con pista: `status` distingue "no hay reproductor" de "hay
        # uno sin metadatos", como siempre.
        status = None if players else _run(["status"])
    except (OSError, subprocess.TimeoutExpired):
        return "No pude leer el reproductor, patrón."
    if status is None:
        # El que suena; si no suena ninguno, el primero (el que daba playerctl).
        _, estado, meta = _playing(players) or players[0]
    elif status.returncode != 0:
        return "No hay nada sonando, patrón. El silencio también es música."
    else:
        estado, meta = (status.stdout or "").strip().lower(), ""
    # Primero el strip() a secas: playerctl termina la línea con "\n" y
    # strip(" —") no lo quitaba, así que la frase salía con un salto de línea
    # pegado ("...: Artista — Canción\n"). Luego se limpia el separador huérfano
    # cuando falta el artista o el título.
    cancion = meta.strip().strip("— ")
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
        selector = _player_arg(player)
        if not selector and _ACTIONS[key][0] in _SOBRE_LO_QUE_SUENA:
            # "vacío = el activo": se pregunta quién suena y se le manda a él,
            # con su nombre tal cual (`-p` distingue mayúsculas). Si no suena
            # nadie, sin `-p`: lo que elija playerctl, como siempre. Dentro del
            # mismo try: si esta pregunta ya se colgó, la orden no va a ir mejor.
            sonando = _playing(_players())
            if sonando is not None:
                selector = ["-p", sonando[0]]
        result = _run([*selector, *_ACTIONS[key]])
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
