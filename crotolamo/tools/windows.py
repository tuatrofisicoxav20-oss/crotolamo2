"""Estado y control de ventanas vía hyprctl (Hyprland).

Todo pasa por `hyprctl` con comandos fijos: nada de bash generado. Las tools de
solo-lectura (listar, estado de una app) son `direct=True` porque su output ya
es una frase lista para el patrón. Enfocar es inofensivo (safe); cerrar ventana
pide confirmación (`safe=False`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from crotolamo.tools.base import normalize_key as _norm
from crotolamo.tools.base import run_cmd, tool

_NO_HYPRCTL = "No tengo hyprctl, patrón. Esto solo funciona dentro de Hyprland."
_MAX_TITLE = 60


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return run_cmd(["hyprctl", *args], timeout=5)


def _clients() -> list[dict[str, Any]] | None:
    """Lista de ventanas de `hyprctl clients -j`, o None si hyprctl falta/falla."""
    if not shutil.which("hyprctl"):
        return None
    try:
        result = _run(["clients", "-j"])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _dispatch(args: list[str]) -> subprocess.CompletedProcess:
    """Corre `hyprctl dispatch ...`. Separado para poder fakearlo en tests."""
    return _run(["dispatch", *args])


def _pgrep(name: str) -> bool:
    """True si hay algún proceso cuyo comando matchea `name` (sin ventana o no)."""
    try:
        result = run_cmd(["pgrep", "-fi", name], timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _workspace_id(client: dict[str, Any]) -> Any:
    ws = client.get("workspace") or {}
    return ws.get("id", "?") if isinstance(ws, dict) else "?"


def _truncate(text: str, limit: int = _MAX_TITLE) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _match(clients: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Ventanas cuyo class o title contiene `name` (sin acentos ni mayúsculas)."""
    query = _norm(name.strip())
    if not query:
        return []
    return [
        c
        for c in clients
        if query in _norm(str(c.get("class", ""))) or query in _norm(str(c.get("title", "")))
    ]


def _label(client: dict[str, Any]) -> str:
    clase = str(client.get("class", "")) or "¿app misteriosa?"
    title = _truncate(str(client.get("title", "")))
    return f"{clase} — {title}" if title else clase


@tool(direct=True)
def list_windows() -> str:
    """Lista las apps/ventanas abiertas ahora mismo y en qué workspace están.

    Úsala cuando el patrón pregunte qué apps o ventanas tiene abiertas.
    """
    clients = _clients()
    if clients is None:
        return _NO_HYPRCTL
    if not clients:
        return "No hay ni una ventana abierta, patrón. Escritorio zen total."
    def _sort_key(c: dict[str, Any]) -> tuple[int, str]:
        ws = _workspace_id(c)
        return (ws, "") if isinstance(ws, int) else (10**9, str(ws))

    lines = [f"workspace {_workspace_id(c)}: {_label(c)}" for c in sorted(clients, key=_sort_key)]
    plural = "ventanas abiertas" if len(lines) != 1 else "ventana abierta"
    return f"Tienes {len(lines)} {plural}, patrón:\n" + "\n".join(lines)


@tool(direct=True)
def app_status(name: str) -> str:
    """Dice si una app está abierta, cuántas ventanas tiene y en qué workspace(s).

    Úsala cuando el patrón pregunte "¿está abierto X?" o "¿tengo X corriendo?".

    Args:
        name: nombre (o pedazo del nombre) de la app o ventana a buscar.
    """
    clients = _clients()
    if clients is None:
        return _NO_HYPRCTL
    matches = _match(clients, name)
    if matches:
        workspaces = sorted({str(_workspace_id(c)) for c in matches})
        ws_txt = f"el workspace {workspaces[0]}" if len(workspaces) == 1 else (
            "los workspaces " + ", ".join(workspaces)
        )
        if len(matches) == 1:
            return f"Sí, patrón: {_label(matches[0])} está abierta en {ws_txt}."
        return (
            f"Sí, patrón: '{name}' está abierta con {len(matches)} ventanas en {ws_txt}."
        )
    if shutil.which("pgrep") and _pgrep(name):
        return (
            f"'{name}' no tiene ninguna ventana abierta, patrón, "
            "pero sí hay un proceso suyo corriendo de fondo."
        )
    return f"'{name}' no está corriendo, patrón. Ni ventana ni proceso."


@tool
def focus_window(name: str) -> str:
    """Enfoca (trae al frente) la ventana de una app, buscándola por nombre.

    Úsala cuando el patrón diga "cámbiate a X", "enfoca X" o "ponme X enfrente".

    Args:
        name: nombre (o pedazo del nombre) de la app o ventana a enfocar.
    """
    clients = _clients()
    if clients is None:
        return _NO_HYPRCTL
    matches = _match(clients, name)
    if not matches:
        return f"No encontré ninguna ventana que suene a '{name}', patrón."
    target = matches[0]
    address = str(target.get("address", ""))
    if not address:
        return f"Encontré '{_label(target)}' pero no tiene dirección, patrón. Raro."
    try:
        result = _dispatch(["focuswindow", f"address:{address}"])
    except (OSError, subprocess.TimeoutExpired):
        return "No pude hablar con Hyprland para enfocar, patrón."
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        return f"Hyprland se quejó al enfocar, patrón: {err or 'sin detalles'}"
    if len(matches) > 1:
        return (
            f"Había {len(matches)} ventanas que matchean '{name}', patrón; "
            f"te enfoqué la primera: {_label(target)} "
            f"(workspace {_workspace_id(target)})."
        )
    return f"Listo, patrón: enfocada {_label(target)} (workspace {_workspace_id(target)})."


@tool(safe=False)
def close_window(name: str) -> str:
    """Cierra la ventana de una app, buscándola por nombre. Pide confirmación.

    Úsala cuando el patrón diga "cierra X" o "ciérrame X". Si hay varias
    ventanas que coinciden, cierra SOLO la primera.

    Args:
        name: nombre (o pedazo del nombre) de la app o ventana a cerrar.
    """
    clients = _clients()
    if clients is None:
        return _NO_HYPRCTL
    matches = _match(clients, name)
    if not matches:
        return f"No encontré ninguna ventana que suene a '{name}', patrón. Nada que cerrar."
    target = matches[0]
    address = str(target.get("address", ""))
    if not address:
        return f"Encontré '{_label(target)}' pero no tiene dirección, patrón. No la toco."
    try:
        result = _dispatch(["closewindow", f"address:{address}"])
    except (OSError, subprocess.TimeoutExpired):
        return "No pude hablar con Hyprland para cerrar, patrón."
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        return f"Hyprland se quejó al cerrar, patrón: {err or 'sin detalles'}"
    if len(matches) > 1:
        return (
            f"Ojo, patrón: había {len(matches)} ventanas que matchean '{name}'. "
            f"Cerré SOLO la primera: {_label(target)}. Las demás siguen abiertas."
        )
    return f"Cerrada, patrón: {_label(target)}."
