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

from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import normalize_key as _norm
from crotolamo.tools.base import run_cmd, tool

log = get_logger("windows")

_NO_HYPRCTL = "No tengo hyprctl, patrón. Esto solo funciona dentro de Hyprland."
_MAX_TITLE = 60
# Tope del detalle de un fallo de hyprctl que se le dice al patrón (una línea).
_MAX_DETALLE = 200


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return run_cmd(["hyprctl", *args], timeout=5)


def _detalle(*salidas: str | None) -> str:
    """Primera línea con texto de lo que imprimió hyprctl, recortada."""
    for salida in salidas:
        for line in (salida or "").splitlines():
            if line.strip():
                return _truncate(line.strip(), _MAX_DETALLE)
    return ""


def _clients() -> list[dict[str, Any]] | str | None:
    """Lista de ventanas de `hyprctl clients -j`.

    None si hyprctl no está instalado. Si está pero falló, la frase para el
    patrón con el motivo. Antes los dos casos daban None y todo se contaba como
    "No tengo hyprctl": con el listener arrancado por systemd sin
    HYPRLAND_INSTANCE_SIGNATURE en el entorno, hyprctl existe y sale con 1, y
    se le decía al patrón que le faltaba algo que tiene, sin dejar en el log lo
    que hyprctl había contestado.
    """
    if not shutil.which("hyprctl"):
        return None
    try:
        result = _run(["clients", "-j"])
    except (OSError, subprocess.TimeoutExpired) as error:
        log.warning("hyprctl clients -j no contestó: %s", error)
        return "No pude hablar con Hyprland, patrón: hyprctl no me contestó."
    if result.returncode != 0:
        # hyprctl escribe sus quejas en stdout ("HYPRLAND_INSTANCE_SIGNATURE not
        # set!"); stderr primero por si alguna versión lo usa.
        detalle = _detalle(result.stderr, result.stdout)
        log.warning("hyprctl clients -j salió con %s: %s", result.returncode,
                    detalle or "sin salida")
        return "hyprctl se quejó, patrón: " + (
            detalle or f"salió con código {result.returncode} sin dar detalles")
    texto = (result.stdout or "").strip()
    try:
        data = json.loads(texto or "[]")
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list):
        # Salida 0 pero no es la lista de ventanas: hay versiones de hyprctl que
        # avisan de un error y aun así salen con 0.
        detalle = _detalle(texto)
        log.warning("hyprctl clients -j no devolvió la lista de ventanas: %s", detalle)
        return f"hyprctl me contestó algo que no entiendo, patrón: {detalle}"
    return data


def _dispatch(args: list[str]) -> subprocess.CompletedProcess:
    """Corre `hyprctl dispatch ...`. Separado para poder fakearlo en tests."""
    return _run(["dispatch", *args])


def _pgrep(name: str) -> bool:
    """True si hay algún proceso cuyo comando matchea `name` (sin ventana o no)."""
    name = name.strip()
    # `pgrep -fi ""` matchea CUALQUIER proceso (patrón vacío = todo): con un
    # nombre vacío se afirmaba que había "un proceso suyo corriendo". Y `--`
    # para que un nombre que empiece por guion no se lea como opción de pgrep.
    if not name:
        return False
    try:
        result = run_cmd(["pgrep", "-fi", "--", name], timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _workspace_id(client: dict[str, Any]) -> Any:
    ws = client.get("workspace") or {}
    return ws.get("id", "?") if isinstance(ws, dict) else "?"


def _truncate(text: str, limit: int = _MAX_TITLE) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _match(clients: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Ventanas cuyo class o title contiene `name` (sin acentos ni mayúsculas).

    De mejor a peor coincidencia: primero la app que se llama así (class
    exacto), luego la app cuyo class lo contiene y al final las que solo lo
    llevan en el título; dentro de cada grupo, el orden de hyprctl. focus_window
    y close_window actúan sobre la PRIMERA: sin este orden, "cierra opera"
    cerraba un kitty con `nvim notas_opera.md` si hyprctl lo listaba antes que
    el navegador.
    """
    query = _norm(name.strip())
    if not query:
        return []
    ranked: list[tuple[int, dict[str, Any]]] = []
    for c in clients:
        clase = _norm(str(c.get("class", "")))
        if clase == query:
            rank = 0
        elif query in clase:
            rank = 1
        elif query in _norm(str(c.get("title", ""))):
            rank = 2
        else:
            continue
        ranked.append((rank, c))
    # sorted() es estable: empates en el orden en que los dio hyprctl.
    return [c for _, c in sorted(ranked, key=lambda par: par[0])]


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
    if not isinstance(clients, list):
        return clients or _NO_HYPRCTL
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
    if not isinstance(clients, list):
        return clients or _NO_HYPRCTL
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
    if not isinstance(clients, list):
        return clients or _NO_HYPRCTL
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
    if not isinstance(clients, list):
        return clients or _NO_HYPRCTL
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
