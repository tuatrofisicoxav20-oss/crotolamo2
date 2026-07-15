"""Tools de cámaras vía Frigate (NVR).

- camera_events: eventos detectados (persona, coche...) en las últimas horas,
  como texto consolidado para que el LLM le resuma al patrón lo relevante.
- camera_snapshot: URL del snapshot del último evento de una cámara.

Los alias amigables ("entrada") se mapean a nombres reales de Frigate vía
[cameras.names] del toml; un nombre sin alias se usa tal cual. La fontanería
HTTP (timeouts, traducción de fallos) vive en _frigate.py.
"""

from __future__ import annotations

import time
from typing import Any

from crotolamo.settings import get_settings
from crotolamo.tools import _frigate
from crotolamo.tools.base import normalize_key, tool

# Etiquetas de Frigate -> español, para que el resumen suene a persona y no a API.
_LABELS_ES = {
    "person": "persona",
    "car": "coche",
    "dog": "perro",
    "cat": "gato",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "bird": "pájaro",
    "package": "paquete",
}

_MAX_EVENTS = 50
_MAX_HOURS = 168  # una semana; más que eso ya no es "qué pasó", es arqueología

_SIN_FILTRO = {"", "todas", "todo", "all"}


def _resolve_camera(camera: str) -> str:
    """Alias amigable -> nombre real de Frigate; sin alias, el nombre va tal cual."""
    names = get_settings().raw.get("cameras", {}).get("names", {})
    if not isinstance(names, dict):
        return camera.strip()
    key = normalize_key(camera)
    for alias, real in names.items():
        if normalize_key(alias) == key:
            return str(real)
    return camera.strip()


def _explica_fallo(error: str) -> str:
    return (f"No pude hablar con Frigate, patrón ({error}). "
            "¿Andan vivas las cámaras?")


def _describe(event: dict[str, Any]) -> str:
    label = str(event.get("label", "algo"))
    etiqueta = _LABELS_ES.get(label, label)
    camara = event.get("camera", "?")
    start = event.get("start_time")
    end = event.get("end_time")
    hora = time.strftime("%H:%M", time.localtime(start)) if start else "??:??"
    if start and end:
        duracion = f"{round(end - start)}s"
    else:
        duracion = "sigue en curso"
    return f"- {etiqueta} en la cámara {camara} a las {hora} ({duracion})"


@tool
def camera_events(camera: str, hours: float = 6) -> str:
    """Consulta los eventos que detectaron las cámaras (personas, coches, etc.)
    en las últimas horas vía Frigate, para resumirle al patrón lo relevante.

    Args:
        camera: alias amigable de la cámara según [cameras.names] de la config
            (por ejemplo "entrada"), o su nombre real en Frigate. Vacío o
            "todas" = todas las cámaras.
        hours: cuántas horas hacia atrás revisar (por defecto 6).
    """
    # Forma positiva a propósito: `hours <= 0 or hours > MAX` deja pasar NaN
    # (toda comparación con NaN da False), y json.loads acepta el literal NaN,
    # así que un tool-call del LLM puede colarlo hasta aquí.
    if not (0 < hours <= _MAX_HOURS):
        return (f"Esas horas no me cuadran, patrón ({hours}). "
                f"Dame algo entre 0 y {_MAX_HOURS}.")

    params: dict[str, Any] = {
        "after": int(time.time() - hours * 3600),
        "limit": _MAX_EVENTS,
    }
    filtra = normalize_key(camera) not in _SIN_FILTRO
    if filtra:
        params["camera"] = _resolve_camera(camera)

    ok, data = _frigate._frigate_get("/api/events", params)
    if not ok:
        return _explica_fallo(data)
    if not isinstance(data, list):
        return _explica_fallo("respuesta_rara")

    donde = f"la cámara '{camera}'" if filtra else "las cámaras"
    if not data:
        return (f"Ni un alma, patrón: cero eventos en {donde} "
                f"en las últimas {hours:g} horas.")

    lines = [f"Eventos de {donde} en las últimas {hours:g} horas "
             f"({len(data)} en total):"]
    lines.extend(_describe(event) for event in data)
    return "\n".join(lines)


@tool
def camera_snapshot(camera: str) -> str:
    """Devuelve la URL del snapshot (foto) del último evento de una cámara,
    para abrirla o pasársela al patrón.

    Args:
        camera: alias amigable de la cámara según [cameras.names] de la config
            (por ejemplo "entrada"), o su nombre real en Frigate.
    """
    real = _resolve_camera(camera)
    if not real:
        return "Necesito saber de qué cámara, patrón."

    ok, data = _frigate._frigate_get("/api/events", {"camera": real, "limit": 1})
    if not ok:
        return _explica_fallo(data)
    # Un dict de error con 200 NO es "sin eventos": distinguirlo evita el
    # mensaje engañoso (mismo trato que en camera_events).
    if not isinstance(data, list):
        return _explica_fallo("respuesta_rara")
    if not data:
        return (f"La cámara '{camera}' no tiene ningún evento registrado, patrón. "
                "No hay foto que enseñar.")

    event = data[0]
    event_id = event.get("id")
    if not event_id:
        return _explica_fallo("respuesta_rara")
    label = _LABELS_ES.get(str(event.get("label", "")), str(event.get("label", "algo")))
    if event.get("has_snapshot") is False:
        return (f"El último evento de la cámara '{camera}' fue de {label}, patrón, "
                "pero Frigate no le guardó foto.")
    url = f"{_frigate.base_url()}/api/events/{event_id}/snapshot.jpg"
    return (f"El último evento de la cámara '{camera}' fue de {label}, patrón. "
            f"Aquí la foto: {url}")
