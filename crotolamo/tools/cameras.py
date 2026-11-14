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

# /api/events mete por defecto la miniatura de cada evento en base64
# (include_thumbnails=1): cientos de kB por consulta que aquí nadie mira, porque
# solo se usan etiqueta, cámara, horas e id.
_SIN_MINIATURAS = {"include_thumbnails": 0}
# Lo que contestaría un Frigate que no aceptara el parámetro (Flask: 400,
# FastAPI: 422). Ver _eventos.
_RECHAZOS = ("http_400", "http_422")


def _alias() -> dict[str, Any]:
    """La tabla [cameras.names] del toml; {} si falta o no es una tabla."""
    names = get_settings().raw.get("cameras", {}).get("names", {})
    return names if isinstance(names, dict) else {}


def _resolve_camera(camera: str) -> str:
    """Alias amigable -> nombre real de Frigate; sin alias, el nombre va tal cual."""
    key = normalize_key(camera)
    for alias, real in _alias().items():
        if normalize_key(alias) == key:
            return str(real)
    return camera.strip()


def _eventos(params: dict[str, Any]) -> tuple[bool, Any]:
    """GET /api/events con `params` (que piden _SIN_MINIATURAS).

    Las versiones conocidas de Frigate aceptan include_thumbnails (y las
    demás ignoran lo que no conocen), pero aquí no hay un Frigate con el que
    comprobarlo en todas. Si contesta 400/422, o algo que no es la lista de
    eventos, se repite UNA vez la consulta sin el parámetro, la de siempre:
    una petición más solo en ese caso, y la tool sigue funcionando.
    """
    ok, data = _frigate._frigate_get("/api/events", params)
    if (ok and isinstance(data, list)) or (not ok and data not in _RECHAZOS):
        return ok, data
    sin_parametro = {k: v for k, v in params.items() if k not in _SIN_MINIATURAS}
    return _frigate._frigate_get("/api/events", sin_parametro)


def _explica_fallo(error: str) -> str:
    return (f"No pude hablar con Frigate, patrón ({error}). "
            "¿Andan vivas las cámaras?")


def _camaras_de_frigate() -> set[str] | None:
    """Los nombres de las cámaras de Frigate (las claves de "cameras" en
    /api/config), o None si no se pudieron saber."""
    ok, config = _frigate._frigate_get("/api/config")
    camaras = config.get("cameras") if ok and isinstance(config, dict) else None
    return {str(nombre) for nombre in camaras} if isinstance(camaras, dict) else None


def _fichadas(reales: set[str], alias: dict[str, Any]) -> str:
    """'entrada (front_door), patio': cada cámara con sus alias, si los tiene."""
    partes = []
    for real in sorted(reales):
        suyos = [str(a) for a, r in alias.items() if str(r) == real]
        partes.append(f"{'/'.join(suyos)} ({real})" if suyos else real)
    return ", ".join(partes)


def _camara_desconocida(camera: str, real: str) -> str | None:
    """Con el filtro por cámara vacío: None si la cámara existe (y el vacío es
    de verdad), o el mensaje para el patrón si no.

    Frigate filtra por el nombre exacto y a uno que no existe le contesta []
    igual que a una cámara tranquila. El LLM no ve [cameras.names] y adivina
    ("puerta", "la entrada"), así que un nombre inventado salía como "ni un
    alma" aunque la cámara de verdad tuviera gente: un falso "todo en orden"
    en una tool de seguridad. La lista de cámaras se le pide a Frigate SOLO
    aquí: el camino con eventos no paga la petición de más.
    """
    alias = _alias()
    # Frigate parte el filtro por comas y compara exacto, sin quitar espacios.
    pedidas = real.split(",")
    camaras = _camaras_de_frigate()
    if camaras is not None:
        faltan = ", ".join(f"'{p}'" for p in pedidas if p not in camaras)
        if not faltan:
            return None
        quien = faltan if real == camera.strip() else f"{faltan} (la que tengo para '{camera}')"
        conocidas = _fichadas(camaras, alias)
        lista = f" Las que tiene: {conocidas}." if conocidas else " No me reporta ninguna."
        return f"Frigate no tiene ninguna cámara llamada {quien}, patrón.{lista}"
    # Sin la lista de Frigate solo queda la tabla de alias: lo que está ahí se
    # da por bueno. Lo demás puede ser una cámara sin alias (que va tal cual a
    # propósito) o un invento: se dice la duda, no un "ni un alma".
    en_tabla = {str(r) for r in alias.values()}
    if all(p in en_tabla for p in pedidas):
        return None
    conocidas = _fichadas(en_tabla, alias)
    lista = f" Las que tengo fichadas: {conocidas}." if conocidas else ""
    return (f"Con el nombre '{camera}' no hay ningún evento, patrón, pero no sé si es una "
            f"cámara: no la tengo fichada y Frigate no me dijo cuáles tiene.{lista}")


def _cuando(start: float, hoy: tuple[int, ...]) -> str:
    """'a las HH:MM' si `start` es de `hoy` (año, mes, día local); si no,
    'el YYYY-MM-DD a las HH:MM'."""
    local = time.localtime(start)
    cuando = time.strftime("a las %H:%M", local)
    # La ventana llega a una semana (_MAX_HOURS) y con solo la hora el evento
    # de hoy y el de hace tres días se leían igual. El que no es de hoy lleva
    # la fecha, en el mismo formato ISO del sello [ahora: ...] que ve el LLM,
    # para que pueda compararlas.
    if local[:3] != hoy:
        cuando = time.strftime("el %Y-%m-%d ", local) + cuando
    return cuando


def _describe(event: dict[str, Any], hoy: tuple[int, ...]) -> str:
    """Una línea por evento. `hoy` es (año, mes, día) local de la consulta."""
    label = str(event.get("label", "algo"))
    etiqueta = _LABELS_ES.get(label, label)
    camara = event.get("camera", "?")
    start = event.get("start_time")
    end = event.get("end_time")
    cuando = _cuando(start, hoy) if start else "a las ??:??"
    if start and end:
        duracion = f"{round(end - start)}s"
    else:
        duracion = "sigue en curso"
    return f"- {etiqueta} en la cámara {camara} {cuando} ({duracion})"


@tool(parallel_safe=True)
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
        # Uno de más que no se lista: Frigate corta en `limit` sin avisar, y es
        # la única forma de saber si los _MAX_EVENTS son todos o solo los
        # últimos.
        "limit": _MAX_EVENTS + 1,
        **_SIN_MINIATURAS,
    }
    filtra = normalize_key(camera) not in _SIN_FILTRO
    if filtra:
        params["camera"] = _resolve_camera(camera)

    ok, data = _eventos(params)
    if not ok:
        return _explica_fallo(data)
    if not isinstance(data, list):
        return _explica_fallo("respuesta_rara")

    donde = f"la cámara '{camera}'" if filtra else "las cámaras"
    if not data:
        desconocida = _camara_desconocida(camera, params["camera"]) if filtra else None
        if desconocida:
            return desconocida
        return (f"Ni un alma, patrón: cero eventos en {donde} "
                f"en las últimas {hours:g} horas.")

    # Frigate devuelve los más recientes primero: si llegó el de más, la lista
    # está cortada y decir "50 en total" era contarle al patrón 50 de 300.
    if len(data) > _MAX_EVENTS:
        data = data[:_MAX_EVENTS]
        cuantos = f"los {_MAX_EVENTS} más recientes; hay más"
    else:
        cuantos = f"{len(data)} en total"
    lines = [f"Eventos de {donde} en las últimas {hours:g} horas ({cuantos}):"]
    hoy = time.localtime()[:3]
    lines.extend(_describe(event, hoy) for event in data)
    return "\n".join(lines)


@tool(parallel_safe=True)
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

    ok, data = _eventos({"camera": real, "limit": 1, **_SIN_MINIATURAS})
    if not ok:
        return _explica_fallo(data)
    # Un dict de error con 200 NO es "sin eventos": distinguirlo evita el
    # mensaje engañoso (mismo trato que en camera_events).
    if not isinstance(data, list):
        return _explica_fallo("respuesta_rara")
    if not data:
        return _camara_desconocida(camera, real) or (
            f"La cámara '{camera}' no tiene ningún evento registrado, patrón. "
            "No hay foto que enseñar.")

    event = data[0]
    event_id = event.get("id")
    if not event_id:
        return _explica_fallo("respuesta_rara")
    label = _LABELS_ES.get(str(event.get("label", "")), str(event.get("label", "algo")))
    # La consulta no lleva ventana: el "último" puede ser de hace días, y sin
    # la hora el LLM lo contaba como de ahorita.
    start = event.get("start_time")
    cuando = f" {_cuando(start, time.localtime()[:3])}" if start else ""
    if event.get("has_snapshot") is False:
        return (f"El último evento de la cámara '{camera}' fue de {label}{cuando}, "
                "patrón, pero Frigate no le guardó foto.")
    url = f"{_frigate.base_url()}/api/events/{event_id}/snapshot.jpg"
    return (f"El último evento de la cámara '{camera}' fue de {label}{cuando}, patrón. "
            f"Aquí la foto: {url}")
