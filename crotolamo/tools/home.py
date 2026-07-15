"""Tools de domótica vía Home Assistant.

- light_control: prende/apaga/alterna una luz por su nombre amigable.
- home_state: lee el estado actual de una entidad mapeada.

Los nombres amigables ("xbox") se mapean a entity_id reales vía [home.lights]
del toml. La fontanería HTTP (bearer, timeouts, traducción de fallos) vive en
_hass.py; aquí quedan solo las @tool.
"""

from __future__ import annotations

from urllib.parse import quote

from crotolamo.settings import get_settings
from crotolamo.tools import _hass
from crotolamo.tools.base import normalize_key, tool

# acción del patrón -> servicio de la REST API (POST /api/services/light/<servicio>).
_SERVICES = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}

_HECHO = {
    "on": "prendí",
    "off": "apagué",
    "toggle": "le di la vuelta a",
}


def _lights() -> dict[str, str]:
    """Mapa nombre amigable (normalizado) -> entity_id, desde [home.lights]."""
    raw = get_settings().raw.get("home", {}).get("lights", {})
    if not isinstance(raw, dict):
        return {}
    return {normalize_key(name): str(eid) for name, eid in raw.items()}


def _resolve_light(target: str) -> tuple[str | None, str | None]:
    """Devuelve (entity_id, mensaje_de_error). Exactamente uno es None."""
    lights = _lights()
    if not lights:
        return None, ("No tengo ninguna luz configurada, patrón. Agrega tus luces "
                      "en [home.lights] de la config y armamos el desmadre.")
    entity_id = lights.get(normalize_key(target))
    if entity_id is None:
        nombres = ", ".join(sorted(lights))
        return None, (f"No conozco ninguna luz llamada '{target}', patrón. "
                      f"Las que traigo fichadas: {nombres}.")
    return entity_id, None


def _explica_fallo(error: str) -> str:
    """Traduce el código de fallo de _hass_call a un mensaje en personaje."""
    if error == "sin_token":
        return ("Me falta el token de la casa, patrón. Exporta CROTOLAMO_HASS_TOKEN "
                "con un token de acceso de Home Assistant y quedamos.")
    if error == "token_rechazado":
        return ("Home Assistant me bateó el token, patrón. Revisa que "
                "CROTOLAMO_HASS_TOKEN siga vigente.")
    if error == "no_encontrado":
        return ("Home Assistant no encontró esa entidad, patrón. Revisa el "
                "entity_id en [home.lights] de la config.")
    return (f"No pude hablar con Home Assistant, patrón ({error}). "
            "¿Anda vivo el cacharro?")


# safe=True: prender/apagar/alternar una luz es reversible al instante, no
# destruye nada ni toca datos; hacer que el guard pida "¿lo hago, patrón?" por
# cada foco mataría la gracia de la domótica por voz. Si algún día se agregan
# cerraduras o el boiler, ESAS sí merecen safe=False.
@tool
def light_control(target: str, action: str) -> str:
    """Prende, apaga o alterna una luz de la casa vía Home Assistant.

    Args:
        target: nombre amigable de la luz, como está en [home.lights] de la config
            (por ejemplo "xbox").
        action: qué hacer con la luz: "on" (prender), "off" (apagar) o
            "toggle" (alternar).
    """
    service = _SERVICES.get(normalize_key(action))
    if service is None:
        return (f"Esa acción '{action}' no me suena, patrón. Con las luces sé hacer "
                "on, off y toggle, nada más.")

    entity_id, problema = _resolve_light(target)
    if entity_id is None:
        return problema or ""

    ok, data = _hass._hass_call(
        "POST", f"/api/services/light/{service}", {"entity_id": entity_id}
    )
    if not ok:
        return _explica_fallo(data)
    # HA responde 200 con la lista de estados que CAMBIARON. Con un entity_id
    # inexistente (typo en la config) responde 200 con []: NO hay que cantar
    # victoria por un "prendí" que no ocurrió.
    if data == []:
        return (f"Home Assistant no reconoció la luz '{target}', patrón. "
                f"Revisa el entity_id '{entity_id}' en [home.lights] de la config.")
    accion = normalize_key(action)
    return f"Listo, patrón: {_HECHO[accion]} la luz '{target}'."


@tool
def home_state(target: str) -> str:
    """Consulta el estado actual de una luz o entidad de la casa (encendida,
    apagada, brillo) vía Home Assistant, para contárselo al patrón.

    Args:
        target: nombre amigable de la entidad, como está en [home.lights] de la
            config (por ejemplo "xbox").
    """
    entity_id, problema = _resolve_light(target)
    if entity_id is None:
        return problema or ""

    # quote: un entity_id con acento o caracteres raros reventaría http.client
    # (UnicodeEncodeError/InvalidURL) antes de tocar la red.
    ok, data = _hass._hass_call("GET", f"/api/states/{quote(entity_id, safe='')}")
    if not ok:
        return _explica_fallo(data)
    if not isinstance(data, dict):
        return _explica_fallo("respuesta_rara")

    state = str(data.get("state", "")).lower()
    if state == "on":
        brillo = (data.get("attributes") or {}).get("brightness")
        if isinstance(brillo, (int, float)) and brillo > 0:
            pct = round(brillo * 100 / 255)
            return f"La luz '{target}' está encendida al {pct}% de brillo, patrón."
        return f"La luz '{target}' está encendida, patrón."
    if state == "off":
        return f"La luz '{target}' está apagada, patrón."
    return (f"La entidad '{target}' anda en estado '{state or 'desconocido'}', patrón. "
            "Ni responde ni avisa: misteriosa.")
