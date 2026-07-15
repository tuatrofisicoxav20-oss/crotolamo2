"""Fontanería HTTP de las tools de cámaras (cameras.py).

Aquí vive lo que NO es una @tool: el GET a la API de Frigate con urllib
(stdlib pura, igual que core/glm.py) y la traducción de fallos de red a
resultados que la tool convierte en mensajes en personaje. cameras.py queda
solo con las funciones @tool que ve el LLM.

OJO: aquí NO aplica el guardia anti-SSRF de _web.py. Allá la URL la elige el
LLM; aquí la base_url viene de la config del patrón ([cameras] del toml) y es
deliberadamente de red local. Frigate no pide token.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from crotolamo.settings import get_settings

DEFAULT_BASE_URL = "http://frigate.local:5000"
# Timeout corto: Frigate vive en la LAN; si no responde en 6s, está caído.
_TIMEOUT = 6.0
_MAX_BODY = 1_500_000  # /api/events con límite de eventos cabe de sobra


def base_url() -> str:
    base = get_settings().raw.get("cameras", {}).get("base_url", DEFAULT_BASE_URL)
    return str(base).rstrip("/")


def _http_json(url: str, timeout: float = _TIMEOUT) -> Any:
    """GET con urllib que devuelve el body parseado como JSON.

    Helper separado para que los tests lo parcheen sin tocar la red (mismo rol
    que _web._http_get). Lanza HTTPError/URLError/OSError si la red falla y
    json.JSONDecodeError si el body no es JSON.
    """
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        raw = response.read(_MAX_BODY)
    return json.loads(raw) if raw.strip() else []


def _frigate_get(path: str, params: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """GET a la API de Frigate. Devuelve (ok, data_o_error).

    Ante fallo NO lanza: devuelve (False, código_corto) y la tool lo traduce a
    un mensaje en personaje. Códigos: "http_<código>", "conexion: <detalle>",
    "respuesta_rara".
    """
    url = base_url() + path
    if params:
        url += "?" + urlencode(params)
    try:
        return True, _http_json(url)
    except urllib.error.HTTPError as error:
        return False, f"http_{error.code}"
    except (TimeoutError, urllib.error.URLError, OSError) as error:
        return False, f"conexion: {error}"
    except json.JSONDecodeError:
        return False, "respuesta_rara"
