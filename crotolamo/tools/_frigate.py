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

import http.client
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


def _build_opener() -> urllib.request.OpenerDirector:
    """Opener SIN proxy.

    urlopen usa el opener global, que trae el ProxyHandler por defecto (lee
    http_proxy/https_proxy del entorno): con un proxy configurado la request
    iba al proxy en vez de a Frigate, que vive en la LAN. ProxyHandler({}) lo
    sustituye por uno vacío. Factoría para que los tests lo reconstruyan.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER: urllib.request.OpenerDirector | None = None


def _opener() -> urllib.request.OpenerDirector:
    """El opener del módulo, construido en el PRIMER uso y no al importar.

    En Python 3.12+ HTTPSHandler crea su SSLContext (y carga el bundle de CAs
    del sistema, ~20 ms) en el constructor: como constante de módulo se pagaba
    en cada arranque del shell/listener/doctor aunque nunca se consultara una
    cámara y la base_url sea http://. Vive en el global `_OPENER` para que
    sustituirlo desde fuera (tests) siga mandando.
    """
    global _OPENER
    if _OPENER is None:
        _OPENER = _build_opener()
    return _OPENER


def _http_json(url: str, timeout: float = _TIMEOUT) -> Any:
    """GET con urllib que devuelve el body parseado como JSON.

    Helper separado para que los tests lo parcheen sin tocar la red (mismo rol
    que _web._http_get). Lanza HTTPError/URLError/OSError si la red falla y
    json.JSONDecodeError si el body no es JSON.
    """
    request = urllib.request.Request(url)
    with _opener().open(request, timeout=timeout) as response:  # noqa: S310
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
    # UnicodeDecodeError: JSON que no es UTF-8 (json.loads lo decodifica antes
    # de parsear y ese fallo es ValueError, NO JSONDecodeError).
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "respuesta_rara"
    # HTTPException (IncompleteRead de un chunked cortado, BadStatusLine de un
    # puerto que no habla HTTP...) no hereda de OSError, y ValueError es lo que
    # lanza Request() con una base_url sin esquema, antes de tocar la red: sin
    # ellos el "NO lanza" de arriba era mentira y la tool salía por el
    # "reventó" genérico del registry.
    except (TimeoutError, urllib.error.URLError, OSError,
            http.client.HTTPException, ValueError) as error:
        return False, f"conexion: {error}"
