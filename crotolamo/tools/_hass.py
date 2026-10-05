"""Fontanería HTTP de las tools de domótica (home.py).

Aquí vive lo que NO es una @tool: la llamada REST a Home Assistant con urllib
(stdlib pura, igual que core/glm.py), el bearer token desde la env y la
traducción de fallos de red a resultados que la tool convierte en mensajes en
personaje. home.py queda solo con las funciones @tool que ve el LLM.

OJO: aquí NO aplica el guardia anti-SSRF de _web.py. Allá la URL la elige el
LLM (a veces a partir de texto de otra página); aquí la base_url viene de la
config del patrón ([home] del toml) y es deliberadamente de red local.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from typing import Any

from crotolamo.settings import get_settings

# El token de acceso de larga duración de Home Assistant. NUNCA va en el toml
# (que va a git): la env es el sitio correcto, igual que CROTOLAMO_GLM_API_KEY.
TOKEN_ENV = "CROTOLAMO_HASS_TOKEN"

DEFAULT_BASE_URL = "http://homeassistant.local:8123"
# Timeout corto: HA vive en la LAN; si no responde en 5s, está caído. Un timeout
# largo dejaría la voz "pensando" por una request que no va a llegar.
_TIMEOUT = 5.0
_MAX_BODY = 500_000  # las respuestas de /api/states son chicas; esto sobra


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Los 3xx se tratan como error, nunca se siguen.

    urlopen sigue los redirects REENVIANDO los headers originales: un 302 desde
    homeassistant.local mandaría el bearer token a otro host. La API de HA no
    redirige en operación normal, así que negarse es gratis y cierra la fuga.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect no permitido", headers, fp)


def _build_opener() -> urllib.request.OpenerDirector:
    """Opener sin redirects y SIN proxy.

    build_opener añade el ProxyHandler por defecto, que lee http_proxy/
    https_proxy del entorno: con un proxy configurado, la request a HA (con el
    bearer token en el header) salía hacia el proxy en vez de a la LAN.
    ProxyHandler({}) lo sustituye por uno vacío. Es una factoría (y no solo la
    constante) para que los tests lo reconstruyan bajo un entorno con proxy.
    """
    return urllib.request.build_opener(_NoRedirectHandler, urllib.request.ProxyHandler({}))


_OPENER = _build_opener()


def _find_token() -> str | None:
    token = os.environ.get(TOKEN_ENV)
    return token.strip() if token and token.strip() else None


def _base_url() -> str:
    base = get_settings().raw.get("home", {}).get("base_url", DEFAULT_BASE_URL)
    return str(base).rstrip("/")


def _http_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: float = _TIMEOUT,
) -> Any:
    """Request HTTP con urllib que devuelve el body parseado como JSON.

    Helper separado para que los tests lo parcheen sin tocar la red (mismo rol
    que _web._http_get). Lanza HTTPError/URLError/OSError si la red falla y
    json.JSONDecodeError si el body no es JSON.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    all_headers = dict(headers)
    if data is not None:
        all_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=all_headers)
    with _OPENER.open(request, timeout=timeout) as response:  # noqa: S310
        raw = response.read(_MAX_BODY)
    return json.loads(raw) if raw.strip() else {}


def _hass_call(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[bool, Any]:
    """Llama a la REST API de Home Assistant. Devuelve (ok, data_o_error).

    Ante fallo NO lanza: devuelve (False, código_corto) y la tool lo traduce a
    un mensaje en personaje. Códigos: "sin_token", "token_rechazado",
    "no_encontrado", "http_<código>", "conexion: <detalle>", "respuesta_rara".
    """
    token = _find_token()
    if token is None:
        return False, "sin_token"

    url = _base_url() + path
    headers = {"Authorization": f"Bearer {token}"}
    try:
        return True, _http_json(method, url, headers, payload)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return False, "token_rechazado"
        if error.code == 404:
            return False, "no_encontrado"
        return False, f"http_{error.code}"
    except json.JSONDecodeError:
        return False, "respuesta_rara"
    # HTTPException cubre InvalidURL (entity_id con \n) y ValueError cubre
    # UnicodeEncodeError (entity_id con acento sin quotear): ninguno hereda de
    # OSError y ambos saltan en putrequest, ANTES de tocar la red.
    except (TimeoutError, urllib.error.URLError, OSError,
            http.client.HTTPException, ValueError) as error:
        return False, f"conexion: {error}"
