"""Tests de light_control / home_state SIN red real.

Se parchea _hass._http_json (el único punto que toca la red), igual que
test_web.py parchea _web._http_get. La config [home] se inyecta en el singleton
de settings (patrón de conftest.fake_project), así los tests no dependen de los
entity_id reales de la casa de Emiliano.
"""

import urllib.error

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import _hass, default_registry, home

TOKEN = "token-de-mentiras"


@pytest.fixture
def home_cfg(monkeypatch):
    """Config [home] de mentiras en el singleton de settings + token en la env."""
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "home", {
        "base_url": "http://hass.test:8123",
        "lights": {"xbox": "light.xbox_led", "sala": "light.sala_techo"},
    })
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    monkeypatch.setenv(_hass.TOKEN_ENV, TOKEN)
    return real


def _patch_http(monkeypatch, respuesta, capture=None):
    """Sustituye _hass._http_json; guarda (method, url, headers, payload) en capture."""
    def fake(method, url, headers, payload=None, timeout=5.0):
        if capture is not None:
            capture.append((method, url, headers, payload))
        return respuesta

    monkeypatch.setattr(_hass, "_http_json", fake)


# ---------------------------------------------------------------------------
# light_control
# ---------------------------------------------------------------------------

# HA responde el POST con la lista de estados que CAMBIARON; no-vacía = actuó.
CAMBIO = [{"entity_id": "light.xbox_led", "state": "on"}]


def test_light_on_camino_feliz(home_cfg, monkeypatch):
    seen = []
    _patch_http(monkeypatch, CAMBIO, capture=seen)
    out = home.light_control("xbox", "on")

    method, url, headers, payload = seen[0]
    assert method == "POST"
    assert url == "http://hass.test:8123/api/services/light/turn_on"
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert payload == {"entity_id": "light.xbox_led"}
    assert "patrón" in out
    assert "xbox" in out


def test_light_off_y_toggle_usan_el_servicio_correcto(home_cfg, monkeypatch):
    seen = []
    _patch_http(monkeypatch, CAMBIO, capture=seen)
    home.light_control("sala", "off")
    home.light_control("sala", "toggle")
    assert seen[0][1].endswith("/api/services/light/turn_off")
    assert seen[1][1].endswith("/api/services/light/toggle")


def test_light_entidad_que_ha_no_reconoce(home_cfg, monkeypatch):
    """HA responde 200 con [] si el entity_id no existe (typo en la config):
    NO hay que cantar victoria con 'prendí la luz'."""
    _patch_http(monkeypatch, [])
    out = home.light_control("xbox", "on")
    assert "prendí" not in out
    assert "entity_id" in out
    assert "patrón" in out


def test_light_accion_invalida_no_toca_la_red(home_cfg, monkeypatch):
    called = []
    _patch_http(monkeypatch, [], capture=called)
    out = home.light_control("xbox", "explota")
    assert called == []
    assert "on" in out and "off" in out and "toggle" in out


def test_light_target_desconocido_lista_las_que_hay(home_cfg, monkeypatch):
    called = []
    _patch_http(monkeypatch, [], capture=called)
    out = home.light_control("cocina", "on")
    assert called == []
    assert "xbox" in out and "sala" in out
    assert "patrón" in out


def test_light_sin_luces_configuradas(home_cfg, monkeypatch):
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "home", {"base_url": "http://hass.test:8123"})
    called = []
    _patch_http(monkeypatch, CAMBIO, capture=called)
    out = home.light_control("xbox", "on")
    assert called == []  # sin luces mapeadas no hay nada que llamar
    assert "config" in out.lower()


def test_light_sin_token_avisa_sin_crashear(home_cfg, monkeypatch):
    monkeypatch.delenv(_hass.TOKEN_ENV, raising=False)
    called = []
    _patch_http(monkeypatch, [], capture=called)
    out = home.light_control("xbox", "on")
    assert called == []  # ni intenta la red sin credencial
    assert _hass.TOKEN_ENV in out
    assert "patrón" in out


def test_light_ha_caido_mensaje_en_personaje(home_cfg, monkeypatch):
    def boom(method, url, headers, payload=None, timeout=5.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(_hass, "_http_json", boom)
    out = home.light_control("xbox", "on")
    assert "patrón" in out
    assert "Traceback" not in out


def test_light_token_rechazado(home_cfg, monkeypatch):
    def boom(method, url, headers, payload=None, timeout=5.0):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(_hass, "_http_json", boom)
    out = home.light_control("xbox", "on")
    # Mensaje del 401 ("vigente"), NO el de token ausente ("me falta el token").
    assert "vigente" in out
    assert "falta" not in out.lower()
    assert "patrón" in out


def test_light_entity_404(home_cfg, monkeypatch):
    def boom(method, url, headers, payload=None, timeout=5.0):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(_hass, "_http_json", boom)
    out = home.home_state("xbox")
    assert "entity_id" in out


def test_light_respuesta_no_json(home_cfg, monkeypatch):
    import json as json_mod

    def boom(method, url, headers, payload=None, timeout=5.0):
        raise json_mod.JSONDecodeError("basura", "<html>", 0)

    monkeypatch.setattr(_hass, "_http_json", boom)
    out = home.light_control("xbox", "on")
    assert "patrón" in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# home_state
# ---------------------------------------------------------------------------

def test_home_state_encendida_con_brillo(home_cfg, monkeypatch):
    seen = []
    _patch_http(
        monkeypatch,
        {"state": "on", "attributes": {"brightness": 128}},
        capture=seen,
    )
    out = home.home_state("xbox")
    assert seen[0][0] == "GET"
    assert seen[0][1] == "http://hass.test:8123/api/states/light.xbox_led"
    assert "encendida" in out.lower()
    assert "50" in out  # 128/255 ≈ 50%


def test_home_state_apagada(home_cfg, monkeypatch):
    _patch_http(monkeypatch, {"state": "off", "attributes": {}})
    out = home.home_state("sala")
    assert "apagada" in out.lower()


def test_home_state_entidad_no_disponible(home_cfg, monkeypatch):
    _patch_http(monkeypatch, {"state": "unavailable", "attributes": {}})
    out = home.home_state("xbox")
    # Debe reportar el estado raro tal cual, no "encendida" ni "apagada".
    assert "unavailable" in out
    assert "encendida" not in out.lower() and "apagada" not in out.lower()


def test_home_state_respuesta_que_no_es_dict(home_cfg, monkeypatch):
    """GET /api/states (entity vacío o proxy raro) devuelve una LISTA: no debe
    reventar con AttributeError, sino fallar en personaje."""
    _patch_http(monkeypatch, [{"state": "on"}])
    out = home.home_state("xbox")
    assert "patrón" in out
    assert "Traceback" not in out


def test_home_state_entity_id_con_acento_va_quoted(home_cfg, monkeypatch):
    """Un entity_id con acento (setup en español) no debe reventar la request:
    se quotea en la URL en vez de dejar que http.client lance UnicodeEncodeError."""
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "home", {
        "base_url": "http://hass.test:8123",
        "lights": {"recamara": "light.recámara"},
    })
    seen = []
    _patch_http(monkeypatch, {"state": "off", "attributes": {}}, capture=seen)
    out = home.home_state("recamara")
    assert "light.rec%C3%A1mara" in seen[0][1]
    assert "apagada" in out.lower()


def test_home_state_target_desconocido(home_cfg, monkeypatch):
    called = []
    _patch_http(monkeypatch, {}, capture=called)
    out = home.home_state("garage")
    assert called == []
    assert "xbox" in out and "sala" in out


def test_home_state_ha_caido(home_cfg, monkeypatch):
    def boom(method, url, headers, payload=None, timeout=5.0):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(_hass, "_http_json", boom)
    out = home.home_state("xbox")
    assert "patrón" in out
    assert "Traceback" not in out


def test_home_state_sin_token(home_cfg, monkeypatch):
    monkeypatch.delenv(_hass.TOKEN_ENV, raising=False)
    called = []
    _patch_http(monkeypatch, {}, capture=called)
    out = home.home_state("xbox")
    assert called == []  # sin credencial ni intenta la red
    assert _hass.TOKEN_ENV in out


# ---------------------------------------------------------------------------
# _hass: los redirects NO se siguen (no reenviar el bearer a otro host)
# ---------------------------------------------------------------------------

def test_hass_no_sigue_redirects(home_cfg, monkeypatch):
    """Un 3xx desde HA debe morir como error, no seguirse: urllib reenvía el
    header Authorization al destino del redirect (fugaría el token)."""
    import http.server
    import threading

    class _Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — nombre que exige BaseHTTPRequestHandler
            self.send_response(302)
            self.send_header("Location", "http://otro-host.test/roba-token")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        real = settings_mod.get_settings()
        monkeypatch.setitem(real.raw, "home", {
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "lights": {"xbox": "light.xbox_led"},
        })
        ok, error = _hass._hass_call("GET", "/api/states/light.xbox_led")
        assert ok is False
        assert "http_302" in error
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# registro en el registry
# ---------------------------------------------------------------------------

def test_tools_de_home_registradas():
    reg = default_registry()
    for name in ("light_control", "home_state"):
        assert name in reg.names()
        # Prender/apagar una luz es reversible al instante: safe=True (sin guard).
        assert reg.get(name).safe is True
