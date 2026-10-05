"""Tests de camera_events / camera_snapshot SIN red real.

Se parchea _frigate._http_json (el único punto que toca la red), igual que
test_web.py parchea _web._http_get. La config [cameras] se inyecta en el
singleton de settings para no depender del Frigate real.
"""

import time
import urllib.error
import urllib.request

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import _frigate, cameras, default_registry

AHORA = time.time()

EVENTOS = [
    {
        "id": "1720000000.123-abc",
        "camera": "front_door",
        "label": "person",
        "start_time": AHORA - 3600,
        "end_time": AHORA - 3590,
    },
    {
        "id": "1720000100.456-def",
        "camera": "front_door",
        "label": "car",
        "start_time": AHORA - 1800,
        "end_time": None,  # evento aún en curso
    },
]


@pytest.fixture
def cams_cfg(monkeypatch):
    """Config [cameras] de mentiras en el singleton de settings."""
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {
        "base_url": "http://frigate.test:5000",
        "names": {"entrada": "front_door"},
    })
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return real


def _patch_http(monkeypatch, respuesta, capture=None):
    def fake(url, timeout=6.0):
        if capture is not None:
            capture.append(url)
        return respuesta

    monkeypatch.setattr(_frigate, "_http_json", fake)


# ---------------------------------------------------------------------------
# camera_events
# ---------------------------------------------------------------------------

def test_eventos_camino_feliz(cams_cfg, monkeypatch):
    seen = []
    _patch_http(monkeypatch, EVENTOS, capture=seen)
    out = cameras.camera_events("entrada", hours=6)

    # El alias amigable se traduce al nombre real de Frigate en la URL.
    assert "camera=front_door" in seen[0]
    assert "after=" in seen[0]
    assert seen[0].startswith("http://frigate.test:5000/api/events")
    # Etiquetas traducidas y datos por evento para que el LLM sintetice.
    assert "persona" in out.lower()
    assert "coche" in out.lower()
    assert "entrada" in out      # el encabezado usa el alias amigable
    assert "front_door" in out   # cada evento trae la cámara real de Frigate
    assert "sigue en curso" in out  # end_time=None = evento aún abierto
    # El 'after' de la URL corresponde a hace ~6 horas.
    import re
    after = int(re.search(r"after=(\d+)", seen[0]).group(1))
    assert abs(after - (time.time() - 6 * 3600)) < 120


def test_eventos_camara_sin_alias_pasa_tal_cual(cams_cfg, monkeypatch):
    seen = []
    _patch_http(monkeypatch, [], capture=seen)
    cameras.camera_events("patio_trasero", hours=2)
    assert "camera=patio_trasero" in seen[0]


def test_eventos_sin_eventos(cams_cfg, monkeypatch):
    _patch_http(monkeypatch, [])
    out = cameras.camera_events("entrada", hours=6)
    assert "cero eventos" in out
    assert "patrón" in out


def test_eventos_respuesta_que_no_es_lista(cams_cfg, monkeypatch):
    """Frigate a veces responde un dict de error con 200: fallo en personaje,
    no un falso 'cero eventos'."""
    _patch_http(monkeypatch, {"message": "internal error"})
    out = cameras.camera_events("entrada", hours=6)
    assert "cero eventos" not in out
    assert "patrón" in out


def test_eventos_frigate_caido(cams_cfg, monkeypatch):
    def boom(url, timeout=6.0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(_frigate, "_http_json", boom)
    out = cameras.camera_events("entrada", hours=6)
    assert "patrón" in out
    assert "Traceback" not in out


@pytest.mark.parametrize("horas", [-3, 0, 9999, float("nan")])
def test_eventos_horas_invalidas_no_toca_la_red(cams_cfg, monkeypatch, horas):
    """Negativas, cero, fuera de rango y NaN (json.loads acepta el literal NaN,
    así que un tool-call del LLM puede colarlo): mensaje de validación, sin red."""
    called = []
    _patch_http(monkeypatch, EVENTOS, capture=called)
    out = cameras.camera_events("entrada", hours=horas)
    assert called == []
    assert "hora" in out.lower()
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# camera_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_camino_feliz(cams_cfg, monkeypatch):
    seen = []
    _patch_http(monkeypatch, EVENTOS, capture=seen)
    out = cameras.camera_snapshot("entrada")
    assert "camera=front_door" in seen[0]
    # URL del snapshot del PRIMER evento que devuelve Frigate (el más reciente).
    assert "http://frigate.test:5000/api/events/1720000000.123-abc/snapshot.jpg" in out
    assert "patrón" in out


def test_snapshot_sin_eventos(cams_cfg, monkeypatch):
    _patch_http(monkeypatch, [])
    out = cameras.camera_snapshot("entrada")
    assert "snapshot.jpg" not in out
    assert "patrón" in out


def test_snapshot_camara_vacia_no_toca_la_red(cams_cfg, monkeypatch):
    called = []
    _patch_http(monkeypatch, EVENTOS, capture=called)
    out = cameras.camera_snapshot("   ")
    assert called == []
    assert "patrón" in out


def test_snapshot_respuesta_que_no_es_lista(cams_cfg, monkeypatch):
    """Un dict de error con 200 debe fallar en personaje, NO afirmar que la
    cámara 'no tiene eventos' (mensaje engañoso)."""
    _patch_http(monkeypatch, {"message": "internal error"})
    out = cameras.camera_snapshot("entrada")
    assert "no tiene ningún evento" not in out
    assert "patrón" in out


def test_snapshot_evento_sin_id(cams_cfg, monkeypatch):
    """Un evento sin 'id' no debe producir '.../None/snapshot.jpg'."""
    _patch_http(monkeypatch, [{"camera": "front_door", "label": "person"}])
    out = cameras.camera_snapshot("entrada")
    assert "None/snapshot.jpg" not in out
    assert "patrón" in out


def test_snapshot_evento_sin_foto(cams_cfg, monkeypatch):
    """has_snapshot=False: Frigate no guardó foto; avisar en vez de dar un 404."""
    _patch_http(monkeypatch, [{"id": "e1", "label": "person",
                               "camera": "front_door", "has_snapshot": False}])
    out = cameras.camera_snapshot("entrada")
    assert "snapshot.jpg" not in out
    assert "patrón" in out


def test_snapshot_frigate_caido(cams_cfg, monkeypatch):
    def boom(url, timeout=6.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(_frigate, "_http_json", boom)
    out = cameras.camera_snapshot("entrada")
    assert "patrón" in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# _frigate: la request NO sale por el proxy del entorno (http_proxy/https_proxy)
# ---------------------------------------------------------------------------

def test_frigate_opener_sin_proxy(monkeypatch):
    """urlopen usaba el opener global con el ProxyHandler por defecto: con
    http_proxy en el entorno la request iba al proxy y no a Frigate (LAN).
    En el opener propio no queda ningún ProxyHandler con proxies, tampoco
    reconstruido con proxies en el entorno. (ProxyHandler({}) no define
    ningún *_open, así que OpenerDirector ni lo registra en .handlers.)"""
    monkeypatch.setenv("http_proxy", "http://proxy.test:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.test:3128")

    def con_proxies(opener):
        return [h for h in opener.handlers
                if isinstance(h, urllib.request.ProxyHandler) and h.proxies]

    # Sanidad del test: un opener "normal" bajo este entorno SÍ llevaría el proxy.
    assert con_proxies(urllib.request.build_opener())
    for opener in (_frigate._OPENER, _frigate._build_opener()):
        assert con_proxies(opener) == []


# ---------------------------------------------------------------------------
# registro en el registry
# ---------------------------------------------------------------------------

def test_tools_de_camaras_registradas():
    reg = default_registry()
    for name in ("camera_events", "camera_snapshot"):
        assert name in reg.names()
        assert reg.get(name).safe is True  # solo lectura: no actúan sobre nada
    # camera_events NO es direct: el LLM debe sintetizar, no volcar la lista cruda.
    assert reg.get("camera_events").direct is False
