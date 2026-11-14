"""Revisión final, unidad 22: crotolamo/tools/cameras.py.

1. Un nombre de cámara desconocido daba un falso "todo en orden". El LLM no ve
   la tabla [cameras.names]: adivina ("puerta", "la entrada", "Front_Door"),
   _resolve_camera pasa tal cual lo que no es alias, Frigate filtra por ese
   nombre y devuelve [] y la tool contestaba "Ni un alma, patrón: cero eventos
   en la cámara 'puerta'" aunque front_door tuviera personas en la ventana. En
   una tool de seguridad. Lo mismo camera_snapshot ("no tiene ningún evento
   registrado"). Ahora, SOLO cuando el filtro por cámara sale vacío, se pide a
   Frigate su lista de cámaras (/api/config) y un nombre que no está ahí se
   contesta como desconocido, con las que sí hay. Si Frigate no da la lista,
   se mira la tabla de alias y lo que no está ahí sale con la duda dicha.

2. camera_snapshot no decía CUÁNDO fue el "último evento" (la consulta no
   lleva ventana: puede tener días). Ahora lleva la hora, y la fecha si no es
   de hoy, con el mismo formato que camera_events.
"""

from __future__ import annotations

import time
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import _frigate, cameras, default_registry

AHORA = time.time()


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


class _FrigateFalso:
    """Frigate de mentiras que filtra como el de verdad.

    /api/events: camera != "all" -> Event.camera << camera.split(",") (exacto,
    distingue mayúsculas, sin strip). /api/config: {"cameras": {nombre: ...}},
    o el fallo/respuesta que se le diga.
    """

    def __init__(self, eventos, camaras=("front_door", "patio"), config=None):
        self.eventos = eventos
        self.config = config if config is not None else {
            "cameras": {nombre: {"enabled": True} for nombre in camaras}}
        self.urls: list[str] = []

    def __call__(self, url, timeout=6.0):
        self.urls.append(url)
        partes = urlsplit(url)
        if partes.path == "/api/config":
            if isinstance(self.config, Exception):
                raise self.config
            return self.config
        assert partes.path == "/api/events", url
        q = parse_qs(partes.query)
        cam = q.get("camera", ["all"])[0]
        limite = int(q.get("limit", ["100"])[0])
        filas = [e for e in self.eventos if cam == "all" or e["camera"] in cam.split(",")]
        return filas[:limite]

    def pidio_config(self) -> bool:
        return any(urlsplit(u).path == "/api/config" for u in self.urls)


def _evento(camara="front_door", hace=600.0, **extra):
    inicio = AHORA - hace
    evento = {"id": f"{inicio:.3f}-abc", "camera": camara, "label": "person",
              "start_time": inicio, "end_time": inicio + 9, "has_snapshot": True}
    evento.update(extra)
    return evento


def _instala(monkeypatch, frigate):
    monkeypatch.setattr(_frigate, "_http_json", frigate)
    return frigate


# --- 1. nombre desconocido: nada de falso "ni un alma" ---------------------------


@pytest.mark.parametrize("nombre", ["puerta", "la entrada", "Front_Door", "todas las cámaras"])
def test_eventos_camara_desconocida_no_es_un_todo_en_orden(cams_cfg, monkeypatch, nombre):
    """front_door SÍ tiene una persona en la ventana: el nombre que adivinó el
    LLM no existe en Frigate, y eso no se puede contestar como 'cero eventos'."""
    frigate = _instala(monkeypatch, _FrigateFalso([_evento()]))
    out = default_registry().run("camera_events", {"camera": nombre, "hours": 6})
    assert "Ni un alma" not in out
    assert "cero eventos" not in out
    assert nombre in out
    # Las que hay de verdad, con su alias, para que el LLM reintente bien.
    assert "entrada (front_door)" in out
    assert "patio" in out
    assert "patrón" in out
    assert frigate.pidio_config()


@pytest.mark.parametrize("nombre", ["entrada", "front_door", "patio"])
def test_eventos_camara_real_sin_eventos_sigue_siendo_ni_un_alma(cams_cfg, monkeypatch, nombre):
    """Alias, nombre real con alias y nombre real SIN alias (el paso tal cual
    es a propósito): si Frigate la tiene y no hay eventos, el 'cero eventos'
    es verdad y se dice igual que antes."""
    _instala(monkeypatch, _FrigateFalso([]))
    out = cameras.camera_events(nombre, hours=6)
    assert out.startswith("Ni un alma, patrón: cero eventos")
    assert f"'{nombre}'" in out


def test_eventos_varias_camaras_reales_separadas_por_coma(cams_cfg, monkeypatch):
    """Frigate acepta 'a,b' en el filtro: si las dos existen, cero es cero."""
    _instala(monkeypatch, _FrigateFalso([]))
    out = cameras.camera_events("front_door,patio", hours=6)
    assert out.startswith("Ni un alma, patrón: cero eventos")


def test_eventos_con_resultados_no_paga_la_lista_de_camaras(cams_cfg, monkeypatch):
    """La comprobación vive solo en el camino del resultado vacío."""
    frigate = _instala(monkeypatch, _FrigateFalso([_evento()]))
    out = cameras.camera_events("entrada", hours=6)
    assert out.startswith("Eventos de la cámara 'entrada'")
    assert not frigate.pidio_config()
    assert len(frigate.urls) == 1


def test_eventos_sin_filtro_vacio_no_pide_la_lista(cams_cfg, monkeypatch):
    frigate = _instala(monkeypatch, _FrigateFalso([]))
    out = cameras.camera_events("todas", hours=6)
    assert out.startswith("Ni un alma, patrón: cero eventos en las cámaras")
    assert not frigate.pidio_config()


def test_eventos_alias_mal_configurado_no_es_un_todo_en_orden(cams_cfg, monkeypatch):
    """La toml de fábrica trae `entrada = "front_door"  # RELLENA`: si el patrón
    no lo cambió y su Frigate llama distinto a la cámara, 'entrada' daba 'ni un
    alma' para siempre."""
    _instala(monkeypatch, _FrigateFalso([_evento("puerta_calle")],
                                         camaras=("puerta_calle",)))
    out = cameras.camera_events("entrada", hours=6)
    assert "Ni un alma" not in out
    assert "front_door" in out      # a qué nombre apunta el alias
    assert "puerta_calle" in out    # la que Frigate sí tiene


@pytest.mark.parametrize("fallo", [
    urllib.error.HTTPError("http://frigate.test:5000/api/config", 404, "nf", {}, None),
    urllib.error.URLError("connection refused"),
])
def test_eventos_sin_lista_de_frigate_y_nombre_fuera_de_la_tabla(cams_cfg, monkeypatch, fallo):
    """Si Frigate no da su lista no se puede saber si 'puerta' es una cámara
    sin alias o un invento: se dice la duda, nunca el 'ni un alma' a secas."""
    _instala(monkeypatch, _FrigateFalso([_evento()], config=fallo))
    out = cameras.camera_events("puerta", hours=6)
    assert "Ni un alma" not in out
    assert "puerta" in out
    assert "entrada (front_door)" in out
    assert "patrón" in out


def test_eventos_sin_lista_de_frigate_y_alias_conocido(cams_cfg, monkeypatch):
    """Sin lista de Frigate, un alias de la tabla se da por bueno: cero es cero."""
    _instala(monkeypatch, _FrigateFalso([], config={"message": "raro"}))
    out = cameras.camera_events("entrada", hours=6)
    assert out.startswith("Ni un alma, patrón: cero eventos")


def test_eventos_sin_lista_y_sin_tabla_de_alias(cams_cfg, monkeypatch):
    """names vacío (la toml lo permite) y Frigate sin lista: la duda, sin
    lista de conocidas que no existe."""
    cams_cfg.raw["cameras"]["names"] = {}
    _instala(monkeypatch, _FrigateFalso([_evento()], config=urllib.error.URLError("x")))
    out = cameras.camera_events("puerta", hours=6)
    assert "Ni un alma" not in out
    assert "puerta" in out
    assert "patrón" in out


def test_snapshot_camara_desconocida(cams_cfg, monkeypatch):
    frigate = _instala(monkeypatch, _FrigateFalso([_evento()]))
    out = default_registry().run("camera_snapshot", {"camera": "puerta"})
    assert "no tiene ningún evento" not in out
    assert "puerta" in out
    assert "entrada (front_door)" in out
    assert frigate.pidio_config()


def test_snapshot_camara_real_sin_eventos_sigue_igual(cams_cfg, monkeypatch):
    _instala(monkeypatch, _FrigateFalso([]))
    out = cameras.camera_snapshot("patio")
    assert "La cámara 'patio' no tiene ningún evento registrado" in out


def test_snapshot_con_evento_no_pide_la_lista(cams_cfg, monkeypatch):
    frigate = _instala(monkeypatch, _FrigateFalso([_evento()]))
    cameras.camera_snapshot("entrada")
    assert not frigate.pidio_config()
    assert len(frigate.urls) == 1


# --- 2. camera_snapshot dice cuándo fue el último evento --------------------------


def test_snapshot_evento_viejo_lleva_la_fecha(cams_cfg, monkeypatch):
    viejo = _evento(hace=3 * 86400)
    _instala(monkeypatch, _FrigateFalso([viejo]))
    out = cameras.camera_snapshot("entrada")
    local = time.localtime(viejo["start_time"])
    assert time.strftime("el %Y-%m-%d a las %H:%M", local) in out
    assert f"/api/events/{viejo['id']}/snapshot.jpg" in out


def test_snapshot_evento_de_hoy_lleva_la_hora_sin_fecha(cams_cfg, monkeypatch):
    # A medianoche "hace 60 s" puede ser ayer: se usa el instante actual.
    ahora = time.time()
    reciente = _evento(hace=AHORA - ahora)  # start_time = ahora
    _instala(monkeypatch, _FrigateFalso([reciente]))
    out = cameras.camera_snapshot("entrada")
    local = time.localtime(reciente["start_time"])
    assert time.strftime("a las %H:%M", local) in out
    assert time.strftime("el %Y-%m-%d", local) not in out


def test_snapshot_sin_foto_tambien_dice_cuando(cams_cfg, monkeypatch):
    viejo = _evento(hace=2 * 86400, has_snapshot=False)
    _instala(monkeypatch, _FrigateFalso([viejo]))
    out = cameras.camera_snapshot("entrada")
    assert "snapshot.jpg" not in out
    assert time.strftime("el %Y-%m-%d a las %H:%M", time.localtime(viejo["start_time"])) in out


def test_snapshot_evento_sin_hora_no_inventa(cams_cfg, monkeypatch):
    sin_hora = {"id": "e1", "camera": "front_door", "label": "person"}
    _instala(monkeypatch, _FrigateFalso([sin_hora]))
    out = cameras.camera_snapshot("entrada")
    assert "/api/events/e1/snapshot.jpg" in out
    assert "??" not in out
