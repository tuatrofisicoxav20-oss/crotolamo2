"""Revisión final, unidad 42: desktop/common/ipc.py (lectura de hud_state.json).

read_hud_state promete no lanzar nunca (corre dentro de callbacks de GLib), pero
un hud_state.json con bytes que no son UTF-8 (disco dañado, otra herramienta)
hacía saltar UnicodeDecodeError, que es ValueError y no OSError:

- HUD: el callback del poll lanzaba y GLib descarta para siempre un timer así
  (sin FileMonitor, el HUD se congelaba hasta reiniciarlo).
- Panel: _apply_state lanzaba con _syncing=True, y los switches de escucha y
  autoarranque y el combo de modo ignoraban los clics en silencio hasta que el
  loop reescribiera el archivo.

HUD y panel REALES sobre el gi falso de test_auditoria_desktop_py (sin display).
"""

from __future__ import annotations

import json

import pytest

from tests import test_auditoria_desktop_py as _desktop
from tests.test_auditoria_desktop_py import _ACTIVO, ipc

# los mismos fixtures: hud.window y panel.window REALES sobre el gi falso
hud = _desktop.hud
panel = _desktop.panel
escucha = _desktop.escucha

_NO_UTF8 = [
    pytest.param(b'{"mode":"speaking","text":"caf\xe9","enabled":false}', id="latin-1"),
    pytest.param(b'{"mode":"speaking","text":"caf\xc3', id="multibyte-truncado"),
]


@pytest.mark.parametrize("contenido", _NO_UTF8)
def test_read_hud_state_con_bytes_que_no_son_utf8_devuelve_vacio(tmp_path, contenido):
    f = tmp_path / "hud_state.json"
    f.write_bytes(contenido)
    assert ipc.read_hud_state(f) == {}


@pytest.mark.parametrize("contenido", _NO_UTF8)
def test_read_hud_state_ruta_por_defecto_con_bytes_que_no_son_utf8(
        tmp_path, monkeypatch, contenido):
    f = tmp_path / "hud_state.json"
    f.write_bytes(contenido)
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", f)
    assert ipc.read_hud_state() == {}


def test_read_hud_state_utf8_valido_sigue_leyendose(tmp_path):
    f = tmp_path / "hud_state.json"
    f.write_bytes('{"mode":"speaking","text":"café"}'.encode())
    assert ipc.read_hud_state(f) == {"mode": "speaking", "text": "café"}


@pytest.mark.parametrize("contenido", _NO_UTF8)
def test_hud_poll_sobrevive_a_un_archivo_que_no_es_utf8(hud, contenido):
    win = hud.ventana()
    hud.publicar("speaking", "hola")
    assert win._current_mode == "speaking"
    hud.window.HUD_STATE_FILE.write_bytes(contenido)
    # GLib descarta para siempre un timer cuyo callback lanza: el poll debe
    # seguir vivo y tratar el archivo ilegible como reposo
    assert win._poll_tick() is True
    assert win._current_mode == "idle"
    # el evento del FileMonitor tampoco lanza
    assert win._read_state_once() is False
    # y cuando el loop vuelve a publicar, el HUD lo refleja
    hud.publicar("thinking", "procesando")
    assert win._current_mode == "thinking"


@pytest.mark.parametrize("contenido", _NO_UTF8)
def test_panel_con_hud_state_que_no_es_utf8_no_se_queda_sordo_a_los_clics(
        panel, escucha, tmp_path, contenido):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is True
    (tmp_path / "hud_state.json").write_bytes(contenido)
    win._on_state_ready(dict(_ACTIVO))
    # el tick termina: _syncing no se queda en True y la etiqueta se pinta
    assert win._syncing is False
    assert win.status.get_label() == "Escuchando"
    # default-SEGURO del contrato: ilegible => escucha activa
    assert win.listen_sw.get_active() is True
    # el clic del switch de escucha llega a control.json
    escucha.clic_escucha(False)
    assert json.loads(escucha.control.read_text()) == {"listening_enabled": False}
