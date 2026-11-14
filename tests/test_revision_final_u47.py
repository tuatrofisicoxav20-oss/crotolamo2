"""Revisión final, unidad 47: desktop/panel/window.py (elegir el modo de voz).

Panel REAL sobre el gi falso de test_auditoria_desktop_py (sin display):

- on_mode_changed no tenía ningún test: guarda el modo en listener.env y
  reinicia el servicio SOLO si está escuchando (nunca arranca uno parado); ni el
  eco de _apply_state (_syncing) ni un combo sin selección escriben nada.
- Si el modo no se podía guardar (disco lleno, sin permiso, la ruta rota), el
  panel reiniciaba igual: voz sorda y modelos recargando unos segundos para
  volver al MISMO modo, «Aplicando modo…», y en el siguiente tick el combo
  rebotaba al anterior sin que nada dijera por qué.
"""

from __future__ import annotations

import subprocess

from tests import test_auditoria_desktop_py as _desktop
from tests.test_auditoria_desktop_py import _ACTIVO, _APAGADO, config, systemd

# el mismo fixture: panel.window REAL sobre el gi falso, _do_async anotado
panel = _desktop.panel


def _elegir(win, key) -> None:
    """El patrón elige `key` en el combo (el gi falso no emite «changed»)."""
    win.mode.set_active_id(key)
    win.on_mode_changed(win.mode)


def _lanzados(p) -> list:
    return [args for args, _on_done in p.lanzados]


# ---------------------------------------------------------------------------
# Camino normal: guarda y reinicia solo si está escuchando.
# ---------------------------------------------------------------------------

def test_modo_con_el_servicio_activo_se_guarda_y_reinicia_una_vez(panel, tmp_path):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    _elegir(win, "simple")
    env = tmp_path / "listener.env"
    assert env.read_text(encoding="utf-8") == "CROTOLAMO_LISTEN_ARGS=--simple\n"
    assert _lanzados(panel) == [("restart", systemd.SERVICE)]
    assert win.status.get_label() == "Aplicando modo…"
    # el reinicio sale bien y el siguiente tick no devuelve el combo al anterior
    panel.lanzados[-1][1](subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    panel.reloj.avanzar(0)
    win._on_state_ready(dict(_ACTIVO))
    assert win.mode.get_active_id() == "simple"
    assert win.status.get_label() == "Escuchando"


def test_modo_con_el_servicio_parado_se_guarda_sin_arrancarlo(panel, tmp_path):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    _elegir(win, "barge")
    env = tmp_path / "listener.env"
    assert env.read_text(encoding="utf-8") == "CROTOLAMO_LISTEN_ARGS=--barge-in\n"
    assert _lanzados(panel) == []  # elegir modo nunca arranca un servicio parado
    win._on_state_ready(dict(_APAGADO))
    assert win.mode.get_active_id() == "barge"
    assert win.status.get_label() == "Apagado"


def test_modo_eco_de_la_sincronizacion_o_sin_seleccion_no_hace_nada(panel, tmp_path):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    env = tmp_path / "listener.env"
    # _apply_state mueve el combo al modo guardado con _syncing puesto: ese
    # «changed» es el eco del tick, no un clic (ni escribe ni reinicia)
    win._syncing = True
    _elegir(win, "barge")
    win._syncing = False
    _elegir(win, None)  # combo sin selección
    assert not env.exists()
    assert _lanzados(panel) == []


# ---------------------------------------------------------------------------
# Modo que no se pudo guardar: ni reinicio para nada ni rebote mudo.
# ---------------------------------------------------------------------------

def test_modo_que_no_se_pudo_guardar_no_reinicia_y_dice_por_que(panel, tmp_path,
                                                                 monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text("CROTOLAMO_LISTEN_ARGS=--simple\n", encoding="utf-8")
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    assert win.mode.get_active_id() == "simple"

    def disco_lleno(path, data):
        raise OSError(28, "No queda espacio en el dispositivo")

    monkeypatch.setattr(config, "_replace_atomically", disco_lleno)
    _elegir(win, "barge")
    assert _lanzados(panel) == []  # nada que aplicar: la voz sigue como estaba
    assert env.read_text(encoding="utf-8") == "CROTOLAMO_LISTEN_ARGS=--simple\n"
    # el combo vuelve YA al modo guardado y el subtítulo dice por qué
    assert win.mode.get_active_id() == "simple"
    assert win.status.get_label() == "Escuchando"  # no «Aplicando modo…»
    assert "no pude guardar el modo" in win.sub.get_label()
    assert "listener.env" in win.sub.get_label()
    # el siguiente tick con el mismo estado no borra el aviso...
    win._on_state_ready(dict(_ACTIVO))
    assert win.mode.get_active_id() == "simple"
    assert "no pude guardar el modo" in win.sub.get_label()
    # ...pero en cuanto el servicio cambia ya no aplica
    win._on_state_ready(dict(_APAGADO))
    assert win.sub.get_label() == "Crotolamo no está escuchando"


def test_modo_con_un_error_real_de_disco_y_luego_arreglado(panel, tmp_path, monkeypatch):
    # OSError de verdad, sin monkeypatch de la escritura: la carpeta del env
    # file es un archivo normal (NotADirectoryError, también como root)
    bloqueo = tmp_path / "no_es_dir"
    bloqueo.write_text("x", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", bloqueo / "listener.env")
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    _elegir(win, "barge")
    assert _lanzados(panel) == []
    assert win.mode.get_active_id() == "half"  # el modo que de verdad aplica
    assert "no pude guardar el modo" in win.sub.get_label()
    # el patrón lo arregla y vuelve a elegir: queda guardado y el aviso viejo
    # no sigue diciendo que no se pudo
    bloqueo.unlink()
    _elegir(win, "barge")
    assert (bloqueo / "listener.env").read_text(encoding="utf-8") == (
        "CROTOLAMO_LISTEN_ARGS=--barge-in\n")
    win._on_state_ready(dict(_APAGADO))
    assert win.mode.get_active_id() == "barge"
    assert win.sub.get_label() == "Crotolamo no está escuchando"
    assert _lanzados(panel) == []  # parado sigue parado
