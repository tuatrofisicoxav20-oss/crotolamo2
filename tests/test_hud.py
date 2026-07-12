"""Tests headless de la lógica pura del HUD (desktop/hud/state.py + common/ipc.py).

No requieren gi ni display: los módulos de desktop/ están partidos para que la
lógica del contrato hud_state.json sea importable sin GTK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# desktop/ no es paquete: se importa igual que en producción (sys.path)
DESKTOP_DIR = str(Path(__file__).resolve().parent.parent / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from common import ipc  # noqa: E402
from hud import state  # noqa: E402


# --- parse_hud_state / extract_mode / extract_text -----------------------------

@pytest.mark.parametrize(
    ("desc", "raw", "exp_mode", "exp_text"),
    [
        ("JSON válido idle",
         '{"mode":"idle","turn_id":0,"text":"","ts":0.0,"pid":0}', "idle", ""),
        ("JSON válido listening",
         '{"mode":"listening","turn_id":1,"text":"qué tiempo hace","ts":1.0,"pid":123}',
         "listening", "qué tiempo hace"),
        ("JSON válido thinking",
         '{"mode":"thinking","turn_id":1,"text":"procesando","ts":2.0,"pid":123}',
         "thinking", "procesando"),
        ("JSON válido speaking",
         '{"mode":"speaking","turn_id":1,"text":"respuesta aquí","ts":3.0,"pid":123}',
         "speaking", "respuesta aquí"),
        ("String vacío (archivo vacío)", "", "idle", ""),
        ("JSON malformado", '{"mode": INVALID}', "idle", ""),
        ("Modo desconocido", '{"mode":"singing","text":"la la la"}', "idle", "la la la"),
        ("Campo text no es str", '{"mode":"listening","text":42}', "listening", ""),
        ("Sin campo mode", '{"text":"hola"}', "idle", "hola"),
        ("text con espacios", '{"mode":"speaking","text":"  hola  "}', "speaking", "hola"),
        ("JSON es lista, no dict", "[1,2,3]", "idle", ""),
    ],
)
def test_contrato_hud_state(desc, raw, exp_mode, exp_text):
    data = state.parse_hud_state(raw)
    assert state.extract_mode(data) == exp_mode, desc
    assert state.extract_text(data) == exp_text, desc


def test_parse_hud_state_dict_valido_se_devuelve_entero():
    data = state.parse_hud_state('{"mode":"idle","extra":1}')
    assert data == {"mode": "idle", "extra": 1}


def test_extract_mode_default_sin_datos():
    assert state.extract_mode({}) == "idle"


def test_extract_text_no_str():
    assert state.extract_text({"text": None}) == ""
    assert state.extract_text({"text": ["a"]}) == ""


def test_valid_modes_completos():
    assert state.VALID_MODES == {"idle", "listening", "thinking", "speaking"}


# --- read_hud_file / read_hud_state con archivos reales -------------------------

def test_read_hud_file_valido(tmp_path):
    f = tmp_path / "hud_state.json"
    f.write_text('{"mode":"speaking","text":"hola"}', encoding="utf-8")
    assert state.read_hud_file(f) == {"mode": "speaking", "text": "hola"}


def test_read_hud_file_inexistente(tmp_path):
    assert state.read_hud_file(tmp_path / "no_existe.json") == {}


def test_read_hud_file_corrupto(tmp_path):
    f = tmp_path / "hud_state.json"
    f.write_text("{esto no es json", encoding="utf-8")
    assert state.read_hud_file(f) == {}


def test_read_hud_file_no_dict(tmp_path):
    f = tmp_path / "hud_state.json"
    f.write_text("[1,2,3]", encoding="utf-8")
    assert state.read_hud_file(f) == {}


def test_read_hud_state_usa_ruta_por_defecto(tmp_path, monkeypatch):
    f = tmp_path / "hud_state.json"
    f.write_text('{"mode":"listening"}', encoding="utf-8")
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", f)
    assert ipc.read_hud_state() == {"mode": "listening"}


def test_read_hud_state_ruta_por_defecto_inexistente(tmp_path, monkeypatch):
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", tmp_path / "nada.json")
    assert ipc.read_hud_state() == {}


# --- higiene: la lógica pura no debe depender de gi ------------------------------

def test_modulos_puros_sin_gi():
    import common.ipc
    import hud.state
    import hud.style

    for mod in (hud.state, hud.style, common.ipc):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import gi" not in src, f"{mod.__name__} no debe importar gi"


def test_style_tablas_coherentes():
    from hud import style

    assert set(style.MODE_LABELS) == state.VALID_MODES
    assert set(style.MODE_ICONS) == state.VALID_MODES
    assert "hud-card" in style.CSS
