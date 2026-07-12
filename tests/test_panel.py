"""Tests headless de la lógica pura del panel (desktop/panel/*.py sin gi).

Mockean el runner de systemctl y las rutas de disco: nunca tocan el systemd
real ni requieren display.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

# desktop/ no es paquete: se importa igual que en producción (sys.path)
DESKTOP_DIR = str(Path(__file__).resolve().parent.parent / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from common import ipc  # noqa: E402
from panel import config, systemd, terminal  # noqa: E402


# --- query_state: parseo de la salida de systemctl show --------------------------

def _fake_runner(stdout: str):
    def runner(*args: str) -> subprocess.CompletedProcess:
        runner.calls.append(args)
        return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")
    runner.calls = []
    return runner


def test_query_state_parsea_show():
    out = ("ActiveState=active\n"
           "SubState=running\n"
           "UnitFileState=enabled\n"
           "ActiveEnterTimestampMonotonic=123456789\n")
    runner = _fake_runner(out)
    st = systemd.query_state(runner=runner)
    assert st == {
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "enabled",
        "ActiveEnterTimestampMonotonic": "123456789",
    }
    # una sola invocación con las propiedades pedidas
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "show"
    assert systemd.SERVICE in runner.calls[0]


def test_query_state_salida_vacia():
    assert systemd.query_state(runner=_fake_runner("")) == {}


def test_query_state_lineas_raras_se_ignoran():
    st = systemd.query_state(runner=_fake_runner("sin_igual\nA=1\n"))
    assert st == {"A": "1"}


def test_query_state_valor_con_igual():
    st = systemd.query_state(runner=_fake_runner("A=x=y\n"))
    assert st == {"A": "x=y"}


def test_run_comando_inexistente_no_lanza():
    res = systemd.run(["/no/existe/binario_falso_xyz"], timeout=1)
    assert res.returncode == 255
    assert res.stdout == ""
    assert res.stderr  # el mensaje de error queda en stderr


# --- _uptime_text ----------------------------------------------------------------

def test_uptime_sin_timestamp():
    assert systemd._uptime_text({}) == "activo"
    assert systemd._uptime_text({"ActiveEnterTimestampMonotonic": "0"}) == "activo"


def test_uptime_segundos():
    st = {"ActiveEnterTimestampMonotonic": "1000000"}  # arrancó en t=1 s
    assert systemd._uptime_text(st, now_us=43_000_000) == "activo desde hace 42s"


def test_uptime_minutos():
    st = {"ActiveEnterTimestampMonotonic": "0000001"}
    now = 1 + 5 * 60 * 1_000_000
    assert systemd._uptime_text(st, now_us=now) == "activo desde hace 5 min"


def test_uptime_horas():
    st = {"ActiveEnterTimestampMonotonic": "1"}
    now = 1 + (2 * 60 + 7) * 60 * 1_000_000  # 2 h 7 min
    assert systemd._uptime_text(st, now_us=now) == "activo desde hace 2 h 7 min"


def test_uptime_negativo_se_recorta_a_cero():
    st = {"ActiveEnterTimestampMonotonic": "9000000000"}
    assert systemd._uptime_text(st, now_us=1) == "activo desde hace 0s"


def test_uptime_timestamp_corrupto():
    st = {"ActiveEnterTimestampMonotonic": "no-numero"}
    assert systemd._uptime_text(st, now_us=1) == "activo"


# --- MODES / ARGS_BY_KEY / KEY_BY_ARGS -------------------------------------------

def test_tablas_de_modo_biyectivas():
    keys = [k for k, _, _ in config.MODES]
    args = [a for _, _, a in config.MODES]
    assert len(set(keys)) == len(keys)
    assert len(set(args)) == len(args)
    assert config.ARGS_BY_KEY == dict(zip(keys, args))
    assert config.KEY_BY_ARGS == dict(zip(args, keys))
    for k, a in config.ARGS_BY_KEY.items():
        assert config.KEY_BY_ARGS[a] == k


def test_modo_por_defecto_existe():
    # el panel usa "half" como fallback: debe seguir existiendo
    assert "half" in config.ARGS_BY_KEY
    assert config.ARGS_BY_KEY["half"] == "--no-barge-in"


# --- read/write_mode_args con tmp_path -------------------------------------------

def test_mode_args_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "listener.env")
    config.write_mode_args("--barge-in")
    assert config.read_mode_args() == "--barge-in"
    contenido = (tmp_path / "listener.env").read_text()
    assert contenido == "CROTOLAMO_LISTEN_ARGS=--barge-in\n"


def test_mode_args_default_sin_archivo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "no_existe.env")
    assert config.read_mode_args() == "--no-barge-in"


def test_mode_args_con_comillas(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text('CROTOLAMO_LISTEN_ARGS="--simple"\n')
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.read_mode_args() == "--simple"


def test_mode_args_cache_se_invalida_al_cambiar(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--barge-in")
    assert config.read_mode_args() == "--barge-in"
    # cambio externo (otro proceso): la caché por mtime debe releer
    env.write_text("CROTOLAMO_LISTEN_ARGS=--simple\n")
    assert config.read_mode_args() == "--simple"


def test_mode_args_cache_no_relee_si_no_cambio(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text("CROTOLAMO_LISTEN_ARGS=--simple\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.read_mode_args() == "--simple"

    lecturas = []
    real_read_text = Path.read_text

    def contando(self, *a, **kw):
        lecturas.append(self)
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", contando)
    assert config.read_mode_args() == "--simple"
    assert env not in lecturas  # mismo mtime → sirvió la caché


# --- read/write_listening_enabled -------------------------------------------------

def test_listening_default_true_sin_archivo(tmp_path, monkeypatch):
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", tmp_path / "hud_state.json")
    assert config.read_listening_enabled() is True


def test_listening_lee_enabled(tmp_path, monkeypatch):
    hud = tmp_path / "hud_state.json"
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", hud)
    hud.write_text('{"mode":"idle","enabled":false}', encoding="utf-8")
    assert config.read_listening_enabled() is False
    hud.write_text('{"mode":"idle","enabled":true}', encoding="utf-8")
    assert config.read_listening_enabled() is True


def test_listening_default_true_corrupto_o_sin_campo(tmp_path, monkeypatch):
    hud = tmp_path / "hud_state.json"
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", hud)
    hud.write_text("{basura", encoding="utf-8")
    assert config.read_listening_enabled() is True
    hud.write_text('{"mode":"idle"}', encoding="utf-8")
    assert config.read_listening_enabled() is True


def test_write_listening_enabled_atomico(tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    monkeypatch.setattr(ipc, "CONTROL_FILE", control)
    config.write_listening_enabled(False)
    assert json.loads(control.read_text()) == {"listening_enabled": False}
    config.write_listening_enabled(True)
    assert json.loads(control.read_text()) == {"listening_enabled": True}
    # sin temporales huérfanos tras escribir
    assert list(tmp_path.glob("*.tmp")) == []


# --- _find_terminal con PATH falso ------------------------------------------------

def _hacer_ejecutable(path: Path) -> None:
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_find_terminal_ninguno(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # dir vacío
    assert terminal._find_terminal() is None


def test_find_terminal_encuentra_foot(tmp_path, monkeypatch):
    _hacer_ejecutable(tmp_path / "foot")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert terminal._find_terminal() == ["foot"]


def test_find_terminal_prioridad(tmp_path, monkeypatch):
    # kitty va antes que xterm en la lista de candidatos
    _hacer_ejecutable(tmp_path / "xterm")
    _hacer_ejecutable(tmp_path / "kitty")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert terminal._find_terminal() == ["kitty", "-e"]


# --- higiene: la lógica pura no debe depender de gi -------------------------------

def test_modulos_puros_sin_gi():
    for mod in (config, systemd, terminal):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import gi" not in src, f"{mod.__name__} no debe importar gi"


def test_rutas_ipc_compartidas():
    # una sola fuente de verdad para las rutas del contrato
    assert ipc.HUD_STATE_FILE.name == "hud_state.json"
    assert ipc.CONTROL_FILE.name == "control.json"
    assert ipc.HUD_STATE_FILE.parent == ipc.CONTROL_FILE.parent


def test_os_replace_disponible():
    # write_listening_enabled depende de os.replace atómico
    assert callable(os.replace)
