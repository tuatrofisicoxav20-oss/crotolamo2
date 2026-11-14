"""Revisión final, unidad 15: crotolamo/mcp/bridge.py.

Contra el fake server (proceso real, sin red):

- Un server que _retire está cerrando (gracia tras el EOF, luego SIGTERM)
  ya no está en _STATE; el apagado por señal (terminate_all + os._exit) no lo
  veía y el server sobrevivía a Crotolamo. close_all (atexit) tampoco.
- Los plazos de la config (`timeout_s`, `startup_timeout_s`) aceptaban `inf`
  (TOML válido) o valores enormes que ninguna espera admite: OverflowError en
  cada llamada ("reventó", con traza) o un server que nunca arrancaba.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

import pytest

from crotolamo.core import router
from crotolamo.mcp import bridge
from crotolamo.mcp import client as mcp_client
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry

FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Ningún server ni grupo dinámico sobrevive de un test a otro."""
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)
    monkeypatch.setattr(router, "_RR_COUNTER", 0)
    yield
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)


def mk_settings(servers: dict, **mcp: Any) -> Settings:
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 5, "servers": servers}
    raw.update(mcp)
    return Settings(raw={"mcp": raw}, user="test", home=Path("/tmp"))


def registrar(servers: dict, **mcp: Any) -> tuple[Registry, list[str]]:
    registry = Registry()
    return registry, bridge.register_mcp_tools(registry, mk_settings(servers, **mcp))


def esperar(cond: Callable[[], bool], plazo: float = 5.0) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ---------------------------------------------------------------------------
# Hallazgo 1: un server a medio cierre en _retire sigue al alcance del apagado
# ---------------------------------------------------------------------------

def _retiro_en_curso(monkeypatch, gracia_s: float | None = None):
    """Conecta un server que tarda 60 s en salir tras el EOF (no lo hace
    solo), lo retira en otro hilo (como el 2º timeout o el aviso on_dead) y
    vuelve cuando ya salió de _STATE y su close() está en marcha.

    `gracia_s` alarga la gracia tras el EOF de close(): con 30 s, si el server
    muere en pocos segundos es que lo mató el apagado, no su propio cierre."""
    if gracia_s is not None:
        monkeypatch.setattr(mcp_client, "_EOF_GRACE_S", gracia_s)
    registry, names = registrar({"fake": {
        "command": FAKE_CMD, "env": {"FAKE_MCP_EOF_DELAY": "60"},
    }})
    assert names
    client = bridge._STATE["fake"].client
    proc = client._proc
    assert proc is not None
    hilo = threading.Thread(
        target=bridge.unregister_server, args=(registry, "fake"), daemon=True,
    )
    hilo.start()
    assert esperar(lambda: client._closed and bridge.connected_servers() == [])
    assert proc.poll() is None  # sigue vivo: en plena gracia tras el EOF
    return proc, hilo


def test_terminate_all_alcanza_al_server_que_retire_esta_cerrando(monkeypatch):
    """La señal de apagado llega mientras _retire espera la gracia del EOF: el
    listener sale con os._exit justo después de terminate_all(), así que el
    SIGTERM que close() iba a mandar luego no sale nunca. terminate_all() tiene
    que mandarlo él; antes no veía al cliente (fuera de _STATE y de _STARTING)."""
    proc, hilo = _retiro_en_curso(monkeypatch, gracia_s=30.0)
    try:
        bridge.terminate_all()
        assert esperar(lambda: proc.poll() is not None, plazo=5.0), (
            "terminate_all() no alcanzó al server a medio cierre: con os._exit, huérfano"
        )
        hilo.join(timeout=10)
        assert not hilo.is_alive()
        assert bridge._CLOSING == set()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        hilo.join(timeout=10)


def test_close_all_espera_al_server_que_retire_esta_cerrando(monkeypatch):
    """atexit -> close_all() con un retiro a medio cerrar en otro hilo (daemon):
    close() de ese cliente ya no hace nada (está en marcha) y volver antes deja
    que el intérprete pare ese hilo antes de su SIGTERM. close_all() espera a
    que el cierre acabe: al volver, el server ya no existe."""
    proc, hilo = _retiro_en_curso(monkeypatch)
    try:
        bridge.close_all()
        assert proc.poll() is not None, "close_all() volvió con el server aún vivo"
        assert bridge._CLOSING == set()
        hilo.join(timeout=10)
        assert not hilo.is_alive()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        hilo.join(timeout=10)


def test_retire_saca_al_cliente_de_closing_aunque_close_falle(monkeypatch):
    registry, names = registrar({"fake": {"command": FAKE_CMD}})
    assert names
    client = bridge._STATE["fake"].client
    original = client.close

    def close_que_falla() -> None:
        original()
        raise RuntimeError("fallo al cerrar")

    monkeypatch.setattr(client, "close", close_que_falla)
    with pytest.raises(RuntimeError):
        bridge.unregister_server(registry, "fake")
    assert bridge._CLOSING == set()
    assert bridge.connected_servers() == []


# ---------------------------------------------------------------------------
# Hallazgo 2: plazos `inf` o más allá de lo que admite cualquier espera
# ---------------------------------------------------------------------------

def _toml(text: str) -> dict[str, Any]:
    return tomllib.loads(text)


@pytest.mark.parametrize("valor", ["inf", "+inf", "nan", "1e10", "1e7"])
def test_plazos_fuera_de_rango_caen_al_default_con_warning(valor, caplog):
    """`inf` es TOML válido y la forma natural de escribir "sin límite"; 1e10
    pasa de threading.TIMEOUT_MAX (Lock.acquire lanza OverflowError) y 1e7
    (115 días) de lo que admite select.poll en ms (~24,8 días)."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    raw = _toml(
        f"enabled = true\ntimeout_s = {valor}\nstartup_timeout_s = {valor}\n"
        f"[servers.a]\ncommand = ['x']\ntimeout_s = {valor}\nstartup_timeout_s = {valor}\n"
    )
    cfg = bridge.load_mcp_config(raw)
    assert cfg.timeout_s == bridge.DEFAULT_TIMEOUT_S
    assert cfg.startup_timeout_s == bridge.DEFAULT_STARTUP_TIMEOUT_S
    (server,) = cfg.servers
    assert server.timeout_s == bridge.DEFAULT_TIMEOUT_S
    assert server.startup_timeout_s == bridge.DEFAULT_STARTUP_TIMEOUT_S
    assert caplog.text.count("timeout_s=") == 4


def test_plazos_grandes_pero_validos_se_respetan():
    cfg = bridge.load_mcp_config({
        "timeout_s": 86_400, "servers": {"a": {
            "command": ["x"], "timeout_s": bridge.MAX_TIMEOUT_S, "startup_timeout_s": 3600.5,
        }},
    })
    assert cfg.timeout_s == 86_400.0
    assert cfg.servers[0].timeout_s == bridge.MAX_TIMEOUT_S
    assert cfg.servers[0].startup_timeout_s == 3600.5


def test_el_tope_cabe_en_todas_las_esperas():
    assert 0 < bridge.MAX_TIMEOUT_S <= threading.TIMEOUT_MAX
    assert bridge.MAX_TIMEOUT_S * 1000 < 2**31 - 1


def test_timeout_inf_no_revienta_cada_llamada():
    registry, names = registrar(
        {"fake": {"command": FAKE_CMD}}, **_toml("timeout_s = inf"),
    )
    assert "mcp_fake_echo" in names
    out = registry.run("mcp_fake_echo", {"text": "hola"})
    assert "reventó" not in out
    assert '"text": "hola"' in out


def test_startup_timeout_inf_no_impide_conectar():
    _, names = registrar(
        {"fake": {"command": FAKE_CMD, **_toml("startup_timeout_s = inf")}},
    )
    assert "mcp_fake_echo" in names
    assert bridge.connected_servers() == ["fake"]


@pytest.mark.parametrize("timeout_s", [1e7, "tope"])
def test_argumento_mayor_que_la_tuberia_con_un_plazo_largo(timeout_s):
    """Un argumento de más de 64 KiB llena la tubería y el escritor espera con
    select.poll, que toma el plazo en ms en un int de C: con 1e7 s (bajo
    threading.TIMEOUT_MAX) lanzaba "timeout is too large". En el tope, cabe."""
    plazo = bridge.MAX_TIMEOUT_S if timeout_s == "tope" else timeout_s
    registry, names = registrar({"fake": {"command": FAKE_CMD, "timeout_s": plazo}})
    assert "mcp_fake_echo" in names
    out = registry.run("mcp_fake_echo", {"text": "x" * 300_000})
    assert "reventó" not in out
    assert out.startswith('{"text": "xxx')
