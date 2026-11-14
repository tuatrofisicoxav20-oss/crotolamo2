"""Revisión final, unidad 16: crotolamo/mcp/client.py.

Contra el fake server (proceso real, sin red):

- Un hilo de tubería (lector de stdout o drenador de stderr) que no arranca
  ("can't start new thread", el caso que el bridge y turn.py ya tratan) dejaba
  un cliente a medio arrancar: el server lanzado y un Thread sin arrancar en
  `_reader`/`_stderr_thread`. close() hacía join() de ese hilo y reventaba con
  "cannot join thread before it is started" antes de cerrar las tuberías, el
  error escapaba del except de bridge._launch y register_mcp_tools caía
  entero: ningún server registrado, tampoco los sanos. Y el lector, al ver
  morir solo al server, hacía el mismo join() en _exit_detail.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from crotolamo.core import router
from crotolamo.mcp import bridge
from crotolamo.mcp.client import StdioMCPClient
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
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 8, "servers": servers}
    raw.update(mcp)
    return Settings(raw={"mcp": raw}, user="test", home=Path("/tmp"))


def esperar(cond: Callable[[], bool], plazo: float = 5.0) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def sin_hilo_para(monkeypatch, nombre: str) -> None:
    """Thread.start() del hilo `nombre` lanza "can't start new thread" (sin
    arrancarlo, como hace CPython cuando no quedan hilos)."""
    real_start = threading.Thread.start

    def start(hilo: threading.Thread) -> None:
        if hilo.name == nombre:
            raise RuntimeError("can't start new thread")
        real_start(hilo)

    monkeypatch.setattr(threading.Thread, "start", start)


# ---------------------------------------------------------------------------
# Hilo de tubería que no arranca
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tuberia", ["stdout", "stderr"])
def test_close_de_un_cliente_a_medio_arrancar_no_lanza_y_cierra_todo(monkeypatch, tuberia):
    """start() falla al arrancar un hilo de tubería; close() no se atraganta
    con el hilo que nunca arrancó: el server termina y las tuberías se cierran."""
    sin_hilo_para(monkeypatch, f"mcp-roto-{tuberia}")
    c = StdioMCPClient("roto", FAKE_CMD)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        c.start()
    proc = c._proc
    assert proc is not None
    c.close()  # antes: RuntimeError "cannot join thread before it is started"
    assert proc.poll() is not None, "el server siguió vivo tras close()"
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed
    assert not c.alive


def test_el_lector_explica_la_muerte_aunque_el_drenador_no_arrancara(monkeypatch):
    """Sin drenador de stderr, el lector que ve morir solo al server sigue
    dando el motivo (código o señal) en vez de reventar en _exit_detail."""
    sin_hilo_para(monkeypatch, "mcp-roto-stderr")
    errores: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errores.append(args.exc_value))
    avisos: list[str] = []
    c = StdioMCPClient("roto", FAKE_CMD, on_dead=lambda _c, motivo: avisos.append(motivo))
    with pytest.raises(RuntimeError, match="can't start new thread"):
        c.start()
    try:
        assert c._proc is not None
        os.kill(c._proc.pid, signal.SIGTERM)
        assert esperar(lambda: bool(avisos)), "el lector no avisó de la muerte"
        reader = c._reader
        assert reader is not None
        reader.join(5)
        assert errores == []
        assert "SIGTERM" in c.dead_reason, c.dead_reason
        assert avisos == [c.dead_reason]
    finally:
        c.close()


@pytest.mark.parametrize("tuberia", ["stdout", "stderr"])
def test_sin_hilo_de_tuberia_en_un_server_los_demas_se_registran(monkeypatch, tuberia):
    """El server que no consigue su hilo se salta (avisado y cerrado); los
    sanos, lanzados antes y después, se registran como siempre."""
    sin_hilo_para(monkeypatch, f"mcp-roto-{tuberia}")
    servers = {n: {"command": FAKE_CMD} for n in ("a", "roto", "c")}
    names = bridge.register_mcp_tools(Registry(), mk_settings(servers))
    assert bridge.connected_servers() == ["a", "c"]
    assert {n.split("_")[1] for n in names} == {"a", "c"}
    assert bridge._STARTING == set()
