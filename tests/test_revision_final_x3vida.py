"""Revisión final, unidad x3 (ciclo de vida): cómo salen los servers MCP.

Contra procesos reales (el fake server y un `sh -c` de wrapper), sin red:

- El apagado por SEÑAL del listener (terminate_all + os._exit) mandaba SIGTERM
  y nada más: un server que lo ignora (o el nieto de un wrapper) sobrevivía a
  Crotolamo en su propia sesión. La salida normal (close) sí escala a SIGKILL.
  `kill_survivors` es esa segunda mitad, para llamarla antes del os._exit.
- close_all() cerraba los servers uno tras otro: la salida costaba N veces la
  gracia del EOF (1 s por cada server que no sale solo) en vez de una.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
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

RAIZ = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]
# Un server que NO sale al cerrarle stdin (se queda "limpiando" 10 min): su
# close() paga la gracia del EOF entera (1 s) y acaba en SIGTERM.
SORDO_AL_EOF = {"command": FAKE_CMD, "env": {"FAKE_MCP_EOF_DELAY": "600"}}

# El server real detrás de un wrapper: habla MCP (es el fake), ignora el
# SIGTERM y deja su pid en argv[1] para que el test lo vigile.
_TERCO = """\
import os, runpy, signal, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
fake = sys.argv[2]
sys.argv = [fake]
runpy.run_path(fake, run_name="__main__")
"""
# Lo mismo sin hablar el protocolo: se queda en el handshake para siempre.
_TERCO_MUDO = """\
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


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


def vivo(pid: int) -> bool:
    """True si el proceso existe y no es un zombi (os.kill(pid, 0) da True a un zombi)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


def _pid(pidfile: Path) -> int:
    assert esperar(pidfile.exists), "el server real no llegó a arrancar"
    return int(pidfile.read_text())


def _rematar(*pids: int) -> None:
    for pid in pids:
        if vivo(pid):
            os.kill(pid, signal.SIGKILL)


def _terco(tmp_path: Path, nombre: str = "terco", guion: str = _TERCO) -> tuple[dict, Path]:
    """Config de un server tras un wrapper `sh -c` (como npx/uvx): el wrapper
    muere con el SIGTERM; el nieto, que es quien retiene las tuberías, no."""
    script = tmp_path / f"{nombre}.py"
    script.write_text(guion, encoding="utf-8")
    pidfile = tmp_path / f"{nombre}.pid"
    # '; true': sh no hace exec y queda de launcher (que SÍ muere con SIGTERM).
    orden = f"{sys.executable} {script} {pidfile} {FAKE}; true"
    return {"command": ["sh", "-c", orden], "env": {"FAKE_MCP_EOF_DELAY": "600"}}, pidfile


# ---------------------------------------------------------------------------
# El apagado por señal escala a SIGKILL (bridge.kill_survivors)
# ---------------------------------------------------------------------------

def test_kill_survivors_mata_al_server_que_ignora_el_sigterm_del_apagado(tmp_path):
    """Lo que hace el listener al recibir una señal: terminate_all() y, tras
    vaciar la Memoria, os._exit. El nieto que ignora el SIGTERM (y el EOF)
    seguía vivo a los 3 s; con kill_survivors() antes del os._exit, no."""
    server, pidfile = _terco(tmp_path)
    _, names = registrar({"terco": server})
    assert "mcp_terco_echo" in names
    pid = _pid(pidfile)
    try:
        t0 = time.monotonic()
        bridge.terminate_all()
        assert time.monotonic() - t0 < 1.0  # terminate_all sigue sin esperar a nadie
        time.sleep(0.3)
        assert vivo(pid)  # el defecto: SIGTERM ignorado, y nadie manda nada más
        t0 = time.monotonic()
        bridge.kill_survivors(grace_s=0.6)
        assert time.monotonic() - t0 < 3.0
        assert esperar(lambda: not vivo(pid), plazo=3.0), "el server sobrevivió al apagado"
    finally:
        _rematar(pid)


def test_kill_survivors_no_espera_ni_mata_si_todos_murieron_con_el_sigterm():
    """Con servers bien hechos no cuesta nada: vuelve en cuanto han salido (ms),
    no agota la gracia, y a ninguno le llega el SIGKILL."""
    _, names = registrar({n: {"command": FAKE_CMD} for n in ("a", "b", "c")})
    assert names
    procs = [bridge._STATE[n].client._proc for n in ("a", "b", "c")]
    bridge.terminate_all()
    t0 = time.monotonic()
    bridge.kill_survivors(grace_s=6.0)
    assert time.monotonic() - t0 < 2.5
    for proc in procs:
        assert proc is not None and proc.poll() is not None  # ya recogido, sin esperar más
        assert proc.returncode != -signal.SIGKILL


def test_la_gracia_de_kill_survivors_cuenta_desde_el_sigterm(tmp_path):
    """El listener vacía la Memoria (hasta 2 s) entre el SIGTERM y esta
    llamada: ese rato ya es gracia. Contarla desde aquí sumaba las dos esperas."""
    server, pidfile = _terco(tmp_path)
    registrar({"terco": server})
    pid = _pid(pidfile)
    try:
        bridge.terminate_all()
        time.sleep(1.0)  # la Memoria vaciándose
        t0 = time.monotonic()
        bridge.kill_survivors(grace_s=0.9)
        assert time.monotonic() - t0 < 0.6, "esperó la gracia entera otra vez"
        assert esperar(lambda: not vivo(pid), plazo=3.0)
    finally:
        _rematar(pid)


def test_kill_survivors_sin_apagado_previo_no_toca_a_nadie():
    """Solo alcanza a los que terminate_all() señaló: llamado sin él (o dos
    veces) no puede matar a un server conectado y sano."""
    registry, names = registrar({"fake": {"command": FAKE_CMD}})
    assert "mcp_fake_echo" in names
    bridge.kill_survivors(grace_s=0.0)
    time.sleep(0.2)
    assert bridge._STATE["fake"].client.alive
    assert '"text": "sigo"' in registry.run("mcp_fake_echo", {"text": "sigo"})


def test_kill_survivors_alcanza_al_server_que_la_senal_pillo_en_el_handshake(tmp_path):
    """Un wrapper (npx descargando) cuyo nieto ignora el SIGTERM, señalado a
    mitad del handshake: su hilo lo da por caído y lo saca de _STARTING al
    momento. Quien escala no puede buscarlo ahí: lo apuntó terminate_all()."""
    server, pidfile = _terco(tmp_path, "mudo", _TERCO_MUDO)
    out: list = []
    hilo = threading.Thread(
        target=lambda: out.append(bridge.register_mcp_tools(
            Registry(), mk_settings({"mudo": server}, startup_timeout_s=20),
        )),
        daemon=True,
    )
    hilo.start()
    pid = _pid(pidfile)
    try:
        assert esperar(lambda: len(bridge._STARTING) == 1)
        bridge.terminate_all()
        hilo.join(timeout=5)
        assert not hilo.is_alive() and out == [[]]
        assert bridge._STARTING == set() and vivo(pid)
        bridge.kill_survivors(grace_s=0.2)
        assert esperar(lambda: not vivo(pid), plazo=3.0), "el server sobrevivió al apagado"
    finally:
        _rematar(pid)


def test_kill_survivors_nunca_lanza_y_sigue_con_los_demas(tmp_path, monkeypatch):
    """Va pegado al os._exit del manejador de señal: un fallo con un server no
    puede propagarse ni dejar sin matar a los que vienen detrás."""
    server, pidfile = _terco(tmp_path)
    registrar({"a": {"command": FAKE_CMD}, "terco": server})
    pid = _pid(pidfile)
    real = StdioMCPClient.kill_if_survives

    def falla_con_a(client: StdioMCPClient, deadline: float) -> bool:
        if client.name == "a":
            raise RuntimeError("fallo al escalar")
        return real(client, deadline)

    monkeypatch.setattr(StdioMCPClient, "kill_if_survives", falla_con_a)
    try:
        bridge.terminate_all()
        bridge.kill_survivors(grace_s=0.2)
        assert esperar(lambda: not vivo(pid), plazo=3.0)
    finally:
        _rematar(pid)


# ---------------------------------------------------------------------------
# close_all() cierra los servers A LA VEZ
# ---------------------------------------------------------------------------

def test_close_all_cierra_los_servers_a_la_vez_no_uno_tras_otro():
    """Tres servers que no salen al ver el EOF: cada close() espera su gracia
    (1 s) antes del SIGTERM. En serie la salida ("salir", Ctrl-C en el prompt,
    el reinicio del servicio) tardaba 3 s; a la vez, lo que el más lento."""
    registry, names = registrar({n: dict(SORDO_AL_EOF) for n in ("a", "b", "c")})
    assert names and bridge.connected_servers() == ["a", "b", "c"]
    procs = [bridge._STATE[n].client._proc for n in ("a", "b", "c")]
    t0 = time.monotonic()
    bridge.close_all()
    transcurrido = time.monotonic() - t0
    assert transcurrido < 2.5, f"close_all tardó {transcurrido:.2f} s: cierra en serie"
    assert all(p is not None and p.poll() is not None for p in procs)
    assert bridge.connected_servers() == [] and bridge._CLOSING == set()
    assert not [n for n in registry.names() if n.startswith("mcp_")]
    assert not router.dynamic_groups()


def test_close_all_cierra_a_la_vez_tambien_a_los_que_siguen_en_el_handshake():
    mudo = {"command": [sys.executable, "-c", "import time; time.sleep(60)"]}
    out: list = []
    hilo = threading.Thread(
        target=lambda: out.append(bridge.register_mcp_tools(
            Registry(), mk_settings({n: dict(mudo) for n in ("a", "b", "c")},
                                    startup_timeout_s=20),
        )),
        daemon=True,
    )
    hilo.start()
    assert esperar(lambda: len(bridge._STARTING) == 3 and not bridge._SPAWNING)
    procs = [c._proc for c in bridge._STARTING]
    t0 = time.monotonic()
    bridge.close_all()
    transcurrido = time.monotonic() - t0
    assert transcurrido < 2.5, f"close_all tardó {transcurrido:.2f} s: cierra en serie"
    assert all(p is not None and p.poll() is not None for p in procs)
    hilo.join(timeout=5)
    assert not hilo.is_alive() and out == [[]]
    assert bridge._STARTING == set()


def test_close_all_sin_hilos_cierra_en_serie(monkeypatch):
    """Sin hilos libres ("can't start new thread"; o un intérprete que ya no
    deja crearlos dentro de atexit) se cierra aquí, en serie como antes: ni
    lanza ni deja un server vivo."""
    real_start = threading.Thread.start
    vistos: list[str] = []

    def start(hilo: threading.Thread) -> None:
        if hilo.name.endswith("-cierre"):
            vistos.append(hilo.name)
            raise RuntimeError("can't start new thread")
        return real_start(hilo)

    registrar({n: {"command": FAKE_CMD} for n in ("a", "b")})
    procs = [bridge._STATE[n].client._proc for n in ("a", "b")]
    monkeypatch.setattr(threading.Thread, "start", start)
    bridge.close_all()
    assert vistos == ["mcp-a-cierre", "mcp-b-cierre"]
    assert all(p is not None and p.poll() is not None for p in procs)
    assert bridge.connected_servers() == []


def test_un_cierre_que_falla_no_deja_a_los_demas_sin_cerrar(monkeypatch, caplog):
    """En serie, el fallo al cerrar el primero cortaba el bucle: los demás
    seguían conectados y, en atexit, vivos al salir. Queda avisado y se sigue."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    registrar({n: {"command": FAKE_CMD} for n in ("a", "b", "c")})
    clients = {n: bridge._STATE[n].client for n in ("a", "b", "c")}
    cerrar_a = clients["a"].close

    def close_que_falla() -> None:
        cerrar_a()
        raise RuntimeError("fallo al cerrar")

    monkeypatch.setattr(clients["a"], "close", close_que_falla)
    bridge.close_all()
    assert bridge.connected_servers() == []
    for name, client in clients.items():
        assert client._proc is not None and client._proc.poll() is not None, name
    assert "MCP 'a': no pude cerrarlo (fallo al cerrar)" in caplog.text


_SALIDA_POR_ATEXIT = r'''
import atexit, sys, time
from pathlib import Path

from crotolamo.mcp import bridge
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry

marca = {}
# atexit corre al revés: este, apuntado ANTES que el close_all del bridge, corre
# después de él; el de abajo, apuntado después, justo antes.
atexit.register(lambda: print("cierre:", time.monotonic() - marca["t0"], flush=True))
servers = {n: {"command": [sys.executable, sys.argv[1]], "env": {"FAKE_MCP_EOF_DELAY": "600"}}
           for n in ("a", "b", "c")}
raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 10, "servers": servers}
names = bridge.register_mcp_tools(
    Registry(), Settings(raw={"mcp": raw}, user="test", home=Path("/tmp")))
assert bridge.connected_servers() == ["a", "b", "c"], names
print("pids:", *(bridge._STATE[n].client._proc.pid for n in ("a", "b", "c")), flush=True)
atexit.register(lambda: marca.__setitem__("t0", time.monotonic()))
'''


def test_la_salida_normal_cierra_los_servers_a_la_vez_dentro_de_atexit():
    """El camino de producción: `salir` en el shell (o el listener que se
    rinde) -> atexit -> close_all. Los hilos del cierre se crean DENTRO de
    atexit; si el intérprete no lo permitiera, el cierre en serie lo cubre."""
    env = {**os.environ, "PYTHONPATH": str(RAIZ)}
    proc = subprocess.run([sys.executable, "-c", _SALIDA_POR_ATEXIT, str(FAKE)], cwd=RAIZ,
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lineas = dict(linea.split(":", 1) for linea in proc.stdout.splitlines() if ":" in linea)
    pids = [int(p) for p in lineas["pids"].split()]
    try:
        assert len(pids) == 3
        assert not any(vivo(p) for p in pids), "quedó un server vivo tras la salida"
        cierre = float(lineas["cierre"])
        assert cierre < 2.5, f"el cierre en atexit tardó {cierre:.2f} s: en serie"
    finally:
        _rematar(*pids)
