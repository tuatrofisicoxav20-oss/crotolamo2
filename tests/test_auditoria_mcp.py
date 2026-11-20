"""Auditoría del grupo MCP (bridge + cliente stdio).

Cada test fija un hallazgo verificado contra el fake server (proceso real, sin
red): política de confirmación ante un typo, keywords cortas a media palabra,
muerte del transporte sin llamada pendiente, servers a mitad del handshake
que el apagado no veía, el presupuesto de arranque de un tools/list paginado,
el cierre educado (EOF antes del SIGTERM) que escala a SIGKILL si el server
real sobrevive a su wrapper, la cancelación de una petición expirada, y que
un server que muere solo deje en el log su código de salida y su stderr.
"""

from __future__ import annotations

import json
import logging
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
from crotolamo.mcp.client import MCPError, MCPTimeout, MCPTransportError, StdioMCPClient
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


def vivo(pid: int) -> bool:
    """True si el proceso existe y no es un zombi (os.kill(pid, 0) da True a un zombi)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


# ---------------------------------------------------------------------------
# mcp#6: un `confirm` inválido no puede bajar la confirmación (fail-closed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typo", ["allways", True, "Always!", 3])
def test_confirm_invalido_por_server_cae_en_always_no_en_un_global_mas_laxo(typo, caplog):
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    cfg = bridge.load_mcp_config(mk_settings(
        {"dudoso": {"command": FAKE_CMD, "confirm": typo}}, confirm="never",
    ))
    assert cfg.servers[0].confirm == "always"
    assert "confirm=" in caplog.text  # el WARNING se mantiene


def test_confirm_global_invalido_cae_en_always():
    cfg = bridge.load_mcp_config(mk_settings({"s": {"command": FAKE_CMD}}, confirm="nunca"))
    assert cfg.confirm == "always"
    assert cfg.servers[0].confirm == "always"  # el server sin `confirm` lo hereda


def test_typo_en_confirm_no_deja_correr_sin_preguntar_una_tool_destructiva():
    """El escenario del hallazgo: global "never" + un server dudoso con typo.
    Antes heredaba "never" y `borrar_mentiroso` corría sin preguntar."""
    registry, names = registrar(
        {"fake": {"command": FAKE_CMD, "confirm": "allways"}}, confirm="never",
    )
    assert names
    assert all(registry.get(n).safe is False for n in names)  # type: ignore[union-attr]


def test_confirm_valido_o_ausente_no_cambia():
    cfg = bridge.load_mcp_config(mk_settings({
        "a": {"command": FAKE_CMD, "confirm": " Never "},
        "b": {"command": FAKE_CMD},
    }, confirm="destructive"))
    assert {s.name: s.confirm for s in cfg.servers} == {"a": "never", "b": "destructive"}
    assert bridge.load_mcp_config(mk_settings({})).confirm == "destructive"


# ---------------------------------------------------------------------------
# mcp#13: el nombre corto de un server ya no matchea A MEDIA palabra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, frase", [
    ("git", "pon musica digital"),
    ("time", "me siento sentimental hoy"),
    ("fs", "que es un offset"),
    ("db", "pon el feedback"),
])
def test_nombre_corto_de_server_no_enruta_dentro_de_otra_palabra(name, frase):
    server = bridge.MCPServerConfig(name=name, command=["x"], prefix=name)
    kws = bridge.default_keywords(server, [{"name": "get_x", "description": ""}])
    router.register_group(f"mcp:{name}", [f"mcp_{name}_get_x"], kws)
    # (Los grupos estáticos pueden matchear la frase, p.ej. "pon musica": da igual.)
    assert f"mcp_{name}_get_x" not in router.select_tool_names(frase)
    # Y nombrarlo sigue enrutando: es lo que de verdad sirve del default.
    assert f"mcp_{name}_get_x" in router.select_tool_names(f"usa {name} para eso")


# ---------------------------------------------------------------------------
# mcp#11: el transporte que muere SOLO retira las tools sin esperar una llamada
# ---------------------------------------------------------------------------

def test_server_que_muere_ocioso_retira_sus_tools_sin_esperar_una_llamada():
    registry, names = registrar({"fake": {"command": FAKE_CMD}})
    assert names
    proc = bridge._STATE["fake"].client._proc
    assert proc is not None
    os.killpg(proc.pid, signal.SIGKILL)
    assert esperar(lambda: bridge.connected_servers() == [] and registry.names() == [])
    assert "mcp:fake" not in router.dynamic_groups()


def test_basura_en_stdout_con_el_server_ocioso_lo_retira_y_para_el_proceso(caplog):
    """Tras una línea no-JSON el transporte es irrecuperable, pero el proceso
    seguía VIVO, sin lector y sin que nadie lo señalara, hasta la próxima
    llamada del modelo (o el apagado)."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    registry, _ = registrar({"fake": {"command": FAKE_CMD}})
    proc = bridge._STATE["fake"].client._proc
    assert proc is not None
    os.kill(proc.pid, signal.SIGUSR1)  # el fake escribe basura y sigue vivo
    assert esperar(lambda: bridge.connected_servers() == [] and registry.names() == [])
    assert "mcp:fake" not in router.dynamic_groups()
    assert esperar(lambda: proc.poll() is not None)
    assert "basura no-JSON" in caplog.text  # la causa real, no "el proceso murió"


def test_la_deteccion_perezosa_loguea_la_causa_real(caplog):
    """Si el aviso del cliente no llegó a tiempo, la llamada lo detecta: el log
    debe decir POR QUÉ (antes decía siempre "el proceso murió", aunque siguiera
    vivo y la causa fuera basura en stdout)."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    registry, _ = registrar({"fake": {"command": FAKE_CMD}})
    client = bridge._STATE["fake"].client
    client._on_dead = None  # simula que el aviso no llegó
    assert client._proc is not None
    os.kill(client._proc.pid, signal.SIGUSR1)
    assert esperar(lambda: not client.alive)
    out = registry.run("mcp_fake_echo", {"text": "x"})
    assert out.startswith("El server MCP 'fake'")
    assert registry.names() == []
    assert "basura no-JSON" in caplog.text
    assert "el proceso murió" not in caplog.text


def test_aviso_tardio_de_un_cliente_viejo_no_tumba_al_que_lo_sustituyo():
    settings = mk_settings({"fake": {"command": FAKE_CMD}})
    registry = Registry()
    bridge.register_mcp_tools(registry, settings)
    viejo = bridge._STATE["fake"].client
    bridge.unregister_server(registry, "fake")
    bridge.register_mcp_tools(registry, settings)
    nuevo = bridge._STATE["fake"].client
    assert nuevo is not viejo
    bridge._on_transport_dead(viejo, "EOF en stdout")
    time.sleep(0.3)
    assert bridge.connected_servers() == ["fake"]
    assert bridge._STATE["fake"].client is nuevo
    assert registry.get("mcp_fake_echo") is not None


def test_cliente_cerrado_por_nosotros_no_avisa_de_muerte():
    avisos: list[str] = []
    client = StdioMCPClient("fake", FAKE_CMD, on_dead=lambda _c, why: avisos.append(why))
    client.start()
    client.initialize(timeout=5)
    client.close()
    time.sleep(0.2)
    assert avisos == []


def test_cliente_avisa_una_vez_cuando_el_transporte_muere_solo():
    avisos: list[tuple[StdioMCPClient, str]] = []
    client = StdioMCPClient("fake", FAKE_CMD, on_dead=lambda c, why: avisos.append((c, why)))
    try:
        client.start()
        client.initialize(timeout=5)
        assert client._proc is not None
        os.kill(client._proc.pid, signal.SIGUSR1)
        assert esperar(lambda: len(avisos) == 1)
        assert avisos[0][0] is client
        assert "basura no-JSON" in avisos[0][1]
        assert client.dead_reason == avisos[0][1]
    finally:
        client.close()
    assert len(avisos) == 1


# ---------------------------------------------------------------------------
# mcp#14: un server a mitad del handshake es visible para el apagado
# ---------------------------------------------------------------------------

def _server_mudo(tmp_path: Path) -> tuple[list[str], Path]:
    """Server que nunca contesta ni lee stdin (ignora el EOF): apunta su pid."""
    pidfile = tmp_path / "pid"
    code = (
        "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    return [sys.executable, "-c", code, str(pidfile)], pidfile


def _arranque_en_hilo(tmp_path: Path) -> tuple[threading.Thread, list, int]:
    command, pidfile = _server_mudo(tmp_path)
    out: list = []
    hilo = threading.Thread(
        target=lambda: out.append(bridge.register_mcp_tools(
            Registry(), mk_settings({"mudo": {"command": command}}, startup_timeout_s=8),
        )),
        daemon=True,
    )
    hilo.start()
    assert esperar(lambda: pidfile.exists() and pidfile.read_text().strip() != "")
    return hilo, out, int(pidfile.read_text())


def test_terminate_all_alcanza_al_server_que_esta_en_el_handshake(tmp_path):
    hilo, out, pid = _arranque_en_hilo(tmp_path)
    try:
        assert bridge.connected_servers() == []  # aún no está en _STATE
        bridge.terminate_all()
        assert esperar(lambda: not vivo(pid), plazo=3.0), "el server quedó huérfano"
        hilo.join(timeout=3)
        assert not hilo.is_alive()  # el handshake abortó, no esperó al plazo
        assert out == [[]]
    finally:
        if vivo(pid):
            os.kill(pid, signal.SIGKILL)


def test_close_all_alcanza_al_server_que_esta_en_el_handshake(tmp_path):
    hilo, out, pid = _arranque_en_hilo(tmp_path)
    try:
        bridge.close_all()
        assert esperar(lambda: not vivo(pid), plazo=3.0), "el server quedó huérfano"
        hilo.join(timeout=3)
        assert not hilo.is_alive()
        assert out == [[]]
    finally:
        if vivo(pid):
            os.kill(pid, signal.SIGKILL)


def test_ctrl_c_en_el_handshake_cierra_el_server(monkeypatch):
    clientes: list[StdioMCPClient] = []

    def initialize(self: StdioMCPClient, timeout: float = 15.0) -> dict:
        clientes.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(StdioMCPClient, "initialize", initialize)
    try:
        with pytest.raises(KeyboardInterrupt):
            registrar({"fake": {"command": FAKE_CMD}})
        assert len(clientes) == 1
        proc = clientes[0]._proc
        assert proc is not None
        assert proc.poll() is not None, "el server siguió vivo tras el Ctrl+C"
        assert bridge._STARTING == set()
    finally:
        for c in clientes:
            c.close()


def test_server_que_muere_justo_tras_el_handshake_no_se_registra(monkeypatch):
    """Si el transporte muere entre tools/list y el registro, el aviso del
    cliente no encuentra a quién retirar: el registro debe verlo y saltarlo."""
    real = StdioMCPClient.list_tools

    def list_tools(self: StdioMCPClient, timeout: float = 15.0) -> list:
        tools = real(self, timeout)
        assert self._proc is not None
        os.killpg(self._proc.pid, signal.SIGKILL)
        assert esperar(lambda: not self.alive)
        return tools

    monkeypatch.setattr(StdioMCPClient, "list_tools", list_tools)
    registry, names = registrar({"fake": {"command": FAKE_CMD}})
    assert names == [] and registry.names() == []
    assert bridge.connected_servers() == []
    assert "mcp:fake" not in router.dynamic_groups()
    assert bridge._STARTING == set()


# ---------------------------------------------------------------------------
# mcp#9: startup_timeout_s es UN presupuesto para tools/list entero, no por página
# ---------------------------------------------------------------------------

# 13 tools de 2 en 2 = 7 páginas de 0.4 s (~2.8 s): cada página cabe en 1 s; la lista, no.
_PAGINAS_LENTAS = {"FAKE_MCP_PAGE_SIZE": "2", "FAKE_MCP_PAGE_DELAY": "0.4"}


def test_list_tools_el_timeout_es_de_la_lista_entera_no_de_cada_pagina():
    client = StdioMCPClient("fake", FAKE_CMD, _PAGINAS_LENTAS)
    try:
        client.start()
        client.initialize(timeout=10)
        t0 = time.monotonic()
        with pytest.raises(MCPTimeout, match="tools/list"):
            client.list_tools(timeout=1.0)
        # Antes: cada página estrenaba el plazo entero y la lista tardaba ~2.8 s.
        assert time.monotonic() - t0 < 2.0
        assert client.alive  # un timeout no rompe el transporte
    finally:
        client.close()


def test_list_tools_paginado_que_cabe_en_el_plazo_sigue_entero():
    client = StdioMCPClient("fake", FAKE_CMD, {"FAKE_MCP_PAGE_SIZE": "2"})
    try:
        client.start()
        client.initialize(timeout=10)
        assert len(client.list_tools(timeout=10)) == 13
    finally:
        client.close()


def test_server_con_catalogo_lento_se_salta_dentro_del_presupuesto_de_arranque(caplog):
    """El escenario del hallazgo: el docstring del bridge, la config y README_M4
    prometen un presupuesto TOTAL (handshake + tools/list); un server paginado
    cuyas páginas caben de una en una bloqueaba el arranque páginas x plazo."""
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    t0 = time.monotonic()
    registry, names = registrar(
        {"lento": {"command": FAKE_CMD, "env": _PAGINAS_LENTAS}}, startup_timeout_s=1,
    )
    assert time.monotonic() - t0 < 2.5
    assert names == [] and registry.names() == []
    assert bridge.connected_servers() == [] and bridge._STARTING == set()
    assert "no arrancó" in caplog.text


# ---------------------------------------------------------------------------
# mcp#7 / x-concurrency-lifecycle#19: close() da al server su cierre educado
# ---------------------------------------------------------------------------

def test_close_deja_al_server_terminar_solo_al_ver_eof_antes_del_sigterm():
    """La spec y el docstring de close(): stdin -> ESPERAR -> SIGTERM -> SIGKILL.
    Antes el SIGTERM salía microsegundos después del EOF y un server que guarda
    su estado al cerrarse moría a media limpieza (returncode -15)."""
    client = StdioMCPClient("fake", FAKE_CMD, {"FAKE_MCP_EOF_DELAY": "0.3"})
    client.start()
    client.initialize(timeout=10)
    proc = client._proc
    assert proc is not None
    client.close()
    assert proc.returncode == 0, f"returncode={proc.returncode}: SIGTERM a media limpieza"
    assert not client.alive


def test_close_de_un_server_que_ignora_el_eof_sigue_acotado():
    """La espera educada tiene plazo: un server que no lee stdin acaba en SIGTERM."""
    client = StdioMCPClient("sordo", [sys.executable, "-c", "import time; time.sleep(60)"])
    client.start()
    proc = client._proc
    assert proc is not None
    t0 = time.monotonic()
    client.close()
    assert time.monotonic() - t0 < 4.0
    assert proc.returncode == -signal.SIGTERM  # la red de seguridad sigue ahí


# ---------------------------------------------------------------------------
# mcp#4: el wrapper muere con SIGTERM pero el server real (nieto) no
# ---------------------------------------------------------------------------

def test_close_escala_a_sigkill_si_el_nieto_sobrevive_al_sigterm(tmp_path):
    """`sh -c` (como npx/uvx) muere con el SIGTERM; el nieto que habla el
    protocolo lo ignora y retiene las tuberías. Antes close() daba el cierre por
    bueno al morir el hijo directo: el server quedaba huérfano (sobrevive a
    Crotolamo), el lector bloqueado para siempre y stdout/stderr sin cerrar."""
    pidfile = tmp_path / "nieto"
    nieto = (
        f"{sys.executable} -c \"import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)\" "
        f"{pidfile}"
    )
    # '; true': sh no hace exec y queda de launcher (que SÍ muere con SIGTERM).
    client = StdioMCPClient("wrap", ["sh", "-c", f"{nieto}; true"])
    client.start()
    assert esperar(lambda: pidfile.exists() and pidfile.read_text().strip() != "")
    pid = int(pidfile.read_text())
    try:
        t0 = time.monotonic()
        client.close()
        assert time.monotonic() - t0 < 8.0
        # Con plazo, como los tests del handshake: close() vuelve en cuanto sus
        # hilos ven EOF en las tuberías, y el kernel las cierra ANTES de dejar
        # al nieto (ya con el SIGKILL dentro) como zombi. Mirado en ese hueco
        # (ms, bajo carga) aún sale "R" y el test fallaba 1 de cada ~15. Sin el
        # SIGKILL sigue fallando: el nieto duerme 60 s ignorando el SIGTERM.
        assert esperar(lambda: not vivo(pid), plazo=3.0), "el server real quedó huérfano"
        assert client._reader is not None and not client._reader.is_alive()
        proc = client._proc
        assert proc is not None and proc.stdout is not None and proc.stderr is not None
        assert proc.stdout.closed and proc.stderr.closed
    finally:
        if vivo(pid):
            os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# mcp#5: un timeout cancela la petición abandonada (notifications/cancelled)
# ---------------------------------------------------------------------------

def test_timeout_manda_notifications_cancelled_con_el_id_abandonado(tmp_path):
    """La spec: al dejar de esperar, el emisor DEBERÍA mandar
    notifications/cancelled. Sin él, el server seguía con la operación
    abandonada (y uno que atiende de uno en uno encolaba detrás la siguiente)."""
    log_path = tmp_path / "stdin.log"
    client = StdioMCPClient("fake", FAKE_CMD, {"FAKE_MCP_LOG": str(log_path)})

    def recibidos() -> list[dict]:
        if not log_path.exists():
            return []
        return [json.loads(raw) for raw in log_path.read_bytes().splitlines() if raw.strip()]

    try:
        client.start()
        client.initialize(timeout=10)
        with pytest.raises(MCPTimeout):
            client.call_tool("lenta", {"segundos": 0.6}, timeout=0.2)
        assert client.alive  # un timeout no rompe el transporte
        assert esperar(lambda: any(
            m.get("method") == "notifications/cancelled" for m in recibidos()
        ))
        llamada = next(m for m in recibidos() if m.get("method") == "tools/call")
        cancel = next(m for m in recibidos() if m.get("method") == "notifications/cancelled")
        assert cancel["params"]["requestId"] == llamada["id"]
        assert "id" not in cancel  # es notificación: el server no la contesta
        assert client.call_tool("echo", {"text": "ok"}, timeout=5)[0] == '{"text": "ok"}'
    finally:
        client.close()


def test_el_initialize_no_se_cancela_nunca():
    """La spec prohíbe al cliente cancelar el initialize; el resto sí."""
    client = StdioMCPClient("x", FAKE_CMD)
    enviados: list[dict] = []

    def enviar(message: dict, deadline: float | None = None) -> None:
        enviados.append(message)

    client._send = enviar  # type: ignore[method-assign]
    with pytest.raises(MCPTimeout):
        client._request("initialize", {}, 0.05)
    assert [m.get("method") for m in enviados] == ["initialize"]
    with pytest.raises(MCPTimeout):
        client._request("tools/call", {"name": "x", "arguments": {}}, 0.05)
    assert [m.get("method") for m in enviados] == [
        "initialize", "tools/call", "notifications/cancelled",
    ]
    assert enviados[-1]["params"]["requestId"] == enviados[1]["id"]


# ---------------------------------------------------------------------------
# mcp#16: un éxito con "error": null no es un error JSON-RPC
# ---------------------------------------------------------------------------

def test_respuesta_con_error_null_es_un_exito_no_un_error():
    """Librerías JSON-RPC laxas (y la 1.0) mandan "error": null junto al
    result. Antes se miraba si la CLAVE existía: el handshake fallaba con
    MCPError('None') y cada llamada volvía "devolvió un error, patrón: None"."""
    client = StdioMCPClient("laxo", FAKE_CMD, {"FAKE_MCP_ERROR_NULL": "1"})
    try:
        client.start()
        assert client.initialize(timeout=10)["serverInfo"]["name"] == "fake-mcp"
        assert len(client.list_tools(timeout=10)) == 13
        assert client.call_tool("echo", {"text": "ok"}, timeout=5) == ('{"text": "ok"}', False)
        # Un error de verdad (objeto) sigue siendo MCPError.
        with pytest.raises(MCPError, match="no_existe"):
            client.call_tool("no_existe", {}, timeout=5)
    finally:
        client.close()


def test_server_laxo_con_error_null_registra_sus_tools():
    laxo = {"command": FAKE_CMD, "env": {"FAKE_MCP_ERROR_NULL": "1"}}
    registry, names = registrar({"laxo": laxo})
    assert "mcp_laxo_echo" in names
    assert registry.run("mcp_laxo_echo", {"text": "hola"}) == '{"text": "hola"}'


# ---------------------------------------------------------------------------
# mcp#15: un sustituto suelto en los argumentos no revienta ni deja pendientes
# ---------------------------------------------------------------------------

def test_sustituto_suelto_en_los_argumentos_llega_como_u_fffd():
    """Medio emoji ("\\ud83d") que json.loads acepta: el encode a UTF-8 de
    _send lanzaba UnicodeEncodeError, la tool "reventaba" y la entrada
    pendiente quedaba huérfana en _pending (una por intento)."""
    client = StdioMCPClient("fake", FAKE_CMD)
    try:
        client.start()
        client.initialize(timeout=10)
        text, is_error = client.call_tool("echo", {"text": "listo \ud83d"}, timeout=5)
        assert json.loads(text) == {"text": "listo �"}
        assert is_error is False
        assert client._pending == {}
        assert client.alive
    finally:
        client.close()


def test_sustituto_suelto_via_bridge_no_es_un_error_duro():
    from crotolamo.core.tool_parsing import is_hard_error, parse_arguments

    registry, _ = registrar({"fake": {"command": FAKE_CMD}})
    out = registry.run("mcp_fake_echo", parse_arguments('{"text": "listo \\ud83d"}'))
    assert not is_hard_error(out)
    assert json.loads(out) == {"text": "listo �"}
    assert bridge._STATE["fake"].client._pending == {}


def test_un_fallo_al_mandar_que_no_es_de_transporte_no_deja_la_peticion_pendiente():
    client = StdioMCPClient("fake", FAKE_CMD)
    try:
        client.start()
        client.initialize(timeout=10)
        with pytest.raises(TypeError):
            client.call_tool("echo", {"x": object()}, timeout=1)  # json.dumps no puede
        assert client._pending == {}
        assert client.call_tool("echo", {"text": "ok"}, timeout=5)[0] == '{"text": "ok"}'
    finally:
        client.close()


# ---------------------------------------------------------------------------
# mcp#8: el plazo cubre también la escritura a stdin, y el cierre no se cuelga
# ---------------------------------------------------------------------------

# Más que la tubería (64 KiB): con el server sin leer, write() se bloqueaba.
_ENORME = {"text": "x" * 200_000}


def _sordo() -> StdioMCPClient:
    """Fake ocupado 60 s en `lenta` (atiende de uno en uno): no lee stdin."""
    client = StdioMCPClient("sordo", FAKE_CMD)
    client.start()
    client.initialize(timeout=10)
    with pytest.raises(MCPTimeout):
        client.call_tool("lenta", {"segundos": 60}, timeout=0.2)
    return client


def _matar(client: StdioMCPClient) -> None:
    proc = client._proc
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _en_hilo(func: Callable[[], Any]) -> tuple[threading.Thread, list]:
    out: list = []

    def run() -> None:
        try:
            out.append(func())
        except BaseException as error:  # noqa: BLE001 - el test lo inspecciona
            out.append(error)

    hilo = threading.Thread(target=run, daemon=True)
    hilo.start()
    return hilo, out


def test_llamada_enorme_a_un_server_que_no_lee_respeta_el_timeout():
    client = _sordo()
    try:
        t0 = time.monotonic()
        hilo, out = _en_hilo(lambda: client.call_tool("echo", _ENORME, timeout=0.5))
        hilo.join(timeout=5)
        assert not hilo.is_alive(), "la escritura a stdin se saltó el timeout"
        assert time.monotonic() - t0 < 3
        # Medio mensaje quedó en la tubería: el stream no se puede resincronizar.
        assert isinstance(out[0], MCPTransportError)
        assert not client.alive
        assert client._pending == {}
    finally:
        _matar(client)
        client.close()


def test_llamada_detras_de_una_escritura_atascada_respeta_su_propio_timeout():
    client = _sordo()
    try:
        atascada, _ = _en_hilo(lambda: client.call_tool("echo", _ENORME, timeout=30))
        assert esperar(client._write_lock.locked)
        t0 = time.monotonic()
        hilo, out = _en_hilo(lambda: client.call_tool("echo", {"text": "y"}, timeout=0.5))
        hilo.join(timeout=5)
        assert not hilo.is_alive(), "esperó el lock de escritura sin plazo"
        assert time.monotonic() - t0 < 3
        assert isinstance(out[0], MCPTimeout)  # no llegó a escribir nada: sigue sano
        assert client.alive
        assert len(client._pending) == 1  # solo la atascada: la suya no quedó huérfana
    finally:
        _matar(client)
        client.close()


def test_close_no_se_cuelga_con_una_escritura_atascada():
    """close() hacía stdin.close(), que espera el lock del BufferedWriter que
    retiene el escritor bloqueado: el atexit close_all no terminaba nunca."""
    client = _sordo()
    proc = client._proc
    assert proc is not None and proc.stdin is not None
    try:
        atascada, errores = _en_hilo(lambda: client.call_tool("echo", _ENORME, timeout=30))
        assert esperar(client._write_lock.locked)
        t0 = time.monotonic()
        cierre, _ = _en_hilo(client.close)
        cierre.join(timeout=8)
        assert not cierre.is_alive(), "close() se colgó"
        assert time.monotonic() - t0 < 6.5
        assert proc.poll() is not None
        atascada.join(timeout=3)
        assert not atascada.is_alive()
        assert isinstance(errores[0], MCPTransportError)
        assert proc.stdin.closed  # el fd no queda abierto
    finally:
        _matar(client)
        client.close()


def test_terminate_now_no_espera_a_una_escritura_atascada():
    """El manejador de SIGTERM/Ctrl+C del listener: antes no llegaba a os._exit."""
    client = _sordo()
    proc = client._proc
    assert proc is not None
    try:
        atascada, _ = _en_hilo(lambda: client.call_tool("echo", _ENORME, timeout=30))
        assert esperar(client._write_lock.locked)
        t0 = time.monotonic()
        hilo, _ = _en_hilo(client.terminate_now)
        hilo.join(timeout=3)
        assert not hilo.is_alive(), "terminate_now() se colgó"
        assert time.monotonic() - t0 < 1.0
        assert esperar(lambda: proc.poll() is not None)  # el SIGTERM salió igual
        atascada.join(timeout=3)
        assert not atascada.is_alive()
    finally:
        _matar(client)
        client.close()


def test_turno_con_argumentos_enormes_a_un_server_colgado_termina_y_lo_retira():
    """El escenario del hallazgo por el bridge: timeout_s=0.3, el server se
    cuelga en una llamada y la siguiente manda >64 KiB. Antes el turno no
    terminaba nunca."""
    registry, names = registrar({"fake": {"command": FAKE_CMD}}, timeout_s=0.3)
    assert names
    client = bridge._STATE["fake"].client
    try:
        assert "tardó demasiado" in registry.run("mcp_fake_lenta", {"segundos": 60})
        t0 = time.monotonic()
        hilo, out = _en_hilo(lambda: registry.run("mcp_fake_echo", _ENORME))
        hilo.join(timeout=8)
        assert not hilo.is_alive(), "el turno no terminó"
        assert time.monotonic() - t0 < 6
        assert out[0].startswith("El server MCP 'fake'")
        assert registry.names() == [] and bridge.connected_servers() == []
    finally:
        _matar(client)


# ---------------------------------------------------------------------------
# mcp#10: el server que muere SOLO deja en el log su código de salida y su stderr
# ---------------------------------------------------------------------------

def _muere_con(code: str) -> list[str]:
    return [sys.executable, "-c", code]


_SIN_TOKEN = _muere_con(
    "import sys; sys.stderr.write('FATAL: falta GITHUB_TOKEN en el entorno\\n'); sys.exit(2)"
)


def test_server_que_muere_al_arrancar_loguea_su_stderr_y_su_codigo(caplog):
    """El escenario del hallazgo: falta un token o un módulo, el server lo dice
    por stderr y sale. El stderr iba solo a DEBUG (el nivel por defecto es INFO)
    y el aviso decía "EOF en stdout (el server terminó)", sin código de salida:
    diagnosticarlo exigía reiniciar con CROTOLAMO_LOG=DEBUG."""
    caplog.set_level(logging.INFO, logger="crotolamo")
    registry, names = registrar({
        "sin_token": {"command": _SIN_TOKEN},
        "sin_modulo": {"command": _muere_con("import paquete_que_no_existe")},
    })
    assert names == [] and registry.names() == []
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    sin_token = next(m for m in avisos if "'sin_token': no arrancó" in m)
    assert "FATAL: falta GITHUB_TOKEN en el entorno" in sin_token
    assert "código de salida 2" in sin_token
    sin_modulo = next(m for m in avisos if "'sin_modulo': no arrancó" in m)
    assert "ModuleNotFoundError" in sin_modulo
    assert "código de salida 1" in sin_modulo


def test_el_stderr_de_un_server_sano_sigue_en_debug(caplog):
    """Solo la muerte sola sube la cola de stderr: los logs normales del server
    (el fake escribe "fake-mcp: listo") no ensucian el log por defecto, ni
    tampoco al cerrarlo nosotros."""
    caplog.set_level(logging.DEBUG, logger="crotolamo")
    client = StdioMCPClient("fake", FAKE_CMD)
    client.start()
    client.initialize(timeout=10)
    assert esperar(lambda: "fake-mcp: listo" in caplog.text)  # en DEBUG, sí
    client.close()
    assert client.dead_reason == "cliente cerrado"
    assert not any(
        "fake-mcp: listo" in r.getMessage() for r in caplog.records if r.levelno > logging.DEBUG
    )


def test_server_matado_por_una_senal_lo_dice_en_la_causa():
    avisos: list[str] = []
    client = StdioMCPClient("fake", FAKE_CMD, on_dead=lambda _c, why: avisos.append(why))
    try:
        client.start()
        client.initialize(timeout=10)
        assert client._proc is not None
        os.killpg(client._proc.pid, signal.SIGKILL)
        assert esperar(lambda: len(avisos) == 1)
        assert "EOF en stdout" in avisos[0]
        assert "SIGKILL" in avisos[0]
        assert "fake-mcp: listo" in avisos[0]  # su stderr, aunque fuera una línea sana
    finally:
        client.close()


def test_la_cola_de_stderr_tiene_tope_y_guarda_lo_ultimo():
    """Un server verborreico (o un volcado minificado de una línea) no puede
    meter megas en el aviso: se guardan las últimas líneas, recortadas."""
    code = (
        "import sys\n"
        "for i in range(200): sys.stderr.write(f'linea {i} ' + 'x' * 5000 + '\\n')\n"
        "sys.stderr.write('la causa de verdad\\n'); sys.exit(3)"
    )
    client = StdioMCPClient("verborreico", _muere_con(code))
    try:
        client.start()
        with pytest.raises(MCPTransportError) as info:
            client.initialize(timeout=10)
        mensaje = str(info.value)
        assert "la causa de verdad" in mensaje and "código de salida 3" in mensaje
        assert "linea 0 " not in mensaje  # lo primero se descarta, no lo último
        assert len(mensaje) < 6000
    finally:
        client.close()


def test_el_diagnostico_no_retrasa_sin_plazo_el_aviso_de_muerte():
    """stdout cerrado con el proceso vivo y stderr abierto: esperar al drenador
    o al código de salida no puede colgar a quien espera la respuesta."""
    code = (
        "import os, sys, time; sys.stderr.write('adios\\n'); sys.stderr.flush(); "
        "os.close(1); time.sleep(30)"
    )
    client = StdioMCPClient("medio", _muere_con(code))
    try:
        client.start()
        t0 = time.monotonic()
        with pytest.raises(MCPTransportError) as info:
            client.initialize(timeout=10)
        assert time.monotonic() - t0 < 3.0
        assert "adios" in str(info.value)
        assert "código de salida" not in str(info.value)  # sigue vivo: no hay código
    finally:
        client.close()


def test_la_cola_de_stderr_no_llega_al_modelo():
    """El aviso al modelo (y de ahí a la nube) sigue siendo el texto fijo: el
    stderr de un server puede llevar rutas o detalles del entorno."""
    registry, _ = registrar({"fake": {"command": FAKE_CMD}})
    out = registry.run("mcp_fake_morir", {})
    assert out == "El server MCP 'fake' se cayó, patrón. Lo desconecté y retiré sus tools."
    assert "fake-mcp" not in out


def test_pipe_rota_al_escribir_tambien_lleva_la_cola_de_stderr():
    """La otra vía por la que el cliente ve morir al server: EPIPE al escribir
    (el server ya no lee stdin). También ahí va su stderr, sin esperar: se
    escribe con el lock de stdin tomado."""
    code = (
        "import os, sys, time; os.close(0); sys.stderr.write('no leo stdin\\n'); "
        "sys.stderr.flush(); time.sleep(30)"
    )
    client = StdioMCPClient("cerrado", _muere_con(code))
    try:
        client.start()
        assert esperar(lambda: "no leo stdin" in client._stderr_tail)
        t0 = time.monotonic()
        with pytest.raises(MCPTransportError) as info:
            client.initialize(timeout=10)
        assert time.monotonic() - t0 < 0.4
        assert "pipe rota al escribir" in str(info.value)
        assert "stderr: no leo stdin" in str(info.value)
    finally:
        client.close()
