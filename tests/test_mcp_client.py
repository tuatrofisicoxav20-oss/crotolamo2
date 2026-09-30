"""Tests del cliente MCP por stdio contra el fake server (sin red, stdlib).

Cada test lanza un proceso Python real con tests/fake_mcp_server.py: se prueba
el transporte de verdad (pipes, hilos, EOF), no un mock del cliente.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from crotolamo.mcp.client import (
    MCPError,
    MCPTimeout,
    MCPTransportError,
    StdioMCPClient,
    _summarize_content,
)

FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]


@pytest.fixture
def client():
    c = StdioMCPClient("fake", FAKE_CMD)
    c.start()
    c.initialize(timeout=5)
    yield c
    c.close()


def test_handshake_guarda_server_info():
    c = StdioMCPClient("fake", FAKE_CMD)
    try:
        c.start()
        result = c.initialize(timeout=5)
        assert result["serverInfo"]["name"] == "fake-mcp"
        assert c.server_info["name"] == "fake-mcp"
        assert c.server_capabilities == {"tools": {}}
        assert c.protocol_version == "2025-06-18"
        assert c.alive
    finally:
        c.close()


def test_list_tools_sigue_la_paginacion(client):
    # El fake pagina de 4 en 4 (13 tools => 4 páginas): si el cliente no
    # siguiera nextCursor solo vería las 4 primeras.
    tools = client.list_tools(timeout=5)
    names = [t["name"] for t in tools]
    assert len(names) == 13
    assert {"echo", "borrar", "lenta", "morir"} <= set(names)
    echo = next(t for t in tools if t["name"] == "echo")
    assert echo["annotations"] == {"readOnlyHint": True}
    assert echo["inputSchema"]["type"] == "object"


def test_call_tool_devuelve_texto_y_flag(client):
    text, is_error = client.call_tool("echo", {"text": "hola"}, timeout=5)
    assert text == '{"text": "hola"}'
    assert is_error is False


def test_call_tool_is_error_llega_como_texto(client):
    text, is_error = client.call_tool("falla", {}, timeout=5)
    assert is_error is True
    assert "propósito" in text


def test_call_tool_aplana_contenido_mixto(client):
    text, _ = client.call_tool("mixto", {}, timeout=5)
    lines = text.splitlines()
    assert lines[0] == "primera línea"
    assert "[imagen image/png]" in lines
    assert "[recurso file:///x.txt]" in lines
    assert "adjunto" in lines  # el texto embebido del recurso sí se conserva
    assert lines[-1] == "última línea"


def test_error_jsonrpc_lanza_mcperror(client):
    with pytest.raises(MCPError) as info:
        client.call_tool("no_existe", {}, timeout=5)
    assert "no_existe" in str(info.value)
    assert not isinstance(info.value, (MCPTimeout, MCPTransportError))


def test_cliente_contesta_el_ping_del_server_e_ignora_notificaciones(client):
    # El fake manda un `ping` (request) y una notificación justo tras initialize.
    text, _ = client.call_tool("pong_recibido", {}, timeout=5)
    assert text == "si"


def test_timeout_lanza_mcptimeout_y_el_cliente_sigue_vivo(client):
    t0 = time.monotonic()
    with pytest.raises(MCPTimeout):
        client.call_tool("lenta", {"segundos": 0.6}, timeout=0.3)
    assert time.monotonic() - t0 < 1.0  # no esperó a que el server terminara (0.6s)
    assert client.alive
    # La respuesta tardía de `lenta` se ignora; la siguiente llamada recibe LA
    # SUYA (el despacho por id no se cruza).
    text, _ = client.call_tool("echo", {"text": "después"}, timeout=5)
    assert text == '{"text": "después"}'


def test_proceso_muerto_lanza_mcptransporterror(client):
    with pytest.raises(MCPTransportError):
        client.call_tool("morir", {}, timeout=5)
    assert not client.alive
    # Ya muerto: cualquier llamada falla al instante, sin esperar timeout.
    t0 = time.monotonic()
    with pytest.raises(MCPTransportError):
        client.call_tool("echo", {"text": "x"}, timeout=5)
    assert time.monotonic() - t0 < 1


def test_basura_en_stdout_es_transporte_roto():
    # La spec exige stdout limpio; una línea no-JSON no se puede resincronizar.
    c = StdioMCPClient("basura", FAKE_CMD, env={"FAKE_MCP_GARBAGE": "1"})
    try:
        c.start()
        with pytest.raises(MCPTransportError):
            c.initialize(timeout=3)
        assert not c.alive
    finally:
        c.close()


def test_comando_inexistente_lanza_mcptransporterror():
    c = StdioMCPClient("nada", ["/no/existe/este/binario"])
    with pytest.raises(MCPTransportError):
        c.start()
    assert not c.alive
    c.close()  # no revienta aunque nunca arrancó


def test_close_es_idempotente_y_termina_el_proceso():
    c = StdioMCPClient("fake", FAKE_CMD)
    c.start()
    c.initialize(timeout=5)
    proc = c._proc
    c.close()
    c.close()
    assert not c.alive
    assert proc is not None and proc.returncode is not None
    with pytest.raises(MCPTransportError):
        c.call_tool("echo", {"text": "x"}, timeout=1)
    with pytest.raises(MCPTransportError):
        c.start()  # un cliente cerrado no revive: se crea otro


def test_llamadas_concurrentes_desde_varios_hilos(client):
    """Varios hilos sobre el mismo cliente: cada uno recibe SU respuesta."""
    results: dict[int, str] = {}
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            text, _ = client.call_tool("echo", {"text": f"hilo-{i}"}, timeout=5)
            results[i] = text
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    for i in range(6):
        assert results[i] == f'{{"text": "hilo-{i}"}}'


def test_summarize_content_resume_lo_que_no_es_texto():
    out = _summarize_content([
        {"type": "text", "text": "a"},
        {"type": "audio", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///y"},
        {"type": "raro"},
        "basura",
    ])
    assert out == "a\n[audio audio/wav]\n[recurso file:///y]\n[raro sin texto]"


def test_summarize_content_usa_structured_content_si_no_hay_texto():
    assert _summarize_content([], {"k": 1}) == '{"k": 1}'
    assert _summarize_content(None) == ""
