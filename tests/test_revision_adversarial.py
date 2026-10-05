"""Arreglos de la revisión adversarial de M4 (MCP) y del wake con música.

Cada test fija un hallazgo verificado: cierre del cliente colgado por un nieto
que retiene las tuberías, guard fail-open en profundidad/contenido anidado,
keywords de 3 letras, strikes que no se reseteaban con un error JSON-RPC,
tools retiradas a mitad de turno, duck aplicado tras el cierre, umbrales con
música por debajo del normal, y playerctl con exit != 0 pero salida útil.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

from crotolamo.core import router
from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.mcp import bridge
from crotolamo.mcp.client import StdioMCPClient
from crotolamo.safety.guard import Guard
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry, Tool
from crotolamo.voice import media_aware
from crotolamo.voice.media_aware import Ducker, PlayerctlBackend, sane_media_threshold

from interfaces.listener import ListenerConfig

FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]


@pytest.fixture(autouse=True)
def limpio():
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)
    yield
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)


def _settings(servers: dict, **mcp) -> Settings:
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 5, "servers": servers}
    raw.update(mcp)
    return Settings(raw={"mcp": raw}, user="test", home=Path("/tmp"))


# --- cliente: cierre acotado aunque un nieto retenga las tuberías ------------------

def test_close_no_se_cuelga_con_un_wrapper_que_deja_un_nieto():
    """`sh -c` que ignora SIGTERM y lanza un hijo (hereda la tubería y el
    SIG_IGN): antes, terminate() mataba solo al sh, el lector nunca veía EOF y
    cerrar stdout se colgaba para siempre. Ahora la señal va al grupo."""
    wrapper = [
        "sh", "-c",
        f"trap '' TERM; {sys.executable} -c 'import time; time.sleep(60)'; sleep 1",
    ]
    client = StdioMCPClient("nieto", wrapper)
    client.start()
    time.sleep(0.3)  # que el nieto exista
    t0 = time.monotonic()
    client.close()
    assert time.monotonic() - t0 < 8.0
    assert not client.alive
    proc = client._proc
    assert proc is not None and proc.poll() is not None


def test_respuesta_con_id_no_hasheable_no_mata_el_lector():
    client = StdioMCPClient("x", FAKE_CMD)
    client._dispatch({"jsonrpc": "2.0", "id": [1], "result": {}})  # no lanza
    client._dispatch({"jsonrpc": "2.0", "id": {"a": 1}, "error": {"code": 1, "message": "x"}})


def test_terminate_all_manda_la_senal_sin_esperar():
    registry = Registry()
    bridge.register_mcp_tools(registry, _settings({"fake": {"command": FAKE_CMD}}))
    state = bridge._STATE["fake"]
    t0 = time.monotonic()
    bridge.terminate_all()
    assert time.monotonic() - t0 < 1.0
    assert not state.client.alive
    fin = time.monotonic() + 3.0
    while state.client._proc is not None and state.client._proc.poll() is None:
        if time.monotonic() > fin:
            pytest.fail("el server no murió tras terminate_all")
        time.sleep(0.05)


# --- bridge: config, keywords, strikes, varios registries -----------------------

def test_enabled_como_string_no_activa_mcp(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.mcp.bridge"):
        cfg = bridge.load_mcp_config({"enabled": "false"})
    assert cfg.enabled is False
    assert "booleano" in caplog.text


def test_variable_sin_definir_en_command_avisa(monkeypatch, caplog):
    monkeypatch.delenv("CROTOLAMO_NO_EXISTE_TOKEN", raising=False)
    with caplog.at_level(logging.WARNING, logger="crotolamo.mcp.bridge"):
        cfg = bridge.load_mcp_config({
            "enabled": True,
            "servers": {"s": {"command": ["x", "--token", "$CROTOLAMO_NO_EXISTE_TOKEN"]}},
        })
    assert cfg.servers[0].command[-1] == "$CROTOLAMO_NO_EXISTE_TOKEN"
    assert "sin definir" in caplog.text


def test_keywords_derivadas_no_incluyen_trozos_de_3_letras():
    server = bridge.MCPServerConfig(name="jira", command=["x"], prefix="jira")
    tools = [{"name": "get_issue", "description": ""}, {"name": "list_runs", "description": ""}]
    kws = bridge.default_keywords(server, tools)
    assert "jira" in kws and "issue" in kws
    assert "get" not in kws and "run" not in kws and "list" not in kws


def test_error_jsonrpc_resetea_los_strikes():
    registry = Registry()
    bridge.register_mcp_tools(registry, _settings({"fake": {"command": FAKE_CMD, "timeout_s": 0.3}}))
    assert registry.run("mcp_fake_lenta", {"segundos": 0.5}).startswith("El server MCP")
    assert bridge.strikes_for("fake") == 1
    time.sleep(1.0)
    # Una tool que el server no conoce responde con error JSON-RPC: está vivo.
    out = bridge._make_caller("fake", "no_existe")()
    assert "devolvió un error" in out
    assert bridge.strikes_for("fake") == 0


def test_desregistrar_retira_las_tools_de_todos_los_registries():
    settings = _settings({"fake": {"command": FAKE_CMD}})
    a = Registry()
    bridge.register_mcp_tools(a, settings)
    b = Registry()
    bridge.register_mcp_tools(b, settings)  # camino idempotente: repone en B
    assert b.get("mcp_fake_echo") is not None
    bridge.unregister_server(a, "fake")
    assert a.get("mcp_fake_echo") is None
    assert b.get("mcp_fake_echo") is None


# --- agente: una tool retirada a mitad de turno deja de verse ---------------------

class _LLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.tools_vistas: list[list[str]] = []

    def chat(self, messages, tools=None):
        self.tools_vistas.append([t["function"]["name"] for t in (tools or [])])
        return self._responses.pop(0)


def test_tool_retirada_a_mitad_de_turno_no_se_manda_en_la_siguiente_iteracion(tmp_path):
    registry = Registry()

    def se_va(**kwargs):
        registry.unregister("efimera")  # como un server MCP que se desconecta
        return "hecho"

    for name in ("efimera", "estable"):
        registry.register(Tool(
            name=name, func=se_va if name == "efimera" else (lambda **k: "ok"),
            description=name, parameters={"type": "object", "properties": {}, "required": []},
        ))
    llm = _LLM([
        ChatResponse(content="", tool_calls=[{"name": "efimera", "arguments": {}}],
                     raw_message={"tool_calls": [{"function": {"name": "efimera",
                                                                "arguments": {}}}]}),
        ChatResponse(content="Listo, patrón."),
    ])
    agent = ToolAgent(llm, Conversation("SYS"), registry=registry,
                      guard=Guard([tmp_path]), fastpath=False)
    assert agent.handle_turn("haz algo") == "Listo, patrón."
    assert "efimera" in llm.tools_vistas[0]
    assert "efimera" not in llm.tools_vistas[1] and "estable" in llm.tools_vistas[1]


# --- guard: contenido anidado y nombres de ruta de los servers MCP ----------------

def test_ruta_anidada_bajo_una_clave_de_contenido_se_inspecciona(tmp_path):
    guard = Guard([tmp_path])
    tool = Tool(name="t", func=lambda **k: "ok", description="t", parameters={})
    assert guard.check(tool, {"value": ["/etc/passwd"]}).allowed is False
    assert guard.check(tool, {"content": {"texto": "/etc/shadow"}}).allowed is False
    # Pero el contenido de PRIMER nivel sigue exento.
    assert guard.check(tool, {"content": "/usr/bin/env python3"}).allowed is True


def test_source_y_destination_relativos_no_salen_del_corral(tmp_path):
    guard = Guard([tmp_path])
    tool = Tool(name="move", func=lambda **k: "ok", description="t", parameters={})
    d = guard.check(tool, {"source": "Documentos/../../../../etc/passwd", "destination": "x"})
    assert not d.allowed or d.needs_confirmation


# --- voz: ducker tras el cierre, avisos una vez, playerctl y umbrales -------------

class _Backend:
    def __init__(self):
        self.volumes = {"spotify": "0.8"}
        self.sets: list[tuple[str, str]] = []

    def playing_players(self):
        return ["spotify", "firefox"]

    def get_volume(self, player):
        return self.volumes.get(player)  # firefox: None (no expone volumen)

    def set_volume(self, player, value):
        self.volumes[player] = value
        self.sets.append((player, value))
        return True


def test_duck_tras_close_es_no_op():
    backend = _Backend()
    ducker = Ducker([backend], factor=0.5)
    ducker.close()
    ducker.duck()
    assert backend.sets == []
    assert not ducker.is_ducked()


def test_reproductor_sin_volumen_avisa_una_sola_vez(caplog):
    backend = _Backend()
    ducker = Ducker([backend], factor=0.5)
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.media_aware"):
        for _ in range(3):
            ducker.duck()
            ducker.restore()
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING and "firefox" in r.getMessage()]
    assert len(avisos) == 1


def test_playerctl_con_exit_distinto_de_cero_pero_salida_util(monkeypatch):
    class _Done:
        returncode = 1
        stdout = "spotify\tPlaying\nfirefox\tPaused\n"

    backend = PlayerctlBackend()
    monkeypatch.setattr(backend, "_run", lambda args: _Done())
    assert backend.playing_players() == ["spotify"]

    class _Empty:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(backend, "_run", lambda args: _Empty())
    assert backend.playing_players() == []


def test_umbral_con_musica_nunca_baja_del_normal(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.media_aware"):
        assert sane_media_threshold(0.9, 0.85, "test") == 0.9
    assert "0.85" in caplog.text
    assert sane_media_threshold(0.72, 0.85, "test") == 0.85
    assert sane_media_threshold(0.5, None, "test") is None


def test_listener_config_sube_el_umbral_de_musica_al_normal():
    s = Settings(raw={"wake": {"threshold": 0.9}, "voice": {"wake_threshold_media": 0.85}},
                 user="t", home=Path("/tmp"))
    cfg = ListenerConfig.from_settings(s)
    assert cfg.threshold == 0.9 and cfg.threshold_media == 0.9


def test_media_aware_exporta_el_helper():
    assert callable(media_aware.sane_media_threshold)
