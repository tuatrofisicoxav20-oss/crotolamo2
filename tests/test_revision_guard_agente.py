"""Tres arreglos de la revisión en guard, agente y settings.

- El guard no toma por ruta el CONTENIDO de un argumento de texto (write_file
  con content="/usr/bin/env ..." se bloqueaba; remember_fact con "~/..." pedía
  confirmación).
- Si el guard o el confirm_fn lanzan, el agente anota igual un resultado de
  tool: antes el historial quedaba con un assistant(tool_calls) huérfano y GLM
  rechazaba todos los turnos siguientes hasta /reset.
- allowed_roots/confirm_roots deben ser listas: un string se iteraba por
  caracteres y metía "/" como raíz permitida (todo el disco en zona libre).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.settings import load_settings
from crotolamo.tools.base import Registry, Tool


def _tool(name: str, func, **params) -> Tool:
    return Tool(
        name=name, func=func, description=name,
        parameters={"type": "object", "properties": {p: {"type": "string"} for p in params},
                    "required": list(params)},
    )


# --- guard: contenido no es ruta -------------------------------------------------

def test_content_que_parece_ruta_no_se_bloquea(tmp_path):
    guard = Guard([tmp_path])
    write_file = _tool("write_file", lambda path, content: "ok", path="", content="")
    decision = guard.check(write_file, {
        "path": str(tmp_path / "run.sh"),
        "content": "/usr/bin/env python3\nprint(1)\n",
    })
    assert decision.allowed and not decision.needs_confirmation


def test_texto_libre_con_tilde_de_home_no_pide_confirmacion(tmp_path):
    guard = Guard([tmp_path])
    remember = _tool("remember_fact", lambda texto: "ok", texto="")
    decision = guard.check(remember, {"texto": "~/proyectos es donde guardo todo"})
    assert decision.allowed and not decision.needs_confirmation


def test_argumento_de_ruta_fuera_del_corral_sigue_bloqueado(tmp_path):
    guard = Guard([tmp_path])
    read_file = _tool("read_file", lambda path: "ok", path="")
    assert not guard.check(read_file, {"path": "/etc/passwd"}).allowed
    # Y un valor con pinta de ruta en un argumento sin nombre conocido también.
    otra = _tool("otra", lambda destino_final: "ok", destino_final="")
    assert not guard.check(otra, {"destino_final": "/etc/passwd"}).allowed


# --- agente: siempre hay resultado de tool ---------------------------------------

class _LLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None):
        return self._responses.pop(0)


class _GuardQueRevienta:
    def check(self, tool, arguments):
        raise ValueError("embedded null byte")


def test_guard_que_lanza_no_deja_el_historial_roto(caplog):
    registry = Registry()
    registry.register(_tool("leer", lambda path: "contenido", path=""))
    conv = Conversation("SYS")
    agent = ToolAgent(
        _LLM([
            ChatResponse(content="", tool_calls=[{"name": "leer", "arguments": {"path": "a\0b"}}],
                         raw_message={"tool_calls": [{"function": {"name": "leer",
                                                                    "arguments": {"path": "a\0b"}}}]}),
            ChatResponse(content="No pude leerlo, patrón."),
        ]),
        conv, registry=registry, guard=_GuardQueRevienta(), fastpath=False,
    )
    reply = agent.handle_turn("lee a\0b")
    assert reply == "No pude leerlo, patrón."
    roles = [m.role for m in conv.history]
    # user -> assistant(tool_calls) -> tool -> assistant: ningún tool_call huérfano.
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert conv.history[2].content.startswith("La tool 'leer' reventó")
    assert "reventó fuera del registry" in caplog.text


def test_confirm_fn_que_lanza_tampoco_rompe_el_turno():
    registry = Registry()
    borrar = _tool("borrar", lambda path: "borrado", path="")
    borrar.safe = False
    registry.register(borrar)

    def confirm_roto(reason):
        raise RuntimeError("sounddevice murió")

    conv = Conversation("SYS")
    agent = ToolAgent(
        _LLM([
            ChatResponse(content="", tool_calls=[{"name": "borrar", "arguments": {"path": "x"}}],
                         raw_message={"tool_calls": [{"function": {"name": "borrar",
                                                                    "arguments": {"path": "x"}}}]}),
            ChatResponse(content="Se me trabó la confirmación, patrón."),
        ]),
        conv, registry=registry, guard=Guard([Path("/")]), confirm_fn=confirm_roto,
        fastpath=False,
    )
    assert agent.handle_turn("borra x") == "Se me trabó la confirmación, patrón."
    assert [m.role for m in conv.history] == ["user", "assistant", "tool", "assistant"]
    assert "borrado" not in conv.history[2].content  # la tool NO corrió


# --- settings: las raíces deben ser listas ---------------------------------------

def test_allowed_roots_como_string_no_arranca(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[paths]\nhome = "~"\nallowed_roots = "~/Documentos"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="allowed_roots"):
        load_settings(cfg)


def test_confirm_roots_con_no_strings_no_arranca(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[paths]\nhome = "~"\nallowed_roots = []\nconfirm_roots = [1]\n',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="confirm_roots"):
        load_settings(cfg)


def test_listas_validas_cargan(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[paths]\nhome = "~"\nallowed_roots = ["~/Documentos"]\n', encoding="utf-8")
    s = load_settings(cfg)
    assert s.allowed_roots == [Path("~/Documentos").expanduser()]
    assert s.confirm_roots == [Path("~").expanduser()]
