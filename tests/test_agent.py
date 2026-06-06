"""Tests del loop agéntico con un LLM falso (sin Ollama)."""

import pytest

from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.tools import default_registry
from crotolamo.tools import desktop


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    monkeypatch.setattr(desktop, "run_detached", lambda args: None)


class FakeLLM:
    """Reproduce una secuencia de respuestas; registra los tools enviados."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self._responses.pop(0)


def _tool_call(name, **args):
    return {"name": name, "arguments": args}


def _make_agent(responses, allowed_roots=None, confirm_fn=None,
                pre_hooks=None, post_hooks=None):
    conv = Conversation("SYS")
    guard = Guard(allowed_roots=allowed_roots or [])
    return ToolAgent(
        FakeLLM(responses),
        conv,
        registry=default_registry(),
        guard=guard,
        max_iterations=6,
        confirm_fn=confirm_fn,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
    ), conv


def test_pre_and_post_hooks_applied_in_order():
    order = []

    def pre1(t):
        order.append("pre1")
        return t + " A"

    def pre2(t):
        order.append("pre2")
        return t + " B"

    def post1(r):
        order.append("post1")
        return r + " X"

    def post2(r):
        order.append("post2")
        return r + " Y"

    agent, conv = _make_agent(
        [ChatResponse(content="reply")],
        pre_hooks=[pre1, pre2], post_hooks=[post1, post2],
    )
    out = agent.handle_turn("hi")
    assert order == ["pre1", "pre2", "post1", "post2"]
    assert out == "reply X Y"
    # El mensaje de usuario llevó los pre-hooks aplicados.
    user_msgs = [m for m in conv.history if m.role == "user"]
    assert user_msgs[0].content == "hi A B"


def test_single_tool_then_final_answer():
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("search_web", query="gatos", engine="youtube")],
                     raw_message={"tool_calls": [{"function": {"name": "search_web"}}]}),
        ChatResponse(content="Listo, patrón."),
    ]
    agent, conv = _make_agent(responses)
    reply = agent.handle_turn("busca gatos en youtube")
    assert reply == "Listo, patrón."
    # En el historial quedó el resultado de la tool con la URL de youtube.
    tool_msgs = [m for m in conv.history if m.role == "tool"]
    assert tool_msgs and "youtube.com" in tool_msgs[0].content


def test_chains_two_tools_in_one_turn():
    responses = [
        ChatResponse(
            content="",
            tool_calls=[
                _tool_call("open_url", url="youtube.com"),
                _tool_call("open_url", url="github.com"),
            ],
            raw_message={"tool_calls": [{"function": {"name": "open_url"}}]},
        ),
        ChatResponse(content="Abrí las dos, patrón."),
    ]
    agent, conv = _make_agent(responses)
    reply = agent.handle_turn("abre youtube y github")
    assert reply == "Abrí las dos, patrón."
    tool_msgs = [m for m in conv.history if m.role == "tool"]
    assert len(tool_msgs) == 2


def test_unknown_tool_is_reported_not_crash():
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("inexistente")],
                     raw_message={"tool_calls": []}),
        ChatResponse(content="Ni modo, patrón."),
    ]
    agent, conv = _make_agent(responses)
    agent.handle_turn("haz algo raro")
    tool_msgs = [m for m in conv.history if m.role == "tool"]
    assert "No tengo una tool" in tool_msgs[0].content


def test_text_emitted_tool_call_is_executed():
    # qwen2.5-coder a veces pone el tool-call como JSON en content, no en tool_calls.
    responses = [
        ChatResponse(content='{"name": "open_folder", "arguments": {"name": "descargas"}}',
                     tool_calls=[], raw_message={}),
        ChatResponse(content="Abrí descargas, patrón."),
    ]
    agent, conv = _make_agent(responses)
    reply = agent.handle_turn("abre la carpeta de descargas")
    assert reply == "Abrí descargas, patrón."
    tool_msgs = [m for m in conv.history if m.role == "tool"]
    assert tool_msgs and "Descargas" in tool_msgs[0].content


def test_text_json_not_a_tool_is_plain_reply():
    # JSON que NO referencia una tool conocida = respuesta normal, no se ejecuta nada.
    responses = [ChatResponse(content='{"dato": 42}', tool_calls=[], raw_message={})]
    agent, conv = _make_agent(responses)
    reply = agent.handle_turn("dame un json")
    assert reply == '{"dato": 42}'
    assert not [m for m in conv.history if m.role == "tool"]


def test_max_iterations_guards_infinite_loop():
    # El LLM siempre pide tool, nunca da respuesta final.
    loop = ChatResponse(content="", tool_calls=[_tool_call("open_url", url="x.com")],
                        raw_message={"tool_calls": [{"function": {"name": "open_url"}}]})
    agent, _ = _make_agent([loop] * 20)
    reply = agent.handle_turn("loop infinito")
    assert "enredé" in reply.lower()
