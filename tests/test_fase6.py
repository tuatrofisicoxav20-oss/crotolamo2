"""Fase 6: streaming, TTS por frases, memoria fuzzy, búsqueda y hotkeys."""

import pytest

from crotolamo.core.agent import ToolAgent, _LiveStreamer
from crotolamo.core.llm import ChatResponse, LLMClient
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.tools import default_registry, desktop


# --- streaming: parser de JSONL ---
def test_consume_stream_parses_and_emits():
    lines = [
        b'{"message":{"content":"Hola"},"done":false}',
        b'{"message":{"content":" patron"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ]
    got = []
    content, last = LLMClient._consume_stream(iter(lines), got.append)
    assert content == "Hola patron"
    assert got == ["Hola", " patron"]


def test_consume_stream_ignores_garbage_lines():
    lines = [b"", b"no-json", b'{"message":{"content":"x"},"done":true}']
    content, _ = LLMClient._consume_stream(iter(lines), None)
    assert content == "x"


# --- _LiveStreamer: streamea texto, retiene tool-calls ---
def test_live_streamer_streams_plain_text():
    out = []
    s = _LiveStreamer(out.append)
    for ch in ["Cla", "ro, ", "patron"]:
        s.feed(ch)
    assert "".join(out) == "Claro, patron"


def test_live_streamer_holds_tool_call_json():
    out = []
    s = _LiveStreamer(out.append)
    for ch in ['{"na', 'me": "open_url"}']:
        s.feed(ch)
    assert out == []  # retenido: no se filtra el JSON


# --- agente con streaming ---
class StreamLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_stream(self, messages, tools=None, on_token=None):
        resp = self._responses.pop(0)
        if on_token and resp.content:
            for ch in resp.content:
                on_token(ch)
        return resp


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    monkeypatch.setattr(desktop, "run_detached", lambda args: None)


def test_agent_streams_final_text():
    agent = ToolAgent(
        StreamLLM([ChatResponse(content="Listo, patron.")]),
        Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), max_iterations=3,
    )
    out = []
    reply = agent.handle_turn("hola", on_token=out.append)
    assert reply == "Listo, patron."
    assert "".join(out) == "Listo, patron."


def test_agent_streaming_does_not_leak_tool_json():
    agent = ToolAgent(
        StreamLLM([
            ChatResponse(content='{"name": "open_url", "arguments": {"url": "x.com"}}'),
            ChatResponse(content="Abierto, patron."),
        ]),
        Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), max_iterations=3,
    )
    out = []
    reply = agent.handle_turn("abre x.com", on_token=out.append)
    assert reply == "Abierto, patron."
    # El JSON del primer turno (tool-call) no debe aparecer en el stream.
    assert "{" not in "".join(out)


# --- memoria fuzzy ---
def test_facts_search_ranks_relevant(tmp_path):
    from crotolamo.persistence import db, facts

    dbp = tmp_path / "m.sqlite"
    db.add_fact("mi proyecto principal es Huevonitis 4", "proyectos", db_path=dbp)
    db.add_fact("me gusta el café", "preferencias", db_path=dbp)
    hits = facts.search("cuál es mi proyecto", db_path=dbp)
    assert hits and "Huevonitis" in hits[0]["texto"]


# --- TTS por frases ---
def test_split_sentences():
    from crotolamo.voice.tts import split_sentences

    out = split_sentences("Hola, patrón. ¿Qué tal? Todo bien.")
    assert len(out) == 3


# --- hotkeys ---
def test_send_hotkey_unknown_combo():
    out = desktop.send_hotkey("combo+inexistente")
    assert "No conozco" in out


def test_send_hotkey_marked_unsafe():
    assert desktop.send_hotkey._crotolamo_tool.safe is False


# --- búsqueda en proyecto ---
def test_find_in_project_finds_symbol():
    from crotolamo.tools import projects

    out = projects.find_in_project("crotolamo", "class ToolAgent")
    assert "agent.py" in out


def test_search_files_empty_query():
    from crotolamo.tools import files

    assert "Dame un nombre" in files.search_files("")
