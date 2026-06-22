"""Tests del short-circuit de retorno directo (Misión 1) y del limpiador de
preámbulos meta (Misión 2). Todo con un LLM falso, sin Ollama real.

El short-circuit: si el modelo pide EXACTAMENTE una tool "directa" (presentacional)
y no falla duro, el agente devuelve su output sin una 2ª llamada al LLM.
"""

import pytest

from crotolamo.core.agent import (
    DEFAULT_DIRECT_TOOLS,
    ToolAgent,
    _is_hard_error,
)
from crotolamo.core.hooks import meta_preamble_cleaner
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.tools import default_registry
from crotolamo.tools import desktop


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    monkeypatch.setattr(desktop, "run_detached", lambda args: None)


class FakeLLM:
    """Reproduce una secuencia de respuestas; cuenta cuántas veces se llamó."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self._responses.pop(0)

    def chat_stream(self, messages, tools=None, on_token=None):
        self.calls.append((messages, tools))
        resp = self._responses.pop(0)
        # Para tool-calls Ollama no emite content; un texto final sí se streamea.
        if on_token and resp.content and not resp.tool_calls:
            on_token(resp.content)
        return resp


def _tool_call(name, **args):
    return {"name": name, "arguments": args}


def _make_agent(responses, direct_tools=None, post_hooks=None):
    conv = Conversation("SYS")
    guard = Guard(allowed_roots=[])
    agent = ToolAgent(
        FakeLLM(responses),
        conv,
        registry=default_registry(),
        guard=guard,
        max_iterations=6,
        direct_tools=direct_tools,
        post_hooks=post_hooks,
    )
    return agent, conv


# --------------------------- short-circuit ---------------------------

def test_direct_tool_short_circuits_one_llm_call():
    # Una sola tool directa (ram_usage) => NO debe haber 2ª llamada al LLM.
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage")],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
        # Esta 2ª respuesta NO debe consumirse si el short-circuit funciona.
        ChatResponse(content="basura que el 3B habría redactado"),
    ]
    agent, conv = _make_agent(responses)
    reply = agent.handle_turn("cuánta RAM uso?")
    # El reply es el output crudo de la tool (presentacional), no el de la 2ª llamada.
    assert "RAM:" in reply
    assert "basura" not in reply
    # Solo se hizo UNA llamada al LLM.
    assert len(agent.llm.calls) == 1
    # En memoria quedó el tool_result Y el assistant final con el mismo texto.
    assert [m for m in conv.history if m.role == "tool"]
    last = [m for m in conv.history if m.role == "assistant"][-1]
    assert last.content == reply


def test_direct_tool_with_hallucinated_arg_still_short_circuits():
    # El 3B a veces pide ram_usage con un `limit` que NO declara (se le pega de
    # list_processes). _execute_call filtra ese kwarg antes de ejecutar, así la
    # tool no revienta y el short-circuit SÍ dispara (1 sola llamada al LLM).
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage", limit=None)],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
        ChatResponse(content="no debería usarse"),
    ]
    agent, _ = _make_agent(responses)
    reply = agent.handle_turn("cuánta RAM uso?")
    assert "RAM:" in reply
    assert len(agent.llm.calls) == 1


def test_non_direct_tool_keeps_second_call():
    # search_web NO es directa => comportamiento clásico de 2 llamadas.
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("search_web", query="x", engine="google")],
                     raw_message={"tool_calls": [{"function": {"name": "search_web"}}]}),
        ChatResponse(content="Te busqué eso, patrón."),
    ]
    agent, _ = _make_agent(responses)
    reply = agent.handle_turn("busca x")
    assert reply == "Te busqué eso, patrón."
    assert len(agent.llm.calls) == 2


def test_two_tools_even_if_direct_keeps_second_call():
    # 2+ tools en el mismo turno => NO short-circuit, aunque sean directas.
    responses = [
        ChatResponse(
            content="",
            tool_calls=[_tool_call("ram_usage"), _tool_call("disk_usage")],
            raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]},
        ),
        ChatResponse(content="Aquí va el resumen, patrón."),
    ]
    agent, _ = _make_agent(responses)
    reply = agent.handle_turn("ram y disco")
    assert reply == "Aquí va el resumen, patrón."
    assert len(agent.llm.calls) == 2


def test_direct_tools_empty_disables_short_circuit():
    # direct_tools=set() => clásico de 2 llamadas (modo medición/baseline).
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage")],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
        ChatResponse(content="Tu RAM, patrón."),
    ]
    agent, _ = _make_agent(responses, direct_tools=set())
    reply = agent.handle_turn("cuánta RAM uso?")
    assert reply == "Tu RAM, patrón."
    assert len(agent.llm.calls) == 2


def test_default_direct_tools_used_when_none():
    agent, _ = _make_agent([ChatResponse(content="hola")])
    assert agent.direct_tools == set(DEFAULT_DIRECT_TOOLS)
    assert "ram_usage" in agent.direct_tools


def test_hard_error_does_not_short_circuit(monkeypatch):
    # Si la tool da un fallo DURO, se deja que el modelo reaccione (2ª llamada).
    import crotolamo.tools.system as system_mod

    def boom():
        raise RuntimeError("kaboom")

    # Forzamos que ram_usage reviente => Registry.run devuelve "La tool '...' reventó".
    monkeypatch.setattr(system_mod.ram_usage._crotolamo_tool, "func", boom)
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage")],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
        ChatResponse(content="Algo falló, patrón, pero te lo cuento."),
    ]
    agent, _ = _make_agent(responses)
    reply = agent.handle_turn("cuánta RAM uso?")
    assert reply == "Algo falló, patrón, pero te lo cuento."
    assert len(agent.llm.calls) == 2


def test_short_circuit_applies_post_hooks():
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage")],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
    ]
    agent, _ = _make_agent(responses, post_hooks=[lambda r: r + " [hooked]"])
    reply = agent.handle_turn("cuánta RAM uso?")
    assert reply.endswith(" [hooked]")
    assert len(agent.llm.calls) == 1


def test_short_circuit_streams_to_on_token():
    # En modo voz/streaming el caller solo lee on_token: el output directo debe emitirse.
    responses = [
        ChatResponse(content="", tool_calls=[_tool_call("ram_usage")],
                     raw_message={"tool_calls": [{"function": {"name": "ram_usage"}}]}),
    ]
    agent, _ = _make_agent(responses)
    chunks = []
    reply = agent.handle_turn("cuánta RAM uso?", on_token=lambda t: chunks.append(t))
    assert "".join(chunks) == reply
    assert len(agent.llm.calls) == 1


# --------------------------- Tool.direct flag ---------------------------

def test_tool_direct_flag_via_decorator_triggers_short_circuit():
    from crotolamo.tools.base import Registry, tool

    @tool(name="presenta_xyz", direct=True)
    def presenta_xyz() -> str:
        """Tool de prueba presentacional."""
        return "Resultado presentacional, patrón."

    reg = Registry()
    reg.register(presenta_xyz._crotolamo_tool)
    conv = Conversation("SYS")
    # direct_tools VACÍO: el short-circuit debe venir del flag Tool.direct.
    agent = ToolAgent(
        FakeLLM([
            ChatResponse(content="", tool_calls=[_tool_call("presenta_xyz")],
                         raw_message={"tool_calls": [{"function": {"name": "presenta_xyz"}}]}),
            ChatResponse(content="no debería usarse"),
        ]),
        conv, registry=reg, guard=Guard(allowed_roots=[]), direct_tools=set(),
    )
    reply = agent.handle_turn("dame eso")
    assert reply == "Resultado presentacional, patrón."
    assert len(agent.llm.calls) == 1


# --------------------------- _is_hard_error ---------------------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "La tool 'ram_usage' reventó, patrón: boom",
    "Argumentos inválidos para 'x', patrón: y",
])
def test_is_hard_error_true(text):
    assert _is_hard_error(text) is True


@pytest.mark.parametrize("text", [
    "RAM: 5G usados de 15G (33%), patrón.",
    "No pude leer /proc/meminfo, patrón.",  # soft-error en personaje: SÍ es directo
    "No hay nada sonando, patrón. El silencio también es música.",
])
def test_is_hard_error_false(text):
    assert _is_hard_error(text) is False


# --------------------------- meta_preamble_cleaner ---------------------------

def test_cleaner_strips_no_params_preamble():
    raw = ("Patrón, parece que la herramienta 'ram_usage' no acepta parámetros. "
           "Te resumo el resultado: RAM: 5G usados de 15G (33%), patrón.")
    out = meta_preamble_cleaner(raw)
    assert out == "RAM: 5G usados de 15G (33%), patrón."


def test_cleaner_leaves_normal_reply_untouched():
    raw = "Claro patrón, tu RAM está al 33%."
    assert meta_preamble_cleaner(raw) == raw


def test_cleaner_wired_by_default_in_toolagent():
    # post_hooks=None (como hace build_agent) => el limpiador meta queda activo.
    conv = Conversation("SYS")
    agent = ToolAgent(FakeLLM([ChatResponse(content="x")]), conv,
                      registry=default_registry(), guard=Guard(allowed_roots=[]))
    assert meta_preamble_cleaner in agent.post_hooks


def test_cleaner_strips_meta_from_final_reply():
    # Respuesta conversacional con preámbulo meta => sale limpia por el post-hook.
    raw = "Parece que la herramienta 'ram_usage' no acepta parámetros. Te resumo el resultado: todo bien, patrón."
    agent, _ = _make_agent([ChatResponse(content=raw)])
    reply = agent.handle_turn("estado")
    assert reply == "todo bien, patrón."


def test_cleaner_never_returns_empty():
    # Si todo el texto fuese preámbulo meta, devolvemos el original (no vacío).
    raw = "Te resumo el resultado:"
    out = meta_preamble_cleaner(raw)
    assert out  # no vacío
