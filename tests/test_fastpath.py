"""Tests del fast-path sin LLM (comandos inequívocos -> tool directa)."""

from crotolamo.core import fastpath


def test_musica_transporte():
    assert fastpath.match("pausa la música") == ("music_control", {"action": "pause"})
    assert fastpath.match("Pausa.") == ("music_control", {"action": "pause"})
    assert fastpath.match("siguiente canción") == ("music_control", {"action": "next"})
    assert fastpath.match("dale play") == ("music_control", {"action": "play"})
    assert fastpath.match("quita la música") == ("music_control", {"action": "stop"})
    assert fastpath.match("qué está sonando?") == ("music_now", {})


def test_ventanas_y_sistema():
    assert fastpath.match("qué apps tengo abiertas") == ("list_windows", {})
    assert fastpath.match("cuánta RAM tengo") == ("ram_usage", {})
    assert fastpath.match("cuánto espacio queda") == ("disk_usage", {})
    assert fastpath.match("cómo está el sistema") == ("system_status", {})


def test_open_app_solo_conocidas():
    # blender está en APP_COMMANDS (defaults del repo) -> atajo
    assert fastpath.match("abre blender") == ("open_app", {"name": "blender"})
    # algo que no es una app conocida -> None (que lo razone el LLM)
    assert fastpath.match("abre la carpeta de descargas") is None
    # "terminal" ya no vive en APP_COMMANDS pero open_app la resuelve
    # vía _detect_terminal: el atajo debe seguir funcionando
    assert fastpath.match("abre la terminal") == ("open_app", {"name": "terminal"})


def test_frases_con_matices_caen_al_llm():
    # cualquier cosa más allá del comando puro NO debe matchear
    assert fastpath.match("pausa la música y dime qué hora es") is None
    assert fastpath.match("mueve el archivo de descargas") is None
    assert fastpath.match("oye, ¿podrías pausar la música cuando acabe?") is None
    assert fastpath.match("") is None
    assert fastpath.match("hola crotolamo") is None


def test_agent_ejecuta_fastpath_sin_llm():
    """Con fastpath, handle_turn NO toca el LLM y devuelve el output de la tool."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation

    class _BoomLLM:  # si el agente lo llama, el test truena
        def chat(self, *a, **k):
            raise AssertionError("el fast-path no debía llegar al LLM")

        chat_stream = chat

    class _FakeTool:
        parameters = {"properties": {"action": {"type": "string"}}}
        safe = True
        direct = True

    class _FakeRegistry:
        def get(self, name):
            return _FakeTool() if name == "music_control" else None

        def run(self, name, args):
            return f"Pausada, patrón. ({name}:{args['action']})"

        def names(self):
            return ["music_control"]

    class _OkGuard:
        def check(self, tool, args):
            class D:
                allowed = True
                needs_confirmation = False
                reason = ""
            return D()

    agent = ToolAgent(
        _BoomLLM(), Conversation("sys"), registry=_FakeRegistry(), guard=_OkGuard(),
        pre_hooks=[], post_hooks=[], fastpath=True,
    )
    tokens: list[str] = []
    reply = agent.handle_turn("pausa la música", on_token=tokens.append)
    assert "Pausada" in reply
    assert tokens == [reply]  # el modo voz recibe el texto vía on_token
    # historial alimentado: user + assistant
    msgs = agent.conversation.to_messages()
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-2]["role"] == "user"


def test_agent_fastpath_apagado_no_intercepta():
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.core.llm import LLMError

    class _DeadLLM:
        def chat(self, *a, **k):
            raise LLMError("sin ollama en tests, patrón")

    class _EmptyRegistry:
        def get(self, name):
            return None

        def names(self):
            return []

        def schemas(self):
            return []

    class _OkGuard:
        def check(self, tool, args):
            class D:
                allowed = True
                needs_confirmation = False
                reason = ""
            return D()

    agent = ToolAgent(
        _DeadLLM(), Conversation("sys"), registry=_EmptyRegistry(), guard=_OkGuard(),
        pre_hooks=[], post_hooks=[], fastpath=False,
    )
    # con fastpath=False, "pausa la música" va al LLM (que aquí truena en personaje)
    assert "patrón" in agent.handle_turn("pausa la música")
