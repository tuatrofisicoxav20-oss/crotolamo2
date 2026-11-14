"""Revisión final, unidad 13: crotolamo/core/tool_parsing.py y crotolamo/core/turn.py.

1. Dos tool-calls escritos como texto, cada uno en su propia cerca ```json.
   coerce_text_tool_calls se quedaba solo con el cuerpo de la PRIMERA cerca
   (re.search) y la segunda llamada se perdía sin dejar rastro: con el cerebro
   local y sus direct_tools por defecto, el único resultado cortaba el turno
   (el patrón oía la RAM y la música no se pausaba); sin direct_tools, el
   modelo veía una sola llamada y podía contestar como si hubieran corrido las
   dos. Las mismas llamadas sin cercas (una por línea, etiquetas de qwen) ya
   corrían las dos. Cada cerca se lee ahora como un bloque aparte: una cerca
   que no es una llamada (un ```bash con llaves) no tumba a las que sí lo son.
"""

from __future__ import annotations

import pytest

from crotolamo.core.llm import ChatResponse
from crotolamo.core.tool_parsing import coerce_text_tool_calls

_TOOLS = {"music_control", "ram_usage", "open_url"}
_RAM = '{"name": "ram_usage", "arguments": {}}'
_PAUSA = '{"name": "music_control", "arguments": {"action": "pause"}}'
_LAS_DOS = [
    {"name": "ram_usage", "arguments": {}},
    {"name": "music_control", "arguments": {"action": "pause"}},
]

_DOS_CERCAS = {
    "seguidas": f"```json\n{_RAM}\n```\n```json\n{_PAUSA}\n```",
    "con prosa entre medias": (
        f"Primero la RAM:\n```json\n{_RAM}\n```\nY luego la música:\n```json\n{_PAUSA}\n```"
    ),
    "sin etiqueta de lenguaje": f"```\n{_RAM}\n```\n```\n{_PAUSA}\n```",
}


@pytest.mark.parametrize("contenido", _DOS_CERCAS.values(), ids=_DOS_CERCAS.keys())
def test_coerce_saca_una_llamada_de_cada_cerca(contenido):
    # Antes: solo [ram_usage]; music_control no se parseaba y nada lo registraba.
    assert coerce_text_tool_calls(contenido, _TOOLS) == _LAS_DOS


def test_una_cerca_que_no_es_json_no_tumba_la_llamada_de_otra_cerca():
    """Cada cerca es un bloque aparte: juntar todas y pasarlas por el escaneo de
    todo-o-nada perdería la llamada por culpa de un ```bash con llaves."""
    llamada_y_bash = f"```json\n{_RAM}\n```\nY en bash:\n```bash\necho ${{HOME}}\n```"
    bash_y_llamada = f"```bash\necho ${{HOME}}\n```\nAhora sí:\n```json\n{_RAM}\n```"

    assert coerce_text_tool_calls(llamada_y_bash, _TOOLS) == [_LAS_DOS[0]]
    assert coerce_text_tool_calls(bash_y_llamada, _TOOLS) == [_LAS_DOS[0]]


@pytest.mark.parametrize("contenido", [
    # Una cerca de datos y otra de código: ninguna es una llamada.
    '```json\n{"edad": 30}\n```\n```python\nd = {"a": 1}\n```',
    # Dentro de una MISMA cerca sigue valiendo el todo-o-nada.
    f"```json\nUn dict se escribe {{clave: valor}}. {_RAM}\n```",
])
def test_las_cercas_sin_llamadas_limpias_siguen_sin_ejecutar_nada(contenido):
    assert coerce_text_tool_calls(contenido, _TOOLS) == []


class _Guion:
    """LLM de mentiras sin streaming: devuelve las respuestas en orden."""

    model = "guion"

    def __init__(self, *respuestas: ChatResponse) -> None:
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def chat(self, messages, tools=None):
        self.llamadas += 1
        return self.respuestas.pop(0)


@pytest.mark.parametrize("direct_tools", [None, set()], ids=["direct por defecto", "sin direct"])
def test_el_turno_ejecuta_las_dos_tools_de_dos_cercas(direct_tools):
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry, Tool

    ejecutadas: list[str] = []

    def ram_usage() -> str:
        ejecutadas.append("ram")
        return "Tienes 8 GB libres, patrón."

    def music_control(action: str = "toggle") -> str:
        ejecutadas.append(f"musica:{action}")
        return "Música en pausa, patrón."

    registry = Registry()
    registry.register(Tool(name="ram_usage", func=ram_usage, description="ram",
                           parameters={"type": "object", "properties": {}, "required": []}))
    registry.register(Tool(
        name="music_control", func=music_control, description="música",
        parameters={"type": "object", "properties": {"action": {"type": "string"}},
                    "required": []},
    ))
    llm = _Guion(
        ChatResponse(content=_DOS_CERCAS["seguidas"]),
        ChatResponse(content="Tienes 8 GB libres y pausé la música, patrón."),
    )
    conversation = Conversation("SYS")
    agent = ToolAgent(llm, conversation, registry=registry, guard=Guard(allowed_roots=[]),
                      fastpath=False, direct_tools=direct_tools)

    agent.handle_turn("dime la ram y pausa la música")

    # Antes: solo corría ram_usage; la música nunca se pausaba.
    assert ejecutadas == ["ram", "musica:pause"]
    pedidas = [
        c["function"]["name"]
        for m in conversation.to_messages() if m.get("role") == "assistant"
        for c in m.get("tool_calls") or []
    ]
    assert pedidas == ["ram_usage", "music_control"]
