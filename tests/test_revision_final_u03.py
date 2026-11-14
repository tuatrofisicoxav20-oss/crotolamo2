"""Revisión final, unidad 03: crotolamo/core/agent.py.

1. Con [voice].stream_release = "sentence", un preámbulo de dos frases sale en
   vivo ("¡Claro que sí, patrón! Voy a revisar la memoria ahora mismo.") y, si
   el modelo pide entonces una tool de retorno directo (ram_usage con Ollama de
   cerebro, o GLM degradado con fallback_shortcuts), el short-circuit devolvía
   el resultado pero solo lo pasaba a `streamer.flush_if_held`, que no emite
   nada si el streamer ya está en vivo. on_token nunca recibía la respuesta y,
   como el StreamSpeaker sí había hablado (el preámbulo), el listener no caía a
   speak_sentences(reply): el patrón oía el preámbulo y jamás la RAM. Ahora el
   resultado de la tool sale por el Emitter del turno, en frase aparte ("\\n")
   si ya se habló algo, como el aviso de un LLMError.

2. En la misma situación, lo que soltaba una llamada salía pegado a lo que
   suelta después el turno: "ahora mismo.Me enredé en demasiados pasos" (el
   TTS no corta ahí y espeak lee el punto como la palabra "punto"). El camino
   de iteraciones agotadas sin la política de repetición llamaba a
   on_token(reply) a pelo, saltándose el Emitter. Ahora va por él, en línea
   aparte si ya se habló, como la frase enlatada de `_exhausted_reply`. Lo
   primero que suelta el streamer de la llamada SIGUIENTE (la respuesta tras la
   tool, la explicación de `_exhausted_reply`) lo separa LiveStreamer._emit
   (core/streaming.py, unidad 12); aquí se fija de punta a punta con el agente.

Con "final" (lo de fábrica) on_token recibe exactamente lo mismo que antes.
"""

from __future__ import annotations

import pytest

from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse, LLMError
from crotolamo.core.memory import Conversation
from crotolamo.core.turn import VIA_DIRECT, VIA_EXHAUSTED
from crotolamo.safety.guard import Guard
from crotolamo.tools.base import Registry, Tool
from crotolamo.voice.tts import split_sentences

_PREAMBULO = "¡Claro que sí, patrón! Voy a revisar la memoria ahora mismo."
_RAM = "Traes 7.6 GiB de 15.3, patrón."
_NOTA = "nota: comprar pan"
_ENLATADA = "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
_SIN_ARGS = {"type": "object", "properties": {}, "required": []}


class _Guion:
    """LLM de guion: `pasos` = [(texto, tool pedida o None)] por llamada; el
    texto sale por on_token en trozos de 4 caracteres, como un stream. Un paso
    que es un LLMError se lanza (el motor cayó)."""

    model = "guion"

    def __init__(self, pasos) -> None:
        self.pasos = list(pasos)
        self.llamadas = 0

    def chat_stream(self, messages, tools=None, on_token=None):
        paso = self.pasos[self.llamadas]
        self.llamadas += 1
        if isinstance(paso, LLMError):
            raise paso
        texto, tool = paso
        for i in range(0, len(texto), 4):
            on_token(texto[i:i + 4])
        if tool is None:
            return ChatResponse(content=texto)
        raw = [{"function": {"name": tool, "arguments": {}}}]
        return ChatResponse(content=texto, tool_calls=[{"name": tool, "arguments": {}}],
                            raw_message={"role": "assistant", "content": texto,
                                         "tool_calls": raw})

    def chat(self, messages, tools=None):
        return self.chat_stream(messages, tools, on_token=lambda t: None)


def _registry(ram=lambda: _RAM) -> Registry:
    reg = Registry()
    # ram_usage está en DEFAULT_DIRECT_TOOLS (la política estática, la que
    # build_agent le da a un cerebro Ollama); leer_nota no es de retorno directo.
    reg.register(Tool(name="ram_usage", func=ram, description="RAM", parameters=_SIN_ARGS))
    reg.register(Tool(name="leer_nota", func=lambda: _NOTA, description="nota",
                      parameters=_SIN_ARGS))
    return reg


def _agente(llm, release: str, reg: Registry | None = None, **kw) -> ToolAgent:
    reg = reg if reg is not None else _registry()
    return ToolAgent(llm, Conversation("SYS"), registry=reg, guard=Guard(allowed_roots=[]),
                     fastpath=False, pre_hooks=[], route_fn=lambda t: reg.schemas(),
                     stream_release=release, **kw)


# --- 1. Short-circuit de una tool directa tras un preámbulo que ya sonó ---

def test_sentence_preambulo_y_tool_directa_se_oye_el_resultado_en_frase_aparte():
    llm = _Guion([(_PREAMBULO, "ram_usage")])
    dichos: list[str] = []
    agente = _agente(llm, "sentence")

    reply = agente.handle_turn("cuánta ram tengo", on_token=dichos.append)

    assert (reply, agente.last_turn.via, llm.llamadas) == (_RAM, VIA_DIRECT, 1)
    junto = "".join(dichos)
    assert junto == _PREAMBULO + "\n" + _RAM
    # Lo que la segmentación del TTS hace de ello: la respuesta es su propia frase.
    assert split_sentences(junto) == ["¡Claro que sí, patrón!",
                                      "Voy a revisar la memoria ahora mismo.", _RAM]


def test_sentence_preambulo_de_una_frase_y_tool_directa_sale_solo_el_resultado():
    """Un preámbulo de UNA frase nunca se suelta: la respuesta sale sola, sin
    separador delante (lo de siempre)."""
    llm = _Guion([("Voy a revisar la memoria.", "ram_usage")])
    dichos: list[str] = []

    reply = _agente(llm, "sentence").handle_turn("cuánta ram", on_token=dichos.append)

    assert reply == _RAM
    assert dichos == [_RAM]


def test_preambulo_suelto_y_directa_que_acierta_en_la_llamada_siguiente():
    """El preámbulo sonó en la 1ª llamada, la tool falló duro (sin
    short-circuit), y en la 2ª la misma tool acierta con el stream retenido:
    la respuesta va aparte del preámbulo, no pegada."""
    intentos: list[int] = []

    def ram() -> str:
        intentos.append(1)
        return "" if len(intentos) == 1 else _RAM   # vacío = fallo duro

    llm = _Guion([(_PREAMBULO, "ram_usage"), ("Otra vez.", "ram_usage")])
    dichos: list[str] = []
    agente = _agente(llm, "sentence", reg=_registry(ram))

    reply = agente.handle_turn("cuánta ram", on_token=dichos.append)

    assert (reply, agente.last_turn.via, llm.llamadas) == (_RAM, VIA_DIRECT, 2)
    assert "".join(dichos) == _PREAMBULO + "\n" + _RAM


def test_final_tool_directa_on_token_recibe_lo_mismo_que_antes():
    """Lo de fábrica: con tools a la vista nada se suelta y la respuesta sale
    de una vez, sin separador."""
    llm = _Guion([(_PREAMBULO, "ram_usage")])
    dichos: list[str] = []

    reply = _agente(llm, "final").handle_turn("cuánta ram", on_token=dichos.append)

    assert reply == _RAM
    assert dichos == [_RAM]


# --- 2. Lo que suelta el turno después no sale pegado al preámbulo ---

def test_sentence_iteraciones_agotadas_la_enlatada_va_en_frase_aparte():
    """Sin la política de repetición: antes on_token(reply) a pelo."""
    llm = _Guion([(_PREAMBULO, "leer_nota")])
    dichos: list[str] = []
    agente = _agente(llm, "sentence", max_iterations=1)

    reply = agente.handle_turn("qué dice mi nota", on_token=dichos.append)

    assert (reply, agente.last_turn.via) == (_ENLATADA, VIA_EXHAUSTED)
    junto = "".join(dichos)
    assert junto == _PREAMBULO + "\n" + _ENLATADA
    assert split_sentences(junto)[1:] == ["Voy a revisar la memoria ahora mismo.",
                                          "Me enredé en demasiados pasos, patrón.",
                                          "Mejor dímelo más simple."]


def test_final_iteraciones_agotadas_on_token_recibe_lo_mismo_que_antes():
    llm = _Guion([(_PREAMBULO, "leer_nota")])
    dichos: list[str] = []

    reply = _agente(llm, "final", max_iterations=1).handle_turn("qué dice mi nota",
                                                                on_token=dichos.append)

    assert reply == _ENLATADA
    assert dichos == [_ENLATADA]


def test_sentence_explicacion_de_exhausted_reply_va_en_frase_aparte():
    """Con la política de repetición, la última petición sin tools (otro
    streamer, en vivo) explica lo que pasó: va aparte de lo ya hablado."""
    explicacion = "Ya revisé la memoria dos veces."
    llm = _Guion([(_PREAMBULO, "leer_nota"), (_PREAMBULO, "leer_nota"),
                  (explicacion, None)])
    dichos: list[str] = []
    agente = _agente(llm, "sentence", max_iterations=2, stop_on_repeated_failure=True)

    reply = agente.handle_turn("qué dice mi nota", on_token=dichos.append)

    assert (reply, llm.llamadas) == (explicacion, 3)
    assert "".join(dichos) == _PREAMBULO + "\n" + _PREAMBULO + "\n" + explicacion


@pytest.mark.parametrize("release, esperado", [
    ("sentence", _PREAMBULO + "\n" + _ENLATADA),
    ("final", _ENLATADA),
])
def test_exhausted_reply_que_falla_dice_la_enlatada_aparte_si_ya_se_hablo(release, esperado):
    llm = _Guion([(_PREAMBULO, "leer_nota"), LLMError("GLM no respondió a tiempo.")])
    dichos: list[str] = []
    agente = _agente(llm, release, max_iterations=1, stop_on_repeated_failure=True)

    reply = agente.handle_turn("qué dice mi nota", on_token=dichos.append)

    assert reply == _ENLATADA
    assert "".join(dichos) == esperado
