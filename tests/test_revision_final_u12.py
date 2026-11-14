"""Revisión final, unidad 12: crotolamo/core/sse.py y crotolamo/core/streaming.py.

1. `parse_sse` daba por fin LIMPIO cualquier `finish_reason`. Z.ai documenta
   `network_error` como "fallo de inferencia del modelo": el stream acababa
   vacío (o a media frase), glm lo devolvía como éxito y FallbackLLM no usaba
   el modelo local; el patrón oía "Me quedé en blanco". Y con un tool_call a
   medio llegar seguido de un `finish_reason` de corte (`length`, `sensitive`,
   `model_context_window_exceeded`...) los argumentos se degradaban a {} y la
   tool corría con sus defaults: el guard de argumentos solo miraba el stream
   SIN cierre. Ahora `network_error` es SSEError (glm lo trata como
   transitorio) y tras un corte el tool_call a medias es SSETruncated, igual
   que sin cierre; un corte aceptado queda dicho en el log.

2. Con [voice].stream_release = "sentence", un preámbulo de dos frases sale en
   vivo y la respuesta de la llamada siguiente (otro LiveStreamer, el mismo
   Emitter del turno) salía pegada: "ahora mismo.Traes la mitad", y espeak lee
   ese punto como la palabra "punto". Ahora lo primero que suelta un streamer
   va en frase aparte ("\n") si el Emitter del turno ya habló.

   Lo que quedaba fuera de esta unidad (core/agent.py) ya está: si la llamada
   del preámbulo acaba en una tool directa (short-circuit), `flush_if_held` no
   emitía el resultado porque ese streamer ya estaba "en vivo". El streamer no
   puede distinguir un resultado de tool del content que ya soltó (los tests
   de fase6 y fallback fijan que en vivo no se repite nada); el agente sí, y
   ahora lo emite él por el Emitter del turno, en frase aparte.
"""

from __future__ import annotations

import json
import logging

import pytest

from crotolamo.core.llm import ChatResponse
from crotolamo.core.sse import SSEError, SSETruncated, parse_sse
from crotolamo.core.streaming import Emitter, LiveStreamer


def _lineas(*objs) -> list[bytes]:
    return [f"data: {json.dumps(o)}\n".encode() for o in objs]


def _fin(reason: str) -> dict:
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def _texto(texto: str) -> dict:
    return {"choices": [{"delta": {"content": texto}}]}


def _tool(args: str) -> dict:
    return {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "c1", "function": {"name": "borrar", "arguments": args},
    }]}}]}


_ARGS_A_MEDIAS = '{"ruta": "/home/user/notas/'
_ARGS_ENTEROS = '{"ruta": "/home/user/notas/a.txt"}'
_DONE = [b"data: [DONE]\n"]


# --- 1. finish_reason: network_error es un fallo; un corte no ejecuta args a medias ---

@pytest.mark.parametrize("previo", [[], [_texto("Traes la mi")], [_tool(_ARGS_A_MEDIAS)]],
                         ids=["vacio", "media_frase", "tool_a_medias"])
@pytest.mark.parametrize("cola", [[], _DONE], ids=["sin_done", "con_done"])
def test_finish_reason_network_error_es_un_fallo_del_servidor(previo, cola):
    stream = _lineas(*previo, _fin("network_error")) + cola

    with pytest.raises(SSEError, match="network_error"):
        parse_sse(iter(stream), None)


@pytest.mark.parametrize("reason", ["length", "sensitive", "model_context_window_exceeded",
                                    "algo_nuevo"])
@pytest.mark.parametrize("cola", [[], _DONE], ids=["sin_done", "con_done"])
def test_corte_con_tool_call_a_medias_jamas_se_ejecuta_con_args_vacios(reason, cola):
    stream = _lineas(_texto("Va."), _tool(_ARGS_A_MEDIAS), _fin(reason)) + cola

    with pytest.raises(SSETruncated, match=reason):
        parse_sse(iter(stream), None)


def test_corte_con_tool_call_sin_argumentos_tampoco_se_sabe_entero():
    stream = _lineas({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "x"}}]}}]}, _fin("length"))

    with pytest.raises(SSETruncated):
        parse_sse(iter(stream), None)


def test_corte_con_tool_call_entero_se_acepta_y_queda_en_el_log(caplog):
    stream = _lineas(_tool(_ARGS_ENTEROS), _fin("length")) + _DONE

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.sse"):
        _, msg, complete = parse_sse(iter(stream), None)

    assert complete is True
    assert msg["tool_calls"][0]["function"] == {
        "name": "borrar", "arguments": {"ruta": "/home/user/notas/a.txt"}}
    assert any("length" in r.getMessage() for r in caplog.records)


def test_corte_con_texto_se_acepta_y_queda_en_el_log(caplog):
    stream = _lineas(_texto("La capital de Francia"), _fin("length"))

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.sse"):
        content, msg, complete = parse_sse(iter(stream), None)

    assert (content, complete) == ("La capital de Francia", True)
    assert "tool_calls" not in msg
    assert any("length" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("reason", ["stop", "tool_calls"])
def test_fin_normal_no_cambia_ni_avisa(reason, caplog):
    """Un tool sin parámetros puede llegar con `arguments` vacío: con el fin
    normal se sigue aceptando como {} (lo de siempre), sin aviso."""
    stream = _lineas(_texto("Va."), {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "ram_usage"}}]}}]}, _fin(reason))

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.sse"):
        content, msg, complete = parse_sse(iter(stream), None)

    assert (content, complete) == ("Va.", True)
    assert msg["tool_calls"][0]["function"] == {"name": "ram_usage", "arguments": {}}
    assert not caplog.records


class _Local:
    model = "local"

    def __init__(self) -> None:
        self.llamadas = 0

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        if on_token is not None:
            on_token("Respuesta del local.")
        return ChatResponse(content="Respuesta del local.")


def _glm_sobre(lineas: list[bytes]):
    from crotolamo.core.glm import GLMClient
    from tests.test_auditoria_core_llm import _Conn, _stream_entero

    glm = GLMClient(api_key="k", base_url="http://localhost:1/v4")
    conn = _Conn([_stream_entero(lineas)])
    glm._transport._connect = lambda: conn
    return glm, conn


@pytest.mark.parametrize("lineas", [
    [_fin("network_error"), b"data: [DONE]\n"],
    [_tool(_ARGS_A_MEDIAS), _fin("length"), b"data: [DONE]\n"],
], ids=["network_error", "tool_a_medias_length"])
def test_de_punta_a_punta_el_fallo_de_la_nube_pasa_al_local(lineas):
    """GLMClient real sobre una conexión de mentiras, envuelto en FallbackLLM:
    antes el turno acababa en "Me quedé en blanco" (o borrando con {}) y el
    local no se llamaba nunca."""
    from crotolamo.core.fallback import FallbackLLM
    from tests.test_auditoria_core_llm import _dato

    glm, conn = _glm_sobre([_dato(o) if isinstance(o, dict) else o for o in lineas])
    local = _Local()
    llm = FallbackLLM(glm, local)
    dichos: list[str] = []
    streamer = LiveStreamer(dichos.append, hold_until_done=True)

    resp = llm.chat_stream([{"role": "user", "content": "hola"}],
                           tools=[{"type": "function", "function": {"name": "borrar"}}],
                           on_token=streamer)
    streamer.flush_if_held(resp.content)

    assert local.llamadas == 1
    assert resp.content == "Respuesta del local." and not resp.tool_calls
    assert dichos == ["Respuesta del local."]
    assert llm.degraded and conn.cerrada


# --- 2. Lo que suelta una llamada no sale pegado a lo que soltó otra del turno ---

_PREAMBULO = "¡Claro que sí, patrón! Voy a revisar la memoria ahora mismo."
_RESPUESTA = "Traes la mitad de la memoria ocupada, patrón."


def _trozos(texto: str) -> list[str]:
    return [texto[i:i + 4] for i in range(0, len(texto), 4)]


def _soltar_preambulo(emit: Emitter) -> LiveStreamer:
    """La 1ª llamada del turno: con "sentence" el preámbulo de 2 frases sale."""
    primero = LiveStreamer(emit, hold_until_done=True, release="sentence")
    for tk in _trozos(_PREAMBULO):
        primero.feed(tk)
    assert emit.released
    return primero


def test_respuesta_retenida_tras_un_preambulo_suelto_va_en_frase_aparte():
    dichos: list[str] = []
    emit = Emitter(dichos.append)
    _soltar_preambulo(emit)

    segundo = LiveStreamer(emit, hold_until_done=True)
    for tk in _trozos(_RESPUESTA):
        assert segundo.feed(tk) is False
    segundo.flush_if_held(_RESPUESTA)

    assert "".join(dichos) == _PREAMBULO + "\n" + _RESPUESTA
    assert dichos[-1] == "\n" + _RESPUESTA


@pytest.mark.parametrize("hold, release", [(True, "sentence"), (False, "final")],
                         ids=["sonda", "en_vivo"])
def test_lo_primero_que_suelta_la_llamada_siguiente_va_aparte(hold, release):
    """La sonda de "sentence" (respuesta de 2 frases) y el habla en vivo (una
    llamada sin tools) sueltan lo acumulado de golpe: también va aparte."""
    dichos: list[str] = []
    emit = Emitter(dichos.append)
    _soltar_preambulo(emit)
    antes = len(dichos)

    segundo = LiveStreamer(emit, hold_until_done=hold, release=release)
    for tk in _trozos(_RESPUESTA + " Todo bien."):
        segundo.feed(tk)
    segundo.flush_if_held(_RESPUESTA + " Todo bien.")

    assert dichos[antes].startswith("\nTrae")
    assert "".join(dichos) == _PREAMBULO + "\n" + _RESPUESTA + " Todo bien."
    assert not any(d.startswith("\n") for d in dichos[antes + 1:])   # una sola vez


@pytest.mark.parametrize("hold", [True, False])
def test_sin_nada_suelto_antes_en_el_turno_no_se_agrega_nada(hold):
    """Lo de fábrica ("final", la 1ª llamada que habla) sale igual que siempre."""
    dichos: list[str] = []
    streamer = LiveStreamer(Emitter(dichos.append), hold_until_done=hold)
    for tk in _trozos(_RESPUESTA):
        streamer.feed(tk)
    streamer.flush_if_held(_RESPUESTA)

    assert "".join(dichos) == _RESPUESTA


def test_si_ya_empieza_con_espacio_o_el_on_token_no_lleva_cuenta_no_se_agrega():
    dichos: list[str] = []
    emit = Emitter(dichos.append)
    _soltar_preambulo(emit)
    segundo = LiveStreamer(emit, hold_until_done=True)
    segundo.flush_if_held("\n" + _RESPUESTA)
    assert "".join(dichos) == _PREAMBULO + "\n" + _RESPUESTA

    sueltos: list[str] = []          # un callable cualquiera: no sabe si se habló
    LiveStreamer(sueltos.append, hold_until_done=True).flush_if_held(_RESPUESTA)
    assert sueltos == [_RESPUESTA]


@pytest.mark.parametrize("final", [
    _PREAMBULO,                              # tal cual
    "Voy a revisar la memoria ahora mismo.",  # recortado por un post-hook
    "",
], ids=["igual", "recortado", "vacio"])
def test_en_vivo_lo_que_ya_sono_no_se_repite(final):
    dichos: list[str] = []
    emit = Emitter(dichos.append)
    primero = _soltar_preambulo(emit)
    antes = list(dichos)

    primero.flush_if_held(final)

    assert dichos == antes


# --- 2 (agente). El turno entero con ToolAgent y la segmentación del TTS ---

def test_agente_sentence_preambulo_tool_y_respuesta_se_hablan_por_separado():
    from crotolamo.voice.tts import split_sentences
    from tests.test_perf_llm_turn_is_one_atomic_blocking_call import _agente, _Guion, _Oido

    llm = _Guion([(_PREAMBULO, True), (_RESPUESTA, False)])
    oido = _Oido(llm)

    reply = _agente(llm, [], stream_release="sentence").handle_turn("cuánta ram",
                                                                    on_token=oido)

    assert reply == _RESPUESTA
    junto = "".join(oido.dichos)
    assert junto == _PREAMBULO + "\n" + _RESPUESTA
    assert split_sentences(junto) == ["¡Claro que sí, patrón!",
                                      "Voy a revisar la memoria ahora mismo.", _RESPUESTA]


def test_agente_sentence_preambulo_y_tool_directa_se_oye_el_resultado():
    # Sin xfail: el short-circuit de core/agent.py emite el resultado por el
    # Emitter del turno, en frase aparte si ya sonó algo (el arreglo de u03).
    # Un solo mecanismo: el streamer no distingue el resultado de una tool del
    # content que ya soltó; el agente sí. Este test lo fija también desde u12.
    from tests.test_perf_llm_turn_is_one_atomic_blocking_call import _agente, _Guion, _Oido

    llm = _Guion([(_PREAMBULO, True)], tools=("ram_probe",))
    oido = _Oido(llm)
    agente = _agente(llm, [], stream_release="sentence")
    agente.direct_tools = {"ram_probe"}

    reply = agente.handle_turn("cuánta ram", on_token=oido)

    assert reply == "ram_probe: 7.6 GiB usados de 15.3 GiB."
    assert "".join(oido.dichos) == _PREAMBULO + "\n" + reply
