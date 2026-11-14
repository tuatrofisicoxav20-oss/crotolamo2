"""Revisión final, unidad 07b: crotolamo/core/glm.py y crotolamo/core/hooks.py.

1. `GLMClient.chat()` (sin stream) daba por buena cualquier respuesta 200. Es
   la ruta de fábrica en voz: el loop concurrente (use_oww = true,
   stream_speak_concurrent = false) llama `handle_turn` sin on_token. Con
   `finish_reason: "network_error"` (Z.ai: fallo de inferencia) el turno salía
   vacío ("Me quedé en blanco") o a media frase y FallbackLLM no probaba el
   local; un objeto `error` con 200 se tragaba como respuesta vacía, y tras un
   corte (`length`, `sensitive`...) un tool_call a medias corría con {}. La
   ruta con stream ya lo trataba así (test_revision_final_u12); estos son sus
   gemelos sin stream.

2. Una API key con un carácter que no va en un header HTTP (comillas tipográficas
   o un espacio invisible al copiarla, un salto de línea) reventaba CADA turno
   con un UnicodeEncodeError/ValueError crudo de http.client: ni el modelo local
   ni un mensaje que hablara de la key. Ahora es una credencial mala más
   (GLMAuthError) y FallbackLLM usa el local, como con una key ASCII rechazada.

3. `meta_preamble_cleaner` (post-hook de fábrica, lo que se habla) recortaba el
   principio de respuestas normales: el separador tras "el resultado" era
   opcional y "...no acepta ... parámetros" se cortaba a media frase.
"""

from __future__ import annotations

import http.client
import json
import logging

import pytest

from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.glm import GLMAuthError, GLMClient
from crotolamo.core.hooks import meta_preamble_cleaner
from crotolamo.core.llm import ChatResponse, LLMError, TransientLLMError
from tests.test_auditoria_core_llm import _con_content_length, _Conn, _stream_entero

_MSGS = [{"role": "user", "content": "hola"}]
_TOOLS = [{"type": "function", "function": {"name": "borrar"}}]
_ARGS_A_MEDIAS = '{"ruta": "/home/user/notas/'
_ARGS_ENTEROS = '{"ruta": "/home/user/notas/a.txt"}'


_SIN_TOOL = object()


def _cuerpo(content: str = "", finish=None, args=_SIN_TOOL, con_args: bool = True) -> dict:
    """El cuerpo de un /chat/completions sin stream, como lo manda Z.ai."""
    message: dict = {"role": "assistant", "content": content}
    if args is not _SIN_TOOL or not con_args:
        fn: dict = {"name": "borrar"}
        if con_args:
            fn["arguments"] = args
        message["tool_calls"] = [{"id": "c1", "type": "function", "function": fn}]
    choice: dict = {"index": 0, "message": message}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}


class _ConnQueValidaHeaders(_Conn):
    """Como `_Conn`, pero los headers pasan por http.client de verdad (latin-1 y
    la validación de valores), que es donde reventaba una key con caracteres raros.
    `putrequest`/`putheader` solo arman el buffer: no tocan la red."""

    def request(self, method, url, body=None, headers=None) -> None:
        real = http.client.HTTPConnection("localhost", 1)
        real.putrequest(method, url)
        for name, value in (headers or {}).items():
            real.putheader(name, value)
        super().request(method, url, body=body, headers=headers)


def _glm(*cuerpos, api_key: str = "k") -> tuple[GLMClient, _Conn]:
    """GLMClient real sobre una conexión de mentiras con respuestas HTTP reales."""
    glm = GLMClient(api_key=api_key, base_url="http://localhost:1/v4")
    resps = [_con_content_length(json.dumps(c).encode()) for c in cuerpos]
    conn = _ConnQueValidaHeaders(resps)
    glm._transport._connect = lambda: conn
    return glm, conn


class _Local:
    model = "local"

    def __init__(self) -> None:
        self.llamadas = 0

    def chat(self, messages, tools=None):
        self.llamadas += 1
        return ChatResponse(content="Respuesta del local.")

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        if on_token is not None:
            on_token("Respuesta del local.")
        return ChatResponse(content="Respuesta del local.")


# --- 1. chat() sin stream: el fallo de la nube con 200 no es una respuesta ---

@pytest.mark.parametrize("cuerpo", [
    _cuerpo("", "network_error"),
    _cuerpo("Traes la mi", "network_error"),
    _cuerpo("", "network_error", args=_ARGS_A_MEDIAS),
], ids=["vacio", "media_frase", "tool_a_medias"])
def test_chat_finish_reason_network_error_es_un_fallo_transitorio(cuerpo):
    glm, _ = _glm(cuerpo)

    with pytest.raises(TransientLLMError, match="network_error"):
        glm.chat(_MSGS, tools=_TOOLS)


def test_chat_un_objeto_error_con_200_es_un_fallo_transitorio():
    glm, _ = _glm({"error": {"code": "1305", "message": "too many requests"}})

    # Antes: ChatResponse(content="") y el agente contestaba "Me quedé en blanco".
    with pytest.raises(TransientLLMError, match="too many requests"):
        glm.chat(_MSGS)


@pytest.mark.parametrize("reason", ["length", "sensitive", "model_context_window_exceeded",
                                    "algo_nuevo"])
def test_chat_corte_con_tool_call_a_medias_jamas_se_ejecuta_con_args_vacios(reason):
    glm, _ = _glm(_cuerpo("Va.", reason, args=_ARGS_A_MEDIAS))

    # Antes: tool_calls == [{"name": "borrar", "arguments": {}}] y la tool corría.
    with pytest.raises(TransientLLMError, match=reason):
        glm.chat(_MSGS, tools=_TOOLS)


@pytest.mark.parametrize("cuerpo", [
    _cuerpo("", "length", con_args=False),
    _cuerpo("", "length", args=""),
    _cuerpo("", "length", args=None),
], ids=["sin_arguments", "arguments_vacio", "arguments_null"])
def test_chat_corte_con_tool_call_sin_argumentos_tampoco_se_sabe_entero(cuerpo):
    glm, _ = _glm(cuerpo)

    with pytest.raises(TransientLLMError):
        glm.chat(_MSGS, tools=_TOOLS)


@pytest.mark.parametrize("args", [_ARGS_ENTEROS, {"ruta": "/home/user/notas/a.txt"}],
                         ids=["texto", "dict"])
def test_chat_corte_con_tool_call_entero_se_acepta_y_queda_en_el_log(args, caplog):
    glm, _ = _glm(_cuerpo("", "length", args=args))

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.glm"):
        resp = glm.chat(_MSGS, tools=_TOOLS)

    assert resp.tool_calls == [{"name": "borrar",
                                "arguments": {"ruta": "/home/user/notas/a.txt"}}]
    assert any("length" in r.getMessage() for r in caplog.records)


def test_chat_corte_con_texto_se_acepta_y_queda_en_el_log(caplog):
    glm, _ = _glm(_cuerpo("La capital de Francia", "length"))

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.glm"):
        resp = glm.chat(_MSGS)

    assert resp.content == "La capital de Francia" and not resp.tool_calls
    assert any("length" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("reason", ["stop", "tool_calls", None], ids=["stop", "tool_calls",
                                                                       "sin_finish_reason"])
def test_chat_fin_normal_no_cambia_ni_avisa(reason, caplog):
    """Un tool sin parámetros puede llegar con `arguments` vacío: con el fin
    normal (o un servidor que no manda finish_reason) se sigue aceptando como
    {} (lo de siempre), sin aviso."""
    glm, _ = _glm(_cuerpo(" Va. ", reason, args=""))

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.glm"):
        resp = glm.chat(_MSGS, tools=_TOOLS)

    assert resp.content == "Va."
    assert resp.tool_calls == [{"name": "borrar", "arguments": {}}]
    assert not caplog.records


def test_chat_tras_un_network_error_la_conexion_se_reutiliza():
    """El cuerpo se leyó entero: el fallo es del modelo, no del socket."""
    glm, conn = _glm(_cuerpo("", "network_error"), _cuerpo("Ya, patrón.", "stop"))

    with pytest.raises(TransientLLMError):
        glm.chat(_MSGS)
    assert glm.chat(_MSGS).content == "Ya, patrón."
    assert not conn.cerrada and len(conn.cuerpos) == 2


@pytest.mark.parametrize("cuerpo", [
    _cuerpo("", "network_error"),
    _cuerpo("Traes la mi", "network_error"),
    _cuerpo("", "length", args=_ARGS_A_MEDIAS),
    {"error": {"code": "1234", "message": "Network error, error id: x"}},
], ids=["network_error_vacio", "network_error_media_frase", "length_tool_a_medias",
        "error_obj_200"])
def test_de_punta_a_punta_sin_stream_el_fallo_de_la_nube_pasa_al_local(cuerpo):
    glm, _ = _glm(cuerpo)
    local = _Local()
    llm = FallbackLLM(glm, local)

    resp = llm.chat(_MSGS, tools=_TOOLS)

    assert local.llamadas == 1
    assert resp.content == "Respuesta del local." and not resp.tool_calls
    assert llm.degraded


def test_el_turno_del_loop_concurrente_lo_contesta_el_local():
    """Como BrainThread: handle_turn sin on_token, así que va por chat()."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    glm, _ = _glm(_cuerpo("", "network_error"))
    local = _Local()
    agent = ToolAgent(FallbackLLM(glm, local), Conversation("SYS"),
                      registry=default_registry(), guard=Guard(allowed_roots=[]))

    # Antes: "Me quedé en blanco, patrón. Repíteme eso." y el local ni se llamaba.
    assert agent.handle_turn("qué hora es") == "Respuesta del local."
    assert local.llamadas == 1


# --- 2. Una key que no cabe en un header es una credencial mala, no un crash ---

_KEYS_RARAS = ["“abc”", "abc​", "abc\ndef", "abc\tdef", "abcé"]
_IDS_RARAS = ["comillas_tipograficas", "espacio_invisible", "salto_de_linea", "tabulador",
              "no_ascii"]


@pytest.mark.parametrize("key", _KEYS_RARAS, ids=_IDS_RARAS)
@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_key_con_caracteres_raros_es_glmautherror_sin_tocar_la_red(key, stream):
    glm, conn = _glm(_cuerpo("hola", "stop"), api_key=key)

    # Antes: UnicodeEncodeError ('latin-1' codec) o ValueError ("Invalid header
    # value") crudos de http.client, en cada turno.
    with pytest.raises(GLMAuthError, match="API key") as info:
        if stream:
            glm.chat_stream(_MSGS)
        else:
            glm.chat(_MSGS)
    assert key not in str(info.value)  # la key no acaba en el log
    assert conn.cuerpos == []


@pytest.mark.parametrize("key", _KEYS_RARAS, ids=_IDS_RARAS)
@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_key_con_caracteres_raros_pasa_al_local(key, stream):
    glm, _ = _glm(api_key=key)
    local = _Local()
    llm = FallbackLLM(glm, local)

    for _ in range(2):  # y en los turnos siguientes ni se vuelve a intentar la nube
        if stream:
            resp = llm.chat_stream(_MSGS, on_token=lambda _t: None)
        else:
            resp = llm.chat(_MSGS)
        assert resp.content == "Respuesta del local."
    assert local.llamadas == 2 and llm.degraded


@pytest.mark.parametrize("key", _KEYS_RARAS, ids=_IDS_RARAS)
def test_preconnect_con_key_rara_no_abre_nada(key):
    """Como sin key: la petición no saldría, no hay handshake que adelantar."""
    glm = GLMClient(api_key=key, base_url="http://localhost:1/v4")
    abiertas: list[object] = []
    glm._transport._connect = lambda: abiertas.append(1)  # type: ignore[assignment]

    assert glm.preconnect() is False
    assert abiertas == []


def test_key_ascii_normal_sigue_igual():
    glm, conn = _glm(_cuerpo("hola", "stop"), api_key="0123abcd.EfGh-_ij")

    assert glm.chat(_MSGS).content == "hola"
    assert len(conn.cuerpos) == 1


def test_key_ascii_normal_en_stream_sigue_igual():
    glm = GLMClient(api_key="0123abcd.EfGh-_ij", base_url="http://localhost:1/v4")
    conn = _Conn([_stream_entero([
        b'data: {"choices":[{"delta":{"content":"hola"},"finish_reason":"stop"}]}\n',
        b"data: [DONE]\n",
    ])])
    glm._transport._connect = lambda: conn

    assert glm.chat_stream(_MSGS).content == "hola"


def test_key_rara_no_cambia_el_error_permanente_de_otros_llmerror():
    """La key rara es GLMAuthError (subclase de LLMError): el agente la contesta
    en personaje como cualquier fallo del motor, nunca un traceback."""
    glm, _ = _glm(api_key="“abc”")
    with pytest.raises(LLMError):
        glm.chat(_MSGS)


# --- 3. meta_preamble_cleaner no toca respuestas normales ---

@pytest.mark.parametrize("respuesta", [
    "Aquí el resultado final fue 3 a 1 para el América.",
    "Te resumo el resultado de la búsqueda: encontré 3 archivos.",
    "Aquí te va el resultado de la búsqueda: encontré 3 archivos.",
    "Te resumo el resultado del partido: ganó el América.",
    "Aquí el resultado del partido: ganó el América 3 a 1.",
    "Aquí va el resultado de tu búsqueda, patrón: tres archivos.",
    "La herramienta de volumen no acepta parámetros negativos. Dime un número de 0 a 100.",
    "La herramienta de búsqueda no acepta más de 10 parámetros, patrón. Pásame menos filtros.",
])
def test_cleaner_no_recorta_respuestas_normales(respuesta):
    assert meta_preamble_cleaner(respuesta) == respuesta


@pytest.mark.parametrize("respuesta, limpia", [
    ("Te resumo el resultado: todo bien.", "todo bien."),
    ("Te resumo el resultado. Todo bien.", "Todo bien."),
    ("Resumo el resultado - todo bien.", "todo bien."),
    ("Aquí va el resultado: 42", "42"),
    ("Aquí te va el resultado: 42", "42"),
    ("aqui el resultado: 42", "42"),
    ("Parece que la herramienta 'ram_usage' no acepta parámetros. Te resumo el "
     "resultado: todo bien, patrón.", "todo bien, patrón."),
    ("Patrón, parece que la herramienta 'x' no acepta ningún parámetro! Tienes 7 GB.",
     "Tienes 7 GB."),
    ("Parece que la herramienta 'ram_usage' no acepta parámetros\nTe resumo el "
     "resultado: todo bien.", "todo bien."),
])
def test_cleaner_sigue_quitando_los_preambulos_meta(respuesta, limpia):
    assert meta_preamble_cleaner(respuesta) == limpia
