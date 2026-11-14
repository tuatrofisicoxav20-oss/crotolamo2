"""Arreglos de la auditoría en los clientes LLM y su transporte HTTP.

Cada bloque fija un hallazgo confirmado; ni un byte de red real: conexiones
falsas inyectadas en `HTTPTransport._connect`.

- El cuerpo de la petición se codifica SIEMPRE: un sustituto suelto en el
  historial (el nombre de un archivo con bytes no-UTF-8 que devolvió una tool,
  un "\\ud83d" suelto de un JSON ajeno) no puede tumbar el turno, y con él
  todos los siguientes mientras ese mensaje siga en el historial.
- "No reintentar si ya se habló" (FallbackLLM) mira lo que de verdad SALIÓ
  hacia el patrón, no cuántas veces se llamó a on_token: con tools a la vista
  el LiveStreamer lo retiene todo, y un corte a media respuesta dejaba el turno
  en el error de la nube sin probar el modelo local aunque no hubiera sonado nada.
- Un cuerpo CORTADO a medias (http.client.IncompleteRead, que es HTTPException
  y no OSError) o ilegible (no-UTF-8) ya no sale como excepción cruda de
  chat()/chat_stream(): es un error en personaje, y el corte abre el breaker.
- El knob [llm.glm].max_tokens llega al payload (y sin él no se manda nada).
- Si la nube se cae a MEDIA frase ya hablada, FallbackLLM no reintenta y el
  error llega igual a on_token: se avisa en vez de dejar la frase en silencio.
- El stream de Ollama entrega el tool_call nativo en un chunk y luego cierra con
  un message vacío: el parser junta el turno entero en vez de quedarse con ese
  último chunk, que borraba la llamada (la tool no corría).
- La coerción de tool-calls escritos como texto no revienta con un JSON de datos
  cuyo "name" es un objeto, lee los argumentos de "parameters" o de un string
  JSON, y saca TODAS las llamadas cuando vienen como objetos seguidos.
- Un stream con un `error` dentro, sin el cierre del motor ([DONE]/finish_reason,
  done=true) y sin nada usable, o un 3xx ya no pasa por una respuesta buena
  (vacía, o con un tool_call a medio llegar que se ejecutaba con {}). Con texto
  ya llegado, el final limpio sin cierre se acepta avisando en el log (criterio
  híbrido; ver test_transport.py).
"""

from __future__ import annotations

import http.client
import io
import json
import os

import pytest

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.glm import GLMAuthError, GLMClient
from crotolamo.core.llm import ChatResponse, LLMClient, LLMError, TransientLLMError
from crotolamo.core.sse import consume_sse
from crotolamo.core.streaming import LiveStreamer
from crotolamo.core.tool_parsing import coerce_text_tool_calls

# --- dobles de prueba ---


class _Resp:
    """Respuesta HTTP de mentiras con la superficie que usan los clientes."""

    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self.status = status
        self._body = body
        self._leido = False

    def read(self) -> bytes:
        self._leido = True
        body, self._body = self._body, b""
        return body

    def getheader(self, name, default=None):
        return default

    def isclosed(self) -> bool:
        return self._leido


class _Conn:
    """Conexión http.client de mentiras que GUARDA el cuerpo de cada petición."""

    def __init__(self, respuestas) -> None:
        self.respuestas = list(respuestas)
        self.cuerpos: list[bytes] = []
        self.cerrada = False

    def request(self, method, url, body=None, headers=None) -> None:
        self.cuerpos.append(body)

    def getresponse(self):
        return self.respuestas.pop(0)

    def close(self) -> None:
        self.cerrada = True


# --- core-llm#5: sustitutos sueltos en el payload ---

# Así nombra Python un archivo cuyo nombre trae bytes que no son UTF-8 (zip
# viejo de Windows, memoria FAT): el byte 0xF3 queda como el sustituto U+DCF3.
_NOMBRE_NO_UTF8 = os.fsdecode(b"canci\xf3n.mp3")
# Y así llega medio emoji escapado en el JSON de un servidor MCP o de una web.
_MEDIO_EMOJI = json.loads('"\\ud83d"')


@pytest.mark.parametrize("roto", [_NOMBRE_NO_UTF8, _MEDIO_EMOJI])
def test_post_json_codifica_aunque_el_payload_traiga_un_sustituto_suelto(monkeypatch, roto):
    t = HTTPTransport("http://localhost:11434", timeout=5)
    conn = _Conn([_Resp()])
    monkeypatch.setattr(t, "_connect", lambda: conn)

    t.post_json("/api/chat", {"content": f"Contenido, patrón:\n📄 {roto} fin"})

    # Antes: UnicodeEncodeError ("surrogates not allowed") sin tocar la red.
    enviado = json.loads(conn.cuerpos[0].decode("utf-8"))["content"]  # UTF-8 y JSON válidos
    # Solo se pierde lo irrepresentable; los acentos y el emoji de al lado, intactos.
    assert enviado.startswith("Contenido, patrón:\n📄 ")
    assert enviado.endswith(" fin")
    assert "\udcf3" not in enviado and "\ud83d" not in enviado
    if roto is _NOMBRE_NO_UTF8:
        assert "canci?n.mp3" in enviado


def test_post_json_sin_sustitutos_manda_los_mismos_bytes_de_siempre(monkeypatch):
    """El arreglo no toca el caso normal: mismo cuerpo, byte a byte."""
    t = HTTPTransport("http://localhost:11434", timeout=5)
    conn = _Conn([_Resp()])
    monkeypatch.setattr(t, "_connect", lambda: conn)
    payload = {"messages": [{"role": "user", "content": "¿Qué hay, patrón? 📄 ñandú"}]}

    t.post_json("/api/chat", payload)

    assert conn.cuerpos[0] == json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _historial_envenenado() -> list[dict]:
    """Un turno ya cerrado cuyo resultado de tool trae el nombre no-UTF-8."""
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "qué hay en la carpeta"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "list_dir", "arguments": {"path": "musica"}}}]},
        {"role": "tool", "name": "list_dir",
         "content": f"Contenido de musica, patrón:\n📄 {_NOMBRE_NO_UTF8}"},
        {"role": "user", "content": "hola"},
    ]


def test_glm_sigue_respondiendo_con_un_nombre_no_utf8_en_el_historial(monkeypatch):
    cuerpo = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hola"}}]})
    client = GLMClient(api_key="k", base_url="http://localhost:1/v4")
    conn = _Conn([_Resp(cuerpo.encode()), _Resp(cuerpo.encode())])
    monkeypatch.setattr(client._transport, "_connect", lambda: conn)

    # El mensaje envenenado sigue en el historial turno tras turno: ninguno revienta.
    assert client.chat(_historial_envenenado()).content == "hola"
    assert client.chat(_historial_envenenado()).content == "hola"
    resultado = json.loads(conn.cuerpos[1].decode("utf-8"))["messages"][3]["content"]
    assert resultado == "Contenido de musica, patrón:\n📄 canci?n.mp3"


def test_ollama_sigue_respondiendo_con_un_nombre_no_utf8_en_el_historial(monkeypatch):
    cuerpo = json.dumps({"message": {"role": "assistant", "content": "hola"}})
    client = LLMClient(host="http://localhost:1")
    conn = _Conn([_Resp(cuerpo.encode())])
    monkeypatch.setattr(client._transport, "_connect", lambda: conn)

    assert client.chat(_historial_envenenado()).content == "hola"
    json.loads(conn.cuerpos[0].decode("utf-8"))  # lo enviado es UTF-8 y JSON válidos


# --- core-llm#7 / core-agent#7: "ya se habló" solo si de verdad salió algo ---


class _Motor:
    """Motor de mentiras para `chat_stream`: suelta `tokens` por on_token y luego
    revienta con `error`, o devuelve lo que soltó si no hay error."""

    def __init__(self, model: str, tokens: tuple[str, ...] = (),
                 error: Exception | None = None) -> None:
        self.model = model
        self.tokens = tokens
        self.error = error
        self.llamadas = 0

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        for token in self.tokens:
            if on_token is not None:
                on_token(token)
        if self.error is not None:
            raise self.error
        return ChatResponse(content="".join(self.tokens))


_CORTES = [
    TransientLLMError("GLM no respondió a tiempo, patrón. (timed out)"),
    GLMAuthError("GLM me rechazó la credencial (401), patrón."),
]


def test_feed_dice_si_el_trozo_salio_o_quedo_retenido():
    """El contrato que consulta FallbackLLM: False = "lo retuve, no sonó"."""
    dichos: list[str] = []

    retiene = LiveStreamer(dichos.append, hold_until_done=True)  # tools a la vista
    assert retiene.feed("Voy a pausar la música. ") is False
    json_crudo = LiveStreamer(dichos.append)                     # tool-call en content
    assert json_crudo.feed('{"name": "open_url"') is False
    assert dichos == []

    en_vivo = LiveStreamer(dichos.append)                        # charla: se habla ya
    assert en_vivo.feed("Hola ") is True
    assert en_vivo.feed("patrón") is True
    assert dichos == ["Hola ", "patrón"]


@pytest.mark.parametrize("corte", _CORTES)
def test_fallback_usa_el_local_si_el_streamer_lo_retuvo_todo(corte):
    nube = _Motor("glm", tokens=("¡Claro que sí, ",), error=corte)
    local = _Motor("ollama", tokens=("Respuesta ", "del local, patrón."))
    dichos: list[str] = []
    streamer = LiveStreamer(dichos.append, hold_until_done=True)
    llm = FallbackLLM(nube, local, cooldown_s=300)

    # Antes: el primer delta marcaba "ya habló" y el corte se propagaba.
    respuesta = llm.chat_stream([], tools=[{"type": "function"}], on_token=streamer.feed)

    assert respuesta.content == "Respuesta del local, patrón."
    assert (nube.llamadas, local.llamadas) == (1, 1)
    assert llm.degraded                    # el breaker se abrió igual
    assert dichos == []                    # sigue retenido: nada sonó a medias
    streamer.flush_if_held(respuesta.content)
    assert dichos == ["Respuesta del local, patrón."]   # y el principio no se repite


def test_fallback_sin_on_token_no_cuenta_los_deltas_como_habla():
    """Si nadie escucha el stream no hay nada que duplicar: se usa el local."""
    nube = _Motor("glm", tokens=("¡Claro ",), error=_CORTES[0])
    local = _Motor("ollama", tokens=("soy el local",))

    assert FallbackLLM(nube, local).chat_stream([]).content == "soy el local"
    assert local.llamadas == 1


def test_fallback_sigue_sin_reintentar_si_el_streamer_ya_solto_texto():
    """La regla original intacta: lo que ya sonó no se repite con el local."""
    nube = _Motor("glm", tokens=("¡Claro que sí, ", "patrón"), error=_CORTES[0])
    local = _Motor("ollama", tokens=("hola",))
    dichos: list[str] = []
    streamer = LiveStreamer(dichos.append)  # sin tools: charla en vivo

    with pytest.raises(LLMError, match="a tiempo"):
        FallbackLLM(nube, local).chat_stream([], on_token=streamer.feed)

    assert dichos == ["¡Claro que sí, ", "patrón"]
    assert local.llamadas == 0


def test_fallback_no_reintenta_con_un_trozo_que_el_streamer_aun_va_a_soltar():
    """Un trozo de 1 carácter queda en el buffer SIN decidir y saldrá pegado al
    siguiente: reintentar con el local haría sonar "HHola". Cuenta como hablado."""
    nube = _Motor("glm", tokens=("H",), error=_CORTES[0])
    local = _Motor("ollama", tokens=("Hola, patrón.",))
    dichos: list[str] = []
    streamer = LiveStreamer(dichos.append)

    with pytest.raises(LLMError):
        FallbackLLM(nube, local).chat_stream([], on_token=streamer.feed)

    assert local.llamadas == 0
    assert dichos == []


def test_toolagent_en_streaming_responde_el_local_si_la_nube_cae_a_media_respuesta():
    """El turno completo, como en voz con GLM de cerebro: todas las tools a la
    vista (el agente retiene), la nube suelta el principio y se cae el wifi."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    nube = _Motor("glm", tokens=("¡Claro que sí, ",), error=_CORTES[0])
    local = _Motor("ollama", tokens=("Respuesta del local, patrón.",))
    agent = ToolAgent(
        FallbackLLM(nube, local), Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), fastpath=False,
    )
    dichos: list[str] = []

    reply = agent.handle_turn("hola", on_token=dichos.append)

    # Antes: reply era "GLM no respondió a tiempo, patrón." y local.llamadas == 0.
    assert reply == "Respuesta del local, patrón."
    assert dichos == ["Respuesta del local, patrón."]
    assert local.llamadas == 1


# --- core-agent#10 / core-llm#4 (C1): cuerpo cortado o ilegible ---


class _Zocalo:
    """Socket de mentiras del que http.client lee una respuesta cruda."""

    def __init__(self, crudo: bytes) -> None:
        self._crudo = crudo

    def makefile(self, mode):
        return io.BufferedReader(io.BytesIO(self._crudo))


def _respuesta_real(crudo: bytes) -> http.client.HTTPResponse:
    """Un HTTPResponse DE VERDAD sobre bytes crudos: el framing, y con él el
    IncompleteRead, lo pone http.client y no un doble que imite la excepción."""
    resp = http.client.HTTPResponse(_Zocalo(crudo))
    resp.begin()
    return resp


_CABECERA = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
_MSGS = [{"role": "user", "content": "hola"}]
# La misma primera línea de stream en el dialecto de cada motor.
_LINEA = {
    "glm": b'data: {"choices":[{"delta":{"content":"Hola "}}]}\n',
    "ollama": b'{"message":{"role":"assistant","content":"Hola "},"done":false}\n',
}
# Lo que contesta el portal cautivo de un wifi ajeno: HTML en latin-1 con 200.
_PORTAL = "<html><body>Inicie sesión para navegar</body></html>\n".encode("latin-1")


def _con_content_length(cuerpo: bytes, declarado: int | None = None) -> http.client.HTTPResponse:
    largo = len(cuerpo) if declarado is None else declarado
    return _respuesta_real(_CABECERA + f"Content-Length: {largo}\r\n\r\n".encode() + cuerpo)


def _cuerpo_cortado() -> http.client.HTTPResponse:
    """Promete 500 bytes, llegan 19 y el servidor (o un proxy) cierra."""
    return _con_content_length(b'{"choices":[{"messa', declarado=500)


def _stream_cortado(linea: bytes) -> http.client.HTTPResponse:
    """Chunked: una línea entera y la siguiente cortada a MEDIO chunk."""
    chunk = f"{len(linea):x}\r\n".encode() + linea + b"\r\n"
    return _respuesta_real(
        _CABECERA + b"Transfer-Encoding: chunked\r\n\r\n" + chunk + chunk[: len(chunk) // 2]
    )


def _cliente(motor: str, monkeypatch, respuesta):
    client = (
        GLMClient(api_key="k", base_url="http://localhost:1/v4") if motor == "glm"
        else LLMClient(host="http://localhost:1")
    )
    conn = _Conn([respuesta])
    monkeypatch.setattr(client._transport, "_connect", lambda: conn)
    return client


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_chat_con_el_cuerpo_cortado_es_un_fallo_transitorio(monkeypatch, motor):
    client = _cliente(motor, monkeypatch, _cuerpo_cortado())

    # Antes: http.client.IncompleteRead cruda, sin mensaje en personaje.
    with pytest.raises(TransientLLMError, match="a medias, patrón"):
        client.chat(_MSGS)


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_chat_stream_cortado_a_medio_chunk_es_un_fallo_transitorio(monkeypatch, motor):
    client = _cliente(motor, monkeypatch, _stream_cortado(_LINEA[motor]))
    dichos: list[str] = []

    with pytest.raises(TransientLLMError, match="a medias, patrón"):
        client.chat_stream(_MSGS, on_token=dichos.append)

    assert dichos == ["Hola "]  # lo que llegó antes del corte sí salió


def test_fallback_responde_el_local_si_a_la_nube_le_cortan_el_cuerpo(monkeypatch):
    """La consecuencia que importaba: el corte es de DISPONIBILIDAD, así que abre
    el breaker y contesta el modelo local en vez de reventar el turno."""

    class _Local:
        model = "ollama"

        def chat(self, messages, tools=None):
            return ChatResponse(content="soy el local")

    llm = FallbackLLM(_cliente("glm", monkeypatch, _cuerpo_cortado()), _Local(), cooldown_s=300)

    assert llm.chat(_MSGS).content == "soy el local"
    assert llm.degraded


def test_el_turno_no_revienta_si_cortan_la_respuesta(monkeypatch):
    from crotolamo.core.agent import Agent
    from crotolamo.core.memory import Conversation

    agent = Agent(_cliente("ollama", monkeypatch, _cuerpo_cortado()), Conversation("SYS"))

    # Antes: la excepción cruda salía de handle_turn (traceback en el shell).
    assert "a medias, patrón" in agent.handle_turn("hola")


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_chat_con_cuerpo_no_utf8_es_basura_no_json(monkeypatch, motor):
    """UnicodeDecodeError es ValueError pero NO JSONDecodeError: se escapaba cruda.
    Es la misma basura que un HTML en UTF-8, y sale con el mismo error."""
    client = _cliente(motor, monkeypatch, _con_content_length(_PORTAL))

    with pytest.raises(LLMError, match="basura no-JSON") as info:
        client.chat(_MSGS)

    assert type(info.value) is LLMError  # permanente: no abre el breaker


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_chat_stream_salta_una_linea_no_utf8_como_cualquier_otra_basura(monkeypatch, motor):
    """Los parsers ya ignoran las líneas que no son JSON; una que además no es
    UTF-8 reventaba en el .decode() antes de llegar a ese filtro."""
    # Con el cierre de cada motor: un stream sin él es un corte (ver core-llm#3).
    cuerpo = _PORTAL + _LINEA[motor] + _CIERRE[motor]
    client = _cliente(motor, monkeypatch, _con_content_length(cuerpo))
    dichos: list[str] = []

    assert client.chat_stream(_MSGS, on_token=dichos.append).content == "Hola"
    assert dichos == ["Hola "]


# --- core-llm#16: [llm.glm].max_tokens, el knob que la config no documentaba ---


def _settings_glm(**glm):
    from types import SimpleNamespace

    return SimpleNamespace(llm={"glm": glm})


def test_max_tokens_de_la_config_llega_al_payload(monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "k")
    con_tope = GLMClient.from_settings(_settings_glm(max_tokens=512))
    sin_tope = GLMClient.from_settings(_settings_glm())

    assert con_tope._payload(_MSGS, None, stream=False)["max_tokens"] == 512
    # Sin la clave (el default) no se manda: decide el tope del propio modelo.
    assert "max_tokens" not in sin_tope._payload(_MSGS, None, stream=True)


# --- core-llm#8 (C2): el corte a media frase se avisa, no deja silencio ---


def test_corte_de_la_nube_a_media_frase_se_avisa_por_on_token():
    """Charla sin tools (se habla en vivo): la nube suelta media frase y se cae.
    FallbackLLM NO reintenta (repetiría el principio) y propaga el error; ese
    error tiene que llegar a on_token, que es lo único que oye la voz."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    nube = _Motor("glm", tokens=("Mira, patrón, ", "la capital de Francia es "), error=_CORTES[0])
    local = _Motor("ollama", tokens=("hola",))
    agent = ToolAgent(
        FallbackLLM(nube, local), Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), fastpath=False, route_fn=lambda texto: [],
    )
    dichos: list[str] = []

    reply = agent.handle_turn("capital de francia", on_token=dichos.append)

    assert local.llamadas == 0  # lo ya dicho no se repite con el local
    assert reply == str(_CORTES[0])
    hablado = "".join(dichos)
    assert hablado.startswith("Mira, patrón, la capital de Francia es ")
    assert hablado.endswith(reply)  # antes: la frase truncada y silencio


# --- core-llm#1 / core-agent#1: tool_calls nativos de Ollama en streaming ---


def _linea_ollama(message: dict, done: bool = False) -> bytes:
    chunk = {"model": "qwen2.5-coder:7b", "message": message, "done": done}
    return json.dumps(chunk).encode() + b"\n"


_PAUSA = {"function": {"name": "pausar_musica", "arguments": {"accion": "pause"}}}
_RAM = {"function": {"name": "ram_usage", "arguments": {}}}
# El chunk con el que Ollama CIERRA todo stream: message vacío, sin tool_calls.
_FIN_OLLAMA = _linea_ollama({"role": "assistant", "content": ""}, done=True)
# La secuencia que documenta Ollama para /api/chat con stream=true y tools: el
# tool_call llega en un chunk con done=false y después viene el cierre vacío.
_STREAM_CON_TOOL = [
    _linea_ollama({"role": "assistant", "content": "", "tool_calls": [_PAUSA]}),
    _FIN_OLLAMA,
]


def _stream_entero(lineas: list[bytes]) -> http.client.HTTPResponse:
    """Chunked y completo: un chunk por línea más el chunk final de tamaño 0."""
    chunks = b"".join(f"{len(linea):x}\r\n".encode() + linea + b"\r\n" for linea in lineas)
    return _respuesta_real(
        _CABECERA + b"Transfer-Encoding: chunked\r\n\r\n" + chunks + b"0\r\n\r\n"
    )


def test_consume_stream_no_pierde_el_tool_call_cuando_llega_el_cierre_vacio():
    content, message = LLMClient._consume_stream(iter(_STREAM_CON_TOOL), None)

    # Antes: message == {"role": "assistant", "content": ""} (el del cierre).
    assert content == ""
    assert message["tool_calls"] == [_PAUSA]


def test_consume_stream_junta_el_texto_y_los_tool_calls_de_varios_chunks():
    lineas = [
        _linea_ollama({"role": "assistant", "content": "Va, "}),
        _linea_ollama({"role": "assistant", "content": "patrón."}),
        _linea_ollama({"role": "assistant", "content": "", "tool_calls": [_PAUSA]}),
        _linea_ollama({"role": "assistant", "content": "", "tool_calls": [_RAM]}),
        _FIN_OLLAMA,
    ]
    dichos: list[str] = []

    content, message = LLMClient._consume_stream(iter(lineas), dichos.append)

    assert dichos == ["Va, ", "patrón."]
    # El mensaje es el del turno ENTERO, no el del último chunk: todo el texto y
    # todas las llamadas, en el orden en que llegaron.
    assert message == {
        "role": "assistant", "content": "Va, patrón.", "tool_calls": [_PAUSA, _RAM],
    }
    assert content == "Va, patrón."


def test_consume_stream_sin_tool_calls_no_inventa_la_clave():
    """Charla normal: el agente decide "nativo o no" por `tool_calls`; una lista
    vacía de más no cambia nada, pero la clave solo va si hubo llamadas (igual
    que en `consume_sse`)."""
    lineas = [_linea_ollama({"role": "assistant", "content": "Hola"}), _FIN_OLLAMA]

    _, message = LLMClient._consume_stream(iter(lineas), None)

    assert message == {"role": "assistant", "content": "Hola"}


def test_chat_stream_de_ollama_devuelve_el_tool_call_nativo(monkeypatch):
    client = _cliente("ollama", monkeypatch, _stream_entero(_STREAM_CON_TOOL))

    respuesta = client.chat_stream(_MSGS, tools=[{"type": "function"}])

    # Antes: tool_calls == [] y raw_message sin la llamada.
    assert respuesta.tool_calls == [{"name": "pausar_musica", "arguments": {"accion": "pause"}}]
    assert respuesta.raw_message["tool_calls"] == [_PAUSA]  # formato Ollama, para el historial


def test_el_turno_en_streaming_con_ollama_ejecuta_la_tool_que_pidio_el_modelo(monkeypatch):
    """De punta a punta, como la voz con stream_speak o `shell --stream` sobre el
    modelo local: el modelo pide la tool por el campo nativo y luego redacta."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry, Tool

    ejecutadas: list[str] = []

    def pausar_musica(accion: str = "pause") -> str:
        ejecutadas.append(accion)
        return "Música en pausa."

    registry = Registry()
    registry.register(Tool(
        name="pausar_musica", func=pausar_musica, description="Pausa la música.",
        parameters={"type": "object", "properties": {"accion": {"type": "string"}}, "required": []},
    ))
    client = LLMClient(host="http://localhost:1")
    conn = _Conn([
        _stream_entero(_STREAM_CON_TOOL),
        _stream_entero([_linea_ollama({"role": "assistant", "content": "Pausada, patrón."}),
                        _FIN_OLLAMA]),
    ])
    monkeypatch.setattr(client._transport, "_connect", lambda: conn)
    agent = ToolAgent(client, Conversation("SYS"), registry=registry,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    dichos: list[str] = []

    reply = agent.handle_turn("pausa la música", on_token=dichos.append)

    # Antes: la tool NO corría y el turno se cerraba en falso con la primera
    # respuesta (vacía), sin llegar a la segunda llamada.
    assert ejecutadas == ["pause"]
    assert reply == "Pausada, patrón."
    assert "".join(dichos) == "Pausada, patrón."
    # La segunda petición lleva la llamada y su resultado, como espera Ollama
    # (con el id estable que el agente le pone a una llamada que llega sin id).
    reenviado = json.loads(conn.cuerpos[1].decode("utf-8"))["messages"]
    assert {"role": "assistant", "content": "",
            "tool_calls": [{**_PAUSA, "id": "call_t1_0"}]} in reenviado
    assert reenviado[-1]["role"] == "tool"
    assert reenviado[-1]["content"] == "Música en pausa."


# --- core-llm#10: JSON de DATOS cuyo "name"/"tool" no es un texto ---

_TOOLS = {"music_control", "list_processes", "open_url", "ram_usage", "disk_usage"}
_JSON_DE_DATOS = [
    '```json\n{"name": {"first": "Ana", "last": "Ruiz"}, "edad": 30}\n```',
    '{"tool": ["martillo", "sierra"]}',
]


@pytest.mark.parametrize("contenido", _JSON_DE_DATOS)
def test_coerce_no_revienta_si_el_name_del_json_no_es_un_texto(contenido):
    # Antes: TypeError (unhashable type) al buscar un dict/lista en el set de tools.
    assert coerce_text_tool_calls(contenido, _TOOLS) == []


class _Guion:
    """LLM de mentiras sin streaming: devuelve las respuestas en orden."""

    model = "guion"

    def __init__(self, *respuestas: ChatResponse) -> None:
        self.respuestas = list(respuestas)

    def chat(self, messages, tools=None):
        return self.respuestas.pop(0)


def _agente_con_tools(llm, **tools):
    """ToolAgent sobre un registry de juguete: una tool por kwarg (nombre=func)."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry, Tool

    registry = Registry()
    for nombre, func in tools.items():
        registry.register(Tool(
            name=nombre, func=func, description=nombre,
            parameters={"type": "object", "properties": {"accion": {"type": "string"}},
                        "required": []},
        ))
    return ToolAgent(llm, Conversation("SYS"), registry=registry,
                     guard=Guard(allowed_roots=[]), fastpath=False)


@pytest.mark.parametrize("contenido", _JSON_DE_DATOS)
def test_el_turno_responde_el_json_de_datos_en_vez_de_reventar(contenido):
    agent = _agente_con_tools(_Guion(ChatResponse(content=contenido)), ram_usage=lambda: "8 GB")

    # Antes: la TypeError salía cruda de handle_turn (traceback en el shell) y el
    # mensaje del patrón quedaba en el historial sin respuesta.
    assert agent.handle_turn("dame los datos en json") == contenido


# --- core-llm#11: argumentos en "parameters" o como string, y varias llamadas ---


def test_coerce_lee_los_argumentos_que_vienen_en_parameters():
    """El formato de texto de Llama 3.x; `strip_leaked_tool_json` ya lo reconoce
    como tool-call, pero aquí se ejecutaba con {} (la tool corría con defaults)."""
    contenido = '{"name": "music_control", "parameters": {"action": "next"}}'

    assert coerce_text_tool_calls(contenido, _TOOLS) == [
        {"name": "music_control", "arguments": {"action": "next"}},
    ]


def test_coerce_prefiere_arguments_si_vienen_las_dos_claves():
    contenido = '{"name": "open_url", "arguments": {"url": "a.com"}, "parameters": {"url": "b"}}'

    assert coerce_text_tool_calls(contenido, _TOOLS)[0]["arguments"] == {"url": "a.com"}


def test_coerce_acepta_arguments_como_string_json():
    """Así los manda el contrato OpenAI; antes la llamada entera se descartaba."""
    con_args = '{"name": "open_url", "arguments": "{\\"url\\": \\"x.com\\"}"}'
    sin_args = '{"name": "ram_usage", "arguments": ""}'

    assert coerce_text_tool_calls(con_args, _TOOLS) == [
        {"name": "open_url", "arguments": {"url": "x.com"}},
    ]
    assert coerce_text_tool_calls(sin_args, _TOOLS) == [{"name": "ram_usage", "arguments": {}}]


@pytest.mark.parametrize("basura", ['"x.com"', '"[1, 2]"', "null", "7"])
def test_coerce_no_inventa_argumentos_vacios_si_no_los_entiende(basura):
    """Un `arguments` ilegible NO se degrada a {}: la tool correría con sus
    defaults, que no es lo que pidió el modelo. Se sigue sin ejecutar nada."""
    contenido = '{"name": "music_control", "arguments": %s}' % basura

    assert coerce_text_tool_calls(contenido, _TOOLS) == []


_DOS_LLAMADAS = {
    "una por línea": (
        '{"name": "ram_usage", "arguments": {}}\n'
        '{"name": "music_control", "arguments": {"action": "pause"}}'
    ),
    "etiquetas de qwen": (
        '<tool_call>\n{"name": "ram_usage", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "music_control", "arguments": {"action": "pause"}}\n</tool_call>'
    ),
    "separadas por ; (llama 3.1)": (
        '{"name": "ram_usage", "parameters": {}}; '
        '{"name": "music_control", "parameters": {"action": "pause"}}'
    ),
}


@pytest.mark.parametrize("contenido", _DOS_LLAMADAS.values(), ids=_DOS_LLAMADAS.keys())
def test_coerce_saca_todas_las_llamadas_escritas_como_objetos_seguidos(contenido):
    # Antes: [] (del primer `{` al último `}` no hay UN documento JSON).
    assert coerce_text_tool_calls(contenido, _TOOLS) == [
        {"name": "ram_usage", "arguments": {}},
        {"name": "music_control", "arguments": {"action": "pause"}},
    ]


@pytest.mark.parametrize("contenido", [
    # Prosa con llaves que no son JSON delante de una llamada bien formada.
    'Un dict se escribe {clave: valor}. {"name": "ram_usage", "arguments": {}}',
    # Objeto cortado: lo de dentro está entero, pero el documento no cerró.
    '{"calls": [{"name": "ram_usage", "arguments": {}}], "motivo": ',
    'En Python una lista es [1, 2] y un dict {a: 1}.',
])
def test_coerce_sigue_sin_ejecutar_nada_si_algun_bloque_no_es_json(contenido):
    """Lo que ya valía y no cambia: sacar varias llamadas no es pescar la primera
    llave que parsee dentro de un texto que no es una lista limpia de llamadas."""
    assert coerce_text_tool_calls(contenido, _TOOLS) == []


def test_el_turno_ejecuta_las_dos_tools_que_el_modelo_escribio_seguidas():
    ejecutadas: list[str] = []

    def ram_usage() -> str:
        ejecutadas.append("ram")
        return "8 GB libres."

    def music_control(accion: str = "toggle") -> str:
        ejecutadas.append(f"musica:{accion}")
        return "Hecho."

    llm = _Guion(
        ChatResponse(content='{"name": "ram_usage", "arguments": {}}\n'
                             '{"name": "music_control", "parameters": {"accion": "pause"}}'),
        ChatResponse(content="Tienes 8 GB libres y pausé la música, patrón."),
    )
    agent = _agente_con_tools(llm, ram_usage=ram_usage, music_control=music_control)

    reply = agent.handle_turn("dime la ram y pausa la música")

    # Antes: no corría ninguna y al patrón le llegaba el primer JSON crudo.
    assert ejecutadas == ["ram", "musica:pause"]
    assert reply == "Tienes 8 GB libres y pausé la música, patrón."


# --- core-llm#3: un stream con error dentro, cortado o redirigido NO es una respuesta ---

_SSE_OK = {"Content-Type": "text/event-stream"}
# El chunk con el que cada motor da el turno por TERMINADO.
_CIERRE = {"glm": b"data: [DONE]\n", "ollama": _FIN_OLLAMA}


def _dato(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _stream_sin_cierre_http(lineas: list[bytes]) -> http.client.HTTPResponse:
    """Chunked al que le cierran el socket justo ENTRE dos chunks (donde cae casi
    cualquier corte: cada evento viaja en su chunk). Iterando por líneas,
    http.client NO lanza IncompleteRead aquí: el stream simplemente se acaba."""
    chunks = b"".join(f"{len(linea):x}\r\n".encode() + linea + b"\r\n" for linea in lineas)
    return _respuesta_real(_CABECERA + b"Transfer-Encoding: chunked\r\n\r\n" + chunks)


_MEDIA_FRASE = {
    "glm": _dato({"choices": [{"delta": {"content": "La capital de Francia es"}}]}),
    "ollama": _linea_ollama({"role": "assistant", "content": "La capital de Francia es"}),
}
_ERROR_DENTRO = {
    "glm": _dato({"error": {"code": "1305", "message": "too many requests"}}),
    "ollama": json.dumps({"error": "model runner has unexpectedly stopped"}).encode() + b"\n",
}
_DETALLE_DEL_ERROR = {"glm": "too many requests", "ollama": "unexpectedly stopped"}


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_un_error_dentro_del_stream_es_un_fallo_y_no_una_respuesta_vacia(monkeypatch, motor):
    """El 200 ya salió y el servidor avisa del fallo con un objeto `error` (Ollama
    lo documenta así; los compatibles con OpenAI, para un límite de ritmo que
    salta con el stream abierto)."""
    client = _cliente(motor, monkeypatch, _stream_entero([_ERROR_DENTRO[motor]]))

    # Antes: ChatResponse(content="") y el agente contestaba como si nada.
    with pytest.raises(TransientLLMError, match=_DETALLE_DEL_ERROR[motor]):
        client.chat_stream(_MSGS)


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_un_error_tras_media_frase_tampoco_se_traga(monkeypatch, motor):
    client = _cliente(
        motor, monkeypatch, _stream_entero([_MEDIA_FRASE[motor], _ERROR_DENTRO[motor]]))
    dichos: list[str] = []

    with pytest.raises(TransientLLMError, match=_DETALLE_DEL_ERROR[motor]):
        client.chat_stream(_MSGS, on_token=dichos.append)

    assert dichos == ["La capital de Francia es"]


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_stream_que_se_acaba_sin_el_cierre_del_motor_es_un_corte(monkeypatch, motor):
    """Ni `[DONE]`/`finish_reason` (OpenAI) ni `done: true` (Ollama) y sin nada
    usable: al stream lo cortaron, aunque el HTTP no se queje (FIN entre chunks,
    o un proxy que cierra bien el chunked de un upstream que se le murió).

    Con texto ya llegado el final limpio se ACEPTA con un aviso en el log
    (criterio híbrido): ver test_transport.py::
    test_stream_truncado_avisa_y_no_marca_finish."""
    for respuesta in (_stream_sin_cierre_http([]), _stream_entero([])):
        client = _cliente(motor, monkeypatch, respuesta)
        dichos: list[str] = []

        # Antes: ChatResponse vacío dado por bueno.
        with pytest.raises(TransientLLMError, match="a medias, patrón"):
            client.chat_stream(_MSGS, on_token=dichos.append)

        assert dichos == []


def test_sse_con_finish_reason_y_sin_done_esta_completo():
    """Hay servidores que cierran tras el chunk de `finish_reason` sin mandar
    `[DONE]`: cualquiera de las dos marcas vale como final."""
    stream = [
        _dato({"choices": [{"delta": {"content": "Hola"}, "finish_reason": None}]}),
        _dato({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _dato({"choices": [], "usage": {"total_tokens": 9}}),
    ]

    content, message = consume_sse(iter(stream), None)

    assert content == "Hola"
    assert message == {"role": "assistant", "content": "Hola"}


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_una_linea_json_que_no_es_un_objeto_se_salta_como_basura(monkeypatch, motor):
    basura = {"glm": b"data: null\n" + b"data: [1]\n", "ollama": b"null\n" + b"[1]\n"}[motor]
    cuerpo = basura + _LINEA[motor] + _CIERRE[motor]
    client = _cliente(motor, monkeypatch, _con_content_length(cuerpo))

    # Antes: AttributeError cruda (None/list no tienen .get).
    assert client.chat_stream(_MSGS).content == "Hola"


_BORRAR_A_MEDIAS = {
    # OpenAI parte los argumentos entre deltas: aquí solo llegó el primer trozo.
    "glm": _dato({"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "c1",
        "function": {"name": "borrar_todo", "arguments": '{"accion": "papel'},
    }]}}]}),
    # Ollama manda la llamada entera, pero algunos modelos ponen los argumentos
    # como texto JSON: aquí llegó a medias y el turno no llegó a cerrarse. (Una
    # llamada entera tras un corte limpio sí vale: ver
    # test_perf_llm_turn_is_one_atomic_blocking_call.py::
    # test_un_tool_call_entero_de_un_stream_cortado_si_se_ejecuta.)
    "ollama": _linea_ollama({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "borrar_todo", "arguments": '{"accion": "papel'}}]}),
}


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_un_tool_call_de_un_stream_cortado_no_se_ejecuta(monkeypatch, motor):
    """La consecuencia fea del corte: los argumentos a medio llegar se degradaban
    a {} y la tool corría igual, con sus defaults."""
    ejecutadas: list[str] = []

    def borrar_todo(accion: str = "definitivo") -> str:
        ejecutadas.append(accion)
        return "Borrado."

    client = _cliente(motor, monkeypatch, _stream_sin_cierre_http([_BORRAR_A_MEDIAS[motor]]))
    agent = _agente_con_tools(client, borrar_todo=borrar_todo)
    dichos: list[str] = []

    reply = agent.handle_turn("manda eso a la papelera", on_token=dichos.append)

    assert ejecutadas == []
    assert "a medias, patrón" in reply
    assert dichos == [reply]


def test_fallback_responde_el_local_si_la_nube_falla_dentro_del_stream(monkeypatch):
    """Lo que no pasaba en ninguno de los casos: ni se abría el breaker ni se
    probaba el modelo local, porque para FallbackLLM la nube había "respondido"."""
    for respuesta in (_stream_entero([_ERROR_DENTRO["glm"]]), _stream_sin_cierre_http([])):
        local = _Motor("ollama", tokens=("Respuesta del local, patrón.",))
        llm = FallbackLLM(_cliente("glm", monkeypatch, respuesta), local, cooldown_s=300)
        dichos: list[str] = []

        respuesta_final = llm.chat_stream(_MSGS, on_token=dichos.append)

        assert respuesta_final.content == "Respuesta del local, patrón."
        assert dichos == ["Respuesta del local, patrón."]
        assert local.llamadas == 1
        assert llm.degraded


def _redireccion(status: int = 301) -> http.client.HTTPResponse:
    return _respuesta_real(
        f"HTTP/1.1 {status} Moved Permanently\r\n".encode()
        + b"Location: https://api.z.ai/api/paas/v4/chat/completions\r\nContent-Length: 0\r\n\r\n"
    )


@pytest.mark.parametrize("motor", ["glm", "ollama"])
@pytest.mark.parametrize("metodo", ["chat", "chat_stream"])
@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_una_redireccion_es_un_error_de_config_y_no_una_respuesta(
        monkeypatch, motor, metodo, status):
    """http.client no sigue redirecciones: el 3xx de un base_url con http:// (o
    con la ruta vieja) llegaba como "respuesta" con el cuerpo vacío."""
    client = _cliente(motor, monkeypatch, _redireccion(status))

    # Antes: chat_stream -> ChatResponse(content=""); chat -> "basura no-JSON".
    with pytest.raises(LLMError, match=str(status)) as info:
        getattr(client, metodo)(_MSGS)

    # Permanente (como un 400): es la config, no una caída; el breaker no se abre.
    assert type(info.value) is LLMError
    assert "https://api.z.ai/api/paas/v4/chat/completions" in str(info.value)
