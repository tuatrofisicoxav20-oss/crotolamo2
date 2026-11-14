"""Causa raíz "el turno del LLM es una llamada atómica y bloqueante".

Pasos 1-3 (parte 1): el ciclo de vida de una respuesta en streaming tiene dueño.
- El transporte reutiliza el keep-alive tras un stream si el cliente avisa del
  fin lógico (`finish`), con un drenaje acotado en bytes y en tiempo TOTAL, y
  solo si la conexión lleva pocos segundos quieta; si no, se descarta como antes.
- Los clientes avisan ese fin, cuelgan el socket si el stream se corta, falla o
  el consumidor aborta, y avisan en el log si el stream acaba limpio sin marca.
- Un stream que acaba sin marca pero con algo usable se ACEPTA (criterio
  híbrido); sin nada usable, o con argumentos a medias, es un corte.

Pasos 4-6 (parte 2): la respuesta en vuelo tiene UN dueño (LiveStreamer).
- El agente le pasa el streamer al motor (no `streamer.feed`), lleva un único
  "ya salió texto en este turno" (Emitter) y acepta `should_abort`: el turno se
  corta en cada línea del stream (también las de un tool-call o un razonamiento,
  que no traen texto), al empezar cada iteración, antes de cada tool y antes de
  que FallbackLLM arranque el local; cierra el historial y no alimenta los
  after-turn hooks.
- `stream_release="sentence"` (opt-in, por defecto "final" = hoy) suelta en
  cuanto la 1ª frase cerró y le sigue texto; con "final" solo se loguea cuánto
  preámbulo habría soltado.
- FallbackLLM reinicia el streamer antes de pasar al local (lo retenido de la
  nube nunca se mezcla con la respuesta local); ver test_fallback.py.

Pasos 7-8 (parte A, el núcleo; el cableado del wake y del barge-in va aparte):
- `HTTPTransport.preconnect` abre UN repuesto desde otro hilo y se lo lleva la
  primera petición (cero handshakes en el turno, y releva al keep-alive propio:
  nunca quedan dos sockets ociosos); uno caducado o caído se tira sin escribir y el
  reintento único sigue conectando de nuevo. La comprobación de vida entiende TLS 1.3 (los
  tickets que llegan tras el handshake no son un cierre).
- Un wake renueva el repuesto al que le quedan menos de 15 s (la orden llega después)
  y el que nadie pide (un wake sin turno LLM) lo cierra su segador a los 30 s.
- `build_agent` pasa [voice].stream_release al agente ("final" de fábrica).

Aquí van las pruebas de punta a punta (contra sockets reales de loopback y con
el agente real); las unitarias están en test_transport.py, test_glm.py,
test_fase6.py y test_fallback.py.
"""

from __future__ import annotations

import json
import logging
import select
import shutil
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crotolamo.core.glm import GLMClient
from crotolamo.core.llm import ChatResponse, LLMClient

MSGS = [{"role": "user", "content": "hola"}]
TOKENS = ["Hola", ", pat", "rón."]
_EVENTOS_SIN_TEXTO = 100   # 1 s de stream sin texto (10 ms entre eventos)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Sin esto, Nagle en el servidor de pruebas + el ACK retardado del cliente
    # meten ~40 ms en cada respuesta por un socket reutilizado (artefacto del
    # servidor de juguete; los de verdad mandan con TCP_NODELAY).
    disable_nagle_algorithm = True

    def log_message(self, *args):  # silencio en la salida de pytest
        pass

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _sin_texto(self, glm: bool, srv) -> None:
        """Un stream largo SIN deltas de texto, como una orden: los `arguments`
        de un tool-call (GLM) o un razonamiento (reasoning_content en GLM,
        thinking en Ollama). Cuenta lo enviado; si nadie cuelga, lo genera todo."""
        for i in range(_EVENTOS_SIN_TEXTO):
            if not glm:
                obj = {"message": {"role": "assistant", "content": "", "thinking": "mmm "},
                       "done": False}
                self._chunk(json.dumps(obj).encode() + b"\n")
            else:
                if srv.mode == "tool_call":
                    fn = {"arguments": "{" if i == 0 else " "}
                    if i == 0:
                        fn["name"] = "ram_probe"
                    delta: dict = {"tool_calls": [{"index": 0, "id": "c1", "function": fn}]}
                else:
                    delta = {"reasoning_content": "mmm "}
                obj = {"choices": [{"delta": delta}]}
                self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
            srv.enviados += 1
            time.sleep(0.01)
        if glm:
            delta = ({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}
                     if srv.mode == "tool_call" else {"content": "Listo."})
            obj = {"choices": [{"delta": delta, "finish_reason": "stop"}]}
            self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self._chunk(b"data: [DONE]\n\n")
        else:
            obj = {"message": {"role": "assistant", "content": "Listo."}, "done": True}
            self._chunk(json.dumps(obj).encode() + b"\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802 - nombre que exige BaseHTTPRequestHandler
        srv = self.server
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        srv.usadas.add(self.connection)
        glm = self.path.endswith("/chat/completions")
        if srv.mode == "entero":         # sin stream (BrainThread): JSON entero de GLM
            cuerpo = json.dumps({"choices": [{"message": {"content": "".join(TOKENS)}}]})
            self.send_response(200)
            self.send_header("Content-Length", str(len(cuerpo.encode())))
            self.end_headers()
            self.wfile.write(cuerpo.encode())
            return
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if srv.mode in ("tool_call", "razonando"):
                self._sin_texto(glm, srv)
                return
            for tk in TOKENS:
                if glm:
                    obj = {"choices": [{"delta": {"content": tk}}]}
                    self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
                else:
                    obj = {"message": {"role": "assistant", "content": tk}, "done": False}
                    self._chunk(json.dumps(obj).encode() + b"\n")
                if srv.mode == "eterno":     # sigue generando hasta que cuelguen
                    time.sleep(0.02)
            if srv.mode == "eterno":
                while not srv.colgado.wait(0.02):
                    self._chunk(b": sigo\n\n" if glm else b"\n")
            elif srv.mode == "sin_marca":
                self.wfile.write(b"0\r\n\r\n")   # chunked bien cerrado, sin [DONE]/done
                self.wfile.flush()
                return
            if glm:
                self._chunk(b"data: [DONE]\n\n")
            else:
                obj = {"message": {"role": "assistant", "content": ""}, "done": True}
                self._chunk(json.dumps(obj).encode() + b"\n")
            if srv.mode == "terminal_lento":     # el fin FÍSICO tarda en llegar
                srv.soltar.wait(3.0)
            elif srv.mode == "goteo":   # tras el fin lógico gotea bytes cada 10 ms
                limite = time.monotonic() + 3.0
                while not srv.soltar.wait(0.01) and time.monotonic() < limite:
                    self._chunk(b": ping\n\n" if glm else b"\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            srv.colgado.set()               # el cliente colgó: deja de "generar"


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode: str = "ok", contexto: ssl.SSLContext | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        if contexto is not None:   # HTTPS: el handshake va en el accept
            self.socket = contexto.wrap_socket(self.socket, server_side=True)
        self._esquema = "https" if contexto is not None else "http"
        self.aceptadas: list[socket.socket] = []
        self.usadas: set[socket.socket] = set()   # conexiones que trajeron alguna petición
        self.mode = mode
        self.conexiones = 0
        # Avisa de cada conexión anotada: el hilo del servidor la anota DESPUÉS del
        # accept (y del handshake TLS), quizá cuando el cliente ya conectó.
        self._cuenta = threading.Condition()
        self.enviados = 0
        self.colgado = threading.Event()
        self.soltar = threading.Event()
        # poll_interval corto: shutdown() espera un ciclo del bucle del servidor.
        self._hilo = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._hilo.start()

    def get_request(self):
        req = super().get_request()
        with self._cuenta:
            self.conexiones += 1
            self.aceptadas.append(req[0])
            self._cuenta.notify_all()
        return req

    def handle_error(self, request, client_address):  # cortes deliberados: sin traza
        pass

    def parar(self) -> None:
        self.soltar.set()
        self.colgado.set()
        self.shutdown()
        self.server_close()
        self._hilo.join(timeout=5)

    @property
    def url(self) -> str:
        return f"{self._esquema}://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def servidor():
    creados: list[_Server] = []

    def crear(mode: str = "ok") -> _Server:
        srv = _Server(mode)
        creados.append(srv)
        return srv

    yield crear
    for srv in creados:
        srv.parar()


def _cliente(motor: str, srv: _Server):
    if motor == "glm":
        return GLMClient(api_key="k", base_url=srv.url + "/api/paas/v4", timeout=5)
    return LLMClient(host=srv.url, timeout=5)


def _conexiones_estables(srv: _Server) -> int:
    time.sleep(0.05)   # deja que el accept pendiente se cuente (no es una medida)
    return srv.conexiones


def _espera_conexiones(srv: _Server, n: int, tope: float = 5.0) -> int:
    """Espera (por evento, con tope) a que el servidor haya anotado `n` conexiones.

    Para mirar la cuenta SIN una respuesta de por medio (tras un preconnect): el
    cliente ya conectó, pero el hilo del servidor puede no haberlo anotado aún.
    Tras una respuesta no hace falta: el servidor anotó antes de contestar.
    """
    with srv._cuenta:
        assert srv._cuenta.wait_for(lambda: srv.conexiones >= n, tope)
        return srv.conexiones


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_seis_streams_usan_una_sola_conexion(servidor, motor):
    """Antes: 6 conexiones (6 handshakes TCP+TLS contra la nube)."""
    srv = servidor()
    client = _cliente(motor, srv)

    for _ in range(6):
        assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."

    assert _conexiones_estables(srv) == 1


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_terminal_lento_no_bloquea_y_cae_a_conexion_nueva(servidor, motor):
    """Si la cola no llega en _DRAIN_TIMEOUT_S, la siguiente petición la tira y
    sigue por un socket nuevo, como antes; nunca espera el timeout de lectura."""
    srv = servidor("terminal_lento")
    client = _cliente(motor, srv)

    t0 = time.perf_counter()
    client.chat_stream(MSGS, on_token=lambda t: None)
    client.chat_stream(MSGS, on_token=lambda t: None)
    transcurrido = time.perf_counter() - t0

    assert transcurrido < 2.0     # el chunk terminal se retiene 3 s en el servidor
    assert _conexiones_estables(srv) == 2


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_un_goteo_tras_el_fin_no_alarga_el_drenaje(servidor, motor):
    """El tope de tiempo del drenaje es TOTAL, no por lectura: un servidor que
    gotea bytes más rápido que _DRAIN_TIMEOUT_S tras el fin lógico retenía la
    siguiente petición hasta 64 KiB de goteo (3 s aquí). Ahora se tira a los
    50 ms y la petición sale por un socket nuevo."""
    srv = servidor("goteo")
    client = _cliente(motor, srv)

    client.chat_stream(MSGS, on_token=lambda t: None)
    t0 = time.perf_counter()
    assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."
    transcurrido = time.perf_counter() - t0

    srv.soltar.set()
    assert transcurrido < 1.0
    assert _conexiones_estables(srv) == 2


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_consumidor_que_aborta_cuelga_y_el_motor_deja_de_generar(servidor, motor):
    srv = servidor("eterno")
    client = _cliente(motor, srv)

    class _Aborto(Exception):
        pass

    def on_token(texto: str) -> None:
        raise _Aborto()

    with pytest.raises(_Aborto):
        client.chat_stream(MSGS, on_token=on_token)

    # Antes: el socket seguía abierto y el servidor generando hasta la
    # siguiente petición de ese hilo (o para siempre, si no había otra).
    assert client._transport._conn is None
    assert srv.colgado.wait(2.0)


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_stream_sin_marca_se_acepta_avisando_y_no_reutiliza(servidor, caplog, motor):
    srv = servidor("sin_marca")
    client = _cliente(motor, srv)
    logger = "crotolamo.core.glm" if motor == "glm" else "crotolamo.core.llm"

    with caplog.at_level(logging.WARNING, logger=logger):
        primera = client.chat_stream(MSGS, on_token=lambda t: None)
        client.chat_stream(MSGS, on_token=lambda t: None)

    assert primera.content == "Hola, patrón."
    assert sum("posiblemente truncada" in r.getMessage() for r in caplog.records) == 2
    assert _conexiones_estables(srv) == 2   # esa conexión no se reutiliza


def test_un_tool_call_entero_de_un_stream_cortado_si_se_ejecuta(monkeypatch, caplog):
    """Híbrido, lado Ollama: cada llamada llega ENTERA en su chunk, así que un
    corte limpio después no la deja a medias. Corre con lo que pidió el modelo
    (nunca con sus defaults) y el log avisa del posible truncado."""
    import http.client
    import io

    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry, Tool

    ejecutadas: list[str] = []

    def borrar_todo(accion: str = "definitivo") -> str:
        ejecutadas.append(accion)
        return "A la papelera."

    registry = Registry()
    registry.register(Tool(
        name="borrar_todo", func=borrar_todo, description="Borra.",
        parameters={"type": "object", "properties": {"accion": {"type": "string"}},
                    "required": []},
    ))

    def _respuesta(lineas: list[bytes], terminado: bool) -> http.client.HTTPResponse:
        chunks = b"".join(f"{len(x):x}\r\n".encode() + x + b"\r\n" for x in lineas)
        crudo = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunks
                 + (b"0\r\n\r\n" if terminado else b""))

        class _Zocalo:
            def makefile(self, mode):
                return io.BufferedReader(io.BytesIO(crudo))

        resp = http.client.HTTPResponse(_Zocalo())  # type: ignore[arg-type]
        resp.begin()
        return resp

    def _linea(message: dict, done: bool = False) -> bytes:
        chunk = {"message": {"role": "assistant", **message}, "done": done}
        return json.dumps(chunk).encode() + b"\n"

    llamada = {"function": {"name": "borrar_todo", "arguments": {"accion": "papelera"}}}
    respuestas = [
        _respuesta([_linea({"content": "", "tool_calls": [llamada]})], terminado=False),
        _respuesta([_linea({"content": "Listo, a la papelera."}),
                    _linea({"content": ""}, done=True)], terminado=True),
    ]

    class _Conn:
        def request(self, *a, **k):
            pass

        def getresponse(self):
            return respuestas.pop(0)

        def close(self):
            pass

    client = LLMClient(host="http://localhost:1")
    monkeypatch.setattr(client._transport, "_connect", _Conn)
    agent = ToolAgent(client, Conversation("SYS"), registry=registry,
                      guard=Guard(allowed_roots=[]), fastpath=False)

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.llm"):
        reply = agent.handle_turn("manda eso a la papelera", on_token=lambda t: None)

    assert ejecutadas == ["papelera"]
    assert reply == "Listo, a la papelera."
    assert any("posiblemente truncada" in r.getMessage() for r in caplog.records)


def test_drenaje_real_usa_el_socket_y_restaura_su_timeout(servidor):
    """Con un socket de verdad: tras el drenaje el timeout de lectura vuelve a
    ser el del cliente (no los 50 ms del drenaje)."""
    srv = servidor()
    client = _cliente("glm", srv)

    client.chat_stream(MSGS, on_token=lambda t: None)
    client.chat_stream(MSGS, on_token=lambda t: None)

    sock = client._transport._conn.sock
    assert isinstance(sock, socket.socket)
    assert sock.gettimeout() == 5
    assert _conexiones_estables(srv) == 1


@pytest.mark.parametrize("hueco, esperadas", [(5.0, [1, 0, 0, 0]), (16.0, [1, 1, 1, 1])],
                         ids=["hueco_5s", "hueco_16s"])
def test_entre_turnos_solo_reutiliza_dentro_del_tope_de_inactividad(servidor, monkeypatch,
                                                                    hueco, esperadas):
    """A4 del bench (charla, tool, charla, tool) con el HUECO entre turnos que
    tiene la voz de verdad (respuesta hablada, wake, orden, STT). Dentro de un
    turno con tools la 2ª llamada siempre reutiliza; ENTRE turnos solo si la
    siguiente petición llega antes de _REUSE_MAX_IDLE_S (15 s). Con un hueco
    mayor, que es lo normal en voz, es una conexión por turno: 4 en vez de las
    6 de antes, no 1 (el 1 del bench es de turnos pegados)."""
    from crotolamo.core import engine

    reloj = {"t": 1000.0}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    srv = servidor()
    client = _cliente("glm", srv)

    por_turno = []
    for llamadas in (1, 2, 1, 2):
        antes = _conexiones_estables(srv)
        for _ in range(llamadas):
            client.chat_stream(MSGS, on_token=lambda t: None)
            reloj["t"] += 0.5            # la tool entre las 2 llamadas del turno
        por_turno.append(_conexiones_estables(srv) - antes)
        reloj["t"] += hueco

    assert por_turno == esperadas


# --- Pasos 4-6: el agente le pregunta al dueño de la respuesta en vuelo ----------

_PREAMBULO_2 = "¡Claro que sí, patrón! Voy a revisar la memoria ahora mismo."
_CHARLA = "¡Qué onda, patrón! Aquí andamos al cien. Dime en qué te echo la mano."
_RESPUESTA = "Traes la mitad de la memoria ocupada, patrón."


def _tokens(texto: str) -> list[str]:
    """Trozos de <= 4 caracteres, como los deltas de un LLM."""
    return [texto[i:i + 4] for i in range(0, len(texto), 4)]


class _Guion:
    """LLM de guion: `pasos` = [(texto, pide_tool)] por llamada. Anota si el
    stream seguía en curso cada vez que algo salió por on_token del agente."""

    model = "guion"

    def __init__(self, pasos, tools=("ram_probe", "disk_probe")):
        self.pasos = list(pasos)
        self.tools = tools
        self.llamadas = 0
        self.generados = 0
        self.en_curso = False
        self.on_token_recibido = None

    def chat_stream(self, messages, tools=None, on_token=None):
        texto, pide_tool = self.pasos[self.llamadas]
        self.llamadas += 1
        self.on_token_recibido = on_token
        self.en_curso = True
        try:
            for tk in _tokens(texto):
                self.generados += 1
                on_token(tk)
        finally:
            self.en_curso = False
        if pide_tool:
            calls = [{"name": n, "arguments": {}} for n in self.tools]
            raw = [{"function": {"name": n, "arguments": {}}} for n in self.tools]
            return ChatResponse(content=texto, tool_calls=calls,
                                raw_message={"role": "assistant", "content": texto,
                                             "tool_calls": raw})
        return ChatResponse(content=texto)

    def chat(self, messages, tools=None):
        return self.chat_stream(messages, tools, on_token=lambda t: None)


def _registry(corridas: list[str], al_correr=None):
    from crotolamo.tools.base import Registry, Tool

    def hacer(nombre):
        def tool() -> str:
            corridas.append(nombre)
            if al_correr is not None:
                al_correr(nombre)
            return f"{nombre}: 7.6 GiB usados de 15.3 GiB."
        return tool

    reg = Registry()
    for nombre in ("ram_probe", "disk_probe"):
        reg.register(Tool(name=nombre, func=hacer(nombre), description="Mide.",
                          parameters={"type": "object", "properties": {}, "required": []}))
    return reg


def _agente(llm, corridas=None, al_correr=None, **kw):
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard

    reg = _registry(corridas if corridas is not None else [], al_correr)
    kw.setdefault("post_hooks", [])
    return ToolAgent(llm, Conversation("SYS"), registry=reg, guard=Guard(allowed_roots=[]),
                     fastpath=False, direct_tools=set(), pre_hooks=[],
                     route_fn=lambda t: reg.schemas(), **kw)


class _Oido:
    """on_token del caller: anota cada texto y si el stream seguía en curso."""

    def __init__(self, llm):
        self.llm = llm
        self.dichos: list[str] = []
        self.durante_el_stream: list[bool] = []

    def __call__(self, texto: str) -> None:
        self.dichos.append(texto)
        self.durante_el_stream.append(self.llm.en_curso)


def test_con_tools_a_la_vista_nada_sale_antes_del_fin_del_stream():
    """El valor por defecto ("final") fija lo de hoy: con tools a la vista no
    suena nada hasta que acaba la llamada, y sale de golpe."""
    llm = _Guion([(_CHARLA, False)])
    oido = _Oido(llm)

    reply = _agente(llm).handle_turn("cuéntame cómo andas", on_token=oido)

    assert reply == _CHARLA
    assert oido.dichos == [_CHARLA]
    assert oido.durante_el_stream == [False]


def test_el_agente_le_pasa_al_motor_el_streamer_entero():
    """FallbackLLM necesita `restart`: con `streamer.feed` no lo alcanzaba."""
    from crotolamo.core.streaming import LiveStreamer

    llm = _Guion([(_CHARLA, False)])
    _agente(llm).handle_turn("hola", on_token=lambda t: None)
    assert isinstance(llm.on_token_recibido, LiveStreamer)


def test_stream_release_sentence_sale_antes_del_fin():
    llm = _Guion([(_CHARLA, False)])
    oido = _Oido(llm)

    reply = _agente(llm, stream_release="sentence").handle_turn("hola", on_token=oido)

    assert reply == _CHARLA
    assert "".join(oido.dichos) == _CHARLA      # todo, una vez, en orden
    assert oido.durante_el_stream[0] is True     # la 1ª frase sonó con el stream en curso
    assert oido.dichos[0].startswith("¡Qué onda, patrón! ")   # la frase entera, junta
    assert len(oido.dichos) > 2                                # y el resto en vivo


def test_stream_release_sentence_calla_el_preambulo_de_una_frase():
    corridas: list[str] = []
    llm = _Guion([("Voy a revisar la memoria por ti.", True), (_RESPUESTA, False)])
    oido = _Oido(llm)

    reply = _agente(llm, corridas, stream_release="sentence").handle_turn(
        "cuánta ram", on_token=oido)

    assert corridas == ["ram_probe", "disk_probe"]
    assert reply == _RESPUESTA
    assert oido.dichos == [_RESPUESTA]


def test_stream_release_desconocido_cae_a_final(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.agent"):
        agent = _agente(_Guion([]), stream_release="sentense")
    assert agent.stream_release == "final"
    assert any("stream_release" in r.getMessage() for r in caplog.records)
    assert _agente(_Guion([])).stream_release == "final"


@pytest.mark.parametrize("preambulo, loguea", [(_PREAMBULO_2, True),
                                                ("Voy a revisar la memoria por ti.", False)])
def test_con_final_se_loguea_el_preambulo_que_sentence_habria_hablado(caplog, preambulo,
                                                                     loguea):
    """El dato para decidir [voice].stream_release con el motor REAL, sin hablarlo."""
    llm = _Guion([(preambulo, True), (_RESPUESTA, False)])
    oido = _Oido(llm)

    with caplog.at_level(logging.INFO, logger="crotolamo.core.agent"):
        _agente(llm).handle_turn("cuánta ram", on_token=oido)

    assert oido.dichos == [_RESPUESTA]           # por defecto nada cambia
    avisos = [r for r in caplog.records
              if r.levelno == logging.INFO and "preámbulo" in r.getMessage()]
    assert bool(avisos) is loguea
    if loguea:
        assert f"{len(_PREAMBULO_2)} caracteres" in avisos[0].getMessage()


def _historial(agent) -> list[dict]:
    return agent.conversation.to_messages()[1:]   # sin el system


def test_should_abort_al_empezar_no_llama_al_llm_y_cierra_el_historial():
    llm = _Guion([(_CHARLA, False)])
    agent = _agente(llm)

    reply = agent.handle_turn("hola", on_token=lambda t: None, should_abort=lambda: True)

    assert reply == ""
    assert llm.llamadas == 0
    assert _historial(agent)[-1] == {"role": "assistant",
                                     "content": "(Turno interrumpido por el patrón.)"}


def test_should_abort_a_mitad_del_stream_corta_en_el_siguiente_token():
    corridas: list[str] = []
    llm = _Guion([(_PREAMBULO_2, True), (_RESPUESTA, False)])
    vigente = {"ok": True}

    def on_token(_texto):
        pass

    class _Corta:
        """El barge-in llega con el 3er token generado."""

        def __call__(self):
            if llm.generados >= 3:
                vigente["ok"] = False
            return not vigente["ok"]

    reply = _agente(llm, corridas).handle_turn("cuánta ram", on_token=on_token,
                                               should_abort=_Corta())

    assert reply == ""
    assert llm.generados == 3          # ni un token más tras el barge-in
    assert llm.llamadas == 1           # ni una llamada más
    assert corridas == []              # ni la tool que iba a pedir


def test_should_abort_antes_de_la_tool_no_la_ejecuta_y_cierra_el_historial():
    corridas: list[str] = []
    vigente = {"ok": True}
    llm = _Guion([("", True), (_RESPUESTA, False)])

    def al_correr(_nombre):
        vigente["ok"] = False          # el patrón interrumpe mientras corre la 1ª

    agent = _agente(llm, corridas, al_correr=al_correr)
    reply = agent.handle_turn("cuánta ram y disco", should_abort=lambda: not vigente["ok"])

    assert reply == ""
    assert corridas == ["ram_probe"]   # la 2ª ya no corre
    assert llm.llamadas == 1
    historial = _historial(agent)
    pedidas = [c["function"]["name"] for c in historial[1]["tool_calls"]]
    resultados = [m for m in historial if m["role"] == "tool"]
    assert len(resultados) == len(pedidas) == 2   # cada tool_call con su resultado
    assert resultados[1]["content"] == "Cancelado: el patrón interrumpió."
    assert historial[-1] == {"role": "assistant",
                             "content": "(Turno interrumpido por el patrón.)"}


def test_turno_cancelado_no_dispara_after_turn_hooks():
    vistos: list[tuple[str, str]] = []
    llm = _Guion([(_CHARLA, False), (_CHARLA, False)])
    agent = _agente(llm, after_turn_hooks=[lambda t, r: vistos.append((t, r))])

    agent.handle_turn("hola", should_abort=lambda: True)
    assert vistos == []
    agent.handle_turn("hola otra vez")           # sin should_abort: como siempre
    assert vistos == [("hola otra vez", _CHARLA)]


def test_cancelar_no_abre_el_breaker_ni_prueba_el_local():
    """TurnCancelled no es un LLMError: FallbackLLM lo deja pasar tal cual."""
    from crotolamo.core.fallback import FallbackLLM

    nube = _Guion([(_CHARLA, False)])
    local = _Guion([(_CHARLA, False)])
    llm = FallbackLLM(nube, local)

    reply = _agente(llm).handle_turn("hola", on_token=lambda t: None,
                                     should_abort=lambda: nube.generados >= 2)

    assert reply == ""
    assert local.llamadas == 0
    assert not llm.degraded


def test_sin_should_abort_ni_stream_release_el_turno_es_el_de_siempre():
    """Llamada de hoy (posicional): misma respuesta, mismos tokens."""
    corridas: list[str] = []
    llm = _Guion([("", True), (_RESPUESTA, False)])
    dichos: list[str] = []

    reply = _agente(llm, corridas).handle_turn("cuánta ram", dichos.append)

    assert (reply, dichos, corridas) == (_RESPUESTA, [_RESPUESTA], ["ram_probe", "disk_probe"])


def test_el_error_tras_texto_retenido_no_lleva_salto():
    """Lo retenido no salió: el aviso de error no va "en línea aparte"."""
    from crotolamo.core.llm import TransientLLMError

    class _SeCae:
        model = "x"

        def chat_stream(self, messages, tools=None, on_token=None):
            on_token("¡Claro que ")
            raise TransientLLMError("GLM no respondió a tiempo, patrón.")

    dichos: list[str] = []
    reply = _agente(_SeCae()).handle_turn("hola", on_token=dichos.append)

    assert dichos == [reply] == ["GLM no respondió a tiempo, patrón."]


@pytest.mark.parametrize("motor", ["glm", "ollama"])
def test_barge_in_a_mitad_del_stream_cuelga_el_socket_real(servidor, motor):
    """De punta a punta: agente real + cliente real. El turno cancelado no deja
    a la nube (ni a Ollama, que quema CPU) generando para nadie."""
    srv = servidor("eterno")
    client = _cliente(motor, srv)
    tokens: list[str] = []

    reply = _agente(client).handle_turn(
        "hola", on_token=lambda t: None,
        should_abort=lambda: len(tokens) >= 2 or bool(tokens.append(1)))

    assert reply == ""
    assert client._transport._conn is None
    assert srv.colgado.wait(2.0)


def test_el_agente_sin_tools_acepta_should_abort():
    """El Agent sin tools (al que cae el shell si el registry no se puede armar)
    acepta el mismo gancho que ToolAgent: quien lo cablee no revienta en el modo
    degradado. Su único punto de control es antes de llamar al LLM."""
    from crotolamo.core.agent import Agent
    from crotolamo.core.memory import Conversation

    class _Chat:
        model = "x"
        llamadas = 0

        def chat(self, messages, tools=None):
            _Chat.llamadas += 1
            return ChatResponse(content="Aquí ando, patrón.")

    agent = Agent(_Chat(), Conversation("SYS"))
    dichos: list[str] = []
    assert agent.handle_turn("hola", on_token=dichos.append,
                             should_abort=lambda: False) == "Aquí ando, patrón."
    assert agent.handle_turn("otra", on_token=dichos.append, should_abort=lambda: True) == ""

    assert _Chat.llamadas == 1
    assert dichos == ["Aquí ando, patrón."]
    assert _historial(agent)[-2:] == [
        {"role": "user", "content": "otra"},
        {"role": "assistant", "content": "(Turno interrumpido por el patrón.)"},
    ]


class _SinLocal:
    """Respaldo que el turno cancelado nunca debe arrancar."""

    model = "local"

    def chat_stream(self, messages, tools=None, on_token=None):
        raise AssertionError("un turno cancelado arrancó el modelo local")

    chat = chat_stream


@pytest.mark.parametrize("motor, modo", [("glm", "tool_call"), ("glm", "razonando"),
                                         ("ollama", "razonando")])
def test_barge_in_corta_un_stream_sin_texto(servidor, motor, modo):
    """Una orden (tool-call en deltas de `arguments`) o un razonamiento no traen
    texto y no pasaban por `feed`, el único punto de control: el turno cancelado
    esperaba el stream entero (y la nube lo generaba entero). Ahora cada línea
    del stream es punto de control: se cuelga en cuanto se cancela."""
    from crotolamo.core.fallback import FallbackLLM

    srv = servidor(modo)
    client = _cliente(motor, srv)
    corridas: list[str] = []
    agent = _agente(FallbackLLM(client, _SinLocal()), corridas)

    reply = agent.handle_turn("cuánta ram", on_token=lambda t: None,
                              should_abort=lambda: srv.enviados >= 2)

    assert reply == ""
    assert srv.colgado.wait(2.0)                       # el servidor vio el cuelgue
    assert srv.enviados < _EVENTOS_SIN_TEXTO // 2      # antes: los 100, todos
    assert client._transport._conn is None
    assert corridas == []
    assert _historial(agent)[-1]["content"] == "(Turno interrumpido por el patrón.)"


def test_barge_in_mientras_la_nube_cae_no_arranca_el_local():
    """El patrón interrumpe y la nube se cae en la misma llamada: FallbackLLM ya
    no arranca el local (segundos de prompt-eval en CPU) para un turno que
    nadie va a oír. El fallo de la nube sí cuenta para el breaker."""
    from crotolamo.core.fallback import FallbackLLM
    from crotolamo.core.llm import TransientLLMError

    vigente = {"ok": True}

    class _NubeQueCae:
        model = "glm"

        def chat_stream(self, messages, tools=None, on_token=None):
            vigente["ok"] = False
            raise TransientLLMError("GLM no respondió a tiempo, patrón. (timed out)")

    corridas: list[str] = []
    llm = FallbackLLM(_NubeQueCae(), _SinLocal())
    agent = _agente(llm, corridas)

    reply = agent.handle_turn("cuánta ram", on_token=lambda t: None,
                              should_abort=lambda: not vigente["ok"])

    assert reply == ""
    assert corridas == []
    assert llm.degraded
    assert _historial(agent)[-1]["content"] == "(Turno interrumpido por el patrón.)"


# --- Paso 7 (parte A): la conexión se abre mientras el patrón dice la orden ------
# HTTPTransport.preconnect abre UN repuesto desde otro hilo (el del aviso del wake);
# la primera petición, de cualquier hilo, se lo lleva. Con sockets de verdad: HTTP de
# loopback y TLS 1.3 de loopback (el de la nube real), sin tocar el transporte.

def _contando_conexiones(monkeypatch, transport) -> dict[str, int]:
    """Cuántas conexiones pide cada hilo (lo que mide la sección F del bench)."""
    por_hilo: dict[str, int] = {}
    original = transport._connect

    def contando():
        nombre = threading.current_thread().name
        por_hilo[nombre] = por_hilo.get(nombre, 0) + 1
        return original()

    monkeypatch.setattr(transport, "_connect", contando)
    return por_hilo


def _al_wake(client) -> None:
    """Lo que hará el aviso del wake: pre-conectar en segundo plano (y aquí, para
    medir, esperar a que termine)."""
    from crotolamo.core.engine import preconnect_in_background

    hilo = preconnect_in_background(client)
    assert hilo is not None
    hilo.join(timeout=5)
    assert not hilo.is_alive()


def test_preconnect_real_el_turno_no_conecta_nada(servidor, monkeypatch):
    """F2 del bench con sockets de verdad: el repuesto lo abre el hilo del wake y
    las dos llamadas del turno (la orden y la de después de la tool) van por él:
    una conexión en total y ningún handshake en el hilo del turno."""
    srv = servidor()
    client = _cliente("glm", srv)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    _al_wake(client)
    assert _espera_conexiones(srv, 1) == 1         # conectado ANTES de la orden
    for _ in range(2):
        assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."

    assert por_hilo == {"Preconnect": 1}
    assert _conexiones_estables(srv) == 1


def test_el_repuesto_va_por_el_mismo_camino_que_cualquier_conexion(servidor, monkeypatch):
    """El repuesto se abre como cualquier conexión (`_connect`): el tope corto para
    CONECTAR, el de leer de siempre después, y keepalive TCP contra la nube (el
    que deja a `_dropped` ver una ruta caída mientras espera la orden)."""
    from crotolamo.core import engine

    srv = servidor()
    reales, crear = socket.getaddrinfo, socket.create_connection
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: reales(
        "127.0.0.1" if host == "nube.test" else host, port, *a, **k))
    topes: list[float] = []

    def creando(address, timeout, *args, **kwargs):
        topes.append(timeout)
        return crear(address, timeout, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", creando)
    client = GLMClient(api_key="k", timeout=60, connect_timeout=2,
                       base_url=f"http://nube.test:{srv.server_address[1]}/api/paas/v4")
    assert client._transport._keepalive                 # "de fuera", como api.z.ai

    assert client.preconnect() is True
    conn = client._transport._spare[0]
    sock = conn.sock

    assert isinstance(conn, engine._HTTPConnection)
    assert topes == [2]                                 # conectar: el tope corto
    assert sock.gettimeout() == 60                      # leer: lo de siempre
    assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
    tcp = {o: sock.getsockopt(socket.IPPROTO_TCP, getattr(socket, o))
           for o in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT")}
    assert tcp == {"TCP_KEEPIDLE": 10, "TCP_KEEPINTVL": 5, "TCP_KEEPCNT": 3}
    assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."
    assert topes == [2] and client._transport._conn is conn   # y el turno va por él


def test_en_el_loop_concurrente_cada_repuesto_se_usa(servidor, monkeypatch):
    """Estado estable del loop concurrente con el wake cableado (parte B): el hilo
    del turno contesta sin stream y su keep-alive no tiene tope de edad, así que
    sigue vivo en cada orden. Antes, tras el primer turno ningún repuesto se usaba
    (4 de 5 handshakes tirados y dos sockets ociosos con sondas de keepalive); ahora
    el turno se lleva el repuesto y cierra su keep-alive, más viejo."""
    from crotolamo.core import engine

    reloj = {"t": 1000.0}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    srv = servidor("entero")
    client = _cliente("glm", srv)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    turnos = 4
    for _ in range(turnos):
        _al_wake(client)
        assert client.chat(MSGS).content == "Hola, patrón."
        assert client._transport._spare is None          # ningún repuesto sin usar
        reloj["t"] += 40.0                               # la orden siguiente

    assert por_hilo == {"Preconnect": turnos}            # cero handshakes en el turno
    assert _espera_conexiones(srv, turnos) == turnos
    assert all(s in srv.usadas for s in srv.aceptadas)   # antes: 1 de 4 usada


def test_sin_preconnect_el_turno_conecta_como_siempre(servidor, monkeypatch):
    srv = servidor()
    client = _cliente("glm", srv)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    for _ in range(2):
        client.chat_stream(MSGS, on_token=lambda t: None)

    assert por_hilo == {"MainThread": 1}
    assert _conexiones_estables(srv) == 1


def test_preconnect_real_sin_servidor_no_lanza_ni_deja_repuesto(monkeypatch):
    """Nube inalcanzable (puerto cerrado): False, sin repuesto, y el turno se
    entera como hoy (el mismo error en personaje)."""
    from crotolamo.core.llm import TransientLLMError

    libre = socket.socket()
    libre.bind(("127.0.0.1", 0))
    puerto = libre.getsockname()[1]
    libre.close()
    client = GLMClient(api_key="k", base_url=f"http://127.0.0.1:{puerto}/v4", timeout=2)

    assert client.preconnect() is False
    assert client._transport._spare is None
    with pytest.raises(TransientLLMError, match="¿Hay internet?"):
        client.chat(MSGS)


def test_un_wake_cerca_de_la_caducidad_del_repuesto_abre_otro(servidor, monkeypatch):
    """Lo que midió la revisión, con cliente, transporte y servidor reales: wake 1 sin
    turno LLM detrás, wake 2 a los 27 s y la orden al LLM 6 s después. Antes el wake 2
    se conformaba con el repuesto de 27 s y el turno lo encontraba caducado:
    {'Preconnect': 1, 'MainThread': 1}. Ahora el wake 2 lo renueva y cierra el viejo."""
    from crotolamo.core import engine

    reloj = {"t": 1000.0}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    srv = servidor("entero")
    client = _cliente("glm", srv)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    _al_wake(client)                                 # wake 1 ("pausa": el fastpath)
    viejo = client._transport._spare[0].sock
    reloj["t"] += 27
    _al_wake(client)                                 # wake 2
    assert viejo.fileno() == -1                      # cerrado: nunca quedan dos
    reloj["t"] += 6                                  # la orden y el STT
    assert client.chat(MSGS).content == "Hola, patrón."

    assert por_hilo == {"Preconnect": 2}             # cero handshakes en el turno
    assert _espera_conexiones(srv, 2) == 2


def test_el_repuesto_real_que_nadie_pide_se_cierra_al_caducar(servidor, monkeypatch):
    """Wake sin turno LLM detrás: el socket del repuesto se cierra al caducar (antes
    seguía abierto, con sondas de keepalive y luego en CLOSE_WAIT, hasta la siguiente
    petición o el siguiente wake) y la orden siguiente conecta como siempre, sin
    petición fallida. Con un tope de edad corto para la prueba."""
    from crotolamo.core.engine import HTTPTransport

    monkeypatch.setattr(HTTPTransport, "_SPARE_MAX_AGE_S", 0.3)
    srv = servidor()
    client = _cliente("glm", srv)
    transporte = client._transport
    por_hilo = _contando_conexiones(monkeypatch, transporte)
    abiertas: list = []
    contando = transporte._connect
    monkeypatch.setattr(transporte, "_connect",
                        lambda: abiertas.append(contando()) or abiertas[-1])
    # El segador avisa al terminar (espera por evento, sin carreras con el reloj).
    segado = threading.Event()
    segar = transporte._expire_spare
    monkeypatch.setattr(transporte, "_expire_spare",
                        lambda conn: (segar(conn), segado.set()))

    _al_wake(client)
    assert segado.wait(timeout=5)

    assert abiertas[0].sock is None and transporte._spare is None   # cerrado
    assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."
    assert por_hilo == {"Preconnect": 1, "MainThread": 1}
    assert _espera_conexiones(srv, 2) == 2


@pytest.fixture(scope="module")
def certificado(tmp_path_factory):
    """Certificado autofirmado para 127.0.0.1, hecho al vuelo: ninguna clave en el
    repo. Sin openssl (no es el caso en Fedora ni en CI) se salta."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("sin openssl para generar el certificado de prueba")
    carpeta = tmp_path_factory.mktemp("tls")
    cert, key = carpeta / "cert.pem", carpeta / "key.pem"
    try:
        subprocess.run(
            [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt",
             "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", str(key),
             "-out", str(cert), "-days", "2", "-subj", "/CN=127.0.0.1",
             "-addext", "subjectAltName=IP:127.0.0.1"],
            check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"openssl no pudo generar el certificado: {error}")
    return cert, key


@pytest.fixture
def servidor_tls(certificado, monkeypatch):
    cert, key = certificado
    # El cliente confía en él por el camino de siempre: el contexto TLS por defecto
    # de http.client lee SSL_CERT_FILE. Del transporte no se toca nada.
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    creados: list[_Server] = []

    def crear(mode: str = "ok", tickets: int | None = None) -> _Server:
        contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        contexto.load_cert_chain(cert, key)
        if tickets is not None:   # OpenSSL manda 2 de fábrica, en dos escrituras
            contexto.num_tickets = tickets
        srv = _Server(mode, contexto=contexto)
        creados.append(srv)
        return srv

    yield crear
    for srv in creados:
        srv.parar()


def _legible(sock, tope_s: float = 2.0) -> bool:
    """¿select() da el socket por legible antes del tope? (espera por evento)."""
    legible, _, _ = select.select([sock], [], [], tope_s)
    return bool(legible)


def test_tls13_el_repuesto_recien_abierto_no_se_tira(servidor_tls, monkeypatch):
    """TLS 1.3 manda los tickets de sesión DESPUÉS del handshake: el repuesto, que
    aún no leyó nada, los tiene pendientes y select() lo da por legible. Tomado
    por un cierre, cada repuesto se tiraba al usarlo y el turno volvía a pagar el
    handshake que el wake ya había pagado."""
    srv = servidor_tls()
    client = GLMClient(api_key="k", base_url=srv.url + "/api/paas/v4", timeout=5)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    _al_wake(client)
    sock = client._transport._spare[0].sock
    assert isinstance(sock, ssl.SSLSocket) and sock.version() == "TLSv1.3"
    assert _legible(sock)                          # los tickets: el caso de verdad

    for _ in range(2):
        assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."
    assert por_hilo == {"Preconnect": 1}
    assert _conexiones_estables(srv) == 1


def test_tls_el_repuesto_que_el_servidor_cerro_se_tira_sin_escribir(servidor_tls,
                                                                    monkeypatch):
    """El servidor cierra el repuesto ocioso (su tope de inactividad): `_dropped`
    lo ve antes de escribir y la orden sale por una conexión nueva, sin petición
    fallida ni reintento."""
    from crotolamo.core import engine

    # Un solo ticket: así se sabe cuándo llegó todo lo de TLS y que lo legible
    # de después es el cierre (con dos, el segundo puede llegar en medio).
    srv = servidor_tls(tickets=1)
    client = GLMClient(api_key="k", base_url=srv.url + "/api/paas/v4", timeout=5)
    _al_wake(client)
    repuesto = client._transport._spare[0]
    assert _legible(repuesto.sock)                 # llegó el ticket...
    assert engine._dropped(repuesto) is False      # ...y no es un cierre
    assert not _legible(repuesto.sock, 0)          # (ya procesado)
    escritas: list[str] = []
    original = repuesto.request
    repuesto.request = lambda *a, **k: escritas.append("x") or original(*a, **k)

    _espera_conexiones(srv, 1)                     # el accept puede ir por detrás
    srv.aceptadas[0].shutdown(socket.SHUT_RDWR)    # el servidor lo cuelga
    assert _legible(repuesto.sock)                 # el EOF llegó

    assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola, patrón."
    assert escritas == []                          # ni un byte al socket muerto
    assert repuesto.sock is None                   # cerrado
    assert _conexiones_estables(srv) == 2


def test_tls_el_keepalive_tras_una_respuesta_se_sigue_reutilizando(servidor_tls,
                                                                  monkeypatch):
    """La comprobación de vida que entiende TLS no cambia la reutilización de
    siempre: N streams sobre HTTPS = 1 conexión."""
    srv = servidor_tls()
    client = GLMClient(api_key="k", base_url=srv.url + "/api/paas/v4", timeout=5)
    por_hilo = _contando_conexiones(monkeypatch, client._transport)

    for _ in range(3):
        client.chat_stream(MSGS, on_token=lambda t: None)

    assert por_hilo == {"MainThread": 1}
    assert _conexiones_estables(srv) == 1


@pytest.mark.parametrize("nombre", [
    "test_preconnect_real_el_turno_no_conecta_nada",
    "test_tls13_el_repuesto_recien_abierto_no_se_tira",
    "test_tls_el_repuesto_que_el_servidor_cerro_se_tira_sin_escribir",
])
def test_las_cuentas_del_servidor_no_dependen_del_reparto_de_cpu(nombre, request,
                                                                  monkeypatch):
    """El hilo del servidor anota la conexión DESPUÉS del accept (y del handshake
    TLS, que va dentro): en una máquina cargada (CI) el cliente ya conectó y la
    cuenta aún no consta. Lo que se mira sin una respuesta de por medio espera a esa
    cuenta (`_espera_conexiones`), no a un sleep; aquí el retraso se fuerza."""
    import inspect
    import socketserver

    original = socketserver.TCPServer.get_request

    def lento(self):
        req = original(self)
        time.sleep(0.1)          # el hilo del servidor tarda en volver a tener CPU
        return req

    monkeypatch.setattr(socketserver.TCPServer, "get_request", lento)
    prueba = globals()[nombre]
    prueba(**{p: monkeypatch if p == "monkeypatch" else request.getfixturevalue(p)
              for p in inspect.signature(prueba).parameters})


# --- Paso 8 (parte A): [voice].stream_release llega al agente ----------------------
# ToolAgent ya sabía soltar en "sentence"; nadie le pasaba la clave, así que con
# GLM de cerebro stream_speak no adelantaba nada y no había forma de pedirlo.

def _build_agent_con_voz(monkeypatch, tmp_path, voz: dict, backend: str = "glm"):
    """build_agent REAL con la [voice] dada; el motor es de mentiras (no conversa)."""
    from crotolamo import settings as settings_mod
    from crotolamo.core import hooks
    from crotolamo.core import memoria as memoria_mod
    from crotolamo.core.fallback import FallbackLLM
    from crotolamo.settings import Settings
    from interfaces import shell

    class _Mudo:
        model = "x"

        def chat(self, messages, tools=None):
            raise AssertionError("este test no conversa")

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": backend}, "voice": voz},
                                 user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    llm = FallbackLLM(_Mudo(), _Mudo()) if backend == "glm" else _Mudo()
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, _ = shell.build_agent()
    agent.close()
    return agent


@pytest.mark.parametrize("backend", ["glm", "ollama"])
def test_build_agent_pasa_stream_release_de_la_voz(monkeypatch, tmp_path, backend):
    from crotolamo.core.agent import ToolAgent

    agent = _build_agent_con_voz(monkeypatch, tmp_path, {"stream_release": "sentence"},
                                 backend)
    assert isinstance(agent, ToolAgent)
    assert agent.stream_release == "sentence"


def test_build_agent_sin_la_clave_retiene_como_siempre(monkeypatch, tmp_path):
    assert _build_agent_con_voz(monkeypatch, tmp_path, {}).stream_release == "final"


@pytest.mark.parametrize("raro", ["sentense", 1, ""])
def test_build_agent_stream_release_raro_cae_a_final_avisando(monkeypatch, tmp_path,
                                                             caplog, raro):
    """Un typo en el toml no deja al patrón sin voz ni habla preámbulos: "final" y
    un WARNING que dice qué valores valen."""
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.agent"):
        agent = _build_agent_con_voz(monkeypatch, tmp_path, {"stream_release": raro})
    assert agent.stream_release == "final"
    assert any("stream_release" in r.getMessage() for r in caplog.records)


def test_config_de_fabrica_documenta_stream_release_y_retiene():
    """La clave está documentada junto a stream_speak, comentada con su valor de
    fábrica ("final" = lo de siempre), con lo que cuesta y lo que gana."""
    import re
    import tomllib

    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert "stream_release" not in tomllib.loads(texto)["voice"]   # de fábrica: "final"
    voz = re.split(r"^\[voice\]$", texto, maxsplit=1, flags=re.M)[1]
    voz = re.split(r"^\[", voz, maxsplit=1, flags=re.M)[0]
    lineas = voz.splitlines()
    i = next(n for n, linea in enumerate(lineas)
             if re.match(r'^#\s*stream_release\s*=\s*"final"\s*$', linea))
    # En el grupo de hablar: tras stream_speak (las claves de hablar del loop
    # concurrente pueden ir en medio) y antes de las ventanas del wake.
    hablar = next(n for n, linea in enumerate(lineas) if linea.startswith("stream_speak ="))
    assert hablar < i
    assert not [linea for linea in lineas[hablar:i] if linea.startswith("wake_")]
    comentario = []
    for linea in reversed(lineas[:i]):
        if not linea.startswith("#"):
            break
        comentario.insert(0, linea.lstrip("#").strip())
    explicacion = " ".join(comentario)
    for dato in ('"sentence"', "preámbulo", "1.7", "0.6", "habría hablado", "INFO"):
        assert dato in explicacion, dato
