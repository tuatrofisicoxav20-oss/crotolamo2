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

Aquí van las pruebas de punta a punta (contra sockets reales de loopback y con
el agente real); las unitarias están en test_transport.py, test_glm.py,
test_fase6.py y test_fallback.py.
"""

from __future__ import annotations

import json
import logging
import socket
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
        glm = self.path.endswith("/chat/completions")
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

    def __init__(self, mode: str = "ok") -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mode = mode
        self.conexiones = 0
        self.enviados = 0
        self.colgado = threading.Event()
        self.soltar = threading.Event()
        # poll_interval corto: shutdown() espera un ciclo del bucle del servidor.
        self._hilo = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._hilo.start()

    def get_request(self):
        req = super().get_request()
        self.conexiones += 1
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
        return f"http://127.0.0.1:{self.server_address[1]}"


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
