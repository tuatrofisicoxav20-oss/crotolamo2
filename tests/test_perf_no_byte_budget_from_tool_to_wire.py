"""Causa raíz de rendimiento: nadie era dueño de un presupuesto de bytes entre la
tool y el request HTTP.

Aquí, el punto único por el que pasan TODOS los resultados de tool antes de
entrar al historial (ToolAgent._safe_execute): el tope TOOL_RESULT_CAP acota a
las tools sin tope propio sin cambiar un byte de lo que ya cabe. Y los bordes
de red: la descarga web con plazo total y gzip, las tools de la LAN con el
mismo lector a trozos, y los esquemas MCP con su propio tope de bytes.
"""

from __future__ import annotations

import gzip
import http.client
import http.server
import json
import socket
import threading
import time
import urllib.error
import zlib

import pytest

from crotolamo.core.agent import ToolAgent
from crotolamo.core.budget import TOOL_RESULT_CAP
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.core.tool_parsing import is_hard_error
from crotolamo.tools.base import Registry, Tool, truncate_for_context

ENORME = "L" * 900_000  # find_in_project/list_project_tree sobre un repo con dependencias


class _LLM:
    """Pide una tool y luego contesta; guarda los bytes de cada request."""

    def __init__(self, *calls: dict):
        self._calls = list(calls)
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.requests.append(messages)
        if self._calls:
            return ChatResponse(content="", tool_calls=[self._calls.pop(0)])
        return ChatResponse(content="Ya está, patrón.")

    def request_bytes(self, i: int) -> int:
        return len(json.dumps(self.requests[i], ensure_ascii=False).encode("utf-8"))


class _Permite:
    def check(self, tool, args):
        class D:
            allowed = True
            needs_confirmation = False
            reason = ""
        return D()


class _Revienta:
    """Un guard que lanza (p.ej. un '\\0' en la ruta) con un mensaje enorme."""

    def check(self, tool, args):
        raise ValueError("ruta rara: " + ENORME)


def _registry(**salidas: str) -> Registry:
    reg = Registry()
    for name, salida in salidas.items():
        reg.register(Tool(
            name=name, func=lambda _s=salida: _s, description=name,
            parameters={"type": "object", "properties": {}, "required": []},
        ))
    return reg


def _agent(llm, registry, guard=None, **kw) -> ToolAgent:
    kw.setdefault("fastpath", False)
    return ToolAgent(llm, Conversation("SYS"), registry=registry, guard=guard or _Permite(),
                     pre_hooks=[], post_hooks=[], **kw)


def _tool_messages(agent: ToolAgent) -> list[str]:
    return [m.content for m in agent.conversation.history if m.role == "tool"]


def test_un_resultado_enorme_entra_al_historial_con_el_tope_y_la_marca_de_siempre():
    llm = _LLM({"name": "list_project_tree", "arguments": {}})
    agent = _agent(llm, _registry(list_project_tree=ENORME))

    assert agent.handle_turn("enséñame el árbol") == "Ya está, patrón."

    [guardado] = _tool_messages(agent)
    assert guardado == ENORME[:TOOL_RESULT_CAP] + "\n...[recortado, 900000 chars en total]"
    # Lo que se reduce: los bytes del request que lleva el resultado al modelo.
    assert llm.request_bytes(1) < TOOL_RESULT_CAP + 1_000
    assert llm.request_bytes(1) < len(ENORME) // 30


@pytest.mark.parametrize("largo", [0, 1, 20_037, TOOL_RESULT_CAP])
def test_lo_que_ya_cabe_no_cambia_un_byte(largo):
    salida = ("ñ" * largo) or "ok"
    llm = _LLM({"name": "read_file", "arguments": {}})
    agent = _agent(llm, _registry(read_file=salida))
    agent.handle_turn("lee")
    assert _tool_messages(agent) == [salida]


def test_el_camino_de_la_excepcion_tambien_se_acota_y_sigue_siendo_fallo_duro():
    llm = _LLM({"name": "read_file", "arguments": {}})
    agent = _agent(llm, _registry(read_file="no llega"), guard=_Revienta())
    agent.handle_turn("lee")

    [guardado] = _tool_messages(agent)
    completo = f"La tool 'read_file' reventó, patrón: ruta rara: {ENORME}"
    assert guardado == truncate_for_context(completo, TOOL_RESULT_CAP)
    assert is_hard_error(guardado)


def test_un_reventon_enorme_dentro_de_la_tool_sigue_siendo_fallo_duro():
    # Registry.run devuelve "La tool '...' reventó" citando la excepción;
    # recortado por la cola, el prefijo que lo marca no cambia.
    def revienta() -> str:
        raise RuntimeError(ENORME)

    llm = _LLM({"name": "read_file", "arguments": {}})
    reg = Registry()
    reg.register(Tool(name="read_file", func=revienta, description="lee",
                      parameters={"type": "object", "properties": {}, "required": []}))
    agent = _agent(llm, reg)
    agent.handle_turn("lee")
    [guardado] = _tool_messages(agent)
    completo = f"La tool 'read_file' reventó, patrón: {ENORME}"
    assert guardado == truncate_for_context(completo, TOOL_RESULT_CAP)
    assert is_hard_error(guardado)


def test_el_short_circuit_devuelve_lo_mismo_que_guarda():
    # Una tool de retorno directo (list_processes con limit=1000): su salida va
    # tal cual al patrón Y al historial (tool + assistant); las dos acotadas.
    salida = "PID COMMAND\n" + "1234 proceso\n" * 3_000
    llm = _LLM({"name": "list_processes", "arguments": {}})
    agent = _agent(llm, _registry(list_processes=salida), direct_tools={"list_processes"})

    reply = agent.handle_turn("procesos")

    assert len(llm.requests) == 1  # sin 2ª llamada, como siempre
    assert reply == truncate_for_context(salida, TOOL_RESULT_CAP)
    roles = [(m.role, m.content) for m in agent.conversation.history]
    assert roles[-2] == ("tool", reply) and roles[-1] == ("assistant", reply)


def test_el_fast_path_pasa_por_el_mismo_tope(monkeypatch):
    from crotolamo.core import fastpath

    monkeypatch.setattr(fastpath, "match", lambda text: ("list_windows", {}))

    class _SinLLM:
        def chat(self, *a, **k):
            raise AssertionError("el fast-path no llega al LLM")

    agent = _agent(_SinLLM(), _registry(list_windows=ENORME), fastpath=True)
    reply = agent.handle_turn("qué ventanas hay")
    assert reply == truncate_for_context(ENORME, TOOL_RESULT_CAP)
    assert agent.conversation.history[-1].content == reply


# ===========================================================================
# Paso 5: la web con presupuesto de bytes Y de tiempo. Antes, _http_get pedía
# el cuerpo sin comprimir (un artículo de 1.3 MB viajaba entero para quedarse
# con 6000 caracteres) y su plazo solo cubría el cuerpo de cada salto: cada
# redirect empezaba de cero y unas cabeceras que gotean no lo disparaban nunca.
# ===========================================================================

from crotolamo.tools import _web, search  # noqa: E402

_ARTICULO = ("<html><head><title>Artículo</title><script>" + "x" * 50_000 + "</script></head>"
             "<body><nav>menú</nav><article>"
             + "".join(f"<p>Párrafo {i}: el patrón lee un texto largo y repetido, con acentos "
                       f"y eñes, para que pese.</p>" for i in range(12_000))
             + "</article></body></html>").encode("utf-8")


class _Web(http.server.BaseHTTPRequestHandler):
    """Rutas de mentiras: /gz comprime si el cliente lo pide, /plano nunca,
    /bomba y /roto mandan un gzip que no es lo que dice ser."""

    protocol_version = "HTTP/1.1"
    rutas: dict[str, tuple[dict[str, str], bytes]] = {}
    vistos: list[tuple[str, str]] = []
    servidos: list[int] = []

    def do_GET(self):  # noqa: N802 — nombre que exige http.server
        self.vistos.append((self.path, self.headers.get("Accept-Encoding", "")))
        headers, body = self.rutas[self.path]
        self.send_response(200)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
            self.servidos.append(len(body))
        except OSError:  # el cliente colgó antes (tope alcanzado): bien
            self.servidos.append(-1)

    def handle(self):
        try:
            super().handle()
        except ConnectionError:  # el cliente colgó con bytes sin leer (tope): bien
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(_Web, "rutas", {})
    monkeypatch.setattr(_Web, "vistos", [])
    monkeypatch.setattr(_Web, "servidos", [])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Web)
    hilo = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                            daemon=True)
    hilo.start()
    monkeypatch.setattr(_web, "_resolve_public_ips", lambda host: ["127.0.0.1"])
    yield f"http://pagina.test:{srv.server_address[1]}", _Web
    srv.shutdown()
    srv.server_close()
    hilo.join(timeout=5)


def test_la_pagina_viaja_comprimida_y_se_lee_igual(web):
    base, srv = web
    html = {"Content-Type": "text/html; charset=utf-8"}
    srv.rutas["/plano"] = (html, _ARTICULO)
    srv.rutas["/gz"] = (dict(html, **{"Content-Encoding": "gzip"}), gzip.compress(_ARTICULO, 6))

    assert _web._http_get(f"{base}/gz", timeout=5.0) == _web._http_get(f"{base}/plano", 5.0)
    assert search.read_page(f"{base}/gz").replace("/gz", "/x") == \
        search.read_page(f"{base}/plano").replace("/plano", "/x")
    # El cliente lo pide, y por el cable viaja una fracción del artículo.
    assert all("gzip" in ae for _path, ae in srv.vistos)
    gz, plano = srv.servidos[0], srv.servidos[1]
    assert gz < plano // 20


def test_sin_content_encoding_el_cuerpo_no_se_toca(web):
    """Pedir gzip no obliga al servidor: si contesta sin comprimir (o con algo
    que no es gzip), el cuerpo llega como siempre."""
    base, srv = web
    srv.rutas["/a"] = ({"Content-Type": "text/plain"}, "ñandú\n".encode() * 10)
    srv.rutas["/b"] = ({"Content-Type": "text/plain", "Content-Encoding": "identity"},
                       b"tal cual")
    assert _web._http_get(f"{base}/a", 5.0) == ("text/plain", "ñandú\n" * 10)
    assert _web._http_get(f"{base}/b", 5.0) == ("text/plain", "tal cual")


def _bomba(descomprimido: int) -> bytes:
    """gzip de `descomprimido` ceros armado a trozos (sin tener los MB en RAM)."""
    comp = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    trozo = bytes(1 << 20)
    partes = [comp.compress(trozo) for _ in range(descomprimido >> 20)]
    return b"".join(partes) + comp.flush()


def test_una_bomba_gzip_se_corta_en_el_tope_descomprimido(web):
    import tracemalloc

    base, srv = web
    bomba = _bomba(200 << 20)  # 200 MB de ceros en ~200 KB
    srv.rutas["/bomba"] = ({"Content-Type": "text/plain", "Content-Encoding": "gzip"}, bomba)
    tracemalloc.start()
    try:
        _ct, body = _web._http_get(f"{base}/bomba", timeout=10.0)
        _actual, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(body) == _web._MAX_DOWNLOAD_BYTES  # el tope cuenta lo DESCOMPRIMIDO
    assert pico < 4 * _web._MAX_DOWNLOAD_BYTES     # no los 200 MB


def test_un_gzip_roto_es_un_fallo_de_conexion(web):
    base, srv = web
    srv.rutas["/roto"] = ({"Content-Type": "text/html", "Content-Encoding": "gzip"},
                          b"esto no es gzip")
    with pytest.raises(urllib.error.URLError):
        _web._http_get(f"{base}/roto", timeout=5.0)
    assert search.read_page(f"{base}/roto").startswith("No pude conectarme a esa página")


def test_un_gzip_cortado_se_lee_hasta_donde_llega(web):
    """Como un cuerpo sin comprimir cortado por el tope o el plazo: lo que se
    alcanzó a inflar es el principio de la página, y vale."""
    base, srv = web
    comprimido = gzip.compress(_ARTICULO, 6)
    srv.rutas["/cortado"] = ({"Content-Type": "text/html", "Content-Encoding": "gzip"},
                             comprimido[: len(comprimido) // 2])
    _ct, body = _web._http_get(f"{base}/cortado", timeout=5.0)
    assert 1000 < len(body) < len(_ARTICULO)
    assert _ARTICULO.decode().startswith(body)


def _servidor_crudo(guion):
    """Servidor TCP de un solo uso: `guion(conn)` decide qué mandar y cuándo."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    alto = threading.Event()

    def loop() -> None:
        while not alto.is_set():
            try:
                conn, _addr = srv.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.recv(65536)
                    guion(conn, alto)
                except OSError:
                    pass

    hilo = threading.Thread(target=loop, daemon=True)
    hilo.start()

    def parar() -> None:
        alto.set()
        srv.close()
        hilo.join(timeout=5)

    return srv.getsockname()[1], parar


def _get_en_hilo(url: str, timeout: float, espera: float):
    """(resultado o excepción, segundos) de _http_get, sin colgar el test."""
    salida: list[object] = []

    def correr() -> None:
        try:
            salida.append(_web._http_get(url, timeout=timeout))
        except Exception as error:  # noqa: BLE001 — se inspecciona abajo
            salida.append(error)

    t0 = time.monotonic()
    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(timeout=espera)
    assert not hilo.is_alive(), f"_http_get seguía {espera} s después con timeout={timeout}"
    return salida[0], time.monotonic() - t0


def test_unas_cabeceras_que_gotean_no_estiran_el_plazo(monkeypatch):
    """Un tarpit: la línea de estado y luego UNA cabecera que nunca acaba, un
    byte cada 50 ms. El timeout del socket es por recv y no salta nunca: antes
    solo paraba http.client al juntar 64 KB de cabecera (~55 minutos)."""

    def guion(conn, alto):
        conn.sendall(b"HTTP/1.1 200 OK\r\nX-Relleno: ")
        while not alto.wait(0.05):
            conn.sendall(b"a")

    port, parar = _servidor_crudo(guion)
    monkeypatch.setattr(_web, "_resolve_public_ips", lambda host: ["127.0.0.1"])
    try:
        error, segundos = _get_en_hilo(f"http://127.0.0.1:{port}/", timeout=0.6, espera=10)
    finally:
        parar()
    assert isinstance(error, TimeoutError)
    assert segundos < 0.6 * 2
    # read_page lo cuenta por la rama de siempre.
    assert isinstance(error, OSError)


def test_los_redirects_comparten_el_plazo(web):
    """Cada salto contestaba a tiempo, pero cada uno estrenaba su timeout: cinco
    redirects de 0.3 s con timeout=0.8 tardaban 1.5 s en vez de rendirse."""
    base, srv = web

    class _Lento(_Web):
        def do_GET(self):  # noqa: N802
            time.sleep(0.3)
            n = int(self.path.strip("/r") or 0)
            if n < 5:
                self.send_response(302)
                self.send_header("Location", f"/r{n + 1}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            super().do_GET()

    srv.rutas["/r5"] = ({"Content-Type": "text/plain"}, b"llegaste")
    lento = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Lento)
    hilo = threading.Thread(target=lento.serve_forever, kwargs={"poll_interval": 0.02},
                            daemon=True)
    hilo.start()
    try:
        url = f"http://pagina.test:{lento.server_address[1]}/r0"
        assert _web._http_get(url, timeout=5.0) == ("text/plain", "llegaste")
        error, segundos = _get_en_hilo(url, timeout=0.8, espera=10)
    finally:
        lento.shutdown()
        lento.server_close()
        hilo.join(timeout=5)
    assert isinstance(error, TimeoutError)
    assert segundos < 0.8 * 1.5


def test_el_ultimo_recv_del_cuerpo_no_se_pasa_del_plazo(monkeypatch):
    """Una gota cada 0.9 s con timeout=1.0: el plazo se miraba ENTRE lecturas y
    la que empezaba a 0.9 s esperaba la siguiente gota (1.8 s en total). Ahora
    el recv solo espera lo que queda, y lo que llegó se devuelve como siempre."""

    def guion(conn, alto):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n")
        while not alto.wait(0.9):
            conn.sendall(b"gota\n")

    port, parar = _servidor_crudo(guion)
    monkeypatch.setattr(_web, "_resolve_public_ips", lambda host: ["127.0.0.1"])
    try:
        resultado, segundos = _get_en_hilo(f"http://127.0.0.1:{port}/", timeout=1.0, espera=10)
    finally:
        parar()
    assert resultado == ("text/plain", "gota\n")
    assert segundos < 1.0 * 1.4


def test_un_cuerpo_que_no_llega_a_tiempo_es_timeout(monkeypatch):
    """Cabeceras a tiempo y ni un byte de cuerpo: no hay principio que leer,
    así que es un fallo de conexión (como el timeout del socket de siempre),
    no una página vacía."""

    def guion(conn, alto):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                     b"Content-Length: 10\r\n\r\n")
        alto.wait(5)

    port, parar = _servidor_crudo(guion)
    monkeypatch.setattr(_web, "_resolve_public_ips", lambda host: ["127.0.0.1"])
    try:
        error, segundos = _get_en_hilo(f"http://127.0.0.1:{port}/", timeout=0.5, espera=10)
    finally:
        parar()
    assert isinstance(error, TimeoutError)
    assert segundos < 0.5 * 2


# ===========================================================================
# Paso 6: las tools de la LAN (Home Assistant, Frigate) leían el cuerpo con un
# único read(tope): el timeout del socket es por recv, así que un servicio que
# gotea tenía el hilo del Brain esperando hasta juntar 0.5-1.5 MB. Ahora leen
# a trozos con el mismo lector que la web y un plazo total para el cuerpo.
# ===========================================================================

from crotolamo import settings as settings_mod  # noqa: E402
from crotolamo.tools import _frigate, _hass, cameras  # noqa: E402


def _json_que_gotea(conn, alto):
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                 b"Content-Length: 100000\r\n\r\n[")
    while not alto.wait(0.05):
        conn.sendall(b"1,")


def _llamar_en_hilo(fn, espera: float):
    salida: list[object] = []

    def correr() -> None:
        try:
            salida.append(fn())
        except Exception as error:  # noqa: BLE001 — se inspecciona abajo
            salida.append(error)

    t0 = time.monotonic()
    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(timeout=espera)
    assert not hilo.is_alive(), f"seguía {espera} s después"
    return salida[0], time.monotonic() - t0


@pytest.mark.parametrize("cliente", ["frigate", "hass"])
def test_un_servicio_de_la_lan_que_gotea_no_cuelga_la_tool(monkeypatch, cliente):
    """Cada gota llega mucho antes del timeout del socket: antes, read(1.5 MB)
    esperaba los 100 KB anunciados (~40 minutos a este ritmo)."""
    port, parar = _servidor_crudo(_json_que_gotea)
    url = f"http://127.0.0.1:{port}/api/x"
    if cliente == "frigate":
        def llamada():
            return _frigate._http_json(url, timeout=0.6)
    else:
        def llamada():
            return _hass._http_json("GET", url, {}, timeout=0.6)
    try:
        error, segundos = _llamar_en_hilo(llamada, espera=10)
    finally:
        parar()
    assert isinstance(error, TimeoutError)
    # Plazo del cuerpo + como mucho un recv más (urllib no suelta su socket).
    assert segundos < 0.6 * 3


def test_el_fallo_por_plazo_se_cuenta_como_conexion(monkeypatch):
    """La tool lo traduce con el código de siempre de un timeout ("conexion:
    timed out"), no como una respuesta rara."""
    port, parar = _servidor_crudo(_json_que_gotea)
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {"base_url": f"http://127.0.0.1:{port}"})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    real_http_json = _frigate._http_json
    monkeypatch.setattr(_frigate, "_http_json", lambda url: real_http_json(url, timeout=0.6))
    try:
        resultado, _segundos = _llamar_en_hilo(
            lambda: _frigate._frigate_get("/api/events", {"limit": 1}), espera=10)
    finally:
        parar()
    assert resultado == (False, "conexion: timed out")


class _JSON(http.server.BaseHTTPRequestHandler):
    cuerpo = b"[]"

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.cuerpo)))
        self.end_headers()
        try:
            self.wfile.write(self.cuerpo)
        except OSError:
            pass

    def do_POST(self):  # noqa: N802
        # Leer el cuerpo de la petición antes de contestar: si el servidor cierra
        # con bytes sin leer, el kernel manda un RST y el cliente podía recibir
        # ConnectionResetError a media respuesta (fallo intermitente del test).
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.do_GET()

    def handle(self):
        try:
            super().handle()
        except ConnectionError:
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def lan(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _JSON)
    hilo = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                            daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv
    srv.shutdown()
    srv.server_close()
    hilo.join(timeout=5)


def _espia_lecturas(monkeypatch) -> list[int]:
    """Bytes de CUERPO que http.client entrega a quien lee (read o read1)."""
    leidos: list[int] = []
    for nombre in ("read", "read1"):
        real = getattr(http.client.HTTPResponse, nombre)

        def espia(self, *a, _real=real, **kw):
            data = _real(self, *a, **kw)
            leidos.append(len(data))
            return data

        monkeypatch.setattr(http.client.HTTPResponse, nombre, espia)
    return leidos


@pytest.mark.parametrize("n", [0, 3, 4_000])
def test_la_lan_lee_el_mismo_json_de_siempre(lan, monkeypatch, n):
    base, srv = lan
    datos = [{"entity_id": f"light.l{i}", "state": "on", "nombre": "Salón ñ"} for i in range(n)]
    monkeypatch.setattr(_JSON, "cuerpo", json.dumps(datos, ensure_ascii=False).encode())
    assert _frigate._http_json(f"{base}/api/events") == datos
    assert _hass._http_json("GET", f"{base}/api/states", {}) == datos
    assert _hass._http_json("POST", f"{base}/api/services/x/y", {}, {"a": 1}) == datos


@pytest.mark.parametrize("modulo", [_frigate, _hass], ids=["frigate", "hass"])
def test_un_cuerpo_mayor_que_el_tope_se_corta_en_el_tope(lan, monkeypatch, modulo):
    """Lo de siempre: más allá del tope no se lee (y lo cortado no es JSON)."""
    base, _srv = lan
    monkeypatch.setattr(_JSON, "cuerpo", b"[" + b"1," * (modulo._MAX_BODY) + b"1]")
    leidos = _espia_lecturas(monkeypatch)
    with pytest.raises(json.JSONDecodeError):
        if modulo is _frigate:
            _frigate._http_json(f"{base}/api/events")
        else:
            _hass._http_json("GET", f"{base}/api/states", {})
    assert sum(leidos) == modulo._MAX_BODY


# ---------------------------------------------------------------------------
# camera_events/camera_snapshot piden include_thumbnails=0 (ya lo hacía la ola
# de arreglos). No hay un Frigate aquí para probar que TODAS las versiones lo
# aceptan: si uno lo rechazara (400/422) o contestara algo que no es la lista,
# se repite UNA vez la consulta sin el parámetro y la tool sigue funcionando.
# ---------------------------------------------------------------------------

_EVENTO = {"id": "ev1", "camera": "calle", "label": "car",
           "start_time": time.time() - 120, "end_time": time.time() - 110,
           "has_snapshot": True}


@pytest.fixture
def cams(monkeypatch):
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {"base_url": "http://frigate.test:5000",
                                              "names": {"entrada": "calle"}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)


def _frigate_que_rechaza(monkeypatch, rechazo):
    """_http_json falso: con include_thumbnails contesta `rechazo` (una
    excepción que se lanza o un valor que se devuelve); sin él, la lista."""
    urls: list[str] = []

    def fake(url, timeout=6.0):
        urls.append(url)
        if "include_thumbnails=" in url:
            if isinstance(rechazo, Exception):
                raise rechazo
            return rechazo
        return [_EVENTO]

    monkeypatch.setattr(_frigate, "_http_json", fake)
    return urls


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://frigate.test:5000/api/events", code, "x", {}, None)


@pytest.mark.parametrize("rechazo", [_http_error(400), _http_error(422),
                                     {"detail": "include_thumbnails no permitido"}],
                         ids=["400", "422", "no-lista"])
@pytest.mark.parametrize("llamada, esperado", [
    (lambda: cameras.camera_events("entrada", hours=6), "coche en la cámara calle"),
    (lambda: cameras.camera_snapshot("entrada"), "/api/events/ev1/snapshot.jpg"),
], ids=["events", "snapshot"])
def test_si_frigate_rechaza_el_parametro_se_repite_sin_el(cams, monkeypatch, rechazo,
                                                         llamada, esperado):
    urls = _frigate_que_rechaza(monkeypatch, rechazo)
    assert esperado in llamada()
    assert len(urls) == 2
    assert "include_thumbnails=0" in urls[0] and "include_thumbnails" not in urls[1]
    # La segunda es la de antes de la ola de arreglos: los mismos parámetros, sin ese.
    assert urls[1] == urls[0].replace("&include_thumbnails=0", "")


@pytest.mark.parametrize("rechazo", [_http_error(500), _http_error(404),
                                     urllib.error.URLError("connection refused")],
                         ids=["500", "404", "caido"])
def test_otros_fallos_no_se_reintentan(cams, monkeypatch, rechazo):
    urls = _frigate_que_rechaza(monkeypatch, rechazo)
    out = cameras.camera_events("entrada", hours=6)
    assert "No pude hablar con Frigate" in out
    assert len(urls) == 1


# ===========================================================================
# Paso 7: el esquema de una tool MCP lo escribe el server y viajaba entero en
# CADA request mientras la tool estuviera a la vista: ejemplos, títulos y
# descripciones de párrafo en cada propiedad (7 KB por una tool de 12 campos).
# Pasado MCP_SCHEMA_CAP se quitan las anotaciones que no cambian qué
# argumentos valen y, si aún no cabe, se recortan las descripciones anidadas.
# ===========================================================================

import copy  # noqa: E402
import logging  # noqa: E402

from crotolamo.mcp import bridge  # noqa: E402


def _bytes(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _sin_tope(schema) -> dict:
    """normalize_schema de antes del tope (lo que sigue haciendo con lo que cabe)."""
    out = dict(schema) if isinstance(schema, dict) else {}
    out.pop("$schema", None)
    out["type"] = "object"
    props = out.get("properties")
    out["properties"] = dict(props) if isinstance(props, dict) else {}
    if "required" in out and not isinstance(out["required"], list):
        out.pop("required")
    return out


def _esquema_verborreico(campos: int = 12) -> dict:
    """El del bench: 12 propiedades con descripción larga, ejemplos y título."""
    return {
        "type": "object",
        "properties": {f"campo_{k}": {
            "type": "string",
            "description": ("Descripción detallada del campo " * 12).strip(),
            "examples": [f"ejemplo {j}" for j in range(6)],
            "title": f"Campo {k}",
        } for k in range(campos)},
        "required": ["campo_0"],
    }


def _esqueleto(schema):
    """Lo que decide qué argumentos valen: nada de esto se toca nunca."""
    if isinstance(schema, list):
        return [_esqueleto(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    fuera = {"description", "examples", "title", "default", "$comment"}
    return {k: (_esqueleto(v) if k not in ("enum", "const") else v)
            for k, v in schema.items() if k not in fuera}


@pytest.mark.parametrize("schema", [
    None,
    {},
    {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
     "properties": {"q": {"type": "string", "title": "Consulta", "default": "x",
                          "examples": ["a", "b"], "description": "Lo que se busca."}},
     "required": ["q"], "additionalProperties": False},
    _esquema_verborreico(campos=3),  # con ejemplos y títulos, pero cabe
    {"type": "object", "properties": {"q": {"type": "string"}}, "required": "q"},
], ids=["none", "vacio", "chico-con-anotaciones", "tres-campos", "required-raro"])
def test_un_esquema_que_cabe_no_cambia_un_byte(schema, caplog):
    original = copy.deepcopy(schema)
    esperado = _sin_tope(schema)
    with caplog.at_level(logging.INFO, logger="crotolamo.mcp.bridge"):
        salida = bridge.normalize_schema(schema)
    assert _bytes(salida) <= bridge.MCP_SCHEMA_CAP
    assert json.dumps(salida, ensure_ascii=False) == json.dumps(esperado, ensure_ascii=False)
    assert schema == original  # la entrada del server no se toca
    assert not [r for r in caplog.records if "esquema" in r.getMessage()]


def test_el_esquema_del_bench_baja_de_7_kb_a_menos_del_tope(caplog):
    raw = [{"name": "crear_registro", "description": "Crea un registro. " * 120,
            "inputSchema": _esquema_verborreico()}]
    original = copy.deepcopy(raw)
    cfg = bridge.MCPServerConfig(name="demo", command=["true"], prefix="demo")
    with caplog.at_level(logging.INFO, logger="crotolamo.mcp.bridge"):
        [tool] = bridge._build_tools(Registry(), cfg, raw)
    antes = _bytes(_sin_tope(original[0]["inputSchema"]))
    despues = _bytes(tool.parameters)
    assert antes > 6_000
    assert despues <= bridge.MCP_SCHEMA_CAP
    assert _bytes(tool.schema()) < 3_000  # el bench medía 7166 B por request
    assert raw == original
    # Mismo esqueleto: tipos, propiedades, required. Solo cambia lo que documenta.
    assert _esqueleto(tool.parameters) == _esqueleto(original[0]["inputSchema"])
    for prop in tool.parameters["properties"].values():
        assert set(prop) == {"type", "description"}
        assert len(prop["description"]) <= 121 and prop["description"].endswith("…")
    # Una línea INFO con el antes y el después, con nombre de tool.
    [linea] = [r.getMessage() for r in caplog.records if "esquema" in r.getMessage()]
    assert "mcp_demo_crear_registro" in linea and str(antes) in linea and str(despues) in linea


def test_si_quitar_anotaciones_basta_las_descripciones_quedan_enteras():
    schema = {"type": "object", "properties": {
        f"c{k}": {"type": "integer", "description": "Un número.",
                  "examples": list(range(200))} for k in range(5)}}
    assert _bytes(schema) > bridge.MCP_SCHEMA_CAP
    salida = bridge.normalize_schema(schema)
    assert _bytes(salida) <= bridge.MCP_SCHEMA_CAP
    assert salida["properties"] == {
        f"c{k}": {"type": "integer", "description": "Un número."} for k in range(5)}


def test_las_palabras_clave_solo_se_quitan_donde_son_anotaciones():
    """Una PROPIEDAD llamada title/default/examples (un create_issue de GitHub
    tiene `title`) es un argumento, no una anotación; y lo que hay dentro de
    enum/const son valores. Tampoco se toca lo que no es esquema conocido."""
    largo = "Texto que documenta y ocupa. " * 8
    schema = {
        "type": "object",
        "description": "Descripción de arriba: " + largo,
        "properties": {
            "title": {"type": "string", "description": largo, "title": "Título"},
            "default": {"type": "boolean", "default": False, "description": largo},
            "examples": {"type": "array", "items": {"type": "string", "examples": ["x"] * 50,
                                                    "description": largo}},
            "modo": {"enum": ["title", "default", "examples"], "description": largo},
            "fijo": {"const": {"title": "se queda", "default": 1}},
            "union": {"anyOf": [{"type": "string", "title": "A", "description": largo},
                                {"$ref": "#/$defs/B"}]},
            "mapa": {"type": "object", "additionalProperties": {
                "type": "number", "title": "valor", "description": largo}},
        },
        "$defs": {"B": {"type": "object", "title": "B", "description": largo,
                        "properties": {"default": {"type": "string", "description": largo}}}},
        "x-vendor": {"title": "no es esquema: se queda", "examples": [1, 2]},
        "required": ["title", "default"],
        "additionalProperties": False,
    }
    schema["properties"].update({f"relleno_{i}": {"type": "string", "description": largo,
                                                  "examples": ["e" * 30] * 4}
                                 for i in range(12)})
    salida = bridge.normalize_schema(copy.deepcopy(schema))
    props = salida["properties"]
    assert set(props) == set(schema["properties"])          # ningún argumento se pierde
    assert salida["required"] == ["title", "default"]
    assert salida["additionalProperties"] is False
    assert props["modo"]["enum"] == ["title", "default", "examples"]
    assert props["fijo"] == {"const": {"title": "se queda", "default": 1}}
    assert "title" not in props["title"] and "default" not in props["default"]
    assert "examples" not in props["examples"]["items"]
    assert "title" not in props["union"]["anyOf"][0]
    assert props["union"]["anyOf"][1] == {"$ref": "#/$defs/B"}
    assert "title" not in props["mapa"]["additionalProperties"]
    assert set(salida["$defs"]["B"]["properties"]) == {"default"}
    assert "title" not in salida["$defs"]["B"]
    assert salida["x-vendor"] == schema["x-vendor"]
    # La descripción de arriba es la de la tool entera: entera. Las anidadas, a 120.
    assert salida["description"] == schema["description"]
    assert len(props["union"]["anyOf"][0]["description"]) <= 121
    assert len(salida["$defs"]["B"]["properties"]["default"]["description"]) <= 121
    assert _esqueleto(salida) == _esqueleto(schema)


# ===========================================================================
# Paso 8: cada cliente ajusta el request a SU ventana. Conversation guarda lo
# mismo de siempre y los dos clientes lo serializaban tal cual: con num_ctx=2048
# el local recibía ~4 600 tokens por request y Ollama recortaba EN SILENCIO
# (el principio del prompt, persona incluida); en la nube, unas cuantas lecturas
# de archivo pasaban de los 128k tokens y el 400 permanente tumbaba cada turno
# hasta que 20 turnos más deslizaban los resultados fuera. Ahora el local se
# ajusta a num_ctx y la nube a [llm.glm].max_context_tokens, sin tocar el
# historial: cuando la nube vuelve, ve la conversación entera otra vez.
# ===========================================================================

from crotolamo.core import budget  # noqa: E402
from crotolamo.core.fallback import FallbackLLM  # noqa: E402
from crotolamo.core.glm import GLMClient  # noqa: E402
from crotolamo.core.llm import LLMClient  # noqa: E402
from crotolamo.core.openai_adapter import to_openai_messages  # noqa: E402

# Persona y 8 esquemas algo más chicos que los reales (2.8 KB y 2.5 KB: ~1 660
# de los 1 792 tokens estimados que deja num_ctx=2048). Con los reales, un turno
# con lectura apenas cabe en la estimación (que peca de grande) y el test no
# podría exigir que quepa.
_PERSONA = "Eres Crotolamo, el asistente del patrón; contestas corto y en personaje. " * 24
_ESQUEMAS = [
    {"type": "function", "function": {
        "name": f"tool_{i}", "description": "Hace algo útil para el patrón. " * 2,
        "parameters": {"type": "object", "properties": {
            "ruta": {"type": "string", "description": "ruta del archivo"}},
            "required": ["ruta"]}}}
    for i in range(8)
]


class _Contesta:
    """Respuesta HTTP mínima: un 200 con el cuerpo de un motor (o líneas de stream)."""

    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self.status = 200
        self._body = body
        self._lines = lines or []

    def read(self, *_a):
        body, self._body = self._body, b""
        return body

    def getheader(self, name, default=None):
        return default

    def isclosed(self):
        return True

    def __iter__(self):
        return iter(self._lines)


class _Cable:
    """Sustituye a HTTPTransport: guarda cada payload; falla si se le pide."""

    def __init__(self, kind: str):
        self.kind = kind
        self.payloads: list[dict] = []
        self.falla: Exception | None = None

    def post_json(self, path, payload, headers=None):
        # Lo que de verdad viaja: serializado en el momento, como el transporte.
        self.payloads.append(json.loads(json.dumps(payload, ensure_ascii=False)))
        if self.falla is not None:
            raise self.falla
        if self.kind == "glm":
            if payload.get("stream"):
                return _Contesta(lines=[
                    b'data: {"choices":[{"delta":{"content":"Listo."},'
                    b'"finish_reason":"stop"}]}', b"data: [DONE]"])
            return _Contesta(json.dumps({"choices": [{"message": {
                "role": "assistant", "content": "Listo, patrón."}}]}).encode())
        if payload.get("stream"):
            return _Contesta(lines=[json.dumps({
                "message": {"role": "assistant", "content": "Listo."}, "done": True}).encode()])
        return _Contesta(json.dumps({"message": {
            "role": "assistant", "content": "Listo, patrón."}, "done": True}).encode())

    def finish(self, resp):
        pass

    def close(self):
        pass


def _local(num_ctx: int | None = 2048) -> tuple[LLMClient, _Cable]:
    cliente = LLMClient(num_ctx=num_ctx)
    cable = _Cable("ollama")
    cliente._transport = cable  # type: ignore[assignment]
    return cliente, cable


def _nube(monkeypatch, **kw) -> tuple[GLMClient, _Cable]:
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    cliente = GLMClient(**kw)
    cable = _Cable("glm")
    cliente._transport = cable  # type: ignore[assignment]
    return cliente, cable


def _conversacion(turnos: int, chars: int, lecturas: int = 1) -> Conversation:
    """La historia como la deja ToolAgent: user, assistant que pide tools, sus
    resultados y la respuesta; el último turno, en curso (sin respuesta)."""
    conv = Conversation(_PERSONA, max_turns=10_000)
    for i in range(turnos):
        conv.add_user(f"[ahora: 2026-11-14 10:{i % 60:02d}]\nlee los archivos, vuelta {i}")
        conv.add_assistant("", tool_calls=[
            {"function": {"name": "tool_0", "arguments": {"ruta": f"/p/{i}_{k}.py"}}}
            for k in range(lecturas)])
        for k in range(lecturas):
            conv.add_tool_result("tool_0", f"{i:03d}-{k}" + "x" * (chars - 5))
        if i < turnos - 1:
            conv.add_assistant(f"Ya leí la vuelta {i}, patrón.")
    return conv


def _sin_huerfanos(msgs: list[dict]) -> bool:
    # El adaptador OpenAI descarta un 'tool' sin su assistant delante.
    return len(to_openai_messages(msgs)) == len(msgs)


@pytest.mark.parametrize("stream", [False, True])
def test_el_local_ajusta_el_request_a_su_num_ctx(stream, caplog):
    conv = _conversacion(30, 20_000)
    msgs = conv.to_messages()
    antes = json.dumps(msgs, ensure_ascii=False)
    assert budget.est_tokens(msgs, _ESQUEMAS) > 30 * 2048  # muy por encima de la ventana
    cliente, cable = _local(2048)

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.llm"):
        if stream:
            cliente.chat_stream(msgs, tools=_ESQUEMAS, on_token=lambda _t: None)
        else:
            cliente.chat(msgs, tools=_ESQUEMAS)

    enviado = cable.payloads[0]["messages"]
    # Cabe, con sitio para la respuesta, y las tools del turno siguen ahí.
    assert budget.est_tokens(enviado, _ESQUEMAS) <= 2048 - budget.REPLY_RESERVE
    assert len(cable.payloads[0]["tools"]) == len(_ESQUEMAS)
    assert cable.payloads[0]["options"]["num_ctx"] == 2048
    # La persona entera y primero; el turno en curso, presente; nada huérfano.
    assert enviado[0] == {"role": "system", "content": _PERSONA}
    assert enviado[-3]["content"].endswith("vuelta 29")
    assert enviado[-1]["role"] == "tool" and enviado[-1]["content"].startswith("029-0")
    assert _sin_huerfanos(enviado)
    # El resultado en curso se recorta con la marca de siempre (dice cuánto había).
    assert "[recortado, 20000 chars en total]" in enviado[-1]["content"]
    # El historial no se toca: la nube lo verá entero cuando vuelva.
    assert json.dumps(conv.to_messages(), ensure_ascii=False) == antes
    # Y queda dicho en el log, con los números para decidir num_ctx.
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    texto = avisos[0].getMessage()
    assert "num_ctx=2048" in texto and "bloque" in texto and "recort" in texto


def test_el_local_que_ya_cabe_va_tal_cual(caplog):
    # Un resultado más largo que el tope de tool del local (2 365 con 2048), pero
    # el request cabe: no se toca nada (es lo que ya funcionaba).
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "lee"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "tool_0", "arguments": {}}}]},
            {"role": "tool", "content": "y" * 3_000, "name": "tool_0"}]
    assert len(msgs[-1]["content"]) > budget.local_tool_cap(2048)
    cliente, cable = _local(2048)
    payload = cliente._payload(msgs, None, stream=False)
    assert payload["messages"] is msgs

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.llm"):
        cliente.chat(msgs)
    assert cable.payloads[0]["messages"] == msgs
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_sin_num_ctx_el_local_no_ajusta_nada():
    # Sin num_ctx no hay ventana que conocer: decide Ollama, como siempre.
    msgs = _conversacion(10, 20_000).to_messages()
    cliente, _ = _local(None)
    assert cliente._payload(msgs, _ESQUEMAS, stream=False)["messages"] is msgs


def test_el_tope_de_tool_del_local_es_el_35_por_ciento_de_la_ventana():
    assert budget.local_tool_cap(2048) == 2365
    assert budget.local_tool_cap(8192) == int(0.35 * 8192 * 3.3)
    assert budget.local_tool_cap(512) == 1200  # nunca menos de 1 200


def _payload_de_siempre(cliente: GLMClient, msgs, tools, stream) -> dict:
    """El payload GLM de antes del paso 8, armado a mano."""
    payload = {"model": cliente.model, "messages": to_openai_messages(msgs),
               "temperature": cliente.temperature, "stream": stream}
    if tools:
        payload["tools"] = tools
    if cliente.max_tokens:
        payload["max_tokens"] = cliente.max_tokens
    if not cliente.thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


@pytest.mark.parametrize("turnos", [1, 20])
def test_la_nube_bajo_su_ventana_no_cambia_un_byte(monkeypatch, caplog, turnos):
    msgs = _conversacion(turnos, 20_000).to_messages()
    cliente, cable = _nube(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.glm"):
        cliente.chat(msgs, tools=_ESQUEMAS)
    esperado = json.dumps(_payload_de_siempre(cliente, msgs, _ESQUEMAS, False),
                          ensure_ascii=False)
    assert json.dumps(cable.payloads[0], ensure_ascii=False) == esperado
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("stream", [False, True])
def test_la_nube_tiene_red_de_seguridad_de_128k(monkeypatch, caplog, stream):
    # 25 turnos x 3 lecturas de 20 KB: ~1.5 MB, muy por encima de 128k tokens.
    conv = _conversacion(25, 20_000, lecturas=3)
    msgs = conv.to_messages()
    antes = json.dumps(msgs, ensure_ascii=False)
    assert budget.est_tokens(msgs, _ESQUEMAS) > 128_000
    cliente, cable = _nube(monkeypatch)
    assert cliente.max_context_tokens == 128_000

    with caplog.at_level(logging.WARNING, logger="crotolamo.core.glm"):
        if stream:
            cliente.chat_stream(msgs, tools=_ESQUEMAS, on_token=lambda _t: None)
        else:
            cliente.chat(msgs, tools=_ESQUEMAS)

    enviado = cable.payloads[0]["messages"]
    assert budget.est_tokens(enviado, _ESQUEMAS) < 128_000
    assert enviado[0] == {"role": "system", "content": _PERSONA}
    # Solo se tira historia vieja por bloques enteros: nada recortado, nada huérfano.
    assert all("[recortado" not in (m.get("content") or "") for m in enviado)
    ids = [c["id"] for m in enviado for c in m.get("tool_calls") or []]
    assert [m["tool_call_id"] for m in enviado if m["role"] == "tool"] == ids
    assert enviado[-1]["content"].startswith("024-2")
    assert json.dumps(conv.to_messages(), ensure_ascii=False) == antes
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_max_context_tokens_sale_de_la_config(monkeypatch):
    class _S:
        llm = {"glm": {"max_context_tokens": 50_000}}

    class _SinClave:
        llm: dict = {"glm": {}}

    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    assert GLMClient.from_settings(_S()).max_context_tokens == 50_000
    assert GLMClient.from_settings(_SinClave()).max_context_tokens == 128_000


def test_tras_la_caida_la_nube_vuelve_a_ver_la_historia_entera(monkeypatch):
    # ~8k tokens: cabe de sobra en la nube, no en el local de 2048.
    conv = _conversacion(4, 6_000)
    nube, cable_nube = _nube(monkeypatch)
    local, cable_local = _local(2048)
    motor = FallbackLLM(nube, local, cooldown_s=0.0)

    cable_nube.falla = TimeoutError("sin internet")
    motor.chat(conv.to_messages(), tools=_ESQUEMAS)
    cable_nube.falla = None
    motor.chat(conv.to_messages(), tools=_ESQUEMAS)

    al_local = cable_local.payloads[0]["messages"]
    assert budget.est_tokens(al_local, _ESQUEMAS) <= 2048 - budget.REPLY_RESERVE
    assert _sin_huerfanos(al_local)
    # La nube, de vuelta, recibe TODO: el ajuste del local no tocó la Conversation.
    assert cable_nube.payloads[1]["messages"] == to_openai_messages(conv.to_messages())


def test_la_config_de_fabrica_documenta_max_context_tokens():
    import re
    import tomllib

    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    glm = re.split(r"^\[llm\.glm\]$", texto, maxsplit=1, flags=re.M)[1]
    glm = re.split(r"^\[", glm, maxsplit=1, flags=re.M)[0]
    m = re.search(r"^#\s*max_context_tokens\s*=\s*(\d+)", glm, re.M)
    assert m, "falta max_context_tokens comentado con su valor de fábrica"
    assert int(m.group(1)) == budget.DEFAULT_CLOUD_CONTEXT_TOKENS
    # Comentada: el default sale del código, no de un valor puesto en el toml.
    assert "max_context_tokens" not in tomllib.loads(texto)["llm"]["glm"]


# ===========================================================================
# Paso 10: los recuerdos de la memoria semántica se anteponían al mensaje del
# patrón en CADA turno y ese texto se guarda en el historial: con top_k=3, el
# request del turno 25 llevaba 60 líneas de recuerdos, 54 repetidas. Ahora el
# pre-hook no repite un recuerdo que el modelo ya va a ver en la ventana; cuando
# esa copia sale de la ventana, el recuerdo vuelve a entrar al recuperarse.
# ===========================================================================

from crotolamo.core import memoria as memoria_mod  # noqa: E402
from crotolamo.core.memoria import (  # noqa: E402
    Memoria,
    MemoriaConfig,
    Recuerdo,
    make_memoria_prehook,
)

_CABECERA = "[recuerdos sobre el patrón que vienen al caso:\n"


class _Recuerda:
    """Doble de mem0: devuelve SIEMPRE los mismos recuerdos, en orden."""

    def __init__(self, *textos: str):
        self.textos = list(textos)

    def buscar(self, query, top_k, umbral):
        return [Recuerdo(f"m{i}", t, 0.05) for i, t in enumerate(self.textos)][:top_k]

    def extraer(self, usuario, respuesta):
        return []

    def todas(self):
        return []


def _memoria_de(*textos: str) -> Memoria:
    return Memoria(MemoriaConfig(enabled=True, umbral=0.03, top_k=3),
                   backend=_Recuerda(*textos))


def _lineas_de_recuerdo(mensajes) -> list[str]:
    lineas: list[str] = []
    for m in mensajes:
        contenido = m["content"] if isinstance(m, dict) else m.content
        if _CABECERA in (contenido or ""):
            bloque = contenido.split(_CABECERA, 1)[1].split("]\n\n", 1)[0]
            lineas += bloque.split("\n")
    return lineas


def test_sin_conversacion_el_prehook_es_el_de_siempre():
    m = _memoria_de("Le gusta el café de olla", "Su perro se llama Chispas")
    assert make_memoria_prehook(m)("hola") == (
        _CABECERA + "- Le gusta el café de olla\n- Su perro se llama Chispas]\n\nhola")


def test_lo_que_ya_esta_en_la_ventana_no_se_repite():
    m = _memoria_de("Le gusta el café de olla", "Su perro se llama Chispas", "Vive en Puebla")
    conv = Conversation("SYS")
    conv.add_user("[ahora: 2026-11-14 10:00]\n" + _CABECERA
                  + "- Le gusta el café de olla\n- Vive en Puebla]\n\nhola")
    conv.add_assistant("Qué onda, patrón.")
    hook = make_memoria_prehook(m, conv)
    # Solo el que falta, con el mismo formato de siempre.
    assert hook("y mi perro?") == _CABECERA + "- Su perro se llama Chispas]\n\ny mi perro?"
    # Si ya están todos, el texto va tal cual: nada que anteponer.
    conv.add_user(hook("y mi perro?"))
    conv.add_assistant("Chispas, patrón.")
    assert hook("otra vez") == "otra vez"


def test_un_recuerdo_parecido_no_cuenta_como_el_mismo():
    # «- Vive en Puebla» no está en la ventana aunque esté «- Vive en Puebla de Zaragoza».
    m = _memoria_de("Vive en Puebla")
    conv = Conversation("SYS")
    conv.add_user(_CABECERA + "- Vive en Puebla de Zaragoza]\n\nhola")
    assert make_memoria_prehook(m, conv)("dónde vivo") == (
        _CABECERA + "- Vive en Puebla]\n\ndónde vivo")


def test_si_la_unica_copia_sale_de_la_ventana_con_este_turno_se_repite():
    m = _memoria_de("Le gusta el café de olla")
    conv = Conversation("SYS", max_turns=2)
    hook = make_memoria_prehook(m, conv)
    conv.add_user(hook("café"))                 # turno 1: entra el recuerdo
    conv.add_assistant("Va.")
    conv.add_user(hook("y luego"))              # turno 2: ya está, no se repite
    conv.add_assistant("Va.")
    assert _lineas_de_recuerdo(conv.to_messages()) == ["- Le gusta el café de olla"]
    # Turno 3: su add_user tira el bloque del turno 1, la única copia: se repite.
    conv.add_user(hook("y ahora"))
    assert _lineas_de_recuerdo(conv.to_messages()) == ["- Le gusta el café de olla"]
    assert conv.to_messages()[-1]["content"].startswith(_CABECERA)


def test_si_el_motor_no_ve_la_ventana_entera_se_repite_como_siempre():
    # El local de 2048 tira los turnos viejos del request (paso 8): ahí no se
    # puede contar con la copia de un turno anterior.
    m = _memoria_de("Le gusta el café de olla")
    conv = Conversation("SYS")
    conv.add_user(make_memoria_prehook(m)("café"))
    entera = {"si": False}
    hook = make_memoria_prehook(m, conv, ventana_entera=lambda: entera["si"])
    assert hook("y luego").startswith(_CABECERA)
    entera["si"] = True
    assert hook("y luego") == "y luego"


class _GLMGraba:
    """El LLM de build_agent: graba los mensajes de cada llamada."""

    model = "glm"

    def __init__(self):
        self.mensajes: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.mensajes.append(messages)
        return ChatResponse(content="Va, patrón.")


def _build_agent_memoria(monkeypatch, tmp_path, backend: str, memoria: Memoria):
    from crotolamo.core import hooks
    from crotolamo.settings import Settings
    from interfaces import shell

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": backend}}, user="test",
                                 home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria", lambda: memoria)
    monkeypatch.setattr(memoria_mod, "mem0_instalado", lambda: True)
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    monkeypatch.setattr(memoria, "precalentar", lambda: None)
    nube, local = _GLMGraba(), _GLMGraba()
    llm = FallbackLLM(nube, local) if backend == "glm" else local
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, conv = shell.build_agent()
    return agent, conv, nube, local, llm


def test_build_agent_con_la_nube_no_repite_recuerdos(monkeypatch, tmp_path):
    m = _memoria_de("Le gusta el café de olla", "Su perro se llama Chispas", "Vive en Puebla")
    agent, conv, nube, _local, _ = _build_agent_memoria(monkeypatch, tmp_path, "glm", m)
    for i in range(25):
        agent.handle_turn(f"cuéntame algo, vuelta {i}")
    lineas = _lineas_de_recuerdo(nube.mensajes[-1])
    # Cada recuerdo, una vez en la ventana (antes: 3 por turno, 60 en el turno 25).
    assert sorted(lineas) == sorted(f"- {t}" for t in _Recuerda(
        "Le gusta el café de olla", "Su perro se llama Chispas", "Vive en Puebla").textos)
    m.cerrar()


def test_build_agent_degradado_al_local_repite_como_siempre(monkeypatch, tmp_path):
    m = _memoria_de("Le gusta el café de olla")
    agent, conv, nube, local, llm = _build_agent_memoria(monkeypatch, tmp_path, "glm", m)
    agent.handle_turn("hola")
    llm._retry_at = time.monotonic() + 60  # breaker abierto: responde el local
    agent.handle_turn("y luego")
    assert len(_lineas_de_recuerdo(local.mensajes[-1])) == 2
    m.cerrar()


def test_build_agent_con_ollama_repite_como_siempre(monkeypatch, tmp_path):
    m = _memoria_de("Le gusta el café de olla")
    agent, conv, _nube, local, _ = _build_agent_memoria(monkeypatch, tmp_path, "ollama", m)
    for i in range(3):
        agent.handle_turn(f"vuelta {i}")
    assert len(_lineas_de_recuerdo(local.mensajes[-1])) == 3
    m.cerrar()


class _ConversacionDeAntes(Conversation):
    """La Conversation con el _trim de antes de este paso, copiado tal cual."""

    def _trim(self) -> None:
        with self._lock:
            user_count = sum(1 for m in self._history if m.role == "user")
            while user_count > self.max_turns and self._history:
                block = [self._history.pop(0)]
                if block[0].role == "user":
                    user_count -= 1
                while self._history and self._history[0].role in {"assistant", "tool"}:
                    block.append(self._history.pop(0))
                if self.compaction and self.summarizer is not None:
                    self._absorb_into_summary(block)


def _al_azar(conv: Conversation, rnd, pasos: int) -> None:
    for i in range(pasos):
        tirada = rnd.random()
        if tirada < 0.35:
            conv.add_user(f"u{i}")
        elif tirada < 0.6:
            conv.add_assistant(f"a{i}")
        elif tirada < 0.75:
            conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
        elif tirada < 0.97:
            conv.add_tool_result("t", f"r{i}")
        else:
            conv.reset()


@pytest.mark.parametrize("semilla", range(40))
def test_el_recorte_es_el_de_siempre(semilla):
    import random

    for compaction in (False, True):
        resumenes: dict[str, list[str]] = {"antes": [], "ahora": []}

        def _resume(clave):
            def resume(texto):
                resumenes[clave].append(texto)
                return f"R{len(resumenes[clave])}"
            return resume

        max_turns = 1 + semilla % 4
        antes = _ConversacionDeAntes("SYS", max_turns=max_turns, compaction=compaction,
                                     summarizer=_resume("antes"))
        ahora = Conversation("SYS", max_turns=max_turns, compaction=compaction,
                             summarizer=_resume("ahora"))
        _al_azar(antes, random.Random(semilla), 120)
        _al_azar(ahora, random.Random(semilla), 120)
        assert ahora.to_messages() == antes.to_messages()
        assert resumenes["ahora"] == resumenes["antes"]


@pytest.mark.parametrize("semilla", range(40))
def test_lo_que_sobrevive_al_proximo_turno_es_lo_que_queda(semilla):
    import random

    rnd = random.Random(semilla)
    conv = Conversation("SYS", max_turns=1 + semilla % 4)
    _al_azar(conv, rnd, rnd.randrange(1, 60))
    previsto = conv.surviving_user_texts()
    conv.add_user("NUEVO")
    assert previsto == [m.content for m in conv.history if m.role == "user"][:-1]


# ===========================================================================
# Paso 11: con [memory].compaction, cada turno con la ventana llena hacía una
# llamada al LLM para resumir DENTRO de add_user, antes de la llamada del turno:
# un viaje de ida y vuelta entero antes de la primera palabra, con el lock de la
# Conversation tomado. Ahora el bloque que sale queda pendiente (el modelo lo
# sigue viendo tal cual, tras el resumen) y lo resume un hilo después de hablar.
# Si el resumidor falla, el bloque espera al siguiente intento (antes se perdía).
# ===========================================================================

from crotolamo.core.memory import CompactionWorker  # noqa: E402


class _Resumidor:
    def __init__(self, falla: bool = False, espera: threading.Event | None = None):
        self.entradas: list[str] = []
        self.hilos: list[str] = []
        self.falla = falla
        self.espera = espera
        self.dentro = threading.Event()

    def __call__(self, texto: str) -> str:
        self.entradas.append(texto)
        self.hilos.append(threading.current_thread().name)
        self.dentro.set()
        if self.espera is not None:
            self.espera.wait(timeout=10)
        if self.falla:
            raise RuntimeError("la nube no contesta")
        return f"RESUMEN {len(self.entradas)}"


def _conv_fondo(resumidor, max_turns: int = 2) -> Conversation:
    return Conversation("SYS", max_turns=max_turns, compaction=True, summarizer=resumidor,
                        summarize_in_background=True)


def _turno(conv: Conversation, i: int, resultado: str | None = None) -> None:
    conv.add_user(f"u{i}")
    if resultado is not None:
        conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
        conv.add_tool_result("t", resultado)
    conv.add_assistant(f"a{i}")


def test_en_segundo_plano_add_user_no_llama_al_resumidor():
    resumidor = _Resumidor()
    conv = _conv_fondo(resumidor)
    for i in range(5):
        _turno(conv, i)
    assert resumidor.entradas == []
    # Los bloques que salieron siguen a la vista, tal cual y en orden: el modelo
    # nunca ve menos que antes.
    assert [m["content"] for m in conv.to_messages()] == [
        "SYS", "u0", "a0", "u1", "a1", "u2", "a2", "u3", "a3", "u4", "a4"]
    assert [m.content for m in conv.history] == ["u3", "a3", "u4", "a4"]


def test_compact_pending_resume_todo_lo_pendiente_en_una_llamada():
    resumidor = _Resumidor()
    conv = _conv_fondo(resumidor)
    _turno(conv, 0, resultado="R" * 5_000)
    for i in range(1, 4):
        _turno(conv, i)
    assert conv.compact_pending() is True
    assert len(resumidor.entradas) == 1
    # Mismo formato que el resumen de siempre; el resultado de tool, recortado a
    # 500 con la marca de siempre (el assistant que solo pide la tool no tiene texto).
    assert resumidor.entradas[0] == (
        "user: u0\ntool: " + truncate_for_context("R" * 5_000, 500)
        + "\nassistant: a0\nuser: u1\nassistant: a1")
    msgs = conv.to_messages()
    assert msgs[1] == {"role": "system",
                       "content": "[resumen de la conversación previa]\nRESUMEN 1"}
    assert [m["content"] for m in msgs[2:]] == ["u2", "a2", "u3", "a3"]
    # Nada pendiente: otra llamada no hace nada.
    assert conv.compact_pending() is False
    # El siguiente resumen parte del anterior: UN solo mensaje de resumen.
    _turno(conv, 4)
    assert conv.compact_pending() is True
    assert resumidor.entradas[1].startswith(
        "[resumen de la conversación previa]\nRESUMEN 1\nuser: u2")
    resumenes = [m for m in conv.to_messages() if "resumen" in m["content"].lower()]
    assert len(resumenes) == 1


def test_si_el_resumidor_falla_el_bloque_espera_y_queda_a_la_vista(caplog):
    resumidor = _Resumidor(falla=True)
    conv = _conv_fondo(resumidor)
    for i in range(3):
        _turno(conv, i)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memory"):
        assert conv.compact_pending() is False
    assert "resum" in caplog.text
    assert [m["content"] for m in conv.to_messages()][:3] == ["SYS", "u0", "a0"]
    # Cuando vuelve, se resume lo que esperaba (antes se había perdido).
    resumidor.falla = False
    _turno(conv, 3)
    assert conv.compact_pending() is True
    assert resumidor.entradas[-1].startswith("user: u0\nassistant: a0\nuser: u1")


def test_lo_pendiente_tiene_tope_y_el_historial_sigue_acotado():
    conv = _conv_fondo(_Resumidor(falla=True))
    for i in range(30):
        _turno(conv, i)
        conv.compact_pending()
    users = [m for m in conv.to_messages() if m["role"] == "user"]
    assert len(users) == conv.max_turns + Conversation.MAX_PENDING_BLOCKS
    # Se tiran los más viejos, como hacía el resumen fallido de antes.
    assert users[-1]["content"] == "u29"


def test_reset_durante_el_resumen_descarta_el_resultado():
    suelta = threading.Event()
    resumidor = _Resumidor(espera=suelta)
    conv = _conv_fondo(resumidor)
    for i in range(3):
        _turno(conv, i)
    hilo = threading.Thread(target=conv.compact_pending)
    hilo.start()
    assert resumidor.dentro.wait(5)
    conv.reset()
    conv.add_user("nuevo")
    suelta.set()
    hilo.join(5)
    assert conv.to_messages() == [{"role": "system", "content": "SYS"},
                                  {"role": "user", "content": "nuevo"}]


def test_el_turno_no_espera_al_resumen_en_curso():
    suelta = threading.Event()
    resumidor = _Resumidor(espera=suelta)
    conv = _conv_fondo(resumidor)
    for i in range(3):
        _turno(conv, i)
    hilo = threading.Thread(target=conv.compact_pending)
    hilo.start()
    assert resumidor.dentro.wait(5)
    try:
        t0 = time.monotonic()
        _turno(conv, 3)                    # add_user/add_assistant con el resumidor colgado
        conv.to_messages()
        assert time.monotonic() - t0 < 0.5
    finally:
        suelta.set()
        hilo.join(5)
    # Lo que entró durante el resumen no se pierde: queda pendiente para el siguiente.
    assert [m["content"] for m in conv.to_messages()] == [
        "SYS", "[resumen de la conversación previa]\nRESUMEN 1", "u1", "a1",
        "u2", "a2", "u3", "a3"]


def test_el_worker_corre_fuera_del_turno_en_un_solo_hilo_hasta_close():
    """UN hilo persistente para todos los resúmenes (revisión: un hilo nuevo por
    resumen era una conexión TCP/TLS nueva por resumen, porque el transporte
    guarda su keep-alive por hilo). Entre resumen y resumen duerme sin sondear;
    close() lo despierta y sale sin esperar al tope."""
    resumidor = _Resumidor()
    conv = _conv_fondo(resumidor)
    worker = CompactionWorker(conv)
    hilos: list[int] = []
    resumir = conv.summarizer

    def _anota(texto):
        hilos.append(threading.get_ident())
        return resumir(texto)

    conv.summarizer = _anota
    for i in range(3):
        _turno(conv, i)
    assert worker.schedule() is True
    assert worker.flush(5)
    assert conv.to_messages()[1]["content"].endswith("RESUMEN 1")
    _turno(conv, 3)
    assert worker.schedule() is True
    assert worker.flush(5)
    assert conv.to_messages()[1]["content"].endswith("RESUMEN 2")
    assert resumidor.hilos == ["CompactionWorker"] * 2
    assert len(hilos) == 2 and hilos[0] == hilos[1] != threading.get_ident()
    vivos = [t for t in threading.enumerate() if t.name == "CompactionWorker"]
    assert [t.ident for t in vivos] == hilos[:1]  # el mismo, esperando el siguiente
    t0 = time.monotonic()
    assert worker.close(1.0) is True
    assert time.monotonic() - t0 < 0.5
    assert not vivos[0].is_alive()
    assert worker.schedule() is False  # cerrado: no arranca más


def test_el_worker_junta_los_avisos_y_sobrevive_a_un_fallo(caplog):
    suelta = threading.Event()
    resumidor = _Resumidor(espera=suelta)
    conv = _conv_fondo(resumidor)
    worker = CompactionWorker(conv)
    for i in range(3):
        _turno(conv, i)
    worker.schedule()
    assert resumidor.dentro.wait(5)
    _turno(conv, 3)
    for _ in range(5):                     # cinco avisos mientras resume: una vuelta más
        worker.schedule()
    suelta.set()
    assert worker.flush(5)
    assert len(resumidor.entradas) == 2

    def _revienta():
        raise ValueError("bug")

    conv.compact_pending = _revienta  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="crotolamo.core.memory"):
        worker.schedule()
        assert worker.flush(5)
    assert "bug" in caplog.text
    worker.close(1.0)


def test_el_cierre_espera_acotado():
    suelta = threading.Event()
    resumidor = _Resumidor(espera=suelta)
    conv = _conv_fondo(resumidor)
    worker = CompactionWorker(conv)
    for i in range(3):
        _turno(conv, i)
    worker.schedule()
    assert resumidor.dentro.wait(5)
    t0 = time.monotonic()
    assert worker.close(0.2) is False      # sigue colgado: no espera más que el tope
    assert time.monotonic() - t0 < 1.0
    suelta.set()


def test_los_resumenes_reusan_la_conexion_keep_alive_del_worker():
    """Revisión: con un hilo nuevo por resumen, cada resumen abría su conexión
    (el keep-alive de HTTPTransport es por hilo): con GLM, TCP+TLS por resumen.
    Con el hilo persistente, dos resúmenes viajan por UNA conexión."""
    conexiones: list[tuple] = []
    peticiones: list[int] = []

    class _Ollama(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            conexiones.append(self.client_address)
            super().setup()

        def log_message(self, *_a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            peticiones.append(threading.get_ident())
            cuerpo = json.dumps({"message": {"role": "assistant",
                                             "content": "lo de antes"}, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
    srv.daemon_threads = True
    hilo_srv = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo_srv.start()
    cliente = LLMClient(host=f"http://127.0.0.1:{srv.server_address[1]}")
    worker = None
    try:
        conv = _conv_fondo(lambda texto: cliente.chat(
            [{"role": "system", "content": "Resume en español"},
             {"role": "user", "content": texto}]).content)
        worker = CompactionWorker(conv)
        for i in range(3):
            _turno(conv, i)
        assert worker.schedule() and worker.flush(5)
        _turno(conv, 3)
        assert worker.schedule() and worker.flush(5)
        assert len(peticiones) == 2
        assert len(conexiones) == 1
        assert conv.to_messages()[1]["content"].endswith("lo de antes")
    finally:
        if worker is not None:
            worker.close(1.0)
        cliente._transport.close()
        srv.shutdown()
        srv.server_close()
        hilo_srv.join(5)


def test_build_agent_con_compaction_resume_despues_de_hablar(monkeypatch, tmp_path):
    from crotolamo.core import hooks
    from crotolamo.settings import Settings
    from interfaces import shell

    class _Motor:
        model = "x"

        def __init__(self):
            self.llamadas: list[str] = []

        def chat(self, messages, tools=None):
            primero = messages[0]["content"]
            es_resumen = primero.startswith("Resume en español")
            self.llamadas.append("resumen" if es_resumen else "turno")
            return ChatResponse(content="lo de antes" if es_resumen else "Va, patrón.")

    motor = _Motor()
    arrancados: list[str] = []
    real_start = threading.Thread.start

    def _cuenta(self):
        arrancados.append(self.name)
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", _cuenta)
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS", Settings(
        raw={"llm": {"backend": "ollama"},
             "memory": {"compaction": True, "max_turns": 2}},
        user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    monkeypatch.setattr(shell, "build_llm", lambda s: motor)
    agent, conv = shell.build_agent()
    assert conv.summarize_in_background is True

    hilos_por_turno = []
    for i in range(6):
        antes = len(motor.llamadas)
        agent.handle_turn(f"vuelta {i}")
        # La primera llamada de cada turno es la del turno: nada antes de hablar.
        assert motor.llamadas[antes] == "turno"
        conv_worker = agent.after_turn_hooks[-1].__self__  # type: ignore[attr-defined]
        assert conv_worker.flush(5)
        hilos_por_turno.append(arrancados.count("CompactionWorker"))
    assert motor.llamadas.count("resumen") == 4
    resumenes = [m for m in conv.to_messages() if "resumen" in m["content"].lower()]
    assert len(resumenes) == 1
    # Ningún hilo en los dos primeros turnos (nada que resumir) y luego UNO solo
    # para los cuatro resúmenes (antes, uno nuevo por resumen).
    assert hilos_por_turno == [0, 0, 1, 1, 1, 1]
    conv_worker.close(1.0)


def test_build_agent_sin_compaction_no_arma_el_worker(monkeypatch, tmp_path):
    agent, conv, *_ = _build_agent_memoria(monkeypatch, tmp_path, "ollama", _memoria_de())
    assert conv.summarize_in_background is False
    assert not any(isinstance(getattr(h, "__self__", None), CompactionWorker)
                   for h in agent.after_turn_hooks)


# ===========================================================================
# Paso 12: la Conversation es la dueña de cómo se ARMA el request. Antes el
# request era el historial guardado tal cual: lo que un hook quería que el modelo
# viera en cada turno tenía que reescribir el texto guardado del patrón (los
# hechos SQLite iban dentro del PRIMER mensaje y morían al salir ese mensaje de
# la ventana o con /reset). Ahora hay huecos anclados tras el system prompt
# (set_pinned) y tres knobs del dueño que, sin su clave, dejan todo como estaba:
# trim_chunk, stale_tool_turns e idle_reset_min.
# ===========================================================================

from crotolamo.core.memory import STALE_TOOL_MARK, Message  # noqa: E402


class _ConversacionOriginal:
    """La Conversation de antes de esta causa raíz, copiada tal cual (HEAD)."""

    def __init__(self, system_prompt, max_turns=20, compaction=False, summarizer=None):
        self._system = Message("system", system_prompt)
        self._history: list[Message] = []
        self.max_turns = max_turns
        self.compaction = compaction
        self.summarizer = summarizer
        self._summary: Message | None = None

    def add_user(self, text):
        self._history.append(Message("user", text))
        self._trim()

    def add_assistant(self, text, tool_calls=None):
        self._history.append(Message("assistant", text, tool_calls=tool_calls))
        self._trim()

    def add_tool_result(self, name, content):
        self._history.append(Message("tool", content, name=name))
        self._trim()

    def reset(self):
        self._history.clear()
        self._summary = None

    def to_messages(self):
        msgs = [self._system.to_dict()]
        if self._summary is not None:
            msgs.append(self._summary.to_dict())
        return msgs + [m.to_dict() for m in self._history]

    def _trim(self):
        user_count = sum(1 for m in self._history if m.role == "user")
        while user_count > self.max_turns and self._history:
            block = [self._history.pop(0)]
            if block[0].role == "user":
                user_count -= 1
            while self._history and self._history[0].role in {"assistant", "tool"}:
                block.append(self._history.pop(0))
            if self.compaction and self.summarizer is not None:
                self._absorb_into_summary(block)

    def _absorb_into_summary(self, block):
        prev = self._summary.content if self._summary is not None else ""
        rendered = "\n".join(f"{m.role}: {m.content}" for m in block if m.content)
        try:
            summary_text = self.summarizer(f"{prev}\n{rendered}".strip())
        except Exception:  # noqa: BLE001
            return
        if summary_text and summary_text.strip():
            self._summary = Message(
                "system", f"[resumen de la conversación previa]\n{summary_text.strip()}")


def _al_azar_largo(conv, rnd, pasos: int) -> None:
    """Como _al_azar, con resultados de tool de todos los tamaños."""
    for i in range(pasos):
        tirada = rnd.random()
        if tirada < 0.3:
            conv.add_user(f"u{i}")
        elif tirada < 0.5:
            conv.add_assistant(f"a{i}")
        elif tirada < 0.65:
            conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
        elif tirada < 0.97:
            conv.add_tool_result("t", f"r{i}:" + "x" * rnd.choice((0, 10, 799, 800, 801, 5000)))
        else:
            conv.reset()


@pytest.mark.parametrize("semilla", range(60))
def test_con_los_valores_de_fabrica_el_request_es_el_de_siempre(semilla):
    """Propiedad: con los knobs sin tocar (y sin nada anclado), to_messages() y lo
    que recibe el resumidor son los de la clase original, para cualquier secuencia."""
    import random

    for compaction in (False, True):
        resumenes: dict[str, list[str]] = {"antes": [], "ahora": []}

        def _resume(clave):
            def resume(texto):
                resumenes[clave].append(texto)
                if len(resumenes[clave]) % 5 == 0:
                    raise RuntimeError("el resumidor también falla a veces")
                return f"R{len(resumenes[clave])}"
            return resume

        max_turns = 1 + semilla % 5
        antes = _ConversacionOriginal("SYS", max_turns=max_turns, compaction=compaction,
                                      summarizer=_resume("antes"))
        ahora = Conversation("SYS", max_turns=max_turns, compaction=compaction,
                             summarizer=_resume("ahora"))
        rnd_a, rnd_b = random.Random(semilla), random.Random(semilla)
        for _ in range(4):
            _al_azar_largo(antes, rnd_a, 40)
            _al_azar_largo(ahora, rnd_b, 40)
            assert ahora.to_messages() == antes.to_messages()
        assert resumenes["ahora"] == resumenes["antes"]


def _bloques_enteros(msgs: list[dict]) -> bool:
    """Ningún tool sin su assistant: el adaptador de OpenAI no tira nada."""
    return len(to_openai_messages(msgs)) == len(msgs)


def _turno_con_tool(conv, i: int, resultado: str = "ok") -> None:
    conv.add_user(f"u{i}")
    conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
    conv.add_tool_result("t", resultado)
    conv.add_assistant(f"a{i}")


def test_trim_chunk_tira_varios_bloques_de_golpe_y_enteros():
    conv = Conversation("SYS", max_turns=5, trim_chunk=3)
    vistos = []
    for i in range(14):
        _turno_con_tool(conv, i)
        msgs = conv.to_messages()
        assert _bloques_enteros(msgs)
        assert msgs[1]["role"] == "user"
        vistos.append(sum(1 for m in msgs if m["role"] == "user"))
    # Se llena hasta 5 y, al pasarse, baja a 5 - 3 + 1 = 3: el principio del
    # historial no cambia en los 2 turnos siguientes (prefijo cacheable).
    assert vistos == [1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
    assert [m.content for m in conv.history if m.role == "user"] == [
        "u9", "u10", "u11", "u12", "u13"]


def test_trim_chunk_no_tira_el_turno_en_curso():
    conv = Conversation("SYS", max_turns=2, trim_chunk=10)
    for i in range(3):
        _turno_con_tool(conv, i)
    assert [m.content for m in conv.history if m.role == "user"] == ["u2"]
    _turno_con_tool(conv, 3)
    assert [m.content for m in conv.history if m.role == "user"] == ["u2", "u3"]


def test_trim_chunk_con_compaction_resume_cada_bloque():
    resumidos: list[str] = []

    def resume(texto):
        resumidos.append(texto)
        return "R"

    conv = Conversation("SYS", max_turns=3, trim_chunk=2, compaction=True, summarizer=resume)
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    # Turno 4: se pasa de 3 y baja a 2 (u0 y u1 al resumen, uno por uno).
    assert len(resumidos) == 2 and "u0" in resumidos[0] and "u1" in resumidos[1]
    assert [m.content for m in conv.history if m.role == "user"] == ["u2", "u3", "u4"]


def test_trim_chunk_con_resumen_en_segundo_plano_no_tira_bloques(caplog):
    """Con trim_chunk=7 un recorte deja 7 bloques pendientes de golpe: el tope de
    pendientes (4) los tiraba sin resumir y avisaba de un resumen que "no avanza"."""
    resumidos: list[str] = []

    def resume(texto):
        resumidos.append(texto)
        return "R"

    conv = Conversation("SYS", max_turns=8, trim_chunk=7, compaction=True, summarizer=resume,
                        summarize_in_background=True)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memory"):
        for i in range(9):
            _turno_con_tool(conv, i)
    assert not caplog.records
    assert [m["content"] for m in conv.to_messages() if m["role"] == "user"] == [
        f"u{i}" for i in range(9)]
    assert conv.compact_pending() is True
    assert len(resumidos) == 1 and all(f"u{i}" in resumidos[0] for i in range(7))
    assert [m.content for m in conv.history if m.role == "user"] == ["u7", "u8"]


def test_lo_que_sobrevive_con_trim_chunk_es_lo_que_queda():
    import random

    for semilla in range(30):
        rnd = random.Random(semilla)
        conv = Conversation("SYS", max_turns=2 + semilla % 4, trim_chunk=1 + semilla % 3)
        _al_azar(conv, rnd, rnd.randrange(1, 60))
        previsto = conv.surviving_user_texts()
        conv.add_user("NUEVO")
        assert previsto == [m.content for m in conv.history if m.role == "user"][:-1]


def test_los_resultados_viejos_viajan_recortados_y_el_historial_queda_entero():
    largo = "L" * 5000
    conv = Conversation("SYS", stale_tool_turns=2)
    for i in range(5):
        _turno_con_tool(conv, i, resultado=f"{i}" + largo)
    conv.add_user("u5")
    msgs = conv.to_messages()
    tools = [m["content"] for m in msgs if m["role"] == "tool"]
    # Turnos 0-2 (más de 2 turnos atrás del actual): primeros 800 + la marca.
    for i in range(3):
        assert tools[i] == (f"{i}" + largo)[:800] + STALE_TOOL_MARK.format(n=5001)
    # Turnos 3 y 4 (los 2 anteriores al actual) van enteros.
    assert tools[3:] == ["3" + largo, "4" + largo]
    # El mensaje tool sigue ahí (solo cambia el content) y los bloques, enteros.
    assert _bloques_enteros(msgs)
    assert len(msgs) == len(conv.to_messages())
    # /historial y el resumen leen el original.
    assert [m.content for m in conv.history if m.role == "tool"] == [
        f"{i}" + largo for i in range(5)]


def test_un_resultado_corto_viejo_no_se_toca():
    conv = Conversation("SYS", stale_tool_turns=0)
    _turno_con_tool(conv, 0, resultado="x" * 800)
    _turno_con_tool(conv, 1, resultado="y" * 801)
    conv.add_user("u2")
    tools = [m["content"] for m in conv.to_messages() if m["role"] == "tool"]
    assert tools == ["x" * 800, "y" * 800 + STALE_TOOL_MARK.format(n=801)]


def test_los_resultados_del_turno_en_curso_nunca_se_recortan():
    conv = Conversation("SYS", stale_tool_turns=0, stale_tool_chars=10)
    conv.add_user("u0")
    conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
    conv.add_tool_result("t", "z" * 50)
    assert conv.to_messages()[-1]["content"] == "z" * 50


def test_lo_anclado_va_tras_el_system_y_sobrevive_al_recorte_y_al_reset():
    conv = Conversation("SYS", max_turns=1, compaction=True, summarizer=lambda texto: "RES")
    conv.set_pinned("hechos", "[H1]")
    conv.set_pinned("otro", "[O1]")
    for i in range(3):
        _turno_con_tool(conv, i)
    msgs = conv.to_messages()
    assert [m["content"] for m in msgs[:4]] == [
        "SYS", "[H1]", "[O1]", "[resumen de la conversación previa]\nRES"]
    assert msgs[1]["role"] == msgs[2]["role"] == "system"
    # Cambiar un hueco lo deja en su sitio; None lo quita.
    conv.set_pinned("hechos", "[H2]")
    conv.set_pinned("otro", None)
    conv.reset()
    assert conv.to_messages() == [{"role": "system", "content": "SYS"},
                                  {"role": "system", "content": "[H2]"}]
    # Lo anclado no es historial: /historial no lo muestra.
    assert conv.history == []


class _Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_tras_un_rato_sin_hablar_la_ventana_empieza_de_cero(caplog):
    reloj = _Reloj()
    conv = Conversation("SYS", idle_reset_s=1800, clock=reloj, compaction=True,
                        summarizer=lambda texto: "RES", max_turns=1)
    conv.set_pinned("hechos", "[H]")
    for i in range(3):
        _turno_con_tool(conv, i)
        reloj.t += 600
    # 600 s desde el último mensaje: no toca nada.
    conv.add_user("sigo aquí")
    conv.add_assistant("va")
    assert any("resumen" in m["content"] for m in conv.to_messages())
    reloj.t += 1801
    with caplog.at_level(logging.INFO, logger="crotolamo.core.memory"):
        conv.add_user("hola")
    assert conv.to_messages() == [
        {"role": "system", "content": "SYS"},
        {"role": "system", "content": "[H]"},
        {"role": "user", "content": "hola"},
    ]
    assert any("30 min" in r.getMessage() for r in caplog.records)


def test_el_rato_cuenta_desde_el_ultimo_mensaje_no_desde_el_ultimo_user():
    reloj = _Reloj()
    conv = Conversation("SYS", idle_reset_s=60, clock=reloj)
    conv.add_user("lee el archivo enorme")
    reloj.t += 50
    conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
    reloj.t += 50
    conv.add_tool_result("t", "ok")
    conv.add_assistant("listo")
    reloj.t += 30
    conv.add_user("¿y qué decía?")
    assert len(conv.history) == 5


def test_la_espera_cuenta_lo_que_la_laptop_estuvo_suspendida(monkeypatch):
    """time.monotonic no avanza mientras Linux está suspendido: 8 h de tapa cerrada
    no contarían como silencio. El reloj de fábrica es el de pared."""
    conv = Conversation("SYS", idle_reset_s=1800)
    _turno_con_tool(conv, 0)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 8 * 3600)
    conv.add_user("buenos días")
    assert [m.content for m in conv.history] == ["buenos días"]


def test_el_reset_por_inactividad_descarta_el_resumen_en_curso():
    reloj = _Reloj()
    dentro, suelta = threading.Event(), threading.Event()

    def resume(texto):
        dentro.set()
        suelta.wait(5)
        return "VIEJO"

    conv = Conversation("SYS", max_turns=1, compaction=True, summarizer=resume,
                        summarize_in_background=True, idle_reset_s=60, clock=reloj)
    _turno_con_tool(conv, 0)
    _turno_con_tool(conv, 1)
    hilo = threading.Thread(target=conv.compact_pending)
    hilo.start()
    assert dentro.wait(5)
    reloj.t += 61
    conv.add_user("hola de nuevo")
    suelta.set()
    hilo.join(5)
    assert [m["content"] for m in conv.to_messages()] == ["SYS", "hola de nuevo"]


def test_lo_que_sobrevive_tras_un_rato_sin_hablar_es_nada():
    """El pre-hook de la memoria semántica pregunta qué seguirá a la vista ANTES del
    add_user: si ese add_user va a vaciar la ventana, nada cuenta como visto."""
    reloj = _Reloj()
    conv = Conversation("SYS", idle_reset_s=60, clock=reloj)
    _turno_con_tool(conv, 0)
    assert conv.surviving_user_texts() == ["u0"]
    reloj.t += 61
    assert conv.surviving_user_texts() == []
    conv.add_user("hola")
    assert [m.content for m in conv.history] == ["hola"]


def test_sin_idle_reset_no_se_consulta_el_reloj():
    def reloj():
        raise AssertionError("sin idle_reset_s no hay por qué mirar la hora")

    conv = Conversation("SYS", clock=reloj)
    _turno_con_tool(conv, 0)
    conv.surviving_user_texts()


# --- build_agent: los knobs salen de [memory] y los hechos van anclados -----------

class _Graba:
    """El LLM de build_agent: graba los mensajes de cada llamada y contesta."""

    model = "x"

    def __init__(self):
        self.mensajes: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.mensajes.append(messages)
        return ChatResponse(content="Va, patrón.")


def _build_agent_hechos(monkeypatch, tmp_path, memory: dict, hechos=()):
    from crotolamo.persistence import facts
    from crotolamo.settings import Settings
    from interfaces import shell

    db = tmp_path / "hechos.sqlite"
    for texto in hechos:
        facts.remember(texto, "gustos", db_path=db)
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS", Settings(
        raw={"llm": {"backend": "ollama"}, "memory": memory},
        user="test", home=tmp_path, paths={"db": db}))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    llm = _Graba()
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, conv = shell.build_agent()
    return agent, conv, llm, db


def _bloque_de_hechos(*textos: str) -> str:
    lineas = "\n".join(f"- (gustos) {t}" for t in textos)
    return f"[contexto que ya sabes sobre el patrón:\n{lineas}]"


def _con_hechos(mensajes: list[dict], bloque: str) -> bool:
    return any(bloque in (m.get("content") or "") for m in mensajes)


def test_los_hechos_siguen_a_la_vista_tras_el_recorte_y_tras_el_reset(monkeypatch, tmp_path):
    """El arreglo funcional: con max_turns=3 los hechos desaparecían en el turno 4
    (salía de la ventana el primer mensaje, que los llevaba) y tras /reset."""
    agent, conv, llm, _ = _build_agent_hechos(
        monkeypatch, tmp_path, {"max_turns": 3}, ["Le gusta el café de olla"])
    bloque = _bloque_de_hechos("Le gusta el café de olla")
    for i in range(5):
        agent.handle_turn(f"vuelta {i}")
    assert all(m[1] == {"role": "system", "content": bloque} for m in llm.mensajes)
    conv.reset()
    agent.handle_turn("hola otra vez")
    assert llm.mensajes[-1][1] == {"role": "system", "content": bloque}
    # Una sola copia por request y ninguna en el texto guardado del patrón.
    assert all(sum(bloque in m["content"] for m in msgs) == 1 for msgs in llm.mensajes)
    assert not any(bloque in m.content for m in conv.history)


def test_el_bloque_anclado_es_el_texto_de_siempre(monkeypatch, tmp_path):
    from crotolamo.core import hooks

    agent, conv, llm, _ = _build_agent_hechos(
        monkeypatch, tmp_path, {}, ["Le gusta el café de olla", "Su perro es Rocky"])
    agent.handle_turn("hola")
    de_antes = hooks.make_facts_prehook()("hola")
    anclado = llm.mensajes[0][1]["content"]
    assert de_antes == f"{anclado}\n\nhola"


def test_los_hechos_anclados_se_releen_cada_turno(monkeypatch, tmp_path):
    """Un hecho nuevo o uno olvidado se nota en el turno siguiente (antes el bloque
    era el del primer turno y un hecho olvidado seguía ahí hasta salir de la ventana)."""
    from crotolamo.persistence import facts

    agent, conv, llm, db = _build_agent_hechos(
        monkeypatch, tmp_path, {}, ["Le gusta el café de olla"])
    agent.handle_turn("hola")
    nuevo = facts.remember("Su perro es Rocky", "gustos", db_path=db)
    agent.handle_turn("qué tal")
    assert llm.mensajes[-1][1]["content"] == _bloque_de_hechos(
        "Le gusta el café de olla", "Su perro es Rocky")
    facts.forget(1, db_path=db)
    facts.forget(nuevo, db_path=db)
    agent.handle_turn("y ahora")
    assert llm.mensajes[-1][1]["role"] == "user"  # sin hechos, sin hueco


def test_si_la_base_falla_se_queda_el_bloque_de_antes(monkeypatch, tmp_path, caplog):
    from crotolamo.persistence import facts

    agent, conv, llm, _ = _build_agent_hechos(
        monkeypatch, tmp_path, {}, ["Le gusta el café de olla"])
    agent.handle_turn("hola")

    def bloqueada(*a, **kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(facts, "facts_context", bloqueada)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.hooks"):
        agent.handle_turn("qué tal")
    assert llm.mensajes[-1][1]["content"] == _bloque_de_hechos("Le gusta el café de olla")
    assert any("database is locked" in r.getMessage() for r in caplog.records)


def test_pin_facts_false_es_la_inyeccion_unica_de_antes(monkeypatch, tmp_path):
    agent, conv, llm, _ = _build_agent_hechos(
        monkeypatch, tmp_path, {"max_turns": 3, "pin_facts": False},
        ["Le gusta el café de olla"])
    bloque = _bloque_de_hechos("Le gusta el café de olla")
    for i in range(4):
        agent.handle_turn(f"vuelta {i}")
    assert [_con_hechos(m, bloque) for m in llm.mensajes] == [True, True, True, False]
    assert llm.mensajes[0][1]["role"] == "user"
    assert f"{bloque}\n\nvuelta 0" in llm.mensajes[0][1]["content"]
    conv.reset()
    agent.handle_turn("hola otra vez")
    assert not _con_hechos(llm.mensajes[-1], bloque)


_GUSTOS = ["le gusta el café de olla sin azúcar",
           "su perro se llama Canelo y es un xoloitzcuintle",
           "trabaja de 9 a 6 en la oficina de Insurgentes",
           "prefiere que le hablen de tú y sin rodeos",
           "su hermana Mariana cumple años el 14 de marzo",
           "usa Fedora en la laptop y Hyprland como escritorio"]


@pytest.mark.parametrize("stream", [False, True])
def test_con_muchos_hechos_anclados_el_local_sigue_cabiendo_en_num_ctx(stream, caplog):
    """Revisión: con pin_facts de fábrica el bloque de hechos va en la cabeza que
    el ajuste no tocaba; persona (~830 tokens) + 8 esquemas enrutados (~700) + 30
    hechos (~540) ya pasaban de 2048 sin una palabra del patrón, y Ollama cortaba
    el PRINCIPIO del prompt: la persona. Ahora el hueco se recorta (el último
    recurso) y se quedan los hechos más nuevos; la persona sale byte a byte."""
    from crotolamo.core.persona import system_prompt
    from crotolamo.tools import default_registry

    persona = system_prompt()
    conv = Conversation(persona)
    conv.set_pinned("facts", _bloque_de_hechos(
        *(f"{_GUSTOS[k % len(_GUSTOS)]} ({k})" for k in range(30))))
    conv.add_user("[ahora: 2026-11-14 10:00]\nlee el archivo del proyecto y busca la función")
    esquemas = router.route_schemas(default_registry(), "lee el archivo del proyecto y busca la función", 8)
    assert len(esquemas) == 8
    msgs = conv.to_messages()
    assert budget.est_tokens(msgs, esquemas) > 2048 - budget.REPLY_RESERVE  # no cabía

    cliente, cable = _local(num_ctx=2048)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.llm"):
        (cliente.chat_stream(msgs, tools=esquemas, on_token=lambda t: None) if stream
         else cliente.chat(msgs, tools=esquemas))
    enviado = cable.payloads[0]["messages"]
    assert budget.est_tokens(enviado, esquemas) <= 2048 - budget.REPLY_RESERVE
    assert json.dumps(enviado[0], ensure_ascii=False) == json.dumps(
        {"role": "system", "content": persona}, ensure_ascii=False)
    hechos = enviado[1]["content"]
    assert hechos.startswith("[contexto que ya sabes sobre el patrón:\n...[recortado, ")
    assert hechos.endswith(f"{_GUSTOS[29 % len(_GUSTOS)]} (29)]")  # el más nuevo
    assert "(0)" not in hechos  # el más viejo es el que sobra
    assert enviado[-1] == msgs[-1]  # lo que dijo el patrón, entero
    assert msgs == conv.to_messages()  # solo el request: la Conversation, igual
    aviso = " ".join(r.getMessage() for r in caplog.records)
    assert "1 hueco(s) anclado(s)" in aviso


def test_build_agent_lee_los_knobs_de_memory(monkeypatch, tmp_path):
    _, conv, _, _ = _build_agent_hechos(
        monkeypatch, tmp_path,
        {"max_turns": 12, "trim_chunk": 4, "stale_tool_turns": 2, "idle_reset_min": 30})
    assert (conv.max_turns, conv.trim_chunk, conv.stale_tool_turns, conv.idle_reset_s) == (
        12, 4, 2, 1800)


def test_sin_las_claves_build_agent_deja_todo_como_estaba(monkeypatch, tmp_path):
    _, conv, _, _ = _build_agent_hechos(monkeypatch, tmp_path, {})
    assert (conv.max_turns, conv.trim_chunk, conv.stale_tool_turns, conv.idle_reset_s) == (
        20, 1, None, None)


def test_un_knob_mal_escrito_se_avisa_y_no_tumba_el_arranque(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.shell"):
        agent, conv, _, _ = _build_agent_hechos(
            monkeypatch, tmp_path,
            {"trim_chunk": "siete", "stale_tool_turns": True, "idle_reset_min": -30})
    assert isinstance(agent, ToolAgent)
    assert (conv.trim_chunk, conv.stale_tool_turns, conv.idle_reset_s) == (1, None, None)
    avisos = " ".join(r.getMessage() for r in caplog.records)
    assert all(k in avisos for k in ("trim_chunk", "stale_tool_turns", "idle_reset_min"))


def test_la_config_de_fabrica_documenta_los_knobs_de_memory():
    import tomllib

    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    memory = tomllib.loads(texto)["memory"]
    assert memory.get("pin_facts", True) is True
    # Los de decisión del dueño van comentados: sin la clave, lo de siempre.
    for clave in ("trim_chunk", "stale_tool_turns", "idle_reset_min"):
        assert clave not in memory
        assert f"# {clave} = " in texto
    assert "pin_facts" in texto


# ===========================================================================
# Paso 13: productores de request con bytes estables. (a) route_schemas emitía
# las tools en el orden de la selección: el mismo set con otro orden (la rotación
# de los grupos MCP, otra frase con otros scores) eran otros bytes y Ollama
# re-evaluaba todo el prompt. Ahora va en el orden del registry. (b) Las llamadas
# sin id (las del texto, las nativas de Ollama) se numeraban por posición al
# traducir para la nube: al salir un turno de la ventana, todas las de después
# cambiaban de id y de bytes. Ahora reciben un id estable al guardarse.
# ===========================================================================

from crotolamo.core import router  # noqa: E402


@pytest.fixture
def dos_servers_mcp():
    reg = Registry()
    for nombre in ("ram_usage", "music_control"):
        reg.register(_tool_simple(nombre))
    nombres = {}
    for srv in ("notas", "tareas"):
        nombres[srv] = [f"mcp_{srv}_tool{k}" for k in range(3)]
        for nombre in nombres[srv]:
            reg.register(_tool_simple(nombre))
        router.register_group(f"mcp:{srv}", nombres[srv], ["pendiente", srv])
    yield reg, nombres
    for srv in nombres:
        router.unregister_group(f"mcp:{srv}")


def _tool_simple(nombre: str) -> Tool:
    return Tool(nombre, lambda **kw: "ok", f"tool {nombre}",
                {"type": "object", "properties": {}, "required": []})


def test_la_misma_consulta_da_los_mismos_bytes(dos_servers_mcp):
    reg, nombres = dos_servers_mcp
    # La rotación MCP sigue haciendo su trabajo en select_tool_names...
    rotadas = {tuple(router.select_tool_names("revisa lo pendiente", 8, reg.names()))
               for _ in range(4)}
    assert len(rotadas) == 2
    # ...pero el bloque de tools que viaja es siempre el mismo, en orden del registry.
    bloques = {json.dumps(router.route_schemas(reg, "revisa lo pendiente", 8),
                          ensure_ascii=False) for _ in range(4)}
    assert len(bloques) == 1
    enviadas = [s["function"]["name"] for s in json.loads(bloques.pop())]
    assert enviadas == nombres["notas"] + nombres["tareas"]


def test_el_mismo_set_por_otra_frase_da_los_mismos_bytes():
    from crotolamo.tools import default_registry

    reg = default_registry()
    a = router.route_schemas(reg, "pausa y dime la ram", 8)
    b = router.route_schemas(reg, "estado del sistema y pausa", 8)
    # La selección los ordena distinto (gana el grupo con más evidencia)...
    assert (router.select_tool_names("pausa y dime la ram", 8, reg.names())
            != router.select_tool_names("estado del sistema y pausa", 8, reg.names()))
    # ...pero es el mismo set y viaja con los mismos bytes.
    assert {s["function"]["name"] for s in a} == {s["function"]["name"] for s in b}
    assert json.dumps(a) == json.dumps(b)
    orden = reg.names()
    indices = [orden.index(s["function"]["name"]) for s in a]
    assert indices == sorted(indices)


def test_el_set_enrutado_no_cambia(dos_servers_mcp):
    """Solo cambia el orden: los nombres son los de select_tool_names, con su tope."""
    from crotolamo.tools import default_registry

    reg = default_registry()
    for frase in ("abre la carpeta de descargas", "acuérdate de la nota del archivo",
                  "cierra la ventana de opera", "lee el archivo notas.txt de documentos",
                  "pon música y abre spotify", "jajaja que buena onda"):
        for tope in (1, 3, 8):
            enviadas = [s["function"]["name"] for s in router.route_schemas(reg, frase, tope)]
            assert sorted(enviadas) == sorted(router.select_tool_names(frase, tope, reg.names()))


class _PideSinId:
    """Un modelo que pide una tool por turno SIN id (Ollama nativo o JSON en el
    texto) y luego contesta; graba lo que vería la nube en cada llamada."""

    def __init__(self, nativo: bool):
        self.nativo = nativo
        self.requests: list[list[dict]] = []
        self.turno = 0

    def chat(self, messages, tools=None):
        self.requests.append(to_openai_messages(messages))
        if messages[-1]["role"] == "tool":
            return ChatResponse(content="Ya está, patrón.")
        self.turno += 1
        llamada = {"name": "lee", "arguments": {"n": self.turno}}
        if self.nativo:
            raw = {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": "lee", "arguments": {"n": self.turno}}}]}
            return ChatResponse(content="", tool_calls=[llamada], raw_message=raw)
        return ChatResponse(content=json.dumps(llamada))


def _ids_por_resultado(openai_msgs: list[dict]) -> dict[str, str]:
    return {m["content"]: m["tool_call_id"] for m in openai_msgs if m["role"] == "tool"}


@pytest.mark.parametrize("nativo", [True, False])
def test_los_ids_de_las_llamadas_no_cambian_al_recortar(nativo):
    llm = _PideSinId(nativo)
    reg = Registry()
    reg.register(Tool("lee", lambda n=0: f"resultado {n}", "lee",
                      {"type": "object", "properties": {"n": {"type": "integer"}},
                       "required": []}))
    agent = ToolAgent(llm, Conversation("SYS", max_turns=2), registry=reg, guard=_Permite(),
                      pre_hooks=[], post_hooks=[], fastpath=False)
    for i in range(5):
        agent.handle_turn(f"vuelta {i}")
    vistos: dict[str, str] = {}
    for req in llm.requests:
        for resultado, cid in _ids_por_resultado(req).items():
            assert vistos.setdefault(resultado, cid) == cid, (resultado, cid)
    assert len(vistos) == 5 and len(set(vistos.values())) == 5
    # Lo que sobrevive al recorte viaja con los mismos bytes: el bloque de la
    # vuelta 3 (user, llamada, resultado) va igual antes y después de que el
    # recorte de la vuelta 4 saque al de la vuelta 2 (antes: call_1 -> call_0).
    antes, despues = llm.requests[-3], llm.requests[-1]
    assert antes[-3]["content"] == despues[1]["content"] == "vuelta 3"
    assert json.dumps(despues[1:4]) == json.dumps(antes[-3:])


def test_los_ids_que_trae_el_motor_se_respetan():
    class _ConId(_PideSinId):
        def chat(self, messages, tools=None):
            resp = super().chat(messages, tools)
            for k, call in enumerate(resp.raw_message.get("tool_calls") or []):
                call["id"] = f"glm-{self.turno}-{k}"
            return resp

    llm = _ConId(True)
    reg = Registry()
    reg.register(Tool("lee", lambda n=0: f"resultado {n}", "lee",
                      {"type": "object", "properties": {}, "required": []}))
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg, guard=_Permite(),
                      pre_hooks=[], post_hooks=[], fastpath=False)
    agent.handle_turn("vuelta")
    guardado = [m for m in agent.conversation.history if m.tool_calls][0]
    assert guardado.tool_calls == [{"function": {"name": "lee", "arguments": {"n": 1}},
                                    "id": "glm-1-0"}]


# ===========================================================================
# Paso 14: observabilidad para decidir los knobs con datos reales. Nada de esto
# cambia un payload ni un valor devuelto: solo el log. Lo que gasta la nube
# (prompt_tokens, cached_tokens), lo que evaluó Ollama (prompt_eval_count, con
# un WARNING si llegó al tope de num_ctx) y, en DEBUG, el tamaño de cada request.
# ===========================================================================

from crotolamo.core import sse  # noqa: E402

_USO_GLM = {"prompt_tokens": 15234, "completion_tokens": 12, "total_tokens": 15246,
            "prompt_tokens_details": {"cached_tokens": 14848}}


class _CableConUso(_Cable):
    """Como _Cable, pero el motor cuenta lo que gastó (usage / prompt_eval_count)."""

    def __init__(self, kind: str, prompt_eval: int = 0):
        super().__init__(kind)
        self.prompt_eval = prompt_eval

    def post_json(self, path, payload, headers=None):
        self.payloads.append(json.loads(json.dumps(payload, ensure_ascii=False)))
        if self.kind == "glm":
            if payload.get("stream"):
                return _Contesta(lines=[
                    b'data: {"choices":[{"delta":{"content":"Listo."},'
                    b'"finish_reason":"stop"}]}',
                    b'data: {"choices":[],"usage":' + json.dumps(_USO_GLM).encode() + b"}",
                    b"data: [DONE]"])
            return _Contesta(json.dumps({"choices": [{"message": {
                "role": "assistant", "content": "Listo, patrón."}}],
                "usage": _USO_GLM}).encode())
        fin = {"message": {"role": "assistant", "content": "" if payload.get("stream")
                           else "Listo, patrón."}, "done": True,
               "prompt_eval_count": self.prompt_eval, "eval_count": 7}
        if payload.get("stream"):
            return _Contesta(lines=[
                json.dumps({"message": {"role": "assistant", "content": "Listo."},
                            "done": False}).encode(),
                json.dumps(fin).encode()])
        return _Contesta(json.dumps(fin).encode())


@pytest.mark.parametrize("stream", [False, True])
def test_glm_cuenta_los_tokens_cacheados(monkeypatch, caplog, stream):
    cliente, _ = _nube(monkeypatch)
    cable = _CableConUso("glm")
    cliente._transport = cable  # type: ignore[assignment]
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hola"}]
    with caplog.at_level(logging.DEBUG, logger="crotolamo.core.glm"):
        resp = (cliente.chat_stream(msgs, on_token=lambda t: None) if stream
                else cliente.chat(msgs))
    assert resp.content in {"Listo.", "Listo, patrón."}
    texto = " ".join(r.getMessage() for r in caplog.records)
    assert "prompt_tokens=15234" in texto and "cached_tokens=14848" in texto
    assert "completion_tokens=12" in texto


def test_sin_usage_la_nube_no_dice_nada(monkeypatch, caplog):
    cliente, cable = _nube(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.core.glm"):
        cliente.chat_stream([{"role": "user", "content": "hola"}], on_token=lambda t: None)
    assert not any("cached_tokens" in r.getMessage() for r in caplog.records)


def test_parse_sse_guarda_el_usage_sin_cambiar_lo_que_devuelve():
    lineas = [b'data: {"choices":[{"delta":{"content":"Hola"},"finish_reason":"stop"}]}',
              b'data: {"choices":[],"usage":' + json.dumps(_USO_GLM).encode() + b"}",
              b"data: [DONE]"]
    sin = sse.parse_sse(list(lineas), None)
    uso: dict = {}
    con = sse.parse_sse(list(lineas), None, usage=uso)
    assert con == sin == ("Hola", {"role": "assistant", "content": "Hola"}, True)
    assert uso == _USO_GLM
    assert sse.consume_sse(list(lineas), None) == ("Hola", {"role": "assistant",
                                                            "content": "Hola"})


@pytest.mark.parametrize("stream", [False, True])
def test_ollama_avisa_si_el_prompt_llego_al_tope_de_num_ctx(caplog, stream):
    cliente, _ = _local(num_ctx=2048)
    cliente._transport = _CableConUso("ollama", prompt_eval=2040)  # type: ignore[assignment]
    msgs = [{"role": "user", "content": "hola"}]
    with caplog.at_level(logging.DEBUG, logger="crotolamo.core.llm"):
        resp = (cliente.chat_stream(msgs, on_token=lambda t: None) if stream
                else cliente.chat(msgs))
    assert resp.content in {"Listo.", "Listo, patrón."}
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    assert "2040" in avisos[0].getMessage() and "num_ctx=2048" in avisos[0].getMessage()


@pytest.mark.parametrize("prompt_eval,num_ctx", [(1791, 2048), (5000, None), (0, 2048)])
def test_ollama_sin_tope_no_avisa(caplog, prompt_eval, num_ctx):
    cliente, _ = _local(num_ctx=num_ctx)
    cliente._transport = _CableConUso("ollama", prompt_eval=prompt_eval)  # type: ignore
    with caplog.at_level(logging.DEBUG, logger="crotolamo.core.llm"):
        cliente.chat([{"role": "user", "content": "hola"}])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(f"prompt_eval_count={prompt_eval}" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("motor", ["ollama", "glm"])
def test_cada_request_cuenta_su_tamano_en_debug(monkeypatch, caplog, motor):
    cliente, cable = _local() if motor == "ollama" else _nube(monkeypatch)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hola"}]
    with caplog.at_level(logging.DEBUG, logger=f"crotolamo.core.{'llm' if motor == 'ollama' else 'glm'}"):
        cliente.chat(msgs, tools=_ESQUEMAS[:3])
    enviado = len(json.dumps(cable.payloads[0], ensure_ascii=False).encode("utf-8"))
    linea = next(r.getMessage() for r in caplog.records if "bytes" in r.getMessage())
    assert f"{enviado} bytes" in linea and "3 tools" in linea and "2 mensajes" in linea
    assert f"~{budget.est_tokens(cable.payloads[0]['messages'], _ESQUEMAS[:3])} tokens" in linea


@pytest.mark.parametrize("motor", ["ollama", "glm"])
def test_sin_debug_no_se_serializa_de_mas(monkeypatch, motor):
    """En producción (INFO) contar el tamaño del request no cuesta ningún json.dumps
    extra; solo con DEBUG se serializa para medirlo."""
    import json as json_mod

    from crotolamo.core import llm as llm_mod

    llamadas = []
    real = json_mod.dumps

    def cuenta(*a, **kw):
        llamadas.append(1)
        return real(*a, **kw)

    def dumps_con(nivel, sin_helper=False):
        cliente, _ = _local() if motor == "ollama" else _nube(monkeypatch)
        logger = logging.getLogger(f"crotolamo.core.{'llm' if motor == 'ollama' else 'glm'}")
        logger.setLevel(nivel)
        llamadas.clear()
        with monkeypatch.context() as m:
            m.setattr(json_mod, "dumps", cuenta)
            if sin_helper:
                m.setattr(llm_mod, "_log_request", lambda *a, **kw: None)
                if motor == "glm":
                    from crotolamo.core import glm as glm_mod
                    m.setattr(glm_mod, "_log_request", lambda *a, **kw: None)
            try:
                cliente.chat([{"role": "user", "content": "hola"}], tools=_ESQUEMAS[:1])
            finally:
                logger.setLevel(logging.NOTSET)
        return len(llamadas)

    referencia = dumps_con(logging.INFO, sin_helper=True)
    assert dumps_con(logging.INFO) == referencia
    assert dumps_con(logging.DEBUG) > referencia


@pytest.mark.parametrize("motor", ["ollama", "glm"])
@pytest.mark.parametrize("stream", [False, True])
def test_la_observabilidad_no_cambia_payload_ni_respuesta(monkeypatch, caplog, motor, stream):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hola"}]
    resultados = []
    for nivel in (logging.WARNING, logging.DEBUG):
        cliente, _ = _local() if motor == "ollama" else _nube(monkeypatch)
        cable = _CableConUso(motor, prompt_eval=2040)
        cliente._transport = cable  # type: ignore[assignment]
        with caplog.at_level(nivel):
            resp = (cliente.chat_stream(msgs, tools=_ESQUEMAS[:2], on_token=lambda t: None)
                    if stream else cliente.chat(msgs, tools=_ESQUEMAS[:2]))
        resultados.append((cable.payloads, resp))
    assert resultados[0] == resultados[1]


# Revisión: la línea DEBUG del request medía con el códec estricto. El historial
# puede traer un sustituto suelto (un "\ud83d" escapado en el JSON de la nube o
# en los argumentos de una tool) que el transporte manda con errors="replace";
# medirlo con .encode("utf-8") a secas lanzaba UnicodeEncodeError ANTES de
# enviar, y como no es LLMError ni FallbackLLM caía al local: cada turno con esa
# respuesta en la ventana salía roto, solo con CROTOLAMO_LOG=DEBUG.
_MEDIO_EMOJI = json.loads('"\\ud83d"')


@pytest.mark.parametrize("motor", ["ollama", "glm"])
@pytest.mark.parametrize("stream", [False, True])
def test_debug_mide_un_historial_con_un_sustituto_suelto_sin_romper_el_turno(
        monkeypatch, caplog, motor, stream):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hola"},
            {"role": "assistant", "content": f"listo {_MEDIO_EMOJI}"},
            {"role": "user", "content": "y ahora"}]
    resultados = []
    for nivel in (logging.INFO, logging.DEBUG):
        cliente, _ = _local() if motor == "ollama" else _nube(monkeypatch)
        cable = _CableConUso(motor)
        cliente._transport = cable  # type: ignore[assignment]
        caplog.clear()
        with caplog.at_level(nivel, logger="crotolamo"):
            resp = (cliente.chat_stream(msgs, tools=_ESQUEMAS[:2], on_token=lambda t: None)
                    if stream else cliente.chat(msgs, tools=_ESQUEMAS[:2]))
        resultados.append((cable.payloads, resp))
    assert resultados[0] == resultados[1]
    # Y la línea sale con los bytes que de verdad viajan (como el transporte).
    enviado = len(json.dumps(resultados[1][0][0], ensure_ascii=False)
                  .encode("utf-8", errors="replace"))
    assert any(f"{enviado} bytes" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("motor", ["ollama", "glm"])
def test_una_linea_de_log_que_revienta_no_tumba_el_request(monkeypatch, caplog, motor):
    # Sin ajuste de ventana: así solo la línea DEBUG pasa por est_tokens.
    cliente, cable = (_local(num_ctx=None) if motor == "ollama"
                      else _nube(monkeypatch, max_context_tokens=None))

    def revienta(*_a, **_kw):
        raise RuntimeError("bug en la medición")

    monkeypatch.setattr(budget, "est_tokens", revienta)
    with caplog.at_level(logging.DEBUG, logger="crotolamo"):
        resp = cliente.chat([{"role": "user", "content": "hola"}])
    assert resp.content == "Listo, patrón."
    assert len(cable.payloads) == 1
