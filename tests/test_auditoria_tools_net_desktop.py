"""Auditoría exhaustiva — grupo tools de red y escritorio (_web, _hass, _frigate,
cameras, home, media, desktop, windows).

Cada test fija un hallazgo confirmado de la auditoría: falla con el código de
antes y pasa con el arreglo. Los comentarios dicen QUÉ se rompía. Todo SIN red
externa: DNS de mentiras y servidores efímeros en 127.0.0.1.
"""

from __future__ import annotations

import http.client
import http.server
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import warnings
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import (
    _frigate,
    _hass,
    _web,
    cameras,
    default_registry,
    desktop,
    home,
    media,
    projects,
    search,
    shortcuts,
    windows,
)

_RAIZ = Path(__file__).resolve().parents[1]


def _dns_fijo(monkeypatch, ip: str = "93.184.216.34") -> list[tuple]:
    """getaddrinfo de mentiras que siempre resuelve a `ip`; devuelve la lista
    de llamadas para poder contarlas."""
    llamadas: list[tuple] = []

    def fake(host, port, *a, **kw):
        llamadas.append((host, port))
        return [(2, 1, 6, "", (ip, 0))]

    monkeypatch.setattr(_web.socket, "getaddrinfo", fake)
    return llamadas


def _sin_tcp(monkeypatch) -> list[tuple]:
    """Corta en el connect (OSError) y apunta a qué IP:puerto se intentó ir."""
    conexiones: list[tuple] = []

    def captura(addr, timeout=None, **kw):
        conexiones.append(addr)
        raise OSError("cortamos aquí: solo interesa la IP destino")

    monkeypatch.setattr(_web.socket, "create_connection", captura)
    return conexiones


def _servidor_crudo(payload: bytes, veces: int = 1) -> tuple[int, socket.socket]:
    """Servidor TCP que contesta `payload` tal cual (sin http.server): sirve
    para respuestas rotas que un servidor HTTP de verdad no sabe emitir."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop() -> None:
        for _ in range(veces):
            try:
                conn, _addr = srv.accept()
            except OSError:
                return
            with conn:
                conn.recv(65536)
                conn.sendall(payload)

    threading.Thread(target=loop, daemon=True).start()
    return srv.getsockname()[1], srv


CHUNK_CORTADO = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                 b"Transfer-Encoding: chunked\r\n\r\n1000\r\n[{\"id\": \"a\"")
# La misma respuesta cortada pero de una PÁGINA: _http_get ya no lee el cuerpo
# de lo que no es texto (un JSON lo rechaza read_page por el tipo), así que el
# corte a medio chunk solo se ve con un Content-Type legible.
CHUNK_CORTADO_HTML = (b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                      b"Transfer-Encoding: chunked\r\n\r\n1000\r\n<html><body><p>hola")
NO_ES_HTTP = b"SSH-2.0-OpenSSH_9.6\r\n"
JSON_NO_UTF8 = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: 6\r\n\r\n[\"\xe9\"]\n")


# ---------------------------------------------------------------------------
# Guardia SSRF: 100.64.0.0/10 (CGNAT y TODOS los nodos de Tailscale) y el
# site-local de IPv6 no son "internet". ipaddress no los marca is_private, así
# que el filtro los dejaba pasar y read_page llegaba a servicios de la tailnet.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ip", [
    "100.64.0.1",        # primer host del espacio compartido (RFC 6598)
    "100.100.100.100",   # MagicDNS / API local de Tailscale
    "100.127.255.254",   # último host del /10
    "fec0::1",           # IPv6 site-local (obsoleto, pero sigue siendo red interna)
])
def test_cgnat_y_site_local_no_son_publicas(monkeypatch, ip):
    assert _web._ip_is_public(ip) is False
    _dns_fijo(monkeypatch, ip)
    assert _web.is_public_url("http://nodo-de-la-tailnet.example:8123/") is False
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(_web.BlockedRedirectError):
        _web._http_get("http://nodo-de-la-tailnet.example:8123/api/", timeout=2.0)
    assert conexiones == []  # ni un SYN hacia la tailnet


@pytest.mark.parametrize("ip", [
    "93.184.216.34", "8.8.8.8",
    "100.63.255.255", "100.128.0.1",   # los vecinos del /10 SÍ son internet
    "2606:4700::1111",
])
def test_las_ips_publicas_siguen_siendo_publicas(ip):
    assert _web._ip_is_public(ip) is True


# ---------------------------------------------------------------------------
# DNS que no resuelve (sin wifi, dominio mal escrito) NO es "red interna": es
# un fallo de conexión. Antes ambos casos eran el mismo None y el patrón oía
# "esa dirección es de la red interna" de una página pública.
# ---------------------------------------------------------------------------

def _sin_dns(monkeypatch) -> None:
    def boom(*a, **kw):
        raise socket.gaierror(-3, "Temporary failure in name resolution")

    monkeypatch.setattr(_web.socket, "getaddrinfo", boom)


def test_dns_caido_es_fallo_de_conexion_no_bloqueo_ssrf(monkeypatch):
    _sin_dns(monkeypatch)
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(urllib.error.URLError) as info:
        _web._http_get("https://es.wikipedia.org/wiki/Ajolote", timeout=2.0)
    assert not isinstance(info.value, _web.BlockedRedirectError)
    assert "es.wikipedia.org" in str(info.value)
    assert conexiones == []
    # El predicado sigue negando ante la duda (fail closed).
    assert _web.is_public_url("https://es.wikipedia.org/wiki/Ajolote") is False


def test_fetch_web_results_sin_internet_dice_conexion(monkeypatch):
    """El camino REAL sin red (test_web.py sustituye _http_get entero y no lo
    ejercita): getaddrinfo falla y la tool debe decir que falló la conexión."""
    _sin_dns(monkeypatch)
    _sin_tcp(monkeypatch)
    out = search.fetch_web_results("clima en monterrey")
    assert "conexión" in out.lower()
    assert "red interna" not in out


@pytest.mark.parametrize("host", ["a..b.example", "x" * 70 + ".example"])
def test_host_que_ni_se_puede_codificar_tampoco_es_red_interna(monkeypatch, host):
    """El getaddrinfo REAL lanza UnicodeError (IDNA) con etiquetas vacías o
    larguísimas, sin llegar a consultar nada: mismo trato que un DNS que no
    resuelve."""
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(urllib.error.URLError) as info:
        _web._http_get(f"http://{host}/", timeout=2.0)
    assert not isinstance(info.value, _web.BlockedRedirectError)
    assert conexiones == []
    assert _web.is_public_url(f"http://{host}/") is False


# ---------------------------------------------------------------------------
# UNA resolución DNS por salto (gate y pin son la misma) y UN contexto TLS por
# proceso: antes eran dos getaddrinfo por salto y un ssl.create_default_context
# (recarga del bundle de CAs, ~20 ms) por cada conexión https.
# ---------------------------------------------------------------------------

def test_una_sola_resolucion_dns_por_salto(monkeypatch):
    llamadas = _dns_fijo(monkeypatch)
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(OSError):
        _web._http_get("http://ejemplo.com/pagina", timeout=2.0)
    assert conexiones == [("93.184.216.34", 80)]
    assert len(llamadas) == 1


def test_resolucion_unica_que_ya_es_privada_se_bloquea_sin_conectar(monkeypatch):
    """La otra mitad de test_web.test_dns_rebinding_se_bloquea: con una sola
    resolución, si esa ya es privada el GET muere antes del connect."""
    llamadas = _dns_fijo(monkeypatch, "127.0.0.1")
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(_web.BlockedRedirectError):
        _web._http_get("http://rebind.evil/", timeout=2.0)
    assert conexiones == []
    assert len(llamadas) == 1


def test_fetch_web_results_resuelve_una_vez(monkeypatch):
    llamadas = _dns_fijo(monkeypatch)
    _sin_tcp(monkeypatch)
    search.fetch_web_results("ajolote")
    assert [host for host, _port in llamadas] == ["html.duckduckgo.com"]


def test_contexto_tls_se_crea_una_vez_y_se_reutiliza(monkeypatch):
    creados: list[object] = []
    real = _web.ssl.create_default_context

    def contando(*a, **kw):
        ctx = real(*a, **kw)
        creados.append(ctx)
        return ctx

    monkeypatch.setattr(_web.ssl, "create_default_context", contando)
    # Como recién importado: que otro test ya lo haya creado no cuente aquí.
    monkeypatch.setattr(_web, "_SSL_CONTEXT", None, raising=False)
    _dns_fijo(monkeypatch)
    conexiones = _sin_tcp(monkeypatch)
    for url in ("https://uno.example/a", "https://dos.example/b", "https://uno.example/c"):
        with pytest.raises(OSError):
            _web._http_get(url, timeout=2.0)
    assert conexiones == [("93.184.216.34", 443)] * 3
    assert len(creados) == 1


def test_contexto_tls_compartido_sigue_verificando_certificados(monkeypatch):
    """El contexto cacheado es el de fábrica: verifica cadena y hostname."""
    monkeypatch.setattr(_web, "_SSL_CONTEXT", None, raising=False)
    conn = _web._PinnedHTTPSConnection("ejemplo.com", "93.184.216.34", None, 2.0)
    ctx = conn._ssl_context
    assert ctx.verify_mode == _web.ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    otra = _web._PinnedHTTPSConnection("otro.example", "93.184.216.34", None, 2.0)
    assert otra._ssl_context is ctx


# ---------------------------------------------------------------------------
# http.client.HTTPException (IncompleteRead, BadStatusLine...) NO hereda de
# OSError: se escapaba de _http_get y de _frigate_get hasta el genérico
# "La tool ... reventó" del registry.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [CHUNK_CORTADO_HTML, NO_ES_HTTP],
                         ids=["chunked-cortado", "no-es-http"])
def test_http_get_respuesta_rota_es_urlerror(monkeypatch, payload):
    """El docstring de _http_get promete OSError/URLError si la red falla."""
    port, srv = _servidor_crudo(payload)
    monkeypatch.setattr(_web, "_resolve_public_ip", lambda host: "127.0.0.1")
    try:
        with pytest.raises(urllib.error.URLError):
            _web._http_get(f"http://127.0.0.1:{port}/x", timeout=5.0)
    finally:
        srv.close()


def test_read_page_respuesta_cortada_responde_en_personaje(monkeypatch):
    port, srv = _servidor_crudo(CHUNK_CORTADO_HTML)
    monkeypatch.setattr(_web, "_resolve_public_ip", lambda host: "127.0.0.1")
    monkeypatch.setattr(_web, "is_public_url", lambda url: True)
    try:
        out = default_registry().run("read_page", {"url": f"http://127.0.0.1:{port}/x"})
    finally:
        srv.close()
    assert "reventó" not in out
    assert "No pude conectarme" in out


@pytest.fixture
def cams_cfg(monkeypatch):
    """Config [cameras] de mentiras en el singleton de settings."""
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {
        "base_url": "http://frigate.test:5000",
        "names": {"entrada": "front_door"},
    })
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return real


@pytest.mark.parametrize("error, codigo", [
    (http.client.IncompleteRead(b""), "conexion"),
    (http.client.BadStatusLine("SSH-2.0-OpenSSH_9.6\r\n"), "conexion"),
    (http.client.InvalidURL("URL can't contain control characters"), "conexion"),
    # base_url sin esquema en la config: Request() lanza ValueError antes de la red.
    (ValueError("unknown url type: 'frigate.local/api/events'"), "conexion"),
    # JSON que no es UTF-8: UnicodeDecodeError es ValueError, no JSONDecodeError.
    (UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte"),
     "respuesta_rara"),
], ids=["incomplete-read", "bad-status-line", "invalid-url", "url-sin-esquema", "no-utf8"])
def test_frigate_get_no_lanza(cams_cfg, monkeypatch, error, codigo):
    """Contrato de _frigate_get: 'Ante fallo NO lanza'."""
    def boom(url, timeout=6.0):
        raise error

    monkeypatch.setattr(_frigate, "_http_json", boom)
    ok, data = _frigate._frigate_get("/api/events", {"limit": 1})
    assert ok is False
    assert str(data).startswith(codigo)
    out = default_registry().run("camera_events", {"camera": "entrada"})
    assert "reventó" not in out
    assert "Frigate" in out


@pytest.mark.parametrize("payload, codigo", [
    (CHUNK_CORTADO, "conexion"),
    (NO_ES_HTTP, "conexion"),
    (JSON_NO_UTF8, "respuesta_rara"),
], ids=["chunked-cortado", "no-es-http", "no-utf8"])
def test_camera_events_con_frigate_roto_de_verdad(monkeypatch, payload, codigo):
    """De punta a punta con un socket real: sin parchear _http_json."""
    port, srv = _servidor_crudo(payload, veces=2)
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {
        "base_url": f"http://127.0.0.1:{port}", "names": {}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    try:
        ok, data = _frigate._frigate_get("/api/events", {"limit": 1})
        out = cameras.camera_events("todas")
    finally:
        srv.close()
    assert ok is False and str(data).startswith(codigo)
    assert "No pude hablar con Frigate" in out


# ---------------------------------------------------------------------------
# _hass / _frigate: el opener se construye en el primer uso, no al importar.
# En Python 3.12+ HTTPSHandler crea su SSLContext (y carga el bundle de CAs)
# en el constructor: ~20 ms por módulo en CADA arranque, se use o no la casa.
# ---------------------------------------------------------------------------

_SONDA_IMPORT = """
import urllib.request
construidos = []
real = urllib.request.build_opener
def contando(*a, **kw):
    construidos.append(1)
    return real(*a, **kw)
urllib.request.build_opener = contando
import crotolamo.tools._hass as h
import crotolamo.tools._frigate as f
import crotolamo.tools.home, crotolamo.tools.cameras
al_importar = len(construidos)
assert h._OPENER is None and f._OPENER is None
uno = h._opener()
assert h._opener() is uno and h._OPENER is uno, "el opener de _hass no se cachea"
dos = f._opener()
assert f._opener() is dos and f._OPENER is dos, "el opener de _frigate no se cachea"
print(al_importar, len(construidos))
"""


def test_importar_las_tools_no_construye_openers():
    proc = subprocess.run(
        [sys.executable, "-c", _SONDA_IMPORT],
        cwd=_RAIZ, capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    al_importar, tras_usar = proc.stdout.split()
    assert al_importar == "0"   # importar no cuesta ni un SSLContext
    assert tras_usar == "2"     # uno por módulo, construido una sola vez


class _RespuestaFalsa:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _RespuestaFalsa:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def read1(self, _n: int = -1) -> bytes:
        # _http_json lee a trozos (read1) hasta que no queda nada.
        body, self._body = self._body, b""
        return body


class _OpenerFalso:
    def __init__(self, body: bytes) -> None:
        self.urls: list[str] = []
        self._body = body

    def open(self, request, timeout=None):
        self.urls.append(request.full_url)
        return _RespuestaFalsa(self._body)


def test_el_opener_parcheado_es_el_que_se_usa(monkeypatch):
    """`_OPENER` sigue siendo EL punto de sustitución (tests, proxies): quien
    lo reemplaza en el módulo manda sobre el perezoso."""
    falso_f = _OpenerFalso(b'[{"id": "e1"}]')
    monkeypatch.setattr(_frigate, "_OPENER", falso_f)
    assert _frigate._http_json("http://frigate.test:5000/api/events") == [{"id": "e1"}]
    assert falso_f.urls == ["http://frigate.test:5000/api/events"]

    falso_h = _OpenerFalso(b'{"state": "on"}')
    monkeypatch.setattr(_hass, "_OPENER", falso_h)
    assert _hass._http_json("GET", "http://hass.test:8123/api/states/x", {}) == {"state": "on"}
    assert falso_h.urls == ["http://hass.test:8123/api/states/x"]


# ===========================================================================
# Lote 2: _web.py (URL -> request-target, lectura del cuerpo, charset, texto).
# ===========================================================================

class _Paginas(http.server.BaseHTTPRequestHandler):
    """Servidor de mentiras: contesta lo que diga `rutas[path]` (cabeceras extra
    y cuerpo en bytes) y apunta cada request-target TAL CUAL llegó por el cable."""

    rutas: dict[str, tuple[int, dict[str, str], bytes]] = {}
    pedidos: list[str] = []

    def do_GET(self):  # noqa: N802 — nombre que exige BaseHTTPRequestHandler
        type(self).pedidos.append(self.path)
        status, headers, body = type(self).rutas.get(
            self.path, (200, {"Content-Type": "text/html; charset=utf-8"}, b"<p>ok</p>"))
        self.send_response(status)
        for nombre, valor in headers.items():
            self.send_header(nombre, valor)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silencio en la salida de pytest
        pass


@pytest.fixture
def paginas(monkeypatch):
    """(base_url, rutas, pedidos) de un servidor local; el pin acepta loopback."""
    _Paginas.rutas = {}
    _Paginas.pedidos = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Paginas)
    # poll_interval corto: shutdown() espera un ciclo entero (0.5 s por defecto).
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    monkeypatch.setattr(_web, "_resolve_public_ip", lambda host: "127.0.0.1")
    monkeypatch.setattr(_web, "is_public_url", lambda url: True)
    yield f"http://127.0.0.1:{server.server_port}", _Paginas.rutas, _Paginas.pedidos
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# El request-target se mandaba tal cual venía en la URL: con acentos http.client
# lanzaba UnicodeEncodeError, con espacios InvalidURL, y urlparse le cortaba el
# ";params" al último segmento (se pedía OTRA página). Las URLs en español que
# escribe el propio LLM llevan acentos casi siempre.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url, esperado", [
    ("https://es.wikipedia.org/wiki/México", "/wiki/M%C3%A9xico"),
    ("https://es.wikipedia.org/wiki/Ciudad de México", "/wiki/Ciudad%20de%20M%C3%A9xico"),
    ("https://example.com/buscar?q=niño feliz&lang=es", "/buscar?q=ni%C3%B1o%20feliz&lang=es"),
    # urlparse separaba ";jsessionid=AB12" como params y se perdía.
    ("https://example.com/doc;jsessionid=AB12?x=1", "/doc;jsessionid=AB12?x=1"),
    ("https://example.com/wiki/Law_%26_Order;_SVU", "/wiki/Law_%26_Order;_SVU"),
    # Lo que ya venía codificado o era puntuación legal NO se toca.
    ("https://es.wikipedia.org/wiki/M%C3%A9xico?a=1&b=%20x", "/wiki/M%C3%A9xico?a=1&b=%20x"),
    ("https://example.com/a/b.c~d?e[]=1&f=a+b,c:d@e!$'()*", "/a/b.c~d?e[]=1&f=a+b,c:d@e!$'()*"),
    ("https://example.com", "/"),
    ("https://example.com?x=1", "/?x=1"),
    ("https://example.com/pagina#seccion", "/pagina"),
])
def test_request_target_se_codifica_sin_tocar_lo_ya_valido(monkeypatch, url, esperado):
    _dns_fijo(monkeypatch)
    conn, target = _web._pinned_connection(url, 2.0)
    conn.close()
    assert target == esperado
    target.encode("ascii")  # http.client exige ASCII en la línea de petición


def test_read_page_con_acentos_y_espacios_llega_al_servidor(paginas):
    base, rutas, pedidos = paginas
    rutas["/wiki/Ciudad%20de%20M%C3%A9xico;v=2?q=%C3%B1"] = (
        200, {"Content-Type": "text/html; charset=utf-8"},
        "<p>La capital del país.</p>".encode())
    out = default_registry().run(
        "read_page", {"url": f"{base}/wiki/Ciudad de México;v=2?q=ñ"})
    assert pedidos == ["/wiki/Ciudad%20de%20M%C3%A9xico;v=2?q=%C3%B1"]
    assert "La capital del país." in out


def test_redirect_con_location_sin_codificar_se_sigue(paginas):
    """Location con UTF-8 crudo y un espacio (servidores viejos lo mandan así):
    http.client entrega el header decodificado como ISO-8859-1, así que hay que
    recuperar los bytes originales antes de codificar, no codificar el mojibake."""
    base, rutas, pedidos = paginas
    crudo = "/destino/México lindo".encode().decode("latin-1")
    rutas["/a"] = (302, {"Location": crudo}, b"")
    rutas["/destino/M%C3%A9xico%20lindo"] = (
        200, {"Content-Type": "text/html; charset=utf-8"}, b"<p>llegaste</p>")
    _content_type, body = _web._http_get(f"{base}/a", timeout=5.0)
    assert pedidos == ["/a", "/destino/M%C3%A9xico%20lindo"]
    assert "llegaste" in body


# ---------------------------------------------------------------------------
# IPv6 literal: con port=None http.client vuelve a partir el host por el último
# ":" y toma el último hexteto como PUERTO ("::1111" -> puerto 1111; "::abcd"
# -> InvalidURL, que ni read_page atrapaba: "La tool reventó").
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url, puerto", [
    ("http://[2606:4700:4700::1111]/", 80),
    ("https://[2606:4700:4700::1111]/", 443),
    ("http://[2606:4700:4700::abcd]/dns-query", 80),
    ("http://[2606:4700:4700::1111]:8080/", 8080),
])
def test_ipv6_literal_no_confunde_el_ultimo_hexteto_con_el_puerto(monkeypatch, url, puerto):
    ip = url.split("[")[1].split("]")[0]
    _dns_fijo(monkeypatch, ip)
    conn, _target = _web._pinned_connection(url, 2.0)
    conn.close()
    assert (conn.host, conn.port) == (ip, puerto)
    conexiones = _sin_tcp(monkeypatch)
    with pytest.raises(OSError):
        _web._http_get(url, timeout=2.0)
    assert conexiones == [(ip, puerto)]


def test_read_page_ipv6_literal_no_revienta(monkeypatch):
    _dns_fijo(monkeypatch, "2606:4700:4700::abcd")
    conexiones = _sin_tcp(monkeypatch)
    out = default_registry().run("read_page", {"url": "http://[2606:4700:4700::abcd]/"})
    assert "reventó" not in out
    assert conexiones == [("2606:4700:4700::abcd", 80)]


def test_puertos_por_defecto_y_explicitos_no_cambian(monkeypatch):
    _dns_fijo(monkeypatch)
    for url, puerto in [("http://ejemplo.com/", 80), ("https://ejemplo.com/", 443),
                        ("http://ejemplo.com:8123/", 8123), ("https://ejemplo.com:8443/", 8443)]:
        conn, _target = _web._pinned_connection(url, 2.0)
        conn.close()
        assert (conn.host, conn.port) == ("ejemplo.com", puerto)


# ---------------------------------------------------------------------------
# El cuerpo se leía ENTERO (hasta 1.5 MB) antes de que nadie mirara el
# Content-Type: un PDF, un zip o una radio por internet se descargaban para
# contestar "eso no es una página". Y el timeout era por recv(): un servidor
# que gotea bytes nunca lo disparaba (el hilo del Brain, colgado).
# ---------------------------------------------------------------------------

def _espia_lecturas(monkeypatch) -> list[int]:
    """Apunta cuántos bytes de CUERPO entrega http.client a quien lee."""
    leidos: list[int] = []
    for nombre in ("read", "read1"):
        real = getattr(http.client.HTTPResponse, nombre)

        def espia(self, *a, _real=real, **kw):
            data = _real(self, *a, **kw)
            leidos.append(len(data))
            return data

        monkeypatch.setattr(http.client.HTTPResponse, nombre, espia)
    return leidos


@pytest.mark.parametrize("mime", ["audio/mpeg", "application/pdf", "application/zip",
                                  "image/png", "application/json"])
def test_contenido_no_textual_no_se_descarga(paginas, monkeypatch, mime):
    base, rutas, _pedidos = paginas
    rutas["/binario"] = (200, {"Content-Type": mime}, b"\x00" * 300_000)
    leidos = _espia_lecturas(monkeypatch)
    content_type, body = _web._http_get(f"{base}/binario", timeout=5.0)
    assert content_type == mime
    assert body == ""
    assert sum(leidos) == 0  # ni un byte del cuerpo
    # read_page contesta lo mismo de siempre, ahora sin bajarse nada.
    out = search.read_page(f"{base}/binario")
    assert "no es una página que pueda leer" in out and mime in out
    assert sum(leidos) == 0


@pytest.mark.parametrize("content_type", [
    "text/html; charset=utf-8", "text/plain", "application/xhtml+xml", "TEXT/HTML", None])
def test_contenido_textual_se_sigue_leyendo_entero(paginas, content_type):
    """Los tipos que read_page SÍ lee (text/*, xhtml, o sin Content-Type) llegan
    completos aunque ocupen muchas lecturas del socket."""
    base, rutas, _pedidos = paginas
    cuerpo = ("<p>" + "palabra " * 40_000 + "FIN</p>").encode()
    headers = {} if content_type is None else {"Content-Type": content_type}
    rutas["/largo"] = (200, headers, cuerpo)
    devuelto, body = _web._http_get(f"{base}/largo", timeout=5.0)
    assert devuelto == (content_type or "").lower()
    assert body == cuerpo.decode()


def test_el_tope_de_descarga_se_respeta_al_byte(paginas, monkeypatch):
    base, rutas, _pedidos = paginas
    rutas["/enorme"] = (200, {"Content-Type": "text/plain"}, b"a" * 500_000)
    monkeypatch.setattr(_web, "_MAX_DOWNLOAD_BYTES", 100_000)
    leidos = _espia_lecturas(monkeypatch)
    _content_type, body = _web._http_get(f"{base}/enorme", timeout=5.0)
    assert len(body) == 100_000
    assert sum(leidos) == 100_000


def _servidor_goteo(cabeceras: bytes, gota: bytes, cada: float):
    """Manda `cabeceras` y luego `gota` cada `cada` segundos, sin acabar nunca
    (hasta que el cliente cuelgue o se llame a parar())."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    alto = threading.Event()

    def loop() -> None:
        try:
            conn, _addr = srv.accept()
        except OSError:
            return
        with conn:
            try:
                conn.recv(65536)
                conn.sendall(cabeceras)
                while not alto.wait(cada):
                    conn.sendall(gota)
            except OSError:
                return

    threading.Thread(target=loop, daemon=True).start()

    def parar() -> None:
        alto.set()
        srv.close()

    return srv.getsockname()[1], parar


def test_servidor_que_gotea_no_cuelga_la_lectura(monkeypatch):
    """Cada gota llega mucho antes del timeout del socket, así que este nunca
    salta: sin un tope TOTAL la lectura seguía hasta juntar 1.5 MB (horas)."""
    port, parar = _servidor_goteo(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n",
        b"<p>gota</p>", cada=0.02)
    monkeypatch.setattr(_web, "_resolve_public_ip", lambda host: "127.0.0.1")
    resultado: list[tuple[str, str]] = []
    hilo = threading.Thread(
        target=lambda: resultado.append(_web._http_get(f"http://127.0.0.1:{port}/", timeout=1.0)),
        daemon=True)
    try:
        hilo.start()
        hilo.join(timeout=15)
        terminado = not hilo.is_alive()
    finally:
        parar()
    assert terminado, "_http_get seguía leyendo 15 s después con timeout=1.0"
    # Lo que alcanzó a llegar se devuelve: sirve para leer el principio.
    assert resultado and "<p>gota</p>" in resultado[0][1]


# ---------------------------------------------------------------------------
# Charset: si el header HTTP no lo trae, manda lo que declare el documento
# (BOM o <meta>). Antes todo eso se leía como UTF-8 y una página ISO-8859-1
# llegaba como "El ni�o comi� pi�a en M�xico".
# ---------------------------------------------------------------------------

_FRASE = "El niño comió piña en México."


@pytest.mark.parametrize("header, cuerpo", [
    ("text/html",
     ('<html><head><meta charset="iso-8859-1"><title>Señal</title></head>'
      f"<body><p>{_FRASE}</p></body></html>").encode("latin-1")),
    ("text/html",
     ("<html><head><META HTTP-EQUIV='Content-Type' CONTENT='text/html; charset=windows-1252'>"
      f"</head><body><p>{_FRASE}</p></body></html>").encode("cp1252")),
    ("text/html",
     (f"<html><head><meta charset=utf-8></head><body><p>{_FRASE}</p></body></html>").encode()),
    # Sin declarar nada: UTF-8, como siempre.
    ("text/html", f"<html><body><p>{_FRASE}</p></body></html>".encode()),
    # El header HTTP manda sobre el <meta> (así lo hacen los navegadores).
    ("text/html; charset=utf-8",
     f'<html><head><meta charset="iso-8859-1"></head><body><p>{_FRASE}</p></body></html>'.encode()),
    ("text/html; charset=iso-8859-1",
     f'<html><head><meta charset="utf-8"></head><body><p>{_FRASE}</p></body></html>'
     .encode("latin-1")),
    # Un <meta> que dice utf-16 miente: si se pudo leer como ASCII, no lo es.
    ("text/html",
     f'<html><head><meta charset="utf-16"></head><body><p>{_FRASE}</p></body></html>'.encode()),
    # Charset inventado: no revienta, cae a UTF-8.
    ("text/html",
     f'<html><head><meta charset="klingon-9"></head><body><p>{_FRASE}</p></body></html>'.encode()),
    # En texto plano no se busca <meta>: ahí es texto, no una declaración.
    ("text/plain", f'Ejemplo: <meta charset="iso-8859-1">\n{_FRASE}'.encode()),
    # BOM UTF-16 sin charset en el header.
    ("text/html", f"<html><body><p>{_FRASE}</p></body></html>".encode("utf-16")),
], ids=["meta-latin1", "http-equiv-cp1252", "meta-utf8", "sin-declarar", "header-gana-utf8",
        "header-gana-latin1", "meta-utf16-miente", "meta-inventado", "text-plain", "bom-utf16"])
def test_charset_declarado_en_el_documento(paginas, header, cuerpo):
    base, rutas, _pedidos = paginas
    rutas["/pagina"] = (200, {"Content-Type": header}, cuerpo)
    _content_type, body = _web._http_get(f"{base}/pagina", timeout=5.0)
    assert _FRASE in body
    assert "�" not in body


def test_bom_utf8_no_se_cuela_en_el_texto(paginas):
    base, rutas, _pedidos = paginas
    rutas["/bom.txt"] = (200, {"Content-Type": "text/plain"}, b"\xef\xbb\xbf" + _FRASE.encode())
    _content_type, body = _web._http_get(f"{base}/bom.txt", timeout=5.0)
    assert body == _FRASE
    assert search.read_page(f"{base}/bom.txt").endswith(":\n" + _FRASE)


def test_read_page_de_pagina_latin1_con_meta(paginas):
    base, rutas, _pedidos = paginas
    rutas["/latin"] = (200, {"Content-Type": "text/html"}, (
        '<html><head><meta charset="iso-8859-1"><title>Señal</title></head>'
        f"<body><p>{_FRASE}</p></body></html>").encode("latin-1"))
    out = search.read_page(f"{base}/latin")
    assert "Señal" in out and _FRASE in out


# ---------------------------------------------------------------------------
# _PageTextExtractor pegaba palabras: <br> no separaba nada y los bloques
# anidados o sin cerrar se unían con "" ("Av. Juárez 5Col. CentroMonterrey").
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("html, esperado", [
    ("<p>Av. Juárez 5<br>Col. Centro<br/>Monterrey</p>", "Av. Juárez 5 Col. Centro Monterrey"),
    ("<blockquote><p>Primera frase.</p><p>Segunda frase.</p></blockquote>",
     "Primera frase. Segunda frase."),
    ("<table><tr><td><p>Precio</p><p>$10</p></td><td>Stock</td></tr></table>",
     "Precio $10\nStock"),
    ("<p>uno<p>dos<p>tres", "uno dos tres"),
    ("<ul><li>uno<ul><li>sub</li></ul></li></ul>", "uno sub"),
    ("<ul><li><div>Nombre</div><div>Precio</div></li></ul>", "Nombre Precio"),
    ("<p>antes<hr>después</p>", "antes después"),
    # Las etiquetas EN LÍNEA no parten palabras.
    ("<p>H<sub>2</sub>O y pala<b>bra</b>, <a href='#'>enla</a>ce</p>", "H2O y palabra, enlace"),
    # Fuera de un bloque de contenido, o dentro de lo que se ignora, no pintan nada.
    ("<div>fuera<br>también</div><p>dentro</p>", "dentro"),
    ("<p>a</p><nav>x<br>y<p>z</p></nav><p>b</p>", "a\nb"),
])
def test_extractor_separa_palabras_en_saltos_y_bloques(html, esperado):
    extractor = _web._PageTextExtractor()
    extractor.feed(html)
    assert extractor.text() == esperado


# ---------------------------------------------------------------------------
# read_page se queda con 6000 caracteres pero el extractor parseaba el
# documento ENTERO (1.5 MB de HTMLParser en Python puro). Con max_chars deja de
# parsear en cuanto ya juntó más de lo que se va a usar.
# ---------------------------------------------------------------------------

def _documento_grande(parrafos: int = 20_000) -> str:
    cuerpo = "".join(
        f"<div class='c'><p>Párrafo {i} con <b>algo</b> de texto.</p><script>var x={i};</script>"
        f"</div>\n" for i in range(parrafos))
    return f"<html><head><title>Largo</title></head><body>{cuerpo}</body></html>"


def _cuenta_etiquetas(monkeypatch) -> list[str]:
    """Apunta cada etiqueta de apertura que el extractor llega a procesar."""
    vistas: list[str] = []
    real = _web._PageTextExtractor.handle_starttag

    def contando(self, tag, attrs):
        vistas.append(tag)
        real(self, tag, attrs)

    monkeypatch.setattr(_web._PageTextExtractor, "handle_starttag", contando)
    return vistas


def test_extractor_con_tope_deja_de_parsear(monkeypatch):
    html = _documento_grande()
    assert len(html) > 1_500_000
    vistas = _cuenta_etiquetas(monkeypatch)
    completo = _web._PageTextExtractor()
    completo.feed(html)
    entero = completo.text()
    total = len(vistas)
    vistas.clear()

    tope = 6_000
    con_tope = _web._PageTextExtractor(max_chars=tope)
    con_tope.feed(html)
    texto = con_tope.text()
    # Mismo resultado en lo que read_page conserva, y sigue notándose que hay
    # más página (len > tope es lo que dispara el "[... texto truncado").
    assert texto[:tope] == entero[:tope]
    assert tope < len(texto) < tope + 200
    # ...parseando una fracción del documento, no el megabyte y pico entero.
    assert total > 80_000
    assert len(vistas) < total // 50
    # Ya lleno, más HTML ni se mira (ni duplica lo que ya tenía).
    con_tope.feed("<p>otra cosa</p>")
    assert con_tope.text() == texto


_HTML_CORTO = ("<html><head><title>T&iacute;tulo</title><script>if (a < b && c) "
               "{ d = '</p>'; }</script></head><body><p class='x'>Uno &amp; dos "
               "<b>tres</b></p><!-- <p>no</p> --><ul><li>cuatro<br>cinco</li></ul></body></html>")


@pytest.mark.parametrize("html", [
    _HTML_CORTO,
    "<html><body><h1>Título</h1><p>Uno &amp; dos.</p><li>tres</li></body></html>" * 3,
    # Un <script> o un comentario gigantes delante del poco texto que hay: lo de
    # después NO se puede perder por tener tope.
    "<html><head><script>" + "var x = {a: 1, b: '<p>no</p>'};\n" * 45_000
    + "</script></head><body><p>Poco texto.</p></body></html>",
    "<html><body><!--" + "x" * 1_400_000 + "--><p>Poco texto.</p></body></html>",
    # Documento cortado a media etiqueta (el tope de descarga cae donde cae).
    "<html><body><p>Entero.</p><p>A medias <a href='/x",
], ids=["corto", "repetido", "script-gigante", "comentario-gigante", "cortado"])
def test_extractor_con_tope_no_cambia_lo_que_cabe(html):
    sin_tope = _web._PageTextExtractor()
    sin_tope.feed(html)
    con_tope = _web._PageTextExtractor(max_chars=6_000)
    con_tope.feed(html)
    assert con_tope.text() == sin_tope.text()
    assert "texto." in sin_tope.text() or "Entero." in sin_tope.text() or "tres" in sin_tope.text()


def test_extractor_tope_justo_en_la_frontera():
    """El corte es por texto YA cerrado: con el tope exacto no se pierde nada, y
    al pasarse por un carácter el bloque que lo rebasa llega entero."""
    html = "<p>" + "a" * 10 + "</p><p>" + "b" * 10 + "</p><p>" + "c" * 10 + "</p>"
    justo = _web._PageTextExtractor(max_chars=32)   # 10 + 1 + 10 + 1 + 10
    justo.feed(html)
    assert justo.text() == "a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10
    corto = _web._PageTextExtractor(max_chars=20)
    corto.feed(html)
    assert corto.text() == "a" * 10 + "\n" + "b" * 10
    # Llegar JUSTO al tope no es pasarse: si parase ahí, quien llama no podría
    # distinguir "la página medía exactamente eso" de "hay más página".
    exacto = _web._PageTextExtractor(max_chars=21)   # 10 + 1 + 10
    exacto.feed(html)
    assert exacto.text() == "a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10


@pytest.mark.parametrize("html", [
    _documento_grande(2_000),                      # larga: read_page la trunca
    _HTML_CORTO,                                   # corta: llega entera
    "<html><body><div>sin bloques</div></body></html>",  # nada legible
], ids=["larga", "corta", "sin-texto"])
def test_read_page_responde_igual_con_el_extractor_topado(monkeypatch, html):
    """El contrato del que depende read_page al pedir max_chars: la respuesta
    de la tool es idéntica, carácter a carácter, a la de parsear todo."""
    monkeypatch.setattr(_web, "_http_get", lambda url, timeout=10.0: ("text/html", html))
    monkeypatch.setattr(_web, "is_public_url", lambda url: True)
    real = _web._PageTextExtractor
    # Las dos variantes ignoran lo que les pase read_page: así el test compara
    # siempre "sin tope" contra "con tope", lo pida read_page o no.
    monkeypatch.setattr(_web, "_PageTextExtractor", lambda max_chars=None: real())
    entero = search.read_page("https://example.com/x")
    monkeypatch.setattr(_web, "_PageTextExtractor",
                        lambda max_chars=None: real(max_chars=search._MAX_PAGE_CHARS))
    assert search.read_page("https://example.com/x") == entero


# ===========================================================================
# Lote 3: cámaras (Frigate) y carpetas de escritorio
# ===========================================================================

def _urls_frigate(monkeypatch, respuesta) -> list[str]:
    """_http_json de mentiras que apunta cada URL pedida a Frigate."""
    urls: list[str] = []

    def fake(url, timeout=6.0):
        urls.append(url)
        return respuesta

    monkeypatch.setattr(_frigate, "_http_json", fake)
    return urls


# ---------------------------------------------------------------------------
# cameras: /api/events mete por defecto la miniatura de cada evento en base64
# (include_thumbnails=1 en Frigate 0.14-0.16) y las tools solo usan
# label/camera/start_time/end_time/id: cientos de kB descargados y parseados
# para imprimir 50 líneas.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("llamada", [
    lambda: cameras.camera_events("entrada", hours=6),
    lambda: cameras.camera_events("todas"),
    lambda: cameras.camera_snapshot("entrada"),
], ids=["events-una-camara", "events-todas", "snapshot"])
def test_camaras_no_piden_miniaturas(cams_cfg, monkeypatch, llamada):
    urls = _urls_frigate(monkeypatch, [])
    llamada()
    assert len(urls) == 1
    assert parse_qs(urlsplit(urls[0]).query)["include_thumbnails"] == ["0"]


class _FrigateFalso(http.server.BaseHTTPRequestHandler):
    """/api/events con la semántica de Frigate: más recientes primero, `limit`
    y la miniatura en base64 salvo que se pida include_thumbnails=0."""

    eventos: list[dict] = []
    bytes_servidos: list[int] = []

    def do_GET(self):  # noqa: N802 — nombre que exige http.server
        query = parse_qs(urlsplit(self.path).query)
        limit = int(query.get("limit", ["100"])[0])
        con_miniatura = int(query.get("include_thumbnails", ["1"])[0])
        filas = [dict(e, thumbnail="A" * 8_000) if con_miniatura else e
                 for e in self.eventos[:limit]]
        body = json.dumps(filas).encode()
        self.bytes_servidos.append(len(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silencio en la salida de pytest
        pass


@pytest.fixture
def frigate_falso(monkeypatch):
    """Frigate de mentiras en 127.0.0.1 con [cameras].base_url apuntándole."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FrigateFalso)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(_FrigateFalso, "eventos", [])
    monkeypatch.setattr(_FrigateFalso, "bytes_servidos", [])
    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "cameras", {
        "base_url": f"http://127.0.0.1:{srv.server_address[1]}", "names": {}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    yield _FrigateFalso
    srv.shutdown()
    srv.server_close()


def _evento(i: int, start: float, camera: str = "calle") -> dict:
    return {"id": f"ev{i}", "camera": camera, "label": "car",
            "start_time": start, "end_time": start + 8}


def test_camera_events_descarga_sin_miniaturas_y_dice_lo_mismo(frigate_falso):
    """De punta a punta contra el Frigate falso: mismos eventos en la respuesta
    y el body que viaja no carga con las miniaturas (8 kB por evento)."""
    ahora = time.time()
    frigate_falso.eventos = [_evento(i, ahora - 60 * (i + 1)) for i in range(40)]
    out = cameras.camera_events("todas", hours=6)
    assert out.count("\n- coche en la cámara calle") == 40
    assert "(40 en total)" in out
    assert len(frigate_falso.bytes_servidos) == 1
    # 40 eventos sin miniatura rondan los 4 kB; con ella serían más de 320 kB.
    assert frigate_falso.bytes_servidos[0] < 10_000


def test_camera_snapshot_descarga_sin_miniatura(frigate_falso):
    frigate_falso.eventos = [_evento(0, time.time() - 60)]
    out = cameras.camera_snapshot("calle")
    assert "/api/events/ev0/snapshot.jpg" in out
    # Un evento pelado son ~100 bytes; con su miniatura, más de 8 kB.
    assert len(frigate_falso.bytes_servidos) == 1
    assert frigate_falso.bytes_servidos[0] < 1_000


# ---------------------------------------------------------------------------
# camera_events: Frigate corta la lista en `limit` y la tool decía "(50 en
# total)" aunque hubiera 300: el LLM le contaba al patrón 50 eventos. Se pide
# uno de más solo para saber si la lista quedó cortada.
# ---------------------------------------------------------------------------

def _frigate_con(monkeypatch, cuantos: int) -> None:
    """_http_json de mentiras que respeta `limit` como Frigate: el evento i
    (0 = el más reciente) viene de la cámara cam<i>."""
    ahora = time.time()
    todos = [_evento(i, ahora - 30 * (i + 1), camera=f"cam{i}") for i in range(cuantos)]

    def fake(url, timeout=6.0):
        return todos[:int(parse_qs(urlsplit(url).query)["limit"][0])]

    monkeypatch.setattr(_frigate, "_http_json", fake)


@pytest.mark.parametrize("cuantos", [51, 300])
def test_camera_events_avisa_cuando_la_lista_esta_cortada(cams_cfg, monkeypatch, cuantos):
    _frigate_con(monkeypatch, cuantos)
    out = cameras.camera_events("todas", hours=6)
    assert "en total" not in out      # 50 NO es el total
    assert "hay más" in out
    assert f"{cameras._MAX_EVENTS} más recientes" in out
    # Se siguen listando 50, no 51: el de más solo delata el corte.
    lineas = out.splitlines()[1:]
    assert len(lineas) == cameras._MAX_EVENTS
    assert "cámara cam0 " in lineas[0] and "cámara cam49 " in lineas[-1]
    assert "cam50" not in out


@pytest.mark.parametrize("cuantos", [1, 49, 50])
def test_camera_events_total_exacto_cuando_cabe(cams_cfg, monkeypatch, cuantos):
    """Justo 50 eventos SÍ son el total: 'hay más' sería mentira al revés."""
    _frigate_con(monkeypatch, cuantos)
    out = cameras.camera_events("todas", hours=6)
    assert f"({cuantos} en total)" in out
    assert "hay más" not in out
    assert out.count("\n- ") == cuantos
    assert f"cámara cam{cuantos - 1} " in out.splitlines()[-1]


# ---------------------------------------------------------------------------
# camera_events acepta ventanas de hasta una semana, pero cada evento solo
# decía "a las 21:14": el de hoy y el de hace tres días eran indistinguibles.
# El que no es de hoy lleva la fecha, en el mismo formato ISO del sello
# [ahora: ...] que ve el LLM.
# ---------------------------------------------------------------------------

# Jueves a media tarde, lejos de la medianoche para que "hoy" no baile.
_AHORA_FIJO = time.mktime((2026, 11, 12, 15, 0, 0, 0, 0, -1))


@pytest.fixture
def reloj_fijo(monkeypatch):
    """Congela el `time` que ve cameras.py (solo su nombre de módulo: el time
    global del proceso no se toca)."""
    falso = types.SimpleNamespace(
        time=lambda: _AHORA_FIJO,
        localtime=lambda t=None: time.localtime(_AHORA_FIJO if t is None else t),
        strftime=time.strftime,
    )
    monkeypatch.setattr(cameras, "time", falso)
    return _AHORA_FIJO


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def test_camera_events_evento_de_otro_dia_lleva_la_fecha(cams_cfg, monkeypatch, reloj_fijo):
    hace_una_hora = reloj_fijo - 3600          # hoy
    anoche = reloj_fijo - 16 * 3600            # ayer a las 23:00
    hace_dos_dias = reloj_fijo - 2 * 86400     # 2026-11-10
    _urls_frigate(monkeypatch, [_evento(0, hace_una_hora), _evento(1, anoche),
                               _evento(2, hace_dos_dias)])
    lineas = cameras.camera_events("todas", hours=72).splitlines()[1:]
    # El de hoy queda como estaba, sin fecha.
    assert lineas[0] == f"- coche en la cámara calle a las {_hhmm(hace_una_hora)} (8s)"
    assert lineas[1] == (f"- coche en la cámara calle el 2026-11-11 "
                         f"a las {_hhmm(anoche)} (8s)")
    assert lineas[2] == (f"- coche en la cámara calle el 2026-11-10 "
                         f"a las {_hhmm(hace_dos_dias)} (8s)")


def test_camera_events_evento_sin_hora_no_revienta(cams_cfg, monkeypatch, reloj_fijo):
    """Un evento sin start_time sigue saliendo con '??:??' (como antes)."""
    _urls_frigate(monkeypatch, [{"id": "x", "camera": "calle", "label": "dog"}])
    out = cameras.camera_events("todas", hours=72)
    assert "- perro en la cámara calle a las ??:?? (sigue en curso)" in out


# ---------------------------------------------------------------------------
# desktop.open_folder: el toml documenta [paths].documentos y [paths].descargas
# ("CERO hardcodeo"), pero FOLDERS se armaba solo con el home. Con
# documentos = "~/Documents", search_files buscaba ahí y "abre documentos"
# respondía "No encontré la carpeta ~/Documentos"; `descargas` no la leía nadie.
# ---------------------------------------------------------------------------

@pytest.fixture
def lanzados(monkeypatch) -> list[list[str]]:
    """Captura lo que open_folder mandaría a xdg-open, sin lanzar nada."""
    args: list[list[str]] = []
    monkeypatch.setattr(desktop, "run_detached", args.append)
    return args


@pytest.fixture
def paths_cfg(monkeypatch):
    """[paths] del singleton de settings, parcheable con setitem."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return real.paths


@pytest.mark.parametrize("clave, pedido", [
    ("documentos", "documentos"),
    ("descargas", "Descargas"),
    # La clave del toml se normaliza igual que el nombre que dicta el patrón.
    ("imágenes", "imagenes"),
], ids=["documentos", "descargas", "clave-con-acento"])
def test_open_folder_respeta_paths_de_la_config(monkeypatch, tmp_path, lanzados,
                                                paths_cfg, clave, pedido):
    carpeta = tmp_path / "Carpeta del patrón"
    carpeta.mkdir()
    monkeypatch.setitem(paths_cfg, clave, carpeta)
    out = desktop.open_folder(pedido)
    assert lanzados == [["xdg-open", str(carpeta)]]
    assert str(carpeta) in out


def test_open_folder_sin_clave_en_paths_usa_el_default(monkeypatch, tmp_path, lanzados,
                                                       paths_cfg):
    """Lo que [paths] no menciona sigue saliendo de FOLDERS."""
    musica = tmp_path / "Música"
    musica.mkdir()
    monkeypatch.setitem(desktop.FOLDERS, "musica", musica)
    monkeypatch.delitem(paths_cfg, "musica", raising=False)
    monkeypatch.delitem(paths_cfg, "música", raising=False)
    desktop.open_folder("música")
    assert lanzados == [["xdg-open", str(musica)]]


def test_open_folder_config_apuntando_a_carpeta_inexistente_lo_dice(monkeypatch, tmp_path,
                                                                    lanzados, paths_cfg):
    fantasma = tmp_path / "no-existe"
    monkeypatch.setitem(paths_cfg, "descargas", fantasma)
    out = desktop.open_folder("descargas")
    assert lanzados == []
    assert f"No encontré la carpeta {fantasma}" in out


@pytest.mark.parametrize("relativa", ["", "Descargas"], ids=["vacia", "sin-raiz"])
def test_open_folder_ignora_una_ruta_relativa_en_paths(monkeypatch, tmp_path, lanzados,
                                                       paths_cfg, relativa):
    """`descargas = ""` es Path('.') y existe siempre: se abría la carpeta
    desde donde se lanzó el proceso. Con una ruta relativa vale el default."""
    descargas = tmp_path / "Descargas"
    descargas.mkdir()
    monkeypatch.setitem(desktop.FOLDERS, "descargas", descargas)
    monkeypatch.setitem(paths_cfg, "descargas", Path(relativa))
    desktop.open_folder("descargas")
    assert lanzados == [["xdg-open", str(descargas)]]


@pytest.mark.parametrize("clave", ["db", "voces", "notas", "home"])
def test_paths_que_no_son_carpetas_del_catalogo_no_se_cuelan(monkeypatch, tmp_path,
                                                             lanzados, paths_cfg, clave):
    """[paths] también trae rutas que no son 'carpetas para abrir' (la sqlite,
    las voces): superponerlas todas habría cambiado el catálogo de open_folder."""
    monkeypatch.setitem(paths_cfg, clave, tmp_path)
    out = desktop.open_folder(clave)
    assert lanzados == []
    assert "No tengo registrada la carpeta" in out
    assert clave not in out.split("Conozco:")[1]
    assert desktop.resolve_target(clave)[0] == "search"


def test_open_folder_sin_config_legible_usa_los_defaults(monkeypatch, tmp_path, lanzados):
    """Mismo trato que _config_apps: sin config, valen los defaults."""
    def boom():
        raise FileNotFoundError("No encuentro la config")

    monkeypatch.setattr(settings_mod, "get_settings", boom)
    docs = tmp_path / "Documentos"
    docs.mkdir()
    monkeypatch.setitem(desktop.FOLDERS, "documentos", docs)
    desktop.open_folder("documentos")
    assert lanzados == [["xdg-open", str(docs)]]


# ---------------------------------------------------------------------------
# desktop.resolve_target (atajos aprendidos): solo miraba APP_COMMANDS, así que
# una app cableada en [apps] —que open_app y el fastpath SÍ conocen— se guardaba
# como búsqueda: "aprende 'chat' = discord" contestaba "Ya puedo dispararlo" y
# luego abría Google. Y el strip() llegaba DESPUÉS de mirar si era una URL (en
# C1 iba antes): " https://x.com" acababa de búsqueda de sí misma.
# ---------------------------------------------------------------------------

@pytest.fixture
def apps_cfg(monkeypatch) -> dict:
    """[apps] del singleton de settings: vacío y rellenable por cada test."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    apps: dict = {}
    monkeypatch.setitem(real.raw, "apps", apps)
    return apps


@pytest.mark.parametrize("clave, pedido", [
    ("discord", "discord"),
    ("Discord", " DISCORD "),
    ("editor de vídeo", "Editor de Video"),
], ids=["tal-cual", "mayusculas-y-espacios", "acentos"])
def test_resolve_target_conoce_las_apps_de_la_config(apps_cfg, clave, pedido):
    apps_cfg[clave] = ["lo-que-sea"]
    assert desktop.resolve_target(pedido) == ("app", {"value": desktop.normalize_key(pedido)})


def test_atajo_a_una_app_de_la_config_abre_la_app(apps_cfg, lanzados, monkeypatch, tmp_path):
    """De punta a punta: lo que se aprende es lo que luego se dispara."""
    monkeypatch.setitem(settings_mod.get_settings().paths, "db", tmp_path / "t.sqlite")
    apps_cfg["discord"] = ["discord", "--start-minimized"]
    shortcuts.learn_shortcut("chat", "discord")
    out = shortcuts.run_shortcut("chat")
    assert lanzados == [["discord", "--start-minimized"]]  # y no xdg-open google.com/search
    assert "Abrí discord" in out


def test_resolve_target_un_sitio_conocido_sigue_ganando_a_la_app(apps_cfg):
    """El orden heredado de C1 (sitios > apps > carpetas) NO se toca: 'spotify'
    está en COMMON_SITES y en el [apps] de fábrica, y el atajo sigue yendo a la web."""
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client"]
    assert desktop.resolve_target("spotify") == ("url", {"value": "https://open.spotify.com"})


@pytest.mark.parametrize("target", [" https://x.com", "\thttp://x.com/a\n", "https://x.com  "],
                         ids=["espacio-delante", "tab-y-salto", "espacios-detras"])
def test_resolve_target_url_con_espacios_alrededor_es_url(apps_cfg, target):
    assert desktop.resolve_target(target) == ("url", {"value": target.strip()})


def test_resolve_target_sin_config_legible_usa_los_defaults(monkeypatch):
    def boom():
        raise FileNotFoundError("No encuentro la config")

    monkeypatch.setattr(settings_mod, "get_settings", boom)
    assert desktop.resolve_target("blender") == ("app", {"value": "blender"})
    assert desktop.resolve_target("discord")[0] == "search"


# ---------------------------------------------------------------------------
# desktop.terminal_exec: open_app("terminal") respeta [apps].terminal, y tanto
# open_app como launch_project contestan "Cablea uno en [apps]" cuando no hay
# terminal... pero terminal_exec solo autodetectaba. Con terminal = ["ptyxis"]
# (el de Fedora, que no está en _TERMINALS) "abre terminal" funcionaba y
# launch_project seguía pidiendo que cablearas lo que ya estaba cableado.
# ---------------------------------------------------------------------------

_INNER = ["bash", "-lc", "echo hola"]


def _solo_instalado(monkeypatch, *binarios: str) -> None:
    """shutil.which de mentiras: solo existen los `binarios` dados."""
    monkeypatch.setattr(desktop.shutil, "which",
                        lambda cmd, *a, **kw: f"/usr/bin/{cmd}" if cmd in binarios else None)


@pytest.mark.parametrize("cableado, esperado", [
    (["ptyxis"], ["ptyxis", "--", *_INNER]),
    (["gnome-terminal"], ["gnome-terminal", "--", *_INNER]),
    (["wezterm"], ["wezterm", "start", "--", *_INNER]),
    (["foot", "--app-id=crotolamo"], ["foot", "--app-id=crotolamo", *_INNER]),
    # La sintaxis sale del NOMBRE del binario, venga con ruta o sin ella.
    (["/usr/bin/kitty"], ["/usr/bin/kitty", *_INNER]),
    # [apps] también admite el comando como una sola cadena.
    ("alacritty --class crot", ["alacritty", "--class", "crot", "-e", *_INNER]),
    # Uno que el código no conoce: -e, la convención de xterm.
    (["ghostty"], ["ghostty", "-e", *_INNER]),
], ids=["ptyxis", "gnome-terminal", "wezterm", "foot-con-args", "ruta-absoluta",
        "cadena", "desconocido"])
def test_terminal_exec_usa_el_terminal_cableado_en_apps(apps_cfg, monkeypatch,
                                                        cableado, esperado):
    apps_cfg["terminal"] = cableado
    _solo_instalado(monkeypatch)  # la autodetección no encuentra nada
    assert desktop.terminal_exec("echo hola") == esperado


def test_terminal_exec_y_open_app_abren_el_mismo_terminal(apps_cfg, monkeypatch, lanzados):
    """Misma precedencia que open_app: config del patrón > terminal detectado."""
    apps_cfg["terminal"] = ["foot"]
    _solo_instalado(monkeypatch, "kitty", "foot")  # kitty va antes en _TERMINALS
    desktop.open_app("terminal")
    assert lanzados == [["foot"]]
    assert desktop.terminal_exec("echo hola") == ["foot", *_INNER]


def test_launch_project_abre_el_terminal_cableado(apps_cfg, monkeypatch, fake_project):
    """El caso del hallazgo: sin terminal autodetectable pero con uno en [apps]."""
    apps_cfg["terminal"] = ["ptyxis"]
    _solo_instalado(monkeypatch)
    lanzados: list[list[str]] = []
    monkeypatch.setattr(desktop, "run_detached", lanzados.append)
    monkeypatch.setattr(projects, "run_detached", lanzados.append, raising=False)
    out = projects.launch_project("crotolamo")
    assert "Cablea uno en [apps]" not in out
    assert lanzados == [["ptyxis", "--", "bash", "-lc",
                         f"cd {shlex.quote(str(fake_project))}; exec bash"]]


@pytest.mark.parametrize("instalado, esperado", [
    ("kitty", ["kitty", *_INNER]),
    ("foot", ["foot", *_INNER]),
    ("alacritty", ["alacritty", "-e", *_INNER]),
    ("wezterm", ["wezterm", "start", "--", *_INNER]),
    ("gnome-terminal", ["gnome-terminal", "--", *_INNER]),
    ("konsole", ["konsole", "-e", *_INNER]),
    ("xterm", ["xterm", "-e", *_INNER]),
])
def test_terminal_exec_autodetectado_no_cambia(apps_cfg, monkeypatch, instalado, esperado):
    """Sin `terminal` en [apps] todo sigue como estaba, terminal por terminal."""
    _solo_instalado(monkeypatch, instalado)
    assert desktop.terminal_exec("echo hola") == esperado


@pytest.mark.parametrize("cableado", [[], ""], ids=["lista-vacia", "cadena-vacia"])
def test_terminal_exec_terminal_vacio_en_apps_cae_a_la_autodeteccion(apps_cfg, monkeypatch,
                                                                     cableado):
    apps_cfg["terminal"] = cableado
    _solo_instalado(monkeypatch, "kitty")
    assert desktop.terminal_exec("echo hola") == ["kitty", *_INNER]


def test_terminal_exec_sin_terminal_cableado_ni_instalado(apps_cfg, monkeypatch):
    _solo_instalado(monkeypatch)
    assert desktop.terminal_exec("echo hola") is None


# ---------------------------------------------------------------------------
# desktop.run_detached soltaba el Popen sin que nadie hiciera wait(): cada
# lanzador que termina (xdg-open sale enseguida) quedaba <defunct> hasta que
# OTRO subprocess disparaba la limpieza interna de subprocess, y Popen.__del__
# avisaba con "ResourceWarning: subprocess N is still running".
# ---------------------------------------------------------------------------

def _espera(condicion, timeout: float = 15.0) -> bool:
    """Sondea hasta que se cumpla; el margen largo es por la máquina cargada
    (solo se agota cuando el test va a fallar)."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if condicion():
            return True
        time.sleep(0.02)
    return bool(condicion())


def _estado_proc(pid: int) -> str | None:
    """Estado del proceso según /proc ('Z' = zombi), o None si ya no existe."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[0]


def _cgroup_falso(monkeypatch, tmp_path, contenido: str) -> None:
    """Hace creer a run_detached que el proceso vive en ese cgroup, y le hace
    olvidar lo que hubiera decidido antes (la decisión se guarda por proceso)."""
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(contenido, encoding="utf-8")
    monkeypatch.setattr(desktop, "_CGROUP", cgroup, raising=False)
    monkeypatch.setattr(desktop, "_scope_cache", None, raising=False)


@pytest.fixture
def fuera_de_servicio(monkeypatch, tmp_path):
    """Sesión normal (terminal, launch.sh): nada de systemd-run de por medio."""
    _cgroup_falso(monkeypatch, tmp_path, "0::/user.slice/user-1000.slice/session-2.scope\n")


@pytest.fixture
def popen_espia(monkeypatch):
    """Popen de verdad, pero guardando cada proceso creado para poder mirarlo."""
    creados: list[subprocess.Popen] = []
    real = subprocess.Popen

    def espia(*args, **kwargs):
        proc = real(*args, **kwargs)
        creados.append(proc)
        return proc

    monkeypatch.setattr(desktop.subprocess, "Popen", espia)
    yield creados
    for proc in creados:  # ningún test deja hijos sueltos, pase o falle
        proc.wait(timeout=10)


def test_run_detached_recoge_al_hijo_que_termina(fuera_de_servicio, popen_espia):
    desktop.run_detached(["true"])
    pid = popen_espia[-1].pid
    _espera(lambda: _estado_proc(pid) is None)
    assert _estado_proc(pid) is None, "el hijo quedó <defunct>: nadie hizo wait()"


def test_run_detached_no_suelta_un_popen_vivo(fuera_de_servicio, monkeypatch):
    pids: list[int] = []
    real = subprocess.Popen

    def espia(*args, **kwargs):
        proc = real(*args, **kwargs)
        pids.append(proc.pid)  # solo el pid: guardar el objeto taparía su __del__
        return proc

    monkeypatch.setattr(desktop.subprocess, "Popen", espia)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        desktop.run_detached(["sleep", "0.3"])
    # Solo el aviso de ESTE hijo: basura ajena recogida mientras tanto no cuenta.
    assert f"subprocess {pids[-1]} is still running" not in [str(a.message) for a in avisos]


# ---------------------------------------------------------------------------
# desktop.run_detached dentro de crotolamo.service: start_new_session cambia la
# sesión, NO el cgroup. Blender, VS Code o el terminal abiertos por voz vivían
# en el cgroup de la unidad y systemd (KillMode=control-group, el default) los
# mataba al parar o reiniciar el servicio (APAGAR / Reiniciar / cambiar el modo
# de escucha en el panel), con lo que tuvieran sin guardar. Ahora arrancan en
# un scope propio (`systemd-run --user --scope`), y solo si una sonda demuestra
# que aquí funciona: si no, se lanza exactamente como antes.
# ---------------------------------------------------------------------------

_SCOPE = "--user --scope --collect --quiet --"
_CGROUP_V2 = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/crotolamo.service\n"
_CGROUP_HIBRIDO = (
    "5:pids:/user.slice/user-1000.slice/user@1000.service\n"
    "1:name=systemd:/user.slice/user-1000.slice/user@1000.service/app.slice/crotolamo.service\n"
    "0::/user.slice/user-1000.slice/user@1000.service/app.slice\n"
)


@pytest.fixture
def systemd_run_falso(monkeypatch, tmp_path):
    """Pone delante en el PATH un `systemd-run` de mentiras que apunta su argv
    y, como el de verdad con --scope, hace exec de lo que va tras `--`. Con
    `falla()` pasa a salir con 1 sin lanzar nada (sin systemd de usuario).
    Devuelve el registro (una línea por invocación) y el interruptor."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    registro = tmp_path / "systemd-run.log"
    roto = tmp_path / "systemd-run.roto"
    (bindir / "systemd-run").write_text(
        "#!/bin/sh\n"
        f"echo \"$*\" >> {shlex.quote(str(registro))}\n"
        f"[ -e {shlex.quote(str(roto))} ] && exit 1\n"
        "while [ \"$1\" != \"--\" ]; do shift; done\n"
        "shift\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    (bindir / "systemd-run").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def lineas() -> list[str]:
        return registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []

    return lineas, roto.touch


@pytest.mark.parametrize("cgroup", [_CGROUP_V2, _CGROUP_HIBRIDO], ids=["cgroup-v2", "hibrido"])
def test_en_el_servicio_lo_lanzado_arranca_en_su_propio_scope(monkeypatch, tmp_path,
                                                             systemd_run_falso, cgroup):
    lineas, _falla = systemd_run_falso
    _cgroup_falso(monkeypatch, tmp_path, cgroup)
    marcas = [tmp_path / "lanzado-1", tmp_path / "lanzado-2"]
    for marca in marcas:
        desktop.run_detached(["touch", str(marca)])
    _espera(lambda: all(m.exists() for m in marcas) and len(lineas()) == 3)
    assert all(m.exists() for m in marcas)  # lo pedido se ejecutó de verdad
    # UNA sonda por proceso (no una por lanzamiento) y cada lanzamiento envuelto.
    assert lineas()[:1] == [f"{_SCOPE} true"]
    assert sorted(lineas()[1:]) == sorted(f"{_SCOPE} touch {m}" for m in marcas)


def test_en_el_servicio_sin_systemd_de_usuario_se_lanza_como_siempre(monkeypatch, tmp_path,
                                                                    systemd_run_falso):
    """systemd-run que no alcanza al systemd del usuario sale con error ANTES de
    lanzar nada: envolver a ciegas dejaría "abre blender" sin abrir nada."""
    lineas, falla = systemd_run_falso
    falla()
    _cgroup_falso(monkeypatch, tmp_path, _CGROUP_V2)
    marcas = [tmp_path / "lanzado-1", tmp_path / "lanzado-2"]
    for marca in marcas:
        desktop.run_detached(["touch", str(marca)])
    _espera(lambda: all(m.exists() for m in marcas))
    assert all(m.exists() for m in marcas)
    assert lineas() == [f"{_SCOPE} true"]  # la sonda, una vez; ningún lanzamiento por ahí


def test_fuera_del_servicio_ni_sonda_ni_scope(fuera_de_servicio, systemd_run_falso, tmp_path):
    lineas, _falla = systemd_run_falso
    marca = tmp_path / "lanzado"
    desktop.run_detached(["touch", str(marca)])
    _espera(marca.exists)
    assert marca.exists()
    assert lineas() == []


def test_sin_cgroup_legible_se_lanza_como_siempre(monkeypatch, tmp_path, systemd_run_falso):
    lineas, _falla = systemd_run_falso
    monkeypatch.setattr(desktop, "_CGROUP", tmp_path / "no-existe", raising=False)
    monkeypatch.setattr(desktop, "_scope_cache", None, raising=False)
    marca = tmp_path / "lanzado"
    desktop.run_detached(["touch", str(marca)])
    _espera(marca.exists)
    assert marca.exists()
    assert lineas() == []


def test_en_el_servicio_un_binario_que_no_existe_sigue_fallando(monkeypatch, tmp_path,
                                                               systemd_run_falso, apps_cfg):
    """open_app cuenta con el OSError de Popen para decir "No pude abrir":
    envuelto en systemd-run, Popen arrancaría bien y contestaría "Abrí"."""
    lineas, _falla = systemd_run_falso
    _cgroup_falso(monkeypatch, tmp_path, _CGROUP_V2)
    with pytest.raises(FileNotFoundError):
        desktop.run_detached(["crotolamo-binario-que-no-existe"])
    apps_cfg["fantasma"] = ["crotolamo-binario-que-no-existe"]
    assert desktop.open_app("fantasma") == "No pude abrir fantasma, patrón."
    assert not any("crotolamo-binario-que-no-existe" in linea for linea in lineas())


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired(cmd="systemd-run", timeout=3),
    PermissionError("systemd-run: permiso denegado"),
], ids=["sonda-colgada", "sonda-sin-permiso"])
def test_en_el_servicio_una_sonda_que_revienta_no_impide_lanzar(monkeypatch, tmp_path,
                                                               systemd_run_falso, error):
    lineas, _falla = systemd_run_falso
    _cgroup_falso(monkeypatch, tmp_path, _CGROUP_V2)
    sondas: list[list[str]] = []

    def sonda_rota(args, timeout=10):
        sondas.append(list(args))
        raise error

    monkeypatch.setattr(desktop, "run_cmd", sonda_rota)
    marcas = [tmp_path / "lanzado-1", tmp_path / "lanzado-2"]
    for marca in marcas:
        desktop.run_detached(["touch", str(marca)])
    _espera(lambda: all(m.exists() for m in marcas))
    assert all(m.exists() for m in marcas)
    # Se intentó UNA vez (no en cada lanzamiento) y nada pasó por systemd-run.
    assert sondas == [["systemd-run", *_SCOPE.split(), "true"]]
    assert lineas() == []


# ===========================================================================
# Lote 5
# ===========================================================================

# ---------------------------------------------------------------------------
# desktop.open_url / resolve_target: el esquema de una URL no distingue
# mayúsculas (RFC 3986 §3.1), pero el `^https?://` era sensible: a
# "HTTPS://Example.com/Ruta" se le anteponía OTRO https:// (y se abría
# "https://HTTPS://Example.com/Ruta") y como atajo se guardaba de búsqueda en
# Google de sí misma.
# ---------------------------------------------------------------------------

@pytest.fixture
def solo_xdg_open(monkeypatch, lanzados) -> list[list[str]]:
    """Sin flatpak: lo que open_url abre va derecho a xdg-open (capturado)."""
    _solo_instalado(monkeypatch, "xdg-open")
    return lanzados


@pytest.mark.parametrize("pedida, abierta", [
    ("HTTPS://Example.com/Ruta", "https://Example.com/Ruta"),
    ("Http://x.com/A?b=C", "http://x.com/A?b=C"),
    ("hTTp://x.com", "http://x.com"),
], ids=["https-mayusculas", "http-capitalizado", "http-mezclado"])
def test_open_url_el_esquema_no_distingue_mayusculas(solo_xdg_open, pedida, abierta):
    """Solo el esquema pasa a minúsculas (su forma canónica); la ruta y la
    query, que sí distinguen mayúsculas, llegan tal cual."""
    out = desktop.open_url(pedida)
    assert solo_xdg_open == [["xdg-open", abierta]]
    assert abierta in out


@pytest.mark.parametrize("pedida, abierta", [
    ("https://x.com/Ruta", "https://x.com/Ruta"),
    ("example.com", "https://example.com"),
    ("gatos persas", "https://www.google.com/search?q=gatos+persas"),
    ("gatos", "https://www.google.com/search?q=gatos"),
], ids=["url-normal", "dominio-pelon", "frase", "palabra"])
def test_open_url_lo_que_ya_funcionaba_no_cambia(solo_xdg_open, pedida, abierta):
    desktop.open_url(pedida)
    assert solo_xdg_open == [["xdg-open", abierta]]


def test_resolve_target_url_con_el_esquema_en_mayusculas_es_url(apps_cfg):
    assert desktop.resolve_target("HTTPS://Example.com/Ruta") == (
        "url", {"value": "HTTPS://Example.com/Ruta"})


def test_atajo_a_una_url_con_el_esquema_en_mayusculas_abre_la_url(apps_cfg, solo_xdg_open,
                                                                  monkeypatch, tmp_path):
    """De punta a punta: se aprende como URL y se abre ESA url, no su búsqueda."""
    monkeypatch.setitem(settings_mod.get_settings().paths, "db", tmp_path / "t.sqlite")
    shortcuts.learn_shortcut("portal", "HTTPS://Example.com/Ruta")
    shortcuts.run_shortcut("portal")
    assert solo_xdg_open == [["xdg-open", "https://Example.com/Ruta"]]


# ---------------------------------------------------------------------------
# desktop.open_app con `flatpak run`: Popen NO falla cuando la app no está
# instalada (flatpak arranca, escribe el error a DEVNULL y sale con 1), así que
# con el [apps] de fábrica (opera y spotify son flatpaks) "abre spotify"
# contestaba "Abrí spotify, patrón." sin abrir nada. open_url ya preguntaba
# antes de lanzar; open_app no.
# ---------------------------------------------------------------------------

_FLATPAK_LIST = ["flatpak", "list", "--all", "--columns=application"]


def _flatpak_falso(monkeypatch, instaladas: set[str], rc: int = 0) -> list[list[str]]:
    """Hay `flatpak`, y `flatpak list` enumera `instaladas` (más un runtime,
    como el de verdad). Devuelve las consultas hechas, para poder contarlas."""
    consultas: list[list[str]] = []
    salida = "".join(f"{ref}\n" for ref in ["org.freedesktop.Platform", *sorted(instaladas)])

    def fake_run_cmd(args, timeout=10):
        consultas.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=rc, stdout=salida, stderr="")

    monkeypatch.setattr(desktop, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(desktop.shutil, "which", lambda cmd, *a, **kw: f"/usr/bin/{cmd}")
    return consultas


def test_open_app_flatpak_sin_la_app_instalada_no_dice_que_abrio(apps_cfg, lanzados,
                                                                 monkeypatch):
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client"]
    consultas = _flatpak_falso(monkeypatch, instaladas={"com.opera.opera-gx"})
    out = desktop.open_app("spotify")
    assert out == "No pude abrir spotify, patrón."
    assert lanzados == []  # ni se lanza lo que se sabe que va a salir con 1
    assert consultas == [_FLATPAK_LIST]


def test_open_app_flatpak_con_la_app_instalada_la_abre(apps_cfg, lanzados, monkeypatch):
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client", "--uri=spotify:home"]
    consultas = _flatpak_falso(monkeypatch, instaladas={"com.spotify.Client"})
    out = desktop.open_app("spotify")
    assert out.endswith("Abrí spotify, patrón.")
    assert lanzados == [["flatpak", "run", "com.spotify.Client", "--uri=spotify:home"]]
    assert consultas == [_FLATPAK_LIST]  # UNA consulta por lanzamiento


def test_open_app_flatpak_con_ruta_pregunta_a_ese_mismo_binario(apps_cfg, lanzados,
                                                                monkeypatch):
    apps_cfg["spotify"] = ["/usr/bin/flatpak", "run", "com.spotify.Client"]
    consultas = _flatpak_falso(monkeypatch, instaladas=set())
    assert desktop.open_app("spotify") == "No pude abrir spotify, patrón."
    assert consultas == [["/usr/bin/flatpak", *_FLATPAK_LIST[1:]]]


def test_open_app_opera_de_fabrica_sin_el_flatpak_lo_dice(apps_cfg, lanzados, monkeypatch):
    """Sin [apps] mandan los defaults de APP_COMMANDS: el mismo `flatpak run`."""
    _flatpak_falso(monkeypatch, instaladas=set())
    assert desktop.open_app("opera") == "No pude abrir opera, patrón."
    assert lanzados == []


def test_open_app_que_no_es_flatpak_no_consulta_nada(apps_cfg, lanzados, monkeypatch):
    apps_cfg["blender"] = ["blender"]
    consultas = _flatpak_falso(monkeypatch, instaladas=set())
    assert desktop.open_app("blender").endswith("Abrí blender, patrón.")
    assert lanzados == [["blender"]]
    assert consultas == []  # ni un proceso de más para las apps de siempre


@pytest.mark.parametrize("argv", [
    ["flatpak", "run", "--branch=stable", "com.spotify.Client"],
    ["flatpak", "run", "com.spotify.Client//stable"],
    ["flatpak", "run"],
    ["flatpak", "update", "com.spotify.Client"],
], ids=["opciones-antes-del-id", "ref-con-rama", "sin-id", "otro-subcomando"])
def test_open_app_flatpak_que_no_se_sabe_leer_se_lanza_como_antes(apps_cfg, lanzados,
                                                                  monkeypatch, argv):
    """Solo se pregunta por un id pelón justo tras `run`: adivinarlo entre
    opciones o refs con rama podría dar por ausente una app que sí está."""
    apps_cfg["cosa"] = argv
    consultas = _flatpak_falso(monkeypatch, instaladas=set())
    assert desktop.open_app("cosa").endswith("Abrí cosa, patrón.")
    assert lanzados == [argv]
    assert consultas == []


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired(cmd="flatpak", timeout=3),
    PermissionError("flatpak: permiso denegado"),
], ids=["consulta-colgada", "consulta-sin-permiso"])
def test_open_app_si_no_se_puede_preguntar_a_flatpak_se_lanza_igual(apps_cfg, lanzados,
                                                                    monkeypatch, error):
    """No poder preguntar no es "no está instalada": se lanza como siempre."""
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client"]

    def consulta_rota(args, timeout=10):
        raise error

    monkeypatch.setattr(desktop, "run_cmd", consulta_rota)
    assert desktop.open_app("spotify").endswith("Abrí spotify, patrón.")
    assert lanzados == [["flatpak", "run", "com.spotify.Client"]]


def test_open_app_si_flatpak_list_falla_se_lanza_igual(apps_cfg, lanzados, monkeypatch):
    """Un listado que sale con error (flatpak viejo sin --columns) no dice nada
    de la app: tampoco ahí se da por ausente."""
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client"]
    _flatpak_falso(monkeypatch, instaladas=set(), rc=1)
    assert desktop.open_app("spotify").endswith("Abrí spotify, patrón.")
    assert lanzados == [["flatpak", "run", "com.spotify.Client"]]


@pytest.fixture
def flatpak_de_mentiras(monkeypatch, tmp_path, fuera_de_servicio):
    """Un `flatpak` ejecutable de verdad delante en el PATH con UNA app
    instalada: `run` de cualquier otra sale con 1 (como el real con una app
    ausente) y `run` de esa deja una marca."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marca = tmp_path / "lanzada"
    (bindir / "flatpak").write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = list ]; then\n"
        "  echo org.freedesktop.Platform; echo com.ejemplo.Instalada; exit 0\n"
        "fi\n"
        "[ \"$1\" = run ] && [ \"$2\" = com.ejemplo.Instalada ] || exit 1\n"
        f"touch {shlex.quote(str(marca))}\n",
        encoding="utf-8",
    )
    (bindir / "flatpak").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return marca


def test_open_app_flatpak_de_punta_a_punta(apps_cfg, flatpak_de_mentiras):
    """Con procesos de verdad: el `flatpak run` de una app ausente arranca y
    sale con 1 sin que Popen se entere; aun así open_app ya no canta victoria."""
    apps_cfg["spotify"] = ["flatpak", "run", "com.spotify.Client"]
    apps_cfg["instalada"] = ["flatpak", "run", "com.ejemplo.Instalada"]
    assert desktop.open_app("spotify") == "No pude abrir spotify, patrón."
    assert desktop.open_app("instalada").endswith("Abrí instalada, patrón.")
    _espera(flatpak_de_mentiras.exists)
    assert flatpak_de_mentiras.exists()  # la que sí está se lanzó de verdad


# ---------------------------------------------------------------------------
# desktop.open_url / open_folder sin xdg-open (o sin poder crear el proceso):
# open_app atrapaba el OSError de Popen y contestaba en personaje, pero los dos
# lanzamientos de xdg-open no, y la tool "reventaba" con el errno crudo del
# registry ("La tool 'open_url' reventó, patrón: [Errno 2] No such file...").
# ---------------------------------------------------------------------------

@pytest.fixture
def path_vacio(monkeypatch, tmp_path, fuera_de_servicio):
    """Un PATH sin nada: ni xdg-open ni flatpak. Popen falla de verdad."""
    vacio = tmp_path / "path-vacio"
    vacio.mkdir()
    monkeypatch.setenv("PATH", str(vacio))


def test_open_url_sin_xdg_open_responde_en_personaje(path_vacio):
    out = default_registry().run("open_url", {"url": "https://example.com"})
    assert not out.startswith("La tool '")
    assert out.startswith("No pude abrir https://example.com, patrón")
    assert "xdg-open" in out  # dice QUÉ falta, que es lo que hay que instalar


def test_search_web_sin_xdg_open_responde_en_personaje(path_vacio):
    out = default_registry().run("search_web", {"query": "gatos"})
    assert out.startswith("No pude abrir https://www.google.com/search?q=gatos, patrón")


def test_open_folder_sin_xdg_open_responde_en_personaje(path_vacio, monkeypatch, tmp_path,
                                                        paths_cfg):
    docs = tmp_path / "Documentos"
    docs.mkdir()
    monkeypatch.setitem(paths_cfg, "documentos", docs)
    out = default_registry().run("open_folder", {"name": "documentos"})
    assert not out.startswith("La tool '")
    assert out.startswith(f"No pude abrir {docs}, patrón")
    assert "xdg-open" in out


# ---------------------------------------------------------------------------
# home.light_control: el POST al servicio devuelve SOLO los estados que
# cambiaron MIENTRAS corría la llamada. [] no es "entity_id inexistente": es
# también lo que llega al prender una luz que ya está prendida y con las
# integraciones que actualizan el estado después (MQTT/Zigbee2MQTT, ESPHome).
# En todos esos casos contestaba "Home Assistant no reconoció la luz... Revisa
# el entity_id" con la orden cumplida y la config bien. Ahora, ante [], se
# pregunta por la entidad: solo un 404 es "no la reconoció".
# ---------------------------------------------------------------------------

@pytest.fixture
def hass_cfg(monkeypatch):
    """[home] de mentiras en el singleton de settings + token en la env."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    monkeypatch.setitem(real.raw, "home", {
        "base_url": "http://hass.test:8123",
        "lights": {"xbox": "light.xbox_led", "recamara": "light.recámara"},
    })
    monkeypatch.setenv(_hass.TOKEN_ENV, "token-de-mentiras")


def _hass_falso(monkeypatch, post, get) -> list[tuple[str, str]]:
    """_http_json de mentiras: `post` contesta el POST y `get` el GET (si es
    una excepción, se lanza). Devuelve las (método, url) pedidas, en orden."""
    pedidas: list[tuple[str, str]] = []

    def fake(method, url, headers, payload=None, timeout=5.0):
        pedidas.append((method, url))
        respuesta = post if method == "POST" else get
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setattr(_hass, "_http_json", fake)
    return pedidas


def _http_404(url: str = "http://hass.test:8123") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 404, "Not Found", None, None)


_POST_ON = ("POST", "http://hass.test:8123/api/services/light/turn_on")
_GET_XBOX = ("GET", "http://hass.test:8123/api/states/light.xbox_led")


@pytest.mark.parametrize("accion, estado, hecho", [
    ("on", "on", "prendí"),          # ya estaba prendida: llamada idempotente
    ("off", "off", "apagué"),
    ("on", "off", "prendí"),         # el estado llega después (MQTT, ESPHome)
    ("toggle", "on", "le di la vuelta a"),
    ("on", "unknown", "prendí"),     # MQTT antes de su primer mensaje de estado
], ids=["on-ya-prendida", "off-ya-apagada", "estado-asincrono", "toggle", "estado-unknown"])
def test_light_sin_cambios_de_una_luz_que_existe_es_exito(hass_cfg, monkeypatch,
                                                          accion, estado, hecho):
    _hass_falso(monkeypatch, post=[], get={"state": estado, "attributes": {}})
    out = home.light_control("xbox", accion)
    assert out == f"Listo, patrón: {hecho} la luz 'xbox'."
    assert "entity_id" not in out


def test_light_sin_cambios_pregunta_por_la_entidad_una_vez(hass_cfg, monkeypatch):
    pedidas = _hass_falso(monkeypatch, post=[], get={"state": "on", "attributes": {}})
    home.light_control("xbox", "on")
    assert pedidas == [_POST_ON, _GET_XBOX]


def test_light_con_cambios_no_hace_ninguna_request_de_mas(hass_cfg, monkeypatch):
    """El camino normal (HA devuelve el estado que cambió) sigue en UNA request."""
    pedidas = _hass_falso(monkeypatch, post=[{"entity_id": "light.xbox_led", "state": "on"}],
                          get=AssertionError("no hacía falta preguntar"))
    assert home.light_control("xbox", "on") == "Listo, patrón: prendí la luz 'xbox'."
    assert pedidas == [_POST_ON]


def test_light_entity_id_inexistente_sigue_sin_cantar_victoria(hass_cfg, monkeypatch):
    """El caso que el aviso quería cazar (typo en [home.lights]): HA contesta
    [] al servicio y 404 al preguntar por la entidad."""
    pedidas = _hass_falso(monkeypatch, post=[], get=_http_404())
    out = home.light_control("xbox", "on")
    assert out == ("Home Assistant no reconoció la luz 'xbox', patrón. "
                   "Revisa el entity_id 'light.xbox_led' en [home.lights] de la config.")
    assert pedidas == [_POST_ON, _GET_XBOX]


def test_light_no_disponible_no_dice_que_prendio(hass_cfg, monkeypatch):
    """HA se salta las entidades `unavailable` (foco desenchufado): [] y no
    pasó nada. Ni "prendí" ni "revisa el entity_id", que está bien."""
    _hass_falso(monkeypatch, post=[], get={"state": "unavailable", "attributes": {}})
    out = home.light_control("xbox", "on")
    assert "prendí" not in out and "entity_id" not in out
    assert "no disponible" in out and "xbox" in out and "patrón" in out


@pytest.mark.parametrize("fallo", [
    urllib.error.URLError("timed out"),
    urllib.error.HTTPError("http://hass.test:8123", 500, "Server Error", None, None),
    json.JSONDecodeError("basura", "<html>", 0),
], ids=["sin-conexion", "http-500", "no-json"])
def test_light_si_la_comprobacion_falla_vale_el_200_del_servicio(hass_cfg, monkeypatch, fallo):
    """HA aceptó la orden (200) y lo único que no se pudo fue comprobar el
    entity_id: no se convierte una orden aceptada en un fallo."""
    _hass_falso(monkeypatch, post=[], get=fallo)
    assert home.light_control("xbox", "on") == "Listo, patrón: prendí la luz 'xbox'."


def test_light_comprobacion_con_respuesta_rara_no_revienta(hass_cfg, monkeypatch):
    """GET que devuelve una lista (proxy raro): no es un 404, no revienta."""
    _hass_falso(monkeypatch, post=[], get=[{"state": "on"}])
    assert home.light_control("xbox", "on") == "Listo, patrón: prendí la luz 'xbox'."


def test_light_la_comprobacion_va_con_el_entity_id_codificado(hass_cfg, monkeypatch):
    """Igual que home_state: un entity_id con acento no puede reventar la URL."""
    pedidas = _hass_falso(monkeypatch, post=[], get={"state": "off", "attributes": {}})
    assert home.light_control("recamara", "off") == "Listo, patrón: apagué la luz 'recamara'."
    assert pedidas[1] == ("GET", "http://hass.test:8123/api/states/light.rec%C3%A1mara")


# ---------------------------------------------------------------------------
# windows: "cierra opera" cerraba la primera ventana que llevara "opera" en
# CUALQUIER parte, en el orden de hyprctl: un kitty con `nvim notas_opera.md`
# abierto le ganaba al navegador. La app (class) va antes que el título.
# ---------------------------------------------------------------------------

def _ventana(clase: str, titulo: str, address: str, ws: int = 1) -> dict:
    return {"class": clase, "title": titulo, "address": address, "workspace": {"id": ws}}


@pytest.fixture
def hypr(monkeypatch):
    """Hyprland de mentiras: `ventanas` es lo que devuelve `hyprctl clients -j`
    y `dispatch` apunta lo que se le mandó hacer."""
    estado = types.SimpleNamespace(ventanas=[], dispatch=[])

    def fake_dispatch(args):
        estado.dispatch.append(args)
        return subprocess.CompletedProcess(["hyprctl", "dispatch", *args], 0, "ok\n", "")

    monkeypatch.setattr(windows, "_clients", lambda: estado.ventanas)
    monkeypatch.setattr(windows, "_dispatch", fake_dispatch)
    return estado


def test_close_window_la_app_le_gana_al_titulo_de_otra_ventana(hypr):
    hypr.ventanas = [
        _ventana("kitty", "nvim notas_opera.md", "0xaaa"),
        _ventana("Opera-GX", "YouTube", "0xbbb", ws=2),
    ]
    out = windows.close_window("opera")
    assert hypr.dispatch == [["closewindow", "address:0xbbb"]]
    assert "Opera-GX" in out and "SOLO la primera" in out


def test_focus_window_la_app_le_gana_al_titulo_de_otra_ventana(hypr):
    hypr.ventanas = [
        _ventana("firefox", "Visual Studio Code - Docs — Mozilla Firefox", "0xaaa", ws=2),
        _ventana("Code", "agent.py — crotolamo2", "0xbbb", ws=3),
    ]
    out = windows.focus_window("code")
    assert hypr.dispatch == [["focuswindow", "address:0xbbb"]]
    assert "Code — agent.py" in out


def test_match_clase_exacta_antes_que_clase_parecida_y_que_titulo(hypr):
    ventanas = [
        _ventana("firefox", "cómo configurar kitty", "0x1"),
        _ventana("kitty-dropdown", "~", "0x2"),
        _ventana("Kitty", "~/Documentos", "0x3"),
        _ventana("kitty", "htop", "0x4"),
    ]
    orden = [c["address"] for c in windows._match(ventanas, "kitty")]
    # Clase exacta (en el orden de hyprctl), luego clase que lo contiene, luego título.
    assert orden == ["0x3", "0x4", "0x2", "0x1"]


def test_match_solo_por_titulo_sigue_funcionando_y_en_orden(hypr):
    hypr.ventanas = [
        _ventana("firefox", "YouTube — Mozilla Firefox", "0x1"),
        _ventana("Opera-GX", "YouTube Music", "0x2"),
    ]
    assert [c["address"] for c in windows._match(hypr.ventanas, "youtube")] == ["0x1", "0x2"]
    windows.focus_window("youtube")
    assert hypr.dispatch == [["focuswindow", "address:0x1"]]


def test_app_status_cuenta_las_mismas_ventanas_que_antes(hypr):
    """El orden nuevo no cambia QUÉ ventanas coinciden, solo cuál va primero."""
    hypr.ventanas = [
        _ventana("kitty", "nvim notas_opera.md", "0xaaa"),
        _ventana("Opera-GX", "YouTube", "0xbbb", ws=2),
    ]
    assert windows.app_status("opera") == (
        "Sí, patrón: 'opera' está abierta con 2 ventanas en los workspaces 1, 2.")


# ---------------------------------------------------------------------------
# windows: cualquier fallo de hyprctl se contaba como "No tengo hyprctl". Con
# el listener arrancado por systemd sin HYPRLAND_INSTANCE_SIGNATURE, hyprctl
# ESTÁ instalado y sale con 1: se le decía al patrón que le faltaba algo que
# tiene, y el motivo real (la salida de hyprctl) se tiraba sin dejar rastro.
# ---------------------------------------------------------------------------

_SIN_FIRMA = "HYPRLAND_INSTANCE_SIGNATURE not set! (is hyprland running?)"
_LAS_CUATRO = [
    lambda: windows.list_windows(),
    lambda: windows.app_status("opera"),
    lambda: windows.focus_window("opera"),
    lambda: windows.close_window("opera"),
]
_IDS_CUATRO = ["list_windows", "app_status", "focus_window", "close_window"]


@pytest.fixture
def hyprctl_falso(monkeypatch):
    """hyprctl instalado cuyo `clients -j` contesta lo que diga `estado.respuesta`
    (un CompletedProcess o una excepción). Apunta cada argv que se le pasa."""
    estado = types.SimpleNamespace(respuesta=None, llamadas=[], instalado=True)

    def fake_run(args):
        estado.llamadas.append(list(args))
        if isinstance(estado.respuesta, BaseException):
            raise estado.respuesta
        return estado.respuesta

    monkeypatch.setattr(windows, "_run", fake_run)
    monkeypatch.setattr(windows.shutil, "which",
                        lambda cmd: f"/usr/bin/{cmd}" if estado.instalado else None)
    return estado


def _hyprctl_dijo(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(["hyprctl", "clients", "-j"], returncode, stdout, stderr)


@pytest.mark.parametrize("llamar", _LAS_CUATRO, ids=_IDS_CUATRO)
def test_hyprctl_instalado_que_falla_no_es_no_tengo_hyprctl(hyprctl_falso, caplog, llamar):
    hyprctl_falso.respuesta = _hyprctl_dijo(1, stdout=_SIN_FIRMA + "\n")
    with caplog.at_level(logging.WARNING, logger="crotolamo"):
        out = llamar()
    assert "No tengo hyprctl" not in out
    assert out == f"hyprctl se quejó, patrón: {_SIN_FIRMA}"
    # Solo se preguntó por las ventanas: sin lista no se enfoca ni se cierra nada.
    assert hyprctl_falso.llamadas == [["clients", "-j"]]
    # Y el motivo queda en el log, con el código de salida.
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(_SIN_FIRMA in a and "1" in a for a in avisos)


def test_hyprctl_que_falla_dice_lo_de_stderr_si_lo_hay(hyprctl_falso):
    hyprctl_falso.respuesta = _hyprctl_dijo(
        3, stderr="\nCouldn't connect to /run/user/1000/hypr/x/.socket.sock. (3)\nmás\n")
    assert windows.list_windows() == (
        "hyprctl se quejó, patrón: Couldn't connect to /run/user/1000/hypr/x/.socket.sock. (3)")


def test_hyprctl_que_falla_sin_decir_nada(hyprctl_falso):
    hyprctl_falso.respuesta = _hyprctl_dijo(1)
    out = windows.list_windows()
    assert out == "hyprctl se quejó, patrón: salió con código 1 sin dar detalles"


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired(cmd="hyprctl", timeout=5),
    OSError(13, "Permission denied"),
], ids=["timeout", "oserror"])
def test_hyprctl_que_no_contesta_no_es_no_tengo_hyprctl(hyprctl_falso, caplog, error):
    hyprctl_falso.respuesta = error
    with caplog.at_level(logging.WARNING, logger="crotolamo"):
        out = windows.app_status("opera")
    assert out == "No pude hablar con Hyprland, patrón: hyprctl no me contestó."
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("stdout", [
    _SIN_FIRMA + "\n",           # versiones de hyprctl que avisan y salen con 0
    '{"error": "algo"}',          # JSON, pero no la lista de ventanas
    '"unknown request"',
], ids=["texto", "objeto", "cadena"])
def test_hyprctl_que_contesta_otra_cosa_no_es_no_tengo_hyprctl(hyprctl_falso, caplog, stdout):
    hyprctl_falso.respuesta = _hyprctl_dijo(0, stdout=stdout)
    with caplog.at_level(logging.WARNING, logger="crotolamo"):
        out = windows.close_window("opera")
    assert "No tengo hyprctl" not in out
    assert out == f"hyprctl me contestó algo que no entiendo, patrón: {stdout.strip()}"
    assert hyprctl_falso.llamadas == [["clients", "-j"]]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_hyprctl_detalle_larguisimo_se_recorta(hyprctl_falso):
    hyprctl_falso.respuesta = _hyprctl_dijo(1, stdout="x" * 5000)
    out = windows.list_windows()
    assert len(out) < 300 and out.endswith("…")


@pytest.mark.parametrize("llamar", _LAS_CUATRO, ids=_IDS_CUATRO)
def test_sin_hyprctl_instalado_se_sigue_diciendo_igual(hyprctl_falso, llamar):
    hyprctl_falso.instalado = False
    assert llamar() == "No tengo hyprctl, patrón. Esto solo funciona dentro de Hyprland."
    assert hyprctl_falso.llamadas == []


def test_hyprctl_sano_no_cambia(hyprctl_falso, caplog):
    hyprctl_falso.respuesta = _hyprctl_dijo(0, stdout=json.dumps([
        _ventana("Opera-GX", "YouTube", "0xbbb", ws=2)]))
    with caplog.at_level(logging.WARNING, logger="crotolamo"):
        assert windows.list_windows() == (
            "Tienes 1 ventana abierta, patrón:\nworkspace 2: Opera-GX — YouTube")
        hyprctl_falso.respuesta = _hyprctl_dijo(0, stdout="[]")
        assert windows.list_windows() == (
            "No hay ni una ventana abierta, patrón. Escritorio zen total.")
        # stdout vacío con salida 0: hyprctl no listó nada, igual que antes.
        for nada in ("", "\n"):
            hyprctl_falso.respuesta = _hyprctl_dijo(0, stdout=nada)
            assert "Escritorio zen" in windows.list_windows()
    assert not caplog.records


# ---------------------------------------------------------------------------
# media: music_now / music_control llamaban a playerctl sin elegir reproductor,
# y playerctl manda entonces la orden al PRIMERO que la admita, suene o no
# (README de playerctl: "will act on the first player it can find"; `pause` la
# admite cualquiera con CanPause). Con una pestaña en pausa delante de Spotify,
# "pausa la música" pausaba la pestaña ya pausada, Spotify seguía sonando y la
# tool contestaba "Pausado, patrón."; "¿qué suena?" describía la pestaña.
# El docstring promete "vacío = el activo".
# ---------------------------------------------------------------------------

def _rep(nombre: str, estado: str, artista: str = "", titulo: str = "", *,
         sin_pista: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(nombre=nombre, estado=estado, artista=artista,
                                 titulo=titulo, sin_pista=sin_pista, saltos=0)


class _PlayerctlFalso:
    """playerctl de mentiras con las reglas del de verdad (playerctl-cli.c):

    - sin -a, la orden va al primer reproductor (de los elegidos con -p, o de
      todos) que la admite, y ahí se acaba; con -a, a todos;
    - `metadata --format` se salta al reproductor que no tiene pista;
    - sin reproductores: salida 1 y "No players found" por stderr.
    """

    def __init__(self, *reproductores: types.SimpleNamespace) -> None:
        self.reproductores = list(reproductores)
        self.argvs: list[list[str]] = []
        self.error: BaseException | None = None

    def __getitem__(self, nombre: str) -> types.SimpleNamespace:
        return next(r for r in self.reproductores if r.nombre == nombre)

    def estados(self) -> dict[str, str]:
        return {r.nombre: r.estado for r in self.reproductores}

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.argvs.append(list(args))
        if self.error is not None:
            raise self.error
        resto = list(args)
        todos = "-a" in resto
        if todos:
            resto.remove("-a")
        elegidos = self.reproductores
        if "-p" in resto:
            i = resto.index("-p")
            nombre = resto[i + 1]
            del resto[i:i + 2]
            # Como playerctl: el nombre tal cual, distinguiendo mayúsculas.
            elegidos = [r for r in elegidos if r.nombre == nombre]
        formato = ""
        if "--format" in resto:
            i = resto.index("--format")
            formato = resto[i + 1]
            del resto[i:i + 2]
        (orden,) = resto
        if not elegidos:
            return subprocess.CompletedProcess(args, 1, "", "No players found\n")
        salida, alguno = "", False
        for rep in elegidos:
            hecho, texto = self._ordena(rep, orden, formato)
            if not hecho:
                continue
            salida += texto
            alguno = True
            if not todos:
                break
        if not alguno:
            return subprocess.CompletedProcess(
                args, 1, "", "No player could handle this command\n")
        return subprocess.CompletedProcess(args, 0, salida, "")

    @staticmethod
    def _ordena(rep: types.SimpleNamespace, orden: str, formato: str) -> tuple[bool, str]:
        if orden == "metadata":
            if rep.sin_pista:
                return False, ""
            linea = formato
            for clave, valor in (("playerName", rep.nombre), ("status", rep.estado),
                                 ("artist", rep.artista), ("title", rep.titulo)):
                linea = linea.replace("{{%s}}" % clave, valor)
            return True, linea + "\n"
        if orden == "status":
            return True, rep.estado + "\n"
        if orden == "play-pause":
            rep.estado = "Paused" if rep.estado == "Playing" else "Playing"
        elif orden in ("next", "previous"):
            rep.saltos += 1 if orden == "next" else -1
        else:
            rep.estado = {"play": "Playing", "pause": "Paused", "stop": "Stopped"}[orden]
        return True, ""


@pytest.fixture
def playerctl(monkeypatch):
    def instala(*reproductores: types.SimpleNamespace) -> _PlayerctlFalso:
        falso = _PlayerctlFalso(*reproductores)
        monkeypatch.setattr(media, "_run", falso)
        monkeypatch.setattr(media.shutil, "which", lambda _cmd: "/usr/bin/playerctl")
        return falso

    return instala


def _pestana_y_spotify() -> list[types.SimpleNamespace]:
    """El montaje del README: una pestaña del navegador en pausa, que playerctl
    lista primero, y Spotify sonando."""
    return [_rep("opera", "Paused", "Canal X", "Video pausado hace una hora"),
            _rep("spotify", "Playing", "Artista", "Cancion")]


def test_music_now_dice_lo_que_suena_no_el_primer_reproductor(playerctl):
    playerctl(*_pestana_y_spotify())
    assert media.music_now() == "Sonando, patrón: Artista — Cancion"


def test_music_now_un_solo_playerctl(playerctl):
    """Antes eran dos procesos (metadata y status) para una sola frase."""
    falso = playerctl(_rep("spotify", "Playing", "Artista", "Cancion"))
    assert media.music_now() == "Sonando, patrón: Artista — Cancion"
    assert len(falso.argvs) == 1


def test_music_now_no_mezcla_dos_reproductores(playerctl):
    """`metadata` se salta al que no tiene pista y `status` no: la frase salía
    con el estado de uno ("Sonando") y la canción de otro."""
    playerctl(_rep("mpv", "Playing", sin_pista=True),
              _rep("spotify", "Paused", "Artista", "Cancion"))
    assert media.music_now() == "En pausa, patrón: Artista — Cancion"


@pytest.mark.parametrize("reproductores, esperado", [
    ([_rep("opera", "Paused", "Canal X", "Video"), _rep("spotify", "Paused", "A", "T")],
     "En pausa, patrón: Canal X — Video"),                    # nadie suena: el primero
    ([_rep("spotify", "Stopped", "A", "T")], "Stopped, patrón: A — T"),
    ([_rep("spotify", "Paused", "", "Solo titulo")], "En pausa, patrón: Solo titulo"),
    ([_rep("spotify", "Paused")], "Reproductor en estado «paused», patrón, pero sin metadatos."),
    ([_rep("vlc", "Stopped", sin_pista=True)],
     "Reproductor en estado «stopped», patrón, pero sin metadatos."),
    ([], "No hay nada sonando, patrón. El silencio también es música."),
    ([_rep("spotify", "Playing", "Artista", "Lado A\tLado B")],
     "Sonando, patrón: Artista — Lado A\tLado B"),            # un tabulador en el título
], ids=["nadie-suena", "parado", "sin-artista", "sin-metadatos", "sin-pista",
        "sin-reproductores", "tab-en-el-titulo"])
def test_music_now_lo_demas_se_dice_igual(playerctl, reproductores, esperado):
    playerctl(*reproductores)
    assert media.music_now() == esperado


@pytest.mark.parametrize("accion, orden, queda", [
    ("pause", "pause", "Paused"),
    ("toggle", "play-pause", "Paused"),
    ("play-pause", "play-pause", "Paused"),
    ("stop", "stop", "Stopped"),
    ("parar", "stop", "Stopped"),
])
def test_music_control_para_lo_que_suena(playerctl, accion, orden, queda):
    falso = playerctl(*_pestana_y_spotify())
    out = media.music_control(accion)
    # Spotify deja de sonar y a la pestaña ni se la toca (con `toggle`, antes
    # arrancaba la pestaña ENCIMA de Spotify).
    assert falso.estados() == {"opera": "Paused", "spotify": queda}
    # Dos playerctl y ni uno más: la pregunta de quién suena y la orden.
    assert falso.argvs == [["-a", "metadata", "--format", media._LISTADO],
                           ["-p", "spotify", orden]]
    assert out.endswith(", patrón.")


@pytest.mark.parametrize("accion, salto", [
    ("next", 1), ("siguiente", 1), ("previous", -1), ("anterior", -1),
])
def test_music_control_cambia_de_cancion_en_lo_que_suena(playerctl, accion, salto):
    falso = playerctl(*_pestana_y_spotify())
    media.music_control(accion)
    assert (falso["opera"].saltos, falso["spotify"].saltos) == (0, salto)


def test_music_control_usa_el_nombre_tal_cual_lo_da_playerctl(playerctl):
    """`-p` distingue mayúsculas: el nombre que imprime playerctl no se toca."""
    falso = playerctl(_rep("firefox", "Paused", "", "Video"),
                      _rep("Lollypop", "Playing", "A", "T"))
    assert media.music_control("pause") == "Pausado, patrón."
    assert falso.argvs[-1] == ["-p", "Lollypop", "pause"]
    assert falso.estados() == {"firefox": "Paused", "Lollypop": "Paused"}


def test_music_control_sin_nada_sonando_hace_lo_de_siempre(playerctl):
    """Nadie en Playing: la orden va sin `-p`, al que elija playerctl."""
    falso = playerctl(_rep("opera", "Paused", "Canal X", "Video"),
                      _rep("spotify", "Paused", "A", "T"))
    assert media.music_control("next") == "Siguiente, patrón."
    assert falso.argvs[-1] == ["next"]
    assert (falso["opera"].saltos, falso["spotify"].saltos) == (1, 0)


def test_music_control_play_no_pregunta_quien_suena(playerctl):
    """`play` es para lo que NO suena: mandárselo al que ya suena no haría
    nada. Se queda como estaba: un solo playerctl, sin elegir."""
    falso = playerctl(*_pestana_y_spotify())
    assert media.music_control("play") == "Dale play, patrón."
    assert falso.argvs == [["play"]]


def test_music_control_con_reproductor_explicito_no_pregunta(playerctl):
    falso = playerctl(_rep("vlc", "Playing", "A", "T"), _rep("spotify", "Playing", "B", "U"))
    assert media.music_control("pause", "VLC") == "Pausado, patrón."
    assert falso.argvs == [["-p", "vlc", "pause"]]
    assert falso.estados() == {"vlc": "Paused", "spotify": "Playing"}


def test_music_control_sin_reproductores_se_dice_igual(playerctl):
    playerctl()
    assert media.music_control("pause") == (
        "No hay ningún reproductor abierto, patrón. Abre Spotify primero.")


def test_music_control_si_playerctl_no_contesta_no_se_espera_dos_veces(playerctl):
    """Si la pregunta de quién suena ya se colgó (8 s), no se lanza la orden a
    colgarse otros 8."""
    falso = playerctl(*_pestana_y_spotify())
    falso.error = subprocess.TimeoutExpired(cmd="playerctl", timeout=8)
    assert media.music_control("pause") == "No pude controlar la música, patrón."
    assert len(falso.argvs) == 1


# playerctl de mentiras para la prueba de punta a punta: los reproductores están
# en un JSON (argv[1]) y vale la regla de playerctl: sin -p/-a, la orden es para
# el primero.
_PLAYERCTL_EJECUTABLE = r"""
import json, sys
ruta = sys.argv[1]
reps = json.load(open(ruta))
args = sys.argv[2:]
todos = "-a" in args
if todos:
    args.remove("-a")
elegidos = reps
if "-p" in args:
    i = args.index("-p")
    elegidos = [r for r in reps if r[0] == args[i + 1]]
    del args[i:i + 2]
formato = ""
if "--format" in args:
    i = args.index("--format")
    formato = args[i + 1]
    del args[i:i + 2]
orden = args[0]
if not elegidos:
    sys.exit("No players found")
for rep in elegidos:
    if orden == "metadata":
        linea = formato
        for clave, valor in zip(("playerName", "status", "artist", "title"), rep):
            linea = linea.replace("{{" + clave + "}}", valor)
        print(linea)
    elif orden == "status":
        print(rep[1])
    else:
        rep[1] = {"pause": "Paused", "play": "Playing", "stop": "Stopped"}[orden]
    if not todos:
        break
json.dump(reps, open(ruta, "w"))
"""


def test_pausa_la_musica_de_punta_a_punta(tmp_path, monkeypatch):
    """Con un `playerctl` ejecutable de verdad en el PATH: el formato con
    tabuladores viaja intacto por argv y la pausa llega a Spotify."""
    estado = tmp_path / "reproductores.json"
    estado.write_text(json.dumps([["opera", "Paused", "Canal X", "Video"],
                                  ["spotify", "Playing", "Artista", "Cancion"]]))
    binarios = tmp_path / "bin"
    binarios.mkdir()
    guion = tmp_path / "playerctl_falso.py"
    guion.write_text(_PLAYERCTL_EJECUTABLE)
    falso = binarios / "playerctl"
    falso.write_text("#!/bin/sh\nexec " + shlex.join([sys.executable, str(guion), str(estado)])
                     + ' "$@"\n')
    falso.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binarios}{os.pathsep}{os.environ.get('PATH', '')}")

    assert media.music_now() == "Sonando, patrón: Artista — Cancion"
    assert media.music_control("pause") == "Pausado, patrón."
    assert [r[:2] for r in json.loads(estado.read_text())] == [
        ["opera", "Paused"], ["spotify", "Paused"]]
    assert media.music_now() == "En pausa, patrón: Canal X — Video"
