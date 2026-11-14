"""Revisión final, unidad 20: crotolamo/tools/_web.py (el GET pinneado de las tools web).

1. El plazo de la llamada ENTERA no cubría conectar ni el handshake TLS. El
   tiempo restante se calculaba ANTES del DNS y ese número viejo era el timeout
   del connect, que además heredaba el handshake TLS (un plazo casi entero
   estrenado). read_page (12 s) podía tener el hilo del Brain DNS + ~24 s.
   Ahora lo que queda del plazo se vuelve a mirar después del DNS (si ya no
   queda, TimeoutError sin conectar), conectar usa lo que queda y el handshake
   TLS también. getaddrinfo en sí no se puede cortar: lo acota el resolver.
2. Se conectaba solo a la PRIMERA IP de la resolución validada. Con un host de
   doble pila o de varias A cuya primera IP no contesta (IPv6 roto en la red
   del portátil, un nodo caído), read_page fallaba aunque la siguiente IP, ya
   validada como pública, funcionara (urllib, curl y los navegadores pasan a la
   siguiente). Ahora se prueban todas en orden, dentro del mismo plazo
   repartido, y sin volver a resolver nunca (el pinning sigue en pie).
3. Ningún test hacía un handshake TLS de verdad por la conexión pinneada: un
   server_hostname con la IP (SNI y verificación del hostname rotos) o un
   _DeadlineIO que fallara sobre un SSLSocket rompía TODAS las lecturas https
   (DuckDuckGo y casi todas las páginas) sin que fallara nada.
"""

from __future__ import annotations

import gzip
import http.server
import shutil
import socket
import ssl
import subprocess
import threading
import time

import pytest

from crotolamo.tools import _web

# Holgura del reloj para las comprobaciones de plazo: la máquina de pruebas es
# compartida. Lo que se mide antes del arreglo está siempre por encima (los
# sleep nunca vuelven antes de tiempo).
_HOLGURA_S = 0.45


def _dns(monkeypatch, nombres: dict[str, list[str]], retraso: float = 0.0) -> list[str]:
    """getaddrinfo de mentiras para `nombres` (host -> IPs, tardando `retraso`);
    el resto (las IPs literales que resuelve create_connection) va al real.
    Las IPs de loopback del servidor local cuentan como públicas. Devuelve la
    lista de hosts consultados."""
    real = socket.getaddrinfo
    consultas: list[str] = []

    def fake(host, port, *a, **kw):
        if host not in nombres:
            return real(host, port, *a, **kw)
        consultas.append(host)
        time.sleep(retraso)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in nombres[host]]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.setattr(_web, "_ip_is_public", lambda ip: True)
    return consultas


def _connect_lento(monkeypatch, retraso: float) -> None:
    """create_connection cuyo SYN-ACK tarda `retraso`: si el timeout no le da,
    espera el timeout y lanza TimeoutError, como un connect de verdad."""
    real = socket.create_connection

    def lento(addr, timeout=None, *a, **kw):
        if timeout is not None and timeout < retraso:
            time.sleep(timeout)
            raise TimeoutError("timed out")
        time.sleep(retraso)
        return real(addr, timeout, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", lento)


@pytest.fixture
def servidor_mudo():
    """Puerto que acepta TCP (el kernel completa el handshake por el backlog)
    y no contesta nunca: ni ServerHello ni respuesta HTTP."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    yield srv.getsockname()[1]
    srv.close()


# ---------------------------------------------------------------------------
# 1. El plazo cubre DNS, connect y handshake TLS
# ---------------------------------------------------------------------------

def test_dns_que_se_come_el_plazo_no_llega_a_conectar(monkeypatch):
    """Antes se conectaba igual, con el timeout calculado antes del DNS."""
    _dns(monkeypatch, {"lento.example": ["93.184.216.34"]}, retraso=0.4)
    conexiones: list[tuple] = []

    def captura(addr, timeout=None, *a, **kw):
        conexiones.append(addr)
        raise OSError("no debía intentar conectar: el plazo ya se acabó")

    monkeypatch.setattr(socket, "create_connection", captura)
    with pytest.raises(TimeoutError):
        _web._http_get("http://lento.example/", timeout=0.2)
    assert conexiones == []


@pytest.mark.parametrize("esquema, dns_s, connect_s", [
    ("https", 0.8, 0.0),
    ("https", 0.0, 0.8),
    ("https", 0.5, 0.4),
    ("http", 0.6, 1.0),
], ids=["https-dns-lento", "https-connect-lento", "https-ambos", "http-ambos"])
def test_el_plazo_cubre_dns_connect_y_handshake(monkeypatch, servidor_mudo,
                                               esquema, dns_s, connect_s):
    """Servidor mudo, timeout=1.0. Antes: https 1.8-1.9 s (el handshake
    estrenaba el timeout calculado antes del DNS), http 1.6 s (el connect
    usaba ese mismo número viejo)."""
    _dns(monkeypatch, {"lento.example": ["127.0.0.1"]}, retraso=dns_s)
    _connect_lento(monkeypatch, connect_s)
    inicio = time.monotonic()
    with pytest.raises(OSError):
        _web._http_get(f"{esquema}://lento.example:{servidor_mudo}/", timeout=1.0)
    transcurrido = time.monotonic() - inicio
    assert transcurrido < 1.0 + _HOLGURA_S, f"{transcurrido:.2f} s con timeout=1.0"


# ---------------------------------------------------------------------------
# 2. Todas las IPs validadas, en orden, dentro del mismo plazo
# ---------------------------------------------------------------------------

class _Pagina(http.server.BaseHTTPRequestHandler):
    """Contesta una página corta a cualquier GET y apunta el Host recibido."""

    hosts: list[str] = []
    cuerpo = "<html><body><p>Página de prueba de la unidad veinte.</p></body></html>"

    def do_GET(self):  # noqa: N802 — nombre que exige BaseHTTPRequestHandler
        type(self).hosts.append(self.headers.get("Host", ""))
        body = type(self).cuerpo.encode()
        if "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body)
            gz = True
        else:
            gz = False
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silencio en la salida de pytest
        pass


@pytest.fixture
def servidor_http(monkeypatch):
    monkeypatch.setattr(_Pagina, "hosts", [])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Pagina)
    hilo = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                            daemon=True)
    hilo.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()
    hilo.join(timeout=5)


def test_si_la_primera_ip_rechaza_se_prueba_la_siguiente(monkeypatch, servidor_http):
    """127.0.0.3 no escucha en ese puerto (RST); 127.0.0.1 sí. Antes:
    ConnectionRefusedError sin probar la segunda."""
    consultas = _dns(monkeypatch, {"dual.example": ["127.0.0.3", "127.0.0.1"]})
    content_type, body = _web._http_get(f"http://dual.example:{servidor_http}/", timeout=5.0)
    assert "text/html" in content_type
    assert "unidad veinte" in body
    assert consultas == ["dual.example"]  # una sola resolución: pinning intacto


def test_ip_en_agujero_negro_no_se_come_el_plazo_de_la_siguiente(monkeypatch, servidor_http):
    """La primera IP no contesta nunca (el SYN se pierde): con el plazo entero
    para ella, la segunda no llegaba a probarse. Se reparte lo que queda."""
    _dns(monkeypatch, {"dual.example": ["127.0.0.3", "127.0.0.1"]})
    real = socket.create_connection
    intentos: list[tuple[str, float]] = []

    def agujero_negro(addr, timeout=None, *a, **kw):
        intentos.append((addr[0], timeout))
        if addr[0] == "127.0.0.3":
            time.sleep(timeout)
            raise TimeoutError("timed out")
        return real(addr, timeout, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", agujero_negro)
    inicio = time.monotonic()
    _content_type, body = _web._http_get(f"http://dual.example:{servidor_http}/", timeout=2.0)
    transcurrido = time.monotonic() - inicio
    assert "unidad veinte" in body
    assert [ip for ip, _t in intentos] == ["127.0.0.3", "127.0.0.1"]
    assert intentos[0][1] <= 1.0  # la mitad de lo que quedaba, no los 2 s
    assert transcurrido < 2.0


def test_se_prueban_todas_las_ips_validadas_y_ninguna_otra(monkeypatch):
    """Sin red: cada IP de la ÚNICA resolución, en su orden, una vez cada una
    (getaddrinfo repite cada IP por tipo de socket), y nada más."""
    resolucion = [(2, 1, 6, "", ("93.184.216.34", 0)),
                  (2, 2, 17, "", ("93.184.216.34", 0)),
                  (2, 3, 0, "", ("93.184.216.34", 0)),
                  (10, 1, 6, "", ("2606:2800:220:1::1", 0, 0, 0))]
    consultas: list[str] = []

    def dns(host, *a, **kw):
        consultas.append(host)
        return resolucion

    monkeypatch.setattr(_web.socket, "getaddrinfo", dns)
    conexiones: list[tuple] = []

    def captura(addr, timeout=None, *a, **kw):
        conexiones.append(addr)
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(_web.socket, "create_connection", captura)
    with pytest.raises(ConnectionRefusedError):
        _web._http_get("http://doble.example/", timeout=2.0)
    assert conexiones == [("93.184.216.34", 80), ("2606:2800:220:1::1", 80)]
    assert consultas == ["doble.example"]


# ---------------------------------------------------------------------------
# 3. Handshake TLS de verdad por la conexión pinneada
# ---------------------------------------------------------------------------

_HOST_TLS = "ejemplo.test"


@pytest.fixture(scope="module")
def certificado(tmp_path_factory):
    """Certificado autofirmado para ejemplo.test, hecho al vuelo: ninguna clave
    en el repo. Sin openssl (no es el caso en Fedora ni en CI) se salta."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("sin openssl para generar el certificado de prueba")
    carpeta = tmp_path_factory.mktemp("tls_u20")
    cert, key = carpeta / "cert.pem", carpeta / "key.pem"
    try:
        subprocess.run(
            [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt",
             "ec_paramgen_curve:prime256v1", "-nodes", "-keyout", str(key),
             "-out", str(cert), "-days", "2", "-subj", f"/CN={_HOST_TLS}",
             "-addext", f"subjectAltName=DNS:{_HOST_TLS}"],
            check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"openssl no pudo generar el certificado: {error}")
    return cert, key


class _PaginaTLS(_Pagina):
    """/ contesta la página; /goteo manda las cabeceras y luego el cuerpo gota
    a gota sin acabar nunca (hasta que el cliente cuelga o el test termina)."""

    alto = threading.Event()

    def do_GET(self):  # noqa: N802
        if self.path != "/goteo":
            super().do_GET()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        try:
            while not type(self).alto.is_set():
                self.wfile.write(b"<p>gota</p>")
                time.sleep(0.05)
        except OSError:  # el cliente colgó al agotar su plazo: bien
            return


@pytest.fixture
def servidor_tls(certificado, monkeypatch):
    """(puerto, SNIs vistos) de un servidor https local con el certificado de
    ejemplo.test; el contexto TLS del módulo confía en él (y solo en él)."""
    cert, key = certificado
    monkeypatch.setattr(_PaginaTLS, "hosts", [])
    monkeypatch.setattr(_PaginaTLS, "alto", threading.Event())
    snis: list[str | None] = []
    contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    contexto.load_cert_chain(cert, key)
    contexto.sni_callback = lambda _sslobj, nombre, _ctx: snis.append(nombre)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PaginaTLS)
    srv.socket = contexto.wrap_socket(srv.socket, server_side=True)
    hilo = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                            daemon=True)
    hilo.start()
    monkeypatch.setattr(_web, "_SSL_CONTEXT", ssl.create_default_context(cafile=str(cert)))
    _dns(monkeypatch, {_HOST_TLS: ["127.0.0.1"], "otro.test": ["127.0.0.1"]})
    yield srv.server_address[1], snis
    _PaginaTLS.alto.set()
    srv.shutdown()
    srv.server_close()
    hilo.join(timeout=5)


def test_https_pinneado_hace_el_handshake_con_el_hostname(servidor_tls):
    """SNI y Host con el hostname (no la IP pinneada), cadena y hostname
    verificados, cuerpo gzip inflado entero."""
    puerto, snis = servidor_tls
    content_type, body = _web._http_get(f"https://{_HOST_TLS}:{puerto}/", timeout=5.0)
    assert "text/html" in content_type
    assert "Página de prueba de la unidad veinte." in body
    assert snis == [_HOST_TLS]
    assert _PaginaTLS.hosts == [f"{_HOST_TLS}:{puerto}"]


def test_https_con_otro_hostname_no_pasa_la_verificacion(servidor_tls):
    """Mismo servidor y misma IP, otro nombre: el certificado no es suyo."""
    puerto, _snis = servidor_tls
    with pytest.raises(ssl.SSLCertVerificationError, match="[Hh]ostname mismatch"):
        _web._http_get(f"https://otro.test:{puerto}/", timeout=5.0)
    assert _PaginaTLS.hosts == []


def test_https_con_certificado_no_confiable_se_rechaza(servidor_tls, monkeypatch):
    """Con las CAs del sistema el autofirmado no vale: el contexto de fábrica
    verifica la cadena."""
    puerto, _snis = servidor_tls
    monkeypatch.setattr(_web, "_SSL_CONTEXT", None)
    with pytest.raises(ssl.SSLCertVerificationError):
        _web._http_get(f"https://{_HOST_TLS}:{puerto}/", timeout=5.0)
    assert _PaginaTLS.hosts == []


def test_https_que_gotea_acaba_en_el_plazo(servidor_tls):
    """_DeadlineIO sobre un SSLSocket: el cuerpo que no acaba nunca se corta en
    el plazo y se devuelve lo que llegó."""
    puerto, _snis = servidor_tls
    inicio = time.monotonic()
    _content_type, body = _web._http_get(f"https://{_HOST_TLS}:{puerto}/goteo", timeout=1.0)
    transcurrido = time.monotonic() - inicio
    assert "<p>gota</p>" in body
    assert transcurrido < 1.0 + _HOLGURA_S, f"{transcurrido:.2f} s con timeout=1.0"
