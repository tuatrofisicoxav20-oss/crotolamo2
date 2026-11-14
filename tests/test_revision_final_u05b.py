"""Revisión final, unidad u05b: el contexto TLS del transporte de los LLM.

Cada conexión https nueva de `HTTPTransport` (la primera de cada hilo, la que
releva a un keep-alive que la nube cerró entre turnos de voz, la de un stream
ocioso más de 15 s, cada reintento y cada preconnect) creaba su propio
SSLContext: http.client, sin `context=`, construye uno por defecto y vuelve a
cargar el bundle de CAs del sistema (~20 ms de CPU) en el hilo del turno, antes
incluso del handshake. Ahora hay uno por transporte, hecho con la receta de
http.client (mismas CAs, misma verificación, ALPN http/1.1, PHA).
"""

from __future__ import annotations

import http.client
import shutil
import ssl
import subprocess
import threading

import pytest

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.glm import GLMClient
from tests.test_perf_llm_turn_is_one_atomic_blocking_call import _Server

MSGS = [{"role": "user", "content": "hola"}]
URL_NUBE = "https://api.z.ai/api/paas/v4"


def _contando_contextos(monkeypatch) -> list[ssl.SSLContext]:
    """Anota cada contexto TLS por defecto que se crea (el gancho de PEP 476 que
    usa http.client cuando no le pasan `context=`)."""
    creados: list[ssl.SSLContext] = []
    real = ssl._create_default_https_context

    def contando(*args, **kwargs):
        ctx = real(*args, **kwargs)
        creados.append(ctx)
        return ctx

    monkeypatch.setattr(ssl, "_create_default_https_context", contando)
    return creados


def test_https_las_conexiones_nuevas_comparten_un_solo_contexto(monkeypatch):
    """Lo del hallazgo: tres conexiones nuevas (y una desde otro hilo, como la del
    preconnect) cargan el bundle de CAs UNA vez, no una por conexión."""
    creados = _contando_contextos(monkeypatch)
    t = HTTPTransport(URL_NUBE, timeout=60)
    assert creados == []          # construir el cliente no lo paga: es perezoso

    conns = [t._connect() for _ in range(3)]
    otro: list[http.client.HTTPConnection] = []
    hilo = threading.Thread(target=lambda: otro.append(t._connect()), name="Preconnect")
    hilo.start()
    hilo.join(timeout=5)
    conns += otro

    assert len(conns) == 4
    assert len({id(c._context) for c in conns}) == 1
    assert len(creados) == 1
    assert conns[0]._context is creados[0]


def test_el_contexto_es_por_transporte(monkeypatch):
    """Uno por transporte, no global: un cliente creado después de cambiar las CAs
    (SSL_CERT_FILE, lo que hacen los tests TLS) usa las suyas."""
    creados = _contando_contextos(monkeypatch)
    a = HTTPTransport(URL_NUBE, timeout=60)._connect()
    b = HTTPTransport(URL_NUBE, timeout=60)._connect()
    assert a._context is not b._context
    assert len(creados) == 2


def test_http_no_crea_contexto_tls(monkeypatch):
    """Ollama en loopback (http://): ni contexto ni bundle de CAs."""
    creados = _contando_contextos(monkeypatch)
    t = HTTPTransport("http://127.0.0.1:11434", timeout=180)
    for _ in range(2):
        conn = t._connect()
        assert not isinstance(conn, http.client.HTTPSConnection)
    assert creados == []


def test_el_contexto_compartido_es_el_de_http_client():
    """Misma receta que http.client: verificación de certificado y de nombre,
    mismas opciones y versiones, y PHA donde OpenSSL lo admite. Compartirlo no
    puede relajar nada."""
    nuestro = HTTPTransport(URL_NUBE, timeout=60)._connect()._context
    suyo = http.client.HTTPSConnection("api.z.ai")._context
    assert nuestro.verify_mode == suyo.verify_mode == ssl.CERT_REQUIRED
    assert nuestro.check_hostname is suyo.check_hostname is True
    assert nuestro.options == suyo.options
    assert nuestro.minimum_version == suyo.minimum_version
    assert nuestro.maximum_version == suyo.maximum_version
    assert nuestro.post_handshake_auth == suyo.post_handshake_auth


# --- Con sockets de verdad: TLS 1.3 de loopback ----------------------------------

@pytest.fixture(scope="module")
def certificado(tmp_path_factory):
    """Certificado autofirmado para 127.0.0.1, hecho al vuelo: ninguna clave en el
    repo. Sin openssl se salta."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("sin openssl para generar el certificado de prueba")
    carpeta = tmp_path_factory.mktemp("tls_u05b")
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
def servidor_tls(certificado):
    cert, key = certificado
    contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    contexto.load_cert_chain(cert, key)
    contexto.set_alpn_protocols(["http/1.1"])   # para ver que el cliente lo ofrece
    srv = _Server("entero", contexto=contexto)
    yield srv
    srv.parar()


def test_tls_real_conexiones_nuevas_con_un_contexto(servidor_tls, certificado,
                                                    monkeypatch):
    """El caso del hallazgo con sockets de verdad: la nube cerró el keep-alive
    ocioso entre turnos (aquí, close()), así que cada turno abre conexión nueva.
    Todas verifican el certificado, negocian ALPN http/1.1 y salen del MISMO
    contexto: el bundle de CAs se carga una vez, no en cada turno."""
    monkeypatch.setenv("SSL_CERT_FILE", str(certificado[0]))
    creados = _contando_contextos(monkeypatch)
    client = GLMClient(api_key="k", base_url=servidor_tls.url + "/api/paas/v4", timeout=5)
    transport = client._transport
    for _ in range(3):
        assert client.chat(MSGS).content
        sock = transport._conn.sock
        assert isinstance(sock, ssl.SSLSocket)
        assert sock.selected_alpn_protocol() == "http/1.1"
        assert sock.context is creados[0]
        transport.close()          # el servidor cerró el keep-alive ocioso
    assert servidor_tls.conexiones == 3
    assert len(creados) == 1


def test_tls_real_sin_confiar_en_el_certificado_falla(servidor_tls, monkeypatch):
    """Compartir el contexto no apaga la verificación: un certificado que nadie
    firmó sigue tumbando la conexión."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    t = HTTPTransport(servidor_tls.url + "/api/paas/v4", timeout=5)
    with pytest.raises(ssl.SSLCertVerificationError):
        t._connect().connect()
