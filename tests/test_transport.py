"""Tests del transporte HTTP keep-alive y del mapeo de errores HTTP por cliente.

Ni un byte de red real: se inyecta un transporte falso en los clientes
(`client._transport`) o conexiones falsas en `HTTPTransport._connect`.
"""

from __future__ import annotations

import http.client
import json

import pytest

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.glm import GLMAuthError, GLMClient
from crotolamo.core.llm import LLMClient, LLMError


# --- dobles de prueba ---

class _Resp:
    """Respuesta HTTP de mentiras con la superficie que usan los clientes."""

    def __init__(self, status=200, body=b"", headers=None, lines=None):
        self.status = status
        self._body = body
        self._headers = headers or {}
        self._lines = lines or []
        self._leido = False

    def read(self):
        self._leido = True
        body, self._body = self._body, b""
        return body

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def isclosed(self):
        return self._leido

    def __iter__(self):
        return iter(self._lines)


class _FakeTransport:
    """Sustituto de HTTPTransport: devuelve respuestas en secuencia."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post_json(self, path, payload, headers=None):
        self.posts.append((path, payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeConn:
    """Conexión http.client de mentiras: request/getresponse en secuencia."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.cerrada = False

    def request(self, method, url, body=None, headers=None):
        self.requests.append((method, url, headers))

    def getresponse(self):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.cerrada = True


def _glm(monkeypatch, responses) -> tuple[GLMClient, _FakeTransport]:
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    client = GLMClient()
    transport = _FakeTransport(responses)
    client._transport = transport
    return client, transport


MSGS = [{"role": "user", "content": "hola"}]


# --- HTTPTransport: keep-alive y reconexión ---

def test_transport_reutiliza_la_conexion(monkeypatch):
    t = HTTPTransport("http://localhost:11434", timeout=5)
    conn = _FakeConn([_Resp(body=b"{}"), _Resp(body=b"{}")])
    monkeypatch.setattr(t, "_connect", lambda: conn)

    r1 = t.post_json("/api/chat", {"a": 1})
    r1.read()  # drenar deja la conexión reutilizable
    t.post_json("/api/chat", {"a": 2})

    assert len(conn.requests) == 2  # mismas dos peticiones por el MISMO socket
    assert not conn.cerrada


def test_transport_reconecta_si_el_servidor_cerro_el_keepalive(monkeypatch):
    """Un BadStatusLine (socket cerrado remoto) provoca UN reintento transparente."""
    t = HTTPTransport("https://api.z.ai/api/paas/v4")
    rota = _FakeConn([http.client.BadStatusLine("")])
    sana = _FakeConn([_Resp(body=b"ok")])
    conns = iter([rota, sana])
    monkeypatch.setattr(t, "_connect", lambda: next(conns))

    resp = t.post_json("/chat/completions", {})
    assert resp.read() == b"ok"
    assert rota.cerrada  # la conexión muerta se tiró


def test_transport_no_reintenta_dos_veces(monkeypatch):
    t = HTTPTransport("http://localhost:11434")
    conns = iter([_FakeConn([ConnectionResetError("reset")]),
                  _FakeConn([ConnectionResetError("reset")])])
    monkeypatch.setattr(t, "_connect", lambda: next(conns))

    with pytest.raises(ConnectionResetError):
        t.post_json("/api/chat", {})


def test_transport_antepone_el_base_path(monkeypatch):
    t = HTTPTransport("https://api.z.ai/api/paas/v4/")
    conn = _FakeConn([_Resp()])
    monkeypatch.setattr(t, "_connect", lambda: conn)
    t.post_json("/chat/completions", {})
    assert conn.requests[0][1] == "/api/paas/v4/chat/completions"


def test_transport_descarta_conexion_con_stream_a_medias(monkeypatch):
    """Si la respuesta anterior no se drenó (streaming cortado), no se reutiliza."""
    t = HTTPTransport("http://localhost:11434")
    sucia = _FakeConn([_Resp(body=b"pendiente")])
    limpia = _FakeConn([_Resp()])
    conns = iter([sucia, limpia])
    monkeypatch.setattr(t, "_connect", lambda: next(conns))

    t.post_json("/api/chat", {})   # NO se lee el cuerpo
    t.post_json("/api/chat", {})   # debe abrir conexión nueva
    assert sucia.cerrada
    assert len(limpia.requests) == 1


# --- GLMClient: mapeo de status HTTP a errores en personaje ---

def test_glm_401_es_auth_error(monkeypatch):
    client, _ = _glm(monkeypatch, [_Resp(status=401, body=b"bad key")])
    with pytest.raises(GLMAuthError, match=r"rechazó la credencial \(401\)"):
        client.chat(MSGS)


def test_glm_500_es_llm_error_con_codigo(monkeypatch):
    client, _ = _glm(monkeypatch, [_Resp(status=500, body=b"boom")])
    with pytest.raises(LLMError, match="GLM respondió 500"):
        client.chat(MSGS)


def test_glm_429_sin_retry_after_lanza(monkeypatch):
    client, transport = _glm(monkeypatch, [_Resp(status=429)])
    with pytest.raises(LLMError, match="limitando el ritmo"):
        client.chat(MSGS)
    assert len(transport.posts) == 1  # sin header no hay reintento


def test_glm_429_con_retry_after_corto_reintenta_y_triunfa(monkeypatch):
    """Un 429 transitorio (Retry-After <= 3s) se espera y reintenta in-place,
    en vez de tumbar el turno y degradar 60s al modelo local vía FallbackLLM."""
    ok = _Resp(body=json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "hola patrón"}}],
    }).encode())
    client, transport = _glm(
        monkeypatch,
        [_Resp(status=429, headers={"Retry-After": "1"}), ok],
    )
    esperas: list[float] = []
    monkeypatch.setattr("crotolamo.core.glm.time.sleep", esperas.append)

    assert client.chat(MSGS).content == "hola patrón"
    assert len(transport.posts) == 2
    assert esperas == [1.0]


def test_glm_429_con_retry_after_largo_no_espera(monkeypatch):
    client, transport = _glm(
        monkeypatch, [_Resp(status=429, headers={"Retry-After": "120"})])
    dormido = []
    monkeypatch.setattr("crotolamo.core.glm.time.sleep", dormido.append)
    with pytest.raises(LLMError, match="limitando el ritmo"):
        client.chat(MSGS)
    assert dormido == []
    assert len(transport.posts) == 1


def test_glm_429_solo_reintenta_una_vez(monkeypatch):
    client, transport = _glm(monkeypatch, [
        _Resp(status=429, headers={"Retry-After": "0"}),
        _Resp(status=429, headers={"Retry-After": "0"}),
    ])
    monkeypatch.setattr("crotolamo.core.glm.time.sleep", lambda s: None)
    with pytest.raises(LLMError, match="limitando el ritmo"):
        client.chat(MSGS)
    assert len(transport.posts) == 2


def test_glm_error_de_red_pregunta_por_internet(monkeypatch):
    client, _ = _glm(monkeypatch, [ConnectionRefusedError("no route")])
    with pytest.raises(LLMError, match="Hay internet"):
        client.chat(MSGS)


def test_glm_timeout_en_personaje(monkeypatch):
    client, _ = _glm(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(LLMError, match="no respondió a tiempo"):
        client.chat(MSGS)


# --- LLMClient (Ollama): un 404/500 ya no se confunde con servicio caído ---

def _ollama(responses) -> LLMClient:
    client = LLMClient(model="qwen2.5-coder:3b")
    client._transport = _FakeTransport(responses)
    return client


def test_ollama_404_habla_del_modelo_no_del_servicio():
    client = _ollama([_Resp(status=404, body=b"model not found")])
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS)
    assert "qwen2.5-coder:3b" in str(exc.value)
    assert "ollama pull" in str(exc.value)
    # El servicio SÍ está vivo: el mensaje viejo despistaba.
    assert "vivo el servicio" not in str(exc.value)


def test_ollama_500_reporta_error_interno():
    client = _ollama([_Resp(status=500, body=b"panic")])
    with pytest.raises(LLMError, match=r"tropezó por dentro \(500\)"):
        client.chat(MSGS)


def test_ollama_conexion_rechazada_pregunta_si_esta_vivo():
    client = _ollama([ConnectionRefusedError("refused")])
    with pytest.raises(LLMError, match="vivo el servicio"):
        client.chat(MSGS)
