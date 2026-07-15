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


# --- HTTPTransport bajo concurrencia (T1): conexión por hilo ---
# En el modo de voz concurrente, WarmLLM y BrainThread pueden llamar chat()
# sobre el MISMO cliente a la vez. Con un socket compartido eso corrompe el
# estado (CannotSendRequest, respuestas cruzadas); con conexión thread-local,
# cada hilo tiene su keep-alive propio y no hay contención.

def test_transport_es_seguro_entre_hilos(monkeypatch):
    import threading

    t = HTTPTransport("http://localhost:11434", timeout=5)
    creadas: list[_FakeConn] = []
    fabrica_lock = threading.Lock()

    def fabrica():
        with fabrica_lock:
            conn = _FakeConn([_Resp(body=f"conn-{len(creadas)}".encode())])
            creadas.append(conn)
        return conn

    monkeypatch.setattr(t, "_connect", fabrica)

    n = 8
    barrera = threading.Barrier(n)
    cuerpos: list[bytes | None] = [None] * n
    errores: list[Exception] = []

    def worker(i: int) -> None:
        try:
            barrera.wait(timeout=10)  # maximizar el solape: todos a la vez
            resp = t.post_json("/api/chat", {"hilo": i})
            cuerpos[i] = resp.read()
        except Exception as error:  # noqa: BLE001 — el assert de abajo lo reporta
            errores.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    assert errores == []
    # Cada hilo recibió UNA respuesta completa y distinta (nada cruzado).
    assert sorted(cuerpos) == sorted(f"conn-{i}".encode() for i in range(n))
    # Cada conexión atendió exactamente una petición: nadie compartió socket.
    assert len(creadas) == n
    assert all(len(conn.requests) == 1 for conn in creadas)


def test_transport_close_solo_afecta_al_hilo_actual(monkeypatch):
    """close() en un hilo no debe tirar la conexión keep-alive de otro."""
    import threading

    t = HTTPTransport("http://localhost:11434", timeout=5)
    conns = []

    def fabrica():
        conn = _FakeConn([_Resp(body=b"{}"), _Resp(body=b"{}")])
        conns.append(conn)
        return conn

    monkeypatch.setattr(t, "_connect", fabrica)

    t.post_json("/api/chat", {}).read()  # hilo principal: abre y drena

    def otro_hilo():
        t.post_json("/api/chat", {}).read()
        t.close()  # cierra SU conexión, no la del principal

    th = threading.Thread(target=otro_hilo)
    th.start()
    th.join(timeout=10)

    assert len(conns) == 2
    assert conns[1].cerrada          # la del hilo secundario se cerró
    assert not conns[0].cerrada      # la del principal sigue viva
    t.post_json("/api/chat", {})     # y se reutiliza sin reconectar
    assert len(conns) == 2


# --- build_primary_llm: el warm-up no comparte socket ni breaker con el loop ---

def test_build_primary_llm_no_envuelve_en_fallback(monkeypatch):
    from crotolamo.core.engine import build_llm, build_primary_llm
    from crotolamo.core.fallback import FallbackLLM
    from crotolamo.settings import get_settings

    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    settings = get_settings()

    warm = build_primary_llm(settings)
    loop_llm = build_llm(settings)

    assert not isinstance(warm, FallbackLLM)   # sin breaker que envenenar
    assert isinstance(loop_llm, FallbackLLM)
    # Instancia y transporte PROPIOS: calentar no toca el socket del loop.
    assert warm is not loop_llm.primary
    assert warm._transport is not loop_llm.primary._transport


def test_build_primary_llm_cae_a_ollama_sin_key(monkeypatch):
    from crotolamo.core.engine import build_primary_llm
    from crotolamo.core.llm import LLMClient
    from crotolamo.settings import get_settings

    for env in ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")

    assert isinstance(build_primary_llm(get_settings()), LLMClient)
