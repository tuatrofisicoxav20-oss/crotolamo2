"""Tests del transporte HTTP keep-alive y del mapeo de errores HTTP por cliente.

Ni un byte de red real: se inyecta un transporte falso en los clientes
(`client._transport`) o conexiones falsas en `HTTPTransport._connect`.
"""

from __future__ import annotations

import http.client
import io
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


# --- HTTPTransport: fin LÓGICO avisado (finish) -> keep-alive reutilizable ---
# Los parsers de stream se detienen en [DONE]/done:true, antes del chunk
# terminal del cuerpo: sin aviso, la respuesta "a medias" tiraba el socket y
# cada chat_stream pagaba TCP+TLS nuevos.

class _StreamResp(_Resp):
    """Stream cuyo parser ya vio el fin lógico: solo queda la cola del cuerpo."""

    def __init__(self, cola=b"\r\n0\r\n\r\n", will_close=False, falla=None, sobra=False):
        super().__init__(body=cola)
        self.will_close = will_close
        self.falla = falla      # excepción que lanza read() (p.ej. el timeout)
        self.sobra = sobra      # True = la cola es más larga que el tope de drenaje
        self.lecturas: list[int | None] = []

    def read(self, amt=None):
        self.lecturas.append(amt)
        if self.falla is not None:
            raise self.falla
        if not self.sobra:
            self._leido = True
        return b"x" * (amt or 0) if self.sobra else super().read()

    # El drenaje lee con read1 (una lectura del socket cada vez, para poder
    # mirar el reloj entre una y otra); aquí da igual cuál se llame.
    read1 = read


class _FakeSock:
    def __init__(self, timeout=60.0):
        self.timeout = timeout
        self.historial: list[float | None] = []

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.historial.append(value)
        self.timeout = value


def _transport_contando(monkeypatch, *conns):
    t = HTTPTransport("http://localhost:11434", timeout=5)
    pendientes = list(conns)
    abiertas: list[_FakeConn] = []

    def fabrica():
        conn = pendientes.pop(0)
        abiertas.append(conn)
        return conn

    monkeypatch.setattr(t, "_connect", fabrica)
    return t, abiertas


def test_transport_reutiliza_tras_finish(monkeypatch):
    stream = _StreamResp()
    conn = _FakeConn([stream, _Resp(body=b"{}")])
    t, abiertas = _transport_contando(monkeypatch, conn)

    resp = t.post_json("/api/chat", {"stream": True})
    t.finish(resp)
    # finish() solo ANOTA: el turno en curso no paga ni una lectura.
    assert stream.lecturas == []
    t.post_json("/api/chat", {})

    assert abiertas == [conn]               # una sola conexión para las dos peticiones
    assert len(conn.requests) == 2
    assert not conn.cerrada
    assert stream.lecturas == [HTTPTransport._DRAIN_BYTES]  # la cola se leyó, acotada


def test_transport_sin_finish_sigue_descartando(monkeypatch):
    """Sin el aviso del consumidor, una respuesta a medias se tira como siempre."""
    stream = _StreamResp()
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    t.post_json("/api/chat", {})
    t.post_json("/api/chat", {})

    assert abiertas == [sucia, limpia]
    assert sucia.cerrada
    assert stream.lecturas == []            # ni se intentó drenar


def test_transport_drenaje_fallido_descarta(monkeypatch):
    """La cola no llegó dentro del tope (el chunk terminal se retrasa): se
    descarta como hoy y la petición sale por una conexión nueva."""
    stream = _StreamResp(falla=TimeoutError("timed out"))
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    t.finish(t.post_json("/api/chat", {}))
    t.post_json("/api/chat", {})

    assert abiertas == [sucia, limpia]
    assert sucia.cerrada
    assert len(limpia.requests) == 1


def test_transport_cola_mas_larga_que_el_tope_descarta(monkeypatch):
    stream = _StreamResp(sobra=True)
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    t.finish(t.post_json("/api/chat", {}))
    t.post_json("/api/chat", {})

    assert stream.lecturas == [HTTPTransport._DRAIN_BYTES]  # nunca más que el tope
    assert abiertas == [sucia, limpia]
    assert sucia.cerrada


def test_transport_will_close_no_drena(monkeypatch):
    """`Connection: close`: el servidor no va a reutilizar ese socket; drenar
    sería esperar para nada."""
    stream = _StreamResp(will_close=True)
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    t.finish(t.post_json("/api/chat", {}))
    t.post_json("/api/chat", {})

    assert stream.lecturas == []
    assert abiertas == [sucia, limpia]
    assert sucia.cerrada


def _reloj_fijo(monkeypatch, t=1000.0):
    """Reloj de mentiras del transporte (drenaje y edad del keep-alive)."""
    from crotolamo.core import engine

    reloj = {"t": t}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    return reloj


def test_transport_drenaje_acotado_en_tiempo_y_restaura_el_timeout(monkeypatch):
    _reloj_fijo(monkeypatch, t=0.0)   # el tiempo no pasa: el tope entero, exacto
    stream = _StreamResp()
    conn = _FakeConn([stream, _Resp()])
    conn.sock = _FakeSock(timeout=60.0)
    t, _ = _transport_contando(monkeypatch, conn)

    t.finish(t.post_json("/api/chat", {}))
    t.post_json("/api/chat", {})

    # Espera como mucho _DRAIN_TIMEOUT_S y deja el timeout de lectura como estaba.
    assert conn.sock.historial == [HTTPTransport._DRAIN_TIMEOUT_S, 60.0]
    assert conn.sock.timeout == 60.0


def test_transport_drenaje_acotado_en_tiempo_total(monkeypatch):
    """Un servidor que gotea la cola más rápido que el timeout de cada lectura
    (y que nunca la termina): el tope de tiempo es del drenaje ENTERO. Antes,
    resp.read(64 KiB) seguía leyendo mientras llegaran bytes: 655 s de goteo."""
    reloj = _reloj_fijo(monkeypatch)

    class _Goteo(_StreamResp):
        def read1(self, amt=-1):
            self.lecturas.append(amt)
            reloj["t"] += 0.01      # un byte cada 10 ms
            return b"x"

        def read(self, amt=None):   # como http.client: junta lecturas hasta `amt`
            return b"".join(self.read1(amt) for _ in range(amt or 0))

    stream = _Goteo()
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    t.finish(t.post_json("/api/chat", {}))
    inicio = reloj["t"]
    t.post_json("/api/chat", {})

    assert reloj["t"] - inicio <= HTTPTransport._DRAIN_TIMEOUT_S + 0.011
    assert abiertas == [sucia, limpia]
    assert sucia.cerrada


@pytest.mark.parametrize("quieta, reutiliza", [
    (HTTPTransport._REUSE_MAX_IDLE_S - 1, True),
    (HTTPTransport._REUSE_MAX_IDLE_S + 1, False),
])
def test_transport_no_reutiliza_un_stream_terminado_hace_rato(monkeypatch, quieta, reutiliza):
    """Un keep-alive muerto en silencio (NAT, cambio de wifi) cuesta el timeout
    de lectura entero y abre el breaker. Antes del paso 1 los streams nunca
    reutilizaban (inmunes); ahora solo dentro de unos segundos: lo que separa
    las llamadas de un turno con tools. Pasado eso, conexión nueva como antes."""
    reloj = _reloj_fijo(monkeypatch)
    stream = _StreamResp()
    vieja, nueva = _FakeConn([stream, _Resp()]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, vieja, nueva)

    t.finish(t.post_json("/api/chat", {"stream": True}))
    reloj["t"] += quieta
    t.post_json("/api/chat", {})

    if reutiliza:
        assert (abiertas, vieja.cerrada) == ([vieja], False)
        assert stream.lecturas == [HTTPTransport._DRAIN_BYTES]
    else:
        assert (abiertas, vieja.cerrada) == ([vieja, nueva], True)
        assert stream.lecturas == []        # ni se intenta drenar


def test_el_reloj_del_keepalive_cuenta_la_suspension(monkeypatch):
    """Tras suspender el portátil, monotonic() no avanzó lo que duró la
    suspensión: un keep-alive de "hace 5 s" podía tener horas. En Linux se
    usa CLOCK_BOOTTIME, que sí la cuenta."""
    import time

    from crotolamo.core import engine

    if not hasattr(time, "CLOCK_BOOTTIME"):
        pytest.skip("CLOCK_BOOTTIME solo existe en Linux")
    pedidos: list[int] = []
    real = time.clock_gettime
    monkeypatch.setattr(time, "clock_gettime", lambda clk: pedidos.append(clk) or real(clk))

    engine._reloj()

    assert pedidos == [time.CLOCK_BOOTTIME]


def test_transport_finish_de_otra_respuesta_no_cuenta(monkeypatch):
    """Solo vale el aviso sobre la ÚLTIMA respuesta de ESTE hilo."""
    import threading

    stream = _StreamResp()
    sucia, limpia = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, sucia, limpia)

    resp = t.post_json("/api/chat", {})
    t.finish(_StreamResp())                         # otra respuesta cualquiera
    hilo = threading.Thread(target=t.finish, args=(resp,))
    hilo.start()
    hilo.join(timeout=10)                           # otro hilo no tiene esa respuesta
    t.post_json("/api/chat", {})

    assert stream.lecturas == []
    assert abiertas == [sucia, limpia]


def test_transport_close_olvida_el_finish(monkeypatch):
    stream = _StreamResp()
    primera, segunda = _FakeConn([stream]), _FakeConn([_Resp()])
    t, abiertas = _transport_contando(monkeypatch, primera, segunda)

    t.finish(t.post_json("/api/chat", {}))
    t.close()
    t.post_json("/api/chat", {})

    assert stream.lecturas == []
    assert abiertas == [primera, segunda]


# --- HTTPTransport: un fallo a media petición cuelga el socket ---

def test_transport_cierra_si_la_respuesta_no_llega_a_tiempo(monkeypatch):
    """Timeout esperando la respuesta (Ollama generando un chat no-stream): antes
    el socket quedaba abierto y el servidor seguía generando para nadie hasta
    la siguiente petición de ese hilo."""
    lenta = _FakeConn([TimeoutError("timed out")])
    t, _ = _transport_contando(monkeypatch, lenta)

    with pytest.raises(TimeoutError):
        t.post_json("/api/chat", {})

    assert lenta.cerrada
    assert t._conn is None


def test_transport_reintenta_un_keepalive_tls_muerto(monkeypatch):
    """Python 3.11/3.12: escribir en un keep-alive TLS que el servidor ya cerró
    lanza ssl.SSLEOFError (no ConnectionError). Es el mismo socket muerto que el
    reintento único ya cubre; sin él, el turno caía al modelo local 60 s."""
    import ssl

    muerta = _FakeConn([ssl.SSLEOFError("EOF occurred in violation of protocol")])
    sana = _FakeConn([_Resp(body=b"ok")])
    t, abiertas = _transport_contando(monkeypatch, muerta, sana)

    assert t.post_json("/chat/completions", {}).read() == b"ok"
    assert abiertas == [muerta, sana]
    assert muerta.cerrada


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


# --- Clientes: cierran el ciclo de la respuesta en streaming ---
# finish() al llegar el fin lógico (keep-alive reutilizable), close() si el
# stream se corta, falla o el consumidor aborta (el servidor deja de generar),
# y un aviso si el stream acaba limpio sin su marca de cierre.

class _Zocalo:
    """Socket de mentiras del que http.client lee una respuesta cruda."""

    def __init__(self, crudo: bytes) -> None:
        self._crudo = crudo

    def makefile(self, mode):
        return io.BufferedReader(io.BytesIO(self._crudo))


def _real(crudo: bytes) -> http.client.HTTPResponse:
    """HTTPResponse DE VERDAD: el framing chunked lo resuelve http.client."""
    resp = http.client.HTTPResponse(_Zocalo(crudo))  # type: ignore[arg-type]
    resp.begin()
    return resp


def _chunked(lineas: list[bytes], terminado: bool = True) -> http.client.HTTPResponse:
    chunks = b"".join(f"{len(linea):x}\r\n".encode() + linea + b"\r\n" for linea in lineas)
    return _real(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunks
        + (b"0\r\n\r\n" if terminado else b"")
    )


_HOLA = {
    "glm": b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n',
    "ollama": b'{"message":{"role":"assistant","content":"Hola"},"done":false}\n',
}
_FIN = {
    "glm": b"data: [DONE]\n\n",
    "ollama": b'{"message":{"role":"assistant","content":""},"done":true}\n',
}
_MOTORES = ["glm", "ollama"]


class _CicloTransport(_FakeTransport):
    """_FakeTransport que además cuenta los avisos de fin y los cuelgues."""

    def __init__(self, responses):
        super().__init__(responses)
        self.finished: list[object] = []
        self.closes = 0

    def finish(self, resp):
        self.finished.append(resp)

    def close(self):
        self.closes += 1


def _cliente(motor, monkeypatch, responses):
    if motor == "glm":
        monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
        client = GLMClient(base_url="http://localhost:1/v4")
    else:
        client = LLMClient(host="http://localhost:1")
    transport = _CicloTransport(responses)
    client._transport = transport
    return client, transport


@pytest.mark.parametrize("motor", _MOTORES)
def test_chat_stream_avisa_fin_logico(monkeypatch, motor):
    resp = _chunked([_HOLA[motor], _FIN[motor]])
    client, transport = _cliente(motor, monkeypatch, [resp])

    assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola"

    assert transport.finished == [resp]   # una vez, sobre ESA respuesta
    assert transport.closes == 0


@pytest.mark.parametrize("motor", _MOTORES)
def test_chat_stream_reutiliza_el_keepalive(monkeypatch, motor):
    """De punta a punta con HTTPTransport real: N streams = 1 conexión (antes N,
    cada una con su handshake TCP+TLS)."""
    n = 4
    conn = _FakeConn([_chunked([_HOLA[motor], _FIN[motor]]) for _ in range(n)])
    client = (
        GLMClient(api_key="k", base_url="http://localhost:1/v4") if motor == "glm"
        else LLMClient(host="http://localhost:1")
    )
    abiertas: list[_FakeConn] = []
    monkeypatch.setattr(client._transport, "_connect", lambda: abiertas.append(conn) or conn)

    for _ in range(n):
        assert client.chat_stream(MSGS, on_token=lambda t: None).content == "Hola"

    assert len(abiertas) == 1
    assert len(conn.requests) == n
    assert not conn.cerrada


@pytest.mark.parametrize("motor", _MOTORES)
def test_chat_stream_cuelga_si_el_consumidor_aborta(monkeypatch, motor):
    """on_token lanza (el patrón interrumpió): se cuelga YA, para que el motor
    deje de generar, y la excepción del consumidor sale tal cual."""
    client, transport = _cliente(motor, monkeypatch, [_chunked([_HOLA[motor], _FIN[motor]])])

    class _Aborto(Exception):
        pass

    def on_token(texto):
        raise _Aborto()

    with pytest.raises(_Aborto):
        client.chat_stream(MSGS, on_token=on_token)

    assert transport.closes == 1
    assert transport.finished == []


@pytest.mark.parametrize("motor", _MOTORES)
@pytest.mark.parametrize("fallo", [
    TimeoutError("timed out"), ConnectionResetError("reset"), http.client.IncompleteRead(b""),
], ids=["timeout", "reset", "incomplete"])
def test_chat_stream_cuelga_si_la_lectura_falla(monkeypatch, motor, fallo):
    class _Rota(_Resp):
        def __iter__(self):
            yield _HOLA[motor]
            raise fallo

    client, transport = _cliente(motor, monkeypatch, [_Rota()])

    with pytest.raises(LLMError) as info:
        client.chat_stream(MSGS, on_token=lambda t: None)

    assert type(info.value).__name__ == "TransientLLMError"   # mismo error de siempre
    assert transport.closes == 1
    assert transport.finished == []


@pytest.mark.parametrize("motor", _MOTORES)
def test_chat_stream_cuelga_si_el_motor_avisa_un_error_dentro(monkeypatch, motor):
    error = {
        "glm": b'data: {"error": {"message": "too many requests"}}\n\n',
        "ollama": b'{"error": "model runner has unexpectedly stopped"}\n',
    }[motor]
    client, transport = _cliente(motor, monkeypatch, [_chunked([error])])

    with pytest.raises(LLMError):
        client.chat_stream(MSGS)

    assert transport.closes == 1
    assert transport.finished == []


@pytest.mark.parametrize("motor", _MOTORES)
@pytest.mark.parametrize("terminado", [False, True], ids=["fin_entre_chunks", "chunked_cerrado"])
def test_stream_truncado_avisa_y_no_marca_finish(monkeypatch, caplog, motor, terminado):
    """Criterio híbrido: un stream que acaba LIMPIO sin su marca
    ([DONE]/finish_reason, done:true) pero con texto se acepta, avisando en el
    log de que puede venir truncado; esa conexión no se reutiliza."""
    import logging

    client, transport = _cliente(motor, monkeypatch, [_chunked([_HOLA[motor]], terminado)])
    dichos: list[str] = []

    logger = f"crotolamo.core.{'glm' if motor == 'glm' else 'llm'}"
    with caplog.at_level(logging.WARNING, logger=logger):
        respuesta = client.chat_stream(MSGS, on_token=dichos.append)

    assert respuesta.content == "Hola"
    assert dichos == ["Hola"]
    avisos = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    assert ("GLM" if motor == "glm" else "Ollama") in avisos[0]
    assert "truncada" in avisos[0]
    assert transport.finished == []      # el transporte NO se entera de un "fin"
    assert transport.closes == 1         # y ese socket no se reutiliza


@pytest.mark.parametrize("motor", _MOTORES)
@pytest.mark.parametrize("fallo", [
    TimeoutError("timed out"), http.client.IncompleteRead(b"{", 40),
], ids=["timeout", "incomplete"])
def test_cuerpo_cortado_es_transitorio_y_cuelga(monkeypatch, motor, fallo):
    """chat() no-stream: un cuerpo que no llega entero es disponibilidad
    (TransientLLMError, con el texto de siempre) y el socket se cuelga."""
    from crotolamo.core.llm import TransientLLMError

    class _Cortada(_Resp):
        def read(self):
            raise fallo

    client, transport = _cliente(motor, monkeypatch, [_Cortada()])
    texto = "no respondió a tiempo" if isinstance(fallo, TimeoutError) else "a medias, patrón"

    with pytest.raises(TransientLLMError, match=texto):
        client.chat(MSGS)

    assert transport.closes == 1


def test_cuerpo_cortado_lo_contesta_el_local(monkeypatch):
    from crotolamo.core.fallback import FallbackLLM
    from crotolamo.core.llm import ChatResponse

    class _Cortada(_Resp):
        def read(self):
            raise http.client.IncompleteRead(b"{", 40)

    class _Local:
        model = "local"
        llamadas = 0

        def chat(self, messages, tools=None):
            self.llamadas += 1
            return ChatResponse(content="soy el local")

    nube, _ = _cliente("glm", monkeypatch, [_Cortada()])
    local = _Local()

    assert FallbackLLM(nube, local).chat(MSGS).content == "soy el local"
    assert local.llamadas == 1
