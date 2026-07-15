"""Tests del respaldo local: si GLM se cae, Crotolamo sigue vivo con Ollama."""

from __future__ import annotations

import pytest

from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.glm import GLMAuthError
from crotolamo.core.llm import ChatResponse, LLMError, TransientLLMError


class _Fake:
    """Cliente de mentiras con el contrato de LLMClient.

    `falla=True` modela un fallo de DISPONIBILIDAD (timeout, wifi caído, 5xx):
    lanza TransientLLMError, que es lo único que debe abrir el breaker. Para
    errores permanentes (auth, 400) se pasa `error=` explícito.
    """

    def __init__(self, name: str, falla: bool = False, tokens: tuple[str, ...] = (),
                 error: Exception | None = None) -> None:
        self.model = name
        self.falla = falla
        self.tokens = tokens
        self.error = error
        self.llamadas = 0

    def _boom(self):
        if self.error is not None:
            raise self.error
        if self.falla:
            raise TransientLLMError(f"{self.model} caído")

    def chat(self, messages, tools=None):
        self.llamadas += 1
        self._boom()
        return ChatResponse(content=f"soy {self.model}")

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        for t in self.tokens:
            if on_token:
                on_token(t)
        self._boom()
        return ChatResponse(content=f"soy {self.model}")


def test_usa_el_primario_cuando_funciona():
    nube, local = _Fake("glm"), _Fake("ollama")
    assert FallbackLLM(nube, local).chat([]).content == "soy glm"
    assert local.llamadas == 0


def test_cae_al_local_si_el_primario_revienta():
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    assert FallbackLLM(nube, local).chat([]).content == "soy ollama"


def test_circuit_breaker_no_reintenta_la_nube_durante_el_cooldown():
    """Sin esto, cada turno pagaría el timeout de la nube caída."""
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)

    for _ in range(4):
        assert llm.chat([]).content == "soy ollama"

    assert nube.llamadas == 1  # solo el primer intento
    assert local.llamadas == 4


def test_tras_el_cooldown_reintenta_la_nube(monkeypatch):
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)

    reloj = {"t": 1000.0}
    monkeypatch.setattr("crotolamo.core.fallback.time.monotonic", lambda: reloj["t"])

    assert llm.chat([]).content == "soy ollama"   # trip
    assert nube.llamadas == 1

    reloj["t"] += 61                              # pasó el cooldown
    nube.falla = False                            # volvió el wifi
    assert llm.chat([]).content == "soy glm"
    assert nube.llamadas == 2


def test_recuperarse_limpia_el_breaker(monkeypatch):
    nube, local = _Fake("glm"), _Fake("ollama")
    llm = FallbackLLM(nube, local)
    llm.chat([])
    assert llm.active is nube


def test_streaming_cae_al_local_si_falla_ANTES_de_emitir():
    nube = _Fake("glm", falla=True, tokens=())      # revienta sin hablar
    local = _Fake("ollama", tokens=("hola ", "patrón"))
    dichos: list[str] = []

    r = FallbackLLM(nube, local).chat_stream([], on_token=dichos.append)
    assert r.content == "soy ollama"
    assert dichos == ["hola ", "patrón"]


def test_streaming_NO_reintenta_si_ya_habló():
    """Reintentar repetiría el principio de la frase por el altavoz."""
    nube = _Fake("glm", falla=True, tokens=("Órale ", "patrón"))
    local = _Fake("ollama", tokens=("hola",))
    dichos: list[str] = []

    with pytest.raises(LLMError):
        FallbackLLM(nube, local).chat_stream([], on_token=dichos.append)

    assert dichos == ["Órale ", "patrón"]  # no se repitió
    assert local.llamadas == 0


def test_model_refleja_el_cliente_activo():
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)
    assert llm.model == "glm"
    llm.chat([])                # trip
    assert llm.model == "ollama"


def test_si_ambos_caen_propaga_el_error_del_local():
    nube, local = _Fake("glm", falla=True), _Fake("ollama", falla=True)
    with pytest.raises(LLMError, match="ollama"):
        FallbackLLM(nube, local).chat([])


def test_toolagent_completa_el_turno_si_el_primario_cae_a_media_iteracion(monkeypatch):
    """Integración ToolAgent + FallbackLLM: la nube pide una tool y REVIENTA en la
    2ª iteración del loop; el respaldo local debe redactar la respuesta final."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry, desktop

    monkeypatch.setattr(desktop, "run_detached", lambda args: None)

    class _Primario:
        model = "glm"

        def __init__(self):
            self.llamadas = 0

        def chat(self, messages, tools=None):
            self.llamadas += 1
            if self.llamadas == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[{"name": "open_url", "arguments": {"url": "x.com"}}],
                    raw_message={"tool_calls": [{
                        "function": {"name": "open_url", "arguments": {"url": "x.com"}},
                    }]},
                )
            raise TransientLLMError("glm caído a media faena")

    class _Local:
        model = "ollama"

        def chat(self, messages, tools=None):
            return ChatResponse(content="Abierta desde el local, patrón.")

    nube = _Primario()
    llm = FallbackLLM(nube, _Local())
    agent = ToolAgent(
        llm, Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), fastpath=False,
    )

    reply = agent.handle_turn("abre x.com")
    assert reply == "Abierta desde el local, patrón."
    assert nube.llamadas == 2  # sirvió la 1ª iteración y reventó en la 2ª


def test_build_llm_envuelve_en_fallback_cuando_hay_key(monkeypatch):
    from crotolamo.core.engine import build_llm
    from crotolamo.core.glm import GLMClient
    from crotolamo.core.llm import LLMClient
    from crotolamo.settings import get_settings

    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")

    llm = build_llm(get_settings())
    assert isinstance(llm, FallbackLLM)
    assert isinstance(llm.primary, GLMClient)
    assert isinstance(llm.secondary, LLMClient)


# --- T6: el breaker SOLO se abre con errores transitorios ---

def test_error_permanente_se_propaga_sin_abrir_el_breaker():
    """Un 400 (payload mal formado) es un BUG, no una caída: degradar 60s a
    Ollama en silencio lo escondería. Se propaga y la nube se reintenta al
    siguiente turno."""
    nube = _Fake("glm", error=LLMError("GLM respondió 400, patrón: payload roto"))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)

    with pytest.raises(LLMError, match="400"):
        llm.chat([])
    assert local.llamadas == 0        # nada de degradar en silencio
    assert llm.active is nube          # el breaker NO se abrió

    with pytest.raises(LLMError, match="400"):
        llm.chat([])
    assert nube.llamadas == 2          # se volvió a intentar (sin cooldown)


def test_auth_error_cae_al_local_sin_reintentar_la_nube():
    """Una key inválida no se arregla sola: reintentarla cada turno es ruido.
    Se marca la nube como caída hasta reinicio y se usa el local."""
    nube = _Fake("glm", error=GLMAuthError("GLM me rechazó la credencial (401), patrón."))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=0.0)  # sin cooldown: probaría la nube SIEMPRE

    for _ in range(3):
        assert llm.chat([]).content == "soy ollama"

    assert nube.llamadas == 1          # un solo intento, no un bucle de 401s
    assert llm.model == "ollama"


def test_streaming_error_permanente_tambien_se_propaga():
    nube = _Fake("glm", error=LLMError("GLM respondió 400, patrón."))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local)
    with pytest.raises(LLMError, match="400"):
        llm.chat_stream([])
    assert local.llamadas == 0
    assert llm.active is nube


# --- T6: clasificación en los clientes ---

def test_clientes_clasifican_transitorio_vs_permanente(monkeypatch):
    from tests.test_transport import _Resp, _glm, _ollama

    # GLM: 500 y timeout son transitorios; 400 es permanente; 401 es auth.
    client, _ = _glm(monkeypatch, [_Resp(status=500, body=b"boom")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client, _ = _glm(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client, _ = _glm(monkeypatch, [_Resp(status=400, body=b"bad request")])
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)

    client, _ = _glm(monkeypatch, [_Resp(status=401, body=b"bad key")])
    with pytest.raises(GLMAuthError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)

    # Ollama: 500/conexión transitorios; 404 (modelo inexistente) permanente.
    client = _ollama([_Resp(status=500, body=b"panic")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client = _ollama([ConnectionRefusedError("refused")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client = _ollama([_Resp(status=404, body=b"model not found")])
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)
