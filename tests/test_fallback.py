"""Tests del respaldo local: si GLM se cae, Crotolamo sigue vivo con Ollama."""

from __future__ import annotations

import pytest

from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.llm import ChatResponse, LLMError


class _Fake:
    """Cliente de mentiras con el contrato de LLMClient."""

    def __init__(self, name: str, falla: bool = False, tokens: tuple[str, ...] = ()) -> None:
        self.model = name
        self.falla = falla
        self.tokens = tokens
        self.llamadas = 0

    def chat(self, messages, tools=None):
        self.llamadas += 1
        if self.falla:
            raise LLMError(f"{self.model} caído")
        return ChatResponse(content=f"soy {self.model}")

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        for t in self.tokens:
            if on_token:
                on_token(t)
        if self.falla:
            raise LLMError(f"{self.model} caído")
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
            raise LLMError("glm caído a media faena")

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
