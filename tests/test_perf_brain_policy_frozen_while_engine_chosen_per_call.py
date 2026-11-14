"""La política del cerebro (atajos + dieta de tools) sigue al motor que RESPONDE.

Con GLM de cerebro y Ollama de respaldo, qué motor contesta lo decide el breaker
de FallbackLLM en CADA llamada; los atajos y la dieta de tools se fijaban al armar
el agente (o una vez por turno, mirando `llm.degraded` ANTES de la llamada que
descubre la caída). Medido con el bench del diseño: 11 de 11 llamadas al modelo
local llevaban las 42 schemas (~5.000 tokens contra num_ctx 2048), y cada turno de
la caída volvía a sondear la nube dos veces.

- ToolAgent consulta una `BrainPolicy` por llamada: el fast-path con la del
  arranque del turno, las tools con la del motor que va a responder (solo se
  recalculan si cambió) y el short-circuit con la del motor que respondió.
- FallbackLLM adapta (`shape_for_secondary`) la petición al local cuando la nube
  cae A MEDIA llamada, y el cooldown cuenta desde que respondió el local.
- build_agent arma una política por motor; [llm.glm].fallback_shortcuts (false de
  fábrica) devuelve al local degradado sus propios atajos.
- HTTPTransport separa el tope de CONECTAR (5 s de fábrica, por dirección del DNS,
  TLS incluido) del de LEER (60 s nube / 180 s local, sin cambios): una nube
  inalcanzable costaba 60 s por dirección en cada sonda.
- Un keep-alive que el kernel ya dio por muerto (el servidor lo cerró, o el
  keepalive TCP vio la ruta caída) se tira ANTES de escribir: antes costaba el
  timeout de lectura entero (60 s) o una escritura fallida más el reintento.
"""

from __future__ import annotations

import json
import logging
import re
import select
import socket
import struct
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import agent as agent_mod
from crotolamo.core import engine
from crotolamo.core.agent import DEFAULT_DIRECT_TOOLS, ToolAgent
from crotolamo.core.engine import HTTPTransport
from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.glm import GLMAuthError, GLMClient
from crotolamo.core.llm import ChatResponse, LLMClient, LLMError, TransientLLMError
from crotolamo.core.memory import Conversation
from crotolamo.core.streaming import TurnCancelled
from crotolamo.safety.guard import Guard
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry, Tool

_RAIZ = Path(__file__).resolve().parents[1]


# --- dobles -----------------------------------------------------------------------

def _tool(name: str, func=None, direct: bool = False) -> Tool:
    return Tool(
        name=name, func=func or (lambda: f"{name} hecho, patrón."), description=name,
        direct=direct, parameters={"type": "object", "properties": {}, "required": []},
    )


def _registry(*tools: Tool) -> Registry:
    reg = Registry()
    for t in tools:
        reg.register(t)
    return reg


def _pide(name: str) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[{"name": name, "arguments": {}}],
        raw_message={"tool_calls": [{"function": {"name": name, "arguments": {}}}]},
    )


class _Motor:
    """Contrato LLMEngine que anota las tools de cada llamada (None = sin tools).

    `al_llamar` corre DENTRO de la llamada: es como se modela que el breaker
    cambió mientras el motor respondía (la nube cayó y contestó el local).
    """

    def __init__(self, name: str = "motor", guion=(), falla: bool = False,
                 error: Exception | None = None, al_llamar=None) -> None:
        self.model = name
        self.guion = list(guion)
        self.falla = falla
        self.error = error
        self.al_llamar = al_llamar
        self.tools: list[int | None] = []
        self.recibidas: list[object] = []

    def chat(self, messages, tools=None):
        self.tools.append(None if tools is None else len(tools))
        self.recibidas.append(tools)
        if self.al_llamar is not None:
            self.al_llamar()
        if self.error is not None:
            raise self.error
        if self.falla:
            raise TransientLLMError(f"{self.model} caído")
        return self.guion.pop(0) if self.guion else ChatResponse(content="Listo, patrón.")

    def chat_stream(self, messages, tools=None, on_token=None):
        resp = self.chat(messages, tools)
        if on_token is not None and resp.content and not resp.tool_calls:
            on_token(resp.content)
        return resp


def _politica(**kw):
    return agent_mod.BrainPolicy(**kw)


def _agente(llm, registry, **kw) -> ToolAgent:
    kw.setdefault("pre_hooks", [])
    kw.setdefault("post_hooks", [])
    return ToolAgent(llm, Conversation("SYS"), registry=registry,
                     guard=Guard(allowed_roots=[]), **kw)


# --- paso 1: ToolAgent evalúa la política en cada llamada ---------------------------

def test_politica_que_cambia_entre_llamadas_recalcula_las_tools():
    """La nube pide una tool y cae el breaker antes de la 2ª llamada: la 2ª lleva
    la dieta del motor que va a responder ahora, no la del arranque del turno."""
    reg = _registry(_tool("a"), _tool("b"), _tool("c"))
    estado = {"degradado": False}
    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset(),
                     honor_tool_direct=False)
    local = _politica(name="degraded", fastpath=False, direct_tools=frozenset(),
                      honor_tool_direct=False,
                      route=lambda texto: [reg.get("a").schema()])

    def _cae() -> None:
        estado["degradado"] = True

    llm = _Motor(guion=[_pide("a"), ChatResponse(content="Ya, patrón.")], al_llamar=_cae)
    agent = _agente(llm, reg, policy=lambda: local if estado["degradado"] else nube)

    assert agent.handle_turn("haz a") == "Ya, patrón."
    assert llm.tools == [3, 1]


def test_politica_que_no_cambia_enruta_una_sola_vez_por_turno():
    """Mientras el motor no cambia, el set de tools queda fijo en todo el turno
    (cache de prefijo): el router corre UNA vez aunque haya tres llamadas."""
    reg = _registry(_tool("a"), _tool("b"))
    rutas: list[str] = []

    def _ruta(texto: str):
        rutas.append(texto)
        return reg.schemas()

    fija = _politica(name="local", fastpath=False, direct_tools=frozenset(), route=_ruta)
    llm = _Motor(guion=[_pide("a"), _pide("b"), ChatResponse(content="Hecho, patrón.")])
    agent = _agente(llm, reg, policy=lambda: fija)

    assert agent.handle_turn("haz a y b") == "Hecho, patrón."
    assert llm.tools == [2, 2, 2]
    assert rutas == ["haz a y b"]


def test_sin_policy_los_kwargs_mandan_como_siempre_y_se_leen_en_vivo():
    """policy=None (todo caller que no es build_agent con GLM): mismo fast-path,
    el router corre una vez por turno sobre el texto LIMPIO, y mutar los atributos
    después de construir sigue surtiendo efecto (se leen en cada turno)."""
    reg = _registry(_tool("ram_usage", lambda: "RAM: 5 GB, patrón."), _tool("otra"))
    rutas: list[str] = []

    def _ruta(texto: str):
        rutas.append(texto)
        return reg.schemas()

    llm = _Motor(guion=[_pide("otra"), ChatResponse(content="Listo, patrón.")])
    agent = _agente(llm, reg, route_fn=_ruta, pre_hooks=[lambda t: f"[ahora: x]\n{t}"])

    # fastpath=True de fábrica: "cuánta ram tengo" no toca el LLM.
    assert agent.handle_turn("cuánta ram tengo") == "RAM: 5 GB, patrón."
    assert llm.tools == [] and rutas == []
    assert agent.routing_text == "cuánta ram tengo"

    agent.fastpath = False
    agent.handle_turn("haz otra cosa")
    assert llm.tools == [2, 2]
    assert rutas == ["haz otra cosa"]  # una vez, sin la fecha del pre-hook

    llm.guion = [_pide("ram_usage"), ChatResponse(content="Redactado, patrón.")]
    agent.direct_tools = set()  # apaga el short-circuit: 2 llamadas
    assert agent.handle_turn("y la ram?") == "Redactado, patrón."


def _llamadas_antes(direct_tools, nombre: str, flag: bool) -> int:
    """Lo que hacía el _is_direct de antes: un set vacío apaga todo, flag incluido."""
    if not direct_tools:
        return 2
    return 1 if (nombre in direct_tools or flag) else 2


@pytest.mark.parametrize("direct_tools", [None, set(), {"en_set"}])
@pytest.mark.parametrize("nombre,flag", [("en_set", False), ("con_flag", True),
                                         ("ram_usage", False), ("normal", False)])
def test_short_circuit_sin_policy_identico_al_de_antes(direct_tools, nombre, flag):
    reg = _registry(_tool(nombre, lambda: "Salida de la tool, patrón.", direct=flag))
    llm = _Motor(guion=[_pide(nombre), ChatResponse(content="Redactado, patrón.")])
    agent = _agente(llm, reg, fastpath=False, direct_tools=direct_tools)
    esperado = direct_tools if direct_tools is not None else set(DEFAULT_DIRECT_TOOLS)

    agent.handle_turn("hazlo")

    assert len(llm.tools) == _llamadas_antes(esperado, nombre, flag)


@pytest.mark.parametrize("honra,llamadas", [(False, 2), (True, 1)])
def test_honor_tool_direct_decide_el_flag_de_la_tool(honra, llamadas):
    """La nube de build_agent NO honra Tool.direct (lo de hoy: list_windows lo
    redacta GLM); honrarlo es la alternativa que queda para el dueño."""
    reg = _registry(_tool("list_windows", lambda: "Firefox, patrón.", direct=True))
    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset(),
                     honor_tool_direct=honra)
    llm = _Motor(guion=[_pide("list_windows"), ChatResponse(content="Solo Firefox, patrón.")])
    agent = _agente(llm, reg, policy=lambda: nube)

    agent.handle_turn("qué ventanas tengo")

    assert len(llm.tools) == llamadas


@pytest.mark.parametrize("cae_a,llamadas", [("local", 1), ("cloud", 2)])
def test_el_short_circuit_lo_decide_el_motor_que_respondio(cae_a, llamadas):
    """La política se evalúa DESPUÉS de la llamada: si la nube cayó dentro y
    respondió el local con atajos, ram_usage vuelve directa (y al revés)."""
    reg = _registry(_tool("ram_usage", lambda: "RAM: 5 GB, patrón."))
    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset(),
                     honor_tool_direct=False)
    local = _politica(name="local", fastpath=True, direct_tools=DEFAULT_DIRECT_TOOLS)
    estado = {"ahora": local if cae_a == "cloud" else nube}

    def _cambia() -> None:
        estado["ahora"] = local if cae_a == "local" else nube

    llm = _Motor(guion=[_pide("ram_usage"), ChatResponse(content="Redactado, patrón.")],
                 al_llamar=_cambia)
    agent = _agente(llm, reg, fastpath=False, policy=lambda: estado["ahora"])

    agent.handle_turn("oye dime la ram")

    assert len(llm.tools) == llamadas


@pytest.mark.parametrize("fastpath,llamadas", [(True, 0), (False, 2)])
def test_el_fastpath_lo_decide_la_politica_y_no_el_kwarg(fastpath, llamadas):
    reg = _registry(_tool("ram_usage", lambda: "RAM: 5 GB, patrón."))
    pol = _politica(name="p", fastpath=fastpath, direct_tools=frozenset(),
                    honor_tool_direct=False)
    llm = _Motor(guion=[_pide("ram_usage"), ChatResponse(content="Redactado, patrón.")])
    # El kwarg dice lo contrario: con `policy` solo queda como atributo de compatibilidad.
    agent = _agente(llm, reg, fastpath=not fastpath, policy=lambda: pol)

    agent.handle_turn("cuánta ram tengo")

    assert len(llm.tools) == llamadas


def test_la_politica_se_reevalua_tras_add_user():
    """Con compaction, add_user llama al resumidor (el LLM compartido) y esa
    llamada puede abrir el breaker: las tools salen de la política de después,
    sin enrutar antes para el motor que ya no va a responder."""
    reg = _registry(_tool("a"), _tool("b"), _tool("c"))
    estado = {"degradado": False}
    rutas: dict[str, list[str]] = {"cloud": [], "degraded": []}

    def _ruta(motor: str, schemas):
        def ruta(texto: str):
            rutas[motor].append(texto)
            return schemas()
        return ruta

    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset(),
                     route=_ruta("cloud", reg.schemas))
    local = _politica(name="degraded", fastpath=False, direct_tools=frozenset(),
                      route=_ruta("degraded", lambda: [reg.get("a").schema()]))

    class _ConvQueResume(Conversation):
        def add_user(self, content: str) -> None:
            estado["degradado"] = True  # el resumidor descubrió la caída
            super().add_user(content)

    llm = _Motor()
    agent = ToolAgent(llm, _ConvQueResume("SYS"), registry=reg, guard=Guard(allowed_roots=[]),
                      pre_hooks=[], post_hooks=[],
                      policy=lambda: local if estado["degradado"] else nube)

    agent.handle_turn("hola")

    assert llm.tools == [1]
    assert rutas == {"cloud": [], "degraded": ["hola"]}


def test_answered_policy_decide_el_short_circuit():
    """`answered_policy` (de fábrica: la misma `policy`) dice qué motor RESPONDIÓ:
    con build_agent lo anota FallbackLLM, no se deduce del breaker después."""
    reg = _registry(_tool("ram_usage", lambda: "RAM: 5 GB, patrón."))
    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset(),
                     honor_tool_direct=False)
    local = _politica(name="local", fastpath=True, direct_tools=DEFAULT_DIRECT_TOOLS)
    llm = _Motor(guion=[_pide("ram_usage"), ChatResponse(content="Redactado, patrón.")])
    agent = _agente(llm, reg, policy=lambda: nube, answered_policy=lambda: local)

    assert agent.handle_turn("oye dime la ram") == "RAM: 5 GB, patrón."
    assert len(llm.tools) == 1


def test_cada_politica_se_enruta_una_vez_por_turno():
    """El motor va y vuelve dentro del turno (nube -> local -> nube): cada política
    se enruta UNA vez y repite su lista; el turno siguiente vuelve a enrutar. El
    router no es puro (el round-robin MCP avanza en cada llamada), así que
    enrutar dos veces daba dos sets distintos dentro del mismo turno."""
    reg = _registry(_tool("a"), _tool("b"), _tool("c"))
    rutas: list[str] = []
    nube = _politica(name="cloud", fastpath=False, direct_tools=frozenset())

    def _ruta(texto: str):
        rutas.append(texto)
        return [reg.get("a").schema()]

    local = _politica(name="degraded", fastpath=False, direct_tools=frozenset(), route=_ruta)
    secuencia = [nube, local, nube, local]
    estado = {"i": 0}

    def _avanza() -> None:
        estado["i"] += 1

    llm = _Motor(guion=[_pide("a"), _pide("b"), _pide("c"), ChatResponse(content="Ya.")])
    agent = _agente(llm, reg, policy=lambda: secuencia[min(estado["i"], 3)])
    vistas: list[object] = []
    # Dentro de cada llamada: lo que pediría el shaper, y el motor cambia.
    llm.al_llamar = lambda: (vistas.append(agent.schemas_for_turn(local)), _avanza())

    agent.handle_turn("haz todo")

    assert llm.tools == [3, 1, 3, 1]
    assert rutas == ["haz todo"]
    # Quien adapte la petición al local en plena llamada (el shaper) ve LA MISMA lista.
    assert all(v is vistas[0] for v in vistas)
    estado["i"] = 1
    llm.guion = [ChatResponse(content="Otra.")]
    agent.handle_turn("otra cosa")
    assert rutas == ["haz todo", "otra cosa"]


def test_cancelar_sigue_cortando_con_politica():
    """should_abort y TurnCancelled no cambian con la política por llamada."""
    reg = _registry(_tool("a"))
    pol = _politica(name="cloud", fastpath=False, direct_tools=frozenset())
    llm = _Motor()
    conv = Conversation("SYS")
    agent = ToolAgent(llm, conv, registry=reg, guard=Guard(allowed_roots=[]),
                      pre_hooks=[], post_hooks=[], policy=lambda: pol)

    assert agent.handle_turn("hola", should_abort=lambda: True) == ""
    assert llm.tools == []
    assert [m.role for m in conv.history] == ["user", "assistant"]


# --- paso 2: FallbackLLM adapta la petición al local y re-arma el cooldown ----------

_TRES = [{"function": {"name": n}} for n in ("a", "b", "c")]


def _llamar(llm: FallbackLLM, modo: str, tools, on_token=None):
    if modo == "chat":
        return llm.chat([{"role": "user", "content": "x"}], tools=tools)
    return llm.chat_stream([{"role": "user", "content": "x"}], tools=tools, on_token=on_token)


@pytest.mark.parametrize("modo", ["chat", "chat_stream"])
@pytest.mark.parametrize("error", [TransientLLMError("caída"), GLMAuthError("401")])
def test_fallo_en_la_llamada_manda_al_local_la_peticion_adaptada(modo, error):
    nube, local = _Motor("glm", error=error), _Motor("ollama")
    llm = FallbackLLM(nube, local)
    vistos: list[tuple] = []

    def _shaper(messages, tools):
        vistos.append((messages, tools))
        return messages, tools[:1]

    llm.shape_for_secondary = _shaper
    _llamar(llm, modo, _TRES)

    assert nube.tools == [3]
    assert local.tools == [1]
    assert vistos == [([{"role": "user", "content": "x"}], _TRES)]


@pytest.mark.parametrize("modo", ["chat", "chat_stream"])
def test_sin_shaper_el_local_recibe_lo_mismo_de_siempre(modo):
    nube, local = _Motor("glm", falla=True), _Motor("ollama")
    _llamar(FallbackLLM(nube, local), modo, _TRES)
    assert local.recibidas == [_TRES] and local.recibidas[0] is _TRES


@pytest.mark.parametrize("modo", ["chat", "chat_stream"])
@pytest.mark.parametrize("tools", [None, []])
def test_sin_tools_no_se_adapta(modo, tools):
    """El resumidor de compaction y el Agent base no mandan tools: nada que adaptar."""
    nube, local = _Motor("glm", falla=True), _Motor("ollama")
    llm = FallbackLLM(nube, local)
    llm.shape_for_secondary = lambda m, t: pytest.fail("no debía adaptarse")
    _llamar(llm, modo, tools)
    assert local.recibidas == [tools]


def test_shaper_roto_se_loguea_y_el_local_recibe_lo_original(caplog):
    nube, local = _Motor("glm", falla=True), _Motor("ollama")
    llm = FallbackLLM(nube, local)

    def _roto(messages, tools):
        raise RuntimeError("router roto")

    llm.shape_for_secondary = _roto
    with caplog.at_level(logging.ERROR, logger="crotolamo.core.fallback"):
        assert llm.chat([], tools=_TRES).content == "Listo, patrón."
    assert local.recibidas == [_TRES]
    assert "router roto" in caplog.text


@pytest.fixture
def reloj(monkeypatch):
    estado = {"t": 1000.0}
    monkeypatch.setattr("crotolamo.core.fallback.time.monotonic", lambda: estado["t"])
    return estado


@pytest.mark.parametrize("modo", ["chat", "chat_stream"])
def test_el_cooldown_cuenta_desde_que_respondio_el_local(reloj, modo):
    """Carga en frío del modelo local (~60-120 s) más larga que el cooldown: antes,
    la siguiente llamada del MISMO turno volvía a sondear la nube caída."""

    def _carga_en_frio() -> None:
        reloj["t"] += 100

    nube, local = _Motor("glm", falla=True), _Motor("ollama", al_llamar=_carga_en_frio)
    llm = FallbackLLM(nube, local, cooldown_s=60)

    _llamar(llm, modo, _TRES)               # la nube falla; el local responde a t+100
    _llamar(llm, modo, _TRES)               # misma ráfaga: no se vuelve a sondear
    assert nube.tools == [3]

    local.al_llamar = None
    reloj["t"] += 61                        # pasó el cooldown desde la respuesta
    _llamar(llm, modo, _TRES)
    assert nube.tools == [3, 3]


def test_con_el_breaker_ya_abierto_ni_se_adapta_ni_se_alarga(reloj):
    """Esas llamadas ya las armó el agente para el local; y alargar el cooldown en
    cada respuesta local dejaría de probar la nube en charla continua."""
    nube, local = _Motor("glm", falla=True), _Motor("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)
    llm.chat([], tools=_TRES)                # abre el breaker (reabre a t=1000)
    llm.shape_for_secondary = lambda m, t: pytest.fail("ya estaba abierto")

    def _lento() -> None:
        reloj["t"] += 30

    local.al_llamar = _lento
    llm.chat([], tools=_TRES)                # t=1030 al terminar
    llm.chat([], tools=_TRES)                # t=1060 al terminar
    assert local.tools == [3, 3, 3]          # tal cual: las armó el agente
    local.al_llamar = None
    nube.falla = False
    # t=1060: el cooldown se contó desde la sonda (1000), no desde cada respuesta.
    assert llm.chat([], tools=_TRES).content == "Listo, patrón."
    assert nube.tools == [3, 3]


def test_un_local_que_tambien_falla_no_deja_el_cooldown_viejo(reloj):
    """El re-armado va en finally: si el local revienta tras su carga en frío, la
    siguiente llamada tampoco paga otra sonda de la nube caída."""

    def _carga_y_falla() -> None:
        reloj["t"] += 100

    nube = _Motor("glm", falla=True)
    local = _Motor("ollama", error=LLMError("Ollama no respondió"), al_llamar=_carga_y_falla)
    llm = FallbackLLM(nube, local, cooldown_s=60)
    with pytest.raises(LLMError):
        llm.chat([], tools=_TRES)
    local.error = None
    local.al_llamar = None
    llm.chat([], tools=_TRES)
    assert nube.tools == [3]


def test_la_auth_rechazada_sigue_caida_hasta_reiniciar(reloj):
    nube, local = _Motor("glm", error=GLMAuthError("401")), _Motor("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)
    llm.chat([], tools=_TRES)
    reloj["t"] += 10_000
    llm.chat([], tools=_TRES)
    assert nube.tools == [3] and llm.degraded


def test_orden_en_el_stream_restart_checkpoint_shaper_local():
    """El shaper va JUSTO antes del local: después del restart() del streamer y
    del punto de control (un turno cancelado no paga ni el router)."""
    orden: list[str] = []

    class _Streamer:
        def __call__(self, chunk):
            return False

        def restart(self):
            orden.append("restart")

        def checkpoint(self):
            orden.append("checkpoint")

    local = _Motor("ollama", al_llamar=lambda: orden.append("local"))
    llm = FallbackLLM(_Motor("glm", falla=True), local)

    def _shaper(messages, tools):
        orden.append("shaper")
        return messages, tools

    llm.shape_for_secondary = _shaper
    llm.chat_stream([], tools=_TRES, on_token=_Streamer())
    assert orden == ["restart", "checkpoint", "shaper", "local"]


def test_cancelado_mientras_la_nube_cae_no_adapta_ni_arranca_el_local():
    class _Cancelado:
        def __call__(self, chunk):
            return False

        def checkpoint(self):
            raise TurnCancelled()

    local = _Motor("ollama")
    llm = FallbackLLM(_Motor("glm", falla=True), local)
    llm.shape_for_secondary = lambda m, t: pytest.fail("cancelado: no se adapta")
    with pytest.raises(TurnCancelled):
        llm.chat_stream([], tools=_TRES, on_token=_Cancelado())
    assert local.tools == []


@pytest.mark.parametrize("modo", ["chat", "chat_stream"])
def test_fallback_anota_que_motor_respondio(reloj, modo):
    """`answered_by_secondary` es de la llamada que ACABA de volver, no del breaker:
    una llamada local que dura más que lo que le quedaba al cooldown vuelve con
    `degraded` ya en False, y aun así la respondió el local."""
    nube, local = _Motor("glm"), _Motor("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)
    assert llm.answered_by_secondary is False

    _llamar(llm, modo, _TRES)
    assert llm.answered_by_secondary is False                 # la nube
    nube.falla = True
    _llamar(llm, modo, _TRES)
    assert llm.answered_by_secondary is True                  # cayó aquí: el local

    def _vence_el_cooldown() -> None:
        reloj["t"] += 100

    local.al_llamar = _vence_el_cooldown
    _llamar(llm, modo, _TRES)                                 # breaker abierto
    assert not llm.degraded and llm.answered_by_secondary is True

    local.al_llamar = None
    nube.falla = False
    _llamar(llm, modo, _TRES)
    assert llm.answered_by_secondary is False                 # volvió la nube


def test_quien_respondio_es_de_cada_hilo():
    """Lo lee quien hizo la llamada, en su hilo (como la conexión del transporte):
    la llamada de otro hilo no le cambia la respuesta."""
    llm = FallbackLLM(_Motor("glm"), _Motor("ollama"))
    llm.chat([], tools=_TRES)
    otro: list[bool] = []

    def _en_otro_hilo() -> None:
        llm.primary.falla = True
        llm.chat([], tools=_TRES)
        otro.append(llm.answered_by_secondary)

    hilo = threading.Thread(target=_en_otro_hilo)
    hilo.start()
    hilo.join(timeout=5)
    assert otro == [True]
    assert llm.answered_by_secondary is False


def test_si_ya_hablo_no_hay_respaldo_ni_shaper():
    class _Habla(_Motor):
        def chat_stream(self, messages, tools=None, on_token=None):
            on_token("Órale, ")
            raise TransientLLMError("se cortó")

    local = _Motor("ollama")
    llm = FallbackLLM(_Habla("glm"), local)
    llm.shape_for_secondary = lambda m, t: pytest.fail("ya habló: no hay respaldo")
    with pytest.raises(TransientLLMError):
        llm.chat_stream([], tools=_TRES, on_token=lambda chunk: None)
    assert local.tools == []


# --- paso 3: build_agent, una política por motor -----------------------------------

class _Resp:
    """Respuesta HTTP mínima de Ollama (/api/chat sin stream)."""

    def __init__(self, message: dict) -> None:
        self.status = 200
        self._body = json.dumps({"message": message, "done": True}).encode()

    def read(self, *a):
        body, self._body = self._body, b""
        return body

    def getheader(self, name, default=None):
        return default


class _OllamaDeMentiras:
    """post_json del LLMClient local: anota las tools y pide ram_usage una vez."""

    def __init__(self, pide: str | None = "ram_usage",
                 dice: str = "Tienes 10 GB libres, patrón.") -> None:
        self.tools: list[int] = []
        self.pide = pide
        self.dice = dice

    def __call__(self, path, payload, headers=None):
        self.tools.append(len(payload.get("tools") or []))
        ultimo = payload["messages"][-1]
        if self.pide and ultimo["role"] != "tool":
            nombres = {t["function"]["name"] for t in payload.get("tools") or []}
            if self.pide in nombres:
                return _Resp({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": self.pide, "arguments": {}}}]})
        return _Resp({"role": "assistant", "content": self.dice})


def _glm_caido(path, payload, headers=None):
    raise TimeoutError("timed out")


def _build(monkeypatch, tmp_path, llm_raw: dict):
    """build_agent REAL con build_llm REAL (GLMClient + LLMClient + FallbackLLM)."""
    from crotolamo.core import hooks
    from crotolamo.core import memoria as memoria_mod
    from interfaces import shell

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "key-de-mentiras")
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": llm_raw}, user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    agent, _ = shell.build_agent()
    agent.registry.get("ram_usage").func = lambda: "RAM: 5.0 GB usados, patrón."
    return agent, agent.llm


def _enrutadas(agent, texto: str) -> int:
    from crotolamo.core.router import route_schemas

    return len(route_schemas(agent.registry, texto, 8))


def test_el_turno_que_descubre_la_caida_manda_la_dieta_chica_al_local(monkeypatch, tmp_path):
    """La nube no responde (timeout del transporte) en el primer turno: antes el
    local recibía las 42 schemas en sus dos llamadas ([42, 42])."""
    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm"})
    assert isinstance(llm, FallbackLLM)
    sondas: list[int] = []

    def _glm(path, payload, headers=None):
        sondas.append(len(payload.get("tools") or []))
        return _glm_caido(path, payload, headers)

    llm.primary._transport.post_json = _glm
    ollama = _OllamaDeMentiras()
    llm.secondary._transport.post_json = ollama

    texto = "oye dime cuanta ram me queda libre"
    assert agent.handle_turn(texto) == "Tienes 10 GB libres, patrón."
    n = _enrutadas(agent, texto)
    assert 0 < n < len(agent.registry.schemas())
    assert sondas == [len(agent.registry.schemas())]     # la nube sí ve todo
    assert ollama.tools == [n, n]                        # antes: [42, 42]


def test_en_voz_el_turno_que_descubre_la_caida_tambien_va_a_dieta(monkeypatch, tmp_path):
    """Lo mismo por chat_stream (stream_speak, el loop simple de voz): el shaper va
    después del restart() del streamer y la respuesta del local se oye."""
    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm"})
    llm.primary = _Motor("glm", falla=True)
    local = _Motor("ollama", guion=[_pide("ram_usage"),
                                    ChatResponse(content="Tienes 10 GB libres, patrón.")])
    llm.secondary = local
    dichos: list[str] = []

    texto = "oye dime cuanta ram me queda libre"
    assert agent.handle_turn(texto, on_token=dichos.append) == "Tienes 10 GB libres, patrón."
    n = _enrutadas(agent, texto)
    assert llm.primary.tools == [len(agent.registry.schemas())]
    assert local.tools == [n, n]
    assert "".join(dichos) == "Tienes 10 GB libres, patrón."


def test_sin_tool_routing_el_local_recibe_todo_como_hoy(monkeypatch, tmp_path):
    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm", "tool_routing": False})
    assert llm.shape_for_secondary is None
    llm.primary._transport.post_json = _glm_caido
    ollama = _OllamaDeMentiras()
    llm.secondary._transport.post_json = ollama
    agent.handle_turn("oye dime cuanta ram me queda libre")
    todas = len(agent.registry.schemas())
    assert ollama.tools == [todas, todas]


def test_degradado_sin_atajos_se_comporta_como_hoy(monkeypatch, tmp_path):
    """Sin la clave (de fábrica): con el breaker abierto, ni fast-path ni
    short-circuit; solo vuelve el routing chico."""
    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm"})
    llm._retry_at = time.monotonic() + 60
    llm.primary._transport.post_json = lambda *a, **k: pytest.fail("breaker abierto")
    ollama = _OllamaDeMentiras()
    llm.secondary._transport.post_json = ollama

    assert agent.handle_turn("cuánta ram tengo") == "Tienes 10 GB libres, patrón."
    n = _enrutadas(agent, "cuánta ram tengo")
    assert ollama.tools == [n, n]


@pytest.mark.parametrize("atajos,llamadas,respuesta", [
    (False, 2, "Solo Firefox con YouTube, patrón."),       # de fábrica: lo redacta el local
    (True, 1, "Tienes 1 ventana abierta, patrón:\nworkspace 2: firefox — YouTube"),
])
def test_degradado_una_tool_con_flag_direct(monkeypatch, tmp_path, atajos, llamadas, respuesta):
    """list_windows lleva Tool.direct=True. Con la config de fábrica el local
    degradado NO honra el flag (lo de hoy: lo redacta el modelo, 2 llamadas); con
    [llm.glm].fallback_shortcuts usa sus atajos y dice la lista de la tool."""
    from crotolamo.tools import windows

    monkeypatch.setattr(windows, "_clients", lambda: [
        {"class": "firefox", "title": "YouTube", "workspace": {"id": 2}},
    ])
    raw: dict = {"backend": "glm"}
    if atajos:
        raw["glm"] = {"fallback_shortcuts": True}
    agent, llm = _build(monkeypatch, tmp_path, raw)
    llm._retry_at = time.monotonic() + 600
    llm.primary._transport.post_json = lambda *a, **k: pytest.fail("breaker abierto")
    ollama = _OllamaDeMentiras(pide="list_windows", dice="Solo Firefox con YouTube, patrón.")
    llm.secondary._transport.post_json = ollama

    # Una frase que el fast-path no atrapa (con atajos, "qué ventanas tengo
    # abiertas" ni llega al modelo): la tool la elige el modelo local.
    assert agent.handle_turn("dime las ventanas que tengo abiertas por favor") == respuesta
    assert len(ollama.tools) == llamadas


@pytest.mark.parametrize("tarda", [2.0, 8.0])
def test_el_short_circuit_es_del_motor_que_respondio_aunque_venza_el_cooldown(
        monkeypatch, tmp_path, reloj, tarda):
    """Con atajos y el breaker a 5 s de cerrarse. Si el local tardaba más que eso,
    al volver `degraded` ya era False y se aplicaba la política de la NUBE a la
    respuesta del LOCAL: sin short-circuit, una 2ª llamada al local y una sonda a
    la nube caída con las 42 tools (antes, con 8 s: [4, 4] y [42])."""
    agent, llm = _build(monkeypatch, tmp_path,
                        {"backend": "glm", "glm": {"fallback_shortcuts": True}})
    llm._retry_at = reloj["t"] + 5
    sondas: list[int] = []

    def _glm(path, payload, headers=None):
        sondas.append(len(payload.get("tools") or []))
        return _glm_caido(path, payload, headers)

    llm.primary._transport.post_json = _glm
    ollama = _OllamaDeMentiras()

    def _local(path, payload, headers=None):
        reloj["t"] += tarda                     # lo que tarda el local en responder
        return ollama(path, payload, headers)

    llm.secondary._transport.post_json = _local

    assert agent.handle_turn("oye dime cuanta ram me queda libre") == "RAM: 5.0 GB usados, patrón."
    assert len(ollama.tools) == 1
    assert sondas == []


def test_el_turno_que_descubre_la_caida_enruta_una_sola_vez(monkeypatch, tmp_path):
    """El shaper (llamada 1) y el agente (llamada 2) comparten la lista enrutada.
    Con dos servers MCP que casan con la frase, el round-robin del router avanzaba
    dos veces y las dos llamadas locales del MISMO turno veían tools distintas
    (sin cache de prefijo en la 2ª, y la tool recién pedida podía desaparecer)."""
    from crotolamo.core import router

    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm"})
    nombres = {srv: [f"{srv}_memoria_{i}" for i in range(4)] for srv in ("zz_a", "zz_b")}
    try:
        for srv, tools in nombres.items():
            for n in tools:
                agent.registry.register(_tool(n))
            router.register_group(f"mcp:{srv}", tools, ["ram"])
        llm.primary._transport.post_json = _glm_caido
        vistas: list[list[str]] = []
        ollama = _OllamaDeMentiras()

        def _local(path, payload, headers=None):
            vistas.append([t["function"]["name"] for t in payload.get("tools") or []])
            return ollama(path, payload, headers)

        llm.secondary._transport.post_json = _local
        antes = router._RR_COUNTER

        agent.handle_turn("oye dime cuanta ram me queda libre")

        assert len(vistas) == 2 and vistas[0] == vistas[1]
        assert any(n.startswith("zz_") for n in vistas[0])   # los grupos MCP sí entraron
        assert router._RR_COUNTER - antes == 1                 # antes: 2
    finally:
        for srv, tools in nombres.items():
            router.unregister_group(f"mcp:{srv}")
            for n in tools:
                agent.registry.unregister(n)


def test_degradado_con_atajos_el_fastpath_no_llama_al_llm(monkeypatch, tmp_path):
    agent, llm = _build(monkeypatch, tmp_path,
                        {"backend": "glm", "glm": {"fallback_shortcuts": True}})
    llm._retry_at = time.monotonic() + 60
    llm.primary._transport.post_json = lambda *a, **k: pytest.fail("breaker abierto")
    ollama = _OllamaDeMentiras()
    llm.secondary._transport.post_json = ollama

    assert agent.handle_turn("cuánta ram tengo") == "RAM: 5.0 GB usados, patrón."
    assert ollama.tools == []
    # Y una tool directa elegida por el local vuelve a su short-circuit: 1 llamada.
    assert agent.handle_turn("oye dime cuanta ram me queda libre") == "RAM: 5.0 GB usados, patrón."
    assert len(ollama.tools) == 1


def test_con_atajos_y_la_nube_sana_nada_cambia(monkeypatch, tmp_path):
    """La clave solo toca al local degradado: GLM sano sigue sin atajos."""
    agent, llm = _build(monkeypatch, tmp_path,
                        {"backend": "glm", "glm": {"fallback_shortcuts": True}})
    nube = _Motor("glm", guion=[_pide("ram_usage"), ChatResponse(content="Redactado, patrón.")])
    llm.primary = nube

    assert agent.handle_turn("cuánta ram tengo") == "Redactado, patrón."
    todas = len(agent.registry.schemas())
    assert nube.tools == [todas, todas]


def test_glm_sano_ve_todo_y_lo_redacta_todo(monkeypatch, tmp_path):
    """Bench G: con la nube arriba, list_windows (Tool.direct) también lo redacta GLM."""
    from crotolamo.tools import windows

    monkeypatch.setattr(windows, "_clients", lambda: [
        {"class": "firefox", "title": "YouTube", "workspace": {"id": 2}},
    ])
    agent, llm = _build(monkeypatch, tmp_path, {"backend": "glm"})
    nube = _Motor("glm", guion=[_pide("list_windows"), ChatResponse(content="Firefox, patrón.")])
    llm.primary = nube
    assert agent.handle_turn("qué apps tengo abiertas") == "Firefox, patrón."
    todas = len(agent.registry.schemas())
    assert nube.tools == [todas, todas]


def test_con_ollama_de_cerebro_no_cambia_nada(monkeypatch, tmp_path):
    from interfaces import shell

    llm = _Motor("ollama")
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, _ = _build(monkeypatch, tmp_path, {"backend": "ollama"})
    assert agent.handle_turn("cuánta ram tengo") == "RAM: 5.0 GB usados, patrón."
    assert llm.tools == []
    llm.guion = [_pide("ram_usage")]
    assert agent.handle_turn("oye dime cuanta ram me queda libre") == "RAM: 5.0 GB usados, patrón."
    assert llm.tools == [_enrutadas(agent, "oye dime cuanta ram me queda libre")]


def test_la_config_de_fabrica_documenta_fallback_shortcuts_apagado():
    texto = (_RAIZ / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    glm = re.split(r"^\[llm\.glm\]$", texto, maxsplit=1, flags=re.M)[1]
    glm = re.split(r"^\[", glm, maxsplit=1, flags=re.M)[0]
    assert re.search(r"^#?\s*fallback_shortcuts\s*=\s*false\b", glm, re.M)
    # De fábrica no se activa: la clave no está puesta a true.
    assert not tomllib.loads(texto)["llm"]["glm"].get("fallback_shortcuts", False)
    # El comentario del fastpath dice que con GLM solo vuelve con esta clave.
    antes = texto[:re.search(r"^fastpath\s*=", texto, re.M).start()]  # type: ignore[union-attr]
    comentario = antes.rstrip("\n").rsplit("\n\n", 1)[-1]
    assert "fallback_shortcuts" in comentario


# --- paso 4: tope de CONECTAR separado del de LEER ---------------------------------
# http.client usa un único timeout para el connect TCP (una vez POR CADA dirección
# que da el DNS), el handshake TLS y cada lectura: una nube en un agujero negro
# (wifi arriba, internet caído) costaba 60 s por dirección en cada sonda.

_MSGS = [{"role": "user", "content": "hola"}]


class _OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive, como Ollama y la nube

    def log_message(self, *args):
        pass

    def _stream(self) -> None:
        """/api/chat con stream: NDJSON chunked, done:true y el chunk terminal."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for obj in ({"message": {"role": "assistant", "content": "ok, patrón."}, "done": False},
                    {"message": {"role": "assistant", "content": ""}, "done": True}):
            linea = json.dumps(obj).encode() + b"\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(linea), linea))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self):
        pedido = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.server.peticiones += 1
        if pedido.get("stream"):
            self._stream()
            return
        body = json.dumps({"message": {"role": "assistant", "content": "ok, patrón."},
                           "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.server.cierre == "cabecera":
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        # "fin"/"rst", sin `Connection: close`: el cliente cree que el keep-alive
        # sigue y el servidor lo cuelga al rato, como hace cualquiera con un ocioso.
        self.close_connection = self.server.cierre is not None


class _Ollama(ThreadingHTTPServer):
    """/api/chat sin stream en 127.0.0.1. `cierre`: None = keep-alive normal;
    "fin" = cuelga tras responder (FIN); "rst" = cuelga con RST (sin FIN);
    "cabecera" = avisa `Connection: close` y cuelga."""

    daemon_threads = True

    def __init__(self, cierre: str | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _OllamaHandler)
        self.cierre = cierre
        self.conexiones = 0
        self.peticiones = 0
        self._hilo = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._hilo.start()

    def get_request(self):
        req = super().get_request()
        self.conexiones += 1
        return req

    def shutdown_request(self, request):
        if self.cierre == "rst":
            request.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            request.close()
            return
        super().shutdown_request(request)

    def handle_error(self, request, client_address):  # cortes deliberados: sin traza
        pass

    def parar(self) -> None:
        self.shutdown()
        self.server_close()
        self._hilo.join(timeout=5)

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def ollama():
    creados: list[_Ollama] = []

    def crear(cierre: str | None = None) -> _Ollama:
        srv = _Ollama(cierre)
        creados.append(srv)
        return srv

    yield crear
    for srv in creados:
        srv.parar()


def _espiar_conexiones(monkeypatch, sockets: list[socket.socket] | None = None,
                       ) -> list[float | None]:
    """Timeouts con los que http.client abre cada conexión (create_connection);
    en `sockets`, los sockets que abrió."""
    vistos: list[float | None] = []
    real = socket.create_connection

    def espia(address, timeout=None, *args, **kwargs):
        vistos.append(timeout)
        sock = real(address, timeout, *args, **kwargs)
        if sockets is not None:
            sockets.append(sock)
        return sock

    monkeypatch.setattr(socket, "create_connection", espia)
    return vistos


def test_transport_conecta_con_tope_corto_y_lee_con_el_largo(monkeypatch, ollama):
    srv = ollama()
    vistos = _espiar_conexiones(monkeypatch)
    client = LLMClient(host=srv.url, timeout=60)

    assert client.chat(_MSGS).content == "ok, patrón."
    assert client.chat(_MSGS).content == "ok, patrón."

    assert vistos == [engine.DEFAULT_CONNECT_TIMEOUT_S]   # antes: [60]
    assert client._transport._conn.sock.gettimeout() == 60  # leer: lo de siempre
    assert srv.conexiones == 1                             # el keep-alive sigue igual


def test_la_reconexion_automatica_de_http_client_tambien_separa_los_topes(monkeypatch,
                                                                          ollama):
    """Con `Connection: close` http.client suelta el socket y la siguiente
    petición reabre ESE MISMO objeto (auto_open): también con el tope corto
    para conectar y el largo para leer, no con el corto para todo."""
    srv = ollama("cabecera")
    sockets: list[socket.socket] = []
    vistos = _espiar_conexiones(monkeypatch, sockets)
    client = LLMClient(host=srv.url, timeout=60)

    client.chat(_MSGS)
    conn = client._transport._conn
    client.chat(_MSGS)

    assert client._transport._conn is conn                 # mismo objeto, reabierto
    assert vistos == [5.0, 5.0]
    assert [s.gettimeout() for s in sockets] == [60, 60]   # leer: lo de siempre
    assert srv.conexiones == 2


def test_una_nube_en_agujero_negro_cuesta_el_tope_por_direccion(monkeypatch):
    """Bench [0b]: el SYN nunca se contesta. Con 2 registros A el connect se
    intenta 2 veces, cada una con su tope: 2 x 5 s en vez de 2 x 60 s."""
    reales, conectar = socket.getaddrinfo, socket.socket.connect
    negras = ("192.0.2.1", "198.51.100.1")   # TEST-NET: nunca salen a la red
    esperas: list[float | None] = []

    def dns(host, port, *args, **kwargs):
        if host == "nube.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in negras]
        return reales(host, port, *args, **kwargs)

    def agujero_negro(sock, address):
        if address[0] not in negras:
            return conectar(sock, address)
        esperas.append(sock.gettimeout())   # lo que esperaría el SYN sin respuesta
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.setattr(socket.socket, "connect", agujero_negro)
    client = GLMClient(api_key="k", base_url="http://nube.test:9/v4", timeout=60)

    with pytest.raises(TransientLLMError, match="GLM no respondió a tiempo"):
        client.chat(_MSGS)

    assert esperas == [5.0, 5.0]                           # antes: [60, 60]


def test_el_handshake_tls_tambien_va_con_el_tope_de_conectar():
    """El TCP conecta pero el TLS no avanza (portal cautivo, proxy colgado): el
    handshake corría con el timeout de lectura entero."""
    mudo = socket.create_server(("127.0.0.1", 0))   # el kernel acepta; nadie habla
    try:
        port = mudo.getsockname()[1]
        client = GLMClient(api_key="k", base_url=f"https://127.0.0.1:{port}/v4",
                           timeout=3, connect_timeout=0.3)
        t0 = time.perf_counter()
        with pytest.raises(TransientLLMError, match="GLM no respondió a tiempo"):
            client.chat(_MSGS)
        transcurrido = time.perf_counter() - t0
    finally:
        mudo.close()

    assert transcurrido < 2.0                              # antes: los 3 s de lectura


@pytest.mark.parametrize("timeout, connect_timeout, esperado", [
    (60, None, 5.0), (180, None, 5.0), (2, None, 2), (60, 1.5, 1.5), (60, 30, 30),
])
def test_tope_de_conectar_por_defecto_y_explicito(timeout, connect_timeout, esperado):
    t = HTTPTransport("http://localhost:11434", timeout=timeout,
                      connect_timeout=connect_timeout)
    assert t.connect_timeout == esperado
    assert t.timeout == timeout                            # el de leer no se toca


@pytest.mark.parametrize("valor", [0, -1, "rapido", float("inf"), float("nan")])
def test_tope_de_conectar_invalido_cae_al_de_fabrica_y_avisa(caplog, valor):
    """Con 0 el connect fallaría siempre al instante y con -1 reventaría con un
    ValueError crudo en cada petición: se avisa y se usa el de fábrica."""
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.engine"):
        t = HTTPTransport("http://localhost:11434", timeout=60, connect_timeout=valor)
    assert t.connect_timeout == engine.DEFAULT_CONNECT_TIMEOUT_S
    assert any("connect_timeout" in r.getMessage() for r in caplog.records)


def test_los_clientes_leen_connect_timeout_de_la_config(tmp_path):
    raw = {"llm": {"timeout": 120, "connect_timeout": 2,
                   "glm": {"timeout": 50, "connect_timeout": 3}}}
    s = Settings(raw=raw, user="test", home=tmp_path)

    local = LLMClient.from_settings(s)._transport
    nube = GLMClient.from_settings(s)._transport

    assert (local.connect_timeout, local.timeout) == (2, 120)
    assert (nube.connect_timeout, nube.timeout) == (3, 50)


def test_con_la_config_de_fabrica_conectar_5s_y_leer_como_siempre():
    s = settings_mod.get_settings()
    local = LLMClient.from_settings(s)._transport
    nube = GLMClient.from_settings(s)._transport
    assert (local.connect_timeout, local.timeout) == (5.0, 180)
    assert (nube.connect_timeout, nube.timeout) == (5.0, 60)


def test_solo_el_motor_local_necesita_calentarse():
    """El warm-up del listener (paso 6, otro dueño) lo leerá: con GLM no hay nada
    que cargar y una llamada real a la nube por arranque no calienta nada."""
    assert LLMClient.needs_warmup is True
    assert GLMClient.needs_warmup is False
    assert LLMClient().needs_warmup is True
    assert GLMClient(api_key="k").needs_warmup is False


# --- paso 5: un keep-alive muerto se tira ANTES de escribir ------------------------

def _hasta_que_llegue_el_cierre(sock: socket.socket) -> None:
    """Espera (acotada) a que el FIN/RST del servidor llegue al socket ocioso."""
    legible, _, _ = select.select([sock], [], [], 5.0)
    assert legible, "el servidor no cerró el keep-alive"


def _contar_envios(monkeypatch, transport: HTTPTransport) -> list[list[str]]:
    """Por cada conexión que abre el transporte, las peticiones que le escribe."""
    por_conexion: list[list[str]] = []
    real = transport._connect

    def fabrica():
        conn = real()
        envios: list[str] = []
        por_conexion.append(envios)
        original = conn.request

        def request(method, url, *args, **kwargs):
            envios.append(url)
            return original(method, url, *args, **kwargs)

        conn.request = request  # type: ignore[method-assign]
        return conn

    monkeypatch.setattr(transport, "_connect", fabrica)
    return por_conexion


@pytest.mark.parametrize("cierre", ["fin", "rst"])
def test_reconecta_sin_enviar_si_el_servidor_cerro_el_ocioso(monkeypatch, ollama, cierre):
    """Antes se escribía la petición en el socket muerto y recién el
    RemoteDisconnected / ConnectionReset disparaba el reintento."""
    srv = ollama(cierre)
    client = LLMClient(host=srv.url, timeout=60)
    envios = _contar_envios(monkeypatch, client._transport)

    assert client.chat(_MSGS).content == "ok, patrón."
    _hasta_que_llegue_el_cierre(client._transport._conn.sock)
    assert client.chat(_MSGS).content == "ok, patrón."

    assert envios == [["/api/chat"], ["/api/chat"]]        # antes: [[..., ...], [...]]
    assert srv.peticiones == 2


def test_un_ocioso_vivo_se_reutiliza_aunque_lleve_rato(monkeypatch, ollama):
    """La política es UNA: el tope de inactividad (_REUSE_MAX_IDLE_S) es de los
    streams drenados; una respuesta leída entera se reutiliza mientras el kernel
    no la dé por muerta, como siempre (sin un handshake más por turno)."""
    reloj = {"t": 1000.0}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    srv = ollama()
    client = LLMClient(host=srv.url, timeout=60)

    client.chat(_MSGS)
    reloj["t"] += HTTPTransport._REUSE_MAX_IDLE_S * 10
    client.chat(_MSGS)

    assert srv.conexiones == 1


def _dns_de_mentiras(monkeypatch, nombre: str = "nube.test") -> None:
    reales = socket.getaddrinfo

    def dns(host, port, *args, **kwargs):
        return reales("127.0.0.1" if host == nombre else host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", dns)


def test_keepalive_tcp_en_la_nube(monkeypatch, ollama):
    """Sin keepalive, el kernel no prueba nunca un socket ocioso (Linux: 2 h por
    defecto): una ruta caída solo se descubría escribiendo y esperando 60 s."""
    srv = ollama()
    _dns_de_mentiras(monkeypatch)
    t = HTTPTransport(f"http://nube.test:{srv.port}/v4", timeout=60)

    t.post_json("/chat/completions", {}).read()
    sock = t._conn.sock

    assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
    tcp = {o: sock.getsockopt(socket.IPPROTO_TCP, getattr(socket, o))
           for o in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT")}
    assert tcp == {"TCP_KEEPIDLE": 10, "TCP_KEEPINTVL": 5, "TCP_KEEPCNT": 3}


def _keepalive(sock: socket.socket) -> bool:
    return bool(sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))


def test_un_stream_terminado_no_sondea_su_keep_alive(monkeypatch, ollama):
    """Un stream terminado solo se reutiliza dentro de _REUSE_MAX_IDLE_S (15 s) y
    el keepalive TCP no da un veredicto antes de 10 + 5*3 = 25 s: ahí solo eran
    sondas (medido en el kernel: 4 en 41 s, y la conexión se tiraba igual). Al
    reutilizarla vuelve a tenerlo: tras una respuesta leída entera, el ocioso no
    tiene tope y el keepalive es lo que descubre la ruta caída."""
    srv = ollama()
    _dns_de_mentiras(monkeypatch)
    client = LLMClient(host=f"http://nube.test:{srv.port}", timeout=60)
    assert client._transport._keepalive

    assert client.chat_stream(_MSGS, on_token=lambda chunk: None).content == "ok, patrón."
    assert not _keepalive(client._transport._conn.sock)       # antes: True

    assert client.chat(_MSGS).content == "ok, patrón."           # dentro del tope: reutiliza
    assert srv.conexiones == 1
    assert _keepalive(client._transport._conn.sock)

    assert client.chat_stream(_MSGS, on_token=lambda chunk: None).content == "ok, patrón."
    assert srv.conexiones == 1 and not _keepalive(client._transport._conn.sock)


def test_un_stream_leido_entero_conserva_el_keepalive(monkeypatch, ollama):
    """Si el consumidor leyó hasta el chunk terminal, la conexión se reutiliza sin
    tope de inactividad (como una respuesta sin stream): el keepalive se queda."""
    srv = ollama()
    _dns_de_mentiras(monkeypatch)
    t = HTTPTransport(f"http://nube.test:{srv.port}", timeout=60)

    resp = t.post_json("/api/chat", {"stream": True})
    resp.read()
    t.finish(resp)

    assert _keepalive(t._conn.sock)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_sin_keepalive_contra_loopback(ollama, host):
    """Ollama local no cierra nunca un keep-alive ocioso y el loopback no se cae
    en silencio: un sondeo cada 10 s, 24/7, sería un despertar sin beneficio."""
    srv = ollama()
    t = HTTPTransport(f"http://{host}:{srv.port}", timeout=60)

    t.post_json("/api/chat", {}).read()

    assert not t._conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)


class _Conn:
    def __init__(self, sock) -> None:
        self.sock = sock


class _SinFd:
    """Doble de socket de los tests viejos: sin fileno()."""


class _FdEnorme:
    def fileno(self) -> int:
        return 1 << 20   # select() no lo admite (FD_SETSIZE)


def test_dropped_sin_socket_que_mirar_cuenta_como_vivo():
    assert engine._dropped(_Conn(None)) is False           # aún sin conectar
    assert engine._dropped(_Conn(_SinFd())) is False        # dobles de los tests
    assert engine._dropped(_Conn(_FdEnorme())) is False     # no se puede saber: como hoy


def test_dropped_socket_cerrado_cuenta_como_muerto():
    sock = socket.socket()
    sock.close()
    assert engine._dropped(_Conn(sock)) is True


def test_dropped_ocioso_sano_no_es_legible():
    a, b = socket.socketpair()
    try:
        assert engine._dropped(_Conn(a)) is False
        b.close()                                          # EOF: legible
        assert engine._dropped(_Conn(a)) is True
    finally:
        a.close()


# --- paso 7: la config de fábrica documenta los topes de conectar ------------------

def test_la_config_de_fabrica_documenta_connect_timeout():
    texto = (_RAIZ / "config" / "crotolamo.toml").read_text(encoding="utf-8")
    llm = re.split(r"^\[llm\]$", texto, maxsplit=1, flags=re.M)[1]
    llm, glm = re.split(r"^\[llm\.glm\]$", llm, maxsplit=1, flags=re.M)
    glm = re.split(r"^\[", glm, maxsplit=1, flags=re.M)[0]
    for seccion in (llm, glm):
        m = re.search(r"^#\s*connect_timeout\s*=\s*([\d.]+)", seccion, re.M)
        assert m, "falta connect_timeout comentado con su valor de fábrica"
        assert float(m.group(1)) == engine.DEFAULT_CONNECT_TIMEOUT_S
    cfg = tomllib.loads(texto)["llm"]
    assert "connect_timeout" not in cfg and "connect_timeout" not in cfg["glm"]


def test_el_readme_no_mete_el_dns_en_el_tope_de_conectar():
    """socket.getaddrinfo corre ANTES de cualquier tope: un resolver que espera a un
    upstream caído suma su espera entera (medido: tope 0.2 s, intento 2.03 s). El
    toml ya lo decía; el README prometía "como mucho" connect_timeout por dirección."""
    texto = (_RAIZ / "README.md").read_text(encoding="utf-8")
    parrafo = next(p for p in texto.split("\n\n") if "`[llm.glm].connect_timeout`" in p)
    assert re.search(r"resolver\s+el\s+nombre", parrafo), parrafo


def test_el_cooldown_documentado_es_el_del_turno_que_descubre_la_caida():
    """El re-armado solo corre en la llamada que descubre la caída: un turno que
    arranca con el breaker a punto de cerrarse sí sondea la nube a media respuesta
    (residuo aceptado en el diseño). El comentario no puede prometer lo contrario."""
    lineas = (_RAIZ / "config" / "crotolamo.toml").read_text(encoding="utf-8").splitlines()
    i = next(n for n, linea in enumerate(lineas) if linea.startswith("fallback_cooldown_s"))
    comentario: list[str] = []
    for linea in reversed(lineas[:i]):
        if not linea.startswith("#"):
            break
        comentario.insert(0, linea.lstrip("#").strip())
    texto = " ".join(comentario)
    assert "el turno que descubre la caída" in texto, texto
    assert "un turno sin internet no vuelve" not in texto
