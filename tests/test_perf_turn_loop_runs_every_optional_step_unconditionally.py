"""Causa raíz "el turno corre cada paso opcional sin condición".

ToolAgent era una tubería fija: todos los pre-hooks, luego el fast-path, luego el
loop {LLM, cada tool en serie} y al final todos los after-turn hooks, sin nada
que describiera el turno. Los únicos hooks con coste real (la búsqueda y la
extracción de mem0) no distinguían "pausa la música" de "me gusta el café de
olla", y la memoria no aprendía de sus fallos.

Parte 1 (pasos 1-4):
- Paso 1: cada turno tiene un TurnInfo (core/turn.py): el comando anclado que
  reconoce fastpath.match (aunque el fast-path esté apagado), por dónde salió
  (`via`), cuántas llamadas al LLM hizo y el resultado de cada tool con una
  clave estable. Los hooks que lo piden (`wants_turn`) lo reciben; los demás se
  llaman exactamente como siempre.
- Paso 2 (tests en test_memoria.py): los hooks de la memoria no buscan ni
  extraen en un comando anclado.
- Paso 3: la búsqueda POR TURNO aprende de sus fallos (una a la vez, cancelada
  si se pasa del presupuesto y con breaker); las explícitas (tools, CLI) no.
- Paso 4: la extracción en segundo plano se pausa tras fallos seguidos, sin
  perder lo encolado, y cerrar() corta la pausa.

Parte 2 (pasos 5-7):
- Paso 5: las tools de solo lectura (Tool.parallel_safe; en MCP, solo
  readOnlyHint) que el modelo pide en un mismo mensaje corren a la vez en un pool
  acotado del agente, con el historial byte a byte igual que en serie.
- Paso 6: [llm].stop_on_repeated_failure (apagada de fábrica): una repetición de
  un fallo no se vuelve a correr ni a preguntar y la siguiente petición va sin
  tools.
- Paso 7 (tests en test_auditoria_mcp.py, de la ola de arreglos): un timeout MCP
  manda notifications/cancelled.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from crotolamo.core import fastpath as fastpath_mod
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse, LLMError
from crotolamo.core.memoria import (
    Memoria,
    MemoriaConfig,
    MemoriaNoDisponible,
    Recuerdo,
    make_memoria_prehook,
)
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Decision
from crotolamo.tools.base import Registry, Tool

_P = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}}


def _ok(x: str = "", y: str = "") -> str:
    return f"bien {x}{y}"


def _rompe(x: str = "", y: str = "") -> str:
    raise RuntimeError("se cayó")


def _soft(x: str = "", y: str = "") -> str:
    from crotolamo.mcp.bridge import _soft as soft_bridge

    return soft_bridge("demo", "reportó un fallo, patrón: boom")


def _registry() -> Registry:
    reg = Registry()
    reg.register(Tool("ok_tool", _ok, "ok", _P))
    reg.register(Tool("rompe", _rompe, "rompe", _P))
    reg.register(Tool("soft", _soft, "soft", _P, strict_args=False))
    reg.register(Tool("ram_usage", lambda: "Tienes 8 GB libres, patrón.", "ram",
                      {"type": "object", "properties": {}}))
    reg.register(Tool("music_control", lambda action="": f"Pausada, patrón. ({action})", "m",
                      {"type": "object", "properties": {"action": {"type": "string"}}}))
    reg.register(Tool("delicada", _ok, "delicada", _P, safe=False))
    reg.register(Tool("prohibida", _ok, "prohibida", _P))
    return reg


class _Guard:
    def check(self, tool, arguments):
        if tool.name == "prohibida":
            return Decision.block("Eso no lo toco, patrón.")
        if not tool.safe:
            return Decision.confirm("¿Le hago la cosa delicada, patrón?")
        return Decision.ok()


def _call(name, **args):
    return {"name": name, "arguments": args}


def _resp(*calls, content=""):
    return ChatResponse(content=content, tool_calls=list(calls), raw_message={"tool_calls": [
        {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})


class _LLM:
    def __init__(self, seq):
        self.seq = list(seq)
        self.llamadas = 0

    def chat(self, messages, tools=None):
        self.llamadas += 1
        item = self.seq.pop(0) if self.seq else ChatResponse(content="fin")
        if isinstance(item, Exception):
            raise item
        return item


def _agente(seq, **kw) -> ToolAgent:
    kw.setdefault("pre_hooks", [])
    kw.setdefault("post_hooks", [])
    return ToolAgent(_LLM(seq), Conversation("SYS"), registry=_registry(), guard=_Guard(), **kw)


# --- paso 1: TurnInfo ---------------------------------------------------------------

def test_call_key_no_depende_del_orden_de_los_argumentos():
    from crotolamo.core.turn import call_key

    assert call_key("t", {"a": 1, "b": "ñ"}) == call_key("t", {"b": "ñ", "a": 1})
    assert call_key("t", {"a": 1}) != call_key("u", {"a": 1})
    assert call_key("t", {"a": 1}) != call_key("t", {"a": 2})
    assert "ñ" in call_key("t", {"b": "ñ"})            # ensure_ascii=False
    assert call_key("t", {"p": object}) .startswith("t{")  # lo no-JSON no revienta


def test_el_comando_anclado_se_reconoce_aunque_el_fastpath_este_apagado():
    agent = _agente([ChatResponse(content="Va."), ChatResponse(content="Rico.")],
                    fastpath=False)
    agent.handle_turn("pausa la música")
    turno = agent.last_turn
    assert turno is not None and turno.command == ("music_control", {"action": "pause"})
    assert turno.text == "pausa la música" and turno.via == "llm" and turno.llm_calls == 1
    agent.handle_turn("me gusta el café de olla")
    assert agent.last_turn is not None and agent.last_turn is not turno
    assert agent.last_turn.command is None


def test_fastpath_match_corre_una_vez_por_turno(monkeypatch):
    """El fast-path reutiliza la clasificación del turno: un solo match."""
    llamadas: list[str] = []
    real = fastpath_mod.match

    def contando(text):
        llamadas.append(text)
        return real(text)

    monkeypatch.setattr(fastpath_mod, "match", contando)
    agent = _agente([], fastpath=True)
    assert agent.handle_turn("pausa la música") == "Pausada, patrón. (pause)"
    assert llamadas == ["pausa la música"]
    assert agent.last_turn is not None and agent.last_turn.via == "fastpath"
    assert agent.last_turn.llm_calls == 0
    assert [c.name for c in agent.last_turn.calls] == ["music_control"]


def test_hooks_que_piden_el_turno_reciben_el_mismo_y_los_de_siempre_igual():
    from crotolamo.core.turn import wants_turn

    vistos: list = []

    @wants_turn
    def pre_turno(text, turn=None):
        vistos.append(("pre_turno", text, turn))
        return text + " +t"

    def pre_viejo(text):
        vistos.append(("pre_viejo", text))
        return text + " +v"

    @wants_turn
    def after_turno(text, reply, turn=None):
        vistos.append(("after_turno", text, reply, turn))

    def after_viejo(text, reply):
        vistos.append(("after_viejo", text, reply))

    agent = _agente([ChatResponse(content="Hola, patrón.")],
                    pre_hooks=[pre_turno, pre_viejo],
                    after_turn_hooks=[after_turno, after_viejo])
    assert agent.handle_turn("hola") == "Hola, patrón."
    turno = agent.last_turn
    assert vistos == [
        ("pre_turno", "hola", turno),
        ("pre_viejo", "hola +t"),
        ("after_turno", "hola", "Hola, patrón.", turno),
        ("after_viejo", "hola", "Hola, patrón."),
    ]
    assert agent.conversation.history[0].content == "hola +t +v"


def test_un_mock_como_hook_se_llama_como_siempre():
    """Solo `wants_turn is True` cuenta: un atributo que da algo truthy (un Mock)
    no cambia la firma con la que se llama al hook."""
    from unittest.mock import MagicMock

    pre = MagicMock(side_effect=lambda t: t)
    after = MagicMock()
    agent = _agente([ChatResponse(content="ok")], pre_hooks=[pre], after_turn_hooks=[after])
    agent.handle_turn("hola")
    pre.assert_called_once_with("hola")
    after.assert_called_once_with("hola", "ok")


def test_cada_llamada_queda_con_su_resultado_y_una_clave_estable():
    from crotolamo.core.turn import call_key

    confirmaciones: list[str] = []
    agent = _agente([
        _resp(_call("ok_tool", x="1", y="2"), _call("no_existe", x="1"),
              _call("prohibida", x="1"), _call("delicada", x="1"),
              _call("rompe", x="1"), _call("soft", x="1")),
        ChatResponse(content="Hice lo que pude."),
    ], confirm_fn=lambda r: confirmaciones.append(r) or False)
    assert agent.handle_turn("haz todo") == "Hice lo que pude."
    turno = agent.last_turn
    assert turno is not None
    assert [(c.name, c.outcome) for c in turno.calls] == [
        ("ok_tool", "ok"), ("no_existe", "unknown_tool"), ("prohibida", "blocked"),
        ("delicada", "declined"), ("rompe", "hard_error"), ("soft", "soft_error"),
    ]
    assert turno.calls[0].key == call_key("ok_tool", {"y": "2", "x": "1"})
    # El resultado del registro es el MISMO que entró al historial.
    tool_msgs = [m.content for m in agent.conversation.history if m.role == "tool"]
    assert [c.result for c in turno.calls] == tool_msgs
    assert confirmaciones == ["¿Le hago la cosa delicada, patrón?"]
    assert turno.failures(call_key("rompe", {"x": "1"})) == 1
    assert turno.failures(call_key("ok_tool", {"x": "1", "y": "2"})) == 0
    ultimo = turno.last_failure(call_key("soft", {"x": "1"}))
    assert ultimo is not None and ultimo.outcome == "soft_error"
    assert turno.last_failure(call_key("ok_tool", {"x": "1", "y": "2"})) is None
    assert turno.llm_calls == 2 and turno.via == "llm"


def test_execute_call_y_safe_execute_devuelven_los_textos_de_siempre():
    """Los strings que ya veían el historial y los tests (wrappers finos)."""
    agent = _agente([], confirm_fn=lambda r: False)
    esperado = {
        "ok_tool": "bien 1",
        "rompe": "La tool 'rompe' reventó, patrón: se cayó",
        "soft": "El server MCP 'demo' reportó un fallo, patrón: boom",
        "no_existe": "No tengo una tool llamada 'no_existe', patrón.",
        "prohibida": "Eso no lo toco, patrón.",
        "delicada": "Cancelado por el patrón.",
    }
    for nombre, texto in esperado.items():
        assert agent._execute_call(nombre, {"x": "1"}) == texto
        assert agent._safe_execute(nombre, {"x": "1"}) == texto


def test_un_guard_que_revienta_queda_como_fallo_duro():
    class _GuardBoom:
        def check(self, tool, arguments):
            raise ValueError("ruta con \\0")

    agent = ToolAgent(_LLM([_resp(_call("ok_tool", x="1")), ChatResponse(content="Falló.")]),
                      Conversation("SYS"), registry=_registry(), guard=_GuardBoom(),
                      pre_hooks=[], post_hooks=[])
    assert agent._safe_execute("ok_tool", {"x": "1"}) == (
        "La tool 'ok_tool' reventó, patrón: ruta con \\0")
    agent.handle_turn("haz")
    turno = agent.last_turn
    assert turno is not None
    assert [(c.name, c.outcome) for c in turno.calls] == [("ok_tool", "hard_error")]


@pytest.mark.parametrize("seq,fastpath,texto,via,llm_calls", [
    ([], True, "pausa la música", "fastpath", 0),
    ([_resp(_call("ram_usage"))], True, "dime la ram", "direct", 1),
    ([ChatResponse(content="Hola.")], True, "hola", "llm", 1),
    ([_resp(_call("ok_tool", x="1"))] * 6, True, "haz algo", "exhausted", 6),
    ([LLMError("sin modelo, patrón")], True, "hola", "llm_error", 1),
    ([_resp(_call("ok_tool", x="1")), LLMError("se cayó")], True, "haz", "llm_error", 2),
])
def test_via_dice_por_donde_salio_el_turno(seq, fastpath, texto, via, llm_calls):
    agent = _agente(seq, fastpath=fastpath)
    agent.handle_turn(texto)
    assert agent.last_turn is not None
    assert (agent.last_turn.via, agent.last_turn.llm_calls) == (via, llm_calls)


def test_un_fastpath_con_fallo_duro_cae_al_llm_con_su_registro():
    reg = _registry()
    reg.register(Tool("music_control", _rompe, "m",
                      {"type": "object", "properties": {"action": {"type": "string"}}}))
    agent = ToolAgent(_LLM([ChatResponse(content="No hay reproductor, patrón.")]),
                      Conversation("SYS"), registry=reg, guard=_Guard(),
                      pre_hooks=[], post_hooks=[], fastpath=True)
    assert agent.handle_turn("pausa la música") == "No hay reproductor, patrón."
    turno = agent.last_turn
    assert turno is not None and turno.via == "llm" and turno.llm_calls == 1
    assert [(c.name, c.outcome) for c in turno.calls] == [("music_control", "hard_error")]


def test_un_turno_cancelado_queda_como_cancelled():
    pasos = {"n": 0}

    def abortar():
        pasos["n"] += 1
        return pasos["n"] > 2

    agent = _agente([_resp(_call("ok_tool", x="1"), _call("ok_tool", x="2"))])
    assert agent.handle_turn("haz", should_abort=abortar) == ""
    turno = agent.last_turn
    assert turno is not None and turno.via == "cancelled"
    assert [c.outcome for c in turno.calls] == ["ok"]   # la 2ª ya no corrió


def test_los_soft_errors_mcp_salen_del_prefijo_compartido():
    from crotolamo.core.tool_parsing import (
        HARD_ERROR_PREFIXES,
        SOFT_ERROR_PREFIXES,
        is_hard_error,
        is_soft_error,
    )
    from crotolamo.mcp.bridge import _soft

    texto = _soft("demo", "tardó demasiado en responder, patrón.")
    assert texto == "El server MCP 'demo' tardó demasiado en responder, patrón."
    assert texto.startswith(SOFT_ERROR_PREFIXES) and is_soft_error(texto)
    assert not is_hard_error(texto)               # sigue siendo texto para el patrón
    assert not set(SOFT_ERROR_PREFIXES) & set(HARD_ERROR_PREFIXES)
    assert not is_soft_error("Pausada, patrón.")


# --- paso 3: la búsqueda por turno aprende de sus fallos --------------------------


class _Almacen:
    """Doble de mem0 para la búsqueda: cuenta, tarda, se bloquea o falla a pedido."""

    def __init__(self) -> None:
        self.items = {"m1": "mi perro se llama Tletl"}
        self.busquedas = 0
        self.en_curso = 0
        self.max_en_curso = 0
        self.tardar_s = 0.0
        self.fallar: Exception | None = None
        self.bloqueo: threading.Event | None = None
        self.extraidos: list[str] = []
        self.fallar_extraccion: Exception | None = None
        self.intentos: list[str] = []
        self._lock = threading.Lock()

    def buscar(self, query: str, top_k: int, umbral: float) -> list[Recuerdo]:
        with self._lock:
            self.busquedas += 1
            self.en_curso += 1
            self.max_en_curso = max(self.max_en_curso, self.en_curso)
        try:
            if self.bloqueo is not None:
                self.bloqueo.wait(timeout=5.0)
            if self.tardar_s:
                time.sleep(self.tardar_s)
            if self.fallar is not None:
                raise self.fallar
            q = set(query.lower().split())
            return [Recuerdo(k, v, 0.05) for k, v in self.items.items()
                    if q & set(v.lower().split())][:top_k]
        finally:
            with self._lock:
                self.en_curso -= 1

    def extraer(self, usuario: str, respuesta: str) -> list[str]:
        self.intentos.append(usuario)
        if self.fallar_extraccion is not None:
            raise self.fallar_extraccion
        self.extraidos.append(usuario)
        return [usuario]

    def guardar(self, texto: str) -> list[str]:
        return []

    def todas(self) -> list[Recuerdo]:
        return [Recuerdo(k, v) for k, v in self.items.items()]

    def borrar(self, memoria_id: str) -> None:
        self.items.pop(memoria_id, None)


@pytest.fixture
def memoria():
    almacenes: list[Memoria] = []

    def _crear(timeout_s: float = 0.1) -> tuple[Memoria, _Almacen]:
        almacen = _Almacen()
        m = Memoria(MemoriaConfig(enabled=True, timeout_busqueda_s=timeout_s), backend=almacen)
        almacenes.append(m)
        return m, almacen

    yield _crear
    for m in almacenes:
        m.cerrar(timeout_s=1.0)


def _esperar(pred, timeout: float = 3.0) -> bool:
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_con_la_memoria_sana_la_busqueda_por_turno_es_la_de_siempre(memoria):
    m, almacen = memoria(timeout_s=1.0)
    hook = make_memoria_prehook(m)
    for _ in range(5):
        assert hook("mi perro") == (
            "[recuerdos sobre el patrón que vienen al caso:\n- mi perro se llama Tletl]"
            "\n\nmi perro")
    assert almacen.busquedas == 5
    assert m.contexto("mi perro") == "\n".join(f"- {r.texto}" for r in m.buscar("mi perro"))


def test_un_almacen_lento_pausa_la_busqueda_por_turno(memoria, caplog):
    """Almacén 10x más lento que el presupuesto: antes CADA turno esperaba el
    presupuesto entero y las búsquedas vencidas se apilaban en el pool."""
    m, almacen = memoria(timeout_s=0.3)
    almacen.bloqueo = threading.Event()               # el almacén no contesta a tiempo
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        t0 = time.monotonic()
        assert m.contexto("mi perro") == ""           # 1: espera el presupuesto
        assert 0.25 < time.monotonic() - t0 < 1.5
        t1 = time.monotonic()
        assert m.contexto("mi perro") == ""           # 2: la 1ª sigue en curso
        assert m.contexto("mi perro") == ""           # 3: breaker abierto
        assert time.monotonic() - t1 < 0.15           # ninguna esperó el presupuesto
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0)
    assert m.contexto("mi perro") == ""               # 4: sigue en pausa, aunque ya acabó
    assert almacen.busquedas == 1 and almacen.max_en_curso == 1
    assert caplog.text.count("tardó más") == 1
    assert caplog.text.count("sin esperarla") == 1     # un solo WARNING de pausa
    assert m.enabled                                  # el breaker no la marca rota


def test_una_sola_busqueda_por_turno_en_vuelo(memoria):
    m, almacen = memoria(timeout_s=0.3)
    almacen.bloqueo = threading.Event()
    assert m.contexto("mi perro") == ""
    assert _esperar(lambda: almacen.en_curso == 1)
    t0 = time.monotonic()
    assert m.contexto("mi perro") == ""               # no encola otra
    assert time.monotonic() - t0 < 0.15               # ni espera el presupuesto
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0)
    assert almacen.busquedas == 1


def test_tras_la_pausa_se_prueba_otra_vez_y_un_acierto_la_cierra(memoria, monkeypatch, caplog):
    monkeypatch.setattr(memoria_mod, "BREAKER_PAUSA_S", 0.15)
    m, almacen = memoria(timeout_s=0.05)
    almacen.fallar = RuntimeError("chroma: disk I/O error")
    assert m.contexto("mi perro") == "" and m.contexto("mi perro") == ""
    assert almacen.busquedas == 2                     # fallos rápidos: sin single-flight
    assert m.contexto("mi perro") == "" and almacen.busquedas == 2   # en pausa
    time.sleep(0.2)
    almacen.fallar = None
    with caplog.at_level(logging.INFO, logger="crotolamo.core.memoria"):
        assert "Tletl" in m.contexto("mi perro")      # medio abierto: prueba y acierta
    assert "volvió a responder" in caplog.text
    almacen.fallar = RuntimeError("otra vez")
    assert m.contexto("mi perro") == ""               # 1 fallo tras cerrar: no pausa
    almacen.fallar = None
    assert "Tletl" in m.contexto("mi perro")
    assert almacen.busquedas == 5


def test_medio_abierto_un_fallo_tras_la_pausa_la_reabre(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "BREAKER_PAUSA_S", 0.15)
    m, almacen = memoria(timeout_s=0.05)
    almacen.fallar = RuntimeError("boom")
    m.contexto("mi perro")
    m.contexto("mi perro")
    time.sleep(0.2)
    assert m.contexto("mi perro") == "" and almacen.busquedas == 3   # la prueba falla
    assert m.contexto("mi perro") == "" and almacen.busquedas == 3   # y reabre


def test_la_busqueda_explicita_no_la_frena_el_breaker(memoria):
    """Las tools (buscar_recuerdos, olvidar) y la CLI contestan con lo que
    encuentren: siguen con su presupuesto y sin breaker."""
    m, almacen = memoria(timeout_s=0.05)
    almacen.fallar = RuntimeError("boom")
    m.contexto("mi perro")
    m.contexto("mi perro")                            # breaker abierto
    almacen.fallar = None
    assert m.contexto("mi perro") == ""
    assert [r.texto for r in m.buscar("mi perro", timeout_s=1.0)] == ["mi perro se llama Tletl"]
    assert [r.texto for r in m.buscar_estricto("mi perro", timeout_s=1.0)] == [
        "mi perro se llama Tletl"]
    assert m.olvidar("mi perro") is not None


def test_memoria_no_disponible_en_el_turno_la_sigue_marcando_rota(memoria):
    m, almacen = memoria(timeout_s=1.0)
    almacen.fallar = MemoriaNoDisponible("falta mem0")
    assert m.contexto("mi perro") == ""
    assert not m.enabled and m.motivo_no_disponible == "falta mem0"
    assert m.contexto("mi perro") == "" and almacen.busquedas == 1


def test_una_busqueda_vencida_que_no_empezo_ya_no_corre(memoria):
    """future.cancel() al vencer: con el pool ocupado, la búsqueda en cola se
    descarta en vez de correr después para nadie (explícitas y por turno)."""
    m, almacen = memoria(timeout_s=0.05)
    almacen.bloqueo = threading.Event()
    assert m.buscar("mi perro", timeout_s=0.05) == []
    assert m.buscar("mi perro", timeout_s=0.05) == []   # los 2 hilos del pool, ocupados
    assert _esperar(lambda: almacen.en_curso == 2)
    assert m.buscar("mi perro", timeout_s=0.05) == []   # explícita en cola: vence
    assert m.contexto("mi perro") == ""                 # por turno en cola: vence
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0)
    time.sleep(0.1)
    assert almacen.busquedas == 2


def test_cerrar_descarta_lo_que_quedo_en_la_cola_del_pool(memoria):
    m, almacen = memoria(timeout_s=0.05)
    almacen.bloqueo = threading.Event()
    m.buscar("mi perro", timeout_s=0.01)
    m.buscar("mi perro", timeout_s=0.01)
    assert _esperar(lambda: almacen.en_curso == 2)
    en_cola = m._pool.submit(m._buscar_sync, "mi perro", 1)   # nadie la espera
    m.cerrar(timeout_s=0.5)
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0)
    assert en_cola.cancelled() and almacen.busquedas == 2


def test_un_fallo_al_precalentar_cuenta_como_uno(memoria):
    m, almacen = memoria(timeout_s=0.05)
    almacen.fallar = RuntimeError("chroma: disk I/O error")
    m.precalentar()
    assert _esperar(lambda: not m._calentando)
    assert m.contexto("mi perro") == ""               # 2º fallo seguido: pausa
    almacen.fallar = None
    assert m.contexto("mi perro") == "" and almacen.busquedas == 2


def test_mientras_mem0_carga_los_turnos_no_abren_el_breaker(memoria):
    """La 1ª búsqueda carga el modelo de embeddings (4-9 s): que el patrón hable
    dos veces en ese rato no es un almacén enfermo. Con la memoria lista, el
    siguiente turno ya trae sus recuerdos (sin la pausa de 60 s)."""
    m, almacen = memoria(timeout_s=0.05)
    almacen.bloqueo = threading.Event()               # "cargando el modelo"
    m.precalentar()
    assert _esperar(lambda: almacen.en_curso == 1)
    assert m.contexto("mi perro") == ""               # espera el presupuesto
    assert m.contexto("mi perro") == ""               # la anterior sigue: al instante
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0 and not m._calentando)
    assert _esperar(lambda: m._busqueda_turno is not None and m._busqueda_turno.done())
    assert "Tletl" in m.contexto("mi perro")


def test_un_almacen_lento_tras_el_arranque_si_abre_el_breaker(memoria):
    m, almacen = memoria(timeout_s=0.05)
    m.precalentar()
    assert _esperar(lambda: not m._calentando)
    almacen.bloqueo = threading.Event()
    m.contexto("mi perro")
    m.contexto("mi perro")
    almacen.bloqueo.set()
    assert _esperar(lambda: almacen.en_curso == 0)
    antes = almacen.busquedas
    assert m.contexto("mi perro") == "" and almacen.busquedas == antes   # en pausa


# --- paso 4: la extracción en segundo plano también aprende de sus fallos -----------

def _fallar_tres(m: Memoria, almacen: _Almacen) -> None:
    almacen.fallar_extraccion = RuntimeError("groq 429: rate limit")
    for texto in ("uno", "dos", "tres"):
        assert m.extraer_en_fondo(texto, "ok")
    assert _esperar(lambda: len(almacen.intentos) == 3)


def test_tres_extracciones_fallidas_pausan_sin_perder_lo_encolado(memoria, monkeypatch, caplog):
    """Antes: tras cada turno se volvía a pegar al proveedor que contestaba 429 (el
    mismo límite que el del cerebro). Ahora espera; lo que llega se queda en cola."""
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 0.4)
    m, almacen = memoria()
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        _fallar_tres(m, almacen)
        almacen.fallar_extraccion = None
        assert m.extraer_en_fondo("me gusta el pozole", "va")
        assert m.extraer_en_fondo("mi perro se llama Tletl", "va")
        time.sleep(0.15)
        assert almacen.intentos == ["uno", "dos", "tres"]   # en pausa: ni lo intenta
        assert m.flush(timeout_s=3.0)
    assert almacen.extraidos == ["me gusta el pozole", "mi perro se llama Tletl"]   # FIFO
    assert caplog.text.count("groq 429") == 3
    assert caplog.text.count("pauso la extracción") == 1


def test_dos_fallos_no_pausan(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 30.0)
    m, almacen = memoria()
    almacen.fallar_extraccion = RuntimeError("groq 500")
    assert m.extraer_en_fondo("uno", "ok") and m.extraer_en_fondo("dos", "ok")
    assert m.flush(timeout_s=2.0)
    almacen.fallar_extraccion = None
    assert m.extraer_en_fondo("tres", "ok")
    assert m.flush(timeout_s=2.0)
    assert almacen.extraidos == ["tres"]


def test_un_acierto_tras_la_pausa_vuelve_a_contar_desde_cero(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 0.2)
    m, almacen = memoria()
    _fallar_tres(m, almacen)
    almacen.fallar_extraccion = None
    assert m.extraer_en_fondo("cuatro", "ok")
    assert m.flush(timeout_s=3.0) and almacen.extraidos == ["cuatro"]
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 30.0)
    almacen.fallar_extraccion = RuntimeError("groq 500")
    assert m.extraer_en_fondo("cinco", "ok") and m.extraer_en_fondo("seis", "ok")
    assert m.flush(timeout_s=2.0)                       # 2 fallos tras el acierto: sin pausa


def test_medio_abierto_un_fallo_tras_la_pausa_vuelve_a_pausar(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 0.5)
    m, almacen = memoria()
    _fallar_tres(m, almacen)
    assert m.extraer_en_fondo("cuatro", "ok")         # tras la pausa: falla y reabre
    assert _esperar(lambda: len(almacen.intentos) == 4)
    almacen.fallar_extraccion = None
    assert m.extraer_en_fondo("cinco", "ok")
    assert m.flush(timeout_s=0.15) is False             # en la 2ª pausa: flush se rinde
    assert almacen.intentos == ["uno", "dos", "tres", "cuatro"]
    assert m.flush(timeout_s=3.0) and almacen.extraidos == ["cinco"]


def test_cerrar_corta_la_pausa(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    _fallar_tres(m, almacen)
    worker = m._worker
    assert worker is not None and worker.is_alive()
    t0 = time.monotonic()
    m.cerrar(timeout_s=2.0)
    assert time.monotonic() - t0 < 1.0
    assert not worker.is_alive()


def test_memoria_no_disponible_al_extraer_no_cuenta_para_la_pausa(memoria, monkeypatch, caplog):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    almacen.fallar_extraccion = MemoriaNoDisponible("falta mem0")
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert m.extraer_en_fondo("uno", "ok")
        assert m.flush(timeout_s=2.0)
    assert not m.enabled and "pauso la extracción" not in caplog.text


# --- paso 5: las tools de solo lectura de un mismo mensaje corren a la vez ---------
# Antes el loop corría las tools de un mensaje una tras otra: tres lecturas web de
# 0.15 s costaban 0.45 s. Ahora, si TODAS son de solo lectura (Tool.parallel_safe),
# el guard las deja sin confirmar y ninguna repite un fallo, corren a la vez en el
# pool del agente; el historial y el registro del turno quedan en el orden pedido.


class _Medidor:
    """Cuántas tools corren a la vez (pico) y la traza de lo que pasó, en orden."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dentro = 0
        self.pico = 0
        self.traza: list[tuple[str, str]] = []
        self.carriles: dict[str, int] = {}
        self.pico_carril = 0

    def tool(self, nombre: str, dormir: float = 0.0, resultado: str | None = None,
             rompe: Exception | None = None, carril: str | None = None):
        def f(x: str = "", y: str = "", **otros) -> str:
            with self.lock:
                self.dentro += 1
                self.pico = max(self.pico, self.dentro)
                self.traza.append(("corre", nombre))
                if carril is not None:
                    self.carriles[carril] = self.carriles.get(carril, 0) + 1
                    self.pico_carril = max(self.pico_carril, self.carriles[carril])
            try:
                time.sleep(dormir)
                if rompe is not None:
                    raise rompe
                return resultado if resultado is not None else f"{nombre}: {x}{y}{otros or ''}"
            finally:
                with self.lock:
                    self.dentro -= 1
                    if carril is not None:
                        self.carriles[carril] -= 1
        return f


def _reg_lectura(m: _Medidor, paralelas: bool = True) -> Registry:
    from crotolamo.core.budget import TOOL_RESULT_CAP

    reg = Registry()

    def lectora(nombre, dormir=0.0, **kw):
        reg.register(Tool(nombre, m.tool(nombre, dormir, **kw), nombre, _P,
                          parallel_safe=paralelas))

    lectora("lee_a", 0.15)                                   # la más lenta va primero
    lectora("lee_b", 0.05)
    lectora("lee_c", 0.10)
    lectora("revienta", 0.02, rompe=RuntimeError("se cayó el disco"))
    lectora("enorme", resultado="x" * (TOOL_RESULT_CAP + 500))
    reg.register(Tool("requiere", lambda x: f"req {x}", "requiere",
                      {"type": "object", "properties": {"x": {"type": "string"}},
                       "required": ["x"]}, parallel_safe=paralelas))
    reg.register(Tool("mcp_lee", m.tool("mcp_lee", 0.03, resultado=(
        "El server MCP 'demo' reportó un fallo, patrón: boom")), "mcp", _P,
        strict_args=False, parallel_safe=paralelas))
    reg.register(Tool("escribe", m.tool("escribe"), "escribe", _P))   # sin el flag
    reg.register(Tool("delicada_lee", m.tool("delicada_lee"), "d", _P, safe=False,
                      parallel_safe=True))
    reg.register(Tool("prohibida", m.tool("prohibida"), "p", _P, parallel_safe=True))
    return reg


def _agente_lectura(m: _Medidor, seq, paralelas: bool = True, **kw) -> ToolAgent:
    kw.setdefault("pre_hooks", [])
    kw.setdefault("post_hooks", [])
    return ToolAgent(_LLM(seq), Conversation("SYS"), registry=_reg_lectura(m, paralelas),
                     guard=kw.pop("guard", _Guard()), **kw)


def _vuelta(agent: ToolAgent) -> dict:
    """Todo lo que deja un turno: lo que va por el cable, la respuesta y el registro."""
    import json

    turno = agent.last_turn
    assert turno is not None
    return {
        "wire": json.dumps(agent.conversation.to_messages(), ensure_ascii=False),
        "calls": [(c.name, c.key, c.outcome, c.result) for c in turno.calls],
        "via": turno.via, "llm_calls": turno.llm_calls,
    }


_LOTE = [
    _call("lee_a", x="1"),
    _call("lee_b", x="2", z="alucinado"),        # strict_args lo filtra igual
    _call("revienta", x="3"),
    _call("enorme"),
    _call("requiere"),                           # falta un argumento requerido
    _call("mcp_lee", x="4", extra={"n": 1}),     # strict_args=False: todo pasa
    _call("lee_c", y="5"),
]


def test_en_paralelo_el_historial_es_byte_a_byte_el_de_en_serie(caplog):
    """La MISMA vuelta en serie (sin el flag) y en paralelo: mismo cable, mismos
    resultados (reventón, recorte, argumentos inválidos, soft error MCP) y mismo
    registro, en el orden en que el modelo las pidió aunque terminen en otro."""
    vueltas = {}
    medidores = {}
    for paralelas in (False, True):
        m = _Medidor()
        agent = _agente_lectura(m, [_resp(*_LOTE), ChatResponse(content="Listo todo.")],
                                paralelas=paralelas)
        with caplog.at_level(logging.ERROR):
            assert agent.handle_turn("léeme todo") == "Listo todo."
        vueltas[paralelas], medidores[paralelas] = _vuelta(agent), m
        agent.close()
    assert vueltas[True] == vueltas[False]
    assert medidores[False].pico == 1
    assert medidores[True].pico >= 2
    # Orden pedido, no el de llegada (lee_a es la más lenta y va primero).
    resultados = [c[3] for c in vueltas[True]["calls"]]
    assert resultados[0] == "lee_a: 1" and resultados[1] == "lee_b: 2"
    assert resultados[2] == "La tool 'revienta' reventó, patrón: se cayó el disco"
    assert resultados[3].endswith("chars en total]")
    assert resultados[4].startswith("Argumentos inválidos para 'requiere'")
    assert [c[2] for c in vueltas[True]["calls"]] == [
        "ok", "ok", "hard_error", "ok", "hard_error", "soft_error", "ok"]


def test_tres_lecturas_cuestan_una(caplog):
    m = _Medidor()
    agent = _agente_lectura(m, [
        _resp(_call("lee_a", x="1"), _call("lee_b", x="2"), _call("lee_c", x="3")),
        ChatResponse(content="Ahí va."),
    ])
    for nombre in ("lee_a", "lee_b", "lee_c"):
        agent.registry.register(Tool(nombre, m.tool(nombre, 0.2), nombre, _P,
                                     parallel_safe=True))
    t0 = time.monotonic()
    assert agent.handle_turn("lee las tres") == "Ahí va."
    transcurrido = time.monotonic() - t0
    assert m.pico == 3
    assert transcurrido < 0.55            # en serie serían 0.6 s como mínimo
    tool_msgs = [(msg.name, msg.content) for msg in agent.conversation.history
                 if msg.role == "tool"]
    assert tool_msgs == [("lee_a", "lee_a: 1"), ("lee_b", "lee_b: 2"), ("lee_c", "lee_c: 3")]
    agent.close()


@pytest.mark.parametrize("intrusa", [
    _call("escribe", x="1"),          # una tool sin el flag (puede cambiar algo)
    _call("delicada_lee", x="1"),     # pide confirmación
    _call("prohibida", x="1"),        # el guard la bloquea
    _call("no_existe", x="1"),        # el modelo inventó el nombre
])
def test_un_lote_con_algo_que_no_es_lectura_libre_corre_en_serie(intrusa):
    m = _Medidor()
    confirmaciones: list[str] = []

    def confirmar(reason):
        with m.lock:
            m.traza.append(("confirma", reason))
            confirmaciones.append(reason)
        return True

    agent = _agente_lectura(m, [
        _resp(_call("lee_a", x="1"), intrusa, _call("lee_b", x="2")),
        ChatResponse(content="Hecho."),
    ], confirm_fn=confirmar)
    assert agent.handle_turn("haz eso") == "Hecho."
    assert m.pico == 1
    assert agent._pool is None                 # ni se crearon hilos
    if intrusa["name"] == "delicada_lee":
        # La confirmación, en su sitio de la serie: tras la 1ª y antes de la 3ª.
        assert m.traza == [("corre", "lee_a"), ("confirma", "¿Le hago la cosa delicada, patrón?"),
                           ("corre", "delicada_lee"), ("corre", "lee_b")]
    else:
        assert confirmaciones == []


def test_un_guard_que_revienta_en_el_lote_cae_a_la_serie_y_anota_el_fallo(caplog):
    class _GuardBoom:
        def check(self, tool, arguments):
            raise ValueError("ruta con \\0")

    m = _Medidor()
    agent = _agente_lectura(m, [
        _resp(_call("lee_a", x="1"), _call("lee_b", x="2")),
        ChatResponse(content="No pude, patrón."),
    ], guard=_GuardBoom())
    with caplog.at_level(logging.ERROR):
        assert agent.handle_turn("lee") == "No pude, patrón."
    assert m.pico == 0
    roles = [msg.role for msg in agent.conversation.history]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    tool_msgs = [msg.content for msg in agent.conversation.history if msg.role == "tool"]
    assert tool_msgs == ["La tool 'lee_a' reventó, patrón: ruta con \\0",
                         "La tool 'lee_b' reventó, patrón: ruta con \\0"]
    assert "reventó fuera del registry" in caplog.text


def test_el_pool_tiene_tope_se_crea_al_primer_lote_se_reusa_y_close_lo_apaga():
    m = _Medidor()
    seis = [_call(n, x=str(i)) for i, n in enumerate(
        ["lee_a", "lee_b", "lee_c", "lee_a", "lee_b", "lee_c"])]
    agent = _agente_lectura(m, [_resp(*seis), ChatResponse(content="uno"),
                                _resp(*seis), ChatResponse(content="dos"),
                                _resp(*seis), ChatResponse(content="tres")])
    for nombre in ("lee_a", "lee_b", "lee_c"):
        agent.registry.register(Tool(nombre, m.tool(nombre, 0.2), nombre, _P,
                                     parallel_safe=True))
    assert agent._pool is None                      # perezoso: sin lote, sin hilos
    assert agent.handle_turn("seis") == "uno"
    assert m.pico == 4                              # acotado
    pool = agent._pool
    assert pool is not None
    hilos = set(pool._threads)
    assert 1 <= len(hilos) <= 4
    assert agent.handle_turn("otra vez") == "dos"
    assert agent._pool is pool and set(pool._threads) == hilos   # el mismo, sin hilos nuevos
    agent.close()
    agent.close()                                   # idempotente
    for hilo in hilos:
        hilo.join(timeout=2.0)
        assert not hilo.is_alive()
    # Tras close() no se crean hilos: el lote corre en serie, con el mismo resultado.
    m.pico = 0
    assert agent.handle_turn("y otra") == "tres"
    assert m.pico == 1 and agent._pool is None
    tool_msgs = [msg.content for msg in agent.conversation.history if msg.role == "tool"]
    assert tool_msgs[-6:] == tool_msgs[:6]


def test_un_lote_cancelado_antes_de_empezar_no_corre_ninguna():
    m = _Medidor()
    pasos = {"n": 0}

    def abortar():
        pasos["n"] += 1
        return pasos["n"] > 1        # la 1ª consulta (antes del LLM) deja seguir

    agent = _agente_lectura(m, [_resp(_call("lee_a", x="1"), _call("lee_b", x="2"))])
    assert agent.handle_turn("lee", should_abort=abortar) == ""
    assert m.traza == []
    tool_msgs = [msg.content for msg in agent.conversation.history if msg.role == "tool"]
    assert tool_msgs == ["Cancelado: el patrón interrumpió."] * 2
    assert agent.conversation.history[-1].content == "(Turno interrumpido por el patrón.)"
    assert agent.last_turn is not None and agent.last_turn.via == "cancelled"


def test_las_de_un_mismo_carril_van_una_tras_otra_y_en_orden():
    """Las tools MCP de un server comparten pipe y contador de strikes: un server
    que atiende de una en una se comería el timeout de la 2ª si se mandan a la
    vez. Van en su carril, en orden; lo demás corre al lado."""
    m = _Medidor()
    reg = Registry()
    for nombre in ("mcp_uno", "mcp_dos"):
        reg.register(Tool(nombre, m.tool(nombre, 0.2, carril="srv"), nombre, _P,
                          strict_args=False, parallel_safe=True, parallel_lane="mcp:srv"))
    reg.register(Tool("lee_a", m.tool("lee_a", 0.3), "a", _P, parallel_safe=True))
    agent = ToolAgent(_LLM([_resp(_call("mcp_uno", x="1"), _call("mcp_dos", x="2"),
                                  _call("lee_a", x="3")),
                            ChatResponse(content="ok")]),
                      Conversation("SYS"), registry=reg, guard=_Guard(),
                      pre_hooks=[], post_hooks=[])
    assert agent.handle_turn("varias") == "ok"
    assert m.pico_carril == 1                     # nunca dos del mismo server a la vez
    assert m.pico == 2                            # pero lee_a corre al lado
    corren = [n for e, n in m.traza if e == "corre" and n.startswith("mcp_")]
    assert corren == ["mcp_uno", "mcp_dos"]
    agent.close()

    # Todo en UN carril: no hay nada que paralelizar, ni se crea el pool.
    m2 = _Medidor()
    reg2 = Registry()
    for nombre in ("mcp_uno", "mcp_dos"):
        reg2.register(Tool(nombre, m2.tool(nombre, carril="srv"), nombre, _P,
                           strict_args=False, parallel_safe=True, parallel_lane="mcp:srv"))
    agent2 = ToolAgent(_LLM([_resp(_call("mcp_uno"), _call("mcp_dos")),
                             ChatResponse(content="ok")]),
                       Conversation("SYS"), registry=reg2, guard=_Guard(),
                       pre_hooks=[], post_hooks=[])
    assert agent2.handle_turn("varias") == "ok"
    assert agent2._pool is None and m2.pico == 1


def test_mcp_solo_readonlyhint_cuenta_como_lectura():
    from crotolamo.mcp import bridge

    server = bridge.MCPServerConfig(name="srv", command=["true"], prefix="srv")
    crudas = [
        {"name": "lee", "annotations": {"readOnlyHint": True}},
        {"name": "no_destruye", "annotations": {"destructiveHint": False}},
        {"name": "borra", "annotations": {"destructiveHint": True}},
        {"name": "sin_hints"},
        {"name": "texto", "annotations": {"readOnlyHint": "true"}},     # no es bool
    ]
    tools = {t.name: t for t in bridge._build_tools(Registry(), server, crudas)}
    assert {n: t.parallel_safe for n, t in tools.items()} == {
        "mcp_srv_lee": True, "mcp_srv_no_destruye": False, "mcp_srv_borra": False,
        "mcp_srv_sin_hints": False, "mcp_srv_texto": False,
    }
    assert {t.parallel_lane for t in tools.values()} == {"mcp:srv"}
    # destructiveHint=False sigue sin pedir confirmación: eso no cambia.
    assert tools["mcp_srv_no_destruye"].safe is True


def test_las_tools_de_fabrica_que_solo_leen_llevan_el_flag():
    from crotolamo.tools import default_registry
    from crotolamo.tools.memoria import memoria_tools

    reg = default_registry()
    con_flag = {n for n in reg.names() if getattr(reg.get(n), "parallel_safe", False)}
    assert con_flag == {
        "fetch_web_results", "read_page", "home_state", "camera_events", "camera_snapshot",
        "music_now", "disk_usage", "ram_usage", "list_processes", "system_status",
        "list_windows", "app_status", "read_file", "list_dir", "search_files",
        "list_projects", "analyze_project", "list_project_tree", "read_project_file",
        "find_in_project", "recall_facts", "search_facts", "list_shortcuts",
    }
    # Las que cambian algo (abrir, escribir, mover, la música, las luces, atajos,
    # ventanas, hechos) NUNCA: aunque sean safe=True, no son de solo lectura.
    for nombre in ("open_app", "open_url", "music_control", "light_control", "write_file",
                   "make_dir", "move_file", "delete_file", "create_note", "launch_project",
                   "run_shortcut", "learn_shortcut", "remember_fact", "forget_fact",
                   "focus_window", "close_window", "search_web"):
        tool = reg.get(nombre)
        assert tool is not None and tool.parallel_safe is False, nombre
    assert {t.name: t.parallel_safe for t in memoria_tools()} == {
        "recordar_de_mi": False, "buscar_recuerdos": True, "olvidar_recuerdo": False}


@pytest.mark.parametrize("entradas", [["salir"], []])     # "salir" y Ctrl+D (EOF)
def test_el_shell_apaga_los_hilos_del_agente_al_salir(monkeypatch, entradas):
    import builtins

    import interfaces.shell as shell

    m = _Medidor()
    agent = _agente_lectura(m, [])
    cerrado: list[bool] = []
    real = agent.close
    monkeypatch.setattr(agent, "close", lambda: cerrado.append(True) or real())
    monkeypatch.setattr(shell, "build_agent", lambda confirm_fn=None: (agent, None))
    cola = iter(entradas)

    def leer(prompt=""):
        try:
            return next(cola)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", leer)
    assert shell.run_shell([]) == 0
    assert cerrado == [True] and agent._pool_closed


# --- paso 6: política de repetición ([llm].stop_on_repeated_failure, apagada) -------
# El loop solo salía por "sin tool-calls" o por max_iterations, sin memoria de lo
# que ya falló: un modelo que repite una llamada rota quemaba las 6 iteraciones
# (~110 KB con GLM) y acababa en la frase enlatada; en voz concurrente el aviso de
# "dímela por el shell" se decía en cada reintento. Con la clave encendida, la
# repetición de un fallo no se vuelve a correr ni a preguntar y la siguiente
# petición va sin tools: el modelo lo explica con sus palabras.

_CANNED = "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
_EXPLICA = "No me salió, patrón: falló la herramienta."


class _Terco:
    """LLM que pide SIEMPRE lo mismo mientras tenga tools a la vista; sin tools,
    explica (o falla, si `final` es una excepción). Anota las tools de cada petición."""

    def __init__(self, *calls, final: str | Exception = _EXPLICA, distintas: bool = False):
        self.calls = list(calls)
        self.final = final
        self.distintas = distintas
        self.tools: list = []

    def chat(self, messages, tools=None):
        self.tools.append(tools)
        if not tools:
            if isinstance(self.final, Exception):
                raise self.final
            return ChatResponse(content=self.final)
        if self.distintas:                    # una llamada distinta en cada iteración
            n = len(self.tools)
            return _resp(*[{**c, "arguments": {"x": str(n)}} for c in self.calls])
        return _resp(*self.calls)

    def chat_stream(self, messages, tools=None, on_token=None):
        response = self.chat(messages, tools)
        if on_token is not None and response.content and not response.tool_calls:
            on_token(response.content)
        return response


def _terco(llm: _Terco, stop: bool, **kw) -> ToolAgent:
    ejecutadas: list[str] = []
    reg = _registry()
    for nombre in ("rompe", "soft", "ok_tool", "prohibida", "delicada"):
        tool = reg.get(nombre)
        assert tool is not None
        func = tool.func

        def contada(*a, _func=func, _nombre=nombre, **k):
            ejecutadas.append(_nombre)
            return _func(*a, **k)

        tool.func = contada
    kw.setdefault("pre_hooks", [])
    kw.setdefault("post_hooks", [])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg, guard=_Guard(),
                      stop_on_repeated_failure=stop, **kw)
    agent.ejecutadas = ejecutadas  # type: ignore[attr-defined]
    return agent


@pytest.mark.parametrize("call,llamadas,ejecuciones", [
    (_call("rompe", x="1"), 4, 2),        # un fallo duro: se permite UN reintento
    (_call("soft", x="1"), 4, 2),         # el aviso de fallo de un server MCP, igual
    (_call("no_existe", x="1"), 3, 0),    # determinista: a la primera repetición
    (_call("prohibida", x="1"), 3, 0),    # el guard la bloquea siempre
])
def test_con_la_politica_un_fallo_repetido_no_quema_las_iteraciones(call, llamadas, ejecuciones):
    llm = _Terco(call)
    agent = _terco(llm, stop=True)
    assert agent.handle_turn("haz la cosa") == _EXPLICA
    assert len(llm.tools) == llamadas
    assert llm.tools[-1] == [] and all(llm.tools[:-1])      # solo la última sin tools
    assert len(agent.ejecutadas) == ejecuciones              # type: ignore[attr-defined]
    turno = agent.last_turn
    assert turno is not None and turno.llm_calls == llamadas
    # La repetición queda en el registro, con el resultado de antes, y ese mismo
    # texto es el que vuelve al modelo.
    assert turno.calls[-1].outcome == "repeated"
    assert turno.calls[-1].result == turno.calls[-2].result
    tool_msgs = [m.content for m in agent.conversation.history if m.role == "tool"]
    assert tool_msgs == [c.result for c in turno.calls]
    assert agent.conversation.history[-1].content == _EXPLICA


@pytest.mark.parametrize("call", [_call("rompe", x="1"), _call("no_existe", x="1"),
                                  _call("prohibida", x="1"), _call("soft", x="1")])
def test_sin_la_politica_todo_sigue_como_siempre(call):
    llm = _Terco(call)
    agent = _terco(llm, stop=False)
    assert agent.handle_turn("haz la cosa") == _CANNED
    assert len(llm.tools) == 6 and all(llm.tools)
    esperado = 6 if call["name"] in ("rompe", "soft") else 0
    assert len(agent.ejecutadas) == esperado                 # type: ignore[attr-defined]
    turno = agent.last_turn
    assert turno is not None and turno.via == "exhausted"
    assert all(c.outcome != "repeated" for c in turno.calls)


def test_una_confirmacion_negada_no_se_vuelve_a_preguntar():
    preguntas: list[str] = []
    llm = _Terco(_call("delicada", x="1"))
    agent = _terco(llm, stop=True, confirm_fn=lambda r: preguntas.append(r) or False)
    assert agent.handle_turn("haz la delicada") == _EXPLICA
    assert preguntas == ["¿Le hago la cosa delicada, patrón?"]
    assert len(llm.tools) == 3 and agent.ejecutadas == []    # type: ignore[attr-defined]
    tool_msgs = [m.content for m in agent.conversation.history if m.role == "tool"]
    assert tool_msgs == ["Cancelado por el patrón."] * 2


def test_otra_llamada_tras_un_fallo_si_corre():
    """Solo se frena la llamada IDÉNTICA (misma tool y argumentos)."""
    llm = _LLM([_resp(_call("no_existe", x="1")), _resp(_call("ok_tool", x="2")),
                _resp(_call("no_existe", x="2")), ChatResponse(content="Ya.")])
    agent = _agente([], stop_on_repeated_failure=True)
    agent.llm = llm
    assert agent.handle_turn("varias") == "Ya."
    turno = agent.last_turn
    assert turno is not None
    assert [c.outcome for c in turno.calls] == ["unknown_tool", "ok", "unknown_tool"]


def test_la_repeticion_dentro_del_mismo_mensaje_tambien_se_frena():
    llm = _Terco(_call("no_existe", x="1"), _call("no_existe", x="1"))
    agent = _terco(llm, stop=True)
    assert agent.handle_turn("dos veces") == _EXPLICA
    assert len(llm.tools) == 2 and llm.tools[-1] == []
    turno = agent.last_turn
    assert turno is not None
    assert [c.outcome for c in turno.calls] == ["unknown_tool", "repeated"]


def test_al_agotar_las_iteraciones_una_ultima_peticion_sin_tools():
    llm = _Terco(_call("ok_tool"), distintas=True, final="Hice seis cosas y me faltó una.")
    agent = _terco(llm, stop=True)
    assert agent.handle_turn("haz mil cosas") == "Hice seis cosas y me faltó una."
    assert len(llm.tools) == 7 and llm.tools[-1] == [] and all(llm.tools[:6])
    assert len(agent.ejecutadas) == 6                        # type: ignore[attr-defined]
    turno = agent.last_turn
    assert turno is not None and turno.via == "exhausted" and turno.llm_calls == 7
    assert agent.conversation.history[-1].content == "Hice seis cosas y me faltó una."


@pytest.mark.parametrize("on_token", [False, True])
def test_si_la_ultima_peticion_falla_queda_la_frase_de_siempre(on_token):
    llm = _Terco(_call("ok_tool"), distintas=True, final=LLMError("se cayó la nube"))
    agent = _terco(llm, stop=True)
    oido: list[str] = []
    reply = agent.handle_turn("haz mil cosas", on_token=oido.append if on_token else None)
    assert reply == _CANNED
    assert len(llm.tools) == 7
    assert agent.conversation.history[-1].content == _CANNED
    if on_token:
        assert "".join(oido) == _CANNED


def test_la_explicacion_sin_tools_se_oye_en_streaming_y_pasa_los_post_hooks():
    llm = _Terco(_call("no_existe", x="1"))
    agent = _terco(llm, stop=True, post_hooks=[lambda t: t + " (limpio)"])
    oido: list[str] = []
    assert agent.handle_turn("haz", on_token=oido.append) == _EXPLICA + " (limpio)"
    assert "".join(oido).startswith(_EXPLICA)


def test_un_lote_en_paralelo_con_una_repeticion_va_en_serie_y_la_frena():
    m = _Medidor()
    agent = _agente_lectura(m, [
        _resp(_call("revienta", x="1")),
        _resp(_call("revienta", x="1")),
        _resp(_call("revienta", x="1"), _call("lee_a", x="2")),
        ChatResponse(content="El disco no responde, patrón."),
    ], stop_on_repeated_failure=True)
    assert agent.handle_turn("lee") == "El disco no responde, patrón."
    turno = agent.last_turn
    assert turno is not None
    assert [(c.name, c.outcome) for c in turno.calls] == [
        ("revienta", "hard_error"), ("revienta", "hard_error"),
        ("revienta", "repeated"), ("lee_a", "ok")]
    assert [n for e, n in m.traza if n == "revienta"] == ["revienta", "revienta"]
    agent.close()


def test_build_agent_lee_la_clave_y_de_fabrica_va_apagada(monkeypatch):
    import interfaces.shell as shell
    from crotolamo import settings as settings_mod

    settings = settings_mod.get_settings()
    llm_cfg = settings.raw.setdefault("llm", {})
    monkeypatch.setitem(llm_cfg, "backend", "ollama")
    monkeypatch.delitem(llm_cfg, "stop_on_repeated_failure", raising=False)
    monkeypatch.setattr(shell, "build_llm", lambda s: _LLM([]))
    agent, _ = shell.build_agent(confirm_fn=lambda r: False)
    assert isinstance(agent, ToolAgent) and agent.stop_on_repeated_failure is False
    monkeypatch.setitem(llm_cfg, "stop_on_repeated_failure", True)
    agent, _ = shell.build_agent(confirm_fn=lambda r: False)
    assert isinstance(agent, ToolAgent) and agent.stop_on_repeated_failure is True


def test_la_config_de_fabrica_documenta_stop_on_repeated_failure_apagado():
    import re
    import tomllib
    from pathlib import Path

    texto = (Path(__file__).resolve().parents[1] / "config" / "crotolamo.toml").read_text(
        encoding="utf-8")
    llm = re.split(r"^\[llm\]$", texto, maxsplit=1, flags=re.M)[1]
    llm = re.split(r"^\[", llm, maxsplit=1, flags=re.M)[0]
    assert re.search(r"^#\s*stop_on_repeated_failure\s*=\s*false\b", llm, re.M)
    assert not tomllib.loads(texto)["llm"].get("stop_on_repeated_failure", False)


# --- revisión: hilos daemon, el carril de la libreta y la repetición dentro del lote --
# Los hilos del lote en paralelo y los de la búsqueda de la memoria eran de un
# ThreadPoolExecutor: no son daemon y concurrent.futures los espera al salir del
# intérprete (aun tras shutdown(wait=False)). En serie las tools corrían en el hilo
# del agente y la salida no las esperaba; con el pool, Ctrl+C en el shell o la
# salida del listener esperaban a la tool más lenta del lote, y una búsqueda de mem0
# colgada no dejaba salir nunca. Ahora son DaemonPool (core/turn.py).

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_RAIZ = Path(__file__).resolve().parents[1]

# Un lote de dos lecturas de 6 s en paralelo; el proceso sale en cuanto las dos
# están corriendo. MODO: "daemon" (el turno en un hilo daemon, como BrainThread, y
# el hilo principal sale como run_listen cuando VoiceLoop.run vuelve), "cerrar"
# (igual, con agent.close() antes) o "ctrlc" (el turno en el hilo principal, como
# run_shell, y llega SIGINT).
_SALIDA_CON_LOTE = r'''
import os, signal, sys, threading, time
from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Decision
from crotolamo.tools.base import Registry, Tool

MODO = sys.argv[1]
P = {"type": "object", "properties": {"x": {"type": "string"}}}
corriendo = []
dos = threading.Event()
lock = threading.Lock()

def lenta(x=""):
    with lock:
        corriendo.append(x)
        if len(corriendo) == 2:
            dos.set()
    time.sleep(6.0)
    return "ok " + x

class LLM:
    n = 0
    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            calls = [{"name": "lee_a", "arguments": {"x": "1"}},
                     {"name": "lee_b", "arguments": {"x": "2"}}]
            return ChatResponse(content="", tool_calls=calls,
                                raw_message={"tool_calls": [{"function": c} for c in calls]})
        return ChatResponse(content="listo")

class Guard:
    def check(self, tool, arguments):
        return Decision.ok()

reg = Registry()
for n in ("lee_a", "lee_b"):
    reg.register(Tool(n, lenta, n, P, parallel_safe=True))
agent = ToolAgent(LLM(), Conversation("S"), registry=reg, guard=Guard(),
                  pre_hooks=[], post_hooks=[])
if MODO == "ctrlc":
    principal = threading.main_thread().ident

    def esperando_el_lote():
        # El hilo principal ya está DENTRO de la espera del lote (Future.result ->
        # Condition.wait), no a punto de entrar: una señal atendida justo antes de
        # que se bloquee en el lock no lo despierta (CPython la ve al soltarse el
        # lock, es decir, cuando acaban las tools: 6 s), en serie igual.
        marco = sys._current_frames().get(principal)
        nombres = []
        while marco is not None:
            nombres.append(marco.f_code.co_name)
            marco = marco.f_back
        return "_run_parallel" in nombres and nombres[0] == "wait"

    def avisar():
        dos.wait(5.0)
        limite = time.monotonic() + 5.0
        while not esperando_el_lote() and time.monotonic() < limite:
            time.sleep(0.01)
        time.sleep(0.2)                     # del Condition.wait al lock bloqueado
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=avisar, daemon=True).start()
    try:
        agent.handle_turn("lee")
    except KeyboardInterrupt:
        print("lote en paralelo:", len(corriendo), flush=True)
else:
    threading.Thread(target=agent.handle_turn, args=("lee",), daemon=True).start()
    assert dos.wait(5.0)
    if MODO == "cerrar":
        agent.close()
    print("lote en paralelo:", len(corriendo), flush=True)
'''


def _correr_y_medir(codigo: str, *args: str, timeout: float = 30.0) -> tuple[float, str]:
    env = {**os.environ, "PYTHONPATH": str(_RAIZ)}
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", codigo, *args], cwd=_RAIZ, env=env,
                          capture_output=True, text=True, timeout=timeout)
    transcurrido = time.monotonic() - t0
    assert proc.returncode in (0, 130), proc.stderr[-2000:]
    return transcurrido, proc.stdout


@pytest.mark.parametrize("modo", ["daemon", "cerrar", "ctrlc"])
def test_la_salida_no_espera_a_las_tools_de_un_lote_en_paralelo(modo):
    """exit_delay.py / ctrl_c_shell.py de la revisión: con el pool, 4.1 s (lo que
    tardaba la tool) en vez de 0.6 s. En serie la salida nunca esperó a la tool."""
    transcurrido, salida = _correr_y_medir(_SALIDA_CON_LOTE, modo)
    assert "lote en paralelo: 2" in salida          # las dos corrían a la vez al salir
    assert transcurrido < 4.0, transcurrido         # las tools tardan 6 s


def test_los_hilos_de_las_tools_son_daemon():
    m = _Medidor()
    agent = _agente_lectura(m, [_resp(_call("lee_a", x="1"), _call("lee_b", x="2")),
                                ChatResponse(content="ok")])
    assert agent.handle_turn("lee") == "ok"
    pool = agent._pool
    assert pool is not None and pool._threads
    assert all(hilo.daemon for hilo in pool._threads)
    assert {hilo.name.rsplit("_", 1)[0] for hilo in pool._threads} == {"ToolAgent-tool"}
    agent.close()


# Un almacén cuya búsqueda no vuelve nunca (mem0/chroma colgado): la búsqueda por
# turno se rinde a tiempo y el breaker la pausa, pero el hilo seguía vivo y, al no
# ser daemon, el intérprete lo esperaba para siempre, con cerrar() o sin él.
_SALIDA_CON_BUSQUEDA_COLGADA = r'''
import sys, threading
from crotolamo.core.memoria import Memoria, MemoriaConfig, make_memoria_prehook

class Colgado:
    def buscar(self, q, k, u):
        threading.Event().wait()
    def extraer(self, u, r):
        return []

m = Memoria(MemoriaConfig(enabled=True, timeout_busqueda_s=0.2), backend=Colgado())
hook = make_memoria_prehook(m)
for texto in ("me gusta el café", "y el pozole", "cuéntame algo"):
    hook(texto)
print("hilos:", sorted({t.daemon for t in threading.enumerate()
                        if t.name.startswith("Memoria")}), flush=True)
if sys.argv[1:] == ["cerrar"]:
    m.cerrar(timeout_s=1.0)
print("fin", flush=True)
'''


@pytest.mark.parametrize("cerrar", [False, True])
def test_una_busqueda_de_memoria_colgada_no_impide_salir(cerrar):
    """memoria_exit_hang.py de la revisión (memoria-persistence#8): el proceso no
    salía nunca (timeout 8 s, rc=124), con cerrar() y sin él."""
    args = ["cerrar"] if cerrar else []
    transcurrido, salida = _correr_y_medir(_SALIDA_CON_BUSQUEDA_COLGADA, *args, timeout=10.0)
    assert "hilos: [True]" in salida and "fin" in salida
    assert transcurrido < 4.0, transcurrido


# --- DaemonPool: el contrato que se usaba de ThreadPoolExecutor -----------------------

def test_daemon_pool_reusa_sus_hilos_con_tope_y_devuelve_futures():
    from concurrent.futures import Future

    from crotolamo.core.turn import DaemonPool

    pool = DaemonPool(2, "Prueba")
    dentro = {"n": 0, "pico": 0}
    lock = threading.Lock()

    def trabajo(i):
        with lock:
            dentro["n"] += 1
            dentro["pico"] = max(dentro["pico"], dentro["n"])
        time.sleep(0.05)
        with lock:
            dentro["n"] -= 1
        return i * 10

    futuros = [pool.submit(trabajo, i) for i in range(5)]
    assert all(isinstance(f, Future) for f in futuros)
    assert [f.result(timeout=5.0) for f in futuros] == [0, 10, 20, 30, 40]
    assert dentro["pico"] == 2 and len(pool._threads) == 2
    assert all(h.daemon and h.name.startswith("Prueba_") for h in pool._threads)
    hilos = set(pool._threads)
    assert pool.submit(trabajo, 7).result(timeout=5.0) == 70
    assert set(pool._threads) == hilos                  # sin hilos nuevos
    pool.shutdown()
    pool.shutdown()                                     # idempotente
    for hilo in hilos:
        hilo.join(timeout=2.0)
        assert not hilo.is_alive()
    with pytest.raises(RuntimeError):
        pool.submit(trabajo, 1)


def test_daemon_pool_la_excepcion_va_al_future_y_el_hilo_sigue():
    from crotolamo.core.turn import DaemonPool

    pool = DaemonPool(1, "Prueba")

    def rompe():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        pool.submit(rompe).result(timeout=5.0)
    assert pool.submit(lambda: "sigue").result(timeout=5.0) == "sigue"
    pool.shutdown()


def test_daemon_pool_lo_encolado_corre_salvo_cancel_futures():
    from crotolamo.core.turn import DaemonPool

    for cancelar in (False, True):
        pool = DaemonPool(1, "Prueba")
        suelta = threading.Event()
        corridos: list[str] = []
        primero = pool.submit(lambda: suelta.wait(5.0) and corridos.append("1"))
        en_cola = pool.submit(lambda: corridos.append("2"))
        pool.shutdown(cancel_futures=cancelar)
        suelta.set()
        primero.result(timeout=5.0)
        hilo = next(iter(pool._threads))
        hilo.join(timeout=2.0)
        assert not hilo.is_alive()
        if cancelar:
            assert en_cola.cancelled() and corridos == ["1"]
        else:
            assert en_cola.result(timeout=1.0) is None and corridos == ["1", "2"]


def test_daemon_pool_un_segundo_shutdown_con_cancel_no_deja_hilos_sin_salida():
    """cancel_futures vacía la cola: los None de un shutdown anterior vuelven a
    ella (si no, el hilo ocupado se quedaba esperando trabajo para siempre)."""
    from crotolamo.core.turn import DaemonPool

    pool = DaemonPool(2, "Prueba")
    suelta = threading.Event()
    ocupados = [pool.submit(suelta.wait, 5.0) for _ in range(2)]
    en_cola = pool.submit(lambda: "tarde")
    pool.shutdown()
    pool.shutdown(cancel_futures=True)
    suelta.set()
    for futuro in ocupados:
        futuro.result(timeout=5.0)
    for hilo in set(pool._threads):
        hilo.join(timeout=2.0)
        assert not hilo.is_alive()
    assert en_cola.cancelled()


def test_daemon_pool_un_cancel_antes_de_empezar_no_corre():
    from crotolamo.core.turn import DaemonPool

    pool = DaemonPool(1, "Prueba")
    suelta = threading.Event()
    corridos: list[str] = []
    ocupado = pool.submit(suelta.wait, 5.0)
    en_cola = pool.submit(lambda: corridos.append("no"))
    assert en_cola.cancel()
    suelta.set()
    ocupado.result(timeout=5.0)
    assert pool.submit(lambda: "sí").result(timeout=5.0) == "sí"
    assert corridos == []
    pool.shutdown()


def test_daemon_pool_sin_hilo_que_arrancar_lanza_y_no_deja_el_trabajo_en_cola(monkeypatch):
    """Si el sistema no da hilos ("can't start new thread"), submit lanza
    RuntimeError (el agente corre ese carril él mismo) y el trabajo NO queda en
    la cola: otro hilo lo correría después, dos veces."""
    from crotolamo.core.turn import DaemonPool

    pool = DaemonPool(2, "Prueba")
    suelta = threading.Event()
    ocupado = pool.submit(suelta.wait, 5.0)            # el único hilo que hubo
    corridos: list[str] = []

    def sin_hilos(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", sin_hilos)
    with pytest.raises(RuntimeError):
        pool.submit(lambda: corridos.append("dos veces"))
    monkeypatch.undo()
    suelta.set()
    ocupado.result(timeout=5.0)
    assert pool.submit(lambda: "sigue").result(timeout=5.0) == "sigue"
    assert corridos == [] and len(pool._threads) == 1
    pool.shutdown()


# --- la libreta SQLite (y la memoria) en un solo carril -------------------------------

_LIBRETA = ("recall_facts", "search_facts", "list_shortcuts", "buscar_recuerdos")


def test_las_lecturas_de_la_libreta_comparten_carril():
    from crotolamo.tools import default_registry
    from crotolamo.tools.memoria import memoria_tools

    reg = default_registry()
    carriles = {n: reg.get(n).parallel_lane for n in reg.names()
                if getattr(reg.get(n), "parallel_safe", False)}
    assert {n for n, c in carriles.items() if c == "libreta"} == {
        "recall_facts", "search_facts", "list_shortcuts"}
    assert {n: c for n, c in carriles.items() if c is not None} == {
        n: "libreta" for n in ("recall_facts", "search_facts", "list_shortcuts")}
    assert {t.name: t.parallel_lane for t in memoria_tools()} == {
        "recordar_de_mi": None, "buscar_recuerdos": "libreta", "olvidar_recuerdo": None}


def _tools_reales(*nombres: str, paralelas: bool = True) -> list[Tool]:
    """Copias de las tools de fábrica (sin tocar las del registry global)."""
    import dataclasses

    from crotolamo.tools import default_registry
    from crotolamo.tools.memoria import memoria_tools

    todas = {t.name: t for t in memoria_tools()}
    reg = default_registry()
    for n in reg.names():
        todas[n] = reg.get(n)
    return [dataclasses.replace(todas[n], parallel_safe=paralelas) for n in nombres]


_LOTE_LIBRETA = [_call("recall_facts"), _call("search_facts", query="perro"),
                 _call("lee_a", x="1"), _call("list_shortcuts"),
                 _call("recall_facts", categoria="x")]


def _agente_libreta(m: _Medidor, paralelas: bool) -> ToolAgent:
    reg = Registry()
    for t in _tools_reales("recall_facts", "search_facts", "list_shortcuts",
                           paralelas=paralelas):
        reg.register(t)
    reg.register(Tool("lee_a", m.tool("lee_a", 0.15), "a", _P, parallel_safe=paralelas))
    return ToolAgent(_LLM([_resp(*_LOTE_LIBRETA), ChatResponse(content="Ahí va.")]),
                     Conversation("SYS"), registry=reg, guard=_Guard(),
                     pre_hooks=[], post_hooks=[])


def test_un_lote_con_la_libreta_no_abre_dos_conexiones_a_la_vez(monkeypatch, tmp_path):
    """sqlite_parallel.py de la revisión: en una libreta recién creada, dos
    conexiones a la vez chocan en `PRAGMA journal_mode=WAL` (SQLITE_BUSY sin
    esperar el busy timeout) y una tool daba "database is locked" donde en serie
    contestaba. En su carril, las lecturas de la libreta van una tras otra; lo
    demás del lote sigue corriendo al lado."""
    from contextlib import contextmanager

    from crotolamo.persistence import db

    abiertas = {"n": 0, "pico": 0}
    lock = threading.Lock()
    real = db.connect

    @contextmanager
    def contada(db_path=None):
        with lock:
            abiertas["n"] += 1
            abiertas["pico"] = max(abiertas["pico"], abiertas["n"])
        try:
            time.sleep(0.03)                         # ensancha la ventana del choque
            with real(db_path) as conn:
                yield conn
        finally:
            with lock:
                abiertas["n"] -= 1

    monkeypatch.setattr(db, "connect", contada)
    vueltas = {}
    for paralelas in (False, True):
        monkeypatch.setattr(db, "_default_db_path",
                            lambda p=paralelas: tmp_path / f"libreta_{p}.sqlite")
        abiertas["pico"] = 0
        m = _Medidor()
        agent = _agente_libreta(m, paralelas)
        assert agent.handle_turn("dime todo") == "Ahí va."
        vueltas[paralelas] = _vuelta(agent)
        agent.close()
        assert abiertas["pico"] == 1, paralelas
    assert vueltas[True] == vueltas[False]
    resultados = [c[3] for c in vueltas[True]["calls"]]
    assert resultados[0] == "No tengo hechos guardados, patrón."
    assert resultados[3] == "No tienes atajos guardados, patrón."


def test_la_libreta_recien_creada_da_lo_mismo_en_paralelo_que_en_serie(monkeypatch, tmp_path):
    """El mismo lote, sin dobles de SQLite, sobre libretas recién creadas."""
    from crotolamo.persistence import db

    for i in range(20):
        vueltas = {}
        for paralelas in (False, True):
            ruta = tmp_path / f"libreta_{i}_{paralelas}.sqlite"
            monkeypatch.setattr(db, "_default_db_path", lambda r=ruta: r)
            m = _Medidor()
            agent = _agente_libreta(m, paralelas)
            agent.handle_turn("dime todo")
            vueltas[paralelas] = _vuelta(agent)
            agent.close()
        assert vueltas[True] == vueltas[False], i


def test_varias_busquedas_de_recuerdos_en_un_lote_tienen_su_presupuesto_entero(
        memoria, monkeypatch):
    """memoria_pool_budget.py de la revisión: la búsqueda explícita cuenta su
    presupuesto desde que se encola en el pool de 2 de la Memoria. Tres
    buscar_recuerdos a la vez: la 3ª esperaba hilo con el reloj corriendo y daba
    "No pude consultar la memoria" donde en serie contestaba. En el carril de la
    libreta van una tras otra, cada una con su presupuesto, como en serie."""
    m, almacen = memoria()
    almacen.items = {"m1": "mi perro se llama Tletl", "m2": "mi café de olla",
                     "m3": "mi coche es rojo"}
    almacen.tardar_s = 0.4
    monkeypatch.setattr(memoria_mod, "_MEMORIA", m)
    real = Memoria.buscar_estricto

    def escalado(self, query, top_k=None, timeout_s=None):
        # El presupuesto de la tool (10 s) x0.06: 0.6 s para un almacén de 0.4 s,
        # la proporción de un almacén de 6.7 s con los 10 s de verdad.
        return real(self, query, top_k, None if timeout_s is None else timeout_s * 0.06)

    monkeypatch.setattr(Memoria, "buscar_estricto", escalado)
    lote = [_call("buscar_recuerdos", pregunta=p) for p in ("mi perro", "mi café", "mi coche")]
    lote.insert(1, _call("lee_a", x="1"))
    vueltas = {}
    for paralelas in (False, True):
        almacen.max_en_curso = 0
        m_ = _Medidor()
        reg = Registry()
        for t in _tools_reales("buscar_recuerdos", paralelas=paralelas):
            reg.register(t)
        reg.register(Tool("lee_a", m_.tool("lee_a", 0.3), "a", _P, parallel_safe=paralelas))
        agent = ToolAgent(_LLM([_resp(*lote), ChatResponse(content="Ya.")]),
                          Conversation("SYS"), registry=reg, guard=_Guard(),
                          pre_hooks=[], post_hooks=[])
        assert agent.handle_turn("qué sabes de mí") == "Ya."
        vueltas[paralelas] = _vuelta(agent)
        agent.close()
        assert almacen.max_en_curso == 1, paralelas
    assert vueltas[True] == vueltas[False]
    assert [c[3].splitlines()[0] for c in vueltas[True]["calls"]] == [
        "Esto recuerdo, patrón:", "lee_a: 1", "Esto recuerdo, patrón:", "Esto recuerdo, patrón:"]


# --- con la política de repetición, la misma llamada repetida en UN lote ---------------

class _LLMVista(_LLM):
    """_LLM que anota cuántas tools llevó cada petición."""

    def __init__(self, seq):
        super().__init__(seq)
        self.tools: list[int] = []

    def chat(self, messages, tools=None):
        self.tools.append(len(tools or []))
        return super().chat(messages, tools)


@pytest.mark.parametrize("politica", [True, False])
def test_la_misma_lectura_rota_tres_veces_en_un_lote_da_lo_de_en_serie(politica):
    """policy_dup.py de la revisión: el plan del lote miraba la política ANTES
    de correr nada, así que tres `revienta(x=1)` en un mensaje corrían las tres y
    la siguiente petición llevaba tools; en serie, la 3ª se frena (repeated) y la
    siguiente va sin tools. Con la política encendida y una clave repetida en el
    lote, el lote va en serie. Sin la política, nada cambia: a la vez, como antes."""
    lote = [_call("revienta", x="1"), _call("revienta", x="1"), _call("revienta", x="1")]
    vueltas, vistas, medidores, con_hilos = {}, {}, {}, {}
    for paralelas in (False, True):
        m = _Medidor()
        llm = _LLMVista([_resp(*lote), ChatResponse(content="El disco no responde, patrón.")])
        agent = ToolAgent(llm, Conversation("SYS"), registry=_reg_lectura(m, paralelas),
                          guard=_Guard(), pre_hooks=[], post_hooks=[],
                          stop_on_repeated_failure=politica)
        assert agent.handle_turn("lee") == "El disco no responde, patrón."
        vueltas[paralelas], vistas[paralelas], medidores[paralelas] = _vuelta(agent), llm.tools, m
        con_hilos[paralelas] = agent._pool is not None
        agent.close()
    assert vueltas[True] == vueltas[False]
    assert vistas[True] == vistas[False]
    corridas = [n for e, n in medidores[True].traza if e == "corre"]
    if politica:
        assert corridas == ["revienta", "revienta"]
        assert [c[2] for c in vueltas[True]["calls"]] == ["hard_error", "hard_error", "repeated"]
        assert vistas[True][-1] == 0                  # la siguiente petición, sin tools
        assert con_hilos[True] is False               # el lote fue en serie
    else:
        assert corridas == ["revienta"] * 3
        assert con_hilos[True] is True                # sin la política, a la vez
        assert all(v > 0 for v in vistas[True])


# --- revisión 3: la pausa de la extracción no puede retener la salida -----------------
# Las dos salidas vacían la extracción con flush(): el shell en atexit
# (_vaciar_al_salir, 5 s) y el listener al recibir SIGTERM
# (_flush_memoria_best_effort, 2 s); ninguna llama a cerrar(), lo único que cortaba
# la pausa. Con el hilo en su pausa de 120 s, flush() solo podía agotar su tiempo:
# exit_after_extract_fail.py / exit_paused.py de la revisión, 5.2 s y 2.0 s con la
# cola VACÍA (0.2 s antes del paso 4), y lo que llegó durante la pausa ni se intentaba.


def _en_la_pausa(m: Memoria) -> bool:
    """El hilo de extracción está DENTRO de la pausa (el Event.wait que llama
    _drenar), no esperando el siguiente item (Queue.get)."""
    hilo = m._worker
    marco = sys._current_frames().get(hilo.ident) if hilo is not None else None
    nombres = []
    while marco is not None:
        nombres.append(marco.f_code.co_name)
        marco = marco.f_back
    return "_drenar" in nombres[1:] and nombres[nombres.index("_drenar") - 1] == "wait"


def test_flush_en_la_pausa_sin_nada_pendiente_vuelve_en_el_acto(memoria, monkeypatch):
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    _fallar_tres(m, almacen)
    assert _esperar(lambda: _en_la_pausa(m))
    t0 = time.monotonic()
    assert m.flush(timeout_s=2.0) is True             # los 3 ya se intentaron
    assert time.monotonic() - t0 < 0.5
    # Con algo encolado durante la pausa, flush() espera como siempre (paso 4).
    assert m.extraer_en_fondo("cuatro", "ok")
    assert m.flush(timeout_s=0.15) is False
    assert almacen.intentos == ["uno", "dos", "tres"]


def test_el_flush_del_listener_en_la_pausa_no_espera_sus_2_s(memoria, monkeypatch):
    """El docstring de _flush_memoria_best_effort lo promete: con la cola vacía
    el apagado no tarda más."""
    from interfaces import listener

    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    _fallar_tres(m, almacen)
    assert _esperar(lambda: _en_la_pausa(m))
    monkeypatch.setattr(memoria_mod, "_MEMORIA", m)
    t0 = time.monotonic()
    listener._flush_memoria_best_effort()
    assert time.monotonic() - t0 < 0.5


def test_el_vaciado_al_salir_corta_la_pausa_y_extrae_lo_encolado(memoria, monkeypatch, caplog):
    """Al salir no hay "después de la pausa": lo que llegó durante ella se intenta
    ya, dentro del presupuesto de la salida, como antes del paso 4."""
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        _fallar_tres(m, almacen)
        assert _esperar(lambda: _en_la_pausa(m))
        almacen.fallar_extraccion = None              # el proveedor ya se recuperó
        assert m.extraer_en_fondo("me gusta el pozole", "va")
        t0 = time.monotonic()
        assert m._vaciar_al_salir() is True
        assert time.monotonic() - t0 < 1.0
    assert almacen.extraidos == ["me gusta el pozole"]
    assert caplog.text.count("pauso la extracción") == 1


def test_tras_el_vaciado_al_salir_no_hay_mas_pausas_ni_avisos_de_pausa(memoria, monkeypatch,
                                                                     caplog):
    """Con el proveedor aún caído, cada fallo tras la salida avisaba "pauso la
    extracción 120 s" sin pausar (el Event ya estaba puesto)."""
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    m, almacen = memoria()
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        _fallar_tres(m, almacen)
        assert _esperar(lambda: _en_la_pausa(m))
        for texto in ("cuatro", "cinco", "seis"):
            assert m.extraer_en_fondo(texto, "ok")
        t0 = time.monotonic()
        assert m._vaciar_al_salir() is True
        assert time.monotonic() - t0 < 1.0
    assert almacen.intentos == ["uno", "dos", "tres", "cuatro", "cinco", "seis"]
    assert caplog.text.count("pauso la extracción") == 1


# El shell de verdad: N turnos con la extracción fallando (429) y el proceso sale
# sin cerrar() (run_shell vuelve con "salir"): el vaciado de atexit. El primer
# atexit apuntado corre el ÚLTIMO, tras el de la Memoria: mide la salida desde ahí.
_SALIDA_EN_PAUSA = r'''
import atexit, logging, sys, threading, time
logging.disable(logging.CRITICAL)
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.memoria import Memoria, MemoriaConfig

memoria_mod._ESPERA_EXTRACCION_AL_SALIR_S = 15.0   # una espera no pasa por salida rápida

class Proveedor429:
    def __init__(self):
        self.intentos = 0
    def buscar(self, q, k, u):
        return []
    def extraer(self, usuario, respuesta):
        self.intentos += 1
        raise RuntimeError("429 Too Many Requests")

n_antes, n_en_pausa = int(sys.argv[1]), int(sys.argv[2])
p = Proveedor429()
t = {}
atexit.register(lambda: print(f"salida {time.monotonic() - t['salir']:.2f} s, "
                              f"intentos {p.intentos}", flush=True))
m = Memoria(MemoriaConfig(enabled=True), backend=p)
for i in range(n_antes):
    m.extraer_en_fondo(f"turno {i}", "ok")
limite = time.monotonic() + 5
while p.intentos < n_antes and time.monotonic() < limite:
    time.sleep(0.01)
time.sleep(0.2)                                    # el hilo entra en su pausa de 120 s
for i in range(n_en_pausa):
    m.extraer_en_fondo(f"en pausa {i}", "ok")
t["salir"] = time.monotonic()
'''


@pytest.mark.parametrize("n_en_pausa", [0, 2])
def test_el_shell_sale_sin_esperar_la_pausa_de_la_extraccion(n_en_pausa):
    import re

    _, salida = _correr_y_medir(_SALIDA_EN_PAUSA, "3", str(n_en_pausa))
    medida = re.search(r"salida ([\d.]+) s, intentos (\d+)", salida)
    assert medida, salida
    assert float(medida.group(1)) < 3.0, salida      # antes: los 15 s enteros
    assert int(medida.group(2)) == 3 + n_en_pausa    # lo de la pausa también se intenta
