"""Revisión final, unidad 42b: interfaces/shell.py (build_agent y _memory_knob).

1. Con GLM de cerebro y [memoria].enabled, el pre-hook no repite un recuerdo que
   ya va en la ventana: decide UNA vez por turno, antes de la llamada, mirando
   `not llm.degraded`. Pero FallbackLLM elige motor llamada a llamada: si la nube
   cae A MEDIA llamada, el local recibe esa petición (y, con el breaker ya
   abierto, las siguientes del turno) y su ajuste a num_ctx=2048 tira los turnos
   viejos, donde iba la única copia del recuerdo. El local contestaba sin él
   (antes de la auditoría el recuerdo se repetía siempre en el mensaje en curso).

2. `trim_chunk = inf` (TOML válido) pasaba _memory_knob y Conversation() hacía
   int(inf): OverflowError fuera del try de build_agent, el listener salía con 1
   y systemd lo reiniciaba en bucle.

build_agent REAL, con un LLMClient REAL (num_ctx=2048, el de fábrica) de local y
un transporte que graba lo que de verdad saldría hacia Ollama.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import tomllib

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.llm import ChatResponse, LLMClient, TransientLLMError
from crotolamo.core.memoria import Memoria
from crotolamo.settings import Settings
from interfaces import shell
from tests.test_perf_no_byte_budget_from_tool_to_wire import (
    _CABECERA,
    _lineas_de_recuerdo,
    _memoria_de,
)

_CAFE = "- Le gusta el café de olla"
# Una respuesta larga de la nube: la historia pasa pronto del ajuste del local.
_LARGO = " ".join(["Pues mire, patrón, le cuento lo que sé de eso con calma."] * 12)


class _Nube:
    """El cliente GLM: contesta largo o, con `cae`, falla como un timeout."""

    model = "glm"

    def __init__(self) -> None:
        self.cae = False
        self.mensajes: list[list[dict]] = []

    def _responde(self, messages) -> ChatResponse:
        self.mensajes.append(messages)
        if self.cae:
            raise TransientLLMError("timed out")
        return ChatResponse(content=_LARGO)

    def chat(self, messages, tools=None):
        return self._responde(messages)

    def chat_stream(self, messages, tools=None, on_token=None):
        respuesta = self._responde(messages)
        if on_token is not None:
            on_token(respuesta.content)
        return respuesta


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


class _Ollama:
    """post_json del LLMClient local: graba el payload YA ajustado a num_ctx."""

    def __init__(self, guion: list[dict] | None = None) -> None:
        self.payloads: list[dict] = []
        self.hilos: list[int] = []
        self.guion = list(guion or [])

    def __call__(self, path, payload, headers=None):
        self.payloads.append(payload)
        self.hilos.append(threading.get_ident())
        if self.guion:
            return _Resp(self.guion.pop(0))
        return _Resp({"role": "assistant", "content": "Café de olla, patrón."})


def _armar(monkeypatch, tmp_path, memoria: Memoria, guion: list[dict] | None = None,
           local=None):
    """build_agent con GLM de cerebro. Sin `local`, un LLMClient REAL de 2048."""
    from crotolamo.core import hooks

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": "glm", "num_ctx": 2048}},
                                 user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria", lambda: memoria)
    monkeypatch.setattr(memoria_mod, "mem0_instalado", lambda: True)
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    monkeypatch.setattr(memoria, "precalentar", lambda: None)
    nube = _Nube()
    ollama = _Ollama(guion)
    if local is None:
        local = LLMClient(num_ctx=2048)
        local._transport.post_json = ollama  # type: ignore[method-assign]
    llm = FallbackLLM(nube, local)
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, conv = shell.build_agent()
    return agent, conv, nube, ollama, llm


def _charla(agent, vueltas: int = 8) -> None:
    for i in range(vueltas):
        agent.handle_turn(f"cuéntame algo, vuelta {i}")


def _ultimo_user(mensajes) -> str:
    return [m["content"] for m in mensajes if m["role"] == "user"][-1]


@pytest.fixture
def memoria():
    m = _memoria_de("Le gusta el café de olla")
    yield m
    m.cerrar()


# --- 1. la nube cae a media llamada: el local recibe los recuerdos -----------------


def test_la_nube_cae_a_media_llamada_y_el_local_ve_el_recuerdo(monkeypatch, tmp_path,
                                                                memoria):
    agent, conv, nube, ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    _charla(agent)
    # Con la nube arriba, el recuerdo viajó UNA vez (turno 1) y no se repitió.
    assert _lineas_de_recuerdo(nube.mensajes[-1]) == [_CAFE]
    nube.cae = True

    assert agent.handle_turn("y qué me gusta tomar?") == "Café de olla, patrón."
    al_local = ollama.payloads[-1]["messages"]
    # El caso existe: el ajuste a 2048 tiró el turno 1, la única copia guardada.
    assert not any("vuelta 0" in (m["content"] or "") for m in al_local)
    # Antes: [] (el local contestaba sin el recuerdo). Ahora, una vez, en el
    # mensaje del turno en curso, que el ajuste no tira nunca.
    assert _lineas_de_recuerdo(al_local) == [_CAFE]
    assert _CAFE in _ultimo_user(al_local)
    assert _ultimo_user(al_local).endswith("y qué me gusta tomar?")


def _recuerdos_en(texto: str) -> set[str]:
    """Las líneas de TODOS los bloques de recuerdos de un mensaje."""
    lineas: set[str] = set()
    for trozo in texto.split(_CABECERA)[1:]:
        lineas.update(trozo.split("]\n\n", 1)[0].split("\n"))
    return lineas


def _al_local_tras_charla(monkeypatch, tmp_path, modo: str) -> list[dict]:
    m = _memoria_de("Le gusta el café de olla", "Su perro se llama Chispas")
    agent, _conv, nube, ollama, llm = _armar(monkeypatch, tmp_path, m)
    _charla(agent)
    # Uno nuevo para este turno: va en el bloque de siempre; el viejo, omitido.
    m._backend.textos.append("Vive en Puebla")
    if modo == "breaker":
        llm._retry_at = time.monotonic() + 60
    else:
        nube.cae = True
    agent.handle_turn("y qué me gusta tomar?")
    m.cerrar()
    return ollama.payloads[-1]["messages"]


def test_ve_lo_mismo_que_con_el_breaker_ya_abierto(monkeypatch, tmp_path):
    """Los recuerdos del turno que recibe el local no dependen de CUÁNDO cayó la
    nube: con el breaker abierto antes del turno o a media llamada."""
    abierto = _recuerdos_en(_ultimo_user(_al_local_tras_charla(monkeypatch, tmp_path,
                                                                "breaker")))
    a_media = _recuerdos_en(_ultimo_user(_al_local_tras_charla(monkeypatch, tmp_path,
                                                                "midcall")))
    assert abierto == {_CAFE, "- Su perro se llama Chispas", "- Vive en Puebla"}
    assert a_media == abierto


def test_todas_las_llamadas_del_local_en_ese_turno_lo_ven(monkeypatch, tmp_path, memoria):
    """La 1ª llamada cae y la repite el local (shaper); la 2ª, tras la tool, va
    directa al local con el breaker abierto: sin shaper, y también la necesita."""
    pide = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "ram_usage", "arguments": {}}}]}
    agent, _conv, nube, ollama, _llm = _armar(monkeypatch, tmp_path, memoria, guion=[pide])
    agent.registry.get("ram_usage").func = lambda: "RAM: 5.0 GB usados, patrón."
    _charla(agent)
    antes = len(ollama.payloads)
    nube.cae = True

    agent.handle_turn("oye dime cuanta ram me queda libre")
    del_turno = ollama.payloads[antes:]
    assert len(del_turno) == 2
    assert del_turno[1]["messages"][-1]["role"] == "tool"
    for payload in del_turno:
        assert _lineas_de_recuerdo(payload["messages"]) == [_CAFE]


class _LocalGraba:
    """Un local sin ajuste: graba lo que recibe por chat y por chat_stream."""

    model = "ollama"

    def __init__(self) -> None:
        self.vistos: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.vistos.append(messages)
        return ChatResponse(content="Café de olla, patrón.")

    def chat_stream(self, messages, tools=None, on_token=None):
        respuesta = self.chat(messages, tools)
        if on_token is not None:
            on_token(respuesta.content)
        return respuesta


def test_por_voz_chat_stream_tambien(monkeypatch, tmp_path, memoria):
    """El loop de voz pide por chat_stream: el local recibe lo mismo."""
    local = _LocalGraba()
    agent, _conv, nube, _ollama, _llm = _armar(monkeypatch, tmp_path, memoria, local=local)
    _charla(agent)
    nube.cae = True
    dichos: list[str] = []
    agent.handle_turn("y qué me gusta tomar?", on_token=dichos.append)
    assert "".join(dichos) == "Café de olla, patrón."
    assert _CAFE in _ultimo_user(local.vistos[-1])


def test_solo_viaja_en_la_peticion_la_conversacion_no_cambia(monkeypatch, tmp_path,
                                                             memoria):
    """La nube, de vuelta, ve la historia de siempre: sin copias de más."""
    agent, conv, nube, _ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    _charla(agent)
    nube.cae = True
    agent.handle_turn("y qué me gusta tomar?")
    assert _CABECERA not in _ultimo_user(conv.to_messages())
    nube.cae = False
    llm._retry_at = 0.0
    agent.handle_turn("y otra cosa")
    assert _lineas_de_recuerdo(nube.mensajes[-1]) == [_CAFE]


def test_con_la_nube_arriba_nada_cambia(monkeypatch, tmp_path, memoria):
    agent, _conv, nube, ollama, _llm = _armar(monkeypatch, tmp_path, memoria)
    _charla(agent, 25)
    assert _lineas_de_recuerdo(nube.mensajes[-1]) == [_CAFE]
    assert ollama.payloads == []


def test_otro_hilo_no_recibe_los_recuerdos_del_turno(monkeypatch, tmp_path, memoria):
    """El resumen de [memory].compaction llama al mismo LLM desde su hilo, tras
    el turno: su petición no es la del patrón y no lleva recuerdos."""
    agent, _conv, nube, ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    _charla(agent)
    nube.cae = True
    agent.handle_turn("y qué me gusta tomar?")
    assert llm.degraded
    hilo = threading.Thread(
        target=lambda: llm.chat([{"role": "user", "content": "resume esto"}]))
    hilo.start()
    hilo.join(timeout=10)
    assert ollama.payloads[-1]["messages"] == [{"role": "user", "content": "resume esto"}]


def test_el_local_envuelto_sigue_siendo_el_mismo_cliente(monkeypatch, tmp_path, memoria):
    """Lo demás del contrato (model, preconnect, el transporte) es el del cliente
    de verdad, y copiar el envoltorio no entra en una recursión."""
    import copy

    _agent, _conv, _nube, ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    llm._retry_at = time.monotonic() + 60
    assert llm.model == LLMClient(num_ctx=2048).model
    assert llm.preconnect() is False                    # el de LLMClient
    assert llm.secondary.num_ctx == 2048
    assert llm.secondary._transport.post_json is ollama
    for envoltorio in (llm.secondary, llm.secondary._recuerdos):
        copia = copy.copy(envoltorio)
        assert type(copia) is type(envoltorio)


def test_un_turno_que_no_busca_no_hereda_los_del_anterior(monkeypatch, tmp_path, memoria):
    """Un comando anclado no busca recuerdos (make_memoria_prehook): si la nube
    cae en él, el local no recibe los que dejó fuera el turno anterior."""
    agent, _conv, nube, ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    _charla(agent)
    nube.cae = True
    agent.handle_turn("pausa la música")
    assert _CABECERA not in _ultimo_user(ollama.payloads[0]["messages"])


def test_turno_con_el_breaker_abierto_no_duplica(monkeypatch, tmp_path, memoria):
    """Con el breaker abierto el pre-hook ya repite todo: nada que devolver."""
    agent, _conv, _nube, ollama, llm = _armar(monkeypatch, tmp_path, memoria)
    agent.handle_turn("hola")
    llm._retry_at = time.monotonic() + 60
    agent.handle_turn("y luego")
    assert _ultimo_user(ollama.payloads[-1]["messages"]).count(_CAFE) == 1


# --- 2. _memory_knob con inf ------------------------------------------------------


@pytest.mark.parametrize("clave,default", [("trim_chunk", 1), ("stale_tool_turns", None),
                                           ("idle_reset_min", None)])
def test_memory_knob_rechaza_infinito(clave, default, caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.shell"):
        assert shell._memory_knob({clave: float("inf")}, clave, default) == default
    assert f"[memory].{clave}" in caplog.text


def test_memory_knob_finito_sigue_valiendo():
    assert shell._memory_knob({"trim_chunk": 5}, "trim_chunk", 1) == 5
    assert shell._memory_knob({"trim_chunk": 2.5}, "trim_chunk", 1) == 2.5
    assert shell._memory_knob({}, "trim_chunk", 1) == 1


def test_build_agent_arranca_con_inf_en_memory(monkeypatch, tmp_path):
    from crotolamo.core import hooks

    raw = tomllib.loads("[llm]\nbackend = \"ollama\"\n"
                        "[memory]\ntrim_chunk = inf\nstale_tool_turns = inf\n"
                        "idle_reset_min = inf\n")
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw=raw, user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    _agent, conv = shell.build_agent()  # antes: OverflowError
    assert conv.trim_chunk == 1
    assert conv.stale_tool_turns is None
    assert conv.idle_reset_s is None
