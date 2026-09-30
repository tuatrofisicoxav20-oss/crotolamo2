"""Memoria semántica (mem0) con el backend y el LLM MOCKEADOS.

Qué se fija: recuperar e inyectar al contexto, fallo/tardanza de la memoria sin
tumbar la respuesta, extracción en segundo plano que no bloquea el turno,
olvidar por descripción, que no se guardan secretos, la regla de contenido en
las tools y el prompt, el registro de tools solo con enabled=true, y la CLI.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest

from crotolamo.core import memoria as memoria_mod
from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import ChatResponse
from crotolamo.core.memoria import (
    INSTRUCCIONES_EXTRACCION,
    REGLA_CONTENIDO,
    Memoria,
    MemoriaConfig,
    MemoriaNoDisponible,
    Recuerdo,
    SecretoRechazado,
    make_memoria_posthook,
    make_memoria_prehook,
    parece_secreto,
)
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry


class FakeBackend:
    """Doble de mem0: ranking por solape de palabras, scores en la escala real."""

    def __init__(self, recuerdos: list[str] | None = None) -> None:
        self.items: dict[str, str] = {}
        self.extraidos: list[tuple[str, str]] = []
        self.borrados: list[str] = []
        self.busquedas = 0
        self.tardar_s = 0.0
        self.fallar: Exception | None = None
        self.bloqueo_extraccion: threading.Event | None = None
        for r in recuerdos or []:
            self.guardar(r)

    def guardar(self, texto: str) -> list[str]:
        mid = f"m{len(self.items) + 1}"
        self.items[mid] = texto
        return [mid]

    def extraer(self, usuario: str, respuesta: str) -> list[str]:
        if self.bloqueo_extraccion is not None:
            self.bloqueo_extraccion.wait(timeout=5.0)
        self.extraidos.append((usuario, respuesta))
        return [usuario]

    def buscar(self, query: str, top_k: int, umbral: float) -> list[Recuerdo]:
        self.busquedas += 1
        if self.fallar is not None:
            raise self.fallar
        if self.tardar_s:
            time.sleep(self.tardar_s)
        q = set(query.lower().split())
        out = []
        for mid, texto in self.items.items():
            solape = len(q & set(texto.lower().split()))
            score = 0.02 + 0.01 * solape  # sin solape: 0.02 (bajo el umbral 0.03)
            if score >= umbral:
                out.append(Recuerdo(mid, texto, score))
        return sorted(out, key=lambda r: r.score, reverse=True)[:top_k]

    def todas(self) -> list[Recuerdo]:
        return [Recuerdo(k, v) for k, v in self.items.items()]

    def borrar(self, memoria_id: str) -> None:
        self.borrados.append(memoria_id)
        self.items.pop(memoria_id, None)


def _memoria(recuerdos=None, **cfg) -> tuple[Memoria, FakeBackend]:
    backend = FakeBackend(recuerdos)
    config = MemoriaConfig(**{"enabled": True, "umbral": 0.03, "top_k": 3, **cfg})
    m = Memoria(config, backend=backend)
    return m, backend


@pytest.fixture(autouse=True)
def _sin_singleton():
    memoria_mod.set_memoria(None)
    yield
    memoria_mod.set_memoria(None)


# --- recuperación e inyección ---------------------------------------------------

def test_prehook_inyecta_los_recuerdos_relevantes():
    m, backend = _memoria(["mi perro se llama Tletl", "me gusta el café sin azúcar",
                           "uso Fedora con Hyprland"])
    hook = make_memoria_prehook(m)
    out = hook("¿cómo se llama mi perro?")
    assert out.startswith("[recuerdos sobre el patrón")
    assert "Tletl" in out and out.endswith("¿cómo se llama mi perro?")
    # Los que no pasan el umbral no entran.
    assert "Fedora" not in out


def test_prehook_respeta_top_k():
    m, backend = _memoria([f"mi perro número {i} se llama Perro{i}" for i in range(6)], top_k=2)
    out = make_memoria_prehook(m)("mi perro")
    assert out.count("- mi perro") == 2


def test_sin_recuerdos_el_texto_va_tal_cual():
    m, backend = _memoria(["uso Fedora"])
    assert make_memoria_prehook(m)("hola qué tal") == "hola qué tal"


def test_memoria_desactivada_es_no_op():
    backend = FakeBackend(["mi perro se llama Tletl"])
    m = Memoria(MemoriaConfig(enabled=False), backend=backend)
    assert make_memoria_prehook(m)("mi perro") == "mi perro"
    assert backend.busquedas == 0
    assert m.extraer_en_fondo("hola", "hola") is False


# --- fallos y tardanza: la respuesta sale igual -------------------------------

def test_fallo_del_backend_no_tumba_la_respuesta(caplog):
    m, backend = _memoria(["mi perro se llama Tletl"])
    backend.fallar = RuntimeError("chroma explotó")
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert make_memoria_prehook(m)("mi perro") == "mi perro"
    assert "chroma explotó" in caplog.text
    assert m.enabled  # un fallo puntual no la marca como rota


def test_backend_no_disponible_se_marca_rota_una_vez(caplog):
    m, backend = _memoria(["x"])
    backend.fallar = MemoriaNoDisponible("falta mem0")
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert m.buscar("x") == []
        assert m.buscar("x") == []
    assert not m.enabled and m.motivo_no_disponible == "falta mem0"
    assert caplog.text.count("no disponible") == 1
    assert backend.busquedas == 1  # la segunda ni lo intenta


def test_busqueda_lenta_se_rinde_al_timeout(caplog):
    m, backend = _memoria(["mi perro se llama Tletl"], timeout_busqueda_s=0.2)
    backend.tardar_s = 1.0
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        out = make_memoria_prehook(m)("mi perro")
    assert out == "mi perro"
    assert time.monotonic() - t0 < 0.8
    assert "tardó más" in caplog.text


class _LLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.mensajes: list = []

    def chat(self, messages, tools=None):
        self.mensajes.append(messages)
        return ChatResponse(content=self.reply)


def _agent(m: Memoria, llm: _LLM, tmp_path: Path) -> ToolAgent:
    return ToolAgent(
        llm, Conversation("SYS"), registry=Registry(), guard=Guard([tmp_path]),
        pre_hooks=[make_memoria_prehook(m)], after_turn_hooks=[make_memoria_posthook(m)],
        fastpath=False, route_fn=lambda t: [],
    )


def test_agente_recibe_los_recuerdos_en_el_mensaje(tmp_path):
    m, backend = _memoria(["mi perro se llama Tletl"])
    llm = _LLM("Tu perro es Tletl, patrón.")
    agent = _agent(m, llm, tmp_path)
    assert agent.handle_turn("¿cómo se llama mi perro?") == "Tu perro es Tletl, patrón."
    user_msg = llm.mensajes[0][-1]["content"]
    assert "Tletl" in user_msg and "recuerdos" in user_msg


def test_agente_responde_aunque_la_memoria_reviente(tmp_path):
    m, backend = _memoria(["x"])
    backend.fallar = RuntimeError("boom")
    agent = _agent(m, _LLM("Sigo vivo, patrón."), tmp_path)
    assert agent.handle_turn("hola") == "Sigo vivo, patrón."


# --- extracción en segundo plano --------------------------------------------------

def test_extraccion_no_bloquea_el_turno(tmp_path):
    m, backend = _memoria()
    backend.bloqueo_extraccion = threading.Event()  # la extracción se queda colgada
    agent = _agent(m, _LLM("Órale, patrón."), tmp_path)
    t0 = time.monotonic()
    reply = agent.handle_turn("me gusta el café de olla")
    assert reply == "Órale, patrón."
    assert time.monotonic() - t0 < 0.5  # no esperó a la extracción
    assert backend.extraidos == []      # sigue bloqueada
    backend.bloqueo_extraccion.set()
    assert m.flush(timeout_s=3.0)
    assert backend.extraidos == [("me gusta el café de olla", "Órale, patrón.")]


def test_extraccion_recibe_el_texto_crudo_sin_prehooks(tmp_path):
    m, backend = _memoria(["mi perro se llama Tletl"])
    agent = _agent(m, _LLM("Sí, patrón."), tmp_path)
    agent.handle_turn("mi perro es muy bravo")
    assert m.flush()
    usuario, _ = backend.extraidos[0]
    assert usuario == "mi perro es muy bravo"  # sin el bloque de recuerdos ni la fecha


def test_extraccion_que_falla_no_mata_el_hilo(caplog):
    m, backend = _memoria()

    def _revienta(usuario, respuesta):
        raise RuntimeError("groq 500")

    backend.extraer = _revienta  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert m.extraer_en_fondo("hola", "hola")
        assert m.flush()
        assert m.extraer_en_fondo("otra", "otra")
        assert m.flush()
    assert caplog.text.count("groq 500") == 2


def test_extraer_false_no_encola():
    m, backend = _memoria(extraer=False)
    assert m.extraer_en_fondo("me gusta el café", "ok") is False


# --- olvidar ------------------------------------------------------------------------

def test_olvidar_borra_el_mejor_candidato_y_solo_uno():
    m, backend = _memoria(["mi perro se llama Tletl", "mi perro le teme a los cohetes",
                           "uso Fedora"])
    borrado = m.olvidar("lo de mi perro Tletl")
    assert borrado is not None and "Tletl" in borrado.texto
    assert backend.borrados == [borrado.id]
    assert len(backend.items) == 2


def test_olvidar_sin_candidato_no_borra_nada():
    m, backend = _memoria(["uso Fedora"])
    assert m.olvidar("mi coche") is None
    assert backend.borrados == []


def test_olvidar_por_voz_via_tool(monkeypatch):
    from crotolamo.tools import memoria as tools_mod

    m, backend = _memoria(["mi perro se llama Tletl"])
    memoria_mod.set_memoria(m)
    out = tools_mod.olvidar_recuerdo("olvida lo de mi perro")
    assert out.startswith("Olvidado, patrón") and "Tletl" in out
    assert backend.items == {}
    assert "No encontré" in tools_mod.olvidar_recuerdo("lo de mi perro")


# --- secretos ------------------------------------------------------------------------

@pytest.mark.parametrize("texto", [
    "mi api key de groq es gsk_abcDEF1234567890abcdef",
    "guarda sk-proj-abcdefghijklmnop1234567890",
    "la contraseña del wifi es casa-2024-segura",
    "password: Tr0ub4dor&3xx",
    "token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123",
    "-----BEGIN RSA PRIVATE KEY-----",
    "mi clave de acceso es 7Hd82kL0pQz9",
    "el jwt es eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0",
    "spak_1234567890abcdefghijklmnop",
])
def test_parece_secreto_detecta(texto):
    assert parece_secreto(texto)


@pytest.mark.parametrize("texto", [
    "me gusta el café de olla sin azúcar",
    "mi perro se llama Tletl y le tiene miedo a los cohetes",
    "la clave del éxito es la constancia",
    "no me gusta compartir contraseñas por chat",
    "mi mamá vive en Puebla desde 2019",
    "mi token favorito es el de Ethereum",
    "el número de mi casa es 42",
    "",
])
def test_parece_secreto_deja_pasar_lo_normal(texto):
    assert not parece_secreto(texto)


def test_recordar_rechaza_secretos():
    m, backend = _memoria()
    with pytest.raises(SecretoRechazado):
        m.recordar("mi api key es gsk_abcDEF1234567890abcdef")
    assert backend.items == {}


def test_turno_con_secreto_no_se_manda_a_extraer(caplog):
    m, backend = _memoria()
    with caplog.at_level(logging.INFO, logger="crotolamo.core.memoria"):
        assert m.extraer_en_fondo("mi contraseña es casa-2024-segura", "no la guardo") is False
        assert m.extraer_en_fondo("hola", "tu token es ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123") is False
    assert m.flush()
    assert backend.extraidos == []
    assert "credencial" in caplog.text


def test_tool_recordar_rechaza_secretos_en_personaje():
    from crotolamo.tools import memoria as tools_mod

    m, backend = _memoria()
    memoria_mod.set_memoria(m)
    out = tools_mod.recordar_de_mi("mi password: Tr0ub4dor&3xx")
    assert "no lo guardo" in out and backend.items == {}
    assert tools_mod.recordar_de_mi("me gusta el pozole").startswith("Ya quedó")
    assert list(backend.items.values()) == ["me gusta el pozole"]


# --- regla de contenido y tools ------------------------------------------------------

def test_regla_de_contenido_en_tools_y_prompt():
    from crotolamo.tools.memoria import memoria_tools

    for palabra in ("proyectos", "Notion", "secretos", "API keys", "contraseñas", "tokens"):
        assert palabra in REGLA_CONTENIDO
    assert REGLA_CONTENIDO in INSTRUCCIONES_EXTRACCION
    tools = {t.name: t for t in memoria_tools()}
    assert set(tools) == {"recordar_de_mi", "buscar_recuerdos", "olvidar_recuerdo"}
    assert "NUNCA guardes secretos" in tools["recordar_de_mi"].description
    assert "hecho" in tools["recordar_de_mi"].parameters["properties"]
    assert tools["olvidar_recuerdo"].parameters["required"] == ["descripcion"]


def test_config_mem0_lleva_instrucciones_y_telemetria_apagada(tmp_path):
    import os

    from crotolamo.core.memoria import Mem0Backend

    cfg = MemoriaConfig(enabled=True, ruta=tmp_path)
    conf = Mem0Backend(cfg, settings=None).config_mem0()
    assert conf["custom_instructions"] == INSTRUCCIONES_EXTRACCION
    assert conf["vector_store"]["config"]["path"] == str(tmp_path / "chroma")
    assert conf["history_db_path"] == str(tmp_path / "history.db")
    assert conf["llm"]["provider"] == "groq"
    assert conf["embedder"]["config"]["model"].endswith("MiniLM-L12-v2")
    assert os.environ["MEM0_TELEMETRY"] == "False"


def test_config_desde_settings_y_defaults():
    s = Settings(raw={"memoria": {"enabled": True, "umbral": 0.05, "top_k": 2,
                                  "ruta": "~/x/mem", "llm_provider": "OpenAI"}},
                 user="t", home=Path("/tmp"))
    cfg = MemoriaConfig.from_settings(s)
    assert cfg.enabled and cfg.umbral == 0.05 and cfg.top_k == 2
    assert cfg.ruta == Path("~/x/mem").expanduser() and cfg.llm_provider == "openai"
    vacio = MemoriaConfig.from_settings(Settings(raw={}, user="t", home=Path("/tmp")))
    assert not vacio.enabled and vacio.umbral == 0.03 and vacio.top_k == 3
    # "false" entre comillas NO activa.
    assert not MemoriaConfig.from_settings({"enabled": "false"}).enabled


def test_registro_de_tools_solo_con_enabled(monkeypatch):
    from crotolamo import settings as settings_mod
    from crotolamo.tools import _register_memoria_if_enabled, default_registry

    base = default_registry().copy()
    assert base.get("remember_fact") is not None
    assert base.get("recordar_de_mi") is None  # de fábrica, apagada

    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "raw", {**real.raw, "memoria": {"enabled": True}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    reg = base.copy()
    _register_memoria_if_enabled(reg)
    assert reg.get("recordar_de_mi") is not None and reg.get("olvidar_recuerdo") is not None
    assert reg.get("remember_fact") is None  # una sola familia de "recordar"
    assert base.get("remember_fact") is not None  # la copia base no se tocó


def test_router_enruta_a_la_familia_registrada():
    from crotolamo.core import router

    names = router.select_tool_names("acuérdate de que me gusta el pozole")
    assert "recordar_de_mi" in names and "remember_fact" in names  # ambos grupos matchean
    reg = Registry()
    from crotolamo.tools.memoria import memoria_tools

    for t in memoria_tools():
        reg.register(t)
    schemas = router.route_schemas(reg, "olvida lo de mi perro")
    assert {s["function"]["name"] for s in schemas} == {
        "recordar_de_mi", "buscar_recuerdos", "olvidar_recuerdo",
    }


def test_cli_lista_busca_y_olvida(monkeypatch, capsys):
    from crotolamo import settings as settings_mod

    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "raw", {**real.raw, "memoria": {"enabled": True}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    backend = FakeBackend(["mi perro se llama Tletl"])
    monkeypatch.setattr(memoria_mod, "Memoria",
                        lambda cfg, backend=None, settings=None: Memoria(cfg, backend=backend
                                                                         or FakeBackend(
                                                                             list(backend_ref.items.values()))))
    backend_ref = backend
    assert memoria_mod.run_cli(["listar"]) == 0
    assert "Tletl" in capsys.readouterr().out
    assert memoria_mod.run_cli(["buscar", "mi perro"]) == 0
    assert "Tletl" in capsys.readouterr().out
    assert memoria_mod.run_cli(["olvidar", "mi perro"]) == 0
    assert "Olvidado" in capsys.readouterr().out


def test_cli_apagada_avisa(monkeypatch, capsys):
    from crotolamo import settings as settings_mod

    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "raw", {**real.raw, "memoria": {"enabled": False}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    assert memoria_mod.run_cli(["listar"]) == 1
    assert "apagada" in capsys.readouterr().out
