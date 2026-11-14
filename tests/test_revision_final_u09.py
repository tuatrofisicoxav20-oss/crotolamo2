"""Revisión final, unidad 09: crotolamo/core/memoria.py (config y key del extractor).

1. La key de un SDK solo cuenta si va a su emisor. mem0 2.1.0 (mem0/llms/openai.py)
   saca la key de OPENAI_API_KEY cuando la config no trae una, pero la manda al
   `openai_base_url` de la config, que aquí es SIEMPRE el de [llm.glm] (Z.ai de
   fábrica). Sin key de la nube y con OPENAI_API_KEY exportada, cada extracción
   mandaba la key de OpenAI del patrón a Z.ai (401) y el doctor lo daba por bueno.
   OPENROUTER_API_KEY la mira antes que todo y se va a OpenRouter ignorando key y
   base_url: con otra nube, la extracción iba al proveedor equivocado.
2. MemoriaConfig.from_settings nunca lanza: un typo en un knob opcional de
   [memoria] (aunque esté apagada) hacía que build_agent arrancara SIN tools.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.memoria import (
    DEFAULT_EMBEDDER_DIMS,
    DEFAULT_TIMEOUT_BUSQUEDA_S,
    DEFAULT_TOP_K,
    DEFAULT_UMBRAL,
    DEFAULT_USUARIO,
    Mem0Backend,
    Memoria,
    MemoriaConfig,
    extraccion_sin_key,
)
from crotolamo.settings import Settings

_ENVS = ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY",
         "OPENROUTER_API_KEY", "OPENROUTER_API_BASE", "GROQ_API_KEY", "OPENAI_BASE_URL")
_KEY_OPENAI = "sk-del-patron-para-openai"
_KEY_OPENROUTER = "sk-or-del-patron"
_ZAI = "https://api.z.ai/api/paas/v4"
_OPENAI = "https://api.openai.com/v1"
_OPENROUTER = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch):
    # Mem0Backend fija FASTEMBED_CACHE_PATH (setdefault): setenv+delenv hace que
    # monkeypatch la restaure al acabar.
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "x")
    monkeypatch.delenv("FASTEMBED_CACHE_PATH")
    for env in _ENVS:
        monkeypatch.delenv(env, raising=False)
    memoria_mod.set_memoria(None)
    yield
    memoria_mod.set_memoria(None)


# ---------------------------------------------------------------------------
# 1. La key del SDK solo cuenta si va a su emisor
# ---------------------------------------------------------------------------

class _Mem0QueEligeClienteComoElReal:
    """`mem0.Memory` con la elección de cliente de mem0 2.1.0, tal cual
    (mem0/llms/openai.py y groq.py): OPENROUTER_API_KEY primero, si no
    `config.api_key or OPENAI_API_KEY` contra `config.openai_base_url or
    OPENAI_BASE_URL or api.openai.com`. Anota a qué host va cada llamada al LLM."""

    def __init__(self, conf: dict[str, Any], llamadas: list[tuple[str, str]]) -> None:
        c = conf["llm"]["config"]
        if conf["llm"]["provider"] == "groq":
            self.key = c.get("api_key") or os.getenv("GROQ_API_KEY")
            self.url = "https://api.groq.com"
        elif os.environ.get("OPENROUTER_API_KEY"):
            self.key = os.environ["OPENROUTER_API_KEY"]
            self.url = (c.get("openrouter_base_url") or os.getenv("OPENROUTER_API_BASE")
                        or _OPENROUTER)
        else:
            self.key = c.get("api_key") or os.getenv("OPENAI_API_KEY")
            self.url = c.get("openai_base_url") or os.getenv("OPENAI_BASE_URL") or _OPENAI
        if not self.key:
            # Como los SDK reales: sin key no se construye el cliente.
            raise RuntimeError("Missing credentials. Please pass an `api_key`")
        self.llamadas = llamadas

    def add(self, mensajes: Any, user_id: str, infer: bool) -> dict[str, Any]:
        if infer:
            self.llamadas.append((self.key, self.url))
            return {"results": [{"id": "e1", "memory": "hecho", "event": "ADD"}]}
        return {"results": [{"id": "m1", "memory": str(mensajes), "event": "ADD"}]}


def _suplantar_mem0(monkeypatch) -> list[tuple[str, str]]:
    llamadas: list[tuple[str, str]] = []
    modulo = types.ModuleType("mem0")

    class Memory:
        @staticmethod
        def from_config(conf: dict[str, Any]) -> Any:
            return _Mem0QueEligeClienteComoElReal(conf, llamadas)

    modulo.Memory = Memory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", modulo)
    return llamadas


def _settings_nube(base_url: str | None, tmp_path: Path) -> Settings | None:
    if base_url is None:
        return None  # sin settings: la nube de fábrica (Z.ai)
    return Settings(raw={"llm": {"glm": {"base_url": base_url}}}, user="t", home=tmp_path)


def _extrae_un_turno(settings: Any, tmp_path: Path) -> tuple[bool, Memoria]:
    m = Memoria(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"), settings=settings)
    encolado = m.extraer_en_fondo("me gusta el pozole", "va, patrón")
    assert m.flush()
    return encolado, m


@pytest.mark.parametrize("base_url", [None, _ZAI, "https://otra-nube.example/v1",
                                      # Que la URL nombre a OpenAI no la hace OpenAI.
                                      "https://evil.example/api.openai.com/v1"])
def test_sin_key_de_nube_la_key_de_openai_no_viaja_a_otra_nube(
        monkeypatch, tmp_path, caplog, base_url):
    monkeypatch.setenv("OPENAI_API_KEY", _KEY_OPENAI)
    llamadas = _suplantar_mem0(monkeypatch)
    settings = _settings_nube(base_url, tmp_path)
    cfg = MemoriaConfig(enabled=True, ruta=tmp_path / "mem")
    # El doctor dice lo que va a pasar: no hay key para ESA nube.
    assert extraccion_sin_key(cfg, settings)
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        _encolado, m = _extrae_un_turno(settings, tmp_path)
    try:
        assert m.enabled  # recordar y buscar siguen
        assert isinstance(m._backend, Mem0Backend) and m._backend.sin_llm
        assert llamadas == []  # ninguna extracción: ni a Z.ai ni a nadie
        assert "extracción automática queda apagada" in caplog.text
    finally:
        m.cerrar()


def test_con_la_nube_de_openai_su_key_del_sdk_si_cuenta(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", _KEY_OPENAI)
    llamadas = _suplantar_mem0(monkeypatch)
    settings = _settings_nube(_OPENAI, tmp_path)
    assert not extraccion_sin_key(MemoriaConfig(enabled=True), settings)
    encolado, m = _extrae_un_turno(settings, tmp_path)
    try:
        assert encolado and not m._backend.sin_llm  # type: ignore[union-attr]
        assert llamadas == [(_KEY_OPENAI, _OPENAI)]  # a su emisor
    finally:
        m.cerrar()


def test_openrouter_api_key_solo_cuenta_si_la_nube_es_openrouter(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", _KEY_OPENROUTER)
    cfg = MemoriaConfig(enabled=True)
    # Con Z.ai, mem0 se iba a OpenRouter con el modelo de Z.ai: proveedor equivocado.
    assert extraccion_sin_key(cfg, _settings_nube(None, tmp_path))
    assert extraccion_sin_key(cfg, _settings_nube(_OPENAI, tmp_path))
    llamadas = _suplantar_mem0(monkeypatch)
    _encolado, m = _extrae_un_turno(_settings_nube(_ZAI, tmp_path), tmp_path)
    m.cerrar()
    assert llamadas == []
    # Con la nube de OpenRouter, la key va a su emisor: la extracción sigue.
    settings = _settings_nube(_OPENROUTER, tmp_path)
    assert not extraccion_sin_key(cfg, settings)
    encolado, m = _extrae_un_turno(settings, tmp_path)
    m.cerrar()
    assert encolado and llamadas == [(_KEY_OPENROUTER, _OPENROUTER)]


def test_con_key_de_la_nube_todo_sigue_igual(monkeypatch, tmp_path):
    """La key de la nube va en la config y a su base_url, haya o no OPENAI_API_KEY."""
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "clave-de-zai")
    monkeypatch.setenv("OPENAI_API_KEY", _KEY_OPENAI)
    llamadas = _suplantar_mem0(monkeypatch)
    assert not extraccion_sin_key(MemoriaConfig(enabled=True), None)
    encolado, m = _extrae_un_turno(None, tmp_path)
    m.cerrar()
    assert encolado and llamadas == [("clave-de-zai", _ZAI)]


def _check_memoria_del_doctor(monkeypatch, tmp_path: Path, llm: dict[str, Any] | None) -> Any:
    from datetime import date

    from scripts import crotolamo_doctor as doctor

    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    # mem0 & cía no están en el venv de la suite: se dan por importables.
    monkeypatch.setattr(doctor, "_import_error", lambda m: None)
    raw: dict[str, Any] = {"memoria": {"enabled": True, "ruta": str(tmp_path / "mem")}}
    if llm is not None:
        raw["llm"] = llm
    settings = Settings(raw=raw, user="t", home=tmp_path, paths={"home": tmp_path})
    ctx = doctor._Ctx(settings=settings, doctor={}, backend="glm", today=date(2026, 11, 14))
    [check] = doctor._check_memoria(ctx)
    return check


@pytest.mark.parametrize("llm", [None, {"glm": {"base_url": _ZAI}}])
def test_el_doctor_no_da_por_buena_la_key_de_openai_con_otra_nube(monkeypatch, tmp_path, llm):
    monkeypatch.setenv("OPENAI_API_KEY", _KEY_OPENAI)
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, llm)
    assert check.ok and check.warn
    assert "sin API key: la extracción automática queda apagada" in check.detail


def test_groq_api_key_sigue_contando_con_groq(monkeypatch, tmp_path):
    """El SDK de Groq siempre va a Groq: su key de la env es la de su emisor."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-del-patron")
    llamadas = _suplantar_mem0(monkeypatch)
    cfg = MemoriaConfig(enabled=True, ruta=tmp_path / "mem", llm_provider="groq")
    assert not extraccion_sin_key(cfg, _settings_nube("https://api.groq.com/openai/v1",
                                                      tmp_path))
    backend = Mem0Backend(cfg, _settings_nube("https://api.groq.com/openai/v1", tmp_path))
    assert backend.extraer("me gusta el pozole", "va") == ["hecho"]
    assert llamadas == [("gsk-del-patron", "https://api.groq.com")]


# ---------------------------------------------------------------------------
# 2. from_settings nunca lanza por un knob opcional mal escrito
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clave, valor, atributo, default", [
    ("umbral", "0,03", "umbral", DEFAULT_UMBRAL),
    ("umbral", None, "umbral", DEFAULT_UMBRAL),
    ("umbral", float("nan"), "umbral", DEFAULT_UMBRAL),
    ("top_k", "tres", "top_k", DEFAULT_TOP_K),
    ("top_k", [3], "top_k", DEFAULT_TOP_K),
    ("top_k", True, "top_k", DEFAULT_TOP_K),
    ("top_k", float("inf"), "top_k", DEFAULT_TOP_K),
    ("dims_embeddings", "384d", "dims_embeddings", DEFAULT_EMBEDDER_DIMS),
    ("timeout_busqueda_s", "1.5s", "timeout_busqueda_s", DEFAULT_TIMEOUT_BUSQUEDA_S),
    ("timeout_busqueda_s", float("inf"), "timeout_busqueda_s", DEFAULT_TIMEOUT_BUSQUEDA_S),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_un_numero_mal_escrito_se_avisa_y_usa_el_default(
        caplog, clave, valor, atributo, default, enabled):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        cfg = MemoriaConfig.from_settings({"enabled": enabled, clave: valor})
    assert getattr(cfg, atributo) == default
    assert cfg.enabled is enabled  # el resto de la config no se pierde
    assert f"[memoria].{clave}" in caplog.text


def test_los_numeros_bien_escritos_no_cambian(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        cfg = MemoriaConfig.from_settings({"umbral": 0.05, "top_k": 2, "dims_embeddings": 768,
                                           "timeout_busqueda_s": 3})
        # Como antes: lo que float()/int() ya entendían sigue valiendo, y top_k >= 1.
        texto = MemoriaConfig.from_settings({"umbral": "0.04", "top_k": 0})
    assert (cfg.umbral, cfg.top_k, cfg.dims_embeddings, cfg.timeout_busqueda_s) == (
        0.05, 2, 768, 3.0)
    assert (texto.umbral, texto.top_k) == (0.04, 1)
    assert caplog.text == ""


@pytest.mark.parametrize("valor, esperado, avisa", [
    (True, True, False),
    (False, False, False),
    ("false", False, True),  # bool("false") es True: extraía con extraer = "false"
    ("true", False, True),  # como enabled: solo un booleano TOML enciende
    (0, False, True),
])
def test_extraer_solo_acepta_un_booleano(caplog, valor, esperado, avisa):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        cfg = MemoriaConfig.from_settings({"enabled": True, "extraer": valor})
    assert cfg.extraer is esperado
    assert ("[memoria].extraer" in caplog.text) is avisa


def test_sin_extraer_en_la_config_se_extrae_como_siempre(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert MemoriaConfig.from_settings({"enabled": True}).extraer is True
    assert caplog.text == ""


def test_enabled_que_no_es_booleano_sigue_apagado_y_se_avisa(caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert MemoriaConfig.from_settings({"enabled": "true"}).enabled is False
    assert "[memoria].enabled" in caplog.text


@pytest.mark.parametrize("valor, esperado, avisa", [
    ("emiliano", "emiliano", False),
    ("  emiliano  ", "emiliano", False),  # mem0 recorta los bordes: lo mismo aquí
    ("el patron", DEFAULT_USUARIO, True),  # mem0 2.1.0 lo rechaza en CADA llamada
    ("el\tpatron", DEFAULT_USUARIO, True),
    ("   ", DEFAULT_USUARIO, True),
    ("", DEFAULT_USUARIO, False),
])
def test_usuario_con_espacios_se_avisa_y_usa_el_default(caplog, valor, esperado, avisa):
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        cfg = MemoriaConfig.from_settings({"enabled": True, "usuario": valor})
    assert cfg.usuario == esperado
    assert ("[memoria].usuario" in caplog.text) is avisa


def test_un_typo_en_memoria_apagada_no_deja_a_crotolamo_sin_tools(monkeypatch, tmp_path):
    """El escenario del dueño: build_agent llama a get_memoria() dentro de su try
    grande y un ValueError de from_settings lo degradaba a un Agent sin tools."""
    from crotolamo.core.agent import ToolAgent
    from interfaces import shell

    class _Mudo:
        model = "x"

        def chat(self, messages, tools=None):
            raise AssertionError("este test no conversa")

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS", Settings(
        raw={"llm": {"backend": "ollama"}, "memory": {"pin_facts": False},
             "memoria": {"enabled": False, "umbral": "0,03", "top_k": "tres"}},
        user="t", home=tmp_path, paths={"home": tmp_path}))
    monkeypatch.setattr(shell, "build_llm", lambda s: _Mudo())
    agent, _ = shell.build_agent()
    assert isinstance(agent, ToolAgent)
    assert agent.registry.names()
