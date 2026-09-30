"""Tests del puente MCP: registro, política de confirmación, strikes, router,
guard recursivo y config. Todo contra el fake server (proceso real, sin red).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import router
from crotolamo.core.agent import ToolAgent
from crotolamo.core.memory import Conversation
from crotolamo.core.tool_parsing import is_hard_error
from crotolamo.mcp import bridge
from crotolamo.safety.guard import Guard
from crotolamo.settings import Settings, _deep_merge
from crotolamo.tools.base import Registry, Tool

FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Ningún server ni grupo dinámico sobrevive de un test a otro."""
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)
    monkeypatch.setattr(router, "_RR_COUNTER", 0)
    yield
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)


def mk_settings(servers: dict, **mcp) -> Settings:
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 5, "servers": servers}
    raw.update(mcp)
    return Settings(raw={"mcp": raw}, user="test", home=Path("/tmp"))


def fake_server(**extra) -> dict:
    return {"command": FAKE_CMD, **extra}


def registrar(servers: dict, **mcp) -> tuple[Registry, list[str]]:
    registry = Registry()
    names = bridge.register_mcp_tools(registry, mk_settings(servers, **mcp))
    return registry, names


# ---------------------------------------------------------------------------
# Registro y nombres
# ---------------------------------------------------------------------------

def test_registra_tools_con_prefijo_y_metadatos():
    registry, names = registrar({"fake": fake_server()})
    assert "mcp_fake_echo" in names
    assert set(names) == set(registry.names())
    echo = registry.get("mcp_fake_echo")
    assert echo is not None
    assert echo.description.endswith("[MCP: fake]")
    assert echo.strict_args is False
    assert echo.direct is False
    # inputSchema normalizado: sin $schema, con type/properties, y lo anidado intacto.
    assert "$schema" not in echo.parameters
    assert echo.parameters["type"] == "object"
    assert echo.parameters["properties"]["opciones"]["type"] == "object"
    assert echo.parameters["additionalProperties"] is True
    assert bridge.connected_servers() == ["fake"]
    assert bridge._ATEXIT_REGISTERED is True


def test_nombres_saneados_recortados_y_sin_colisiones():
    registry, names = registrar({"mi-server": fake_server()})
    assert "mcp_mi_server_echo" in names
    assert "mcp_mi_server_Nombre_raro_x" in names
    # a.b y a-b sanean igual: el segundo lleva sufijo.
    assert "mcp_mi_server_a_b" in names
    assert "mcp_mi_server_a_b_2" in names
    for name in names:
        assert len(name) <= 64
        assert all(ch.isalnum() or ch == "_" for ch in name)
    largo = [n for n in names if n.startswith("mcp_mi_server_tool_con_un_nombre")]
    assert len(largo) == 1 and len(largo[0]) == 64
    # Descripción larga recortada a ~300 + sufijo del server.
    tool = registry.get(largo[0])
    assert tool is not None and len(tool.description) < 330


def test_prefix_explicito():
    _, names = registrar({"fake": fake_server(prefix="fs")})
    assert "mcp_fs_echo" in names


def test_una_llamada_de_verdad_pasa_por_el_registry():
    registry, _ = registrar({"fake": fake_server()})
    out = registry.run("mcp_fake_echo", {"text": "hola"})
    assert out == '{"text": "hola"}'
    assert not is_hard_error(out)


def test_es_idempotente_no_relanza_ni_duplica():
    registry, names = registrar({"fake": fake_server()})
    client = bridge._STATE["fake"].client
    again = bridge.register_mcp_tools(registry, mk_settings({"fake": fake_server()}))
    assert sorted(again) == sorted(names)
    assert len(registry.names()) == len(names)
    assert bridge._STATE["fake"].client is client  # mismo proceso
    # En un registry NUEVO repone las tools sin lanzar otro proceso.
    otro = Registry()
    bridge.register_mcp_tools(otro, mk_settings({"fake": fake_server()}))
    assert set(otro.names()) == set(names)
    assert bridge._STATE["fake"].client is client


def test_unregister_server_quita_tools_grupo_y_cierra():
    registry, names = registrar({"fake": fake_server()})
    client = bridge._STATE["fake"].client
    assert bridge.unregister_server(registry, "fake") is True
    assert registry.names() == []
    assert "mcp:fake" not in router.dynamic_groups()
    assert not client.alive
    assert bridge.unregister_server(registry, "fake") is False  # idempotente


def test_registry_unregister():
    registry = Registry()
    registry.register(Tool(name="t", func=lambda: "ok", description="d", parameters={}))
    assert registry.unregister("t") is True
    assert registry.unregister("t") is False
    assert registry.get("t") is None


def test_server_que_no_arranca_no_impide_los_demas(caplog):
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    servers = {
        "roto": {"command": ["/no/existe/binario"]},
        "muere": {"command": [sys.executable, "-c", "import sys; sys.exit(3)"]},
        "fake": fake_server(),
    }
    registry, names = registrar(servers, startup_timeout_s=3)
    assert "mcp_fake_echo" in names
    assert not any(n.startswith("mcp_roto_") or n.startswith("mcp_muere_") for n in names)
    assert bridge.connected_servers() == ["fake"]
    assert "'roto'" in caplog.text and "'muere'" in caplog.text


def test_desactivado_no_lanza_nada():
    registry, names = registrar({"fake": fake_server()}, enabled=False)
    assert names == [] and registry.names() == []
    assert bridge.connected_servers() == []


# ---------------------------------------------------------------------------
# Política de confirmación
# ---------------------------------------------------------------------------

def _safe(registry: Registry, name: str) -> bool:
    tool = registry.get(name)
    assert tool is not None
    return tool.safe


def test_confirm_destructive_por_defecto_respeta_los_hints():
    registry, _ = registrar({"fake": fake_server()})
    assert _safe(registry, "mcp_fake_echo") is True        # readOnlyHint: true
    assert _safe(registry, "mcp_fake_borrar") is False     # destructiveHint: true
    assert _safe(registry, "mcp_fake_sin_hints") is False  # sin annotations => pregunta


def test_confirm_always_todo_pide_confirmacion():
    registry, names = registrar({"fake": fake_server()}, confirm="always")
    assert all(_safe(registry, n) is False for n in names)


def test_confirm_never_nada_pide_confirmacion():
    registry, names = registrar({"fake": fake_server()}, confirm="never")
    assert all(_safe(registry, n) is True for n in names)


def test_confirm_por_server_pisa_el_global():
    registry, names = registrar(
        {"fake": fake_server(confirm="always"), "otro": fake_server(confirm="never")},
        confirm="never",
    )
    assert all(_safe(registry, n) is False for n in names if n.startswith("mcp_fake_"))
    assert all(_safe(registry, n) is True for n in names if n.startswith("mcp_otro_"))


def test_server_mentiroso_se_salta_la_confirmacion_bajo_destructive():
    """DOCUMENTA el comportamiento esperado (ver README_M4): los hints los declara
    EL SERVER y son advisory. `borrar_mentiroso` BORRA pero jura readOnlyHint=true:
    bajo "destructive" corre sin preguntar. Contra servers así: confirm="always"."""
    registry, _ = registrar({"fake": fake_server()})
    assert _safe(registry, "mcp_fake_borrar_mentiroso") is True  # se lo creyó

    bridge.close_all()
    registry, _ = registrar({"fake": fake_server(confirm="always")})
    assert _safe(registry, "mcp_fake_borrar_mentiroso") is False  # única defensa real


def test_tool_is_safe_tabla():
    assert bridge.tool_is_safe("destructive", {"readOnlyHint": True}) is True
    assert bridge.tool_is_safe("destructive", {"destructiveHint": False}) is True
    assert bridge.tool_is_safe("destructive", {"destructiveHint": True}) is False
    assert bridge.tool_is_safe("destructive", None) is False
    assert bridge.tool_is_safe("destructive", {"readOnlyHint": "true"}) is False  # no es bool
    assert bridge.tool_is_safe("always", {"readOnlyHint": True}) is False
    assert bridge.tool_is_safe("never", {"destructiveHint": True}) is True


# ---------------------------------------------------------------------------
# strict_args y guard recursivo (de punta a punta con ToolAgent._execute_call)
# ---------------------------------------------------------------------------

def _agent(registry: Registry, tmp_path: Path) -> ToolAgent:
    return ToolAgent(
        llm=None, conversation=Conversation("SYS"), registry=registry,
        guard=Guard(allowed_roots=[tmp_path / "libre"], confirm_roots=[tmp_path / "confirmable"]),
        confirm_fn=lambda _reason: False,
    )


def test_strict_args_false_deja_pasar_kwargs_anidados_y_extra(tmp_path):
    registry, _ = registrar({"fake": fake_server()})
    registry.register(Tool(
        name="plana", func=lambda **kw: str(sorted(kw)), description="d",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
    ))
    agent = _agent(registry, tmp_path)
    args = {"text": "hola", "opciones": {"mayusculas": True}, "extra": 1}
    out = agent._execute_call("mcp_fake_echo", dict(args))
    assert out == '{"extra": 1, "opciones": {"mayusculas": true}, "text": "hola"}'
    # Una tool normal (strict_args=True) sigue filtrando lo no declarado.
    assert agent._execute_call("plana", {"a": "x", "extra": 1}) == "['a']"


def test_guard_recursivo_bloquea_rutas_anidadas_y_permite_las_del_corral(tmp_path):
    libre = tmp_path / "libre"
    guard = Guard(allowed_roots=[libre], confirm_roots=[tmp_path / "confirmable"])
    tool = Tool(name="t", func=lambda **k: "ok", description="d", parameters={})

    # Lista bajo un nombre de ruta plural: una fuera del corral bloquea todo.
    assert guard.check(tool, {"paths": [str(libre / "a.txt"), "/etc/passwd"]}).allowed is False
    assert guard.check(tool, {"paths": [str(libre / "a.txt"), str(libre / "b")]}).allowed is True
    # Dict anidado con clave de ruta.
    assert guard.check(tool, {"opciones": {"ruta": "/etc/shadow"}}).allowed is False
    assert guard.check(tool, {"opciones": {"ruta": str(libre / "x")}}).allowed is True
    # Lista de dicts (JSON típico de un server MCP), sin nombre de ruta pero con
    # valor que parece un path.
    assert guard.check(tool, {"items": [{"nombre": "x"}, {"nombre": "../../etc"}]}).allowed is False
    # Zona de confirmación también aplica anidada.
    d = guard.check(tool, {"batch": [{"path": str(tmp_path / "confirmable" / "n.md")}]})
    assert d.allowed and d.needs_confirmation
    # Sin rutas por ningún lado: pasa.
    assert guard.check(tool, {"query": "hola", "n": 3, "tags": ["a", "b"]}).allowed is True


def test_guard_recursivo_para_en_profundidad_8(tmp_path):
    """El tope existe para que un JSON patológico no reviente la pila: más hondo
    de 8 niveles ya no se inspecciona (documentado en README_M4)."""
    guard = Guard(allowed_roots=[tmp_path])
    tool = Tool(name="t", func=lambda **k: "ok", description="d", parameters={})

    def anidar(niveles: int, hoja):
        value = hoja
        for _ in range(niveles):
            value = {"path": value}
        return value

    assert guard.check(tool, anidar(6, "/etc/passwd")).allowed is False
    assert guard.check(tool, anidar(12, "/etc/passwd")).allowed is True


def test_guard_bloquea_una_tool_mcp_antes_de_llegar_al_server(tmp_path):
    registry, _ = registrar({"fake": fake_server(confirm="never")})
    agent = _agent(registry, tmp_path)
    out = agent._execute_call("mcp_fake_borrar", {"path": "/etc/passwd"})
    assert "corral" in out.lower() or "permitidas" in out.lower()
    out = agent._execute_call("mcp_fake_borrar", {"path": str(tmp_path / "libre" / "x")})
    assert out.startswith("borrado ")


def test_tool_mcp_no_safe_pide_confirmacion(tmp_path):
    registry, _ = registrar({"fake": fake_server()})
    agent = _agent(registry, tmp_path)
    out = agent._execute_call("mcp_fake_sin_hints", {"x": "1"})
    assert out == "Cancelado por el patrón."


# ---------------------------------------------------------------------------
# Timeouts, strikes y transporte
# ---------------------------------------------------------------------------

def test_timeout_es_soft_error_y_el_segundo_seguido_desregistra(caplog):
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    registry, _ = registrar({"fake": fake_server(timeout_s=0.3)})
    client = bridge._STATE["fake"].client

    out = registry.run("mcp_fake_lenta", {"segundos": 1})
    assert out.startswith("El server MCP 'fake'")
    assert "patrón" in out
    assert not is_hard_error(out)
    assert bridge.strikes_for("fake") == 1
    assert registry.get("mcp_fake_lenta") is not None  # sigue registrada

    out = registry.run("mcp_fake_lenta", {"segundos": 1})
    assert out.startswith("El server MCP 'fake'")
    assert not is_hard_error(out)
    assert registry.get("mcp_fake_lenta") is None      # TODAS fuera
    assert registry.names() == []
    assert "mcp:fake" not in router.dynamic_groups()
    assert bridge.connected_servers() == []
    assert not client.alive
    assert "timeouts seguidos" in caplog.text


def test_exito_resetea_los_strikes():
    registry, _ = registrar({"fake": fake_server(timeout_s=0.3)})
    assert registry.run("mcp_fake_lenta", {"segundos": 0.5}).startswith("El server MCP")
    assert bridge.strikes_for("fake") == 1
    time.sleep(0.4)  # el fake es monohilo: que termine de dormir
    assert registry.run("mcp_fake_echo", {"text": "ok"}) == '{"text": "ok"}'
    assert bridge.strikes_for("fake") == 0
    # Otro timeout vuelve a ser el PRIMERO: no desregistra.
    assert registry.run("mcp_fake_lenta", {"segundos": 0.5}).startswith("El server MCP")
    assert bridge.strikes_for("fake") == 1
    assert registry.get("mcp_fake_echo") is not None


def test_transporte_roto_desregistra_de_inmediato(caplog):
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    registry, _ = registrar({"fake": fake_server()})
    out = registry.run("mcp_fake_morir", {})
    assert out.startswith("El server MCP 'fake'")
    assert not is_hard_error(out)
    assert registry.names() == []
    assert bridge.connected_servers() == []
    assert "mcp:fake" not in router.dynamic_groups()
    assert "transporte roto" in caplog.text


def test_is_error_del_server_es_soft_error_con_su_texto():
    registry, _ = registrar({"fake": fake_server()})
    out = registry.run("mcp_fake_falla", {})
    assert out.startswith("El server MCP 'fake'")
    assert "propósito" in out
    assert not is_hard_error(out)


def test_tool_de_server_ya_desconectado_responde_en_personaje():
    registry, _ = registrar({"fake": fake_server()})
    echo = registry.get("mcp_fake_echo")
    assert echo is not None
    bridge.unregister_server(registry, "fake")
    out = echo.run({"text": "x"})  # alguien se quedó con la Tool en la mano
    assert out.startswith("El server MCP 'fake'") and "patrón" in out


def test_soft_errors_no_son_hard_errors():
    from crotolamo.core.tool_parsing import HARD_ERROR_PREFIXES

    msg = bridge._soft("x", "tardó demasiado, patrón.")
    assert not msg.startswith(HARD_ERROR_PREFIXES)
    assert not is_hard_error(msg)


# ---------------------------------------------------------------------------
# Router: grupos dinámicos y round-robin
# ---------------------------------------------------------------------------

def test_registra_grupo_en_el_router_con_keywords_de_la_config():
    registrar({"fake": fake_server(keywords=["Cosas Raras", "échale"])})
    grupo = router.dynamic_groups()["mcp:fake"]
    assert grupo["keywords"] == ["cosas raras", "echale"]  # normalizadas
    assert "mcp_fake_echo" in grupo["tools"]
    assert "mcp_fake_echo" in router.select_tool_names("échale ganas")


def test_keywords_por_defecto_derivan_del_server_y_sus_tools():
    registrar({"fake": fake_server()})
    kws = router.dynamic_groups()["mcp:fake"]["keywords"]
    assert "fake" in kws
    assert "echo" in kws and "borrar" in kws and "sin" in kws and "hints" in kws
    assert len(kws) <= bridge.MAX_DERIVED_KEYWORDS
    assert "mcp_fake_echo" in router.select_tool_names("usa el server fake")


def test_round_robin_solo_entre_grupos_mcp_matcheados():
    router.register_group("mcp:a", ["a1", "a2"], ["zutano"])
    router.register_group("mcp:b", ["b1", "b2"], ["zutano"])
    assert router.select_tool_names("zutano", max_tools=2) == ["a1", "a2"]
    assert router.select_tool_names("zutano", max_tools=2) == ["b1", "b2"]
    assert router.select_tool_names("zutano", max_tools=2) == ["a1", "a2"]
    # Sin tope, los dos entran; solo cambia el orden.
    assert router.select_tool_names("zutano") == ["b1", "b2", "a1", "a2"]


def test_sin_grupos_mcp_matcheados_el_routing_es_identico_y_no_rota():
    router.register_group("mcp:a", ["a1"], ["zutano"])
    before = router._RR_COUNTER
    base = router.select_tool_names("cuanta RAM estoy usando?")
    assert router.select_tool_names("cuanta RAM estoy usando?") == base
    assert "ram_usage" in base and "a1" not in base
    assert router._RR_COUNTER == before  # el contador no avanzó
    # Charla: sigue sin tools.
    assert router.select_tool_names("hola crotolamo, como estas?") == []


def test_estaticos_conservan_su_orden_y_los_mcp_rotan_en_sus_huecos():
    router.register_group("mcp:a", ["a1"], ["ram"])
    router.register_group("mcp:b", ["b1"], ["ram"])
    first = router.select_tool_names("cuanta ram")
    second = router.select_tool_names("cuanta ram")
    # El grupo estático system matchea más ("ram" + "cuanta"... al menos igual)
    # y va primero en ambas; solo a1/b1 intercambian lugar.
    assert first[:4] == second[:4] == ["disk_usage", "ram_usage", "list_processes", "system_status"]
    assert first[4:] == ["a1", "b1"] and second[4:] == ["b1", "a1"]


def test_register_group_rechaza_nombres_estaticos_y_unregister_es_idempotente():
    with pytest.raises(ValueError):
        router.register_group("files", ["x"], ["y"])
    router.register_group("mcp:z", ["x"], ["y"])
    assert router.unregister_group("mcp:z") is True
    assert router.unregister_group("mcp:z") is False


# ---------------------------------------------------------------------------
# Config: tablas nombradas, merge con local.toml y validación
# ---------------------------------------------------------------------------

def test_servers_como_tablas_nombradas_se_fusionan_con_el_local():
    base = {"mcp": {"enabled": False, "servers": {
        "archivos": {"command": ["npx", "server-filesystem", "~/Documentos"], "timeout_s": 20},
    }}}
    local = {"mcp": {"enabled": True, "servers": {
        "archivos": {"timeout_s": 5},                      # retoca sin repetir command
        "jira": {"command": ["jira-mcp"], "confirm": "always"},  # añade otro
    }}}
    cfg = bridge.load_mcp_config(_deep_merge(base, local)["mcp"])
    assert cfg.enabled is True
    by_name = {s.name: s for s in cfg.servers}
    assert set(by_name) == {"archivos", "jira"}
    assert by_name["archivos"].timeout_s == 5
    assert by_name["archivos"].command[1] == "server-filesystem"
    assert by_name["jira"].confirm == "always"


def test_load_mcp_config_expande_y_valida(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="crotolamo.mcp.bridge")
    monkeypatch.setenv("CROTO_TEST_DIR", "/opt/x")
    monkeypatch.setenv("CROTO_TEST_TOKEN", "secreto")
    cfg = bridge.load_mcp_config(mk_settings({
        "ok": {
            "command": ["srv", "~/Documentos", "$CROTO_TEST_DIR/bin"],
            "env": {"TOKEN": "$CROTO_TEST_TOKEN", "N": 3, "MAL": {"x": 1}},
            "cwd": "~", "keywords": ["a", 2, ""], "confirm": "raro",
            "timeout_s": -1, "startup_timeout_s": "x",
        },
        "sin_command": {"env": {}},
        "command_vacio": {"command": []},
        "command_no_lista": {"command": "srv"},
        "no_tabla": 42,
    }, timeout_s=7, confirm="never"))
    assert [s.name for s in cfg.servers] == ["ok"]
    ok = cfg.servers[0]
    assert ok.command[1] == str(Path("~/Documentos").expanduser())
    assert ok.command[2] == "/opt/x/bin"
    assert ok.env == {"TOKEN": "secreto", "N": "3"}
    assert ok.cwd == str(Path("~").expanduser())
    assert ok.keywords == ["a"]
    assert ok.confirm == "never"      # inválido => hereda el global
    assert ok.timeout_s == 7          # inválido => hereda el global
    assert ok.startup_timeout_s == 5  # inválido => hereda el global
    assert ok.prefix == "ok"
    for name in ("sin_command", "command_vacio", "command_no_lista", "no_tabla"):
        assert name in caplog.text


def test_load_mcp_config_defaults_sin_seccion():
    cfg = bridge.load_mcp_config(Settings(raw={}, user="t", home=Path("/tmp")))
    assert cfg.enabled is False
    assert cfg.timeout_s == bridge.DEFAULT_TIMEOUT_S
    assert cfg.startup_timeout_s == bridge.DEFAULT_STARTUP_TIMEOUT_S
    assert cfg.confirm == "destructive"
    assert cfg.servers == []


def test_settings_mcp_property_y_toml_de_ejemplo():
    real = settings_mod.load_settings()
    assert real.mcp.get("enabled") is False  # apagado por defecto en el toml
    assert real.mcp.get("confirm") == "destructive"


def test_default_registry_registra_mcp_solo_si_enabled(monkeypatch):
    from crotolamo.tools import GLOBAL_REGISTRY, default_registry

    real = settings_mod.get_settings()
    monkeypatch.setitem(real.raw, "mcp", {"enabled": False, "servers": {"fake": fake_server()}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    default_registry()
    assert not any(n.startswith("mcp_") for n in GLOBAL_REGISTRY.names())

    monkeypatch.setitem(real.raw, "mcp", {
        "enabled": True, "startup_timeout_s": 5, "servers": {"fake": fake_server()},
    })
    try:
        default_registry()
        assert "mcp_fake_echo" in GLOBAL_REGISTRY.names()
        default_registry()  # idempotente
        assert GLOBAL_REGISTRY.names().count("mcp_fake_echo") == 1
    finally:
        bridge.close_all()
    assert not any(n.startswith("mcp_") for n in GLOBAL_REGISTRY.names())
