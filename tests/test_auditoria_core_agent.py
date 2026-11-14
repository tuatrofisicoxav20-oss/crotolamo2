"""Arreglos de la auditoría en el núcleo del agente (core/agent.py y vecinos).

Cada bloque fija un hallazgo confirmado; todo con LLMs falsos, sin Ollama ni red.

- Un LLMError es la RESPUESTA del turno: debe salir por on_token (el shell
  --stream y la voz en streaming solo leen eso) y cerrar el turno en el
  historial con un assistant, igual que el resto de salidas de handle_turn.
- direct_tools=set() apaga el short-circuit ENTERO, también el de las tools
  marcadas con Tool.direct=True (con GLM como cerebro, todo lo redacta él).
- Una respuesta VACÍA del modelo sin haber ejecutado nada no puede salir como
  "Listo, patrón.": ese cierre solo vale si en el turno corrió alguna tool.
- El short-circuit de retorno directo es para cuando esa tool ES el turno: si
  antes ya corrió otra, su resultado no puede quedarse sin contar.
- El pre-hook de hechos no se da por hecho si la lectura de la DB falló.
"""

from __future__ import annotations

import builtins

from crotolamo.core.agent import Agent, ToolAgent
from crotolamo.core.llm import ChatResponse, LLMError, TransientLLMError
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.tools.base import Registry, Tool

_ERROR = "No pude hablar con Ollama, patrón. ¿Está vivo el servicio?"


class _DeadLLM:
    """El motor está caído: toda llamada lanza el error en personaje."""

    model = "x"

    def chat(self, messages, tools=None):
        raise TransientLLMError(_ERROR)

    def chat_stream(self, messages, tools=None, on_token=None):
        raise TransientLLMError(_ERROR)


def _registry(*tools: Tool) -> Registry:
    reg = Registry()
    for t in tools:
        reg.register(t)
    return reg


def _tool(name: str, func, direct: bool = False) -> Tool:
    return Tool(
        name=name, func=func, description=name, direct=direct,
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _tool_call_response(name: str) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[{"name": name, "arguments": {}}],
        raw_message={"tool_calls": [{"function": {"name": name, "arguments": {}}}]},
    )


def _roles(conv: Conversation) -> list[str]:
    return [m.role for m in conv.history]


# --- LLMError: sale por on_token y cierra el turno en el historial --------------

def test_error_del_llm_sale_por_on_token():
    """El caller en streaming (shell --stream, StreamSpeaker) solo lee on_token:
    si el error solo se devuelve, el patrón ve/oye NADA."""
    agent = ToolAgent(_DeadLLM(), Conversation("SYS"), registry=_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    tokens: list[str] = []
    reply = agent.handle_turn("hola", on_token=tokens.append)
    assert reply == _ERROR
    assert tokens == [_ERROR]


def test_error_del_llm_con_tools_a_la_vista_tambien_sale_por_on_token():
    """Con tools a la vista el streamer retiene: el error igual tiene que salir."""
    reg = _registry(_tool("ram_usage", lambda: "RAM: 5G, patrón."))
    agent = ToolAgent(_DeadLLM(), Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    tokens: list[str] = []
    reply = agent.handle_turn("cuánta RAM uso?", on_token=tokens.append)
    assert "".join(tokens) == reply == _ERROR


def test_error_del_llm_cierra_el_turno_en_el_historial():
    """Antes el user quedaba colgado sin assistant: tras N turnos fallidos el
    modelo recibía N órdenes sin contestar de golpe (y podía ejecutarlas tarde)."""
    conv = Conversation("SYS")
    agent = ToolAgent(_DeadLLM(), conv, registry=_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    agent.handle_turn("apaga la compu")
    agent.handle_turn("hola")
    assert _roles(conv) == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in conv.history if m.role == "assistant"] == [_ERROR, _ERROR]


def test_error_del_llm_tras_una_tool_no_deja_el_bloque_tool_abierto():
    """El LLM pide una tool, se ejecuta, y la 2ª llamada revienta: el historial
    no puede acabar en un `tool` sin assistant (mismo invariante que al agotar
    las iteraciones)."""

    class _SeCaeEnLaSegunda:
        model = "x"

        def __init__(self):
            self.llamadas = 0

        def chat(self, messages, tools=None):
            self.llamadas += 1
            if self.llamadas == 1:
                return _tool_call_response("abre_algo")
            raise LLMError("GLM respondió 400, patrón: payload roto")

    reg = _registry(_tool("abre_algo", lambda: "Abierto, patrón."))
    conv = Conversation("SYS")
    agent = ToolAgent(_SeCaeEnLaSegunda(), conv, registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False, direct_tools=set())
    reply = agent.handle_turn("abre eso")
    assert "400" in reply
    assert _roles(conv) == ["user", "assistant", "tool", "assistant"]
    assert conv.history[-1].content == reply


def test_error_a_media_frase_sale_en_linea_aparte():
    """Charla sin tools: el streamer habla en vivo. Si el motor se cae a media
    frase, el error se emite IGUAL (antes: frase truncada y silencio), separado
    de lo ya dicho para que no se pegue a la palabra cortada."""

    class _SeCortaAMedias:
        model = "x"

        def chat_stream(self, messages, tools=None, on_token=None):
            on_token("Hoy amaneció ")
            on_token("bon")
            raise TransientLLMError(_ERROR)

    conv = Conversation("SYS")
    agent = ToolAgent(_SeCortaAMedias(), conv, registry=_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    tokens: list[str] = []
    reply = agent.handle_turn("cómo amaneció?", on_token=tokens.append)
    assert reply == _ERROR
    assert "".join(tokens) == f"Hoy amaneció bon\n{_ERROR}"
    assert _roles(conv) == ["user", "assistant"]


def test_agente_sin_tools_error_sale_por_on_token_y_queda_en_historial():
    """El Agent base (el de respaldo si las tools no arman) tenía el mismo hueco."""
    conv = Conversation("SYS")
    agent = Agent(_DeadLLM(), conv)
    tokens: list[str] = []
    reply = agent.handle_turn("hola", on_token=tokens.append)
    assert reply == _ERROR
    assert tokens == [_ERROR]
    assert _roles(conv) == ["user", "assistant"]


def test_shell_stream_muestra_el_error_del_llm(monkeypatch, capsys):
    """El síntoma que veía el patrón: `shell --stream` imprimía 'Crotolamo > ' vacío."""
    import interfaces.shell as shell

    conv = Conversation("SYS")
    agent = ToolAgent(_DeadLLM(), conv, registry=_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    monkeypatch.setattr(shell, "build_agent", lambda confirm_fn=None: (agent, conv))
    entradas = iter(["hola", "salir"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(entradas))

    assert shell.run_shell(["--stream"]) == 0
    assert f"Crotolamo > {_ERROR}" in capsys.readouterr().out


# --- direct_tools=set() apaga TODO el short-circuit -----------------------------

class _LLMDeGuion:
    """Reproduce una secuencia de respuestas y cuenta las llamadas."""

    model = "x"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        return self._responses.pop(0)

    def chat_stream(self, messages, tools=None, on_token=None):
        # Como Ollama: un tool-call no emite content; un texto final sí.
        resp = self.chat(messages, tools)
        if on_token and resp.content and not resp.tool_calls:
            on_token(resp.content)
        return resp


def test_direct_tools_vacio_apaga_tambien_el_flag_direct():
    """interfaces/shell.py pasa direct_tools=set() con GLM como cerebro ("todo lo
    razona y redacta él"). Una tool con Tool.direct=True se saltaba esa regla."""
    reg = _registry(_tool("presenta_xyz", lambda: "Lista cruda, patrón.", direct=True))
    llm = _LLMDeGuion([
        _tool_call_response("presenta_xyz"),
        ChatResponse(content="Redactado por el modelo, patrón."),
    ])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False, direct_tools=set())
    reply = agent.handle_turn("dame eso")
    assert reply == "Redactado por el modelo, patrón."
    assert llm.calls == 2


def test_list_windows_con_el_cableado_de_glm_lo_redacta_el_modelo(monkeypatch):
    """El caso real: list_windows lleva direct=True y con GLM se leía la lista
    cruda ("workspace 2: firefox — YouTube") con una sola llamada."""
    from crotolamo.tools import default_registry, windows

    monkeypatch.setattr(windows, "_clients", lambda: [
        {"class": "firefox", "title": "YouTube", "workspace": {"id": 2}},
    ])
    llm = _LLMDeGuion([
        _tool_call_response("list_windows"),
        ChatResponse(content="Solo tienes Firefox con YouTube, patrón."),
    ])
    # Exactamente lo que cablea build_agent cuando el cerebro es GLM.
    agent = ToolAgent(llm, Conversation("SYS"), registry=default_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False, direct_tools=set())
    reply = agent.handle_turn("qué ventanas tengo abiertas?")
    assert reply == "Solo tienes Firefox con YouTube, patrón."
    assert llm.calls == 2


def test_flag_direct_sigue_valiendo_con_el_default():
    """Sin tocar direct_tools (motor local), el flag Tool.direct sigue dando
    short-circuit aunque el nombre no esté en DEFAULT_DIRECT_TOOLS."""
    reg = _registry(_tool("presenta_xyz", lambda: "Lista cruda, patrón.", direct=True))
    llm = _LLMDeGuion([_tool_call_response("presenta_xyz")])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    assert agent.handle_turn("dame eso") == "Lista cruda, patrón."
    assert llm.calls == 1


# --- respuesta vacía del modelo: "Listo" solo si en el turno se hizo algo --------

_EN_BLANCO = "Me quedé en blanco, patrón. Repíteme eso."


def test_respuesta_vacia_sin_tools_no_presume_de_listo():
    """El modelo devuelve content vacío y ninguna tool (contexto desbordado, una
    respuesta solo-razonamiento de GLM). Decir "Listo, patrón." es afirmar que
    se hizo algo: no se hizo nada. Mismo texto que el Agent base en ese caso."""
    conv = Conversation("SYS")
    agent = ToolAgent(_LLMDeGuion([ChatResponse(content="")]), conv, registry=_registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    reply = agent.handle_turn("¿cuál es la capital de Francia?")
    assert reply == _EN_BLANCO
    assert conv.history[-1].content == _EN_BLANCO
    # Y es exactamente lo que contesta el Agent sin tools ante lo mismo.
    assert Agent(_LLMDeGuion([ChatResponse(content="")]), Conversation("SYS")).handle_turn(
        "¿cuál es la capital de Francia?") == _EN_BLANCO


def test_respuesta_vacia_sin_tools_en_streaming_tampoco_dice_listo():
    """La voz solo lee on_token: lo que se habla también tiene que ser honesto,
    con tools a la vista (el streamer retiene) y sin ellas (habla en vivo)."""
    for reg in (_registry(), _registry(_tool("abre_algo", lambda: "Abierto, patrón."))):
        agent = ToolAgent(_LLMDeGuion([ChatResponse(content="")]), Conversation("SYS"),
                          registry=reg, guard=Guard(allowed_roots=[]), fastpath=False)
        tokens: list[str] = []
        reply = agent.handle_turn("¿qué hora es en Tokio?", on_token=tokens.append)
        assert "".join(tokens) == reply == _EN_BLANCO


def test_respuesta_vacia_tras_ejecutar_una_tool_sigue_siendo_listo():
    """El caso para el que existe "Listo, patrón.": la tool corrió y el modelo no
    tiene nada que añadir. Eso no cambia."""
    reg = _registry(_tool("abre_algo", lambda: "Abierto, patrón."))
    llm = _LLMDeGuion([_tool_call_response("abre_algo"), ChatResponse(content="")])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    assert agent.handle_turn("abre eso") == "Listo, patrón."
    assert llm.calls == 2


# --- short-circuit: solo si la tool directa ES todo el turno ---------------------

_RAM = "RAM: 5G usados de 15G (33%), patrón."


def test_short_circuit_no_se_come_el_resultado_de_una_tool_anterior():
    """"busca mi tesis y dime cuánta RAM uso": search_files en la 1ª iteración y
    ram_usage sola en la 2ª. Antes el short-circuit devolvía la RAM cruda y el
    resultado de la búsqueda no se le contaba nunca al patrón."""
    reg = _registry(
        _tool("search_files", lambda: "Encontré: /home/u/Documentos/tesis_final.pdf"),
        _tool("ram_usage", lambda: _RAM),
    )
    llm = _LLMDeGuion([
        _tool_call_response("search_files"),
        _tool_call_response("ram_usage"),
        ChatResponse(content="Tu tesis está en Documentos y usas 5G de RAM, patrón."),
    ])
    conv = Conversation("SYS")
    agent = ToolAgent(llm, conv, registry=reg, guard=Guard(allowed_roots=[]), fastpath=False)
    reply = agent.handle_turn("busca mi tesis y dime cuánta RAM uso")
    assert reply == "Tu tesis está en Documentos y usas 5G de RAM, patrón."
    assert llm.calls == 3
    assert _roles(conv) == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


def test_short_circuit_no_tapa_el_fallo_de_otra_tool_del_turno():
    """Si lo anterior fue OTRA tool que reventó, el patrón tiene que enterarse:
    devolver solo la RAM se tragaba el error."""

    def _revienta():
        raise RuntimeError("sin display")

    reg = _registry(_tool("abre_algo", _revienta), _tool("ram_usage", lambda: _RAM))
    llm = _LLMDeGuion([
        _tool_call_response("abre_algo"),
        _tool_call_response("ram_usage"),
        ChatResponse(content="No pude abrirlo, patrón; de RAM vas en 5G."),
    ])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    reply = agent.handle_turn("abre eso y dime la RAM")
    assert reply == "No pude abrirlo, patrón; de RAM vas en 5G."
    assert llm.calls == 3


def test_short_circuit_se_conserva_al_reintentar_la_misma_tool():
    """La tool directa falla duro y el modelo la REINTENTA bien: del intento
    fallido no hay nada que contar, así que el atajo sigue valiendo (2 llamadas,
    no 3: son ~7-9s de CPU)."""
    intentos: list[int] = []

    def _ram():
        intentos.append(1)
        if len(intentos) == 1:
            raise RuntimeError("meminfo ocupado")
        return _RAM

    reg = _registry(_tool("ram_usage", _ram))
    llm = _LLMDeGuion([
        _tool_call_response("ram_usage"),
        _tool_call_response("ram_usage"),
        ChatResponse(content="no debería usarse"),
    ])
    agent = ToolAgent(llm, Conversation("SYS"), registry=reg,
                      guard=Guard(allowed_roots=[]), fastpath=False)
    assert agent.handle_turn("cuánta RAM uso?") == _RAM
    assert llm.calls == 2


# --- pre-hook de hechos: una lectura fallida no cuenta como hecha ----------------

def test_facts_prehook_reintenta_si_la_primera_lectura_falla(monkeypatch):
    """La marca de "ya inyecté" se ponía ANTES de leer la DB: un fallo puntual en
    el primer turno (sqlite bloqueada) dejaba al proceso sin hechos para siempre."""
    from crotolamo.core.hooks import make_facts_prehook
    from crotolamo.persistence import facts

    lecturas: list[int] = []

    def _facts_context(*args, **kwargs):
        lecturas.append(1)
        if len(lecturas) == 1:
            raise OSError("database is locked")
        return "- (general) Su perro se llama Rocky"

    monkeypatch.setattr(facts, "facts_context", _facts_context)
    hook = make_facts_prehook()
    assert hook("hola") == "hola"  # la lectura falló: el turno sigue, sin hechos
    segundo = hook("cómo se llama mi perro")
    assert "Rocky" in segundo and segundo.endswith("cómo se llama mi perro")
    # Inyectados una vez, no se vuelve a leer la DB en cada turno.
    assert hook("gracias") == "gracias"
    assert len(lecturas) == 2


# --- router: las keywords matchean al INICIO de palabra, no a media palabra ------

def test_router_charla_con_keyword_a_media_palabra_no_cuelga_tools():
    """"pon" dentro de "responde", "ram" dentro de "gramática", "olvida" dentro
    de "inolvidable", "prende" dentro de "comprendes": es charla. Con tools
    colgando el turno local pasa de ~1-2s a ~30s, no se habla en vivo y el 3B
    se tienta a llamar una tool (lo que el propio router dice querer evitar)."""
    from crotolamo.core.router import select_tool_names

    for charla in (
        "responde más corto",
        "explícame la gramática del inglés",
        "fue una noche inolvidable",
        "¿me comprendes?",
        "no sabré qué decirte",
    ):
        assert select_tool_names(charla) == [], charla


def test_router_keyword_a_media_palabra_no_arrastra_otro_grupo():
    """La orden sí lleva tools, pero no las del grupo que solo matcheaba de
    rebote: "tema" en "sistema", "ram" en "telegram", "camara" en "recámara"."""
    from crotolamo.core.router import select_tool_names

    sistema = select_tool_names("qué opinas del sistema solar")
    assert "system_status" in sistema and "music_control" not in sistema

    telegram = select_tool_names("abre telegram")
    assert "open_app" in telegram and "ram_usage" not in telegram

    recamara = select_tool_names("prende la luz de la recámara")
    assert "light_control" in recamara and "camera_events" not in recamara


def test_router_prefijos_y_encliticos_siguen_entrando():
    """Solo se exige el INICIO de palabra: el final queda libre, porque las
    keywords son prefijos a propósito ("tengo abiert", "compu") y el español
    pega el pronombre al imperativo ("ponme", "ábrelo", "ciérrala")."""
    from crotolamo.core.router import select_tool_names

    assert "music_control" in select_tool_names("ponme algo de música")
    assert "open_app" in select_tool_names("ábrelo otra vez")
    assert "close_window" in select_tool_names("ciérrala ya")
    assert "light_control" in select_tool_names("préndela")
    assert "list_windows" in select_tool_names("qué tengo abierto ahorita")
    assert "system_status" in select_tool_names("cómo anda la computadora")
    # Tras un signo o un separador también empieza palabra.
    assert "read_file" in select_tool_names("¿lee tareas.md?")
    assert "send_hotkey" in select_tool_names("haz ctrl+alt+tab")


def test_router_variantes_con_prefijo_que_entraban_de_rebote_siguen_enrutando():
    """"anota" llegaba a create_note solo porque contiene "nota", y "reabre" a
    open_app porque contiene "abre". Siguen siendo órdenes: ahora van declaradas."""
    from crotolamo.core.router import select_tool_names

    assert "create_note" in select_tool_names("anota comprar leche")
    assert "create_note" in select_tool_names("anótame la cita del martes")
    assert "open_app" in select_tool_names("reabre firefox")
    assert "open_app" in select_tool_names("relanza steam")
    assert "write_file" in select_tool_names("sobrescribe config.toml con esto")
    assert "list_dir" in select_tool_names("dime qué tiene esa subcarpeta")


def test_router_grupo_dinamico_tambien_matchea_al_inicio_de_palabra(monkeypatch):
    """Los grupos MCP usan el mismo matcher: "eco" no puede saltar con "seco"."""
    from crotolamo.core import router

    # Un grupo MCP matcheado avanza el round-robin global: que el test lo devuelva.
    monkeypatch.setattr(router, "_RR_COUNTER", router._RR_COUNTER)
    router.register_group("mcp:auditoria_eco", ["mcp_eco_1"], ["eco", ".cfg"])
    try:
        assert router.select_tool_names("qué seco está el clima") == []
        assert router.select_tool_names("hazme eco de esto") == ["mcp_eco_1"]
        # Una keyword que no empieza por letra no tiene "inicio de palabra" que
        # exigir: sigue valiendo como subcadena.
        assert router.select_tool_names("revisa ajustes.cfg") == ["mcp_eco_1"]
    finally:
        router.unregister_group("mcp:auditoria_eco")


# --- router: "abre la carpeta X" tiene que ofrecer open_folder -------------------

_FILES = ["create_note", "read_file", "write_file", "list_dir", "make_dir",
          "move_file", "delete_file", "search_files"]


def test_router_abre_la_carpeta_ofrece_open_folder():
    """La frase de bandera del proyecto. "carpeta" (files, 7 letras) le ganaba a
    "abre" (desktop, 4) y las 8 tools de files llenaban el tope de 8: al modelo
    local nunca le llegaba open_folder (ni open_app ni open_url)."""
    from crotolamo.core.router import route_schemas, select_tool_names
    from crotolamo.tools import default_registry

    for frase in (
        "abre la carpeta de descargas",
        "ábreme la carpeta de documentos",
        "abre mi carpeta de música",
        "puedes abrir la carpeta de imágenes",
    ):
        assert "open_folder" in select_tool_names(frase), frase

    # Y llega hasta las schemas que ve el modelo, dentro del tope.
    schemas = route_schemas(default_registry(), "abre la carpeta de descargas")
    nombres = [s["function"]["name"] for s in schemas]
    assert "open_folder" in nombres and len(nombres) <= 8


def test_router_las_ordenes_de_archivos_sobre_una_carpeta_no_cambian():
    """El arreglo va por la frase completa ("abre la carpeta"): sin el verbo de
    abrir, "carpeta" sigue siendo de files con sus 8 tools, como antes."""
    from crotolamo.core.router import select_tool_names

    for frase in (
        "qué hay en la carpeta de descargas",
        "crea una carpeta que se llame fotos",
        "borra la carpeta vieja",
        "en qué carpeta guardé la tesis",
    ):
        assert select_tool_names(frase) == _FILES, frase


# --- router: el tope de max_tools cuenta solo las tools REGISTRADAS -------------

_FACTS = ["remember_fact", "recall_facts", "search_facts", "forget_fact"]
_MEMORIA = ["recordar_de_mi", "buscar_recuerdos", "olvidar_recuerdo"]


def _registry_con_una_familia(presente: list[str], ausente: list[str]) -> Registry:
    """Copia del registry real con UNA sola familia de "recordar", como en
    producción (tools/__init__.py registra la de mem0 y retira la de SQLite, o
    deja la de SQLite). Aislada: no depende de [memoria].enabled del equipo."""
    from crotolamo.tools import default_registry

    reg = default_registry().copy()
    for name in ausente:
        reg.unregister(name)
    for name in presente:
        if reg.get(name) is None:
            reg.register(_tool(name, lambda: "ok"))
    return reg


def _enviadas(reg: Registry, frase: str, max_tools: int = 8) -> list[str]:
    from crotolamo.core.router import route_schemas

    return [s["function"]["name"] for s in route_schemas(reg, frase, max_tools)]


def test_router_la_familia_de_recordar_ausente_no_ocupa_cupo():
    """"memoria" y "facts" comparten disparadores y solo una está registrada. El
    tope se aplicaba ANTES de filtrar por registry: los 3 nombres fantasma de
    mem0 ocupaban hueco y de files solo llegaba create_note (5 tools de 8)."""
    reg = _registry_con_una_familia(presente=_FACTS, ausente=_MEMORIA)
    nombres = _enviadas(reg, "acuérdate de la nota del archivo")
    # Las schemas van en el orden del registry (bytes estables, ver route_schemas).
    esperadas = set(_FACTS + ["create_note", "read_file", "write_file", "list_dir"])
    assert nombres == [n for n in reg.names() if n in esperadas]


def test_router_remember_fact_llega_si_sobra_cupo():
    """El caso que rompía la orden: 6 de projects + 2 fantasmas llenaban el tope
    y remember_fact quedaba fuera con 2 huecos libres; el modelo no podía guardar
    el hecho que el patrón le estaba dictando."""
    reg = _registry_con_una_familia(presente=_FACTS, ausente=_MEMORIA)
    nombres = _enviadas(
        reg, "acuérdate de que mi proyecto principal es Huevonitis y lista mis proyectos")
    assert "remember_fact" in nombres
    assert "list_projects" in nombres and len(nombres) == 8


def test_router_con_mem0_los_nombres_de_sqlite_tampoco_ocupan_cupo():
    """El espejo: con la memoria semántica registrada, los 4 nombres de hechos
    SQLite (retirados del registry) eran los fantasmas."""
    reg = _registry_con_una_familia(presente=_MEMORIA, ausente=_FACTS)
    nombres = _enviadas(reg, "acuérdate de la nota del archivo")
    esperadas = set(_MEMORIA + ["create_note", "read_file", "write_file", "list_dir", "make_dir"])
    assert nombres == [n for n in reg.names() if n in esperadas]


def test_router_el_tope_se_sigue_respetando_con_el_filtro():
    reg = _registry_con_una_familia(presente=_FACTS, ausente=_MEMORIA)
    for tope in (1, 3, 8):
        nombres = _enviadas(reg, "acuérdate de la nota del archivo", max_tools=tope)
        assert len(nombres) == tope
        assert set(nombres) <= set(reg.names())


def test_select_tool_names_sin_registry_no_cambia():
    """Quien solo inspecciona (scripts/stt_harness.py, los tests) llama sin
    registry: mismos nombres y mismo tope que siempre, ambas familias incluidas."""
    from crotolamo.core.router import select_tool_names

    nombres = select_tool_names("acuérdate de la nota del archivo")
    assert nombres == _MEMORIA + _FACTS + ["create_note"]
    # Y con el conjunto disponible, el filtro va ANTES de contar.
    solo_facts = select_tool_names("acuérdate de la nota del archivo", 5,
                                   available=_FACTS + ["create_note", "read_file"])
    assert solo_facts == _FACTS + ["create_note"]
    assert select_tool_names("acuérdate de eso", available=[]) == []


# --- fastpath: sin las alternativas redundantes, los mismos atajos ---------------

def test_fastpath_formas_cortas_siguen_siendo_atajo():
    """Se quitaron `|^siguiente$` y `|^que suena$` (ya los cubría la primera
    alternativa de su regla) y el '.' de la clase de "abre <app>" (_norm nunca
    deja un punto). Nada de eso matcheaba por su cuenta: los atajos no cambian."""
    from crotolamo.core import fastpath

    assert fastpath.match("Siguiente.") == ("music_control", {"action": "next"})
    assert fastpath.match("sáltala") == ("music_control", {"action": "next"})
    assert fastpath.match("¿Qué suena?") == ("music_now", {})
    assert fastpath.match("qué rola suena ahorita") == ("music_now", {})
    # El punto final de Whisper se va en _norm, antes de la regex.
    assert fastpath.match("Abre Blender.") == ("open_app", {"name": "blender"})
    assert fastpath.match("abre notas.txt") is None


# --- router: en un server MCP más grande que el tope, entra la tool que se nombra --

_SRV = [f"mcp_srv_tool_{i}" for i in range(10)] + [
    "mcp_srv_pong_recibido", "mcp_srv_create_pull_request", "mcp_srv_list_pull_requests"]
# Como las deriva el bridge: nombre del server y trozos de 5+ letras de sus tools.
_SRV_KEYWORDS = ["srv", "recibido", "create", "request", "requests"]


def _grupo_srv(monkeypatch):
    from crotolamo.core import router

    # Un grupo MCP matcheado avanza el round-robin global: que el test lo devuelva.
    monkeypatch.setattr(router, "_RR_COUNTER", router._RR_COUNTER)
    router.register_group("mcp:auditoria_srv", list(_SRV), _SRV_KEYWORDS)
    return router


def test_router_mcp_la_tool_nombrada_entra_aunque_quede_tras_el_tope(monkeypatch):
    """Un server trae TODAS sus tools en un solo grupo, en el orden de tools/list,
    y el tope corta por el final: con 13 tools y max_tools=8 las 5 últimas no le
    llegaban nunca al modelo local, ni cuando la frase decía su nombre ("recibido"
    es una keyword derivada del nombre de `pong_recibido` y aun así se ofrecían
    las 8 primeras, sin ella)."""
    router = _grupo_srv(monkeypatch)
    try:
        nombres = router.select_tool_names("srv acuse recibido", 8)
        assert "mcp_srv_pong_recibido" in nombres and len(nombres) == 8
        # La que más palabras propias comparte con la frase va primero.
        pulls = router.select_tool_names("lista los pull requests de srv", 8)
        assert pulls[:2] == ["mcp_srv_list_pull_requests", "mcp_srv_create_pull_request"]
        # Y llega hasta las schemas que ve el modelo.
        reg = _registry(*[_tool(n, lambda: "ok") for n in _SRV])
        schemas = router.route_schemas(reg, "srv acuse recibido", 8)
        assert "mcp_srv_pong_recibido" in [s["function"]["name"] for s in schemas]
    finally:
        router.unregister_group("mcp:auditoria_srv")


def test_router_mcp_sin_palabra_propia_de_ninguna_tool_el_orden_no_cambia(monkeypatch):
    """Si la frase solo nombra al server (o a todas por igual, vía el prefijo), el
    orden es el de tools/list, exactamente como antes."""
    router = _grupo_srv(monkeypatch)
    try:
        assert router.select_tool_names("pregúntale a srv", 8) == _SRV[:8]
        assert router.select_tool_names("pregúntale a srv", 20) == _SRV
    finally:
        router.unregister_group("mcp:auditoria_srv")


# --- código muerto de la superficie de tools: sin llamadores, fuera ---------------

def test_api_muerta_del_nucleo_no_vuelve():
    """Cero llamadores en crotolamo/, interfaces/, desktop/, scripts/ y tests/:
    router.names_for (su docstring decía "usado en tests" y no lo estaba), el
    re-export agent._HARD_ERROR_PREFIXES (quien lo necesita lo importa de
    tool_parsing) y Conversation.set_system (el system prompt se fija al crear
    la Conversation; nadie lo cambiaba en caliente)."""
    from crotolamo.core import agent, router

    assert not hasattr(router, "names_for")
    assert not hasattr(agent, "_HARD_ERROR_PREFIXES")
    assert not hasattr(Conversation, "set_system")


def test_el_system_prompt_es_el_de_la_construccion():
    """Lo que sí se usa, igual que siempre: el system prompt del constructor
    encabeza el payload y reset() no lo toca."""
    conv = Conversation("eres Crotolamo", max_turns=2)
    conv.add_user("hola")
    conv.reset()
    assert conv.to_messages() == [{"role": "system", "content": "eres Crotolamo"}]
