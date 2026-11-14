"""Fase 6: streaming, TTS por frases, memoria fuzzy, búsqueda y hotkeys."""

import pytest

from crotolamo.core.agent import ToolAgent, _LiveStreamer
from crotolamo.core.llm import ChatResponse, LLMClient
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Guard
from crotolamo.tools import default_registry, desktop


# --- streaming: parser de JSONL ---
def test_consume_stream_parses_and_emits():
    lines = [
        b'{"message":{"content":"Hola"},"done":false}',
        b'{"message":{"content":" patron"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ]
    got = []
    content, last = LLMClient._consume_stream(iter(lines), got.append)
    assert content == "Hola patron"
    assert got == ["Hola", " patron"]


def test_consume_stream_ignores_garbage_lines():
    lines = [b"", b"no-json", b'{"message":{"content":"x"},"done":true}']
    content, _ = LLMClient._consume_stream(iter(lines), None)
    assert content == "x"


_CALL = {"function": {"name": "ram_usage", "arguments": {}}}


def _chunk(message: dict, done: bool) -> bytes:
    import json

    return json.dumps({"message": {"role": "assistant", **message}, "done": done}).encode()


def test_consume_stream_tool_call_en_chunk_no_final():
    """Ollama reciente: la llamada en un chunk done=false y un cierre vacío."""
    lines = [_chunk({"content": "", "tool_calls": [_CALL]}, False), _chunk({"content": ""}, True)]
    _, message = LLMClient._consume_stream(iter(lines), None)
    assert message["tool_calls"] == [_CALL]


def test_consume_stream_el_chunk_final_manda_y_no_duplica():
    """La misma llamada en un chunk intermedio Y en el final: si el final trae
    las suyas, mandan esas (formato de siempre); nunca se ejecuta dos veces."""
    lines = [_chunk({"content": "", "tool_calls": [_CALL]}, False),
             _chunk({"content": "", "tool_calls": [_CALL]}, True)]
    _, message = LLMClient._consume_stream(iter(lines), None)
    assert message["tool_calls"] == [_CALL]


def test_parse_stream_el_chunk_final_manda_aunque_traiga_otra_llamada():
    """Llamadas DISTINTAS en un chunk intermedio y en el final: corren solo las
    del final (criterio del lead), ni las de antes ni las dos sumadas. Con la
    misma llamada en los dos chunks (el test de arriba) cualquier orden pasaba."""
    temprana = {"function": {"name": "pausar_musica", "arguments": {}}}
    lines = [_chunk({"content": "", "tool_calls": [temprana]}, False),
             _chunk({"content": "", "tool_calls": [_CALL]}, True)]

    _, message, complete = LLMClient._parse_stream(iter(lines), None)

    assert complete is True
    assert message["tool_calls"] == [_CALL]


def test_parse_stream_reporta_done():
    completo = [_chunk({"content": "Hola"}, False), _chunk({"content": ""}, True)]
    truncado = [_chunk({"content": "Hola"}, False)]

    assert LLMClient._parse_stream(iter(completo), None) == (
        "Hola", {"role": "assistant", "content": "Hola"}, True)
    # Corte limpio con texto: se acepta (el cliente avisa), pero sin done.
    assert LLMClient._parse_stream(iter(truncado), None) == (
        "Hola", {"role": "assistant", "content": "Hola"}, False)


def test_parse_stream_truncado_con_tool_call_entero_se_acepta():
    """Ollama manda cada llamada ENTERA: un corte limpio después no la deja a medias."""
    lines = [_chunk({"content": "", "tool_calls": [_CALL]}, False)]
    _, message, complete = LLMClient._parse_stream(iter(lines), None)
    assert complete is False
    assert message["tool_calls"] == [_CALL]


@pytest.mark.parametrize("lines", [
    [],
    [_chunk({"content": " "}, False)],
    # Argumentos como texto JSON a medias (algunos modelos los mandan así).
    [_chunk({"content": "", "tool_calls": [
        {"function": {"name": "borrar", "arguments": '{"ruta": "/ho'}}]}, False)],
], ids=["vacio", "espacios", "args_a_medias"])
def test_parse_stream_truncado_sin_nada_usable_sigue_siendo_un_corte(lines):
    from crotolamo.core.llm import TransientLLMError

    with pytest.raises(TransientLLMError, match="a medias, patrón"):
        LLMClient._parse_stream(iter(lines), None)


# --- _LiveStreamer: streamea texto, retiene tool-calls ---
def test_live_streamer_streams_plain_text():
    out = []
    s = _LiveStreamer(out.append)
    for ch in ["Cla", "ro, ", "patron"]:
        s.feed(ch)
    assert "".join(out) == "Claro, patron"


def test_live_streamer_holds_tool_call_json():
    out = []
    s = _LiveStreamer(out.append)
    for ch in ['{"na', 'me": "open_url"}']:
        s.feed(ch)
    assert out == []  # retenido: no se filtra el JSON


# --- agente con streaming ---
class StreamLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_stream(self, messages, tools=None, on_token=None):
        resp = self._responses.pop(0)
        if on_token and resp.content:
            for ch in resp.content:
                on_token(ch)
        return resp


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    monkeypatch.setattr(desktop, "run_detached", lambda args: None)


def test_agent_streams_final_text():
    agent = ToolAgent(
        StreamLLM([ChatResponse(content="Listo, patron.")]),
        Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), max_iterations=3,
    )
    out = []
    reply = agent.handle_turn("hola", on_token=out.append)
    assert reply == "Listo, patron."
    assert "".join(out) == "Listo, patron."


def test_agent_streaming_does_not_leak_tool_json():
    agent = ToolAgent(
        StreamLLM([
            ChatResponse(content='{"name": "open_url", "arguments": {"url": "x.com"}}'),
            ChatResponse(content="Abierto, patron."),
        ]),
        Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), max_iterations=3,
    )
    out = []
    reply = agent.handle_turn("abre x.com", on_token=out.append)
    assert reply == "Abierto, patron."
    # El JSON del primer turno (tool-call) no debe aparecer en el stream.
    assert "{" not in "".join(out)


# --- memoria fuzzy ---
def test_facts_search_ranks_relevant(tmp_path):
    from crotolamo.persistence import db, facts

    dbp = tmp_path / "m.sqlite"
    db.add_fact("mi proyecto principal es Huevonitis 4", "proyectos", db_path=dbp)
    db.add_fact("me gusta el café", "preferencias", db_path=dbp)
    hits = facts.search("cuál es mi proyecto", db_path=dbp)
    assert hits and "Huevonitis" in hits[0]["texto"]


# --- TTS por frases ---
def test_split_sentences():
    from crotolamo.voice.tts import split_sentences

    out = split_sentences("Hola, patrón. ¿Qué tal? Todo bien.")
    assert len(out) == 3


# --- hotkeys ---
def test_send_hotkey_unknown_combo():
    out = desktop.send_hotkey("combo+inexistente")
    assert "No conozco" in out


def test_send_hotkey_marked_unsafe():
    assert desktop.send_hotkey._crotolamo_tool.safe is False


# --- búsqueda en proyecto ---
def test_find_in_project_finds_symbol(fake_project):
    from crotolamo.tools import projects

    out = projects.find_in_project("crotolamo", "class ToolAgent")
    assert "agent.py" in out


def test_search_files_empty_query():
    from crotolamo.tools import files

    assert "Dame un nombre" in files.search_files("")


# --- El preámbulo hablado (GLM verbaliza su intención antes de pedir la tool) ---

def test_streamer_retiene_todo_si_se_enviaron_tools():
    """En voz, "Voy a pausar la música por ti" no debe sonar antes del resultado."""
    from crotolamo.core.agent import _LiveStreamer

    dichos = []
    s = _LiveStreamer(dichos.append, hold_until_done=True)
    for t in ("¡Claro", " que sí!", " Voy a pausar."):
        s.feed(t)
    assert dichos == []  # nada salió por el altavoz


def test_streamer_habla_en_vivo_si_no_hay_tools():
    """Charla pura: lo que sale ES la respuesta, se habla ya (stream_speak)."""
    from crotolamo.core.agent import _LiveStreamer

    dichos = []
    s = _LiveStreamer(dichos.append, hold_until_done=False)
    s.feed("Hola ")
    s.feed("patrón")
    assert "".join(dichos) == "Hola patrón"


def test_streamer_retenido_suelta_el_texto_final():
    from crotolamo.core.agent import _LiveStreamer

    dichos = []
    s = _LiveStreamer(dichos.append, hold_until_done=True)
    s.feed("preámbulo que no debe sonar")
    s.flush_if_held("Listo, pausé la música, patrón.")
    assert dichos == ["Listo, pausé la música, patrón."]


# --- El streamer como DUEÑO de la respuesta en vuelo (causa raíz "el turno del
# LLM es una llamada atómica y bloqueante", paso 4) ---------------------------


class _StreamerDeAntes:
    """Copia literal del LiveStreamer previo al paso 4: la referencia contra la
    que se comprueba que, con los valores por defecto, nada cambia."""

    def __init__(self, on_token, hold_until_done=False):
        self._on_token = on_token
        self._buf = []
        self._decision = "hold" if hold_until_done else None

    def feed(self, chunk):
        self._buf.append(chunk)
        if self._decision == "stream":
            self._on_token(chunk)
            return True
        if self._decision == "hold":
            return False
        head = "".join(self._buf).lstrip()
        if len(head) < 2:
            return True
        if head[0] in "{[" or head.startswith("```"):
            self._decision = "hold"
            return False
        self._decision = "stream"
        self._on_token(head)
        return True

    def flush_if_held(self, final_text):
        if self._decision != "stream":
            self._on_token(final_text)


_STREAMS = [
    ["Cla", "ro, ", "patrón."],
    ["¡Qué onda", ", patrón!", " Aquí", " andamos.", " Dime."],
    ["H", "ola", " patrón"],
    [" ", "\n", "Hola", "."],
    ['{"na', 'me": "open_url"}'],
    ["[", "1, 2]"],
    ["```", "json\n{}", "```"],
    ["Voy a revisar la memoria por ti."],
    ["", "Ho", "", "la"],
    [],
]


@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("tokens", _STREAMS)
def test_streamer_por_defecto_emite_y_devuelve_lo_mismo_que_antes(tokens, hold):
    antes, ahora = [], []
    viejo, nuevo = _StreamerDeAntes(antes.append, hold), _LiveStreamer(ahora.append, hold)

    devuelto_antes = [viejo.feed(t) for t in tokens]
    devuelto_ahora = [nuevo.feed(t) for t in tokens]
    viejo.flush_if_held("Texto final, patrón.")
    nuevo.flush_if_held("Texto final, patrón.")

    assert devuelto_ahora == devuelto_antes
    assert ahora == antes


def test_streamer_es_invocable_como_on_token():
    """El agente pasa el OBJETO al motor (FallbackLLM le pide `restart`)."""
    dichos = []
    s = _LiveStreamer(dichos.append)
    assert s("Hola ") is True
    assert s("patrón") is True
    assert dichos == ["Hola ", "patrón"]


def test_emitter_marca_released_solo_con_texto():
    from crotolamo.core.streaming import Emitter

    dichos = []
    emit = Emitter(dichos.append)
    emit("")
    assert emit.released is False
    emit("Hola")
    assert emit.released is True
    assert dichos == ["", "Hola"]  # reenvía todo, tal cual


def test_streamer_released_y_restart():
    dichos = []
    s = _LiveStreamer(dichos.append, hold_until_done=True)
    s.feed("Preámbulo de la nube que no sonó")
    assert s.released is False

    s.restart()  # otro motor repite la llamada: lo retenido ya no vale
    s.feed("Respuesta ")
    s.feed("del local.")
    assert s.released is False
    s.flush_if_held("Respuesta del local.")
    assert dichos == ["Respuesta del local."]
    assert s.released is True

    en_vivo = _LiveStreamer(dichos.append)
    en_vivo.feed("{")             # aún sin decidir
    en_vivo.restart()
    en_vivo.feed("Ho")
    en_vivo.feed("la")
    assert dichos[-2:] == ["Ho", "la"]  # el "{" del primer intento no se pega


def test_streamer_should_abort_lanza_turn_cancelled():
    from crotolamo.core.llm import LLMError
    from crotolamo.core.streaming import TurnCancelled

    # Un LLMError haría que FallbackLLM abriera el breaker o probara el local.
    assert not issubclass(TurnCancelled, LLMError)
    vigente = {"ok": True}
    dichos = []
    s = _LiveStreamer(dichos.append, should_abort=lambda: not vigente["ok"])
    s.feed("Hola ")
    vigente["ok"] = False
    with pytest.raises(TurnCancelled):
        s.feed("patrón")
    assert dichos == ["Hola "]


def test_streamer_checkpoint_es_el_punto_de_control_sin_tocar_los_tokens():
    """`checkpoint` lo llaman los parsers con CADA línea del stream (un tool-call
    o un razonamiento no traen texto y no pasan por `feed`). Sin should_abort no
    hace nada; con él, corta igual que `feed`."""
    from crotolamo.core.streaming import TurnCancelled

    dichos: list[str] = []
    s = _LiveStreamer(dichos.append)
    s.checkpoint()                                   # sin should_abort: nada
    s.feed("Hola")
    assert dichos == ["Hola"]

    vigente = {"ok": True}
    s = _LiveStreamer(dichos.append, hold_until_done=True,
                      should_abort=lambda: not vigente["ok"])
    s.checkpoint()
    vigente["ok"] = False
    with pytest.raises(TurnCancelled):
        s.checkpoint()


class _ConControl:
    """on_token con punto de control (como LiveStreamer): cuenta las consultas
    y, a partir de `corta_en`, cancela."""

    def __init__(self, corta_en: int | None = None) -> None:
        self.dichos: list[str] = []
        self.consultas = 0
        self.corta_en = corta_en

    def __call__(self, texto: str) -> None:
        self.dichos.append(texto)

    def checkpoint(self) -> None:
        from crotolamo.core.streaming import TurnCancelled

        self.consultas += 1
        if self.corta_en is not None and self.consultas >= self.corta_en:
            raise TurnCancelled()


def _leidas(lines: list[bytes], contador: list[int]):
    for line in lines:
        contador.append(1)
        yield line


def test_parse_stream_consulta_el_punto_de_control_en_cada_linea():
    """Ollama razonando (`thinking`) manda chunks con content vacío que no pasan
    por on_token: también son punto de control. Los tokens no cambian."""
    lines = [_chunk({"content": "", "thinking": "Veamos"}, False),
             _chunk({"content": "", "thinking": " la ram"}, False),
             _chunk({"content": "Hola"}, False),
             _chunk({"content": ""}, True)]
    sink = _ConControl()

    content, _, complete = LLMClient._parse_stream(iter(lines), sink)

    assert (content, complete, sink.dichos) == ("Hola", True, ["Hola"])
    assert sink.consultas == len(lines)


def test_parse_stream_el_punto_de_control_corta_un_stream_sin_texto():
    from crotolamo.core.streaming import TurnCancelled

    lines = [_chunk({"content": "", "thinking": "x" * 8}, False)] * 50 + [
        _chunk({"content": ""}, True)]
    leidas: list[int] = []

    with pytest.raises(TurnCancelled):
        LLMClient._parse_stream(_leidas(lines, leidas), _ConControl(corta_en=2))

    assert len(leidas) == 2          # ni una línea más tras el barge-in


def _sentence(dichos, **kw):
    return _LiveStreamer(dichos.append, hold_until_done=True, release="sentence", **kw)


def test_streamer_sentence_suelta_al_cerrar_la_primera_frase():
    dichos = []
    s = _sentence(dichos)
    assert s.feed("¡Qué onda") is False
    assert s.feed(", patrón!") is False      # cerrada, pero aún no sigue nada
    assert s.feed(" Aquí") is True           # cerrada Y sigue texto: se suelta
    assert dichos == ["¡Qué onda, patrón! Aquí"]
    assert s.feed(" andamos.") is True
    assert dichos == ["¡Qué onda, patrón! Aquí", " andamos."]
    assert s.released and s.would_release_early
    s.flush_if_held("¡Qué onda, patrón! Aquí andamos.")  # ya en vivo: nada más
    assert len(dichos) == 2


@pytest.mark.parametrize("tokens", [
    ("Voy a revisar", " la memoria", " por ti."),
    # El cierre y su espacio en el MISMO trozo: la frase ya cerró, pero no le
    # sigue nada; "cerrada" sola no basta para soltar (mutante M33).
    ("¡Claro que sí, patrón", "! "),
    ("¿Te abro la memoria", "? "),
    ("Voy a revisar la memoria", ". "),
    ("Listo", ".\n\n"),
])
def test_streamer_sentence_no_suelta_preambulo_de_una_frase(tokens):
    dichos = []
    s = _sentence(dichos)
    for t in tokens:
        assert s.feed(t) is False
    assert dichos == []
    s.flush_if_held("Traes la mitad ocupada, patrón.")
    assert dichos == ["Traes la mitad ocupada, patrón."]


@pytest.mark.parametrize("tokens", [
    ['{"name": "ram"', ', "arguments": {}}', ". Y luego", " más."],
    ["[1, 2]", ". Sigue", " texto."],
    ["```json", "\n{}```", ". Más", " texto."],
])
def test_streamer_sentence_retiene_json(tokens):
    dichos = []
    s = _sentence(dichos)
    for t in tokens:
        assert s.feed(t) is False
    assert dichos == [] and not s.would_release_early


def test_streamer_sentence_limpia_el_principio_sin_comerse_el_espacio():
    from crotolamo.core.hooks import meta_preamble_cleaner

    dichos = []
    s = _sentence(dichos, clean_head=meta_preamble_cleaner)
    for t in ("Te resumo el resultado: ", "Tienes 7 GB libres.", " Y "):
        s.feed(t)
    s.feed("sobra.")
    assert "".join(dichos) == "Tienes 7 GB libres. Y sobra."


def test_streamer_sentence_con_limpiador_roto_suelta_el_texto_tal_cual():
    def roto(_texto):
        raise RuntimeError("hook roto")

    dichos = []
    s = _sentence(dichos, clean_head=roto)
    for t in ("Hola, patrón.", " Ya", " está."):
        s.feed(t)
    assert "".join(dichos) == "Hola, patrón. Ya está."


def test_streamer_sentence_restart_no_mezcla_lo_del_primer_motor():
    dichos = []
    s = _sentence(dichos)
    s.feed("La nube iba diciendo algo sin cerrar")
    s.restart()
    for t in ("Hola, patrón.", " Todo", " bien."):
        s.feed(t)
    assert "".join(dichos) == "Hola, patrón. Todo bien."


@pytest.mark.parametrize("tokens, esperado", [
    (["¡Claro que sí", ", patrón!", " Voy a revisar", " la memoria."], True),
    (["Voy a revisar", " la memoria", " por ti."], False),
    (['{"name": "ram"}', ". Y", " más."], False),
    (["Dr.", " López", " dice"], False),        # una abreviatura no cierra frase
])
def test_streamer_would_release_early_no_emite(tokens, esperado):
    """Solo diagnóstico con la política por defecto: dice si "sentence" habría
    soltado texto antes del fin del stream, sin soltar nada."""
    dichos = []
    s = _LiveStreamer(dichos.append, hold_until_done=True)
    for t in tokens:
        assert s.feed(t) is False
    assert s.would_release_early is esperado
    assert dichos == []
    s.restart()
    assert s.would_release_early is False


def test_los_signos_de_cierre_del_streamer_cubren_los_del_tts():
    """El filtro barato del streamer no puede saltarse un cierre que el TTS sí ve."""
    import string

    from crotolamo.core.streaming import _CIERRES
    from crotolamo.voice.tts import _split

    for c in set(string.punctuation) | set("¿¡…«»"):
        for sep in (" ", "** ", "_\t"):
            texto = f"Hola{c}{sep}mundo"
            piezas, _resto = _split(texto, final=False)
            if piezas:  # el TTS cerró una frase: el filtro tiene que haberlo visto
                assert not _CIERRES.isdisjoint(texto), repr(texto)


class _CandidatoContado:
    """`tts._CANDIDATO` que cuenta cuántos caracteres escanea cada búsqueda."""

    def __init__(self, patron):
        self.patron = patron
        self.escaneados = 0

    def finditer(self, text, pos=0):
        self.escaneados += len(text) - pos
        return self.patron.finditer(text, pos)


_LARGOS = {
    "frase_larga": ("palabra " * 250).strip() + ".",
    "lista_sin_puntos": ", ".join(f"elemento {i}" for i in range(300)),
    "charla": "¡Qué onda, patrón! Aquí andamos al cien y listos para la chamba.",
    # Un signo de cierre que NO cierra frase al principio y luego mucho texto:
    # antes, desde ese signo, cada token re-segmentaba todo lo acumulado.
    "abreviatura": "El Sr. Pérez " + ("dice que la cosa va bien y " * 150).strip(),
    "decimal": "La versión 3.5 trae " + ", ".join(f"cosa {i}" for i in range(500)),
    "pregunta_abierta": "¿Sabías que " + ("la palabra sigue y sigue " * 150).strip(),
    # Lista numerada sin punto final: cada "N. " es candidato y ninguno cierra.
    "lista_numerada": "\n".join(f"{i % 10}. elemento numero {i}" for i in range(400)),
}


def test_el_enfasis_de_la_sonda_es_el_del_tts():
    """La sonda incremental vuelve a mirar la racha final de énfasis ("Listo.**"
    espera su espacio): tiene que ser la misma que admite `_CANDIDATO`."""
    from crotolamo.core.streaming import _ENFASIS
    from crotolamo.voice.tts import _CANDIDATO

    assert f"[{_ENFASIS}]*" in _CANDIDATO.pattern


@pytest.mark.parametrize("release", ["final", "sentence"])
@pytest.mark.parametrize("nombre", sorted(_LARGOS))
def test_el_diagnostico_no_resegmenta_cada_token(monkeypatch, nombre, release):
    """Con la política por defecto el diagnóstico corre en TODOS los turnos con
    tools a la vista (y "sentence" usa la misma sonda): re-segmentar lo
    acumulado en cada token era cuadrático (0,4 s de CPU para la lista numerada
    de 9 KB). Cada carácter se escanea un número acotado de veces, y cada
    candidato se decide una vez más, como mucho, por trozo que llega."""
    from crotolamo.voice import tts

    texto = _LARGOS[nombre]
    contado = _CandidatoContado(tts._CANDIDATO)
    decisiones = []
    original = tts._cierra_frase

    def cierra(text, start, end):
        decisiones.append(start)
        return original(text, start, end)

    monkeypatch.setattr(tts, "_CANDIDATO", contado)
    monkeypatch.setattr(tts, "_cierra_frase", cierra)
    trozos = [texto[i:i + 4] for i in range(0, len(texto), 4)]
    s = _LiveStreamer(lambda t: None, hold_until_done=True, release=release)
    for t in trozos:
        s.feed(t)

    assert contado.escaneados <= 2 * len(texto)
    assert len(decisiones) <= len(trozos)


class _SondaDeReferencia:
    """La regla tal como la define el diseño, sin atajos: lo acumulado entero
    por `_split(final=False)` en cada trozo (lo que hacía la sonda antes)."""

    def __init__(self):
        self.buf = []
        self.json = None

    def __call__(self, trozo):
        from crotolamo.voice.tts import _split

        self.buf.append(trozo)
        head = "".join(self.buf).lstrip()
        if self.json is None:
            if len(head) < 2:
                return "wait"
            self.json = head[0] in "{[" or head.startswith("```")
        if self.json:
            return "json"
        piezas, resto = _split(head, final=False)
        return "release" if piezas and resto.strip() else "wait"


_PIEZAS = [
    "Hola", "patrón", "Sr", "Dr", "etc", "p.ej", "No", "no", "40", "3", "12", "x",
    ".", ".", ".", "!", "?", "¿", "¡", "\n", "\n\n", " ", " ", " ", "  ", "\t",
    "**", "_", "~", "1.", "12.", "{", "[", "```", ":", ",", "\xa0", " ",
]


def _caso(rnd):
    texto = "".join(rnd.choice(_PIEZAS) for _ in range(rnd.randint(0, 24)))
    trozos, i = [], 0
    while i < len(texto):
        n = rnd.randint(0, 6)            # trozos vacíos también llegan
        trozos.append(texto[i:i + n])
        i += n
    return trozos


def test_la_sonda_incremental_decide_igual_que_resegmentar_todo():
    """La sonda incremental es una optimización pura: mismo veredicto, trozo a
    trozo, que re-segmentar todo lo acumulado; con las dos políticas, también
    tras un `restart` a mitad de camino (como lo usa FallbackLLM: solo si no
    salió nada)."""
    import random

    rnd = random.Random(20261114)
    for _ in range(4000):
        trozos = _caso(rnd)
        corte = rnd.randint(0, len(trozos))
        for release in ("final", "sentence"):
            dichos = []
            s = _LiveStreamer(dichos.append, hold_until_done=True, release=release)
            for t in trozos[:corte]:
                s.feed(t)
            if s.released:      # FallbackLLM no reintenta lo que ya sonó
                continue
            s.restart()
            ref, veredicto, esperado = _SondaDeReferencia(), "wait", []
            caso = (release, trozos, corte)
            for t in trozos[corte:]:
                devuelto = s.feed(t)
                if veredicto == "wait":
                    veredicto = ref(t)
                    if veredicto == "release":
                        esperado.append("".join(ref.buf).lstrip())
                elif veredicto == "release":
                    esperado.append(t)      # ya en vivo: cada trozo tal cual
                assert s.would_release_early is (veredicto == "release"), caso
                if release == "sentence":
                    assert devuelto is (veredicto == "release"), caso
            assert dichos == (esperado if release == "sentence" else []), caso
