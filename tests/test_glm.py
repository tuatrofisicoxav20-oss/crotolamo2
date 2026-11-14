"""Tests del adaptador GLM: traducción de contrato Ollama <-> OpenAI."""

from __future__ import annotations

import json

import pytest

from crotolamo.core.engine import GLM, OLLAMA, resolve_backend
from crotolamo.core.glm import GLMAuthError, GLMClient
from crotolamo.core.openai_adapter import from_openai_message, to_openai_messages
from crotolamo.core.sse import consume_sse


class _FakeSettings:
    def __init__(self, llm: dict) -> None:
        self._llm = llm

    @property
    def llm(self) -> dict:
        return self._llm


# --- traducción de mensajes hacia OpenAI ---

def test_mensajes_simples_pasan_igual():
    msgs = [
        {"role": "system", "content": "eres crotolamo"},
        {"role": "user", "content": "hola"},
    ]
    assert to_openai_messages(msgs) == msgs


def test_assistant_con_tool_calls_gana_id_y_arguments_string():
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "ram_usage", "arguments": {"limit": 5}}}],
    }]
    out = to_openai_messages(msgs)
    call = out[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"] == "call_0"
    # OpenAI exige `arguments` como string JSON, no como dict.
    assert json.loads(call["function"]["arguments"]) == {"limit": 5}


def test_tool_result_se_correlaciona_con_su_llamada():
    """El mensaje `tool` debe llevar el tool_call_id de la llamada que lo pidió."""
    msgs = [
        {"role": "user", "content": "ram y disco"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "ram_usage", "arguments": {}}},
                {"function": {"name": "disk_usage", "arguments": {}}},
            ],
        },
        {"role": "tool", "name": "ram_usage", "content": "8 GB"},
        {"role": "tool", "name": "disk_usage", "content": "200 GB"},
    ]
    out = to_openai_messages(msgs)
    assert out[2]["tool_call_id"] == "call_0"
    assert out[2]["content"] == "8 GB"
    assert out[3]["tool_call_id"] == "call_1"
    # El rol tool de OpenAI no lleva `name`.
    assert "name" not in out[2]


def test_ids_no_se_repiten_entre_bloques_de_tools():
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
        {"role": "tool", "name": "a", "content": "1"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "b", "arguments": {}}}]},
        {"role": "tool", "name": "b", "content": "2"},
    ]
    out = to_openai_messages(msgs)
    assert out[0]["tool_calls"][0]["id"] == "call_0"
    assert out[2]["tool_calls"][0]["id"] == "call_1"
    assert out[1]["tool_call_id"] == "call_0"
    assert out[3]["tool_call_id"] == "call_1"


def test_tool_huerfano_se_descarta_en_vez_de_colgar_un_id():
    """Un `tool` sin su assistant produciría un tool_call_id que apunta a nada, y
    GLM devolvería un 400 opaco. `_trim()` hoy no lo genera, pero no dependemos
    de eso."""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "tool", "name": "t", "content": "huérfano"},
        {"role": "user", "content": "hola"},
    ]
    out = to_openai_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user"]


def test_trim_nunca_deja_tool_sin_su_assistant():
    """Invariante conjunta: recortar la ventana no rompe la correlación de ids."""
    from crotolamo.core.memory import Conversation

    conv = Conversation("sys", max_turns=2)
    for i in range(10):  # fuerza varios recortes
        conv.add_user(f"p{i}")
        conv.add_assistant("", tool_calls=[{"function": {"name": "t", "arguments": {}}}])
        conv.add_tool_result("t", f"r{i}")
        conv.add_assistant(f"a{i}")

    declarados: set[str] = set()
    for msg in to_openai_messages(conv.to_messages()):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            declarados.update(tc["id"] for tc in msg["tool_calls"])
        if msg["role"] == "tool":
            assert msg["tool_call_id"] in declarados


def test_arguments_ya_string_no_se_doble_serializa():
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "x", "arguments": '{"a":1}'}}]}]
    out = to_openai_messages(msgs)
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


# --- normalización de la respuesta hacia el formato canónico (Ollama) ---

def test_respuesta_openai_se_normaliza_a_dict_de_arguments():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "music_control", "arguments": '{"action":"pause"}'},
        }],
    }
    out = from_openai_message(message)
    assert out["content"] == ""
    assert out["tool_calls"][0]["function"]["arguments"] == {"action": "pause"}


def test_arguments_corruptos_no_revientan():
    message = {"role": "assistant", "content": "",
               "tool_calls": [{"function": {"name": "x", "arguments": "{roto"}}]}
    assert from_openai_message(message)["tool_calls"][0]["function"]["arguments"] == {}


def test_roundtrip_historial_sobrevive_al_reinyectado():
    """ToolAgent guarda raw_message en el historial y lo vuelve a mandar: el
    formato canónico (dict) debe re-traducirse a OpenAI sin perder nada."""
    openai_msg = {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "call_x", "type": "function",
                        "function": {"name": "ram_usage", "arguments": '{"limit":3}'}}],
    }
    canonical = from_openai_message(openai_msg)
    back = to_openai_messages([canonical])[0]
    assert back["tool_calls"][0]["function"]["name"] == "ram_usage"
    assert json.loads(back["tool_calls"][0]["function"]["arguments"]) == {"limit": 3}
    # El id original se conserva, no se reinventa.
    assert back["tool_calls"][0]["id"] == "call_x"


# --- streaming SSE ---

def _sse(*objs) -> list[bytes]:
    lines = [f"data: {json.dumps(o)}\n".encode() for o in objs]
    return lines + [b"data: [DONE]\n"]


def test_sse_acumula_texto_y_llama_on_token():
    stream = _sse(
        {"choices": [{"delta": {"content": "Hola "}}]},
        {"choices": [{"delta": {"content": "patrón"}}]},
    )
    got: list[str] = []
    content, msg = consume_sse(iter(stream), got.append)
    assert content == "Hola patrón"
    assert got == ["Hola ", "patrón"]
    assert "tool_calls" not in msg


def test_sse_reensambla_tool_call_fragmentado():
    """OpenAI parte `arguments` entre deltas; hay que unirlos por `index`."""
    stream = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "music_control", "arguments": '{"act'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ion":"pause"}'}}]}}]},
    )
    _, msg = consume_sse(iter(stream), None)
    call = msg["tool_calls"][0]
    assert call["function"]["name"] == "music_control"
    assert call["function"]["arguments"] == {"action": "pause"}


def test_sse_varias_tool_calls_se_separan_por_index():
    stream = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "a", "arguments": "{}"}},
            {"index": 1, "function": {"name": "b", "arguments": "{}"}},
        ]}}]},
    )
    _, msg = consume_sse(iter(stream), None)
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["a", "b"]


def test_sse_ignora_lineas_basura_y_done():
    stream = [b"\n", b": comentario\n", b"data: no-json\n",
              b'data: {"choices":[{"delta":{"content":"ok"}}]}\n', b"data: [DONE]\n"]
    content, _ = consume_sse(iter(stream), None)
    assert content == "ok"


# --- parse_sse: el fin LÓGICO del stream, para que el cliente cierre el ciclo ---

_HOLA = {"choices": [{"delta": {"content": "Hola"}}]}


def _lineas(*objs) -> list[bytes]:
    return [f"data: {json.dumps(o)}\n".encode() for o in objs]


@pytest.mark.parametrize("stream, completo", [
    (_lineas(_HOLA) + [b"data: [DONE]\n"], True),
    # Hay servidores que cierran tras el chunk de finish_reason sin [DONE].
    (_lineas(_HOLA, {"choices": [{"delta": {}, "finish_reason": "stop"}]}), True),
    # Ni lo uno ni lo otro: el cuerpo se acabó antes (respuesta truncada).
    (_lineas(_HOLA), False),
], ids=["done", "finish_reason", "ninguno"])
def test_parse_sse_reporta_fin_logico(stream, completo):
    from crotolamo.core.sse import parse_sse

    got: list[str] = []
    content, msg, complete = parse_sse(iter(stream), got.append)

    assert complete is completo
    assert content == "Hola" and got == ["Hola"]
    assert msg == {"role": "assistant", "content": "Hola"}
    # El envoltorio de 2 valores devuelve lo mismo, sin el aviso.
    assert consume_sse(iter(stream), None) == (content, msg)


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


def _tool_delta(args: str, primero: bool = False) -> dict:
    call: dict = {"index": 0, "function": {"arguments": args}}
    if primero:
        call = {"index": 0, "id": "c1", "function": {"name": "escribir", "arguments": args}}
    return {"choices": [{"delta": {"tool_calls": [call]}}]}


def test_parse_sse_consulta_el_punto_de_control_en_cada_linea():
    """Un tool-call (deltas de `arguments`), un razonamiento (reasoning_content,
    con [llm.glm].thinking) o un delta vacío no traen texto y no pasan por
    on_token: antes un turno de orden no se podía cancelar hasta el final del
    stream. Cada línea leída es punto de control; los tokens no cambian."""
    from crotolamo.core.sse import parse_sse

    stream = _lineas(
        {"choices": [{"delta": {"reasoning_content": "Veamos..."}}]},
        _tool_delta('{"texto": ', primero=True),
        _tool_delta('"hola"}'),
        {"choices": [{"delta": {}}]},
        _HOLA,
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    sink = _ConControl()

    content, msg, complete = parse_sse(iter(stream), sink)

    assert (content, complete, sink.dichos) == ("Hola", True, ["Hola"])
    assert msg["tool_calls"][0]["function"] == {"name": "escribir", "arguments": {"texto": "hola"}}
    assert sink.consultas == len(stream)


@pytest.mark.parametrize("delta", [_tool_delta("xxxxxxxx"),
                                   {"choices": [{"delta": {"reasoning_content": "mmm"}}]}])
def test_parse_sse_el_punto_de_control_corta_un_stream_sin_texto(delta):
    from crotolamo.core.sse import parse_sse
    from crotolamo.core.streaming import TurnCancelled

    stream = _lineas(_tool_delta('{"texto": "', primero=True), *[delta] * 50)
    leidas: list[int] = []

    def _iter():
        for line in stream:
            leidas.append(1)
            yield line

    with pytest.raises(TurnCancelled):
        parse_sse(_iter(), _ConControl(corta_en=2))

    assert len(leidas) == 2          # ni una línea más tras el barge-in


def test_parse_sse_truncado_con_tool_call_entero_se_acepta():
    """Corte limpio tras un tool_call cuyos argumentos llegaron ENTEROS: la
    respuesta vale (con el aviso de truncado, que lo da el cliente)."""
    from crotolamo.core.sse import parse_sse

    stream = _lineas({"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "c1", "function": {"name": "music_control", "arguments": '{"a": 1}'},
    }]}}]})

    _, msg, complete = parse_sse(iter(stream), None)

    assert complete is False
    assert msg["tool_calls"][0]["function"] == {"name": "music_control", "arguments": {"a": 1}}


@pytest.mark.parametrize("stream", [
    [],                                                      # nada de nada
    _lineas({"choices": [{"delta": {"content": "  "}}]}),    # solo espacios
    # Argumentos a medio llegar: jamás se ejecutan degradados a {}.
    _lineas({"choices": [{"delta": {"content": "Va."}}]}, {"choices": [{"delta": {
        "tool_calls": [{"index": 0, "function": {"name": "borrar", "arguments": '{"ruta": "/ho'}}],
    }}]}),
    # Un tool_call que ni trajo argumentos tampoco se sabe entero.
    _lineas({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "x"}}]}}]}),
], ids=["vacio", "espacios", "args_a_medias", "sin_args"])
def test_parse_sse_truncado_sin_nada_usable_sigue_siendo_un_corte(stream):
    from crotolamo.core.sse import SSETruncated, parse_sse

    with pytest.raises(SSETruncated):
        parse_sse(iter(stream), None)


# --- credenciales y selección de backend ---

def test_sin_api_key_error_en_personaje(monkeypatch):
    for env in ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    client = GLMClient()
    with pytest.raises(GLMAuthError, match="API key"):
        client._headers()


def test_api_key_desde_env(monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    assert GLMClient()._headers()["Authorization"] == "Bearer secreta"


def test_from_settings_lee_subtabla_glm():
    settings = _FakeSettings({
        "temperature": 0.5,
        "glm": {"model": "glm-5.2", "base_url": "https://x/v4/", "timeout": 30},
    })
    client = GLMClient.from_settings(settings)
    assert client.model == "glm-5.2"
    assert client.base_url == "https://x/v4"  # sin barra final
    assert client.temperature == 0.5
    assert client.timeout == 30


def test_thinking_apagado_por_defecto():
    """GLM-4.7 razona por defecto: 6.9s vs 1.2s por turno. En voz eso no sirve."""
    payload = GLMClient()._payload([{"role": "user", "content": "hola"}], None, False)
    assert payload["thinking"] == {"type": "disabled"}


def test_thinking_encendido_no_manda_el_campo():
    payload = GLMClient(thinking=True)._payload(
        [{"role": "user", "content": "hola"}], None, False)
    assert "thinking" not in payload


def test_from_settings_lee_thinking():
    settings = _FakeSettings({"glm": {"thinking": True}})
    assert GLMClient.from_settings(settings).thinking is True


def test_backend_env_pisa_el_toml(monkeypatch):
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    assert resolve_backend(_FakeSettings({"backend": "ollama"})) == GLM


def test_backend_desconocido_cae_a_ollama(monkeypatch):
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    assert resolve_backend(_FakeSettings({"backend": "gemini"})) == OLLAMA


def test_backend_por_defecto_es_ollama(monkeypatch):
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    assert resolve_backend(_FakeSettings({})) == OLLAMA


def test_build_llm_sin_key_cae_a_ollama(monkeypatch):
    """Más vale un Crotolamo lento que uno mudo."""
    from crotolamo.core.engine import build_llm
    from crotolamo.core.llm import LLMClient

    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    for env in ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert isinstance(build_llm(_FakeSettings({})), LLMClient)


def test_build_llm_con_key_usa_glm_con_respaldo_local(monkeypatch):
    """Con key, el cliente va envuelto: GLM primero, Ollama si la nube se cae."""
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")
    from crotolamo.core.engine import build_llm
    from crotolamo.core.fallback import FallbackLLM

    llm = build_llm(_FakeSettings({}))
    assert isinstance(llm, FallbackLLM)
    assert isinstance(llm.primary, GLMClient)
