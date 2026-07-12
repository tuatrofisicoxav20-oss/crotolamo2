"""Tests del adaptador GLM: traducción de contrato Ollama <-> OpenAI."""

from __future__ import annotations

import json

import pytest

from crotolamo.core.engine import GLM, OLLAMA, resolve_backend
from crotolamo.core.glm import (
    GLMAuthError,
    GLMClient,
    _from_openai_message,
    to_openai_messages,
)


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
    out = _from_openai_message(message)
    assert out["content"] == ""
    assert out["tool_calls"][0]["function"]["arguments"] == {"action": "pause"}


def test_arguments_corruptos_no_revientan():
    message = {"role": "assistant", "content": "",
               "tool_calls": [{"function": {"name": "x", "arguments": "{roto"}}]}
    assert _from_openai_message(message)["tool_calls"][0]["function"]["arguments"] == {}


def test_roundtrip_historial_sobrevive_al_reinyectado():
    """ToolAgent guarda raw_message en el historial y lo vuelve a mandar: el
    formato canónico (dict) debe re-traducirse a OpenAI sin perder nada."""
    openai_msg = {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "call_x", "type": "function",
                        "function": {"name": "ram_usage", "arguments": '{"limit":3}'}}],
    }
    canonical = _from_openai_message(openai_msg)
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
    content, msg = GLMClient._consume_sse(iter(stream), got.append)
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
    _, msg = GLMClient._consume_sse(iter(stream), None)
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
    _, msg = GLMClient._consume_sse(iter(stream), None)
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["a", "b"]


def test_sse_ignora_lineas_basura_y_done():
    stream = [b"\n", b": comentario\n", b"data: no-json\n",
              b'data: {"choices":[{"delta":{"content":"ok"}}]}\n', b"data: [DONE]\n"]
    content, _ = GLMClient._consume_sse(iter(stream), None)
    assert content == "ok"


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
