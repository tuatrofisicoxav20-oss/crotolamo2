"""Tests de la confirmación de tools delicadas según el modo de voz (T2).

En el modo CONCURRENTE el EarThread ya tiene el micrófono abierto de forma
persistente: el listen_once de voice_confirm abriría un segundo sd.InputStream
sobre el mismo device (chocan). El confirm_fn de ese modo debe denegar CON
AVISO hablado, sin tocar jamás el STT.
"""

from __future__ import annotations

from crotolamo.core.agent import ToolAgent
from crotolamo.core.memory import Conversation
from crotolamo.safety.guard import Decision
from crotolamo.tools.base import Registry, Tool

from interfaces.listener import make_deny_with_notice


def test_deny_with_notice_avisa_y_deniega():
    hablado: list[str] = []
    deny = make_deny_with_notice(hablado.append)

    assert deny("Voy a borrar el archivo.") is False
    assert len(hablado) == 1
    assert "Voy a borrar el archivo." in hablado[0]
    assert "patrón" in hablado[0]
    assert "shell" in hablado[0]  # le dice el camino para confirmarla de verdad


def test_deny_with_notice_no_puede_abrir_microfono():
    """Por construcción: la factory solo recibe say(); no hay STT que llamar.
    Este test fija esa garantía contra regresiones que le inyecten el STT."""
    import inspect

    params = inspect.signature(make_deny_with_notice).parameters
    assert list(params) == ["say"]


class _GuardQuePideConfirmacion:
    def check(self, tool, arguments) -> Decision:
        return Decision.confirm("Esa acción está delicada.")


class _STTQueNoDebeUsarse:
    def listen_once(self, *args, **kwargs):
        raise AssertionError("el modo concurrente intentó abrir un segundo micrófono")


def test_tool_delicada_se_rechaza_con_aviso_en_modo_concurrente():
    """De punta a punta: una tool safe=False bajo el confirm_fn concurrente se
    rechaza, el aviso se habla, y el STT jamás se usa."""
    registry = Registry()
    registry.register(Tool(
        name="accion_delicada",
        func=lambda: "hecho",
        description="una tool delicada",
        parameters={"type": "object", "properties": {}, "required": []},
        safe=False,
    ))

    hablado: list[str] = []
    _stt = _STTQueNoDebeUsarse()  # presente en el entorno, nunca invocado
    agent = ToolAgent(
        llm=None,
        conversation=Conversation("system prompt de prueba"),
        registry=registry,
        guard=_GuardQuePideConfirmacion(),
        confirm_fn=make_deny_with_notice(hablado.append),
    )

    out = agent._execute_call("accion_delicada", {})
    assert "hecho" not in out          # la tool NO corrió
    assert "patrón" in out.lower() or "cancelado" in out.lower()
    assert len(hablado) == 1           # el aviso sí se habló
    assert "delicada" in hablado[0]
