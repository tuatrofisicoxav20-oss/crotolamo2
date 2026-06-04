"""El agente. En la Fase 1 es LLM + memoria; la Fase 2 le añade el loop de tools.

handle_turn(text) -> respuesta final en texto.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from crotolamo.core.llm import LLMClient, LLMError
from crotolamo.core.memory import Conversation


def _coerce_text_tool_calls(content: str, known_names: set[str]) -> list[dict[str, Any]]:
    """Fallback: algunos modelos (qwen2.5-coder en Ollama) emiten el tool-call
    como JSON dentro de `content` en vez de en el campo nativo `tool_calls`.

    Parseamos ese texto y, solo si referencia una tool conocida, lo tratamos
    como llamada. Devuelve [] si el contenido es texto conversacional normal.
    """
    if not content or "{" not in content:
        return []

    # Quitar cercas de código ```json ... ``` si las hay.
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Intentar el bloque {...} o [...] más externo.
    start = min((text.find(c) for c in "{[" if c in text), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        return []

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    candidates = parsed if isinstance(parsed, list) else [parsed]
    calls: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        args = item.get("arguments", item.get("args", {}))
        if name in known_names and isinstance(args, dict):
            calls.append({"name": name, "arguments": args})
    return calls


class Agent:
    def __init__(self, llm: LLMClient, conversation: Conversation) -> None:
        self.llm = llm
        self.conversation = conversation

    def handle_turn(self, text: str) -> str:
        """Un turno conversacional con memoria. Sin tools todavía (Fase 1)."""
        self.conversation.add_user(text)

        try:
            response = self.llm.chat(self.conversation.to_messages())
        except LLMError as error:
            return str(error)

        reply = response.content or "Me quedé en blanco, patrón. Repíteme eso."
        self.conversation.add_assistant(reply)
        return reply


# Callback de confirmación: recibe el motivo, devuelve True si el patrón acepta.
ConfirmFn = Callable[[str], bool]


def _deny(_reason: str) -> bool:
    """Confirmación por defecto: negar (lo seguro)."""
    return False


class ToolAgent(Agent):
    """El loop agéntico: el LLM pide tools, las ejecutamos (bajo guard) y le
    devolvemos el resultado para que decida el siguiente paso.
    """

    def __init__(
        self,
        llm: LLMClient,
        conversation: Conversation,
        registry,
        guard,
        max_iterations: int = 6,
        confirm_fn: ConfirmFn | None = None,
    ) -> None:
        super().__init__(llm, conversation)
        self.registry = registry
        self.guard = guard
        self.max_iterations = max_iterations
        self.confirm_fn = confirm_fn or _deny

    def _execute_call(self, name: str, arguments: dict) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"No tengo una tool llamada '{name}', patrón."

        decision = self.guard.check(tool, arguments)
        if not decision.allowed:
            return decision.reason
        if decision.needs_confirmation and not self.confirm_fn(decision.reason):
            return "Cancelado por el patrón."

        return self.registry.run(name, arguments)

    def handle_turn(self, text: str) -> str:
        self.conversation.add_user(text)
        schemas = self.registry.schemas()
        known = set(self.registry.names())

        for _ in range(self.max_iterations):
            try:
                response = self.llm.chat(self.conversation.to_messages(), tools=schemas)
            except LLMError as error:
                return str(error)

            calls = response.tool_calls
            native = bool(calls)
            # Fallback: el modelo puso el tool-call como JSON en content.
            if not calls:
                calls = _coerce_text_tool_calls(response.content, known)

            if not calls:
                reply = response.content or "Listo, patrón."
                self.conversation.add_assistant(reply)
                return reply

            # Reinyectamos el mensaje del asistente con sus tool_calls (formato Ollama).
            tool_calls_payload = response.raw_message.get("tool_calls") if native else [
                {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
            ]
            self.conversation.add_assistant(
                response.content if native else "",
                tool_calls=tool_calls_payload,
            )

            for call in calls:
                result = self._execute_call(call["name"], call.get("arguments", {}))
                self.conversation.add_tool_result(call["name"], result)
            # Volvemos a pedirle al LLM que decida con los resultados a la vista.

        return "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
