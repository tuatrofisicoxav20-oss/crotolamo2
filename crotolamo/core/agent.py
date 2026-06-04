"""El agente. En la Fase 1 es LLM + memoria; la Fase 2 le añade el loop de tools.

handle_turn(text) -> respuesta final en texto.
"""

from __future__ import annotations

from crotolamo.core.llm import LLMClient, LLMError
from crotolamo.core.memory import Conversation


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
