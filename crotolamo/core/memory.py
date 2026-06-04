"""Memoria conversacional de corto plazo.

Esto es lo que C1 no tenía: contexto entre turnos. Una `Conversation` guarda
el historial con roles y lo recorta con una ventana deslizante para no reventar
el contexto del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    # Para mensajes del asistente que piden tools (tool-calling, Fase 2).
    tool_calls: list[dict[str, Any]] | None = None
    # Para mensajes de rol "tool": nombre de la tool que produjo el resultado.
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.name:
            msg["name"] = self.name
        return msg


class Conversation:
    """Historial de la sesión con ventana deslizante por nº de turnos."""

    def __init__(self, system_prompt: str, max_turns: int = 20) -> None:
        self._system = Message("system", system_prompt)
        self._history: list[Message] = []
        self.max_turns = max_turns

    # --- escritura ---
    def add_user(self, text: str) -> None:
        self._history.append(Message("user", text))
        self._trim()

    def add_assistant(self, text: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self._history.append(Message("assistant", text, tool_calls=tool_calls))

    def add_tool_result(self, name: str, content: str) -> None:
        self._history.append(Message("tool", content, name=name))

    def set_system(self, text: str) -> None:
        self._system = Message("system", text)

    def reset(self) -> None:
        self._history.clear()

    # --- lectura ---
    def to_messages(self) -> list[dict[str, Any]]:
        """Payload para el LLM: [system, ...historial]."""
        return [self._system.to_dict()] + [m.to_dict() for m in self._history]

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def _trim(self) -> None:
        """Recorta a max_turns mensajes de usuario, preservando bloques completos.

        Contamos turnos por mensajes de usuario; al exceder, soltamos desde el
        principio (sin tocar el system, que vive aparte).
        """
        user_count = sum(1 for m in self._history if m.role == "user")
        while user_count > self.max_turns and self._history:
            dropped = self._history.pop(0)
            if dropped.role == "user":
                user_count -= 1
            # Tras soltar un user, soltamos su respuesta/tools encadenados hasta
            # el próximo user para no dejar mensajes huérfanos de tool.
            while self._history and self._history[0].role in {"assistant", "tool"}:
                self._history.pop(0)
