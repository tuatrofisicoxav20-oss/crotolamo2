"""Memoria conversacional de corto plazo.

Esto es lo que C1 no tenía: contexto entre turnos. Una `Conversation` guarda
el historial con roles y lo recorta con una ventana deslizante para no reventar
el contexto del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Historial de la sesión con ventana deslizante por nº de turnos.

    M5 (de GLaDOS): con compaction=True y un summarizer, los turnos viejos no se
    tiran en seco; se RESUMEN en un mensaje anclado tras el system prompt.
    """

    def __init__(self, system_prompt: str, max_turns: int = 20,
                 compaction: bool = False, summarizer=None) -> None:
        self._system = Message("system", system_prompt)
        self._history: list[Message] = []
        self.max_turns = max_turns
        self.compaction = compaction
        self.summarizer = summarizer  # Callable[[str], str] | None
        self._summary: Message | None = None

    # --- escritura ---
    def add_user(self, text: str) -> None:
        self._history.append(Message("user", text))
        self._trim()

    def add_assistant(self, text: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self._history.append(Message("assistant", text, tool_calls=tool_calls))
        self._trim()  # M5: compactar también tras assistant/tool (arregla m2 del audit)

    def add_tool_result(self, name: str, content: str) -> None:
        self._history.append(Message("tool", content, name=name))
        self._trim()

    def set_system(self, text: str) -> None:
        self._system = Message("system", text)

    def reset(self) -> None:
        self._history.clear()
        self._summary = None

    # --- lectura ---
    def to_messages(self) -> list[dict[str, Any]]:
        """Payload para el LLM: [system, (resumen?), ...historial]."""
        msgs = [self._system.to_dict()]
        if self._summary is not None:
            msgs.append(self._summary.to_dict())
        return msgs + [m.to_dict() for m in self._history]

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def _trim(self) -> None:
        """Recorta a max_turns turnos de usuario, preservando bloques completos.

        Con compaction activa, el bloque viejo se RESUME en vez de descartarse.
        """
        user_count = sum(1 for m in self._history if m.role == "user")
        while user_count > self.max_turns and self._history:
            block = [self._history.pop(0)]
            if block[0].role == "user":
                user_count -= 1
            # Arrastramos su respuesta/tools encadenados hasta el próximo user.
            while self._history and self._history[0].role in {"assistant", "tool"}:
                block.append(self._history.pop(0))

            if self.compaction and self.summarizer is not None:
                self._absorb_into_summary(block)
            # Sin compaction: el bloque simplemente se descarta (comportamiento viejo).

    def _absorb_into_summary(self, block: list["Message"]) -> None:
        """Funde un bloque de mensajes viejos en el resumen anclado (M5)."""
        prev = self._summary.content if self._summary is not None else ""
        rendered = "\n".join(
            f"{m.role}: {m.content}" for m in block if m.content
        )
        try:
            summary_text = self.summarizer(f"{prev}\n{rendered}".strip())
        except Exception:  # noqa: BLE001 - si el resumen falla, no perdemos el turno actual
            return
        if summary_text and summary_text.strip():
            self._summary = Message(
                "system", f"[resumen de la conversación previa]\n{summary_text.strip()}"
            )
