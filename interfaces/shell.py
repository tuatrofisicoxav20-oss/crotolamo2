"""REPL de texto sobre el agente. Reemplaza chapi_shell.main de C1.

Comandos del shell:
  salir / exit / quit   termina
  /reset                limpia la memoria de la sesión
  /historial            muestra el contexto actual
"""

from __future__ import annotations

from crotolamo.core.agent import Agent
from crotolamo.core.llm import LLMClient
from crotolamo.core.memory import Conversation
from crotolamo.core.persona import system_prompt
from crotolamo.settings import get_settings


def _build_agent() -> tuple[Agent, Conversation]:
    settings = get_settings()
    conversation = Conversation(
        system_prompt(),
        max_turns=settings.memory.get("max_turns", 20),
    )
    llm = LLMClient.from_settings(settings)

    # La Fase 2 cablea el registry de tools; si está disponible, lo usamos.
    try:
        from crotolamo.core.agent import ToolAgent  # type: ignore
        from crotolamo.tools import default_registry
        from crotolamo.safety.guard import Guard

        agent: Agent = ToolAgent(
            llm,
            conversation,
            registry=default_registry(),
            guard=Guard.from_settings(settings),
            max_iterations=settings.llm.get("max_iterations", 6),
        )
    except Exception:
        agent = Agent(llm, conversation)

    return agent, conversation


def _show_history(conversation: Conversation) -> None:
    if not conversation.history:
        print("(memoria vacía, patrón)")
        return
    for msg in conversation.history:
        tag = {"user": "Patrón", "assistant": "Crotolamo", "tool": "tool"}.get(msg.role, msg.role)
        content = msg.content
        if msg.role == "assistant" and msg.tool_calls and not content:
            names = ", ".join(c.get("name", "?") for c in msg.tool_calls)
            content = f"[pide tools: {names}]"
        print(f"  {tag}: {content}")


def run_shell(argv: list[str] | None = None) -> int:
    agent, conversation = _build_agent()

    print("Crotolamo Shell. 'salir' para terminar. /reset limpia memoria, /historial la muestra.\n")

    while True:
        try:
            user = input("Patrón > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCrotolamo apagado, patrón.")
            return 0

        if not user:
            continue

        low = user.lower()
        if low in {"salir", "exit", "quit"}:
            print("Crotolamo apagado, patrón.")
            return 0
        if low == "/reset":
            conversation.reset()
            print("Memoria limpia, patrón.")
            continue
        if low == "/historial":
            _show_history(conversation)
            continue

        reply = agent.handle_turn(user)
        print(f"\nCrotolamo > {reply}\n")


if __name__ == "__main__":
    raise SystemExit(run_shell())
