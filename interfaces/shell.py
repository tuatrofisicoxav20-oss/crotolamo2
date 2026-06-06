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
from crotolamo.persistence import facts
from crotolamo.settings import get_settings


def build_agent(confirm_fn=None) -> tuple[Agent, Conversation]:
    """Construye el agente con tools, guard y memoria persistente inyectada.

    Reusable por el shell (texto) y el listener (voz). confirm_fn decide cómo se
    pide confirmación para tools inseguras (texto o voz).
    """
    settings = get_settings()
    llm = LLMClient.from_settings(settings)

    # M5: compaction por resumen (opcional, [memory].compaction). El summarizer es una
    # llamada corta al LLM que condensa los turnos viejos en vez de tirarlos.
    compaction = settings.memory.get("compaction", False)

    def _summarize(text: str) -> str:
        resp = llm.chat([
            {"role": "system", "content": "Resume en español, muy breve, los hechos y "
             "decisiones clave de esta conversación para conservar contexto. Sin relleno."},
            {"role": "user", "content": text},
        ])
        return resp.content

    # M4.2: los hechos ya NO se inyectan en el system prompt aquí, sino vía pre-hook.
    conversation = Conversation(
        system_prompt(),
        max_turns=settings.memory.get("max_turns", 20),
        compaction=compaction,
        summarizer=_summarize if compaction else None,
    )

    # La Fase 2 cablea el registry de tools; si está disponible, lo usamos.
    try:
        from crotolamo.core.agent import ToolAgent  # type: ignore
        from crotolamo.core.hooks import datetime_prehook, make_facts_prehook
        from crotolamo.tools import default_registry
        from crotolamo.safety.guard import Guard

        agent: Agent = ToolAgent(
            llm,
            conversation,
            registry=default_registry(),
            guard=Guard.from_settings(settings),
            max_iterations=settings.llm.get("max_iterations", 6),
            confirm_fn=confirm_fn or _text_confirm,
            pre_hooks=[make_facts_prehook(), datetime_prehook],
        )
    except Exception:
        agent = Agent(llm, conversation)

    return agent, conversation


def _text_confirm(reason: str) -> bool:
    """Pide confirmación por texto antes de una acción marcada como insegura."""
    answer = input(f"{reason} [s/N] ").strip().lower()
    return answer in {"s", "si", "sí", "y", "yes"}


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
    from crotolamo.core.agent import ToolAgent

    argv = argv or []
    stream = "--stream" in argv

    agent, conversation = build_agent()
    can_stream = stream and isinstance(agent, ToolAgent)

    extra = " (streaming)" if can_stream else ""
    print(f"Crotolamo Shell{extra}. 'salir' para terminar. /reset limpia memoria, /historial la muestra.\n")

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

        # Fase 4: fast-path "acuérdate que ..." — guarda sin depender del LLM.
        fact = facts.detect_remember(user)
        if fact:
            facts.remember(fact)
            print(f"\nCrotolamo > Anotado, patrón. Lo recordaré: «{fact}».\n")
            continue

        if can_stream:
            print("\nCrotolamo > ", end="", flush=True)
            agent.handle_turn(user, on_token=lambda t: print(t, end="", flush=True))
            print("\n")
        else:
            reply = agent.handle_turn(user)
            print(f"\nCrotolamo > {reply}\n")


if __name__ == "__main__":
    raise SystemExit(run_shell())
