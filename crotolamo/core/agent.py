"""El agente. En la Fase 1 es LLM + memoria; la Fase 2 le añade el loop de tools.

handle_turn(text) -> respuesta final en texto.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from crotolamo.core.llm import LLMClient, LLMError
from crotolamo.logging_setup import get_logger
from crotolamo.core.memory import Conversation

log = get_logger("core.agent")


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


class _LiveStreamer:
    """Streamea tokens al patrón en vivo, salvo que la respuesta empiece como un
    tool-call JSON ('{', '[' o cerca de código) — en ese caso retiene, para no
    filtrar el JSON crudo de los tool-calls que qwen emite en `content`.
    """

    def __init__(self, on_token: Callable[[str], None]) -> None:
        self._on_token = on_token
        self._buf: list[str] = []
        self._decision: str | None = None  # None | "stream" | "hold"

    def feed(self, chunk: str) -> None:
        self._buf.append(chunk)
        if self._decision == "stream":
            self._on_token(chunk)
            return
        if self._decision == "hold":
            return
        head = "".join(self._buf).lstrip()
        if len(head) < 2:
            return  # aún no hay suficiente para decidir
        if head[0] in "{[" or head.startswith("```"):
            self._decision = "hold"
        else:
            self._decision = "stream"
            self._on_token(head)  # soltamos lo acumulado de golpe y seguimos en vivo

    def flush_if_held(self, final_text: str) -> None:
        """Si retuvimos pero resultó ser texto final, lo emitimos completo."""
        if self._decision != "stream":
            self._on_token(final_text)


# Callback de confirmación: recibe el motivo, devuelve True si el patrón acepta.
ConfirmFn = Callable[[str], bool]


def _deny(_reason: str) -> bool:
    """Confirmación por defecto: negar (lo seguro)."""
    return False


# Tools "de retorno directo" por defecto: su output YA es una frase lista para el
# patrón (presentacional), así que cuando el modelo pide EXACTAMENTE una de ellas
# devolvemos su resultado tal cual, sin una 2ª llamada al LLM para que lo redacte.
# Esto corta ~7-9s por turno en CPU. Inyectable vía ToolAgent(direct_tools=...).
DEFAULT_DIRECT_TOOLS: frozenset[str] = frozenset({
    "ram_usage",
    "system_status",
    "disk_usage",
    "music_now",
    "music_control",
    "list_processes",
})

# Prefijos con los que la capa de ejecución (Registry.run) marca un fallo DURO de
# la tool (excepción no controlada o argumentos inválidos). En esos casos NO se
# hace short-circuit: dejamos que el modelo reaccione. Los "soft-errors" en
# personaje de las propias tools ("No pude leer /proc/meminfo, patrón.") SÍ son
# texto listo para el patrón, así que esos sí se devuelven directos.
_HARD_ERROR_PREFIXES: tuple[str, ...] = (
    "La tool '",            # "...reventó, patrón: ..."
    "Argumentos inválidos para '",
    "No tengo una tool ",   # nombre desconocido (defensivo; no debería pasar aquí)
)


def _is_hard_error(result: str) -> bool:
    """True si el resultado de una tool es un fallo duro (no apto para short-circuit)."""
    if not result or not result.strip():
        return True
    return result.lstrip().startswith(_HARD_ERROR_PREFIXES)


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
        pre_hooks: list[Callable[[str], str]] | None = None,
        post_hooks: list[Callable[[str], str]] | None = None,
        route_fn: Callable[[str], list[dict[str, Any]]] | None = None,
        direct_tools: set[str] | None = None,
    ) -> None:
        super().__init__(llm, conversation)
        self.registry = registry
        self.guard = guard
        self.max_iterations = max_iterations
        self.confirm_fn = confirm_fn or _deny
        # Short-circuit de retorno directo: nombres de tools "presentacionales"
        # cuyo output se devuelve tal cual (sin 2ª llamada al LLM). Si es None usa
        # el default sensato; pasa un set vacío para DESACTIVAR el short-circuit
        # (comportamiento clásico de 2 llamadas, útil p.ej. para medir/comparar).
        self.direct_tools: set[str] = (
            set(DEFAULT_DIRECT_TOOLS) if direct_tools is None else set(direct_tools)
        )
        # Tool routing: si está, devuelve solo las tool-schemas relevantes a la
        # consulta (prompt chico => rápido en CPU). Si es None, se mandan todas.
        self.route_fn = route_fn
        # M4 (de Open WebUI pipelines): hooks que enriquecen la entrada (pre) y
        # limpian/transforman la respuesta final (post). Se aplican en orden.
        self.pre_hooks = pre_hooks or []
        # Misión 2: por defecto (post_hooks=None) montamos el limpiador de
        # preámbulos meta del 3B ("parece que la herramienta no acepta
        # parámetros...", "te resumo el resultado:"). Es conservador (anclado al
        # inicio y nunca devuelve vacío). Si el caller pasa post_hooks explícitos,
        # se respetan tal cual (los tests inyectan los suyos).
        if post_hooks is None:
            from crotolamo.core.hooks import (
                meta_preamble_cleaner,
                strip_leaked_tool_json,
            )
            # Orden: primero limpiar preámbulos meta; luego, si lo que queda es un
            # tool-call JSON crudo filtrado, sustituirlo por un mensaje en personaje.
            self.post_hooks = [meta_preamble_cleaner, strip_leaked_tool_json]
        else:
            self.post_hooks = post_hooks

    def _execute_call(self, name: str, arguments: dict) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"No tengo una tool llamada '{name}', patrón."

        # El 3B a veces alucina argumentos que la tool NO declara (p.ej. pide
        # `ram_usage` con un `limit` que se le "pega" de `list_processes`, vecina
        # en el routing). Eso haría reventar a func(**args) con un TypeError, lo
        # que rompería el short-circuit (lo trataría como fallo duro). Filtramos
        # los kwargs no declarados por ESTA tool antes de ejecutar. No oculta
        # errores de args REQUERIDOS ausentes: esos siguen reventando como antes.
        declared = set(tool.parameters.get("properties", {}).keys())
        arguments = {k: v for k, v in arguments.items() if k in declared}

        decision = self.guard.check(tool, arguments)
        if not decision.allowed:
            return decision.reason
        if decision.needs_confirmation and not self.confirm_fn(decision.reason):
            return "Cancelado por el patrón."

        return self.registry.run(name, arguments)

    def _is_direct(self, name: str) -> bool:
        """True si la tool es de retorno directo: o bien está en el set inyectado
        `direct_tools`, o bien su definición lleva el flag `Tool.direct=True`.
        """
        if name in self.direct_tools:
            return True
        tool = self.registry.get(name)
        return bool(tool is not None and getattr(tool, "direct", False))

    def _apply(self, hooks, value: str) -> str:
        for hook in hooks:
            try:
                value = hook(value)
            except Exception as error:  # noqa: BLE001 - un hook roto no mata el turno
                log.warning("hook falló: %s", error)
        return value

    def handle_turn(self, text: str, on_token=None) -> str:
        # Enrutamos sobre el texto LIMPIO del patrón (antes de que los pre-hooks le
        # antepongan fecha/hechos), que es la señal real de intención. El set de
        # tools se fija UNA vez por turno y se mantiene en todas las iteraciones,
        # para no romper el cache de prefijo dentro del turno.
        routing_text = text
        # M4: pre-hooks enriquecen la entrada antes de llegar al LLM.
        text = self._apply(self.pre_hooks, text)
        self.conversation.add_user(text)
        schemas = self.route_fn(routing_text) if self.route_fn is not None else self.registry.schemas()
        known = set(self.registry.names())

        for _ in range(self.max_iterations):
            streamer = _LiveStreamer(on_token) if on_token is not None else None
            try:
                if streamer is not None:
                    response = self.llm.chat_stream(
                        self.conversation.to_messages(), tools=schemas, on_token=streamer.feed,
                    )
                else:
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
                # M4: post-hooks transforman/limpian la respuesta final.
                reply = self._apply(self.post_hooks, reply)
                # Si retuvimos por sospecha de tool-call pero era texto, lo soltamos ahora.
                if streamer is not None:
                    streamer.flush_if_held(reply)
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

            results: list[tuple[str, str]] = []
            for call in calls:
                result = self._execute_call(call["name"], call.get("arguments", {}))
                self.conversation.add_tool_result(call["name"], result)
                results.append((call["name"], result))

            # Short-circuit (Misión 1): si el modelo pidió EXACTAMENTE una tool,
            # esa tool es de retorno directo y NO falló duro, devolvemos su output
            # tal cual como respuesta final, ahorrándonos la 2ª llamada al LLM
            # (~7-9s en CPU). Casos con 2+ tools, tool no-directa o fallo duro
            # caen al comportamiento normal (otra iteración => 2ª llamada).
            if len(results) == 1:
                name, result = results[0]
                if self._is_direct(name) and not _is_hard_error(result):
                    # Mismos post-hooks y misma actualización de memoria que el
                    # camino final normal (líneas del bloque `if not calls`).
                    reply = self._apply(self.post_hooks, result)
                    if streamer is not None:
                        # En modo voz/streaming el caller solo lee on_token; hay
                        # que emitir el texto o el patrón se queda en silencio.
                        streamer.flush_if_held(reply)
                    self.conversation.add_assistant(reply)
                    return reply
            # Si no hubo short-circuit, volvemos a pedirle al LLM que decida con
            # los resultados a la vista.

        return "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
