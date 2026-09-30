"""El agente. En la Fase 1 es LLM + memoria; la Fase 2 le añade el loop de tools.

handle_turn(text) -> respuesta final en texto.
"""

from __future__ import annotations

from typing import Any, Callable

from crotolamo.core.llm import LLMClient, LLMError
from crotolamo.core.streaming import LiveStreamer as _LiveStreamer
from crotolamo.core.tool_parsing import (
    HARD_ERROR_PREFIXES as _HARD_ERROR_PREFIXES,  # noqa: F401 - re-export (compat)
    coerce_text_tool_calls as _coerce_text_tool_calls,
    is_hard_error as _is_hard_error,
)
from crotolamo.logging_setup import get_logger
from crotolamo.core.memory import Conversation

log = get_logger("core.agent")


class Agent:
    def __init__(self, llm: LLMClient, conversation: Conversation) -> None:
        self.llm = llm
        self.conversation = conversation

    def handle_turn(self, text: str, on_token=None) -> str:
        """Un turno conversacional con memoria. Sin tools todavía (Fase 1).

        on_token: misma firma que en ToolAgent (el listener lo usa para hablar
        en streaming); aquí el fallback emite la respuesta completa de golpe.
        """
        self.conversation.add_user(text)

        try:
            response = self.llm.chat(self.conversation.to_messages())
        except LLMError as error:
            return str(error)

        reply = response.content or "Me quedé en blanco, patrón. Repíteme eso."
        self.conversation.add_assistant(reply)
        if on_token is not None:
            on_token(reply)
        return reply


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
        fastpath: bool = True,
    ) -> None:
        super().__init__(llm, conversation)
        self.registry = registry
        self.guard = guard
        self.max_iterations = max_iterations
        self.confirm_fn = confirm_fn or _deny
        # Misión velocidad: atajos regex SIN LLM para comandos inequívocos
        # ("pausa la música" -> music_control). El comando pasa igual por el
        # guard y el registry; solo se salta las ~17-22s del LLM en CPU.
        self.fastpath = fastpath
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
        self.post_hooks: list[Callable[[str], str]]
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
        # Las tools con strict_args=False (MCP, M4) se saltan el filtro: su
        # esquema lo dicta el server y puede ser anidado o abierto
        # (additionalProperties); mutilarlo rompería llamadas legítimas.
        if getattr(tool, "strict_args", True):
            declared = set(tool.parameters.get("properties", {}).keys())
            arguments = {k: v for k, v in arguments.items() if k in declared}

        decision = self.guard.check(tool, arguments)
        if not decision.allowed:
            return decision.reason
        if decision.needs_confirmation and not self.confirm_fn(decision.reason):
            return "Cancelado por el patrón."

        return self.registry.run(name, arguments)

    def _safe_execute(self, name: str, arguments: dict) -> str:
        """_execute_call que NUNCA propaga: siempre hay un resultado que anotar.

        Registry.run ya atrapa lo que revienta DENTRO de la tool, pero el guard
        (p.ej. un '\\0' en la ruta) o el confirm_fn (STT/sounddevice) pueden
        lanzar antes. Si eso escapaba tras add_assistant(tool_calls=...), el
        historial quedaba con un assistant pidiendo tools sin su resultado y
        las APIs OpenAI-compatibles (GLM) rechazaban TODOS los turnos siguientes
        hasta /reset. El prefijo "La tool '" lo marca como fallo duro (sin
        short-circuit), igual que un reventón dentro de la tool.
        """
        try:
            return self._execute_call(name, arguments)
        except Exception as error:  # noqa: BLE001 - un fallo del guard/confirm no rompe el turno
            log.exception("la tool '%s' reventó fuera del registry", name)
            return f"La tool '{name}' reventó, patrón: {error}"

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

        # Fast-path (misión velocidad): comando inequívoco -> tool directa sin
        # LLM (~0.1s en vez de ~17-22s). Solo si la regla cubre el comando
        # COMPLETO; cualquier matiz cae al loop normal de abajo. El historial
        # se alimenta igual para que la conversación no pierda el turno.
        if self.fastpath:
            from crotolamo.core import fastpath as fastpath_mod

            hit = fastpath_mod.match(routing_text)
            if hit is not None:
                fast_name, fast_args = hit
                result = self._safe_execute(fast_name, fast_args)
                if not _is_hard_error(result):
                    reply = self._apply(self.post_hooks, result)
                    self.conversation.add_user(text)
                    self.conversation.add_assistant(reply)
                    if on_token is not None:
                        on_token(reply)
                    return reply
                # Fallo duro del atajo (p.ej. hyprctl ausente): que lo razone
                # el LLM como siempre, sin ensuciar el historial.

        self.conversation.add_user(text)
        schemas = self.route_fn(routing_text) if self.route_fn is not None else self.registry.schemas()
        known = set(self.registry.names())

        for _ in range(self.max_iterations):
            # Una tool retirada a mitad de turno (un server MCP desconectado por
            # timeouts, M4) no debe seguir a la vista del modelo en la siguiente
            # iteración: filtramos por presencia en el registry (barato).
            if schemas:
                schemas = [
                    s for s in schemas
                    if self.registry.get(s.get("function", {}).get("name", "")) is not None
                ]
            # Con tools a la vista, el modelo puede anunciar lo que va a hacer antes
            # de pedirla; retenemos hasta saber si hubo tool_call. Sin tools (charla),
            # se habla en vivo.
            streamer = (
                _LiveStreamer(on_token, hold_until_done=bool(schemas))
                if on_token is not None else None
            )
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
                result = self._safe_execute(call["name"], call.get("arguments", {}))
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

        # Agotadas las iteraciones. El historial acaba en un bloque tool sin
        # respuesta del asistente; si no cerramos con un assistant, el turno
        # siguiente arranca con una secuencia inconsistente (y algunos motores
        # rechazan un `tool` sin `assistant` que lo suceda).
        reply = "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
        self.conversation.add_assistant(reply)
        if on_token is not None:
            on_token(reply)
        return reply
