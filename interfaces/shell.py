"""REPL de texto sobre el agente. Reemplaza chapi_shell.main de C1.

Comandos del shell:
  salir / exit / quit   termina
  /reset                limpia la memoria de la sesión
  /historial            muestra el contexto actual
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Callable, cast

from crotolamo.core.agent import Agent
from crotolamo.core.engine import build_llm
from crotolamo.core.memory import CompactionWorker, Conversation
from crotolamo.core.persona import system_prompt
from crotolamo.logging_setup import get_logger
from crotolamo.settings import get_settings

if TYPE_CHECKING:
    from crotolamo.core.memoria import Memoria

log = get_logger("shell")

# Lo que oye el patrón cuando un turno revienta por un bug (la misma frase que la voz).
_TURNO_ROTO = "Se me trabaron los cables, patrón. Vuelve a intentarlo."


def _memory_knob(memory: dict[str, Any], key: str, default: int | None) -> Any:
    """Un número opcional (>= 0) de [memory]. Un valor que no lo es (un typo en el
    toml) se avisa y se ignora: un knob opcional no debe tumbar el arranque del
    daemon, ni un negativo vaciar la conversación en cada turno. Tampoco `inf`
    (TOML válido): Conversation hace int() y el OverflowError, fuera del try de
    build_agent, dejaba al listener en un bucle de reinicios de systemd."""
    value = memory.get(key, default)
    if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                         and value >= 0 and math.isfinite(value)):
        return value
    log.warning("[memory].%s = %r no es un número >= 0; lo ignoro (uso %r)",
                key, value, default)
    return default


# El bloque de recuerdos con el formato del pre-hook (make_memoria_prehook).
_BLOQUE_RECUERDOS = "[recuerdos sobre el patrón que vienen al caso:\n{}]\n\n"


class _RecuerdosOmitidos:
    """La memoria del pre-hook de recuerdos con GLM de cerebro: busca con la de
    verdad y apunta los recuerdos que el pre-hook dejó FUERA por ir ya en la
    ventana (make_memoria_prehook con `ventana_entera`).

    POR QUÉ: el pre-hook decide una vez por turno, ANTES de la llamada, si el
    motor ve la ventana entera; FallbackLLM elige motor llamada a llamada. Si la
    nube cae A MEDIA llamada, el local recibe esa petición (y, con el breaker ya
    abierto, las siguientes del turno) y su ajuste a num_ctx tira los turnos
    viejos, donde iba la única copia. _LocalConRecuerdos se los devuelve con lo
    apuntado aquí. Apuntar y no buscar otra vez: una 2ª búsqueda en el turno,
    con la 1ª aún en curso, contaría como otro fallo para el breaker de la memoria.

    Por hilo (el del turno; el resumen de compaction llama al LLM desde el suyo)
    y por turno: uno que no busca (un comando anclado) no hereda los del anterior.
    """

    def __init__(self, memoria: Memoria, turno: Callable[[], object]) -> None:
        self._memoria = memoria
        self._turno = turno
        self._hilo = threading.local()

    def __getattr__(self, name: str) -> Any:
        # El resto del contrato, el de la memoria de verdad. Sin ella aún (copy
        # crea el objeto sin __init__), AttributeError y no una recursión.
        if name == "_memoria":
            raise AttributeError(name)
        return getattr(self._memoria, name)

    def contexto(self, query: str, ya_visto: Callable[[str], bool] | None = None) -> str:
        omitidas: list[str] = []
        self._hilo.apunte = (self._turno(), omitidas)
        if ya_visto is None:
            return self._memoria.contexto(query)

        def visto(linea: str) -> bool:
            if ya_visto(linea):
                omitidas.append(linea)
                return True
            return False

        return self._memoria.contexto(query, visto)

    def omitidas(self) -> list[str]:
        """Las que dejó fuera el pre-hook de ESTE turno en ESTE hilo ([] si no buscó)."""
        apunte = getattr(self._hilo, "apunte", None)
        if apunte is None or apunte[0] is not self._turno():
            return []
        return list(apunte[1])


class _LocalConRecuerdos:
    """El motor local con GLM de cerebro: la petición que le llega en un turno
    cuyo pre-hook dejó recuerdos fuera (ver _RecuerdosOmitidos) los lleva de
    vuelta delante del último mensaje del patrón, el que su ajuste a num_ctx no
    tira nunca: lo mismo que ve con el breaker abierto desde el principio del
    turno, cuando el pre-hook ya los repite. Solo en la petición: la
    Conversation (y la nube, cuando vuelve) sigue sin copias de más.
    """

    def __init__(self, local: Any, recuerdos: _RecuerdosOmitidos) -> None:
        self._local = local
        self._recuerdos = recuerdos

    def __getattr__(self, name: str) -> Any:
        # model, preconnect, el transporte...: los del cliente de verdad (sin él
        # aún, como en _RecuerdosOmitidos, AttributeError y no una recursión).
        if name == "_local":
            raise AttributeError(name)
        return getattr(self._local, name)

    def _con_recuerdos(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lineas = self._recuerdos.omitidas()
        if not lineas:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                bloque = _BLOQUE_RECUERDOS.format("\n".join(lineas))
                out = list(messages)
                out[i] = {**messages[i], "content": bloque + (messages[i].get("content") or "")}
                return out
        return messages

    def chat(self, messages, tools=None):
        return self._local.chat(self._con_recuerdos(messages), tools=tools)

    def chat_stream(self, messages, tools=None, on_token=None):
        return self._local.chat_stream(self._con_recuerdos(messages), tools=tools,
                                       on_token=on_token)


def build_agent(confirm_fn=None) -> tuple[Agent, Conversation]:
    """Construye el agente con tools, guard y memoria persistente inyectada.

    Reusable por el shell (texto) y el listener (voz). confirm_fn decide cómo se
    pide confirmación para tools inseguras (texto o voz).
    """
    settings = get_settings()
    # Ollama (local) o GLM (nube), según [llm].backend. Ver crotolamo/core/engine.py.
    llm = build_llm(settings)

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
    # Los knobs de [memory] que deciden qué viaja en cada request (ver la Conversation
    # y config/crotolamo.toml); sin sus claves, el request es el de siempre.
    idle_min = _memory_knob(settings.memory, "idle_reset_min", None)
    conversation = Conversation(
        system_prompt(),
        max_turns=settings.memory.get("max_turns", 20),
        compaction=compaction,
        summarizer=_summarize if compaction else None,
        trim_chunk=_memory_knob(settings.memory, "trim_chunk", 1),
        stale_tool_turns=_memory_knob(settings.memory, "stale_tool_turns", None),
        idle_reset_s=idle_min * 60 if idle_min else None,
    )

    # La Fase 2 cablea el registry de tools; si está disponible, lo usamos.
    try:
        from crotolamo.core.agent import DEFAULT_DIRECT_TOOLS, BrainPolicy, ToolAgent
        from crotolamo.core.hooks import (
            datetime_prehook,
            make_facts_prehook,
            make_memoria_posthook,
            make_memoria_prehook,
            make_pinned_facts_prehook,
        )
        from crotolamo.core.memoria import get_memoria, mem0_instalado
        from crotolamo.core.router import route_schemas
        from crotolamo.tools import default_registry
        from crotolamo.safety.guard import Guard

        registry = default_registry()

        # ¿El cerebro es GLM (nube)? Entonces GLM RAZONA TODO: ve todas las
        # tools en cada turno (también en charla), redacta él cada respuesta y
        # no hay atajos regex sin LLM. Los atajos (fastpath, routing chico,
        # direct_tools) existían por el modelo LOCAL lento en CPU; con la nube
        # a ~1-2s solo le quitaban criterio al modelo.
        from crotolamo.core.engine import GLM, resolve_backend
        from crotolamo.core.fallback import FallbackLLM

        glm_brain = resolve_backend(settings) == GLM and isinstance(llm, FallbackLLM)

        # Tool routing (rendimiento en CPU): manda solo las tools relevantes a
        # cada consulta para que el turno local sea rápido (~2-5s vs ~130s).
        use_routing = settings.llm.get("tool_routing", True)
        max_tools = settings.llm.get("max_tools", 8)
        local_route: Callable[[str], list[dict[str, Any]]] | None = (
            (lambda t: route_schemas(registry, t, max_tools)) if use_routing else None
        )

        # Con GLM, UNA política por motor (ver BrainPolicy): el agente la consulta
        # en cada llamada, así que atajos y dieta de tools siguen al motor que de
        # verdad responde, que el breaker de FallbackLLM decide llamada a llamada.
        # Antes se miraba `llm.degraded` una vez por turno, ANTES de la llamada
        # que descubre la caída: ese turno mandaba las 42 tools al modelo local.
        # Con Ollama de cerebro, la de los kwargs de siempre (policy=None).
        policy_fn: Callable[[], BrainPolicy] | None = None
        answered_fn: Callable[[], BrainPolicy] | None = None
        route_fn = local_route
        # La dieta que recibe el local si la nube cae A MEDIA llamada (ver abajo).
        failover_policy: BrainPolicy | None = None
        if glm_brain:
            cloud_policy = BrainPolicy("cloud", fastpath=False, direct_tools=frozenset(),
                                       honor_tool_direct=False)
            # Degradado a Ollama (sin internet): el routing chico, siempre. Sus
            # atajos (fastpath y retorno directo) solo si el patrón lo pide con
            # [llm.glm].fallback_shortcuts; sin la clave, como siempre: sin atajos.
            if settings.llm.get("glm", {}).get("fallback_shortcuts", False):
                degraded_policy = BrainPolicy(
                    "local", fastpath=settings.llm.get("fastpath", True),
                    direct_tools=DEFAULT_DIRECT_TOOLS, route=local_route,
                )
            else:
                degraded_policy = BrainPolicy("degraded", fastpath=False,
                                              direct_tools=frozenset(),
                                              honor_tool_direct=False, route=local_route)

            def _policy_glm() -> BrainPolicy:
                return degraded_policy if llm.degraded else cloud_policy

            def _answered_glm() -> BrainPolicy:
                # El motor que RESPONDIÓ, según FallbackLLM: no `degraded` mirado
                # después, que ya es False si el cooldown venció en plena llamada local.
                return degraded_policy if llm.answered_by_secondary else cloud_policy

            def _route_now(text: str) -> list[dict[str, Any]]:
                # Solo compatibilidad (agent.route_fn): las tools que vería AHORA
                # el motor que respondería; el agente usa la política directamente.
                return _policy_glm().schemas_for(registry, text)

            policy_fn = _policy_glm
            answered_fn = _answered_glm
            route_fn = _route_now
            if local_route is not None:
                failover_policy = degraded_policy

        # Memoria semántica (mem0): con [memoria].enabled, los recuerdos
        # relevantes entran en CADA turno (pre-hook, con presupuesto de tiempo)
        # y la extracción de hechos va en segundo plano tras responder
        # (after-turn hook). Sustituye a la inyección única de hechos SQLite:
        # tras `crotolamo memoria migrar` ya viven aquí. Sin ella, todo como antes.
        memoria = get_memoria()
        recuerdos: _RecuerdosOmitidos | None = None
        if memoria.enabled and mem0_instalado():
            memoria.precalentar()
            if glm_brain:
                # Un recuerdo que ya va en la ventana no se repite en cada turno
                # (ver make_memoria_prehook). Mientras responde el local (breaker
                # abierto), sí: el ajuste a su num_ctx tira los turnos viejos. Si
                # la nube cae A MEDIA llamada, el local recibe los que se dejaron
                # fuera (_RecuerdosOmitidos; ver abajo, tras armar el agente).
                recuerdos = _RecuerdosOmitidos(memoria, lambda: tool_agent.last_turn)
                recall = make_memoria_prehook(cast("Memoria", recuerdos), conversation,
                                              ventana_entera=lambda: not llm.degraded)
            else:
                recall = make_memoria_prehook(memoria)
            pre_hooks = [recall, datetime_prehook]
            after_turn_hooks = [make_memoria_posthook(memoria)]
        else:
            # Los hechos SQLite, anclados tras el system prompt ([memory].pin_facts,
            # de fábrica): en el primer mensaje, como antes, se perdían al salir
            # ese mensaje de la ventana (turno max_turns+1) y con /reset.
            if settings.memory.get("pin_facts", True):
                facts_hook = make_pinned_facts_prehook(conversation)
            else:
                facts_hook = make_facts_prehook()
            pre_hooks = [facts_hook, datetime_prehook]
            after_turn_hooks = []

        tool_agent = ToolAgent(
            llm,
            conversation,
            registry=registry,
            guard=Guard.from_settings(settings),
            max_iterations=settings.llm.get("max_iterations", 6),
            confirm_fn=confirm_fn or _text_confirm,
            pre_hooks=pre_hooks,
            after_turn_hooks=after_turn_hooks,
            route_fn=route_fn,
            # Con GLM no hay comandos prehechos: todo lo razona y redacta él.
            # (Con `policy`, estos dos quedan como atributos de compatibilidad.)
            fastpath=False if glm_brain else settings.llm.get("fastpath", True),
            direct_tools=set() if glm_brain else None,
            policy=policy_fn,
            answered_policy=answered_fn,
            # Apagada de fábrica: la decide el patrón (ver config/crotolamo.toml).
            stop_on_repeated_failure=settings.llm.get("stop_on_repeated_failure", False),
            # Cuándo suelta el streaming con tools a la vista (con GLM, siempre):
            # "final" de fábrica = retener como siempre; "sentence" habla antes a
            # riesgo de decir un preámbulo (ver [voice] en config/crotolamo.toml).
            # Un valor raro lo rechaza ToolAgent: "final" y un aviso en el log.
            stream_release=str(settings.voice.get("stream_release", "final")),
        )
        if recuerdos is not None:
            # Cada petición que llegue al local en un turno cuyo pre-hook dejó
            # recuerdos fuera los lleva de vuelta: la de la caída (tras el shaper)
            # y las siguientes del turno, que van directas con el breaker abierto.
            llm.secondary = _LocalConRecuerdos(llm.secondary, recuerdos)
        if failover_policy is not None:
            # La nube cayó A MEDIA llamada: esa petición iba armada para GLM (todas
            # las tools). El local recibe la dieta chica del texto limpio del turno,
            # la MISMA lista que el agente le dará en las llamadas siguientes (se
            # enruta una vez por turno). Sin tool_routing no se instala: el local
            # recibe todo, como siempre.
            def _shape(messages, tools):
                return messages, tool_agent.schemas_for_turn(failover_policy)

            llm.shape_for_secondary = _shape
        if compaction:
            # El resumen de [memory].compaction, DESPUÉS de hablar: antes era una
            # llamada al LLM dentro de add_user, antes de la del turno (un viaje
            # de ida y vuelta entero antes de la primera palabra, en cada turno
            # con la ventana llena). Lo que sale de la ventana sigue a la vista
            # tal cual hasta que el resumen lo sustituye. Al final del armado: si
            # algo de arriba falla, el Agent sin tools resume como siempre.
            conversation.summarize_in_background = True
            tool_agent.after_turn_hooks.append(CompactionWorker(conversation).after_turn)
        agent: Agent = tool_agent
    except Exception:
        # Degradar a un agente SIN tools mantiene a Crotolamo hablando, pero no
        # puede ser silencioso: un ImportError en un módulo de tools dejaba al
        # asistente sordo a "abre X" sin ninguna pista en el log.
        log.exception(
            "no pude armar las tools; Crotolamo arranca SIN herramientas (solo charla)"
        )
        agent = Agent(llm, conversation)

    return agent, conversation


def _cerrar_agente(agent) -> None:
    """Al salir, el agente apaga sus hilos (los de las tools en paralelo, ver
    ToolAgent.close). Un fallo aquí no puede tapar la salida: queda en el log."""
    cerrar = getattr(agent, "close", None)
    if cerrar is None:
        return
    try:
        cerrar()
    except Exception as error:  # noqa: BLE001 - salir siempre
        log.warning("no pude cerrar el agente al salir: %s", error)


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
            # ToolAgent guarda los tool_calls en formato Ollama (GLM se normaliza
            # a lo mismo): el nombre va DENTRO de "function", no arriba.
            names = ", ".join(
                (c.get("function") or {}).get("name") or "?" for c in msg.tool_calls
            )
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
            _cerrar_agente(agent)
            return 0

        if not user:
            continue

        low = user.lower()
        if low in {"salir", "exit", "quit"}:
            print("Crotolamo apagado, patrón.")
            _cerrar_agente(agent)
            return 0
        if low == "/reset":
            conversation.reset()
            print("Memoria limpia, patrón.")
            continue
        if low == "/historial":
            _show_history(conversation)
            continue

        try:
            if can_stream and isinstance(agent, ToolAgent):
                print("\nCrotolamo > ", end="", flush=True)
                agent.handle_turn(user, on_token=lambda t: print(t, end="", flush=True))
                print("\n")
            else:
                reply = agent.handle_turn(user)
                print(f"\nCrotolamo > {reply}\n")
        except Exception as error:  # noqa: BLE001 - un turno roto no mata el shell
            # Como el listener: el traceback al log y al patrón una frase en
            # personaje. Sin esto, UNA excepción imprevista (un hook, la DB, un
            # print de texto que no se puede codificar) cerraba la sesión entera
            # con su historial. Los LLMError ya vuelven como respuesta normal.
            log.exception("el turno reventó: %s", error)
            # En streaming el "Crotolamo > " (y quizá media frase) ya salió.
            print(f"\n{_TURNO_ROTO}\n" if can_stream else f"\nCrotolamo > {_TURNO_ROTO}\n")


if __name__ == "__main__":
    import sys

    from crotolamo.logging_setup import setup_logging

    # Como script (`python interfaces/shell.py --stream`) no se pasa por
    # crotolamo.__main__: run_shell() a secas ignoraba --stream, y sin
    # setup_logging los INFO se perdían y los WARNING salían sin formato.
    setup_logging()
    raise SystemExit(run_shell(sys.argv[1:]))
