"""El agente. En la Fase 1 es LLM + memoria; la Fase 2 le añade el loop de tools.

handle_turn(text) -> respuesta final en texto.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from crotolamo.core import fastpath as fastpath_mod
from crotolamo.core.budget import TOOL_RESULT_CAP
from crotolamo.core.engine import LLMEngine
from crotolamo.core.llm import LLMError
from crotolamo.core.streaming import (
    RELEASE_FINAL,
    RELEASE_POLICIES,
    Emitter as _Emitter,
    LiveStreamer as _LiveStreamer,
    TurnCancelled,
)
from crotolamo.core.tool_parsing import (
    coerce_text_tool_calls as _coerce_text_tool_calls,
    is_hard_error as _is_hard_error,
    is_soft_error as _is_soft_error,
)
from crotolamo.core.turn import (
    BLOCKED,
    DECLINED,
    HARD_ERROR,
    OK,
    REPEATED,
    SOFT_ERROR,
    UNKNOWN_TOOL,
    VIA_CANCELLED,
    VIA_DIRECT,
    VIA_EXHAUSTED,
    VIA_FASTPATH,
    VIA_LLM_ERROR,
    CallRecord,
    DaemonPool,
    TurnInfo,
    call_key,
    hook_wants_turn,
)
from crotolamo.logging_setup import get_logger
from crotolamo.core.memory import Conversation
from crotolamo.tools.base import truncate_for_context

log = get_logger("core.agent")

# Lo que se contesta cuando el modelo devuelve content vacío sin haber hecho nada.
_EN_BLANCO = "Me quedé en blanco, patrón. Repíteme eso."
# Cierre del historial de un turno cancelado (barge-in) y resultado de las tools
# que ya no se ejecutaron: cada tool_call necesita su respuesta.
_INTERRUMPIDO = "(Turno interrumpido por el patrón.)"
_TOOL_CANCELADA = "Cancelado: el patrón interrumpió."
# Tope de hilos para las tools de solo lectura de un mismo mensaje del modelo
# (ver ToolAgent._parallel_plan): las que pasen esperan turno en el pool.
_PARALLEL_TOOLS = 4


class Agent:
    # `llm` es el CONTRATO (LLMEngine), no el cliente de Ollama: build_agent
    # también pasa aquí un GLMClient o un FallbackLLM.
    def __init__(self, llm: LLMEngine, conversation: Conversation) -> None:
        self.llm = llm
        self.conversation = conversation

    def handle_turn(self, text: str, on_token=None, should_abort=None) -> str:
        """Un turno conversacional con memoria. Sin tools todavía (Fase 1).

        on_token: misma firma que en ToolAgent (el listener lo usa para hablar
        en streaming); aquí el fallback emite la respuesta completa de golpe.
        should_abort: también la de ToolAgent, para que quien la cablee no
        reviente con este agente (el shell cae a él si el registry falla). Hay
        un solo punto de control, antes de la llamada al LLM, que es una y no
        es streaming.
        """
        self.conversation.add_user(text)
        if should_abort is not None and should_abort():
            self.conversation.add_assistant(_INTERRUMPIDO)
            return ""

        try:
            response = self.llm.chat(self.conversation.to_messages())
            reply = response.content or _EN_BLANCO
        except LLMError as error:
            # El error ES la respuesta del turno: se anota y se emite como
            # cualquier otra (ver el mismo caso en ToolAgent._handle_turn).
            reply = str(error)

        self.conversation.add_assistant(reply)
        if on_token is not None:
            on_token(reply)
        return reply

    def close(self) -> None:
        """Para quien apaga el agente sin saber cuál le tocó: este no tiene hilos."""


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

RouteFn = Callable[[str], list[dict[str, Any]]]


@dataclass(frozen=True)
class BrainPolicy:
    """Atajos y dieta de tools del motor que va a RESPONDER, no del configurado.

    POR QUÉ: con GLM de cerebro y Ollama de respaldo, qué motor contesta lo decide
    el breaker de FallbackLLM en CADA llamada. Si los atajos se fijaban al armar
    el agente (o una vez por turno, antes de la llamada que descubre la caída), el
    modelo local heredaba la política de la nube: las 42 schemas contra un
    num_ctx de 2048 en CPU. ToolAgent la consulta antes de cada llamada y después
    de cada respuesta (ver `_handle_turn`).

    route=None manda todas las tools del registry. honor_tool_direct: si el flag
    `Tool.direct=True` de una tool cuenta como retorno directo además del set.
    """

    name: str
    fastpath: bool
    direct_tools: frozenset[str]
    honor_tool_direct: bool = True
    route: RouteFn | None = None

    def schemas_for(self, registry, routing_text: str) -> list[dict[str, Any]]:
        """Las tool-schemas que vería este motor para el texto LIMPIO del turno."""
        return self.route(routing_text) if self.route is not None else registry.schemas()


class ToolAgent(Agent):
    """El loop agéntico: el LLM pide tools, las ejecutamos (bajo guard) y le
    devolvemos el resultado para que decida el siguiente paso.
    """

    def __init__(
        self,
        llm: LLMEngine,
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
        after_turn_hooks: list[Callable[[str, str], None]] | None = None,
        stream_release: str = RELEASE_FINAL,
        policy: Callable[[], BrainPolicy] | None = None,
        answered_policy: Callable[[], BrainPolicy] | None = None,
        stop_on_repeated_failure: bool = False,
    ) -> None:
        super().__init__(llm, conversation)
        # Cuándo soltar el texto en streaming si la llamada lleva tools a la
        # vista: "final" (retener hasta saber que no hubo tool-call, lo de
        # siempre) o "sentence" (en cuanto cierra la 1ª frase y le sigue más
        # texto; ver LiveStreamer). Un valor desconocido no puede dejar al
        # patrón sin voz ni hablar preámbulos: cae a "final" avisando.
        if stream_release not in RELEASE_POLICIES:
            log.warning("stream_release=%r no existe (valen %s); uso %r",
                        stream_release, ", ".join(RELEASE_POLICIES), RELEASE_FINAL)
            stream_release = RELEASE_FINAL
        self.stream_release = stream_release
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
        # (comportamiento clásico de 2 llamadas, útil p.ej. para medir/comparar),
        # también el de las tools con el flag Tool.direct: ver _is_direct.
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
        # Hooks DESPUÉS del turno: reciben (texto del patrón, respuesta) ya con
        # la respuesta devuelta al caller. Hoy: la memoria semántica encola la
        # extracción de hechos. Nunca bloquean ni alteran la respuesta.
        self.after_turn_hooks: list[Callable[[str, str], None]] = list(after_turn_hooks or [])
        # Política por llamada (ver BrainPolicy). Sin `policy`, la de los kwargs de
        # arriba (fastpath, direct_tools, route_fn), leída en vivo de los atributos:
        # mutarlos tras construir sigue funcionando como siempre. Con `policy`,
        # esos tres quedan solo como atributos de compatibilidad.
        self._policy_fn: Callable[[], BrainPolicy] = policy or self._static_policy
        # La política del motor que RESPONDIÓ la llamada que acaba de volver (decide
        # el short-circuit). Sin ella, la misma `policy` evaluada tras la llamada.
        # Con GLM, build_agent la saca de lo que anotó FallbackLLM: deducirla del
        # breaker fallaba si el cooldown vencía MIENTRAS respondía el local.
        self._answered_fn: Callable[[], BrainPolicy] = answered_policy or self._policy_fn
        # Texto LIMPIO del turno en curso (sin pre-hooks): con él enruta el shaper
        # que FallbackLLM aplica si la nube cae A MEDIA llamada (ver build_agent).
        self.routing_text: str = ""
        # (política, schemas) ya enrutadas en ESTE turno: ver schemas_for_turn.
        self._turn_schemas: list[tuple[BrainPolicy, list[dict[str, Any]]]] = []
        # Nº de turno, para los ids estables de las tool_calls que llegan sin id.
        self._turn_seq = 0
        # El registro del último turno (core/turn.py): qué comando anclado era,
        # por dónde salió y cómo acabó cada tool. Lo reciben los hooks que lo
        # piden (wants_turn); el siguiente turno lo sustituye.
        self.last_turn: TurnInfo | None = None
        # Hilos de las tools de solo lectura que el modelo pide juntas (ver
        # _parallel_plan). Del agente: se crean con el primer lote que los usa, se
        # reusan en los siguientes y close() los apaga (después, todo en serie).
        # Daemon (DaemonPool): la salida del proceso no espera a una tool a medias,
        # igual que cuando corrían en serie en el hilo del agente.
        self._pool: DaemonPool | None = None
        self._pool_lock = threading.Lock()
        self._pool_closed = False
        # Política de repetición ([llm].stop_on_repeated_failure, apagada de
        # fábrica: la decide el patrón). Ver _repeated_failure.
        self.stop_on_repeated_failure = stop_on_repeated_failure

    def close(self) -> None:
        """Apaga los hilos de las tools en paralelo. Idempotente y sin esperar: los
        ociosos salen ya; uno a media tool acaba la suya (acotada por el timeout
        de la propia tool) y sale. Lo ya encolado se termina: un turno en curso
        no se queda sin resultados. Después, los lotes corren en serie."""
        with self._pool_lock:
            self._pool_closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown()

    def _tool_pool(self) -> DaemonPool | None:
        """El pool del agente (perezoso), o None si ya se cerró."""
        with self._pool_lock:
            if self._pool_closed:
                return None
            if self._pool is None:
                self._pool = DaemonPool(_PARALLEL_TOOLS, "ToolAgent-tool")
            return self._pool

    def _static_policy(self) -> BrainPolicy:
        # Un direct_tools VACÍO apaga también el flag Tool.direct (ver _is_direct).
        return BrainPolicy("static", self.fastpath, frozenset(self.direct_tools),
                           honor_tool_direct=bool(self.direct_tools), route=self.route_fn)

    def schemas_for_turn(self, policy: BrainPolicy) -> list[dict[str, Any]]:
        """Las tools de `policy` para el turno en curso, enrutadas UNA vez por turno.

        El router no es puro (el round-robin entre servers MCP avanza en cada
        consulta): enrutar dos veces daba dos sets distintos dentro del mismo
        turno, sin cache de prefijo para la 2ª llamada. Lo comparten el agente y
        el shaper de FallbackLLM (build_agent), que adapta la llamada en la que
        cae la nube: el local ve la misma lista en todas las llamadas del turno.
        """
        for vista, schemas in self._turn_schemas:
            if vista == policy:
                return schemas
        schemas = policy.schemas_for(self.registry, self.routing_text)
        self._turn_schemas.append((policy, schemas))
        return schemas

    def _execute_call(self, name: str, arguments: dict) -> str:
        return self._execute_call_rec(name, arguments)[0]

    def _execute_call_rec(self, name: str, arguments: dict) -> tuple[str, str]:
        """(resultado, cómo acabó): el texto que va al historial, el mismo de
        siempre, y su clasificación para el registro del turno (core/turn.py)."""
        tool, arguments, decision = self._prepare_call(name, arguments)
        if tool is None:
            return f"No tengo una tool llamada '{name}', patrón.", UNKNOWN_TOOL
        if not decision.allowed:
            return decision.reason, BLOCKED
        if decision.needs_confirmation and not self.confirm_fn(decision.reason):
            return "Cancelado por el patrón.", DECLINED
        return self._run_prepared(name, arguments)

    def _prepare_call(self, name: str, arguments: dict) -> tuple[Any, dict, Any]:
        """(tool, argumentos que le llegan, decisión del guard) sin ejecutar nada
        ni preguntar al patrón; (None, arguments, None) si la tool no existe. El
        guard es puro: lo puede consultar el plan de un lote en paralelo antes de
        correr nada y decidir con lo mismo que decidirá la ejecución."""
        tool = self.registry.get(name)
        if tool is None:
            return None, arguments, None

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

        return tool, arguments, self.guard.check(tool, arguments)

    def _run_prepared(self, name: str, arguments: dict) -> tuple[str, str]:
        """Corre una tool ya filtrada, permitida y confirmada, y clasifica su
        resultado."""
        result = self.registry.run(name, arguments)
        if _is_hard_error(result):
            return result, HARD_ERROR
        return result, SOFT_ERROR if _is_soft_error(result) else OK

    def _safe_execute(self, name: str, arguments: dict) -> str:
        return self._safe_execute_rec(name, arguments)[0]

    def _safe_execute_rec(self, name: str, arguments: dict,
                          prepared: bool = False) -> tuple[str, str]:
        """_execute_call que NUNCA propaga: siempre hay un resultado que anotar.

        Registry.run ya atrapa lo que revienta DENTRO de la tool, pero el guard
        (p.ej. un '\\0' en la ruta) o el confirm_fn (STT/sounddevice) pueden
        lanzar antes. Si eso escapaba tras add_assistant(tool_calls=...), el
        historial quedaba con un assistant pidiendo tools sin su resultado y
        las APIs OpenAI-compatibles (GLM) rechazaban TODOS los turnos siguientes
        hasta /reset. El prefijo "La tool '" lo marca como fallo duro (sin
        short-circuit), igual que un reventón dentro de la tool.

        Es también el punto único por el que pasa TODO resultado (fast-path,
        tool-calls nativos y coaccionados, MCP) antes de entrar al historial, y
        de ahí a cada request de los próximos turnos: aquí se aplica el tope
        TOOL_RESULT_CAP. Acota las tools sin tope propio (find_in_project,
        recall_facts, list_project_tree, list_processes...), que mandaban
        cientos de KB; lo que ya cabe no cambia un byte, y el recorte guarda la
        cabeza, donde están los prefijos que marcan un fallo duro.

        prepared=True: `arguments` ya salieron de _prepare_call con el guard a
        favor y sin confirmación pendiente (un lote en paralelo): solo se corre.
        """
        try:
            if prepared:
                result, outcome = self._run_prepared(name, arguments)
            else:
                result, outcome = self._execute_call_rec(name, arguments)
        except Exception as error:  # noqa: BLE001 - un fallo del guard/confirm no rompe el turno
            log.exception("la tool '%s' reventó fuera del registry", name)
            result, outcome = f"La tool '{name}' reventó, patrón: {error}", HARD_ERROR
        return truncate_for_context(result, TOOL_RESULT_CAP), outcome

    def _run_call(self, turn: TurnInfo, name: str, arguments: dict) -> str:
        """Ejecuta UNA tool del turno y deja su registro (clave con los argumentos
        tal como los pidió el modelo, antes del filtro de strict_args)."""
        previo = self._repeated_failure(turn, name, arguments)
        if previo is not None:
            # Ni se corre ni se vuelve a preguntar: al modelo le vuelve el mismo
            # resultado y la siguiente petición va sin tools (ver _handle_turn).
            log.info("no repito '%s': ya acabó en %s en este turno", name, previo.outcome)
            turn.calls.append(CallRecord(name, previo.key, REPEATED, previo.result))
            return previo.result
        result, outcome = self._safe_execute_rec(name, arguments)
        turn.calls.append(CallRecord(name, call_key(name, arguments), outcome, result))
        return result

    def _repeated_failure(self, turn: TurnInfo, name: str,
                          arguments: dict) -> CallRecord | None:
        """El fallo anterior de ESTA llamada (misma tool y argumentos) si, con
        [llm].stop_on_repeated_failure, repetirla no tiene sentido; None = se corre.

        Sin la política el loop no recordaba qué había fallado: un modelo que
        repite una llamada rota quemaba todas las iteraciones (~110 KB con GLM)
        para acabar en la frase enlatada, y en voz concurrente el aviso de
        "dímela por el shell" se decía en cada reintento. Una tool que no existe,
        una bloqueada por el guard o una que el patrón no confirmó se frenan a la
        primera repetición (darían lo mismo); un fallo de la tool o el aviso de un
        server MCP, tras UN reintento (puede ser pasajero).
        """
        if not self.stop_on_repeated_failure:
            return None
        key = call_key(name, arguments)
        previo = turn.last_failure(key)
        if previo is None:
            return None
        if previo.outcome in (UNKNOWN_TOOL, BLOCKED, DECLINED, REPEATED):
            return previo
        if previo.outcome in (HARD_ERROR, SOFT_ERROR) and turn.failures(key) >= 2:
            return previo
        return None

    def _parallel_plan(self, turn: TurnInfo,
                       calls: list[dict]) -> list[tuple[str, dict, dict, Any]] | None:
        """Las llamadas de UN mensaje del modelo, listas para correr a la vez:
        [(nombre, argumentos pedidos, argumentos filtrados, carril)], o None para
        correrlas en serie como siempre.

        Antes el loop las corría una tras otra aunque fueran lecturas sueltas: tres
        páginas web pagaban tres esperas de red. A la vez solo si TODAS son de
        solo lectura (Tool.parallel_safe; `safe` no basta: open_app y
        music_control son safe y cambian cosas), el guard las deja sin confirmar
        y hay al menos dos carriles. Lo demás (una tool que cambia algo, una que
        no existe, una bloqueada o que pide confirmación, una repetición que la
        política frena, cualquier excepción aquí) va en serie, sin tocar: las
        confirmaciones nunca se cruzan con nada y se preguntan en su orden.

        Con la política de repetición, también la MISMA llamada dos veces en el
        lote: si la 1ª falla, la serie frena la 3ª (repeated) y la siguiente
        petición va sin tools; el plan se decide antes de correr nada y las
        habría corrido todas.
        """
        if len(calls) < 2 or self._pool_closed:
            return None
        try:
            # Primero el flag (barato): si una no lo lleva, ni se consulta el guard.
            tools = [self.registry.get(call["name"]) for call in calls]
            if not all(getattr(tool, "parallel_safe", False) is True for tool in tools):
                return None
            claves: set[str] = set()
            plan: list[tuple[str, dict, dict, Any]] = []
            for i, call in enumerate(calls):
                name, pedidos = call["name"], call.get("arguments", {})
                tool, filtrados, decision = self._prepare_call(name, pedidos)
                if tool is None or not decision.allowed or decision.needs_confirmation:
                    return None
                if self._repeated_failure(turn, name, pedidos) is not None:
                    return None
                if self.stop_on_repeated_failure:
                    clave = call_key(name, pedidos)
                    if clave in claves:
                        return None
                    claves.add(clave)
                lane = getattr(tool, "parallel_lane", None)
                plan.append((name, pedidos, filtrados, ("carril", lane) if lane else i))
        except Exception as error:  # noqa: BLE001 - la serie lo repite y anota el fallo
            log.debug("el lote va en serie: no pude planearlo (%s)", error)
            return None
        if len({lane for *_, lane in plan}) < 2:
            return None
        return plan

    def _run_parallel(self, turn: TurnInfo,
                      plan: list[tuple[str, dict, dict, Any]]) -> Iterator[tuple[str, str]]:
        """Corre el plan en el pool del agente y da (nombre, resultado) en el ORDEN
        de las llamadas, que es el del historial y el del registro del turno.

        Cada carril va en un hilo, sus llamadas una tras otra. Cada llamada pasa por
        _safe_execute_rec: el mismo texto (reventón, recorte) que en serie. Si el
        pool no da hilo (cerrado, sin hilos libres en el sistema), ese carril
        corre aquí mismo, en serie.
        """
        carriles: dict[Any, list[int]] = {}
        for i, (*_, lane) in enumerate(plan):
            carriles.setdefault(lane, []).append(i)

        def correr(indices: list[int]) -> list[tuple[str, str]]:
            return [self._safe_execute_rec(plan[i][0], plan[i][2], prepared=True)
                    for i in indices]

        pool = self._tool_pool()
        futuros: dict[Any, Future[list[tuple[str, str]]] | None] = {}
        for lane, indices in carriles.items():
            try:
                futuros[lane] = pool.submit(correr, indices) if pool is not None else None
            except RuntimeError as error:   # pool cerrado a medias, "can't start new thread"
                log.warning("tools en serie: el pool no aceptó el carril (%s)", error)
                futuros[lane] = None
        hechos: dict[int, tuple[str, str]] = {}
        for i, (name, pedidos, _filtrados, lane) in enumerate(plan):
            if i not in hechos:
                futuro = futuros[lane]
                salidas = futuro.result() if futuro is not None else correr(carriles[lane])
                hechos.update(zip(carriles[lane], salidas))
            result, outcome = hechos[i]
            turn.calls.append(CallRecord(name, call_key(name, pedidos), outcome, result))
            yield name, result

    def _is_direct(self, name: str, policy: BrainPolicy | None = None) -> bool:
        """True si la tool es de retorno directo para la política del motor que
        RESPONDIÓ: está en su `direct_tools`, o la política honra el flag
        `Tool.direct=True` y la definición de la tool lo lleva.

        Con los kwargs del constructor, un `direct_tools` VACÍO apaga el
        short-circuit entero, flag incluido (honor_tool_direct=False): es el
        interruptor de "todo lo razona y redacta el modelo". Antes el flag se
        colaba por debajo y list_windows / app_status devolvían su lista cruda
        sin pasar por el modelo; la política de la nube de build_agent tampoco
        lo honra.
        """
        if policy is None:
            policy = self._policy_fn()
        if name in policy.direct_tools:
            return True
        if not policy.honor_tool_direct:
            return False
        tool = self.registry.get(name)
        return bool(tool is not None and getattr(tool, "direct", False))

    def _apply(self, hooks, value: str, turn: TurnInfo | None = None) -> str:
        for hook in hooks:
            try:
                # Solo el hook que lo pide recibe el turno; los demás, como siempre.
                if turn is not None and hook_wants_turn(hook):
                    value = hook(value, turn=turn)
                else:
                    value = hook(value)
            except Exception as error:  # noqa: BLE001 - un hook roto no mata el turno
                log.warning("hook falló: %s", error)
        return value

    def _log_release(self, calls, content: str) -> None:
        """El dato para decidir [voice].stream_release con el motor REAL: cuánto
        preámbulo suelta (o soltaría) "sentence" antes de un tool-call."""
        if not calls:
            log.debug("stream_release: 'sentence' adelanta la primera frase")
            return
        verbo = "soltó" if self.stream_release != RELEASE_FINAL else "habría hablado"
        log.info("stream_release: 'sentence' %s un preámbulo de %d caracteres "
                 "antes del tool-call", verbo, len(content))

    def handle_turn(self, text: str, on_token=None, should_abort=None) -> str:
        """should_abort: callable sin argumentos; si devuelve True el turno se
        corta en el siguiente punto de control (cada línea del stream, haya
        texto o no; cada iteración; antes de cada tool; antes de que
        FallbackLLM arranque el local) y devuelve "". None = nunca se cancela."""
        # Lo primero, el registro del turno: qué comando anclado es (el
        # clasificador del fast-path, microsegundos, aunque el atajo esté
        # apagado), sobre el texto CRUDO. Los hooks que lo piden lo leen.
        turn = TurnInfo(text=text, command=fastpath_mod.match(text))
        self.last_turn = turn
        try:
            reply = self._handle_turn(text, turn, on_token, should_abort)
        except TurnCancelled:
            # El historial cierra con un assistant (como al agotar iteraciones)
            # y un turno que el patrón canceló no alimenta la memoria semántica.
            turn.via = VIA_CANCELLED
            self.conversation.add_assistant(_INTERRUMPIDO)
            return ""
        # El texto CRUDO del patrón (sin pre-hooks) y la respuesta final. Un hook
        # roto no toca la respuesta: ya está calculada y ya se habló.
        for hook in self.after_turn_hooks:
            try:
                if hook_wants_turn(hook):
                    con_turno: Callable[..., None] = hook
                    con_turno(text, reply, turn=turn)
                else:
                    hook(text, reply)
            except Exception as error:  # noqa: BLE001
                log.warning("after-turn hook falló: %s", error)
        return reply

    def _handle_turn(self, text: str, turn: TurnInfo, on_token=None,
                     should_abort=None) -> str:
        # Enrutamos sobre el texto LIMPIO del patrón (antes de que los pre-hooks le
        # antepongan fecha/hechos), que es la señal real de intención. El set de
        # tools se mantiene en todas las iteraciones mientras no cambie el motor
        # que responde, para no romper el cache de prefijo dentro del turno.
        routing_text = text
        self.routing_text = routing_text
        self._turn_schemas = []
        self._turn_seq += 1
        ids_dados = 0  # ids estables asignados en este turno (ver abajo)
        # M4: pre-hooks enriquecen la entrada antes de llegar al LLM.
        text = self._apply(self.pre_hooks, text, turn)
        # Política del motor que respondería AHORA (con GLM de cerebro: la nube
        # o, con el breaker abierto, el local), con sus atajos y su dieta.
        policy = self._policy_fn()

        # Fast-path (misión velocidad): comando inequívoco -> tool directa sin
        # LLM (~0.1s en vez de ~17-22s). Solo si la regla cubre el comando
        # COMPLETO; cualquier matiz cae al loop normal de abajo. El historial
        # se alimenta igual para que la conversación no pierda el turno.
        # El match ya está en el turno (handle_turn): no se repite.
        if policy.fastpath:
            hit = turn.command
            if hit is not None:
                fast_name, fast_args = hit
                result = self._run_call(turn, fast_name, dict(fast_args))
                if not _is_hard_error(result):
                    turn.via = VIA_FASTPATH
                    reply = self._apply(self.post_hooks, result)
                    self.conversation.add_user(text)
                    self.conversation.add_assistant(reply)
                    if on_token is not None:
                        on_token(reply)
                    return reply
                # Fallo duro del atajo (p.ej. hyprctl ausente): que lo razone
                # el LLM como siempre, sin ensuciar el historial.

        self.conversation.add_user(text)
        # add_user pudo llamar al LLM (el resumen de [memory].compaction) y esa
        # llamada abrir o cerrar el breaker: las tools salen de la política de ahora.
        policy = self._policy_fn()
        schemas = self.schemas_for_turn(policy)
        known = set(self.registry.names())

        # ¿Ya salió texto por on_token en este turno? Lo necesita la salida por
        # LLMError de abajo para no pegar el error a una frase cortada a medias.
        # Uno por TURNO; el streamer (uno por llamada) emite a través de él.
        emit = _Emitter(on_token) if on_token is not None else None
        # Con "sentence" lo que se suelta en vivo no pasa por los post-hooks
        # (solo alcanzan al texto final): el limpiador de preámbulos meta se
        # aplica a lo acumulado antes de soltarlo, si el agente lo usa.
        clean_head = None
        if self.stream_release != RELEASE_FINAL:
            from crotolamo.core.hooks import meta_preamble_cleaner

            if meta_preamble_cleaner in self.post_hooks:
                clean_head = meta_preamble_cleaner

        # (tool, resultado) de las iteraciones ANTERIORES de este turno. Lo miran
        # el cierre por content vacío y el short-circuit: ambos necesitan saber
        # si en el turno ya se hizo algo que el patrón todavía no ha oído.
        previos: list[tuple[str, str]] = []
        # Con la política de repetición: la próxima petición va sin tools (ver
        # _repeated_failure). Sin la política, siempre False.
        sin_tools = False

        for _ in range(self.max_iterations):
            if should_abort is not None and should_abort():
                raise TurnCancelled()
            # El motor pudo cambiar desde la llamada anterior (la nube cayó a
            # media llamada y respondió el local, o volvió): la dieta sigue al
            # que va a responder. Si no cambió, el set queda fijo (cache de prefijo);
            # si cambió, es el que ya le dio el shaper a ese motor en este turno.
            current = self._policy_fn()
            if current != policy:
                policy = current
                schemas = self.schemas_for_turn(policy)
            # Una tool retirada a mitad de turno (un server MCP desconectado por
            # timeouts, M4) no debe seguir a la vista del modelo en la siguiente
            # iteración: filtramos por presencia en el registry (barato).
            if schemas:
                schemas = [
                    s for s in schemas
                    if self.registry.get(s.get("function", {}).get("name", "")) is not None
                ]
            # Tras frenar una repetición, sin tools: el modelo explica con sus
            # palabras lo que no salió (y, sin tools a la vista, se habla en vivo).
            a_la_vista = [] if sin_tools else schemas
            sin_tools = False
            # Con tools a la vista, el modelo puede anunciar lo que va a hacer antes
            # de pedirla; retenemos hasta saber si hubo tool_call (o, con
            # "sentence", hasta que cierre la 1ª frase). Sin tools (charla), se
            # habla en vivo. Al motor va el streamer ENTERO, no su `feed`:
            # FallbackLLM le pide `restart` antes de repetir la llamada en local
            # y los parsers consultan su `checkpoint` con cada línea del stream.
            streamer = (
                _LiveStreamer(emit, hold_until_done=bool(a_la_vista),
                              release=self.stream_release, should_abort=should_abort,
                              clean_head=clean_head)
                if emit is not None else None
            )
            turn.llm_calls += 1
            try:
                if streamer is not None:
                    response = self.llm.chat_stream(
                        self.conversation.to_messages(), tools=a_la_vista, on_token=streamer,
                    )
                else:
                    response = self.llm.chat(self.conversation.to_messages(), tools=a_la_vista)
            except LLMError as error:
                # El error ES la respuesta del turno y sale como las demás. Antes
                # solo se devolvía: (1) quien lee on_token (shell --stream, la
                # voz en streaming) no veía ni oía nada, o una frase truncada y
                # silencio; (2) el user quedaba sin assistant (o un bloque tool
                # abierto si falló tras ejecutar tools) y al volver el motor el
                # modelo recibía de golpe las órdenes sin contestar.
                reply = str(error)
                turn.via = VIA_LLM_ERROR
                self.conversation.add_assistant(reply)
                if emit is not None:
                    # Si ya se habló media frase en vivo, el error va en línea aparte.
                    emit(f"\n{reply}" if emit.released else reply)
                return reply
            # Política del motor que ACABA de responder (pudo caer al local a
            # media llamada): es la que decide el short-circuit de abajo.
            answered = self._answered_fn()

            calls = response.tool_calls
            native = bool(calls)
            # Fallback: el modelo puso el tool-call como JSON en content.
            if not calls:
                calls = _coerce_text_tool_calls(response.content, known)
            if streamer is not None and streamer.would_release_early:
                self._log_release(calls, response.content)

            if not calls:
                # Content vacío: "Listo" solo si en este turno corrió alguna tool
                # (el modelo no tiene nada que añadir a lo hecho). Sin ninguna NO
                # se hizo nada (contexto desbordado, respuesta solo-razonamiento) y
                # "Listo" daba la orden por cumplida; se dice lo que el Agent base.
                reply = response.content or ("Listo, patrón." if previos else _EN_BLANCO)
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
            # Una llamada sin id (las del texto, las nativas de Ollama) se guarda con
            # uno estable. Sin él, to_openai_messages las numera por POSICIÓN al
            # armar cada request para la nube: al salir un turno de la ventana,
            # todas las de después cambiaban de id (y de bytes) y el cache de
            # prefijo del proveedor se perdía. Los ids que trae el motor, tal cual.
            if tool_calls_payload:
                con_id = []
                for pedida in tool_calls_payload:
                    if not pedida.get("id"):
                        pedida = {**pedida, "id": f"call_t{self._turn_seq}_{ids_dados}"}
                        ids_dados += 1
                    con_id.append(pedida)
                tool_calls_payload = con_id
            self.conversation.add_assistant(
                response.content if native else "",
                tool_calls=tool_calls_payload,
            )

            results: list[tuple[str, str]] = []
            antes = len(turn.calls)
            plan = self._parallel_plan(turn, calls)
            if plan is not None:
                # Lecturas sueltas: a la vez (ver _parallel_plan), con el punto de
                # control que la serie tiene antes de la 1ª tool.
                if should_abort is not None and should_abort():
                    for pending in calls:
                        self.conversation.add_tool_result(pending["name"], _TOOL_CANCELADA)
                    raise TurnCancelled()
                for name, result in self._run_parallel(turn, plan):
                    self.conversation.add_tool_result(name, result)
                    results.append((name, result))
            else:
                for call in calls:
                    if should_abort is not None and should_abort():
                        # Las tools que faltan ya no corren (el patrón interrumpió),
                        # pero cada tool_call necesita su resultado en el historial.
                        for pending in calls[len(results):]:
                            self.conversation.add_tool_result(pending["name"], _TOOL_CANCELADA)
                        raise TurnCancelled()
                    result = self._run_call(turn, call["name"], call.get("arguments", {}))
                    self.conversation.add_tool_result(call["name"], result)
                    results.append((call["name"], result))

            # Short-circuit (Misión 1): si el modelo pidió EXACTAMENTE una tool,
            # esa tool es de retorno directo y NO falló duro, devolvemos su output
            # tal cual como respuesta final, ahorrándonos la 2ª llamada al LLM
            # (~7-9s en CPU). Casos con 2+ tools, tool no-directa o fallo duro
            # caen al comportamiento normal (otra iteración => 2ª llamada).
            # "Exactamente una" es del TURNO, no de la iteración: si antes ya
            # corrió otra tool ("busca mi tesis y dime la RAM"), devolver solo
            # esta dejaba aquel resultado (o su fallo) sin contar; ahí redacta
            # el modelo. Lo único previo que no estorba son los intentos fallidos
            # de ESTA misma tool: de un reintento no hay nada más que decir.
            if len(results) == 1:
                name, result = results[0]
                if (
                    self._is_direct(name, answered)
                    and not _is_hard_error(result)
                    and all(n == name and _is_hard_error(r) for n, r in previos)
                ):
                    # Mismos post-hooks y misma actualización de memoria que el
                    # camino final normal (líneas del bloque `if not calls`).
                    turn.via = VIA_DIRECT
                    reply = self._apply(self.post_hooks, result)
                    if streamer is not None:
                        # En modo voz/streaming el caller solo lee on_token; hay
                        # que emitir el texto o el patrón se queda en silencio.
                        streamer.flush_if_held(reply)
                    self.conversation.add_assistant(reply)
                    return reply
            # Si no hubo short-circuit, volvemos a pedirle al LLM que decida con
            # los resultados a la vista.
            previos.extend(results)
            sin_tools = any(c.outcome == REPEATED for c in turn.calls[antes:])

        # Agotadas las iteraciones. El historial acaba en un bloque tool sin
        # respuesta del asistente; si no cerramos con un assistant, el turno
        # siguiente arranca con una secuencia inconsistente (y algunos motores
        # rechazan un `tool` sin `assistant` que lo suceda).
        reply = "Me enredé en demasiados pasos, patrón. Mejor dímelo más simple."
        turn.via = VIA_EXHAUSTED
        if self.stop_on_repeated_failure:
            return self._exhausted_reply(turn, reply, emit, should_abort, clean_head)
        self.conversation.add_assistant(reply)
        if on_token is not None:
            on_token(reply)
        return reply

    def _exhausted_reply(self, turn: TurnInfo, canned: str, emit, should_abort,
                         clean_head) -> str:
        """Con la política de repetición, al agotar las iteraciones: UNA petición
        más, sin tools, para que el modelo cuente con sus palabras qué hizo y qué
        no, en vez de la frase enlatada. Si falla o no dice nada, la enlatada."""
        if should_abort is not None and should_abort():
            raise TurnCancelled()
        # Sin tools a la vista: se habla en vivo, como la charla.
        streamer = (
            _LiveStreamer(emit, hold_until_done=False, release=self.stream_release,
                          should_abort=should_abort, clean_head=clean_head)
            if emit is not None else None
        )
        turn.llm_calls += 1
        try:
            if streamer is not None:
                response = self.llm.chat_stream(self.conversation.to_messages(), tools=[],
                                                on_token=streamer)
            else:
                response = self.llm.chat(self.conversation.to_messages(), tools=[])
        except LLMError as error:
            log.warning("la última petición sin tools falló (%s); queda la frase de siempre",
                        error)
            response = None
        if response is not None and response.content:
            reply = self._apply(self.post_hooks, response.content)
            if streamer is not None:
                streamer.flush_if_held(reply)
            self.conversation.add_assistant(reply)
            return reply
        self.conversation.add_assistant(canned)
        if emit is not None:
            # Si ya se habló media frase en vivo, la enlatada va en línea aparte.
            emit(f"\n{canned}" if emit.released else canned)
        return canned
