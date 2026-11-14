"""La descripción de UN turno del agente: qué pidió el patrón y qué pasó en él.

POR QUÉ: ToolAgent era una tubería fija (pre-hooks, fast-path, LLM y tools,
after-turn hooks) y los hooks solo recibían strings. Los únicos con coste de
verdad (la búsqueda y la extracción de la memoria semántica) no podían
distinguir "pausa la música" de "me gusta el café de olla", y nada guardaba qué
tool falló con qué argumentos, así que repetir un fallo se veía igual que
avanzar. TurnInfo es ese registro: lo arma el agente al empezar el turno, lo
llena en cada salida y se lo pasa a los hooks que lo piden (`wants_turn`).
"""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

# Resultado de una llamada a tool (CallRecord.outcome).
OK = "ok"
UNKNOWN_TOOL = "unknown_tool"   # el modelo pidió una tool que no existe
BLOCKED = "blocked"             # el guard no la dejó correr
DECLINED = "declined"           # el patrón no la confirmó
HARD_ERROR = "hard_error"       # reventó o argumentos inválidos (HARD_ERROR_PREFIXES)
SOFT_ERROR = "soft_error"       # un server MCP avisó de un fallo (SOFT_ERROR_PREFIXES)
# El modelo repitió una llamada que ya había fallado y no se volvió a correr
# ([llm].stop_on_repeated_failure): su resultado es el del fallo anterior.
REPEATED = "repeated"

# Por dónde salió el turno (TurnInfo.via).
VIA_FASTPATH = "fastpath"       # atajo regex sin LLM
VIA_DIRECT = "direct"           # short-circuit: el output de la tool tal cual
VIA_LLM = "llm"                 # el modelo redactó la respuesta final
VIA_EXHAUSTED = "exhausted"     # se agotaron las iteraciones
VIA_LLM_ERROR = "llm_error"     # el motor falló y su error es la respuesta
VIA_CANCELLED = "cancelled"     # el patrón interrumpió (should_abort)


def call_key(name: str, args: dict[str, Any]) -> str:
    """Clave estable de una llamada: el mismo pedido da la misma clave aunque el
    modelo escriba los argumentos en otro orden. Lo que no es JSON va como str."""
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class CallRecord:
    """Una tool que corrió en el turno: qué se pidió y cómo acabó."""

    name: str
    key: str
    outcome: str
    result: str

    @property
    def failed(self) -> bool:
        return self.outcome != OK


@dataclass
class TurnInfo:
    """El turno en curso. `command` es el (tool, args) que reconoce
    fastpath.match sobre el texto LIMPIO del patrón: el clasificador de comandos
    anclados (la orden completa, ^...$), que se calcula aunque el fast-path esté
    apagado (con GLM de cerebro) porque cuesta microsegundos."""

    text: str
    command: tuple[str, dict[str, Any]] | None
    via: str = VIA_LLM
    llm_calls: int = 0
    calls: list[CallRecord] = field(default_factory=list)

    def failures(self, key: str) -> int:
        """Cuántas veces falló en este turno la llamada con esa clave."""
        return sum(1 for record in self.calls if record.key == key and record.failed)

    def last_failure(self, key: str) -> CallRecord | None:
        """El último registro de esa clave si fue un fallo; None si no corrió o
        si la última vez salió bien."""
        for record in reversed(self.calls):
            if record.key == key:
                return record if record.failed else None
        return None


_Hook = TypeVar("_Hook", bound=Callable[..., Any])


def wants_turn(hook: _Hook) -> _Hook:
    """Marca un hook que quiere el TurnInfo: el agente lo llama con `turn=`
    (pre-hook: hook(text, turn=...); after-turn: hook(text, reply, turn=...)).
    Sin la marca, el hook se llama exactamente como siempre."""
    hook.wants_turn = True  # type: ignore[attr-defined]
    return hook


def hook_wants_turn(hook: Any) -> bool:
    # `is True`: un doble que contesta a cualquier atributo (un Mock) no cuenta.
    return getattr(hook, "wants_turn", False) is True


_T = TypeVar("_T")
_Trabajo = tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]


class DaemonPool:
    """Hilos DAEMON con dueño y tope que completan Futures de concurrent.futures.

    Los usan los pasos opcionales del turno que corren fuera del hilo del agente:
    las tools de solo lectura de un lote (ToolAgent) y la búsqueda de la memoria
    semántica (Memoria). POR QUÉ no ThreadPoolExecutor: sus hilos no son daemon y
    concurrent.futures los espera al salir del intérprete aunque se haya hecho
    shutdown(wait=False). Lo que corre en ellos no tiene cómo cortarse: con un
    lote a medias, Ctrl+C en el shell o la salida del listener (VoiceLoop.run
    vuelve cuando muere un hilo del loop) esperaban a la tool más lenta, hasta su
    timeout (read_page 12 s, un carril MCP 2x20 s), y una búsqueda de mem0
    colgada no dejaba salir nunca, con cerrar() o sin él. En serie, esas mismas
    tools corrían en el hilo del agente (el principal en el shell, el Brain
    daemon en voz) y la salida no las esperaba: aquí tampoco.

    Lo que se usaba del executor sigue igual: submit() da un Future (result con
    timeout; cancel() de lo que aún no empezó hace que ya no corra), los hilos se
    crean cuando no hay uno ocioso, hasta `max_workers`, y se reusan; shutdown()
    no espera (lo ya encolado corre, salvo cancel_futures=True) y un submit()
    después lanza RuntimeError.
    """

    def __init__(self, max_workers: int, name: str) -> None:
        self._max = max_workers
        self._name = name
        self._cola: queue.SimpleQueue[_Trabajo | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        # Hilos que acabaron su trabajo y esperan otro (como el semáforo de
        # ociosos del executor): con uno libre no se crea otro.
        self._ociosos = 0
        self._cerrado = False

    def submit(self, fn: Callable[..., _T], /, *args: Any) -> Future[_T]:
        future: Future[_T] = Future()
        with self._lock:
            if self._cerrado:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._cola.put((future, fn, args))
            if self._ociosos > 0:
                self._ociosos -= 1
                return future
            if len(self._threads) >= self._max:
                return future
            hilo = threading.Thread(target=self._trabajar, daemon=True,
                                    name=f"{self._name}_{len(self._threads)}")
            try:
                hilo.start()
            except RuntimeError:
                # "can't start new thread": quien llama lo corre él mismo; si
                # quedara vivo en la cola, otro hilo lo correría DESPUÉS otra vez.
                future.cancel()
                raise
            self._threads.add(hilo)
        return future

    def _trabajar(self) -> None:
        while True:
            trabajo = self._cola.get()
            if trabajo is None:
                return
            future, fn, args = trabajo
            resultado: Any = None
            # False = lo cancelaron mientras esperaba: ya no corre.
            if future.set_running_or_notify_cancel():
                try:
                    resultado = fn(*args)
                except BaseException as error:  # noqa: BLE001 - al Future: lo ve quien espera
                    future.set_exception(error)
                else:
                    future.set_result(resultado)
            # Sin referencias al trabajo (ni a su resultado) mientras espera otro.
            del trabajo, future, fn, args, resultado
            with self._lock:
                self._ociosos += 1

    def shutdown(self, cancel_futures: bool = False) -> None:
        """Sin esperar: los ociosos salen ya y el que está a media tarea la acaba
        y sale. Idempotente. cancel_futures=True: lo que seguía en la cola no
        corre (sus Futures quedan cancelados)."""
        with self._lock:
            # Un None por hilo, detrás de lo encolado: cada hilo sale al tomar el suyo.
            centinelas = 0 if self._cerrado else len(self._threads)
            self._cerrado = True
            if cancel_futures:
                while True:
                    try:
                        trabajo = self._cola.get_nowait()
                    except queue.Empty:
                        break
                    if trabajo is None:
                        centinelas += 1     # de un shutdown anterior: vuelve a la cola
                    else:
                        trabajo[0].cancel()
            for _ in range(centinelas):
                self._cola.put(None)
