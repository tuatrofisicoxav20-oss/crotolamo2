"""Memoria conversacional de corto plazo.

Esto es lo que C1 no tenía: contexto entre turnos. Una `Conversation` guarda
el historial con roles y lo recorta con una ventana deslizante para no reventar
el contexto del modelo.
"""

from __future__ import annotations

import atexit
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from crotolamo.core.budget import SUMMARY_TOOL_CHARS
from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import truncate_for_context

log = get_logger("core.memory")

# Lo que viaja de un resultado de tool viejo con stale_tool_turns: sus primeros
# stale_tool_chars caracteres y esta marca. Solo en el request; el historial
# (/historial, el resumen) guarda el original.
STALE_TOOL_MARK = "\n[resultado ya usado: {n} chars; vuelve a llamar la tool si lo necesitas]"


def _ahora() -> float:
    """El reloj de idle_reset_s. De pared, no monotonic: en Linux time.monotonic
    no avanza con la laptop suspendida y una noche de tapa cerrada no contaría
    como silencio. Se busca en cada llamada (los tests y el bench lo mueven)."""
    return time.time()


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


def _evicted_ends(history: list[Message], max_turns: int, trim_chunk: int = 1) -> list[int]:
    """Dónde acaba cada bloque que sobra cuando hay más de `max_turns` turnos de usuario.

    Del más viejo al más nuevo; un bloque es el mensaje de cabeza más su cadena
    de assistant/tool hasta el próximo user. La ÚNICA regla de recorte: la usan
    `_trim` y quien necesita saber qué seguirá a la vista tras el próximo turno.
    Al pasarse se baja a max_turns - trim_chunk + 1 turnos (nunca menos de uno):
    con trim_chunk=1, a max_turns, un bloque por turno como siempre.
    """
    users = sum(1 for m in history if m.role == "user")
    target = max_turns
    if users > max_turns and trim_chunk > 1:
        target = max(max_turns - trim_chunk + 1, min(max_turns, 1))
    ends: list[int] = []
    i = 0
    while users > target and i < len(history):
        if history[i].role == "user":
            users -= 1
        i += 1
        # Arrastramos su respuesta/tools encadenados hasta el próximo user.
        while i < len(history) and history[i].role in {"assistant", "tool"}:
            i += 1
        ends.append(i)
    return ends


def _render(block: list[Message], tool_chars: int | None = None) -> str:
    """Un bloque como texto para el resumidor: "rol: contenido" por línea."""
    return "\n".join(
        f"{m.role}: "
        + (truncate_for_context(m.content, tool_chars)
           if tool_chars is not None and m.role == "tool" else m.content)
        for m in block if m.content
    )


class Conversation:
    """Historial de la sesión con ventana deslizante por nº de turnos.

    M5 (de GLaDOS): con compaction=True y un summarizer, los turnos viejos no se
    tiran en seco; se RESUMEN en un mensaje anclado tras el system prompt.

    SEGURO ENTRE HILOS: todas las operaciones toman un RLock interno. Hoy en voz
    solo BrainThread la muta, pero ese invariante no estaba escrito ni defendido;
    si mañana un segundo consumidor (HUD, log, un panel) lee el historial mientras
    se escribe, sin lock sería un "list changed size during iteration" aleatorio.
    Es RLock (reentrante) porque los add_* llaman a _trim por dentro. El overhead
    con un solo hilo es despreciable.

    summarize_in_background=True saca el resumen del turno: el bloque que sale
    de la ventana queda PENDIENTE (el modelo lo sigue viendo tal cual, tras el
    resumen) y `compact_pending()` lo resume cuando alguien lo llame, fuera del
    lock (ver CompactionWorker). Con False, el resumen síncrono de siempre.

    Es la dueña de cómo se ARMA el request (to_messages), no solo de lo que se
    guarda. Lo que el modelo debe ver en cada turno sin ser historial va en un
    hueco anclado (set_pinned): los hechos del patrón ya no tienen que viajar
    dentro del primer mensaje guardado, que sale de la ventana. Los knobs del
    dueño ([memory] en config/crotolamo.toml), con sus valores por defecto, dejan
    el request byte a byte como antes:
    - trim_chunk: al pasarse de max_turns tira varios bloques de golpe y deja
      max_turns - trim_chunk + 1 turnos; el principio del historial se queda
      quieto unos turnos y el cache de prefijo del proveedor lo aprovecha.
    - stale_tool_turns: los resultados de tool de turnos más viejos que eso
      viajan recortados (ver STALE_TOOL_MARK); el historial guarda el original.
    - idle_reset_s: tras ese silencio, el próximo add_user empieza de cero
      (lo anclado se queda).
    """

    # Bloques pendientes de resumir como mucho. Lo normal es 0-1 (se resumen tras
    # cada turno); los que se juntan mientras el resumidor falla esperan al
    # siguiente intento, pero sin tope la ventana crecería sin límite si la caída
    # dura: pasado el tope se tiran los más viejos, como el resumen fallido de antes.
    MAX_PENDING_BLOCKS = 4

    def __init__(self, system_prompt: str, max_turns: int = 20,
                 compaction: bool = False, summarizer=None, *,
                 trim_chunk: int = 1, stale_tool_turns: int | None = None,
                 stale_tool_chars: int = 800, idle_reset_s: float | None = None,
                 summarize_in_background: bool = False,
                 clock: Callable[[], float] | None = None) -> None:
        self._system = Message("system", system_prompt)
        self._history: list[Message] = []
        self.max_turns = max_turns
        self.compaction = compaction
        self.summarizer = summarizer  # Callable[[str], str] | None
        self.trim_chunk = max(1, int(trim_chunk))
        self.stale_tool_turns = None if stale_tool_turns is None else max(0, int(stale_tool_turns))
        self.stale_tool_chars = max(0, int(stale_tool_chars))
        self.idle_reset_s = (float(idle_reset_s)
                             if idle_reset_s is not None and idle_reset_s > 0 else None)
        self.summarize_in_background = summarize_in_background
        self._summary: Message | None = None
        # Huecos anclados tras el system prompt, en orden de alta (set_pinned).
        self._pinned: dict[str, Message] = {}
        self._clock = clock or _ahora
        # Cuándo se guardó el último mensaje; solo se mira con idle_reset_s.
        self._last_activity: float | None = None
        # Bloques que salieron de la ventana y esperan al resumen en segundo plano.
        self._pending: list[list[Message]] = []
        # Sube con cada reset(): un resumen que empezó antes no se aplica después.
        self._epoch = 0
        self._lock = threading.RLock()

    # --- escritura ---
    def add_user(self, text: str) -> None:
        with self._lock:
            self._reset_if_idle()
            self._history.append(Message("user", text))
            self._trim()
            self._touch()

    def add_assistant(self, text: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        with self._lock:
            self._history.append(Message("assistant", text, tool_calls=tool_calls))
            self._trim()  # M5: compactar también tras assistant/tool (arregla m2 del audit)
            self._touch()

    def add_tool_result(self, name: str, content: str) -> None:
        with self._lock:
            self._history.append(Message("tool", content, name=name))
            self._trim()
            self._touch()

    def reset(self) -> None:
        """Vacía la conversación (historial, resumen, pendientes). Lo anclado se queda."""
        with self._lock:
            self._clear()

    def set_pinned(self, key: str, text: str | None) -> None:
        """Ancla (o, con None o "", quita) un mensaje de sistema tras el system prompt.

        Va en CADA request, antes del resumen y del historial, y no lo tocan ni
        el recorte de la ventana ni reset(): es para lo que el modelo debe saber
        todo el rato sin ser conversación (los hechos del patrón). Cambiar el
        texto de un hueco lo deja en su sitio. Mientras no cambie, sus bytes
        tampoco: no rompe el cache de prefijo. Solo el ajuste del request a una
        ventana que ni sin historia alcanza (budget.fit, el local con num_ctx
        chico) lo manda recortado, con sus líneas más nuevas; aquí sigue entero.
        """
        with self._lock:
            if text:
                self._pinned[key] = Message("system", text)
            else:
                self._pinned.pop(key, None)

    def _clear(self) -> None:
        self._history.clear()
        self._summary = None
        self._pending.clear()
        self._epoch += 1  # un resumen en curso ya no se aplica

    def _touch(self) -> None:
        if self.idle_reset_s:
            self._last_activity = self._clock()

    def _reset_if_idle(self) -> None:
        """Con idle_reset_s: si pasó más que eso desde el último mensaje guardado,
        la conversación empieza de cero (lo anclado se queda)."""
        if not self.idle_reset_s or self._last_activity is None:
            return
        silencio = self._clock() - self._last_activity
        if silencio <= self.idle_reset_s:
            return
        self._last_activity = None
        if self._history or self._summary is not None or self._pending:
            log.info("%.0f min sin hablar: la conversación empieza de cero "
                     "([memory].idle_reset_min)", silencio / 60)
            self._clear()

    # --- lectura ---
    def to_messages(self) -> list[dict[str, Any]]:
        """Payload para el LLM: [system, ...anclados, (resumen?), ...pendientes, ...historial].

        Con stale_tool_turns, los resultados de tool de los turnos de usuario más
        viejos que eso salen recortados AQUÍ, al armar el request: el historial
        (/historial, el resumen) sigue guardando el original.
        """
        with self._lock:
            msgs = [self._system.to_dict()]
            msgs += [m.to_dict() for m in self._pinned.values()]
            if self._summary is not None:
                msgs.append(self._summary.to_dict())
            rest = [m for block in self._pending for m in block] + self._history
            stale_until = self._stale_until(rest)
            return msgs + [
                self._stale_dict(m) if i < stale_until and m.role == "tool" else m.to_dict()
                for i, m in enumerate(rest)
            ]

    def _stale_until(self, rest: list[Message]) -> int:
        """Índice del user a partir del cual los resultados de tool viajan enteros:
        el del turno en curso y los stale_tool_turns anteriores. 0 = todos enteros."""
        if self.stale_tool_turns is None:
            return 0
        users = [i for i, m in enumerate(rest) if m.role == "user"]
        if len(users) <= self.stale_tool_turns:
            return 0
        return users[-(self.stale_tool_turns + 1)]

    def _stale_dict(self, message: Message) -> dict[str, Any]:
        msg = message.to_dict()
        if len(message.content) > self.stale_tool_chars:
            msg["content"] = (message.content[:self.stale_tool_chars]
                              + STALE_TOOL_MARK.format(n=len(message.content)))
        return msg

    @property
    def history(self) -> list[Message]:
        with self._lock:
            return list(self._history)

    def has_pending(self) -> bool:
        """True si hay bloques esperando al resumen en segundo plano."""
        with self._lock:
            return bool(self._pending)

    def surviving_user_texts(self) -> list[str]:
        """Los mensajes de usuario que seguirán en el historial tras el PRÓXIMO add_user.

        Para quien prepara ese mensaje (el pre-hook de la memoria semántica):
        lo que está en un bloque que ese add_user va a recortar ya no cuenta
        como "a la vista" del modelo. Con idle_reset_s, si ese add_user va a
        empezar de cero, el vaciado se hace YA: si no, el pre-hook daría por
        vistos recuerdos que el modelo no va a ver.
        """
        with self._lock:
            self._reset_if_idle()
            history = [*self._history, Message("user")]
            ends = _evicted_ends(history, self.max_turns, self.trim_chunk)
            start = ends[-1] if ends else 0
            return [m.content for m in history[start:-1] if m.role == "user"]

    def _trim(self) -> None:
        """Recorta a max_turns turnos de usuario, preservando bloques completos
        (con trim_chunk, de golpe a max_turns - trim_chunk + 1; ver _evicted_ends).

        Con compaction activa, el bloque viejo se RESUME en vez de descartarse.
        Siempre se llama con el lock ya tomado (desde los add_*; RLock).
        """
        with self._lock:
            start = 0
            for end in _evicted_ends(self._history, self.max_turns, self.trim_chunk):
                block = self._history[start:end]
                start = end
                if self.compaction and self.summarizer is not None:
                    if self.summarize_in_background:
                        self._pending.append(block)
                    else:
                        self._absorb_into_summary(block)
                # Sin compaction: el bloque simplemente se descarta (comportamiento viejo).
            del self._history[:start]
            # Con trim_chunk, un recorte deja trim_chunk bloques pendientes de golpe:
            # caben siempre enteros, más el margen de siempre para los fallos.
            sobran = len(self._pending) - (self.MAX_PENDING_BLOCKS + self.trim_chunk - 1)
            if sobran > 0:
                del self._pending[:sobran]
                log.warning("el resumen de la conversación no avanza; tiro %d bloque(s) "
                            "viejo(s) sin resumir", sobran)

    def compact_pending(self) -> bool:
        """Resume los bloques pendientes en UNA llamada y los cambia por el resumen.

        El resumidor corre FUERA del lock: el turno siguiente puede leer y
        escribir mientras tanto (lo que entre, queda para la próxima vez). Al
        terminar, el resumen nuevo y la retirada de lo resumido van juntos bajo
        el lock. Los resultados de tool van recortados a SUMMARY_TOOL_CHARS: el
        resumen guarda hechos y decisiones, no archivos enteros. Si el resumidor
        falla (o devuelve nada), los bloques siguen pendientes y a la vista; si
        hubo un reset() entretanto, el resultado se descarta. True si resumió.
        """
        with self._lock:
            if not self._pending or not self.compaction or self.summarizer is None:
                return False
            blocks = list(self._pending)
            prev = self._summary
            epoch = self._epoch
            summarizer = self.summarizer
        rendered = "\n".join(_render(b, SUMMARY_TOOL_CHARS) for b in blocks)
        try:
            summary_text = summarizer(f"{prev.content if prev else ''}\n{rendered}".strip())
        except Exception as error:  # noqa: BLE001 - los bloques esperan al siguiente intento
            log.warning("no pude resumir la conversación (%s); lo reintento tras el "
                        "próximo turno", error)
            return False
        if not summary_text or not summary_text.strip():
            log.warning("el resumen de la conversación salió vacío; lo reintento tras "
                        "el próximo turno")
            return False
        with self._lock:
            if self._epoch != epoch or self._summary is not prev:
                return False  # reset() o, en un uso concurrente, otro resumen ganó
            self._summary = Message(
                "system", f"[resumen de la conversación previa]\n{summary_text.strip()}"
            )
            # Por identidad: el tope pudo tirar alguno mientras se resumía.
            self._pending = [b for b in self._pending if not any(b is r for r in blocks)]
        return True

    def _absorb_into_summary(self, block: list["Message"]) -> None:
        """Funde un bloque de mensajes viejos en el resumen anclado (M5)."""
        prev = self._summary.content if self._summary is not None else ""
        rendered = _render(block)
        try:
            summary_text = self.summarizer(f"{prev}\n{rendered}".strip())
        except Exception:  # noqa: BLE001 - si el resumen falla, no perdemos el turno actual
            return
        if summary_text and summary_text.strip():
            self._summary = Message(
                "system", f"[resumen de la conversación previa]\n{summary_text.strip()}"
            )


# Lo más que el proceso espera AL SALIR a un resumen en curso (se descarta igual).
_ESPERA_AL_SALIR_S = 1.0


class CompactionWorker:
    """Dueño del hilo que corre `conversation.compact_pending()` después de hablar.

    Mismo patrón que la extracción de Memoria: el turno solo AVISA (after-turn
    hook) y UN hilo daemon persistente resume; entre resumen y resumen duerme en
    una Condition sin timeout (ni sondeo ni hilos nuevos). Persistente porque
    HTTPTransport guarda su conexión keep-alive por hilo: un hilo nuevo por
    resumen abría una conexión (con GLM, TCP+TLS) por resumen. Los avisos que
    llegan mientras resume se juntan en una vuelta más. close() lo despierta y
    lo espera con tope: al salir del proceso un resumen a medias no vale nada,
    la Conversation muere con él.
    """

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._again = False   # hay aviso sin atender
        self._busy = False    # resumiendo ahora
        self._closed = False
        self._atexit = False

    def schedule(self) -> bool:
        """Pide un resumen de lo pendiente. False si el worker ya está cerrado."""
        with self._cond:
            if self._closed:
                return False
            if self._thread is None:
                thread = threading.Thread(target=self._run, name="CompactionWorker",
                                          daemon=True)
                try:
                    thread.start()
                except RuntimeError as error:  # sin hilos libres: se reintenta en otro turno
                    log.warning("no pude arrancar el resumen en segundo plano: %s", error)
                    return False
                self._thread = thread
                if not self._atexit:
                    atexit.register(self.close, _ESPERA_AL_SALIR_S)
                    self._atexit = True
            self._again = True
            self._cond.notify_all()
            return True

    def after_turn(self, _usuario: str, _respuesta: str) -> None:
        """La firma de los after-turn hooks de ToolAgent. Sin nada pendiente (los
        primeros max_turns turnos, o ya resumido) no arranca ningún hilo."""
        if self._conversation.has_pending():
            self.schedule()

    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._again and not self._closed:
                        self._cond.wait()
                    if self._closed:
                        return
                    self._again = False
                    self._busy = True
                try:
                    self._conversation.compact_pending()
                except Exception:  # noqa: BLE001 - un bug no mata al worker ni al turno
                    log.exception("el resumen de la conversación en segundo plano reventó")
                finally:
                    with self._cond:
                        self._busy = False
                        self._cond.notify_all()
        finally:
            # Se va (close, o algo que ni el except de arriba atrapó): el próximo
            # schedule() arranca otro, y quien espere en flush() deja de esperar.
            with self._cond:
                self._thread = None
                self._busy = False
                self._cond.notify_all()

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Espera (con tope) a que no quede resumen en curso ni avisado. Tests y apagado."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while (self._thread is not None and self._thread is not threading.current_thread()
                   and (self._busy or (self._again and not self._closed))):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def close(self, timeout_s: float = 2.0) -> bool:
        """No arranca más resúmenes, despierta al hilo y lo espera, con tope."""
        with self._cond:
            self._closed = True
            thread = self._thread
            if self._atexit:
                atexit.unregister(self.close)
                self._atexit = False
            self._cond.notify_all()
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout_s)
        return not thread.is_alive()
