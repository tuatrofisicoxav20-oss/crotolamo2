"""Streaming de tokens hacia el patrón, con retención de tool-calls.

`LiveStreamer` es el DUEÑO de la respuesta mientras está en vuelo: sabe qué se
ha soltado ya hacia el patrón (`released`), decide cuándo soltar y con qué
evidencia, se deja reiniciar si otro motor repite la llamada (`restart`) y es
el punto por donde un turno se cancela a mitad de stream (`should_abort`).
Antes era un callable suelto y cada capa (FallbackLLM, el agente) llevaba su
propia idea de "ya se habló".
"""

from __future__ import annotations

import re
from typing import Callable

from crotolamo.logging_setup import get_logger

log = get_logger("core.streaming")

# Políticas de liberación cuando a la llamada se le enviaron tools.
RELEASE_FINAL = "final"        # retener hasta saber que no hubo tool-call (lo de siempre)
RELEASE_SENTENCE = "sentence"  # soltar al cerrarse la 1ª frase si le sigue más texto
RELEASE_POLICIES = (RELEASE_FINAL, RELEASE_SENTENCE)


# Signos tras los que la segmentación del TTS puede cerrar frase (el lookbehind
# de `_CANDIDATO` en voice/tts.py; un test lo vigila). Sin ninguno en el texto no
# hay frase cerrada y la sonda ni siquiera busca candidatos.
_CIERRES = frozenset(".!?¿¡\n")
# Algo que no es espacio: lo mismo que mira `str.strip()`, sin copiar el trozo.
_TEXTO = re.compile(r"\S")
# Lo que `_CANDIDATO` admite entre el signo de cierre y el espacio (cierres de
# énfasis markdown); un test lo vigila.
_ENFASIS = "*_~"


class TurnCancelled(Exception):
    """El turno se invalidó (barge-in) mientras el LLM generaba o antes de una tool.

    NO es un LLMError a propósito: FallbackLLM no debe abrir el breaker ni
    reintentar con el local un turno que el patrón canceló, y el agente no debe
    contestarlo como si fuera un fallo del motor.
    """


class Emitter:
    """El on_token del caller con memoria: `released` = ya salió texto en este turno.

    Uno por TURNO (el streamer es uno por llamada al LLM): lo consulta el agente
    para no pegar un aviso de error a una frase que ya sonó a medias.
    """

    def __init__(self, on_token: Callable[[str], None]) -> None:
        self._on_token = on_token
        self.released = False

    def __call__(self, text: str) -> None:
        if text:
            self.released = True
        self._on_token(text)


class LiveStreamer:
    """Streamea tokens al patrón en vivo, salvo que la respuesta empiece como un
    tool-call JSON ('{', '[' o cerca de código) — en ese caso retiene, para no
    filtrar el JSON crudo de los tool-calls que qwen emite en `content`.

    `hold_until_done=True` retiene SIEMPRE. Se usa cuando a esta llamada se le
    enviaron tools: el modelo puede verbalizar su intención ANTES de pedir la tool
    ("Voy a pausar la música por ti") y en voz eso se hablaría, seguido del
    resultado real. Medido contra GLM: en streaming emite ese preámbulo; en
    no-streaming, `content` viene vacío. Si no se enviaron tools, lo que salga ES
    la respuesta final y se habla en vivo, que es el objetivo de `stream_speak`.

    `release="sentence"` (opt-in, decisión del dueño) relaja esa retención con
    evidencia del propio stream: suelta en cuanto la 1ª frase está cerrada y le
    sigue más texto. Un preámbulo de UNA frase seguido del tool-call nunca sale;
    uno de dos o más, sí (no hay señal en el stream que lo descarte antes del
    final: los deltas de tool_calls llegan DESPUÉS del texto). `clean_head`
    limpia lo acumulado antes de soltarlo (el limpiador de preámbulos meta),
    porque los post-hooks solo alcanzan al texto que se suelta al final.

    Con la política por defecto, `would_release_early` dice si "sentence"
    habría soltado algo antes del fin del stream, sin soltar nada: el agente lo
    loguea para medir con el motor real cuánto preámbulo se colaría.

    `should_abort`: si devuelve True en un punto de control (`checkpoint`: cada
    token en `feed` y cada línea del stream en los parsers de los motores),
    lanza TurnCancelled; el cliente cuelga el socket y el motor deja de generar.
    """

    def __init__(self, on_token: Callable[[str], None],
                 hold_until_done: bool = False, *,
                 release: str = RELEASE_FINAL,
                 should_abort: Callable[[], bool] | None = None,
                 clean_head: Callable[[str], str] | None = None) -> None:
        self._on_token = on_token
        self._should_abort = should_abort
        self._clean_head = clean_head
        self._buf: list[str] = []
        # Estado al empezar (y al que vuelve `restart`): None = sin decidir,
        # "hold" = retener hasta el final, "probe" = retener hasta la evidencia
        # de "sentence". Durante el stream pasa a "stream" (en vivo) o "hold".
        self._initial: str | None = None
        if hold_until_done:
            self._initial = "probe" if release == RELEASE_SENTENCE else "hold"
        self._decision: str | None = self._initial
        self.released = False
        self.would_release_early = False
        self._reset_probe()

    def __call__(self, chunk: str) -> bool:
        return self.feed(chunk)

    def checkpoint(self) -> None:
        """Punto de control del turno: TurnCancelled si `should_abort` lo pide.

        Lo llama `feed` con cada token y los parsers de los motores (sse.py,
        LLMClient) con CADA línea del stream, que lo buscan como atributo del
        on_token: un tool-call (deltas de `arguments`), un razonamiento o un
        chunk vacío no traen texto y no pasan por `feed`, así que un turno de
        orden no se podía cortar hasta que acababa el stream. Sin should_abort
        no hace nada.
        """
        if self._should_abort is not None and self._should_abort():
            raise TurnCancelled()

    def _emit(self, text: str) -> None:
        if text:
            self.released = True
        self._on_token(text)

    def restart(self) -> None:
        """Otro motor va a repetir ESTA llamada: lo acumulado del intento fallido
        ya no vale y no debe salir pegado a la respuesta nueva.

        Lo llama FallbackLLM antes de pasar al local, y solo si no salió nada
        (`feed` devolvió False en todo el intento): lo ya soltado no se recoge.
        """
        self._buf.clear()
        # Con "sentence", lo que la nube dejó a medias no puede decidir por el
        # local. Con la política por defecto la decisión tomada se queda (si
        # la nube empezó como JSON, el local se retiene y suena de una vez con
        # los post-hooks, como antes del paso 6): on_token recibe lo mismo.
        if self._initial == "probe":
            self._decision = self._initial
        self.would_release_early = False
        self._reset_probe()

    def _reset_probe(self) -> None:
        # El diagnóstico deja de mirar en cuanto sabe la respuesta (la regla es
        # monótona). `_looks_json` se decide una vez, con 2 caracteres, igual
        # que la decisión en vivo; `_may_close` = ya pasó algún signo de cierre.
        self._diagnosed = False
        self._looks_json: bool | None = None
        self._may_close = False
        # Estado de la segmentación incremental (ver `_first_sentence_followed`):
        # lo acumulado sin el espacio inicial, desde dónde buscar candidatos,
        # dónde empieza la frase en curso y si ya cerró alguna.
        self._head = ""
        self._scan = 0
        self._cut = 0
        self._closed = False

    def _probe(self, chunk: str) -> str:
        """La regla de "sentence" sobre lo acumulado: "json" (pinta a tool-call:
        retener siempre), "wait" (aún sin evidencia) o "release"."""
        # Soltar la referencia del atributo antes del `+=` deja a CPython
        # ampliar la cadena en sitio; `self._head + chunk` la copiaba entera
        # en cada token (cuadrático en una respuesta larga).
        head, self._head = self._head, ""
        head += chunk if head else chunk.lstrip()
        self._head = head
        if not self._may_close and not _CIERRES.isdisjoint(chunk):
            self._may_close = True
        if self._looks_json is None:
            if len(self._head) < 2:
                return "wait"
            self._looks_json = self._head[0] in "{[" or self._head.startswith("```")
        if self._looks_json:
            return "json"
        # Sin signo de cierre no hay frase cerrada, y un trozo en blanco no
        # puede volver verdadera la regla (hace falta texto tras la frase).
        if not self._may_close or not chunk.strip():
            return "wait"
        return "release" if self._first_sentence_followed() else "wait"

    def _first_sentence_followed(self) -> bool:
        """La regla de "sentence": la 1ª frase ya cerró y le sigue más texto.

        Es `_split(head, final=False)` de voice/tts.py (los mismos candidatos y
        el mismo `_cierra_frase`, para que "frase cerrada" signifique aquí lo
        mismo que en lo que se habla) hecho INCREMENTAL: re-segmentar todo lo
        acumulado en cada token era cuadrático (0,4 s de CPU para una lista
        numerada de 9 KB, en cada turno con tools a la vista). Un candidato
        seguido ya de algo que no es espacio está decidido para siempre:
        `_cierra_frase` solo mira hacia atrás y un carácter adelante. Así cada
        token solo escanea lo nuevo y la cola aún indecisa.
        """
        # Perezoso: core no carga el paquete de voz al importarse.
        from crotolamo.voice import tts

        head, end = self._head, len(self._head)
        closed, cut = self._closed, self._cut
        floor = None
        for m in tts._CANDIDATO.finditer(head, self._scan):
            start, stop = m.span()
            if stop >= end:
                # Toca el final: puede alargarse con más espacio y, si es un
                # punto, sin ver lo que sigue no se sabe si cierra (_split
                # hace `break`). Se evalúa ahora y se re-evalúa con lo que venga.
                floor = start
                if head[start - 1] == ".":
                    break
            if tts._cierra_frase(head, start, stop):
                # `_split` descarta las piezas en blanco, pero la 1ª nunca lo
                # está (`head` empieza sin espacio): cerrar una ya es "cerró".
                closed, cut = True, stop
            if floor is None:
                self._scan, self._closed, self._cut = stop, closed, cut
        if floor is None:
            # Sin candidato indeciso, uno nuevo solo puede empezar en la racha
            # final de énfasis ("Listo.**" espera su espacio).
            floor = end
            while floor > self._scan and head[floor - 1] in _ENFASIS:
                floor -= 1
        self._scan = floor
        return closed and _TEXTO.search(head, cut) is not None

    def feed(self, chunk: str) -> bool:
        """Recibe un delta. Devuelve False SOLO si quedó retenido (no salió ni
        saldrá por su cuenta: lo único que se emite luego es `flush_if_held`).

        Lo consulta `FallbackLLM` para su regla de "no reintentar si ya se habló":
        contar llamadas a on_token daba por hablado lo que aquí seguía retenido,
        y un corte de la nube a media respuesta ya no probaba el modelo local.
        Si reintenta, llama a `restart` para que lo retenido no se mezcle.
        """
        self.checkpoint()
        self._buf.append(chunk)
        if self._decision == "stream":
            self._emit(chunk)
            return True
        if self._decision == "hold":
            if self._initial == "hold" and not self._diagnosed:
                verdict = self._probe(chunk)
                if verdict != "wait":
                    self._diagnosed = True
                    self.would_release_early = verdict == "release"
            return False
        if self._decision == "probe":
            verdict = self._probe(chunk)
            if verdict == "json":
                self._decision = "hold"
            if verdict != "release":
                return False
            self.would_release_early = True
            self._decision = "stream"
            self._emit(self._cleaned(self._head))
            return True
        head = "".join(self._buf).lstrip()
        if len(head) < 2:
            # Aún no hay suficiente para decidir. Cuenta como hablado: sigue en
            # el buffer y saldría pegado al principio de lo que venga después.
            return True
        if head[0] in "{[" or head.startswith("```"):
            self._decision = "hold"
            return False
        self._decision = "stream"
        self._emit(head)  # soltamos lo acumulado de golpe y seguimos en vivo
        return True

    def _cleaned(self, head: str) -> str:
        """`clean_head` aplicado a lo acumulado, sin comerse el espacio final
        (el siguiente token saldría pegado: "Y" + "sobra" = "Ysobra"). Un
        limpiador roto no tumba el turno: se suelta el texto tal cual."""
        if self._clean_head is None:
            return head
        body = head.rstrip()
        try:
            cleaned = self._clean_head(body)
        except Exception as error:  # noqa: BLE001 - mismo trato que un post-hook roto
            log.warning("el limpiador del principio falló: %s", error)
            return head
        return cleaned + head[len(body):]

    def flush_if_held(self, final_text: str) -> None:
        """Si retuvimos pero resultó ser texto final, lo emitimos completo."""
        if self._decision != "stream":
            self._emit(final_text)
