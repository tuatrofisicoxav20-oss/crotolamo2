"""Motor con respaldo: GLM (nube) y, si se cae, Ollama (local).

POR QUÉ: `engine.build_llm()` decide el motor UNA vez, al arrancar, mirando si hay
API key. Pero un asistente de voz corre durante horas y el wifi se cae a media
sesión. Sin esto, cada frase respondía "¿Hay internet, patrón?" en vez de usar el
modelo local que ya está instalado.

DISEÑO: envuelve dos clientes con el mismo contrato y expone ese mismo contrato,
así que `ToolAgent` no se entera. Dos decisiones que no son obvias:

1. **Circuit breaker.** Tras un fallo del primario, no reintentamos en cada turno:
   iríamos a pagar el timeout una y otra vez. Se marca como caído y se usa el
   respaldo durante `cooldown_s`, luego se vuelve a probar.

2. **No reintentar si ya se habló.** En voz, `chat_stream` va soltando tokens al
   TTS. Si el primario revienta A MEDIA frase, reintentar con el respaldo haría que
   Crotolamo repitiera el principio. Si ya se emitió un token, el error se propaga.
   "Emitido" es lo que SALIÓ: un token que el consumidor retuvo (on_token devolvió
   False, ver `LiveStreamer.feed`) no sonó, y ahí sí se usa el respaldo. Antes de
   repetir la llamada se le pide al consumidor `restart()` (si lo tiene): lo que
   retuvo de la nube no puede salir pegado a la respuesta del local.

El punto de control del consumidor (`checkpoint`, ver `LiveStreamer`) sigue a la
vista del primario a través del envoltorio, y se consulta antes de arrancar el
local: un turno cancelado (barge-in) no paga el prompt-eval en CPU para nadie.

Cuando el primario cae DENTRO de una llamada, la petición venía armada para él
(con GLM: todas las tools). `shape_for_secondary` la adapta al respaldo antes de
repetirla, y el cooldown se cuenta desde que el respaldo YA respondió: su carga
en frío (~60-120 s) duraba más que el cooldown y la siguiente llamada del mismo
turno volvía a probar la nube caída.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from crotolamo.core.glm import GLMAuthError
from crotolamo.core.llm import ChatResponse, TransientLLMError
from crotolamo.logging_setup import get_logger

log = get_logger("core.fallback")

Messages = list[dict[str, Any]]
Tools = list[dict[str, Any]] | None


class FallbackLLM:
    """Intenta `primary`; ante un fallo TRANSITORIO usa `secondary`.

    Solo los errores de disponibilidad (TransientLLMError: timeout, conexión,
    5xx, 429) abren el breaker. Un error PERMANENTE (400 por payload roto,
    no-JSON) se propaga con su mensaje en personaje: degradar 60s en silencio
    solo escondería el bug. Una credencial rechazada (GLMAuthError) tampoco se
    arregla sola: la nube se marca caída hasta reinicio (sin bucle de 401s) y
    se usa el local, avisando una vez en el log.
    """

    def __init__(self, primary, secondary, cooldown_s: float = 60.0) -> None:
        self.primary = primary
        self.secondary = secondary
        self.cooldown_s = cooldown_s
        # monotonic del instante en que el primario vuelve a estar disponible.
        # 0.0 = disponible ya. Se usa monotonic (no wall clock) para que un ajuste
        # de la hora del sistema no deje el breaker abierto para siempre.
        self._retry_at: float = 0.0
        # Auth rechazada EN CALIENTE: caído hasta reinicio (el cooldown no aplica;
        # la key no va a revivir sola). El caso "sin key al arrancar" ya lo maneja
        # engine.build_llm, que ni siquiera envuelve en FallbackLLM.
        self._auth_down: bool = False
        # Adapta (messages, tools) al respaldo cuando el primario cae A MEDIA
        # llamada: esa petición iba armada para la nube y el modelo local en CPU
        # necesita su dieta chica de tools. Lo instala build_agent. None = se
        # reenvía tal cual (lo de siempre).
        self.shape_for_secondary: Callable[[Messages, Tools], tuple[Messages, Tools]] | None = None
        # Qué motor respondió la última llamada de CADA hilo (ver answered_by_secondary).
        self._local = threading.local()

    @property
    def answered_by_secondary(self) -> bool:
        """True si la última llamada de ESTE hilo la respondió el respaldo local.

        No es lo mismo que `degraded` mirado después: una llamada con el breaker
        ya abierto no lo re-arma, así que si el cooldown vence mientras el local
        responde, al volver `degraded` ya es False y la respuesta era del local.
        Lo lee el agente para decidir el short-circuit con la política de quien
        respondió. Por hilo, como la conexión del transporte.
        """
        return bool(getattr(self._local, "secondary", False))

    # `model` es parte del contrato `LLMEngine`: los clientes concretos lo traen
    # como atributo y aquí se delega en el que respondería ahora mismo. Hoy solo
    # lo consultan los tests; el log de arranque lee el de cada cliente y el
    # doctor saca el nombre de la config.
    @property
    def model(self) -> str:
        return self.active.model

    @property
    def active(self):
        """El cliente que se usaría ahora mismo."""
        return self.secondary if self._primary_down() else self.primary

    @property
    def degraded(self) -> bool:
        """True si ahora mismo respondería el respaldo local (breaker abierto).

        Lo consulta el armado del agente para decidir la dieta de tools: con GLM
        (nube) se mandan TODAS las tools (razona él); degradado a Ollama local se
        vuelve al tool-routing chico para que el turno no cueste ~130s en CPU.
        """
        return self._primary_down()

    def _primary_down(self) -> bool:
        return self._auth_down or self._retry_at > time.monotonic()

    def _trip_auth(self, error: Exception) -> None:
        self._auth_down = True
        log.warning(
            "el motor principal rechazó la credencial (%s); uso el local hasta "
            "reiniciar. Revisa la API key, patrón.", error,
        )

    def _trip(self, error: Exception) -> None:
        self._retry_at = time.monotonic() + self.cooldown_s
        log.warning(
            "el motor principal falló (%s); uso el local %ss",
            error, int(self.cooldown_s),
        )

    def _recover(self) -> None:
        if self._retry_at:
            log.info("el motor principal volvió")
        self._retry_at = 0.0

    def _for_secondary(self, messages: Messages, tools: Tools) -> tuple[Messages, Tools]:
        """(messages, tools) para el respaldo tras un fallo del primario EN esta
        llamada. Sin tools (el resumen de compaction, el Agent base) no hay nada
        que adaptar; un shaper roto no puede dejar al patrón sin respuesta."""
        if self.shape_for_secondary is None or not tools:
            return messages, tools
        try:
            return self.shape_for_secondary(messages, tools)
        except Exception:  # noqa: BLE001 - se usa la petición original
            log.exception("no pude adaptar la petición al motor local; va tal cual")
            return messages, tools

    def _rearm(self) -> None:
        """El cooldown cuenta desde que el respaldo YA respondió, no desde el fallo.

        Si la primera respuesta local tarda más que cooldown_s (carga en frío del
        modelo), el breaker se re-armaba antes de acabar el propio turno y la
        siguiente llamada volvía a pagar la sonda de la nube caída. Las llamadas
        con el breaker ya abierto no lo alargan: en charla continua la nube se
        sigue probando cada cooldown_s. Una auth rechazada no usa el cooldown.
        """
        if self._retry_at:
            self._retry_at = max(self._retry_at, time.monotonic() + self.cooldown_s)

    def preconnect(self) -> bool:
        """Abre por adelantado la conexión del motor que respondería ahora.

        Nunca lanza: es una optimización, y sin soporte (o sin red) el turno
        conecta como siempre. Devuelve True si quedó una conexión lista.
        """
        preconnect = getattr(self.active, "preconnect", None)
        if not callable(preconnect):
            return False
        try:
            return bool(preconnect())
        except Exception as error:  # noqa: BLE001 - nunca tumba a quien la llama
            log.debug("preconnect falló: %s", error)
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        if not self._primary_down():
            try:
                response = self.primary.chat(messages, tools=tools)
            except TransientLLMError as error:
                self._trip(error)
            except GLMAuthError as error:
                self._trip_auth(error)
            # Cualquier otro LLMError (400, no-JSON) se propaga: es un bug, no
            # una caída, y el patrón debe verlo en vez de un Ollama lento.
            else:
                self._recover()
                self._local.secondary = False
                return response
            messages, tools = self._for_secondary(messages, tools)
            self._local.secondary = True
            try:
                return self.secondary.chat(messages, tools=tools)
            finally:
                self._rearm()
        self._local.secondary = True
        return self.secondary.chat(messages, tools=tools)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token=None,
    ) -> ChatResponse:
        checkpoint = getattr(on_token, "checkpoint", None)
        cayo_aqui = False  # el primario se intentó EN esta llamada y falló
        if not self._primary_down():
            hablo = False

            def _marcar(chunk: str) -> None:
                nonlocal hablo
                # "Habló" = algo SALIÓ hacia el patrón, no "se llamó a on_token".
                # Quien retiene el trozo lo avisa devolviendo False (LiveStreamer
                # con tools a la vista: nada suena hasta acabar la llamada), y sin
                # on_token no escucha nadie. Un callback cualquiera devuelve None:
                # se asume que habló, como siempre.
                if on_token is not None and on_token(chunk) is not False:
                    hablo = True

            # Los parsers buscan el punto de control como atributo del on_token
            # que reciben (este envoltorio): sin él, un tool-call o un
            # razonamiento del primario no se podían cancelar a medio stream.
            _marcar.checkpoint = checkpoint  # type: ignore[attr-defined]

            try:
                response = self.primary.chat_stream(messages, tools=tools, on_token=_marcar)
            except TransientLLMError as error:
                self._trip(error)
                if hablo:
                    # Ya salió audio/texto por on_token. Reintentar duplicaría el
                    # principio de la frase; mejor fallar este turno.
                    raise
            except GLMAuthError as error:
                self._trip_auth(error)
                if hablo:
                    raise
            else:
                self._recover()
                self._local.secondary = False
                return response
            # No salió nada, pero el consumidor puede tener retenido el
            # principio que soltó la nube (LiveStreamer con "sentence" lo
            # soltaría al cerrar la frase, pegado a la respuesta del local).
            restart = getattr(on_token, "restart", None)
            if callable(restart):
                restart()
            cayo_aqui = True
        if checkpoint is not None:
            # Cancelado mientras el primario fallaba: el local no arranca. El
            # fallo del primario ya contó para el breaker (fue real).
            checkpoint()
        self._local.secondary = True
        if not cayo_aqui:
            return self.secondary.chat_stream(messages, tools=tools, on_token=on_token)
        messages, tools = self._for_secondary(messages, tools)
        try:
            return self.secondary.chat_stream(messages, tools=tools, on_token=on_token)
        finally:
            self._rearm()
