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
"""

from __future__ import annotations

import time
from typing import Any

from crotolamo.core.llm import ChatResponse, LLMError
from crotolamo.logging_setup import get_logger

log = get_logger("core.fallback")


class FallbackLLM:
    """Intenta `primary`; ante `LLMError` usa `secondary`."""

    def __init__(self, primary, secondary, cooldown_s: float = 60.0) -> None:
        self.primary = primary
        self.secondary = secondary
        self.cooldown_s = cooldown_s
        # monotonic del instante en que el primario vuelve a estar disponible.
        # 0.0 = disponible ya. Se usa monotonic (no wall clock) para que un ajuste
        # de la hora del sistema no deje el breaker abierto para siempre.
        self._retry_at: float = 0.0

    # `model` (parte del contrato `LLMEngine`) lo leen el logging y el doctor;
    # delegamos en el cliente activo en este momento.
    @property
    def model(self) -> str:
        return self.active.model

    @property
    def active(self):
        """El cliente que se usaría ahora mismo."""
        return self.secondary if self._primary_down() else self.primary

    def _primary_down(self) -> bool:
        return self._retry_at > time.monotonic()

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

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        if not self._primary_down():
            try:
                response = self.primary.chat(messages, tools=tools)
            except LLMError as error:
                self._trip(error)
            else:
                self._recover()
                return response
        return self.secondary.chat(messages, tools=tools)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token=None,
    ) -> ChatResponse:
        if not self._primary_down():
            hablo = False

            def _marcar(chunk: str) -> None:
                nonlocal hablo
                hablo = True
                if on_token is not None:
                    on_token(chunk)

            try:
                response = self.primary.chat_stream(messages, tools=tools, on_token=_marcar)
            except LLMError as error:
                self._trip(error)
                if hablo:
                    # Ya salió audio/texto por on_token. Reintentar duplicaría el
                    # principio de la frase; mejor fallar este turno.
                    raise
            else:
                self._recover()
                return response
        return self.secondary.chat_stream(messages, tools=tools, on_token=on_token)
