"""Selección del motor de inferencia: Ollama (local) o GLM (nube).

Un único punto donde se decide con qué LLM habla Crotolamo. Ambos clientes
exponen el mismo contrato (chat, chat_stream -> ChatResponse), así que el resto
del código —`ToolAgent`, `Conversation`, las tools— no distingue cuál corre.

Se elige con [llm].backend en la config, o con la env CROTOLAMO_LLM_BACKEND
(que manda sobre el toml, útil para probar sin editar archivos).
"""

from __future__ import annotations

import os

from crotolamo.logging_setup import get_logger

log = get_logger("core.engine")

OLLAMA = "ollama"
GLM = "glm"
VALID_BACKENDS = (OLLAMA, GLM)


def resolve_backend(settings) -> str:
    """Nombre del backend a usar. La env pisa el toml; un valor raro cae a ollama."""
    raw = os.environ.get("CROTOLAMO_LLM_BACKEND") or settings.llm.get("backend", OLLAMA)
    backend = str(raw).strip().lower()
    if backend not in VALID_BACKENDS:
        log.warning("backend '%s' desconocido; uso '%s'", backend, OLLAMA)
        return OLLAMA
    return backend


def build_llm(settings):
    """Construye el cliente del backend configurado.

    Con GLM, el cliente va envuelto en `FallbackLLM`: si la nube falla EN CALIENTE
    (se cayó el wifi, la key caducó, un 429), los turnos siguientes los atiende
    Ollama local. Sin eso, un corte de internet dejaba a Crotolamo respondiendo
    "¿Hay internet, patrón?" en vez de usar el modelo que ya está instalado.

    Si se pide GLM y no hay API key, ni siquiera se intenta: se usa Ollama directo.
    Más vale un Crotolamo lento que un Crotolamo mudo.
    """
    backend = resolve_backend(settings)

    from crotolamo.core.llm import LLMClient

    local = LLMClient.from_settings(settings)

    if backend == GLM:
        from crotolamo.core.glm import GLMClient, _find_api_key

        if _find_api_key() is None:
            log.warning(
                "backend=glm pero no hay API key (CROTOLAMO_GLM_API_KEY). "
                "Caigo a Ollama local."
            )
        else:
            from crotolamo.core.fallback import FallbackLLM

            remote = GLMClient.from_settings(settings)
            cooldown = settings.llm.get("glm", {}).get("fallback_cooldown_s", 60.0)
            log.info(
                "motor: GLM (%s) vía %s, con respaldo local (%s)",
                remote.model, remote.base_url, local.model,
            )
            return FallbackLLM(remote, local, cooldown_s=cooldown)

    log.info("motor: Ollama (%s) en %s", local.model, local.host)
    return local
