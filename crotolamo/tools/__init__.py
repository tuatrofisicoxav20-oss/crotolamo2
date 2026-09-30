"""Registry de tools. Importar este paquete registra las tools disponibles."""

from __future__ import annotations

from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import GLOBAL_REGISTRY, Registry, Tool, tool

log = get_logger("tools")


def _register_mcp_if_enabled() -> None:
    """M4: suma las tools de los servers MCP de `[mcp]` al registry global.

    Solo si `[mcp].enabled = true`; y pase lo que pase (config rota, server que
    no arranca, bug del puente) aquí se loguea y se sigue: el arranque de
    Crotolamo no depende de ningún proceso externo. El bridge es idempotente,
    así que llamar default_registry() varias veces no relanza nada.
    """
    try:
        from crotolamo.settings import get_settings

        settings = get_settings()
        if not settings.mcp.get("enabled", False):
            return
        from crotolamo.mcp.bridge import register_mcp_tools

        register_mcp_tools(GLOBAL_REGISTRY, settings)
    except Exception as error:  # noqa: BLE001 — MCP es opcional; nunca tumba el arranque
        log.warning("no pude registrar las tools MCP: %s", error)


def default_registry() -> Registry:
    """Importa los módulos de tools (lo que las registra) y devuelve el registry."""
    # Los imports tienen efecto colateral: cada @tool se registra al importarse.
    from crotolamo.tools import (  # noqa: F401
        cameras,
        desktop,
        facts,
        files,
        home,
        media,
        projects,
        search,
        shortcuts,
        system,
        windows,
    )

    _register_mcp_if_enabled()
    return GLOBAL_REGISTRY


def build_registry() -> Registry:
    """Registry NUEVO y aislado con todas las tools (m4). Mutarlo no afecta al GLOBAL;
    útil para tests o usos aislados.
    """
    return default_registry().copy()


__all__ = [
    "GLOBAL_REGISTRY", "Registry", "Tool", "tool", "default_registry", "build_registry",
]
