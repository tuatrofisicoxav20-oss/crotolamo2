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


def _register_memoria_if_enabled(registry: Registry) -> None:
    """Memoria semántica (mem0): con [memoria].enabled = true registra sus tools
    (recordar_de_mi / buscar_recuerdos / olvidar_recuerdo) y RETIRA las de hechos
    SQLite, para que el modelo vea una sola familia de "recordar/olvidar". Un
    fallo aquí se loguea y se sigue: la memoria es opcional.
    """
    try:
        from crotolamo.settings import get_settings

        if get_settings().memoria.get("enabled") is not True:
            return
        from crotolamo.core.memoria import mem0_instalado

        if not mem0_instalado():
            # Sin la extra instalada, retirar las tools SQLite dejaría al
            # asistente sin NINGUNA forma de recordar: se quedan las de siempre.
            log.warning("[memoria].enabled = true pero mem0 no está instalado "
                        "(pip install -e '.[memoria]'); sigo con los hechos SQLite")
            return
        from crotolamo.tools.memoria import TOOLS_SQLITE_SUSTITUIDAS, memoria_tools

        for t in memoria_tools():
            registry.register(t)
        for name in TOOLS_SQLITE_SUSTITUIDAS:
            registry.unregister(name)
    except Exception as error:  # noqa: BLE001 — la memoria es opcional
        log.warning("no pude registrar las tools de memoria semántica: %s", error)


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

    _register_memoria_if_enabled(GLOBAL_REGISTRY)
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
