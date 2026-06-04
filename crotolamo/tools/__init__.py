"""Registry de tools. Importar este paquete registra las tools disponibles."""

from __future__ import annotations

from crotolamo.tools.base import GLOBAL_REGISTRY, Registry, Tool, tool


def default_registry() -> Registry:
    """Importa los módulos de tools (lo que las registra) y devuelve el registry."""
    # Los imports tienen efecto colateral: cada @tool se registra al importarse.
    from crotolamo.tools import (  # noqa: F401
        desktop,
        facts,
        files,
        projects,
        search,
        shortcuts,
        system,
    )

    return GLOBAL_REGISTRY


__all__ = ["GLOBAL_REGISTRY", "Registry", "Tool", "tool", "default_registry"]
