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


def build_registry() -> Registry:
    """Registry NUEVO y aislado con todas las tools (m4). Mutarlo no afecta al GLOBAL;
    útil para tests o usos aislados.
    """
    return default_registry().copy()


__all__ = [
    "GLOBAL_REGISTRY", "Registry", "Tool", "tool", "default_registry", "build_registry",
]
