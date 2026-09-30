"""Validación de rutas contra la allowlist. Reutilizable por el guard (en el agente)
y por las propias tools de archivo (defensa en profundidad, M2).
"""

from __future__ import annotations

from pathlib import Path


def path_inside_roots(candidate, roots) -> bool:
    """True si `candidate`, ya resuelto (sigue symlinks y ../), cae dentro de alguna
    de las raíces dadas. Resolver primero neutraliza el path-traversal.
    """
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        # ValueError: un '\0' en la ruta ("embedded null byte"). La ruta la
        # elige el LLM; sin capturarlo la excepción subía hasta el agente en
        # vez de negar. Lo que no se puede resolver, no está en el corral.
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return True
        except (ValueError, OSError, RuntimeError):
            continue
    return False


def path_inside_allowed_roots(candidate, allowed_roots) -> bool:
    """Alias histórico de path_inside_roots (API original de M2)."""
    return path_inside_roots(candidate, allowed_roots)
