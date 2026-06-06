"""Validación de rutas contra la allowlist. Reutilizable por el guard (en el agente)
y por las propias tools de archivo (defensa en profundidad, M2).
"""

from __future__ import annotations

from pathlib import Path


def path_inside_allowed_roots(candidate, allowed_roots) -> bool:
    """True si `candidate`, ya resuelto (sigue symlinks y ../), cae dentro de alguna
    de las raíces permitidas. Resolver primero neutraliza el path-traversal.
    """
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return True
        except (ValueError, OSError, RuntimeError):
            continue
    return False
