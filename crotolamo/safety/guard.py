"""Validación de seguridad por ALLOWLIST, no por blocklist de regex.

C1 dependía de buscar substrings peligrosos ("rm -rf"...) en bash crudo: un
sandbox de papel. C2 no ejecuta bash arbitrario; las tools son funciones. El
guard decide, por tool y por argumentos, si la acción:
  - corre directo (safe),
  - necesita confirmación del patrón,
  - o se bloquea (p.ej. una ruta fuera de las raíces permitidas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crotolamo.tools.base import Tool

# Nombres de argumentos que típicamente contienen rutas de archivo (Fase 3).
_PATH_ARG_NAMES = {"path", "ruta", "file", "archivo", "dest", "destino", "src", "origen", "dir"}


@dataclass
class Decision:
    allowed: bool
    needs_confirmation: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "Decision":
        return cls(True, False, "")

    @classmethod
    def confirm(cls, reason: str) -> "Decision":
        return cls(True, True, reason)

    @classmethod
    def block(cls, reason: str) -> "Decision":
        return cls(False, False, reason)


class Guard:
    def __init__(self, allowed_roots: list[Path]) -> None:
        self.allowed_roots = [p.expanduser().resolve() for p in allowed_roots]

    @classmethod
    def from_settings(cls, settings) -> "Guard":
        return cls(settings.allowed_roots)

    def _path_inside_allowed(self, candidate: Path) -> bool:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def check(self, tool: Tool, arguments: dict) -> Decision:
        """Decide si una llamada a tool puede correr."""
        # 1) Validar cualquier argumento que parezca una ruta contra el allowlist.
        for arg_name, value in arguments.items():
            if arg_name.lower() in _PATH_ARG_NAMES and isinstance(value, str) and value:
                if not self._path_inside_allowed(Path(value)):
                    return Decision.block(
                        f"La ruta '{value}' está fuera de las zonas permitidas, patrón. "
                        "No salgo del corral."
                    )

        # 2) Tools marcadas como no-safe piden confirmación explícita.
        if not tool.safe:
            return Decision.confirm(
                f"La acción '{tool.name}' puede ser destructiva, patrón. ¿La confirmo?"
            )

        return Decision.ok()
