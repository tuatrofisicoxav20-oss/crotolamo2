"""Validación de seguridad por ALLOWLIST, no por blocklist de regex.

C1 dependía de buscar substrings peligrosos ("rm -rf"...) en bash crudo: un
sandbox de papel. C2 no ejecuta bash arbitrario; las tools son funciones. El
guard decide, por tool y por argumentos, si la acción:
  - corre directo (safe),
  - necesita confirmación del patrón,
  - o se bloquea (p.ej. una ruta fuera de las raíces permitidas).

Tres zonas por ruta (M6):
  - dentro de allowed_roots  -> libre (corre directo si la tool es safe),
  - dentro de confirm_roots  -> pide confirmación al patrón,
  - fuera de ambas           -> bloqueada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from crotolamo.safety.paths import path_inside_roots
from crotolamo.tools.base import Tool

# Nombres de argumentos que típicamente contienen rutas de archivo (Fase 3).
# Los plurales (M4) cubren las listas de rutas de los servers MCP
# (p.ej. read_multiple_files(paths=[...]) del server de filesystem).
_PATH_ARG_NAMES = {
    "path", "ruta", "file", "archivo", "dest", "destino", "src", "origen", "dir",
    "paths", "rutas",
    # Nombres que usan los servers MCP habituales (filesystem: move_file(source,
    # destination), read_file(path)...): una ruta RELATIVA ahí no lleva prefijo
    # y solo el nombre la delata.
    "source", "destination", "filename", "file_path", "filepath", "directory",
    "folder", "carpeta", "old_path", "new_path", "target_path",
}

# Prefijos que delatan una ruta del sistema de archivos (señal aparte del nombre).
_PATH_PREFIXES = ("/", "~/", "./", "../")

# Tope de anidamiento al recorrer argumentos (M4). Los argumentos de una tool MCP
# son JSON arbitrario; con 8 niveles sobra para cualquier esquema real y un JSON
# patológico (miles de niveles) no nos revienta la pila. Más hondo, no se mira.
_MAX_DEPTH = 8


# Argumentos que llevan CONTENIDO (texto libre), nunca rutas: a ellos no se les
# aplica la heurística de prefijo. Sin esta lista, write_file(content="/usr/bin/
# env python3 ...") se bloqueaba como "ruta fuera del corral" y remember_fact(
# texto="~/proyectos es donde guardo todo") pedía confirmación (y en el modo
# concurrente se cancelaba). El nombre de argumento que SÍ es ruta
# (_PATH_ARG_NAMES) sigue mandando aunque el valor no lleve prefijo.
_CONTENT_ARG_NAMES = {
    "content", "contenido", "text", "texto", "query", "pattern", "patron", "patrón",
    "title", "titulo", "título", "body", "message", "mensaje", "prompt", "reason",
    "note", "nota", "fact", "hecho", "value", "valor",
}


def _looks_like_path(value: str) -> bool:
    # Un valor con saltos de línea es contenido (un script, una nota), no una ruta.
    return value.startswith(_PATH_PREFIXES) and "\n" not in value


def _iter_strings(
    value: Any, key: str, depth: int = 0, too_deep: list[str] | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Recorre un argumento (posiblemente anidado) y suelta (nombre, string, nivel).

    Cada string lleva como "nombre" la clave del dict más cercano que lo contiene
    (los elementos de una lista heredan la clave de la lista): así una lista bajo
    `paths` se clasifica igual que un `path` suelto. Recursión simple con tope;
    lo que quede más hondo NO se inspecciona, y se anota en `too_deep` para que
    el guard no lo deje pasar en silencio (fail-open).
    """
    if isinstance(value, str):
        if value:
            yield key, value, depth
    elif depth >= _MAX_DEPTH:
        if too_deep is not None:
            too_deep.append(key)
        return
    elif isinstance(value, dict):
        for sub_key, sub_value in value.items():
            yield from _iter_strings(sub_value, str(sub_key), depth + 1, too_deep)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item, key, depth + 1, too_deep)


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
    def __init__(
        self,
        allowed_roots: list[Path],
        confirm_roots: list[Path] | None = None,
    ) -> None:
        self.allowed_roots = [p.expanduser().resolve() for p in allowed_roots]
        # Zona de confirmación (M6). Default: el home del patrón — Crotolamo puede
        # usar toda la lap, pero fuera de la zona libre pide permiso primero.
        if confirm_roots is None:
            confirm_roots = [Path("~")]
        self.confirm_roots = [p.expanduser().resolve() for p in confirm_roots]

    @classmethod
    def from_settings(cls, settings) -> "Guard":
        return cls(settings.allowed_roots, getattr(settings, "confirm_roots", None))

    def _path_inside_allowed(self, candidate: Path) -> bool:
        return path_inside_roots(candidate, self.allowed_roots)

    def _path_inside_confirm(self, candidate: Path) -> bool:
        return path_inside_roots(candidate, self.confirm_roots)

    def check(self, tool: Tool, arguments: dict) -> Decision:
        """Decide si una llamada a tool puede correr."""
        # 1) Clasificar en zonas cualquier argumento que sea una ruta:
        #    por nombre conocido (señal fuerte) O porque el valor parece un path.
        #    Se recorren también listas y dicts anidados (M4): las tools MCP
        #    reciben JSON arbitrario y una ruta escondida en {"opciones":
        #    {"ruta": "/etc/passwd"}} merece el mismo corral que una de primer nivel.
        confirm_reason = ""
        too_deep: list[str] = []
        for arg_name, value, depth in _iter_strings(arguments, "", 0, too_deep):
            name = arg_name.lower()
            # La exención de contenido solo aplica al argumento de PRIMER nivel
            # (content="..." de write_file): un string escondido en una lista o
            # dict bajo "value" sigue mirándose como cualquier otro.
            is_content = depth == 1 and name in _CONTENT_ARG_NAMES
            is_path_arg = name in _PATH_ARG_NAMES or (
                not is_content and _looks_like_path(value)
            )
            if not is_path_arg:
                continue
            candidate = Path(value)
            if self._path_inside_allowed(candidate):
                continue
            if self._path_inside_confirm(candidate):
                if not confirm_reason:
                    confirm_reason = f"fuera de la zona libre: {value}. ¿Lo hago?"
                continue
            return Decision.block(
                f"La ruta '{value}' está fuera de las zonas permitidas, patrón. "
                "No salgo del corral."
            )

        # Argumentos más hondos que el tope no se inspeccionaron: no se permite
        # a ciegas, se pregunta (fail-closed). Un JSON legítimo casi nunca pasa
        # de 8 niveles; uno que lo hace merece que el patrón lo vea.
        if too_deep and not confirm_reason:
            confirm_reason = (
                f"argumentos anidados más de {_MAX_DEPTH} niveles ({too_deep[0] or 'raíz'}): "
                "no puedo revisarlos. ¿Lo hago?"
            )

        # 2) Tools marcadas como no-safe piden confirmación explícita SIEMPRE,
        #    sin importar en qué zona caigan sus rutas.
        if not tool.safe:
            return Decision.confirm(
                f"La acción '{tool.name}' puede ser destructiva, patrón. ¿La confirmo?"
            )

        # 3) Rutas en la zona de confirmación: se puede, pero preguntando.
        if confirm_reason:
            return Decision.confirm(confirm_reason)

        return Decision.ok()
