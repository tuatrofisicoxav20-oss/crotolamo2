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

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit

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

# Un URI de archivo local también es una ruta: un server MCP de navegador o de
# fetch abre file:///home/... igual que un path. Como no empieza por "/", pasaba
# LIBRE bajo cualquier nombre de argumento (y para las tools MCP el guard es el
# único corral). Se lee como lo lee un parser de URLs (los navegadores; urllib
# igual), que ignora los controles C0 y espacios de las puntas y los tabs/saltos
# de línea de cualquier sitio: para quien lo abre, "fi\tle:///etc/pass\nwd" es
# /etc/passwd. La barra tras el esquema NO hace falta: un navegador abre
# "file:etc/passwd" y "file:\etc\passwd" como /etc/passwd, y urllib abre
# "file:../x" relativo a su CWD; pedirla dejaba pasar esas formas sin
# clasificar. Basta que siga algo que no sea un espacio, para no tomar por URI
# un texto como "file: x".
_URL_EDGE_JUNK = "".join(map(chr, range(0x21)))
_URL_INNER_JUNK = ("\t", "\r", "\n")
_FILE_URI_RE = re.compile(
    r"[\x00-\x20]*" + r"[\t\r\n]*".join("file:") + r"(?=[\t\r\n]*\S)",
    re.ASCII | re.IGNORECASE,
)

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
    # Texto libre de las tools de memoria (buscar_recuerdos, olvidar_recuerdo) y
    # la etiqueta de un hecho: "olvida lo de ~/proyectos" es una frase, no una ruta.
    "pregunta", "descripcion", "descripción", "categoria", "categoría",
}


def _uri_path(uri: str) -> str:
    """La ruta de un URI file: ya limpio, con los %xx decodificados."""
    try:
        path = urlsplit(uri).path
    except ValueError:
        # URI malformado (p.ej. un host IPv6 a medias): no se deja pasar por no
        # saber leerlo; se mira lo que venga tras el esquema.
        path = uri[len("file:"):]
    return unquote(path)


def _file_uri_paths(value: str) -> list[str]:
    """Las rutas locales que abre un URI file: ([] si `value` no lo es).

    Dos lecturas, y el guard clasifica las dos: la de un parser WHATWG (los
    navegadores, Node), que toma "\\" por "/" y lee desde la raíz lo que no
    lleva barra ("file:etc/passwd" es /etc/passwd), y la de urllib, que abre el
    resto tal cual, relativo a su CWD si no empieza por "/". Con barra suelen
    coincidir y queda una sola.
    """
    if not _FILE_URI_RE.match(value):
        return []
    uri = value.strip(_URL_EDGE_JUNK)
    for junk in _URL_INNER_JUNK:
        uri = uri.replace(junk, "")
    browser = _uri_path("file:" + uri[len("file:"):].replace("\\", "/"))
    if not browser.startswith("/"):
        browser = "/" + browser
    plain = _uri_path(uri)
    return [browser] if plain == browser else [browser, plain]


def _looks_like_path(value: str) -> bool:
    # Se mira recortado: " /etc/passwd" y "/etc/passwd\n" son rutas para un
    # server que haga trim. Un salto de línea ENTRE medias sí es contenido (un
    # script, una nota), no una ruta...
    text = value.strip()
    if text.startswith(_PATH_PREFIXES) and "\n" not in text:
        return True
    # ...salvo en un URI file: (quien lo abre le quita los saltos de línea).
    return _FILE_URI_RE.match(value) is not None


def _path_forms(value: str, path_name: bool) -> list[str]:
    """Las rutas locales que `value` puede acabar siendo para quien lo reciba.

    El guard las clasifica TODAS y manda la zona más estricta:
    - Un URI file: es su ruta (o sus dos lecturas, ver _file_uri_paths). Bajo
      un nombre de ruta (`path_name`) cuenta además tal cual: una tool nativa
      no entiende el esquema y lo abriría como ruta relativa.
    - Un valor con espacios o saltos de línea en las puntas se mira recortado
      (lo que abriría un server que haga trim) y tal cual (las tools nativas no
      recortan).
    """
    local = _file_uri_paths(value)
    if local:
        return [*local, value] if path_name else local
    text = value.strip()
    return [text] if text == value else [text, value]


def _is_relative(path: Path) -> bool:
    """True si `path` sigue sin ser absoluta tras expandir el ~: es la que se
    resuelve contra una base (el CWD del proceso, o el ancla del guard)."""
    try:
        return not path.expanduser().is_absolute()
    except RuntimeError:
        # "~usuario" sin home conocido: no se sabe qué ruta es. Que la juzgue
        # el corral tal cual: tampoco podrá resolverla, y la bloquea.
        return False


def _iter_strings(
    value: Any, key: str, depth: int = 0, too_deep: list[str] | None = None,
    top: str = "",
) -> Iterator[tuple[str, str, int, str]]:
    """Recorre un argumento (posiblemente anidado) y suelta (nombre, string,
    nivel, argumento de primer nivel del que cuelga).

    Cada string lleva como "nombre" la clave del dict más cercano que lo contiene
    (los elementos de una lista heredan la clave de la lista): así una lista bajo
    `paths` se clasifica igual que un `path` suelto. Recursión simple con tope;
    lo que quede más hondo NO se inspecciona, y se anota en `too_deep` para que
    el guard no lo deje pasar en silencio (fail-open).
    """
    if isinstance(value, str):
        # También el vacío: bajo un nombre de ruta, "" es la carpeta base
        # (Path("") == "."), y saltárselo dejaba a list_dir(path="") listar el
        # CWD sin pasar por las zonas.
        yield key, value, depth, top
    elif depth >= _MAX_DEPTH:
        if too_deep is not None:
            too_deep.append(key)
        return
    elif isinstance(value, dict):
        for sub_key, sub_value in value.items():
            yield from _iter_strings(
                sub_value, str(sub_key), depth + 1, too_deep, top if depth else str(sub_key),
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item, key, depth + 1, too_deep, top)


# La pregunta de una tool destructiva dice QUÉ va a tocar (ver _destructive_question):
# hasta 2 rutas, o hasta 3 argumentos de 60 caracteres. Es lo que se oye por
# voz antes de decir "sí": lo justo para distinguir una nota de ~/.bashrc, la
# ventana o el comando, sin leer en voz alta un documento entero.
_SHOWN_TARGETS = 2
_SHOWN_ARGS = 3
_SHOWN_CHARS = 60
# Una ruta se recorta por en medio y con más margen: el principio dice la zona
# (~/...) y el final, el archivo; cortarle la cola escondería justo el nombre.
_SHOWN_PATH_CHARS = 100

# Controles C0 y DEL. En la pregunta van escapados ("\n"): un salto de línea en
# un argumento partía lo que imprime el shell y podía esconder lo que sigue.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _visible(text: str, limit: int, middle: bool = False) -> str:
    """`text` legible en una sola línea y recortado a unos `limit` caracteres."""
    text = _CONTROL_RE.sub(lambda m: repr(m.group())[1:-1], text)
    if len(text) <= limit:
        return text
    if middle:
        head = limit // 3
        return text[:head] + "…" + text[len(text) - (limit - head - 1):]
    return text[:limit] + "…"


def _homes() -> list[Path]:
    """El home tal cual y resuelto (en Silverblue /home es un enlace a
    /var/home): una ruta se nombra con ~ si cae en cualquiera de los dos."""
    try:
        home = Path.home()
        real = home.resolve()
    except (OSError, RuntimeError, KeyError, ValueError):
        return []
    return [home] if real == home else [home, real]


def _with_tilde(path: Path, homes: list[Path]) -> str:
    for home in homes:
        try:
            rel = path.relative_to(home)
        except ValueError:
            continue
        return f"~/{rel}" if rel.parts else "~"
    return str(path)


def _target_name(candidate: Path, outside: bool, homes: list[Path]) -> str:
    """Cómo se nombra una ruta en la pregunta de una tool destructiva.

    Se nombra la que de verdad se toca, con enlaces y '..' resueltos (la misma
    que clasificó path_inside_roots): "~/Documentos/../.bashrc" es ~/.bashrc.
    Si la escrita apunta a otro sitio por un enlace, van las dos: write_file
    escribe en el destino y delete_file/move_file quitan o mueven el enlace, y
    "~/Documentos/link" a secas sonaría a zona libre aunque lleve a ~/.ssh.
    """
    notes = ["fuera de la zona libre"] if outside else []
    try:
        expanded = candidate.expanduser()
        written = _with_tilde(Path(os.path.abspath(expanded)), homes)
        real = _with_tilde(expanded.resolve(), homes)
    except (OSError, RuntimeError, ValueError):
        written = real = str(candidate)
    if written != real:
        notes.insert(0, f"que apunta a '{_visible(real, _SHOWN_PATH_CHARS, middle=True)}'")
    name = f"'{_visible(written, _SHOWN_PATH_CHARS, middle=True)}'"
    return name + (f" ({', '.join(notes)})" if notes else "")


def _argument_summary(arguments: dict, skip: set[str], with_text: bool) -> str:
    """Los argumentos que no se nombraron como ruta, recortados: "name=opera",
    "command=rm -rf ...". El texto libre (content= de write_file) no se lee:
    es lo que se escribe, no dónde. Solo si no hay otra cosa que decir
    (`with_text`, sin rutas ni otros argumentos) se lee también: la consulta de
    una tool MCP de SQL va en query="DROP TABLE ...", y callarla dejaba la
    pregunta otra vez a ciegas."""
    if not isinstance(arguments, dict):
        return ""
    items = [(str(k), v) for k, v in arguments.items() if str(k) not in skip]
    shown = [(k, v) for k, v in items if k.lower() not in _CONTENT_ARG_NAMES]
    if not shown and with_text:
        shown = items
    parts = []
    for key, value in shown[:_SHOWN_ARGS]:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(value)
        parts.append(f"{_visible(key, _SHOWN_CHARS)}={_visible(text, _SHOWN_CHARS)}")
    if len(shown) > _SHOWN_ARGS:
        parts.append("…")
    return ", ".join(parts)


def _destructive_question(
    tool_name: str, arguments: dict, targets: list[tuple[Path, bool]],
    named: set[str], too_deep: list[str],
) -> str:
    """La pregunta de una tool no-safe: la acción, sobre qué rutas (las de
    fuera de la zona libre primero, marcadas) y con qué argumentos.

    Era una sola frase para todo ("La acción 'write_file' puede ser
    destructiva"), y es lo único que oye, lee o se le avisa al patrón antes de
    decidir: una nota de ~/Documentos y ~/.bashrc pedían el mismo "sí".
    """
    # Una ruta repetida (dos formas del mismo valor, la misma en dos
    # argumentos) se nombra una vez; las de fuera de la zona libre, delante.
    unique: dict[str, tuple[Path, bool]] = {}
    for candidate, outside in targets:
        unique.setdefault(str(candidate), (candidate, outside))
    ordered = sorted(unique.values(), key=lambda t: not t[1])
    homes = _homes() if ordered else []
    names = [_target_name(c, outside, homes) for c, outside in ordered[:_SHOWN_TARGETS]]
    if len(ordered) > _SHOWN_TARGETS:
        names.append(f"{len(ordered) - _SHOWN_TARGETS} más")
    sobre = ""
    if names:
        sobre = " sobre " + (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + " y " + names[-1]
        )
    summary = _argument_summary(arguments, named, with_text=not ordered)
    con = f" con {summary}" if summary else ""
    ojo = ""
    if too_deep:
        ojo = (
            f" Ojo: hay argumentos anidados más de {_MAX_DEPTH} niveles "
            f"({_visible(too_deep[0] or 'raíz', _SHOWN_CHARS)}) que no puedo revisar."
        )
    return f"La acción '{tool_name}'{sobre}{con} puede ser destructiva, patrón.{ojo} ¿La confirmo?"


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
        relative_base: Path | None = None,
    ) -> None:
        self.allowed_roots = [p.expanduser().resolve() for p in allowed_roots]
        # Zona de confirmación (M6). Default: el home del patrón — Crotolamo puede
        # usar toda la lap, pero fuera de la zona libre pide permiso primero.
        if confirm_roots is None:
            confirm_roots = [Path("~")]
        self.confirm_roots = [p.expanduser().resolve() for p in confirm_roots]
        # Ancla de las rutas RELATIVAS de las tools nativas (T10): la misma
        # que usa files._resolve ([paths].relative_base). None (de fábrica) =
        # el CWD del proceso, que es contra lo que el sistema la resuelve.
        self.relative_base = (
            relative_base.expanduser().resolve() if relative_base is not None else None
        )

    @classmethod
    def from_settings(cls, settings) -> "Guard":
        return cls(
            settings.allowed_roots,
            getattr(settings, "confirm_roots", None),
            getattr(settings, "relative_base", None),
        )

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
        # Lo que la llamada toca (ruta, ¿fuera de la zona libre?) y los
        # argumentos de primer nivel de los que sale: la pregunta de una tool
        # destructiva nombra lo primero y resume el resto (paso 2).
        targets: list[tuple[Path, bool]] = []
        named: set[str] = set()
        # Una ruta relativa se clasifica donde la va a abrir quien la recibe
        # (T10). Un server MCP (strict_args=False) la resuelve contra su
        # directorio de trabajo: el `cwd` de su config si lo tiene (el bridge
        # lo deja en tool.path_base) y, si no, el que hereda de este proceso,
        # el CWD. Clasificarla siempre contra el CWD dejaba que un server con
        # cwd = ~/proyectos/privado leyera ".env" sin preguntar (el CWD del
        # .service es el repo, zona libre). Las tools nativas no pueden depender
        # del CWD (la misma orden pasaba libre desde el repo, pedía
        # confirmación desde el home y se bloqueaba desde /; como servicio,
        # make_dir("x") caía dentro del repo de Crotolamo): ver más abajo.
        native = getattr(tool, "strict_args", True)
        server_cwd = None if native else getattr(tool, "path_base", None)
        server_base = Path(server_cwd) if server_cwd else None
        for arg_name, value, depth, top in _iter_strings(arguments, "", 0, too_deep):
            name = arg_name.lower()
            # La exención de contenido solo aplica al argumento de PRIMER nivel
            # (content="..." de write_file): un string escondido en una lista o
            # dict bajo "value" sigue mirándose como cualquier otro.
            is_content = depth == 1 and name in _CONTENT_ARG_NAMES
            path_name = name in _PATH_ARG_NAMES
            is_path_arg = path_name or (not is_content and _looks_like_path(value))
            if not is_path_arg:
                continue
            for form in _path_forms(value, path_name):
                candidate = Path(form)
                # Lo que oye el patrón: la ruta tal como vino ("" es la base).
                shown = form or "."
                if native and _is_relative(candidate):
                    # Bajo un nombre que no es de ruta, una tool nativa no abre
                    # el valor contra el CWD: read_project_file valida su
                    # relative_path contra la raíz del proyecto, que solo ella
                    # conoce. No hay nada que clasificar aquí. (Solo el
                    # argumento de primer nivel, como la exención de contenido.)
                    if depth == 1 and not path_name:
                        continue
                    # Si las de archivo la anclan ([paths].relative_base), se
                    # clasifica (y se nombra) la ruta que de verdad van a tocar.
                    if self.relative_base is not None:
                        candidate = self.relative_base / candidate
                        shown = str(candidate)
                elif server_base is not None and _is_relative(candidate):
                    # La del server MCP con cwd propio: se clasifica (y se
                    # nombra) la que él va a abrir.
                    candidate = server_base / candidate
                    shown = str(candidate)
                named.add(top)
                if self._path_inside_allowed(candidate):
                    targets.append((candidate, False))
                    continue
                if self._path_inside_confirm(candidate):
                    targets.append((candidate, True))
                    if not confirm_reason:
                        confirm_reason = f"fuera de la zona libre: {shown}. ¿Lo hago?"
                    continue
                return Decision.block(
                    f"La ruta '{shown}' está fuera de las zonas permitidas, patrón. "
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
        #    sin importar en qué zona caigan sus rutas. La pregunta nombra lo que
        #    tocan y lleva el aviso de zona (o el de anidado) que el paso 1 ya
        #    sabía: con una frase genérica, el "sí" a una nota valía igual para
        #    pisar ~/.bashrc o sacar la llave de ~/.ssh a la zona libre.
        if not tool.safe:
            return Decision.confirm(
                _destructive_question(tool.name, arguments, targets, named, too_deep)
            )

        # 3) Rutas en la zona de confirmación: se puede, pero preguntando.
        if confirm_reason:
            return Decision.confirm(confirm_reason)

        return Decision.ok()
