"""Tools para leer y razonar sobre los proyectos del patrón.

Migrado/ampliado de C1::skills.py (analyze_project, launch_project). Lo nuevo:
read_project_file y list_project_tree -> Crotolamo deja de ser un abridor de
carpetas y pasa a poder responder "¿qué hace el archivo X de Huevonitis?".

Los proyectos se leen de [projects] en la config (cero hardcodeo).
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
from pathlib import Path

from crotolamo.settings import get_settings
from crotolamo.tools.base import (
    SKIP_DIRS,
    normalize_key,
    read_text_capped,
    tool,
    truncate_for_context,
)
from crotolamo.tools.desktop import run_detached, terminal_exec

# Entradas de una subcarpeta que enseña list_project_tree.
_TREE_CHILDREN = 12

# Tope de caracteres de cada coincidencia que enseña find_in_project. Sin él,
# un JSON o un .min.js de una sola línea iba ENTERO al LLM (megas); con él,
# el peor caso (25 líneas recortadas) queda por debajo de READ_CAP.
_LINE_CAP = 500

# Metacaracteres con los que grep, leyendo el patrón como regex (BRE), puede NO
# casar con ese mismo texto escrito tal cual. El punto no está: `a.b` leído
# como regex también casa con "a.b".
_REGEX_META = frozenset("[*^$\\")


def _walk_files(root: Path) -> "list[Path]":
    """Archivos del proyecto, PODANDO los directorios de SKIP_DIRS antes de
    descender (un rglob + filtro posterior se traga .venv/node_modules enteros)."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        files.extend(Path(dirpath) / f for f in sorted(filenames))
    return files


def _projects() -> dict[str, Path]:
    return {normalize_key(name): path for name, path in get_settings().projects.items()}


def _resolve_project(name: str) -> Path | None:
    return _projects().get(normalize_key(name))


@tool
def list_projects() -> str:
    """Lista los proyectos que Crotolamo conoce."""
    projects = _projects()
    if not projects:
        return "No tengo proyectos registrados, patrón. Agrégalos en [projects] de la config."
    lines = [f"- {name}: {path}" for name, path in projects.items()]
    return "Proyectos que conozco, patrón:\n" + "\n".join(lines)


@tool
def analyze_project(name: str) -> str:
    """Resumen rápido de un proyecto: ruta y archivos Python/shell que contiene.

    Args:
        name: nombre del proyecto (crotolamo, huevonitis, ...).
    """
    project = _resolve_project(name)
    if project is None:
        return f"No tengo registrado el proyecto '{name}', patrón."
    if not project.exists():
        return f"No encontré el proyecto {name}: {project}"

    py_files, sh_files = [], []
    for path in _walk_files(project):
        if path.suffix == ".py":
            py_files.append(path)
        elif path.suffix == ".sh":
            sh_files.append(path)

    lines = [
        f"Análisis de {name}, patrón:",
        f"Ruta: {project}",
        f"Archivos Python: {len(py_files)} (muestro 20)",
    ]
    lines += [f"PY: {f.relative_to(project)}" for f in py_files[:20]]
    if sh_files:
        lines.append(f"Scripts shell: {len(sh_files)} (muestro 15)")
        lines += [f"SH: {f.relative_to(project)}" for f in sh_files[:15]]
    return "\n".join(lines)


@tool
def list_project_tree(name: str) -> str:
    """Árbol de carpetas y archivos de primer/segundo nivel de un proyecto.

    Args:
        name: nombre del proyecto.
    """
    project = _resolve_project(name)
    if project is None:
        return f"No tengo registrado el proyecto '{name}', patrón."
    if not project.exists():
        return f"No encontré el proyecto {name}: {project}"

    # iterdir()/is_dir() lanzan PermissionError (u otro OSError) sobre carpetas
    # sin permisos; sin red de seguridad la tool "reventaba" en vez de
    # responder en personaje.
    try:
        lines = _tree_lines(project)
    except OSError as error:
        return f"No pude leer el árbol de {name}, patrón: {error}"
    return "\n".join(lines)


def _tree_lines(project: Path) -> list[str]:
    """Líneas del árbol (primer y segundo nivel). Puede lanzar OSError."""
    lines: list[str] = [f"{project.name}/"]
    for top in sorted(project.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if top.name in SKIP_DIRS or top.name.startswith("."):
            continue
        lines.append(f"  {'📁' if top.is_dir() else '📄'} {top.name}")
        if not top.is_dir():
            continue
        # Se filtra ANTES de recortar: cortando primero, los ocultos gastaban
        # el cupo y una carpeta con 12 dotfiles y dos documentos salía vacía.
        try:
            children = [
                child for child in sorted(top.iterdir())
                if child.name not in SKIP_DIRS and not child.name.startswith(".")
            ]
        except OSError:
            # Una subcarpeta sin permisos no debe tirar el árbol entero.
            lines.append("    (no pude leer esta carpeta, patrón)")
            continue
        for child in children[:_TREE_CHILDREN]:
            lines.append(f"    {'📁' if child.is_dir() else '📄'} {child.name}")
        # Sin esta marca el modelo daba el recorte por la carpeta completa
        # ("tests/ tiene 12 archivos", "no existe test_projects.py").
        if len(children) > _TREE_CHILDREN:
            lines.append(f"    ...y {len(children) - _TREE_CHILDREN} más.")
    return lines


@tool
def read_project_file(project: str, relative_path: str) -> str:
    """Lee un archivo de un proyecto y devuelve su contenido para razonar sobre él.

    Args:
        project: nombre del proyecto (crotolamo, huevonitis, ...).
        relative_path: ruta del archivo relativa a la raíz del proyecto.
    """
    base = _resolve_project(project)
    if base is None:
        return f"No tengo registrado el proyecto '{project}', patrón."
    if not base.exists():
        return f"No encontré el proyecto {project}: {base}"

    base = base.resolve()
    target = (base / relative_path).resolve()
    # Anti-traversal: el archivo debe quedar DENTRO del proyecto.
    if base != target and base not in target.parents:
        return f"Esa ruta se sale del proyecto {project}, patrón. No salgo del corral."
    if not target.exists():
        return f"No existe {relative_path} en {project}, patrón."
    if target.is_dir():
        return f"{relative_path} es una carpeta, patrón. Usa list_project_tree."
    # Un FIFO sin escritor bloquea el open() para siempre: solo archivos normales.
    if not target.is_file():
        return f"{relative_path} no es un archivo normal, patrón, no lo puedo leer."

    try:
        text = read_text_capped(target)
    except OSError as error:
        return f"No pude leer {target}, patrón: {error}"
    return f"Contenido de {project}/{relative_path}:\n{text}"


@tool
def find_in_project(project: str, pattern: str) -> str:
    """Busca un texto/patrón dentro de los archivos de un proyecto (read-only).

    Args:
        project: nombre del proyecto.
        pattern: el texto a buscar.
    """
    base = _resolve_project(project)
    if base is None:
        return f"No tengo registrado el proyecto '{project}', patrón."
    if not base.exists():
        return f"No encontré el proyecto {project}: {base}"
    if not pattern.strip():
        return "Dame algo que buscar, patrón."

    try:
        result = _grep(base, pattern)
        lines = result.stdout.strip().splitlines()
        # El modelo pasa código tal cual, y grep lo lee como regex: `list[str]`
        # es "lis" + una de {s,t,r} y no casa con "list[str]"; `arr[` ni
        # siquiera es un regex (grep sale con 2 y sin salida). Si como regex no
        # salió nada y el patrón lleva metacaracteres, se busca como texto.
        if not lines and _REGEX_META.intersection(pattern):
            result = _grep(base, pattern, literal=True)
            lines = result.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return "La búsqueda falló o tardó demasiado, patrón."

    if not lines:
        # rc=1 es "sin coincidencias"; rc>=2, que grep falló (un archivo que no
        # pudo leer): la búsqueda no fue completa y hay que decirlo.
        if result.returncode >= 2 and (aviso := result.stderr.strip()):
            return (f"No encontré «{pattern}» en {project}, patrón, pero la búsqueda "
                    f"no fue completa: {aviso.splitlines()[0][:200]}")
        return f"No encontré «{pattern}» en {project}, patrón."
    shown = lines[:25]
    out = "\n".join(
        truncate_for_context(s.replace(str(base) + "/", ""), _LINE_CAP) for s in shown
    )
    extra = f"\n...y {len(lines) - 25} coincidencias más." if len(lines) > 25 else ""
    return f"Encontré «{pattern}» en {project}, patrón:\n{out}{extra}"


def _grep(base: Path, pattern: str, literal: bool = False) -> subprocess.CompletedProcess:
    """grep recursivo del proyecto; con `literal`, el patrón es texto y no regex.
    Lanza OSError/TimeoutExpired."""
    # `-e` y `--`: el patrón lo elige el LLM. Sin ellos, un patrón que empieza
    # por guion se interpreta como opción de grep; con "-r" la ruta del
    # proyecto pasaba a ser "el patrón" y grep buscaba en el CWD del proceso,
    # fuera del proyecto (y del corral).
    # Las exclusiones salen de SKIP_DIRS: repetidas a mano se quedaron sin
    # `venv` ni `node_modules` y las coincidencias de las dependencias copaban
    # las líneas que se muestran. El nombre de la propia base no se excluye:
    # grep aplica --exclude-dir también a la carpeta que se le pasa, y un
    # proyecto llamado así se quedaba sin buscar.
    skip = [f"--exclude-dir={name}" for name in sorted(SKIP_DIRS - {base.name})]
    cmd = ["grep", "-rniI", *(["-F"] if literal else []), "--max-count=3", *skip,
           "-e", pattern, "--", str(base)]
    # errors="replace": grep imprime el nombre del archivo y la línea tal cual;
    # un byte que no sea UTF-8 lanzaba UnicodeDecodeError con text=True a secas.
    return subprocess.run(
        cmd, text=True, errors="replace", capture_output=True, timeout=20,
    )


def _spawn(name: str, argv: list[str], done: str) -> str:
    """Lanza `argv` y devuelve `done`; si no arranca, lo dice en personaje."""
    # run_detached deja pasar el OSError de Popen cuando el binario no existe
    # o no se puede ejecutar (gtk-launch sin instalar, un terminal de [apps]
    # mal escrito): sin red, la tool "reventaba" con el errno crudo.
    try:
        run_detached(argv)
    except (OSError, subprocess.SubprocessError) as error:
        return f"No pude lanzar {name}, patrón: {error}"
    return done


@tool
def launch_project(name: str) -> str:
    """Lanza un proyecto buscando su launcher (.sh/.desktop) y abriéndolo.

    Args:
        name: nombre del proyecto.
    """
    project = _resolve_project(name)
    if project is None:
        return f"No tengo registrado el proyecto '{name}', patrón."
    if not project.exists():
        return f"No encontré el proyecto {name}: {project}"

    # UN solo recorrido (podado) matcheando los 5 patrones; se conserva la
    # prioridad original: launch*.sh gana a run*.sh, y estos a *.desktop.
    patterns = ("launch*.sh", "*launcher*.sh", "run*.sh", "start*.sh", "*.desktop")
    candidates: list[tuple[int, Path]] = []
    for path in _walk_files(project):
        for rank, pattern in enumerate(patterns):
            if fnmatch.fnmatchcase(path.name, pattern):
                candidates.append((rank, path))
                break
    candidates.sort(key=lambda item: item[0])  # sort estable: conserva el orden del walk
    launchers = [path for _rank, path in candidates]

    if not launchers:
        # shlex.quote: `terminal_exec` arma `bash -lc <cadena>`. Sin escapar, un
        # nombre de ruta con comillas o `$(...)` se ejecutaría como sintaxis de
        # shell en vez de tratarse como un literal.
        argv = terminal_exec(f"cd {shlex.quote(str(project))}; exec bash")
        if argv is None:
            return f"No hallé launcher ni terminal para {name}, patrón. Cablea uno en [apps]."
        return _spawn(
            name, argv, f"No hallé launcher claro para {name}, patrón. Te abrí una terminal ahí.",
        )

    launcher = launchers[0]
    if launcher.suffix == ".desktop":
        return _spawn(
            name, ["gtk-launch", launcher.stem], f"Lancé {name} con {launcher.name}, patrón.",
        )
    # `launcher` sale de un rglob del proyecto: es un nombre de archivo del disco,
    # no una constante. Un archivo llamado `run";rm -rf ~;".sh` se ejecutaría.
    argv = terminal_exec(
        f"cd {shlex.quote(str(project))} && {shlex.quote(str(launcher))}; exec bash"
    )
    if argv is None:
        return f"No encontré un terminal para lanzar {name}, patrón. Cablea uno en [apps]."
    return _spawn(
        name, argv, f"Lancé {name} con {launcher.name}, patrón. Que el código tenga piedad.",
    )
