"""Tools para leer y razonar sobre los proyectos del patrón.

Migrado/ampliado de C1::skills.py (analyze_project, launch_project). Lo nuevo:
read_project_file y list_project_tree -> Crotolamo deja de ser un abridor de
carpetas y pasa a poder responder "¿qué hace el archivo X de Huevonitis?".

Los proyectos se leen de [projects] en la config (cero hardcodeo).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from crotolamo.settings import get_settings
from crotolamo.tools.base import tool
from crotolamo.tools.desktop import normalize_key, run_detached, terminal_exec

_SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules"}
_READ_CAP = 20_000


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
    for path in project.rglob("*"):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
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

    lines: list[str] = [f"{project.name}/"]
    for top in sorted(project.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if top.name in _SKIP_DIRS or top.name.startswith("."):
            continue
        lines.append(f"  {'📁' if top.is_dir() else '📄'} {top.name}")
        if top.is_dir():
            for child in sorted(top.iterdir())[:12]:
                if child.name in _SKIP_DIRS or child.name.startswith("."):
                    continue
                lines.append(f"    {'📁' if child.is_dir() else '📄'} {child.name}")
    return "\n".join(lines)


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

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"No pude leer {target}, patrón: {error}"
    if len(text) > _READ_CAP:
        text = text[:_READ_CAP] + f"\n...[recortado, {len(text)} chars en total]"
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

    cmd = ["grep", "-rniI", "--max-count=3",
           "--exclude-dir=.git", "--exclude-dir=.venv", "--exclude-dir=__pycache__",
           pattern, str(base)]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return "La búsqueda falló o tardó demasiado, patrón."

    lines = result.stdout.strip().splitlines()
    if not lines:
        return f"No encontré «{pattern}» en {project}, patrón."
    shown = lines[:25]
    out = "\n".join(s.replace(str(base) + "/", "") for s in shown)
    extra = f"\n...y {len(lines) - 25} coincidencias más." if len(lines) > 25 else ""
    return f"Encontré «{pattern}» en {project}, patrón:\n{out}{extra}"


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

    launchers: list[Path] = []
    for pattern in ("launch*.sh", "*launcher*.sh", "run*.sh", "start*.sh", "*.desktop"):
        for path in project.rglob(pattern):
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            launchers.append(path)

    if not launchers:
        argv = terminal_exec(f'cd "{project}"; exec bash')
        if argv is None:
            return f"No hallé launcher ni terminal para {name}, patrón. Cablea uno en [apps]."
        run_detached(argv)
        return f"No hallé launcher claro para {name}, patrón. Te abrí una terminal ahí."

    launcher = launchers[0]
    if launcher.suffix == ".desktop":
        run_detached(["gtk-launch", launcher.stem])
        return f"Lancé {name} con {launcher.name}, patrón."
    argv = terminal_exec(f'cd "{project}" && "{launcher}"; exec bash')
    if argv is None:
        return f"No encontré un terminal para lanzar {name}, patrón. Cablea uno en [apps]."
    run_detached(argv)
    return f"Lancé {name} con {launcher.name}, patrón. Que el código tenga piedad."
