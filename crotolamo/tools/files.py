"""Operaciones de archivo SEGURAS. Reemplazan el bash crudo de C1.

Cada tool que recibe una ruta usa un nombre de argumento que el guard reconoce
(`path`, `src`, `dest`...) y valida contra la allowlist de [paths].allowed_roots.
Las destructivas (delete/move) van marcadas safe=False -> el guard pide confirmación.
Aquí NO se revalida la allowlist: esa es responsabilidad del guard en el agente;
las funciones son usables directamente (p.ej. en tests) sin corral.
"""

from __future__ import annotations

from pathlib import Path

from crotolamo.settings import get_settings
from crotolamo.tools.base import tool

# Tope de lectura para no inundar el contexto del LLM.
_READ_CAP = 20_000


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


@tool
def read_file(path: str) -> str:
    """Lee un archivo de texto y devuelve su contenido (recortado si es enorme).

    Args:
        path: ruta del archivo a leer.
    """
    p = _resolve(path)
    if not p.exists():
        return f"No existe el archivo, patrón: {p}"
    if p.is_dir():
        return f"Eso es una carpeta, patrón, no un archivo: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"No pude leer {p}, patrón: {error}"
    if len(text) > _READ_CAP:
        text = text[:_READ_CAP] + f"\n...[recortado, {len(text)} chars en total]"
    return text


@tool
def write_file(path: str, content: str) -> str:
    """Escribe (crea o sobrescribe) un archivo de texto.

    Args:
        path: ruta del archivo a escribir.
        content: contenido a guardar.
    """
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"No pude escribir {p}, patrón: {error}"
    return f"Escribí {len(content)} chars en {p}, patrón."


@tool
def list_dir(path: str) -> str:
    """Lista el contenido de una carpeta.

    Args:
        path: ruta de la carpeta.
    """
    p = _resolve(path)
    if not p.exists():
        return f"No existe la carpeta, patrón: {p}"
    if not p.is_dir():
        return f"Eso no es una carpeta, patrón: {p}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    if not entries:
        return f"{p} está vacía, patrón."
    lines = [f"{'📁' if e.is_dir() else '📄'} {e.name}" for e in entries[:100]]
    extra = f"\n...y {len(entries) - 100} más." if len(entries) > 100 else ""
    return f"Contenido de {p}, patrón:\n" + "\n".join(lines) + extra


@tool
def make_dir(path: str) -> str:
    """Crea una carpeta (y las intermedias que falten).

    Args:
        path: ruta de la carpeta a crear.
    """
    p = _resolve(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return f"No pude crear {p}, patrón: {error}"
    return f"Carpeta lista, patrón: {p}"


@tool(safe=False)
def move_file(src: str, dest: str) -> str:
    """Mueve o renombra un archivo o carpeta. Operación destructiva (pide confirmación).

    Args:
        src: ruta origen.
        dest: ruta destino.
    """
    source = _resolve(src)
    target = _resolve(dest)
    if not source.exists():
        return f"No existe el origen, patrón: {source}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    except OSError as error:
        return f"No pude mover {source}, patrón: {error}"
    return f"Moví {source} -> {target}, patrón."


@tool(safe=False)
def delete_file(path: str) -> str:
    """Borra un archivo. Operación destructiva (pide confirmación).

    Args:
        path: ruta del archivo a borrar.
    """
    p = _resolve(path)
    if not p.exists():
        return f"No existe, patrón, así que nada que borrar: {p}"
    if p.is_dir():
        return f"Es una carpeta, patrón. No borro carpetas con esta tool por seguridad: {p}"
    try:
        p.unlink()
    except OSError as error:
        return f"No pude borrar {p}, patrón: {error}"
    return f"Borrado, patrón: {p}"


@tool
def search_files(query: str) -> str:
    """Busca archivos por nombre dentro de Documentos del patrón (read-only).

    Args:
        query: parte del nombre del archivo a buscar.
    """
    import subprocess

    base = get_settings().paths.get("documentos", Path.home() / "Documentos")
    query = query.strip()
    if not query:
        return "Dame un nombre que buscar, patrón."
    if not base.exists():
        return f"No existe la base de búsqueda, patrón: {base}"
    try:
        result = subprocess.run(
            ["find", str(base), "-iname", f"*{query}*"],
            text=True, capture_output=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "La búsqueda tardó demasiado, patrón."
    lines = result.stdout.strip().splitlines()
    if not lines:
        return f"No encontré archivos con «{query}», patrón."
    shown = lines[:20]
    extra = f"\n...y {len(lines) - 20} más." if len(lines) > 20 else ""
    return f"Encontré esto, patrón:\n" + "\n".join(shown) + extra


@tool
def create_note(title: str, content: str = "") -> str:
    """Crea una nota en markdown en la carpeta de notas del patrón.

    Args:
        title: título de la nota (se usa como nombre de archivo).
        content: cuerpo opcional de la nota.
    """
    notas_dir = get_settings().paths.get("notas", Path.home() / "Documentos" / "crotolamo_notas")
    notas_dir.mkdir(parents=True, exist_ok=True)

    clean = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
    clean = clean.replace(" ", "_") or "nota"
    path = notas_dir / f"{clean}.md"

    if not content.strip():
        content = f"# {title}\n\n"
    path.write_text(content, encoding="utf-8")
    return f"Nota creada, patrón: {path}"
