"""Operaciones de archivo SEGURAS. Reemplazan el bash crudo de C1.

Defensa en profundidad (M2): además del guard del agente, CADA tool de archivo
revalida ella misma la ruta contra el corral. Dos cerrojos: aunque el guard no
intercepte (p.ej. llamada directa), la tool rechaza rutas fuera del corral.
Las destructivas (delete/move/write) van marcadas safe=False.

Corral ampliado (M6): el corral de la tool = allowed_roots + confirm_roots. Si
una ruta cae en confirm_roots, el guard YA pidió confirmación al patrón antes de
ejecutar la tool; aquí solo se rechaza lo que queda fuera de ambas zonas.
"""

from __future__ import annotations

from pathlib import Path

from crotolamo.safety.paths import path_inside_roots
from crotolamo.settings import get_settings
from crotolamo.tools.base import SKIP_DIRS, read_text_capped, replace_surrogates, tool

# Tope del nombre de una nota, en BYTES: NAME_MAX es 255 en ext4/btrfs/xfs y hay
# que dejar sitio a ".md" y al sufijo "_N" de las notas con título repetido.
_NOTE_STEM_MAX_BYTES = 240


def _resolve(path: str) -> Path:
    """Ruta canónica: ~ expandido y symlinks/`..` resueltos.

    Se resuelve UNA vez, antes de validar, y las tools operan sobre ESA misma
    ruta. Validar la ruta resuelta y luego leer/escribir/mover la ruta cruda
    dejaba una ventana (TOCTOU): un enlace simbólico cambiado entre la
    validación y la operación apuntaría fuera del corral. Si no se puede
    resolver (bucle de enlaces, '\\0' en la ruta...), se devuelve tal cual y
    _outside_corral la rechaza, porque path_inside_roots tampoco podrá.

    Una ruta relativa se resuelve contra [paths].relative_base si está puesta
    (T10): sin ancla, "crea la carpeta fotos" cae dentro del repo de Crotolamo
    cuando corre como servicio (WorkingDirectory) y en otro sitio desde un
    shell. De fábrica no hay ancla (el CWD, como siempre): cambiarlo mueve
    dónde caen los archivos y, si el ancla es el home, añade una confirmación.
    El guard clasifica esa misma ruta (Guard.relative_base).
    """
    p = Path(path).expanduser()
    base = get_settings().relative_base
    if base is not None and not p.is_absolute():
        p = base / p
    try:
        return p.resolve()
    except (OSError, RuntimeError, ValueError):
        return p


def _outside_corral(p: Path) -> str | None:
    """Devuelve un mensaje de bloqueo si `p` está fuera del corral; si no, None.

    El corral acepta allowed_roots (zona libre) Y confirm_roots (zona con
    confirmación): si la ruta está en la segunda, el guard ya preguntó al patrón.
    """
    settings = get_settings()
    roots = list(settings.allowed_roots) + list(settings.confirm_roots)
    if not path_inside_roots(p, roots):
        return (f"La ruta '{p}' está fuera de las zonas permitidas, patrón. "
                "No salgo del corral.")
    return None


@tool
def read_file(path: str) -> str:
    """Lee un archivo de texto y devuelve su contenido (recortado si es enorme).

    Args:
        path: ruta del archivo a leer.
    """
    p = _resolve(path)
    if (blocked := _outside_corral(p)):
        return blocked
    if not p.exists():
        return f"No existe el archivo, patrón: {p}"
    if p.is_dir():
        return f"Eso es una carpeta, patrón, no un archivo: {p}"
    # Un FIFO sin escritor bloquea el open() para siempre (y con él, el hilo
    # del cerebro): solo se abren archivos normales.
    if not p.is_file():
        return f"Eso no es un archivo normal, patrón, no lo puedo leer: {p}"
    try:
        return read_text_capped(p)
    except OSError as error:
        return f"No pude leer {p}, patrón: {error}"


@tool(safe=False)
def write_file(path: str, content: str) -> str:
    """Escribe (crea o sobrescribe) un archivo de texto. Puede pisar contenido
    existente, así que pide confirmación.

    Args:
        path: ruta del archivo a escribir.
        content: contenido a guardar.
    """
    p = _resolve(path)
    if (blocked := _outside_corral(p)):
        return blocked
    # ANTES de abrir: write_text trunca el archivo y después codifica. Con un
    # sustituto suelto en el contenido (el "\ud83d" de un tool-call con el JSON
    # cortado) lanzaba UnicodeEncodeError con el archivo que iba a pisar ya vacío.
    # (Lo que no es texto lo rechaza write_text por su cuenta, sin abrir nada.)
    if isinstance(content, str):
        content = replace_surrogates(content)
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
    if (blocked := _outside_corral(p)):
        return blocked
    if not p.exists():
        return f"No existe la carpeta, patrón: {p}"
    if not p.is_dir():
        return f"Eso no es una carpeta, patrón: {p}"
    # iterdir()/is_dir() lanzan PermissionError (u otro OSError) sobre carpetas
    # sin permisos; sin esto la tool "reventaba" en vez de responder en personaje.
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"{'📁' if e.is_dir() else '📄'} {e.name}" for e in entries[:100]]
    except OSError as error:
        return f"No pude listar {p}, patrón: {error}"
    if not entries:
        return f"{p} está vacía, patrón."
    extra = f"\n...y {len(entries) - 100} más." if len(entries) > 100 else ""
    return f"Contenido de {p}, patrón:\n" + "\n".join(lines) + extra


@tool
def make_dir(path: str) -> str:
    """Crea una carpeta (y las intermedias que falten).

    Args:
        path: ruta de la carpeta a crear.
    """
    p = _resolve(path)
    if (blocked := _outside_corral(p)):
        return blocked
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
    if (blocked := _outside_corral(source) or _outside_corral(target)):
        return blocked
    if not source.exists():
        return f"No existe el origen, patrón: {source}"
    # Semántica de `mv`: si el destino es una carpeta que ya existe, se mueve
    # DENTRO de ella. Con rename a secas, "mueve informe.pdf a respaldo" fallaba
    # ("Is a directory") y una carpeta SUSTITUÍA a la carpeta destino si estaba
    # vacía. La ruta final pasa por el mismo cerrojo que las otras dos.
    if target.is_dir() and target != source:
        target = target / source.name
        if (blocked := _outside_corral(target)):
            return blocked
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
    if (blocked := _outside_corral(p)):
        return blocked
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
    # Poda: find bajaba a .venv/.git/node_modules de cada repo bajo Documentos
    # (decenas de miles de entradas) y sus coincidencias copaban las 20 rutas
    # que se muestran. La carpeta podada SÍ sale si es su nombre el que casa
    # ("busca node_modules"); lo que no se recorre es su contenido. El nombre
    # de la propia base no se poda: find evalúa la expresión también sobre ella.
    skip: list[str] = []
    for name in sorted(SKIP_DIRS - {base.name}):
        skip += ["-o", "-name", name] if skip else ["-name", name]
    match = ["-iname", f"*{query}*", "-print"]
    try:
        # errors="replace": un solo nombre con bytes que no son UTF-8 (un ZIP
        # hecho en Windows) y la decodificación estricta de text=True lanzaba
        # UnicodeDecodeError: la búsqueda entera "reventaba".
        result = subprocess.run(
            ["find", str(base), "(", *skip, ")", "-prune", *match, "-o", *match],
            text=True, errors="replace", capture_output=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "La búsqueda tardó demasiado, patrón."
    except OSError as error:
        # find ausente o no ejecutable: la búsqueda ni empezó, no es que "tardara".
        return f"No pude lanzar la búsqueda, patrón: {error}"
    lines = result.stdout.strip().splitlines()
    if not lines:
        return f"No encontré archivos con «{query}», patrón."
    shown = lines[:20]
    extra = f"\n...y {len(lines) - 20} más." if len(lines) > 20 else ""
    return "Encontré esto, patrón:\n" + "\n".join(shown) + extra


@tool
def create_note(title: str, content: str = "") -> str:
    """Crea una nota en markdown en la carpeta de notas del patrón.

    Args:
        title: título de la nota (se usa como nombre de archivo).
        content: cuerpo opcional de la nota.
    """
    notas_dir = get_settings().paths.get("notas", Path.home() / "Documentos" / "crotolamo_notas")

    clean = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
    clean = clean.replace(" ", "_") or "nota"
    # Al dictar, el modelo mete la frase entera en el título, y un nombre de
    # más de 255 bytes no se puede crear ("File name too long"). Se acorta el
    # NOMBRE; el título va completo dentro de la nota.
    clean = clean.encode("utf-8")[:_NOTE_STEM_MAX_BYTES].decode("utf-8", errors="ignore")
    path = notas_dir / f"{clean}.md"
    # Mismo cerrojo que las demás tools de escritura: la ruta final debe caer
    # dentro del corral (aunque salga de la config, no nos fiamos).
    if (blocked := _outside_corral(path)):
        return blocked

    # `or ""`: los modelos mandan `null` en un parámetro opcional y eso es "sin
    # cuerpo", no un None.strip() que revienta la tool.
    if not (content or "").strip():
        content = f"# {title}\n\n"
    # Como en write_file: codificable antes de abrir la nota.
    content = replace_surrogates(content)
    # Modo "x" (crear en exclusiva, sin seguir enlaces): con write_text, una
    # segunda nota con el mismo título —o con otro que se sanea igual ("ideas:
    # juego" e "ideas/ juego")— PISABA la anterior sin avisar, y esta tool no
    # pide confirmación. Si el nombre está cogido se prueba con _2, _3...
    target = path
    try:
        notas_dir.mkdir(parents=True, exist_ok=True)
        for n in range(2, 1000):
            try:
                with target.open("x", encoding="utf-8") as fh:
                    fh.write(content)
            except FileExistsError:
                target = notas_dir / f"{clean}_{n}.md"
                # El cerrojo de arriba valió para `path`, que podía ser un
                # enlace hacia dentro del corral desde una carpeta de fuera.
                if (blocked := _outside_corral(target)):
                    return blocked
                continue
            break
        else:
            return f"Ya hay demasiadas notas llamadas «{clean}», patrón. Dame otro título."
    except OSError as error:
        return f"No pude crear la nota, patrón: {error}"
    if target != path:
        return f"Nota creada, patrón: {target} (ya había una con ese título y no la pisé)"
    return f"Nota creada, patrón: {path}"
