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

import fnmatch
import os
import re
from pathlib import Path

from crotolamo.safety.paths import path_inside_roots
from crotolamo.settings import get_settings
from crotolamo.tools.base import (
    SKIP_DIRS,
    normalize_key,
    read_text_capped,
    replace_surrogates,
    tool,
)

# Tope del nombre de una nota, en BYTES: NAME_MAX es 255 en ext4/btrfs/xfs y hay
# que dejar sitio a ".md" y al sufijo "_N" de las notas con título repetido.
_NOTE_STEM_MAX_BYTES = 240

# Cuánto mira read_file para decidir si un archivo es binario (lo mismo que git).
_BINARY_SNIFF_BYTES = 8192

# Lo que separa palabras en un nombre de archivo: create_note guarda "Lista de
# compras" como Lista_de_compras.md, y otros usan '-'.
_NAME_SEPARATORS = str.maketrans("_-", "  ")

# Las letras que en un nombre pueden ir con tilde, diéresis, virgulilla o
# cedilla, y esas variantes: en el patrón de find cada una va como clase (ver
# _find_glob).
_ACCENTED = {
    "a": "áàâäãå", "e": "éèêë", "i": "íìîï", "o": "óòôöõ", "u": "úùûü",
    "n": "ñ", "c": "ç", "y": "ýÿ",
}


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
    p = _anchored(path)
    try:
        return p.resolve()
    except (OSError, RuntimeError, ValueError):
        return p


def _anchored(path: str) -> Path:
    """`path` con el ~ expandido y, si es relativa, anclada como en _resolve."""
    p = Path(path).expanduser()
    base = get_settings().relative_base
    if base is not None and not p.is_absolute():
        p = base / p
    return p


def _entry(path: str) -> Path:
    """La ENTRADA que nombra `path`: su carpeta resuelta y el último nombre tal
    cual, sin seguirlo si es un enlace simbólico.

    Borrar y mover actúan sobre lo que nombró el patrón. Con _resolve, "borra el
    acceso directo" borraba el archivo al que apuntaba (el guard pregunta sin
    decir ruta) y dejaba el enlace roto, que luego ya no se podía borrar ("No
    existe"); mover un enlace se llevaba el archivo real. unlink() y rename() no
    siguen el último nombre y la carpeta va resuelta una sola vez, igual que en
    _resolve, así que la ventana TOCTOU sigue cerrada. '.', '..' y la raíz no
    nombran una entrada: esas se resuelven enteras.
    """
    p = _anchored(path)
    if p.name in ("", ".", ".."):
        return _resolve(path)
    try:
        return p.parent.resolve() / p.name
    except (OSError, RuntimeError, ValueError):
        return p


def _outside_corral(p: Path, *, entry: bool = False) -> str | None:
    """Devuelve un mensaje de bloqueo si `p` está fuera del corral; si no, None.

    El corral acepta allowed_roots (zona libre) Y confirm_roots (zona con
    confirmación): si la ruta está en la segunda, el guard ya preguntó al patrón.

    Con entry=True, `p` viene de _entry: si es un enlace, lo que se va a tocar
    es el enlace, así que cuenta la carpeta que lo contiene y no adónde apunta.
    (os.path.islink, no Path.is_symlink: este no se traga un EACCES y la tool
    reventaría en vez de responder que la ruta está fuera.)
    """
    settings = get_settings()
    roots = list(settings.allowed_roots) + list(settings.confirm_roots)
    checked = p.parent if entry and os.path.islink(p) else p
    if not path_inside_roots(checked, roots):
        return (f"La ruta '{p}' está fuera de las zonas permitidas, patrón. "
                "No salgo del corral.")
    return None


@tool(parallel_safe=True)
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
        # Un PDF, un .docx o una sqlite decodificados "con reemplazo" eran
        # 20 000 caracteres de U+FFFD y NUL que el modelo no puede usar y que
        # viajaban en cada petición siguiente (50-120 KB de JSON). El criterio
        # de git: un NUL en los primeros 8 KB es binario. Un texto en Latin-1
        # no lleva NUL y se sigue leyendo; uno en UTF-16 sí (y tampoco se leía:
        # salía un \u0000 entre letra y letra).
        with p.open("rb") as fh:
            if b"\x00" in fh.read(_BINARY_SNIFF_BYTES):
                return f"Eso no es texto, patrón: parece un archivo binario, no lo puedo leer: {p}"
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


@tool(parallel_safe=True)
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
    # El origen es la entrada (_entry): mover un enlace mueve el enlace, como mv.
    source = _entry(src)
    target = _resolve(dest)
    if (blocked := _outside_corral(source, entry=True) or _outside_corral(target)):
        return blocked
    # lexists: un enlace roto también se puede mover.
    if not os.path.lexists(source):
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
    # La entrada (_entry): borrar un acceso directo quita el enlace, no el original.
    p = _entry(path)
    if (blocked := _outside_corral(p, entry=True)):
        return blocked
    # lexists: un enlace roto también es algo que borrar.
    if not os.path.lexists(p):
        return f"No existe, patrón, así que nada que borrar: {p}"
    # Un enlace a una carpeta es un enlace: se quita él y la carpeta queda.
    if not os.path.islink(p) and p.is_dir():
        return f"Es una carpeta, patrón. No borro carpetas con esta tool por seguridad: {p}"
    try:
        p.unlink()
    except OSError as error:
        return f"No pude borrar {p}, patrón: {error}"
    return f"Borrado, patrón: {p}"


@tool(parallel_safe=True)
def search_files(query: str) -> str:
    """Busca archivos por nombre dentro de Documentos del patrón (read-only).

    Args:
        query: parte del nombre del archivo a buscar.
    """
    import subprocess

    base = get_settings().paths.get("documentos", Path.home() / "Documentos")
    query = query.strip()
    # Plegada como los nombres (_fold_name): "lista de compras" tiene que dar
    # con Lista_de_compras.md, y "reunión" (la tilde la pone Whisper) con
    # reunion_lunes.txt. Una consulta que solo era '_' o '-' no busca nada.
    folded = _fold_name(query)
    if not folded:
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
    match = ["-iname", _find_glob(folded), "-print"]
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
    # find es el colador grueso; quien decide es esto, sobre los nombres
    # plegados igual que la consulta. Los comodines que mande el modelo
    # ("*.pdf") siguen valiendo.
    wanted = re.compile(fnmatch.translate(f"*{folded}*"))
    lines = [
        line for line in result.stdout.strip().splitlines()
        if wanted.match(_fold_name(os.path.basename(line)))
    ]
    if not lines:
        return f"No encontré archivos con «{query}», patrón."
    shown = lines[:20]
    extra = f"\n...y {len(lines) - 20} más." if len(lines) > 20 else ""
    return "Encontré esto, patrón:\n" + "\n".join(shown) + extra


def _find_glob(folded: str) -> str:
    """Patrón -iname para find que deja pasar todo nombre cuyo plegado casa con
    `folded` (la consulta ya plegada), y alguno más: el filtro de Python decide.

    Cada letra que puede llevar tilde va como clase con sus variantes y un '*'
    detrás, que se traga la tilde suelta de un nombre en NFD (o + U+0301) y, en
    el locale C, donde find compara bytes, el segundo byte de la 'ó' en UTF-8 (la
    clase casa con el primero). Así sirve en cualquier locale y con nombres que
    no son UTF-8. Lo que no es letra o cifra ASCII (un separador, un comodín, un
    carácter de otro alfabeto) va como '*'.
    """
    parts: list[str] = []
    for char in folded:
        if char in _ACCENTED:
            variants = char + _ACCENTED[char]
            parts.append(f"[{variants}{variants.upper()}]*")
        elif char.isascii() and char.isalnum():
            parts.append(char)
        else:
            parts.append("*")
    return re.sub(r"\*+", "*", "*" + "".join(parts) + "*")


def _fold_name(text: str) -> str:
    """Forma comparable de un nombre de archivo: sin mayúsculas ni tildes (también
    las de un nombre en NFD) y con '_', '-' y los espacios como un solo espacio."""
    # Corre sobre cada nombre que deja pasar find (miles, con una consulta corta
    # en un Documentos grande): quitar tildes es lo caro, y en un nombre ASCII no
    # hay ninguna que quitar.
    if not text.isascii():
        text = normalize_key(text)
    return " ".join(text.lower().translate(_NAME_SEPARATORS).split())


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
