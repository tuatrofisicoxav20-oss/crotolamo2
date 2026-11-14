"""Tools de escritorio: abrir apps, carpetas y URLs. Migrado de C1::skills.py.

Lo que cambia vs C1: ya no hay parse_* con regex. El LLM llama estas funciones
tipadas vía tool-calling. La lógica de lanzamiento (run_detached, APP_COMMANDS,
funny_line) se conserva.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import threading
from pathlib import Path

from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import normalize_key, run_cmd, tool

log = get_logger("desktop")

HOME = Path.home()

# Carpetas estándar XDG, derivadas del home actual (no hardcodeado a un usuario).
# Son los defaults y el catálogo de nombres: la ruta de cada una la puede pisar
# [paths] de la config (ver _folders).
FOLDERS: dict[str, Path] = {
    "documentos": HOME / "Documentos",
    "descargas": HOME / "Descargas",
    "escritorio": HOME / "Escritorio",
    "desktop": HOME / "Desktop",
    "imagenes": HOME / "Imágenes",
    "musica": HOME / "Música",
    "videos": HOME / "Vídeos",
}

COMMON_SITES: dict[str, str] = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
    "gmail": "https://mail.google.com",
    "drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "classroom": "https://classroom.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "wikipedia": "https://www.wikipedia.org",
    "spotify": "https://open.spotify.com",
    "ollama": "https://ollama.com",
    "arxiv": "https://arxiv.org",
}

# Id de flatpak del navegador preferido del patrón (también abre las URLs).
_OPERA_GX_FLATPAK = "com.opera.opera-gx"

APP_COMMANDS: dict[str, list[list[str]]] = {
    "opera": [["flatpak", "run", _OPERA_GX_FLATPAK]],
    "opera gx": [["flatpak", "run", _OPERA_GX_FLATPAK]],
    "archivos": [["nautilus"]],
    "nautilus": [["nautilus"]],
    "blender": [["blender"]],
    "vscode": [["code"]],
    "visual studio code": [["code"]],
    "geany": [["geany"]],
    "libreoffice": [["libreoffice"]],
    "calculadora": [["gnome-calculator"]],
}

_FUNNY_OPEN = [
    "Abriendo eso, patrón. Internet vuelve a recibir nuestras malas decisiones.",
    "Va, patrón. Abriendo antes de que cambie de opinión el universo.",
    "Hecho, patrón. Otra pestaña más para alimentar al monstruo.",
    "Abriendo, patrón. Fedora sigue vivo, contra todo pronóstico.",
    "Listo, patrón. La civilización avanza medio centímetro.",
]


def funny_line() -> str:
    return random.choice(_FUNNY_OPEN)


# El esquema de una URL no distingue mayúsculas (RFC 3986 §3.1): "HTTPS://x.com"
# ya es una URL, no un dominio al que anteponerle otro https://.
_HTTP_SCHEME = re.compile(r"https?://", re.IGNORECASE)


def resolve_target(target: str) -> tuple[str, dict]:
    """Clasifica un destino genérico en (tipo, payload): url/app/folder/search.

    Punto único de resolución para los atajos aprendidos (shortcuts.py) y
    cualquier otro consumidor: aquí viven APP_COMMANDS/COMMON_SITES/FOLDERS,
    así que la clasificación se decide aquí y no leyendo los dicts desde fuera.
    """
    # strip ANTES de clasificar (como hacía C1): con un espacio delante la URL
    # no pasaba el startswith y se guardaba como búsqueda de sí misma.
    target = target.strip()
    key = normalize_key(target)
    if key in COMMON_SITES:
        return "url", {"value": COMMON_SITES[key]}
    # "terminal" no vive en APP_COMMANDS (lo resuelve _detect_terminal en
    # open_app), pero sigue siendo una app válida como destino de atajo. Igual
    # las de [apps]: open_app y el fastpath las abren, así que un atajo a una
    # de ellas es una app y no una búsqueda en Google de su nombre.
    if key in APP_COMMANDS or key == "terminal" or key in _config_apps():
        return "app", {"value": key}
    if key in FOLDERS:
        return "folder", {"value": key}
    if _HTTP_SCHEME.match(target):
        return "url", {"value": target}
    return "search", {"engine": "google", "query": target}


# `systemd-run --scope` registra un scope propio en el systemd del usuario y
# hace exec del comando dentro de él: mismo entorno y mismo PID, otro cgroup.
_SCOPE_CMD = ["systemd-run", "--user", "--scope", "--collect", "--quiet", "--"]
_CGROUP = Path("/proc/self/cgroup")
# Lo que decidió _scope_prefix(); None = aún sin decidir (una vez por proceso).
_scope_cache: list[str] | None = None


def _scope_prefix() -> list[str]:
    """Prefijo que saca del cgroup del servicio a lo que lanza run_detached.

    start_new_session cambia la sesión, NO el cgroup: corriendo como
    crotolamo.service, Blender, VS Code o el terminal abiertos por voz quedan
    dentro de la unidad y systemd (KillMode=control-group, el default) los mata
    al parar o reiniciar el servicio, con lo que tengan sin guardar. Fuera de
    una unidad .service no hay nada que arreglar y devuelve [].

    Solo se usa si una sonda (`... -- true`) demuestra que aquí funciona: un
    systemd-run que no alcanza al systemd del usuario sale con error ANTES de
    lanzar nada y Popen no se entera (el mismo problema que `flatpak run`).
    """
    global _scope_cache
    if _scope_cache is None:
        prefix: list[str] = []
        try:
            lines = _CGROUP.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(line.endswith(".service") for line in lines) and shutil.which(_SCOPE_CMD[0]):
                if run_cmd([*_SCOPE_CMD, "true"], timeout=3).returncode == 0:
                    prefix = _SCOPE_CMD
        except (OSError, subprocess.SubprocessError) as error:
            log.debug("sin scope de systemd, lanzo como siempre: %s", error)
        _scope_cache = prefix
    return _scope_cache


def run_detached(args: list[str]) -> None:
    prefix = _scope_prefix()
    # Con el binario ausente no se envuelve: Popen debe fallar con el OSError
    # de siempre (open_app lo usa para decir "No pude abrir"). Envuelto,
    # systemd-run arrancaría bien y el fallo del exec se perdería en DEVNULL.
    if prefix and not shutil.which(args[0]):
        prefix = []
    proc = subprocess.Popen(
        [*prefix, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Nadie más espera a este hijo: sin wait() queda <defunct> al terminar
    # (xdg-open sale enseguida) hasta que OTRO subprocess dispara la limpieza
    # interna de subprocess, y Popen.__del__ avisa con ResourceWarning.
    threading.Thread(target=proc.wait, name="crotolamo-reap", daemon=True).start()


# M3: el patrón usa Hyprland (Wayland), no GNOME. Detectamos el terminal instalado
# en vez de asumir gnome-terminal. Orden: terminales de tiling primero.
_TERMINALS = ("kitty", "foot", "alacritty", "wezterm", "gnome-terminal", "konsole", "xterm")


def _detect_terminal() -> list[str] | None:
    for term in _TERMINALS:
        if shutil.which(term):
            return [term]
    return None


def terminal_exec(command: str) -> list[str] | None:
    """Argv para correr `bash -lc command` en el terminal del patrón.

    La sintaxis para ejecutar un comando varía entre terminales (--, -e, start --);
    aquí se normaliza. Devuelve None si no hay terminal cableado ni instalado.
    """
    # Misma precedencia que open_app: [apps].terminal > terminal detectado. Sin
    # esto, "Cablea uno en [apps]" (lo que contestan open_app y launch_project)
    # arreglaba "abre terminal" pero launch_project seguía sin terminal.
    term = _config_apps().get("terminal") or _detect_terminal()
    if not term:
        return None
    # La sintaxis sale del nombre del binario: lo cableado puede traer ruta y
    # argumentos propios (["/usr/bin/foot", "--app-id=x"]), que se conservan.
    name = Path(term[0]).name
    inner = ["bash", "-lc", command]
    if name in ("gnome-terminal", "ptyxis"):
        return [*term, "--", *inner]
    if name == "wezterm":
        return [*term, "start", "--", *inner]
    if name in ("kitty", "foot"):
        # kitty, foot: ejecutan los args directamente.
        return [*term, *inner]
    # alacritty, konsole, xterm... y cualquier otro cableado en [apps]: -e es
    # la convención de xterm, la que más terminales respetan.
    return [*term, "-e", *inner]


def _config_apps() -> dict[str, list[str]]:
    """Apps definidas por el patrón en [apps] de la config (clave -> lista de args).

    Permite a Emiliano cablear sus apps de Hyprland sin tocar código (M3). Se
    superponen sobre APP_COMMANDS (defaults).
    """
    try:
        from crotolamo.settings import get_settings

        raw = get_settings().raw.get("apps", {})
    except Exception as error:  # noqa: BLE001 — sin config, valen los defaults
        log.debug("no pude leer [apps] de la config: %s", error)
        return {}
    apps: dict[str, list[str]] = {}
    for name, cmd in raw.items():
        if isinstance(cmd, str):
            apps[normalize_key(name)] = cmd.split()
        elif isinstance(cmd, list):
            apps[normalize_key(name)] = [str(x) for x in cmd]
    return apps


def _folders() -> dict[str, Path]:
    """FOLDERS con las rutas de [paths] de la config superpuestas.

    El toml documenta [paths].documentos y [paths].descargas, pero FOLDERS se
    arma solo con el home: quien apuntaba documentos a ~/Documents veía a
    search_files buscar ahí y a open_folder decir que la carpeta no existía.
    Solo se pisan nombres que YA están en FOLDERS: [paths] también trae rutas
    que no son carpetas para abrir (db, voces) y el catálogo no debe cambiar.
    """
    try:
        from crotolamo.settings import get_settings

        paths = get_settings().paths
    except Exception as error:  # noqa: BLE001 — sin config, valen los defaults
        log.debug("no pude leer [paths] de la config: %s", error)
        return FOLDERS
    folders = dict(FOLDERS)
    for name, path in paths.items():
        key = normalize_key(name)
        # Una ruta relativa (o "" = ".") se resolvería contra el CWD del
        # proceso: "abre descargas" abriría la carpeta desde donde se lanzó.
        if key in folders and path.is_absolute():
            folders[key] = path
    return folders


def _opera_gx_flatpak_available() -> bool:
    """True solo si hay `flatpak` Y Opera GX está instalada como flatpak.

    Popen NO falla cuando `flatpak run` no encuentra la app: el proceso arranca,
    escribe el error a DEVNULL y sale con 1, así que un fallback a xdg-open
    "por excepción" era código muerto y la URL no se abría en ninguna parte.
    Hay que preguntar ANTES con `flatpak info` (timeout corto: es local).
    """
    if not shutil.which("flatpak"):
        return False
    try:
        result = run_cmd(["flatpak", "info", _OPERA_GX_FLATPAK], timeout=3)
    except (OSError, subprocess.SubprocessError) as error:
        log.debug("no pude consultar flatpak info: %s", error)
        return False
    return result.returncode == 0


def _flatpak_app_missing(args: list[str]) -> bool:
    """True si `args` es `flatpak run <id>` y flatpak no tiene instalado ese id.

    El mismo problema que resuelve _opera_gx_flatpak_available para las URLs:
    con una app ausente `flatpak run` arranca y sale con 1 a espaldas de Popen,
    y open_app contestaba "Abrí" sin abrir nada (el [apps] de fábrica trae
    opera y spotify como flatpaks).

    Solo se da por ausente lo que es seguro: un id pelón, justo tras `run`,
    que flatpak NO lista en ninguna instalación. Con opciones delante, un ref
    con rama (`id//rama`) o sin poder preguntar (timeout, permisos, un flatpak
    que no entiende --columns) se lanza como siempre: dar por ausente una app
    que sí está sería peor que el fallo que esto arregla.
    """
    if len(args) < 3 or Path(args[0]).name != "flatpak" or args[1] != "run":
        return False
    app_id = args[2]
    if app_id.startswith("-") or "/" in app_id:
        return False
    try:
        # --all: que no se quede fuera ni lo que `list` esconde por defecto.
        result = run_cmd([args[0], "list", "--all", "--columns=application"], timeout=3)
    except (OSError, subprocess.SubprocessError) as error:
        log.debug("no pude consultar flatpak list: %s", error)
        return False
    return result.returncode == 0 and app_id not in result.stdout.split()


def _open_local_browser(url: str) -> str:
    if _opera_gx_flatpak_available():
        try:
            run_detached(["flatpak", "run", _OPERA_GX_FLATPAK, url])
            return f"{funny_line()}\nAbrí esta pestaña, patrón: {url}"
        except (OSError, subprocess.SubprocessError) as error:
            log.debug("flatpak/opera falló, caigo a xdg-open: %s", error)
    # Sin xdg-utils (o sin poder crear el proceso) Popen lanza OSError: sin
    # atraparlo, la tool "reventaba" con el mensaje genérico del registry.
    try:
        run_detached(["xdg-open", url])
    except (OSError, subprocess.SubprocessError) as error:
        return f"No pude abrir {url}, patrón: {error}"
    return f"{funny_line()}\nAbrí esta URL con el navegador por defecto, patrón: {url}"


@tool
def open_url(url: str) -> str:
    """Abre una URL en el navegador. Si no parece URL, la trata como búsqueda en Google.

    Args:
        url: la dirección o término a abrir.
    """
    url = url.strip()
    if not url:
        return "No recibí una URL, patrón."

    scheme = _HTTP_SCHEME.match(url)
    if scheme:
        # Solo el esquema pasa a minúsculas (su forma canónica): así lo que
        # recibe el lanzador no depende de que sepa leer "HTTPS://". La ruta y
        # la query sí distinguen mayúsculas y no se tocan.
        url = scheme.group().lower() + url[scheme.end():]
    elif "." in url and " " not in url:
        url = "https://" + url
    else:
        from urllib.parse import quote_plus

        url = f"https://www.google.com/search?q={quote_plus(url)}"

    return _open_local_browser(url)


@tool
def open_app(name: str) -> str:
    """Abre una aplicación instalada por su nombre (opera, terminal, vscode, blender, etc.).

    Args:
        name: nombre de la app.
    """
    key = normalize_key(name)
    config_apps = _config_apps()

    # Precedencia: config del patrón > terminal detectado > defaults APP_COMMANDS.
    if key in config_apps:
        candidates = [config_apps[key]]
    elif key == "terminal":
        term = _detect_terminal()
        if term is None:
            return "No encontré ningún terminal instalado, patrón. Cablea uno en [apps]."
        candidates = [term]
    elif key in APP_COMMANDS:
        candidates = APP_COMMANDS[key]
    else:
        disponibles = ", ".join(sorted(set(APP_COMMANDS) | set(config_apps)))
        return f"No tengo registrada la app '{name}', patrón. Conozco: {disponibles}."

    for args in candidates:
        # Un `flatpak run` de una app sin instalar no hace fallar a Popen: se
        # pregunta antes, para no contestar "Abrí" por algo que no se abrió.
        if _flatpak_app_missing(args):
            log.debug("flatpak no tiene instalada %s: no la lanzo", args[2])
            continue
        try:
            run_detached(args)
            return f"{funny_line()}\nAbrí {name}, patrón."
        except (OSError, subprocess.SubprocessError) as error:
            log.debug("no pude lanzar %s: %s", args, error)
            continue
    return f"No pude abrir {name}, patrón."


@tool
def open_folder(name: str) -> str:
    """Abre una carpeta del sistema en el explorador (documentos, descargas, escritorio, etc.).

    Args:
        name: nombre de la carpeta.
    """
    key = normalize_key(name)
    if key not in FOLDERS:
        disponibles = ", ".join(sorted(FOLDERS))
        return f"No tengo registrada la carpeta '{name}', patrón. Conozco: {disponibles}."

    folder = _folders()[key]
    if key == "escritorio" and not folder.exists():
        folder = HOME / "Desktop"

    if not folder.exists():
        return f"No encontré la carpeta {folder}, patrón."

    # Mismo trato que en _open_local_browser: sin xdg-open, en personaje.
    try:
        run_detached(["xdg-open", str(folder)])
    except (OSError, subprocess.SubprocessError) as error:
        return f"No pude abrir {folder}, patrón: {error}"
    return f"{funny_line()}\nAbrí {folder}, patrón."


# Combos de teclas conocidos -> secuencia de keycodes Linux para ydotool
# (formato keycode:estado, 1=presionar 0=soltar).
_HOTKEYS: dict[str, list[str]] = {
    "alt+tab": ["56:1", "15:1", "15:0", "56:0"],
    "super": ["125:1", "125:0"],
    "ctrl+alt+t": ["29:1", "56:1", "20:1", "20:0", "56:0", "29:0"],
    "alt+f4": ["56:1", "62:1", "62:0", "56:0"],
}


@tool(safe=False)
def send_hotkey(combo: str) -> str:
    """Envía un atajo de teclado al escritorio vía ydotool (control de ventanas).

    Combos soportados: alt+tab, super, ctrl+alt+t, alt+f4. Acción insegura
    (afecta a todo el escritorio): pide confirmación.

    Args:
        combo: el atajo a enviar.
    """
    key = normalize_key(combo).replace(" ", "")
    if key not in _HOTKEYS:
        return f"No conozco el atajo '{combo}', patrón. Tengo: {', '.join(_HOTKEYS)}."
    if not shutil.which("ydotool"):
        return "No tengo ydotool, patrón. Instálalo con `sudo dnf install ydotool`."
    try:
        subprocess.run(["ydotool", "key", *_HOTKEYS[key]], check=True,
                       capture_output=True, text=True, timeout=5)
    except subprocess.CalledProcessError:
        return ("Falló ydotool, patrón. ¿Está corriendo el demonio ydotoold? "
                "(`systemctl --user start ydotoold`).")
    except (OSError, subprocess.TimeoutExpired):
        return "No pude enviar el atajo, patrón."
    return f"Envié {combo}, patrón."
