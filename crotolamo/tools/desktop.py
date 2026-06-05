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
import unicodedata
from pathlib import Path

from crotolamo.tools.base import tool

HOME = Path.home()

# Carpetas estándar XDG, derivadas del home actual (no hardcodeado a un usuario).
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

APP_COMMANDS: dict[str, list[list[str]]] = {
    "opera": [["flatpak", "run", "com.opera.opera-gx"]],
    "opera gx": [["flatpak", "run", "com.opera.opera-gx"]],
    "terminal": [["gnome-terminal"]],
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


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_key(text: str) -> str:
    """Normaliza una clave de app/carpeta/sitio para emparejar (acentos, espacios)."""
    return re.sub(r"\s+", " ", _strip_accents(text.lower())).strip()


def run_detached(args: list[str]) -> None:
    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _open_local_browser(url: str) -> str:
    if shutil.which("flatpak"):
        try:
            run_detached(["flatpak", "run", "com.opera.opera-gx", url])
            return f"{funny_line()}\nAbrí esta pestaña, patrón: {url}"
        except Exception:
            pass
    run_detached(["xdg-open", url])
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

    if not re.match(r"^https?://", url):
        if "." in url and " " not in url:
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
    if key not in APP_COMMANDS:
        disponibles = ", ".join(sorted(APP_COMMANDS))
        return f"No tengo registrada la app '{name}', patrón. Conozco: {disponibles}."

    for args in APP_COMMANDS[key]:
        try:
            run_detached(args)
            return f"{funny_line()}\nAbrí {name}, patrón."
        except Exception:
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

    folder = FOLDERS[key]
    if key == "escritorio" and not folder.exists():
        folder = HOME / "Desktop"

    if not folder.exists():
        return f"No encontré la carpeta {folder}, patrón."

    run_detached(["xdg-open", str(folder)])
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
