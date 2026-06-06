import json
import random
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus


HOME = Path.home()
BASE_DIR = HOME / "Documentos" / "chapi_assistant"
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

LEARNED_COMMANDS_PATH = CONFIG_DIR / "learned_commands.json"
CHAPI_INBOX = BASE_DIR / "chapi_inbox"
CHAPI_INBOX.mkdir(parents=True, exist_ok=True)


PROJECTS = {
    "tletl": HOME / "Documentos" / "tletl_control_v4_1_ai_integrado",
    "huevonitis": HOME / "Documentos" / "huevonitis version 2.1",
    "crotolamo": HOME / "Documentos" / "chapi_assistant",
    "chapi": HOME / "Documentos" / "chapi_assistant",
}

FOLDERS = {
    "documentos": HOME / "Documentos",
    "descargas": HOME / "Descargas",
    "escritorio": HOME / "Escritorio",
    "desktop": HOME / "Desktop",
    "imagenes": HOME / "Imágenes",
    "imágenes": HOME / "Imágenes",
    "musica": HOME / "Música",
    "música": HOME / "Música",
    "videos": HOME / "Vídeos",
    "vídeos": HOME / "Vídeos",
}

COMMON_SITES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
    "chapi": "https://chatgpt.com",
    "gmail": "https://mail.google.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "google docs": "https://docs.google.com",
    "sheets": "https://sheets.google.com",
    "google sheets": "https://sheets.google.com",
    "classroom": "https://classroom.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "wikipedia": "https://www.wikipedia.org",
    "stack overflow": "https://stackoverflow.com",
    "ollama": "https://ollama.com",
    "hugging face": "https://huggingface.co",
    "spotify": "https://open.spotify.com",
    "nasa": "https://www.nasa.gov",
    "arxiv": "https://arxiv.org",
}

SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q={query}",
    "youtube": "https://www.youtube.com/results?search_query={query}",
    "github": "https://github.com/search?q={query}",
    "wikipedia": "https://es.wikipedia.org/wiki/Special:Search?search={query}",
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "spotify": "https://open.spotify.com/search/{query}",
    "stack overflow": "https://stackoverflow.com/search?q={query}",
    "maps": "https://www.google.com/maps/search/{query}",
    "imagenes": "https://www.google.com/search?tbm=isch&q={query}",
    "imágenes": "https://www.google.com/search?tbm=isch&q={query}",
    "traductor": "https://translate.google.com/?sl=auto&tl=es&text={query}&op=translate",
    "arxiv": "https://arxiv.org/search/?query={query}&searchtype=all",
}

APP_COMMANDS = {
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

BLOCKED_SEARCH_TERMS = [
    "porn",
    "porno",
    "xxx",
    "xvideos",
    "pornhub",
    "hackear contraseña",
    "robar contraseña",
    "malware para",
    "ransomware",
    "exploit para romper",
    "bomba",
    "arma casera",
]


FUNNY_OPENING_LINES = [
    "Abriendo eso, patrón. Internet vuelve a recibir nuestras malas decisiones.",
    "Va, patrón. Abriendo la página antes de que cambie de opinión el universo.",
    "Hecho, patrón. Otra pestaña más para alimentar al monstruo.",
    "Abriendo, patrón. Fedora sigue vivo, contra todo pronóstico.",
    "Listo, patrón. La civilización avanza medio centímetro.",
]

FUNNY_SEARCH_LINES = [
    "Buscando eso, patrón. A ver si internet se comporta por una vez.",
    "Va, patrón. Invocando al oráculo lleno de anuncios.",
    "Buscando, patrón. Prometo ignorar las páginas que parecen hechas en 2007.",
    "Hecho, patrón. Una búsqueda más para el archivo mental del caos.",
    "Buscando eso. Que los algoritmos tengan piedad.",
]


def remove_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def base_normalize(text: str) -> str:
    text = text.strip()
    text = remove_accents(text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_learned_commands() -> dict:
    if not LEARNED_COMMANDS_PATH.exists():
        return {}

    try:
        return json.loads(LEARNED_COMMANDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_learned_commands(data: dict) -> None:
    LEARNED_COMMANDS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_blocked_query(text: str) -> bool:
    lower = base_normalize(text)
    return any(term in lower for term in BLOCKED_SEARCH_TERMS)


def run_detached(args: list[str]) -> None:
    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def run_in_terminal(command: str, title: str = "Crotolamo") -> None:
    terminal_cmd = [
        "gnome-terminal",
        "--title",
        title,
        "--",
        "bash",
        "-lc",
        f"{command}; echo; echo 'Presiona Enter para cerrar...'; read",
    ]
    run_detached(terminal_cmd)


def funny_line(kind: str = "open") -> str:
    if kind == "search":
        return random.choice(FUNNY_SEARCH_LINES)

    return random.choice(FUNNY_OPENING_LINES)


def clean_activation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    lower = base_normalize(text)

    wake_words = [
        "crotolamo",
        "coto y amo",
        "coto lamo",
        "croto lamo",
        "corto lamo",
        "control amo",
        "oye crotolamo",
        "oye coto y amo",
    ]

    for wake in wake_words:
        wake_norm = base_normalize(wake)
        if lower.startswith(wake_norm):
            words_to_remove = len(wake_norm.split())
            return " ".join(text.split()[words_to_remove:]).strip(" ,.:;-")

    return text.strip()


def normalize(text: str) -> str:
    text = clean_activation(text)
    lower = base_normalize(text)

    replacements = {
        "habre ": "abre ",
        "abrele ": "abre ",
        "abreme ": "abre ",
        "abrir ": "abre ",
        "buscar ": "busca ",
        "buscame ": "busca ",
        "buscalo ": "busca ",
        "googlealo ": "googlea ",
        "opea": "opera",
        "opela": "opera",
        "nopela": "en opera",
        "you tube": "youtube",
        "yutub": "youtube",
        "yutube": "youtube",
        "git hub": "github",
        "guit hub": "github",
        "gijub": "github",
        "spotifi": "spotify",
        "espoti": "spotify",
        "espoti fai": "spotify",
        "spoti": "spotify",
        "pestana": "pestana",
        "pestanas": "pestanas",
        "pagina": "pestana",
        "paginas": "pestanas",
        "una pagina": "una pestana",
        "una nueva pagina": "una nueva pestana",
        "musica": "música",
        "tlet": "tletl",
        "tle tl": "tletl",
        "huevonitis": "huevonitis",
        "guevonitis": "huevonitis",
    }

    for wrong, right in replacements.items():
        lower = lower.replace(wrong, right)

    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def open_url(url: str) -> str:
    url = url.strip()

    if not url:
        return "No recibí una URL, patrón."

    if not re.match(r"^https?://", url):
        if "." in url and " " not in url:
            url = "https://" + url
        else:
            url = build_search_url("google", url)

    if shutil.which("flatpak"):
        try:
            run_detached(["flatpak", "run", "com.opera.opera-gx", url])
            return f"{funny_line('open')}\nAbrí esta pestaña, patrón: {url}"
        except Exception:
            pass

    run_detached(["xdg-open", url])
    return f"{funny_line('open')}\nAbrí esta URL con el navegador predeterminado, patrón: {url}"


def build_search_url(engine: str, query: str) -> str:
    engine = normalize(engine)
    query = query.strip()

    if engine not in SEARCH_ENGINES:
        engine = "google"

    # Spotify prefiere path encoded, no query normal.
    if engine == "spotify":
        return SEARCH_ENGINES[engine].format(query=quote_plus(query).replace("+", "%20"))

    return SEARCH_ENGINES[engine].format(query=quote_plus(query))


def search_web(query: str, engine: str = "google") -> str:
    query = query.strip()

    if not query:
        return "Necesito algo que buscar, patrón."

    if is_blocked_query(query):
        return "No voy a abrir ni buscar eso, patrón. Huele a desastre."

    url = build_search_url(engine, query)
    opened = open_url(url)
    return f"{funny_line('search')}\n{opened}"


def open_app(name: str) -> str:
    key = normalize(name)

    if key not in APP_COMMANDS:
        return f"No tengo registrada la app '{name}', patrón."

    for args in APP_COMMANDS[key]:
        try:
            run_detached(args)
            return f"{funny_line('open')}\nAbrí {name}, patrón."
        except Exception:
            continue

    return f"No pude abrir {name}, patrón."


def open_path(path: Path) -> str:
    expanded = path.expanduser()

    if not expanded.exists():
        return f"No encontré la ruta: {expanded}"

    run_detached(["xdg-open", str(expanded)])
    return f"{funny_line('open')}\nAbrí {expanded}, patrón."


def open_project(name: str) -> str:
    key = normalize(name)

    if key not in PROJECTS:
        return f"No tengo registrado el proyecto '{name}', patrón."

    return open_path(PROJECTS[key])


def open_folder(name: str) -> str:
    key = normalize(name)

    if key not in FOLDERS:
        return f"No tengo registrada la carpeta '{name}', patrón."

    folder = FOLDERS[key]

    if key == "escritorio" and not folder.exists():
        folder = HOME / "Desktop"

    return open_path(folder)


def find_project_launchers(project_path: Path) -> list[Path]:
    if not project_path.exists():
        return []

    patterns = [
        "launch*.sh",
        "*launcher*.sh",
        "run*.sh",
        "start*.sh",
        "*.desktop",
    ]

    found = []

    for pattern in patterns:
        found.extend(project_path.rglob(pattern))

    # Evitar cosas dentro de venv, backups enormes, etc.
    filtered = []

    for path in found:
        parts = set(path.parts)
        if ".venv" in parts or "venv" in parts or "__pycache__" in parts:
            continue
        filtered.append(path)

    return filtered[:10]


def launch_project(name: str) -> str:
    key = normalize(name)

    if key not in PROJECTS:
        return f"No tengo registrado el proyecto '{name}', patrón."

    project = PROJECTS[key]

    if not project.exists():
        return f"No encontré el proyecto {name}: {project}"

    launchers = find_project_launchers(project)

    if not launchers:
        run_in_terminal(f'cd "{project}" && ls -la', f"Crotolamo - {name}")
        return (
            f"No encontré launcher claro para {name}, patrón. "
            f"Te abrí una terminal en el proyecto para revisar."
        )

    launcher = launchers[0]

    if launcher.suffix == ".desktop":
        run_detached(["gtk-launch", launcher.stem])
        return f"Lancé {name} usando {launcher.name}, patrón."

    launcher.chmod(launcher.stat().st_mode | 0o111)
    run_in_terminal(f'cd "{project}" && "{launcher}"', f"Crotolamo - {name}")

    return f"Lancé {name} con {launcher.name}, patrón. Que el código tenga piedad."


def analyze_project(name: str) -> str:
    key = normalize(name)

    if key not in PROJECTS:
        return f"No tengo registrado el proyecto '{name}', patrón."

    project = PROJECTS[key]

    if not project.exists():
        return f"No encontré el proyecto {name}: {project}"

    py_files = []
    sh_files = []

    for path in project.rglob("*"):
        if any(skip in path.parts for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue

        if path.suffix == ".py":
            py_files.append(path)
        elif path.suffix == ".sh":
            sh_files.append(path)

    py_files = py_files[:20]
    sh_files = sh_files[:15]

    lines = [
        f"Análisis rápido de {name}, patrón:",
        f"Ruta: {project}",
        f"Archivos Python encontrados: {len(py_files)} mostrados máximo 20",
    ]

    for file in py_files:
        lines.append(f"PY: {file.relative_to(project)}")

    if sh_files:
        lines.append(f"Launchers/scripts shell encontrados: {len(sh_files)} mostrados máximo 15")
        for file in sh_files:
            lines.append(f"SH: {file.relative_to(project)}")

    return "\n".join(lines)


def search_files(query: str, base: Path | None = None) -> str:
    base = base or (HOME / "Documentos")
    query = query.strip()

    if not query:
        return "Necesito algo que buscar, patrón."

    if not base.exists():
        return f"No existe la carpeta base: {base}"

    cmd = ["find", str(base), "-iname", f"*{query}*", "-print"]

    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "La búsqueda tardó demasiado, patrón. No voy a quedarme excavando como topo."

    lines = result.stdout.strip().splitlines()

    if not lines:
        return f"No encontré archivos con '{query}' en {base}, patrón."

    limited = lines[:15]
    output = "\n".join(limited)

    if len(lines) > 15:
        output += f"\n...y {len(lines) - 15} resultados más."

    return f"Encontré esto, patrón:\n{output}"


def create_note(title: str, content: str = "") -> str:
    notes_dir = HOME / "Documentos" / "crotolamo_notas"
    notes_dir.mkdir(parents=True, exist_ok=True)

    clean_title = "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_"
        for c in title
    ).strip().replace(" ", "_")

    if not clean_title:
        clean_title = "nota"

    path = notes_dir / f"{clean_title}.md"

    if not content.strip():
        content = f"# {title}\n\n"

    path.write_text(content, encoding="utf-8")
    return f"Nota creada, patrón: {path}"


def create_chapi_request(task: str) -> str:
    task = task.strip()

    if not task:
        return "Necesito que me digas qué le voy a pedir a Chapi, patrón."

    path = CHAPI_INBOX / "latest_request.md"

    content = f"""ORIGEN: CROTOLAMO
PATRÓN: Emiliano / Caos Orbital
MODO: petición generada por voz/local
TAREA:

{task}

INSTRUCCIÓN PARA CHAPI:
Responde sabiendo que esta petición viene de Crotolamo, no escrita directamente por el patrón.
"""

    path.write_text(content, encoding="utf-8")

    open_url("https://chatgpt.com")

    return (
        f"Preparé una petición para Chapi, patrón: {path}\n"
        "También abrí ChatGPT. Copia el contenido de ese archivo y pégalo en el chat."
    )


def learn_shortcut(alias: str, target: str) -> str:
    alias_key = normalize(alias)
    target = target.strip()

    if not alias_key or not target:
        return "No pude aprender ese atajo, patrón. Me faltó alias o destino."

    learned = load_learned_commands()

    action = None

    norm_target = normalize(target)

    if norm_target in COMMON_SITES:
        action = {"type": "url", "value": COMMON_SITES[norm_target]}
    elif norm_target in APP_COMMANDS:
        action = {"type": "app", "value": norm_target}
    elif norm_target in PROJECTS:
        action = {"type": "project", "value": norm_target}
    elif norm_target in FOLDERS:
        action = {"type": "folder", "value": norm_target}
    elif target.startswith("http://") or target.startswith("https://"):
        action = {"type": "url", "value": target}
    else:
        action = {"type": "search", "engine": "google", "query": target}

    learned[alias_key] = action
    save_learned_commands(learned)

    return f"Aprendí el atajo '{alias}', patrón. Ya puedo usarlo después."


def run_learned_shortcut(text: str) -> str | None:
    key = normalize(text)
    learned = load_learned_commands()

    if key not in learned:
        return None

    action = learned[key]
    action_type = action.get("type")

    if action_type == "url":
        return open_url(action.get("value", ""))

    if action_type == "app":
        return open_app(action.get("value", ""))

    if action_type == "project":
        return open_project(action.get("value", ""))

    if action_type == "folder":
        return open_folder(action.get("value", ""))

    if action_type == "search":
        return search_web(action.get("query", ""), action.get("engine", "google"))

    return "Encontré el atajo, patrón, pero está mal guardado. Qué sorpresa, memoria con drama."


def system_status() -> str:
    commands = {
        "Disco": "df -h ~ | tail -n 1",
        "RAM": "free -h | awk '/Mem:/ {print $3 \" usados de \" $2}'",
        "Carga": "uptime",
    }

    parts = []

    for label, cmd in commands.items():
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            executable="/bin/bash",
        )
        value = result.stdout.strip() or result.stderr.strip()
        parts.append(f"{label}: {value}")

    return "Estado del sistema, patrón:\n" + "\n".join(parts)


def parse_learning_command(text: str) -> str | None:
    lower = normalize(text)

    match = re.match(r"^aprende atajo (.+?) para abrir (.+)$", lower)
    if match:
        alias, target = match.groups()
        return learn_shortcut(alias, target)

    match = re.match(r"^aprende comando (.+?) para abrir (.+)$", lower)
    if match:
        alias, target = match.groups()
        return learn_shortcut(alias, target)

    match = re.match(r"^recuerda que cuando diga (.+?) abre (.+)$", lower)
    if match:
        alias, target = match.groups()
        return learn_shortcut(alias, target)

    return None


def parse_chapi_command(text: str) -> str | None:
    lower = normalize(text)

    patterns = [
        r"^dile a chapi que (.+)$",
        r"^pidele a chapi que (.+)$",
        r"^preguntale a chapi (.+)$",
        r"^que chapi me ayude a (.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, lower)
        if match:
            return create_chapi_request(match.group(1))

    return None


def parse_search_command(text: str) -> str | None:
    lower = normalize(text)

    # Spotify específico:
    # "pon dtmf de bad bunny en spotify"
    # "reproduce no digas nada en spotify"
    # "busca una canción de latin mafia en spotify"
    match = re.match(
        r"^(?:pon|reproduce|toca|busca cancion|busca una cancion|busca la cancion) (.+?)(?: en spotify)?$",
        lower,
    )
    if match:
        query = match.group(1).strip()
        query = re.sub(r"^(la cancion|una cancion|cancion)\s+", "", query).strip()
        return search_web(query, "spotify")

    # "spotify dtmf de bad bunny"
    match = re.match(r"^spotify (.+)$", lower)
    if match:
        return search_web(match.group(1), "spotify")

    # "abre spotify y busca dtmf de bad bunny"
    match = re.match(r"^abre spotify y busca (.+)$", lower)
    if match:
        return search_web(match.group(1), "spotify")

    # Búsquedas generales por motor
    match = re.match(r"^abre (google|youtube|github|wikipedia|duckduckgo|spotify|stack overflow|maps|imagenes|traductor|arxiv) y busca (.+)$", lower)
    if match:
        engine, query = match.groups()
        return search_web(query, engine)

    match = re.match(r"^abre (?:una )?pestana en (google|youtube|github|wikipedia|duckduckgo|spotify|stack overflow|maps|imagenes|traductor|arxiv) y busca (.+)$", lower)
    if match:
        engine, query = match.groups()
        return search_web(query, engine)

    match = re.match(r"^busca en (google|youtube|github|wikipedia|duckduckgo|spotify|stack overflow|maps|imagenes|traductor|arxiv) (.+)$", lower)
    if match:
        engine, query = match.groups()
        return search_web(query, engine)

    match = re.match(r"^busca (.+) en (google|youtube|github|wikipedia|duckduckgo|spotify|stack overflow|maps|imagenes|traductor|arxiv)$", lower)
    if match:
        query, engine = match.groups()
        return search_web(query, engine)

    match = re.match(r"^googlea (.+)$", lower)
    if match:
        return search_web(match.group(1), "google")

    match = re.match(r"^youtube (.+)$", lower)
    if match:
        return search_web(match.group(1), "youtube")

    match = re.match(r"^abre opera y busca (.+)$", lower)
    if match:
        return search_web(match.group(1), "google")

    match = re.match(r"^investiga (.+)$", lower)
    if match:
        return search_web(match.group(1), "google")

    match = re.match(r"^busca (.+)$", lower)
    if match:
        return search_web(match.group(1), "google")

    return None


def parse_open_command(text: str) -> str | None:
    lower = normalize(text)

    if lower in {
        "abre una pestana en opera",
        "abre pestana en opera",
        "abre una pestana nueva en opera",
        "abre pestana nueva en opera",
        "abre una pagina en opera",
    }:
        return open_url("https://www.google.com")

    match = re.match(r"^abre (?:una )?pestana(?: de| en)? (.+)$", lower)
    if match:
        target = match.group(1).strip()

        if target in COMMON_SITES:
            return open_url(COMMON_SITES[target])

        return open_url(target)

    match = re.match(r"^abre proyecto (.+)$", lower)
    if match:
        return open_project(match.group(1))

    match = re.match(r"^(?:lanza|ejecuta|corre|inicia) (?:el )?(?:launcher de |proyecto )?(.+)$", lower)
    if match:
        target = match.group(1).strip()

        if target in PROJECTS:
            return launch_project(target)

    match = re.match(r"^analiza proyecto (.+)$", lower)
    if match:
        return analyze_project(match.group(1))

    match = re.match(r"^abre (.+)$", lower)
    if match:
        target = match.group(1).strip()

        if target in COMMON_SITES:
            return open_url(COMMON_SITES[target])

        if target in APP_COMMANDS:
            return open_app(target)

        if target in PROJECTS:
            return open_project(target)

        if target in FOLDERS:
            return open_folder(target)

        if target.startswith("~/") or target.startswith("/"):
            return open_path(Path(target))

    return None


def handle_direct_skill(text: str) -> str | None:
    raw = clean_activation(text)
    lower = normalize(raw)

    if not lower:
        return None

    learned_result = run_learned_shortcut(lower)
    if learned_result is not None:
        return learned_result

    result = parse_learning_command(raw)
    if result is not None:
        return result

    result = parse_chapi_command(raw)
    if result is not None:
        return result

    if lower in {
        "estado",
        "estado del sistema",
        "como esta la laptop",
        "como esta mi laptop",
        "estado de la laptop",
    }:
        return system_status()

    result = parse_search_command(raw)
    if result is not None:
        return result

    result = parse_open_command(raw)
    if result is not None:
        return result

    if lower.startswith("busca archivo "):
        query = raw.replace("busca archivo ", "", 1).strip()
        return search_files(query)

    if lower.startswith("encuentra archivo "):
        query = raw.replace("encuentra archivo ", "", 1).strip()
        return search_files(query)

    if lower.startswith("crea nota "):
        title = raw.replace("crea nota ", "", 1).strip()
        return create_note(title)

    if lower in {"abre mis documentos", "abre documentos"}:
        return open_folder("documentos")

    if lower in {"abre descargas", "abre mis descargas"}:
        return open_folder("descargas")

    if lower in {"abre escritorio", "abre mi escritorio"}:
        return open_folder("escritorio")

    return None
