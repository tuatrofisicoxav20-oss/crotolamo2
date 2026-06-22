"""Tool routing: elige qué tools enviarle al LLM según la consulta.

PROBLEMA que resuelve: en CPU, mandar las ~32 tool-schemas (~2500 tokens) hace
que la PRIMERA llamada de cada turno tarde ~130-156s evaluando el prompt. Con
solo 5-8 tools relevantes (~600-900 tokens) ese costo baja a ~2-5s. Medido en
este equipo: 6 tools = 2.4s vs 30 tools = 142s.

CÓMO: keyword-matching insensible a acentos en español. Cada "grupo" (≈ un
módulo de tools) tiene disparadores; la consulta activa uno o varios grupos y se
envía la unión de sus tools, con un tope. Si nada matchea, se manda un set común
para que las acciones básicas nunca se rompan. Cero dependencias (stdlib).

Bonus: con menos tools a la vista, el modelo 3B elige mejor (menos distractores),
lo que mitiga errores de selección (p.ej. pedir "siguiente canción" y que llame
a 'music_now' en vez de 'music_control').
"""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable


def _norm(text: str) -> str:
    """minúsculas + sin acentos, para matchear robusto la voz/typos del patrón."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Grupo -> (tools del grupo, palabras clave que lo activan). Las keywords se
# comparan ya normalizadas (sin acentos, minúsculas) como subcadena de la consulta.
GROUPS: dict[str, dict[str, Any]] = {
    "system": {
        "tools": ["disk_usage", "ram_usage", "list_processes", "system_status"],
        "keywords": [
            "ram", "memoria", "disco", "espacio", "almacenamiento", "proceso",
            "cpu", "sistema", "rendimiento", "lenta", "lento", "compu", "equipo",
            "computadora", "laptop", "lap", "estado", "uso de", "consume",
        ],
    },
    "desktop": {
        "tools": ["open_app", "open_folder", "open_url", "send_hotkey"],
        "keywords": [
            "abre", "abrir", "abreme", "abrime", "lanza", "lanzar", "ejecuta",
            "app", "aplicacion", "programa", "ventana", "navegador", "pagina",
            "sitio", "url", "link", "enlace", "opera", "vscode", "geany",
            "blender", "libreoffice", "calculadora", "nautilus", "terminal",
            "atajo", "tecla", "alt+tab", "alt tab", "minimiza", "cierra ventana",
            "escritorio",
        ],
    },
    "media": {
        "tools": ["music_now", "music_control"],
        "keywords": [
            "musica", "cancion", "canciones", "spotify", "reproduce", "reproducir",
            "pon", "ponme", "pausa", "pausala", "play", "dale play", "siguiente",
            "anterior", "suena", "sonando", "tema", "escuchar", "rola", "salta",
            "siguiente cancion", "para la musica", "quita la musica", "toca",
        ],
    },
    "files": {
        "tools": [
            "create_note", "read_file", "write_file", "list_dir", "make_dir",
            "move_file", "delete_file", "search_files",
        ],
        "keywords": [
            "archivo", "archivos", "fichero", "nota", "notas", "apunta",
            "lee", "leer", "escribe", "escribir", "guarda en", "crea un",
            "crea una", "borra", "elimina", "mueve", "renombra", "lista",
            "directorio", "txt", "markdown", "md", "documento",
            # frases comunes para "ver qué hay" en una carpeta (gap del routing):
            "carpeta", "que hay en", "que tengo en", "muestra", "muestrame",
            "contenido de", "ensename los", "ver los archivos",
        ],
    },
    "facts": {
        "tools": ["remember_fact", "recall_facts", "search_facts", "forget_fact"],
        "keywords": [
            "recuerda", "acuerdate", "acuerda", "recordar", "recuerdas",
            "olvida", "que sabes de mi", "que sabes sobre mi", "anota que",
            "memoriza", "ten en cuenta", "hecho sobre",
        ],
    },
    "projects": {
        "tools": [
            "list_projects", "analyze_project", "list_project_tree",
            "find_in_project", "read_project_file", "launch_project",
        ],
        "keywords": [
            # OJO: "crotolamo" NO va aquí: es el nombre del asistente y aparece en
            # saludos ("hola crotolamo"), que deben caer al set común, no a projects.
            "proyecto", "proyectos", "huevonitis", "tletl",
            "repo", "repositorio", "codigo", "analiza el proyecto", "abre el proyecto",
            "lanza el proyecto", "estructura del proyecto",
        ],
    },
    "shortcuts": {
        "tools": ["learn_shortcut", "list_shortcuts", "run_shortcut"],
        "keywords": [
            "atajo", "atajos", "alias", "ensena", "ensename", "aprende",
            "shortcut", "acceso directo",
        ],
    },
    "search": {
        "tools": ["search_web"],
        "keywords": [
            "busca en", "buscar en", "google", "internet", "en la web",
            "investiga", "buscame", "busca informacion",
        ],
    },
}

# Tope de tools por turno. ~8 mantiene el prompt chico (rápido) sin castrar al
# modelo. Si varios grupos se activan, se respeta este límite por orden de grupo.
MAX_TOOLS_DEFAULT = 8


def select_tool_names(text: str, max_tools: int = MAX_TOOLS_DEFAULT) -> list[str]:
    """Nombres de tools relevantes a la consulta (con tope).

    Si NINGÚN grupo se activa, devuelve [] a propósito: es charla/saludo. En ese
    caso NO le colgamos tools al modelo, por dos razones medidas en este equipo:
    (1) velocidad — sin tools el turno baja de ~30s a ~1-2s; (2) calidad — con
    tools colgando, el 3B alucina y llama una tool aunque solo le saluden
    ("hola" -> abría apps al azar). Sin tools, simplemente conversa.
    """
    norm = _norm(text)
    selected: list[str] = []
    seen: set[str] = set()

    for group in GROUPS.values():
        if any(kw in norm for kw in group["keywords"]):
            for name in group["tools"]:
                if name not in seen:
                    seen.add(name)
                    selected.append(name)

    return selected[:max_tools]


def route_schemas(
    registry,
    text: str,
    max_tools: int = MAX_TOOLS_DEFAULT,
) -> list[dict[str, Any]]:
    """Esquemas (formato Ollama) de las tools enrutadas para esta consulta.

    Solo incluye tools que existen en el registry (ignora nombres desconocidos,
    p.ej. si una tool se quita). Lista vacía es intencional (charla): significa
    "sin tools este turno", NO "manda todas".
    """
    names = select_tool_names(text, max_tools)
    if not names:
        return []  # charla: turno rápido, sin tentar al modelo a alucinar tools
    schemas: list[dict[str, Any]] = []
    available = set(registry.names())
    for name in names:
        if name in available:
            tool = registry.get(name)
            if tool is not None:
                schemas.append(tool.schema())
    return schemas


def all_schemas(registry) -> list[dict[str, Any]]:
    """Sin routing: todas las tools (comportamiento previo)."""
    return registry.schemas()


def names_for(text: str, registry=None, max_tools: int = MAX_TOOLS_DEFAULT) -> Iterable[str]:
    """Helper para depurar/inspeccionar qué se enrutaría (usado en tests)."""
    names = select_tool_names(text, max_tools)
    if registry is not None:
        available = set(registry.names())
        return [n for n in names if n in available]
    return names
