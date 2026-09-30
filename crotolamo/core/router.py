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

M4: además de los grupos estáticos de abajo, los servers MCP se registran como
grupos DINÁMICOS (`register_group` / `unregister_group`) y, cuando varios de
ellos matchean, rotan en round-robin para que el tope no mate siempre al mismo.
"""

from __future__ import annotations

import threading
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
            # frases naturales de voz que no disparaban (gap de validación):
            "quita la cancion", "quitala", "parale", "cortale",
        ],
    },
    "windows": {
        "tools": ["list_windows", "app_status", "focus_window", "close_window"],
        "keywords": [
            "ventana", "ventanas", "que apps", "tengo abiert", "abiertas",
            "abiertos", "esta abierto", "esta abierta", "sigue abierto",
            "sigue abierta", "cierra", "cierrame", "enfoca", "enfocame",
            "cambiate a", "cambia a", "trae al frente", "pon enfrente",
            "workspace", "esta corriendo",
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
            # imperativos naturales de voz (gap de validación):
            "mueveme", "muevelo", "pasame", "pasalo", "manda", "mandame",
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
    "home": {
        "tools": ["light_control", "home_state"],
        "keywords": [
            "luz", "luces", "foco", "focos", "lampara", "prende", "prendeme",
            "prendida", "apaga", "apagame", "apagada", "enciende", "encendida",
            "ilumina", "brillo", "domotica", "home assistant", "casa inteligente",
        ],
    },
    "cameras": {
        "tools": ["camera_events", "camera_snapshot"],
        "keywords": [
            "camara", "camaras", "frigate", "vigilancia", "movimiento",
            "quien paso", "alguien paso", "alguien entro", "quien anduvo",
            "detectaron", "detecto", "snapshot", "que vieron", "eventos de",
            "foto de la entrada", "algo raro afuera",
        ],
    },
    "search": {
        # search_web abre pestaña; fetch_web_results/read_page LEEN el contenido
        # para poder respondérselo al patrón por voz.
        "tools": ["search_web", "fetch_web_results", "read_page"],
        "keywords": [
            "busca en", "buscar en", "google", "internet", "en la web",
            "investiga", "buscame", "busca informacion",
            "que dice internet", "informacion sobre", "informacion de",
            "leeme la pagina", "lee la pagina", "leeme el articulo",
            "averigua", "consulta en internet", "quien es", "que es ",
        ],
    },
}

# Tope de tools por turno. ~8 mantiene el prompt chico (rápido) sin castrar al
# modelo. Si varios grupos se activan, se respeta este límite por orden de grupo.
MAX_TOOLS_DEFAULT = 8


# ---------------------------------------------------------------------------
# Grupos DINÁMICOS (M4): los servers MCP se registran al arrancar (y se retiran
# en caliente si se caen), así que no pueden vivir en el dict estático GROUPS.
# Van con el mismo formato {tools, keywords}. Hoy todo grupo dinámico es un
# server MCP; eso es lo que decide el round-robin de abajo.
# ---------------------------------------------------------------------------
_DYNAMIC_GROUPS: dict[str, dict[str, Any]] = {}
_DYNAMIC_LOCK = threading.Lock()

# Contador del round-robin entre grupos MCP matcheados. Solo avanza cuando en
# la consulta matcheó al menos un grupo MCP: sin ellos, el routing es EXACTAMENTE
# el de siempre (determinista, mismo orden de especificidad).
_RR_COUNTER = 0


def register_group(name: str, tools: list[str], keywords: list[str]) -> None:
    """Registra (o reemplaza) un grupo dinámico de routing.

    Las keywords se normalizan aquí (minúsculas, sin acentos) para que el bridge
    MCP o la config del patrón puedan escribirlas "con acentos y todo". El nombre
    no puede chocar con un grupo estático: el bridge usa el espacio "mcp:<server>".
    """
    if name in GROUPS:
        raise ValueError(f"'{name}' ya es un grupo estático del router")
    norm_keywords: list[str] = []
    for kw in keywords:
        k = _norm(str(kw)).strip()
        if k and k not in norm_keywords:
            norm_keywords.append(k)
    with _DYNAMIC_LOCK:
        _DYNAMIC_GROUPS[name] = {"tools": list(tools), "keywords": norm_keywords}


def unregister_group(name: str) -> bool:
    """Quita un grupo dinámico. True si existía."""
    with _DYNAMIC_LOCK:
        return _DYNAMIC_GROUPS.pop(name, None) is not None


def dynamic_groups() -> dict[str, dict[str, Any]]:
    """Copia de los grupos dinámicos registrados (inspección/tests)."""
    with _DYNAMIC_LOCK:
        return {name: dict(group) for name, group in _DYNAMIC_GROUPS.items()}


def _rotate_mcp_groups(ordered: list[tuple[int, int, int, dict[str, Any], bool]]) -> None:
    """Rota IN PLACE, en round-robin, los grupos MCP dentro de la lista ordenada.

    POR QUÉ: con el tope de max_tools, dos servers MCP que matchean la misma
    consulta competirían siempre en el mismo orden y el segundo jamás vería
    la luz. Rotar solo el orden RELATIVO de los grupos MCP (ocupan las mismas
    posiciones que les dio el score) reparte las oportunidades entre llamadas
    sin tocar el lugar de los grupos estáticos.
    """
    global _RR_COUNTER
    positions = [i for i, entry in enumerate(ordered) if entry[4]]
    if not positions:
        return
    with _DYNAMIC_LOCK:
        shift = _RR_COUNTER % len(positions)
        _RR_COUNTER += 1
    mcp_entries = [ordered[i] for i in positions]
    rotated = mcp_entries[shift:] + mcp_entries[:shift]
    for pos, entry in zip(positions, rotated):
        ordered[pos] = entry


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

    # Especificidad primero: el grupo con MÁS y MÁS LARGOS keywords matcheados
    # va antes en la unión. Con el tope de max_tools, esto evita que un grupo
    # genérico ("pagina" -> desktop, "lee" -> files) desplace al grupo que
    # matcheó la frase completa ("leeme la pagina" -> search). Score = suma de
    # longitudes de los matches (evidencia total), desempate por número de
    # matches y luego por orden de declaración, como antes. Los grupos
    # dinámicos (MCP) van DESPUÉS de los estáticos en el orden de declaración.
    scored: list[tuple[int, int, int, dict[str, Any], bool]] = []
    with _DYNAMIC_LOCK:
        dynamic = list(_DYNAMIC_GROUPS.values())
    static_groups = [(g, False) for g in GROUPS.values()]
    all_groups = static_groups + [(g, True) for g in dynamic]
    for idx, (group, is_mcp) in enumerate(all_groups):
        matched = [kw for kw in group["keywords"] if kw in norm]
        if matched:
            score = sum(len(kw) for kw in matched)
            scored.append((score, len(matched), idx, group, is_mcp))

    ordered = sorted(scored, key=lambda t: (-t[0], -t[1], t[2]))
    _rotate_mcp_groups(ordered)

    for _, _, _, group, _ in ordered:
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


def names_for(text: str, registry=None, max_tools: int = MAX_TOOLS_DEFAULT) -> Iterable[str]:
    """Helper para depurar/inspeccionar qué se enrutaría (usado en tests)."""
    names = select_tool_names(text, max_tools)
    if registry is not None:
        available = set(registry.names())
        return [n for n in names if n in available]
    return names
