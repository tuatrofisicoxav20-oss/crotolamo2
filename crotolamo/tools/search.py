"""Tool de búsqueda web. Migrado de C1::skills.py.search_web.

Construye la URL del motor pedido y la abre con open_url. Bloquea términos
obviamente tóxicos (la única blocklist que sobrevive: contenido, no comandos).
"""

from __future__ import annotations

from urllib.parse import quote_plus

from crotolamo.tools.base import tool
from crotolamo.tools.desktop import normalize_key, open_url

SEARCH_ENGINES: dict[str, str] = {
    "google": "https://www.google.com/search?q={query}",
    "youtube": "https://www.youtube.com/results?search_query={query}",
    "github": "https://github.com/search?q={query}",
    "wikipedia": "https://es.wikipedia.org/wiki/Special:Search?search={query}",
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "spotify": "https://open.spotify.com/search/{query}",
    "stackoverflow": "https://stackoverflow.com/search?q={query}",
    "maps": "https://www.google.com/maps/search/{query}",
    "imagenes": "https://www.google.com/search?tbm=isch&q={query}",
    "traductor": "https://translate.google.com/?sl=auto&tl=es&text={query}&op=translate",
    "arxiv": "https://arxiv.org/search/?query={query}&searchtype=all",
}

_BLOCKED_TERMS = [
    "porn", "porno", "xxx", "xvideos", "pornhub",
    "hackear contraseña", "robar contraseña", "malware para",
    "ransomware", "exploit para romper", "bomba", "arma casera",
]


def is_blocked_query(text: str) -> bool:
    lower = normalize_key(text)
    return any(normalize_key(term) in lower for term in _BLOCKED_TERMS)


def build_search_url(engine: str, query: str) -> str:
    key = normalize_key(engine)
    if key not in SEARCH_ENGINES:
        key = "google"
    if key == "spotify":
        # Spotify prefiere el término en el path, encoded con %20.
        return SEARCH_ENGINES[key].format(query=quote_plus(query).replace("+", "%20"))
    return SEARCH_ENGINES[key].format(query=quote_plus(query))


@tool
def search_web(query: str, engine: str = "google") -> str:
    """Busca algo en la web y abre los resultados en el navegador.

    Args:
        query: lo que se quiere buscar.
        engine: motor a usar (google, youtube, github, spotify, wikipedia, maps, etc.).
    """
    query = query.strip()
    if not query:
        return "Necesito algo que buscar, patrón."
    if is_blocked_query(query):
        return "No voy a buscar eso, patrón. Huele a desastre."

    url = build_search_url(engine, query)
    return open_url(url)
