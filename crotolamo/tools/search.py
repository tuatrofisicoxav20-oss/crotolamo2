"""Tools de búsqueda web. Migrado de C1::skills.py.search_web.

- search_web: construye la URL del motor pedido y la abre con open_url (el patrón
  la ve en su navegador; el asistente no).
- fetch_web_results: búsqueda real contra html.duckduckgo.com — devuelve título,
  URL y snippet de los primeros resultados como texto, para que el LLM se los
  lea al patrón.
- read_page: descarga una URL y devuelve su texto plano (sin scripts ni menús)
  para que el LLM lo resuma por voz.

La fontanería (GET anti-SSRF, parsers de HTML, filtro de URLs internas) vive en
_web.py; aquí quedan solo las @tool. Bloquea términos obviamente tóxicos (la
única blocklist que sobrevive: contenido, no comandos).
"""

from __future__ import annotations

import urllib.error
from urllib.parse import quote_plus, urlparse

from crotolamo.tools import _web
from crotolamo.tools._web import (  # noqa: F401 — re-exports por compat
    BlockedRedirectError,
    _DDGResultsParser,
    _decode_ddg_href,
    _http_get,
    _PageTextExtractor,
    is_public_url,
)
from crotolamo.tools.base import normalize_key, tool
from crotolamo.tools.desktop import open_url

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

_MAX_PAGE_CHARS = 6_000
_MAX_RESULTS = 5

_BLOQUEO_RED_INTERNA = ("Esa dirección es de la red interna, patrón. No leo ahí: "
                        "solo páginas públicas de internet.")


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


@tool
def fetch_web_results(query: str) -> str:
    """Busca en internet y devuelve los primeros resultados (título, URL y resumen)
    como texto, para poder leérselos o contárselos al patrón sin abrir el navegador.

    Args:
        query: lo que se quiere buscar en internet.
    """
    query = query.strip()
    if not query:
        return "Necesito algo que buscar, patrón."
    if is_blocked_query(query):
        return "No voy a buscar eso, patrón. Huele a desastre."

    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        _content_type, body = _web._http_get(url, timeout=10.0)
    except _web.BlockedRedirectError:
        return _BLOQUEO_RED_INTERNA
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"No pude buscar en internet, patrón: falló la conexión ({error})."

    parser = _web._DDGResultsParser()
    parser.feed(body)
    results = parser.results[:_MAX_RESULTS]
    if not results:
        return f"No encontré resultados para '{query}', patrón."

    lines = [f"Resultados de la búsqueda '{query}':"]
    for i, item in enumerate(results, start=1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   URL: {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
    lines.append("Puedo leerte cualquiera de estas páginas completas con read_page, patrón.")
    return "\n".join(lines)


@tool
def read_page(url: str) -> str:
    """Descarga una página web y devuelve su contenido como texto plano,
    para poder resumírselo o leérselo al patrón.

    Args:
        url: dirección de la página a leer (con o sin https://).
    """
    url = url.strip()
    if not url:
        return "Necesito una URL que leer, patrón."
    if "://" not in url:
        url = "https://" + url

    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"Solo puedo leer páginas http o https, patrón, no '{scheme}://'."

    if not _web.is_public_url(url):
        return _BLOQUEO_RED_INTERNA

    try:
        content_type, body = _web._http_get(url, timeout=12.0)
    except _web.BlockedRedirectError:
        # Un 3xx quiso llevarnos a la red interna: mismo bloqueo que la URL directa.
        return _BLOQUEO_RED_INTERNA
    except urllib.error.HTTPError as error:
        return f"La página respondió con error {error.code}, patrón. No pude leerla."
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"No pude conectarme a esa página, patrón ({error})."

    mime = content_type.split(";")[0].strip()
    if mime and not (mime.startswith("text/") or mime in ("application/xhtml+xml",)):
        return f"Eso no es una página que pueda leer, patrón (es contenido tipo {mime})."

    if mime == "text/plain":
        text = "\n".join(" ".join(line.split()) for line in body.splitlines() if line.strip())
    else:
        extractor = _web._PageTextExtractor()
        extractor.feed(body)
        text = extractor.text()

    if not text.strip():
        return "La página no tiene texto legible, patrón. Puro adorno."

    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS].rstrip()
        text += "\n[... texto truncado, la página sigue, patrón]"
    return f"Contenido de {url}:\n{text}"
