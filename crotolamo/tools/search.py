"""Tools de búsqueda web. Migrado de C1::skills.py.search_web.

- search_web: construye la URL del motor pedido y la abre con open_url (el patrón
  la ve en su navegador; el asistente no).
- fetch_web_results: búsqueda real contra html.duckduckgo.com — devuelve título,
  URL y snippet de los primeros resultados como texto, para que el LLM se los
  lea al patrón.
- read_page: descarga una URL y devuelve su texto plano (sin scripts ni menús)
  para que el LLM lo resuma por voz.

Todo con stdlib pura (urllib + html.parser). Bloquea términos obviamente
tóxicos (la única blocklist que sobrevive: contenido, no comandos).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urlparse

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


# ---------------------------------------------------------------------------
# Lectura real de la web (fetch_web_results / read_page)
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_MAX_DOWNLOAD_BYTES = 1_500_000  # ~1.5 MB: suficiente para cualquier artículo
_MAX_PAGE_CHARS = 6_000
_MAX_RESULTS = 5


def _http_get(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """GET con urllib. Devuelve (content_type, body_texto).

    Helper separado para que los tests lo parcheen sin tocar la red.
    Lanza OSError/urllib.error.URLError si la red falla.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        content_type = (response.headers.get("Content-Type") or "").lower()
        raw = response.read(_MAX_DOWNLOAD_BYTES)
        charset = response.headers.get_content_charset() or "utf-8"
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:  # charset inventado por el servidor
        body = raw.decode("utf-8", errors="replace")
    return content_type, body


def _decode_ddg_href(href: str) -> str:
    """Los links de html.duckduckgo.com vienen como //duckduckgo.com/l/?uddg=<url>.

    Extrae y decodifica el parámetro uddg; si no existe, devuelve el href tal cual
    (normalizando el esquema si empieza con //).
    """
    parsed = urlparse(href, scheme="https")
    uddg = parse_qs(parsed.query).get("uddg")
    if uddg and uddg[0]:
        return uddg[0]
    if href.startswith("//"):
        return "https:" + href
    return href


class _DDGResultsParser(HTMLParser):
    """Saca (título, url, snippet) del HTML de html.duckduckgo.com.

    Estructura real: <a class="result__a" href="//duckduckgo.com/l/?uddg=...">Título</a>
    y <a class="result__snippet" ...>snippet</a> (a veces es un <div>).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._mode: str | None = None  # "title" | "snippet" | None
        self._depth = 0  # anidamiento dentro del elemento capturado (b, span...)
        self._buffer: list[str] = []
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._mode is not None:
            self._depth += 1
            return
        classes = (dict(attrs).get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            href = dict(attrs).get("href") or ""
            self._current = {"title": "", "url": _decode_ddg_href(href), "snippet": ""}
            self._mode = "title"
            self._buffer = []
        elif tag in ("a", "div", "span") and "result__snippet" in classes:
            self._mode = "snippet"
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._mode is None:
            return
        if self._depth > 0:
            self._depth -= 1
            return
        text = " ".join("".join(self._buffer).split())
        if self._mode == "title" and self._current is not None:
            self._current["title"] = text
            self.results.append(self._current)
        elif self._mode == "snippet" and self.results:
            last = self.results[-1]
            if not last["snippet"]:
                last["snippet"] = text
        self._mode = None
        self._current = None
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._mode is not None:
            self._buffer.append(data)


class _PageTextExtractor(HTMLParser):
    """Extrae texto legible de una página: párrafos, títulos, listas y tablas.

    Ignora por completo script/style/nav/header/footer/aside/noscript.
    """

    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
    _CONTENT_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "td", "th", "blockquote", "pre", "figcaption", "title",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._content_depth = 0
        self._chunks: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self._CONTENT_TAGS:
            self._content_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0 and tag in self._CONTENT_TAGS:
            self._content_depth = max(0, self._content_depth - 1)
            if self._content_depth == 0:
                self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._content_depth > 0:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer = []
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        self._flush()
        return "\n".join(self._chunks)


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
        _content_type, body = _http_get(url, timeout=10.0)
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"No pude buscar en internet, patrón: falló la conexión ({error})."

    parser = _DDGResultsParser()
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

    try:
        content_type, body = _http_get(url, timeout=12.0)
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
        extractor = _PageTextExtractor()
        extractor.feed(body)
        text = extractor.text()

    if not text.strip():
        return "La página no tiene texto legible, patrón. Puro adorno."

    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS].rstrip()
        text += "\n[... texto truncado, la página sigue, patrón]"
    return f"Contenido de {url}:\n{text}"
