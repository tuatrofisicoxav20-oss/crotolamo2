"""Fontanería HTTP/HTML de las tools de web (search.py).

Aquí vive lo que NO es una @tool: el GET con guardia anti-SSRF, los parsers de
HTML (resultados de DuckDuckGo, texto de página) y el filtro de URLs internas.
search.py queda solo con las funciones @tool que ve el LLM.

Todo con stdlib pura (urllib + html.parser).
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlparse

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_MAX_DOWNLOAD_BYTES = 1_500_000  # ~1.5 MB: suficiente para cualquier artículo


def is_public_url(url: str) -> bool:
    """False si la URL apunta a la red interna (loopback, LAN, link-local...).

    El LLM elige a qué URL llamar, y a veces la elige a partir de texto que vio en
    otra página. Sin este filtro podría pedir `http://localhost:11434` (el propio
    Ollama) o `http://169.254.169.254` (metadatos de nube): eso es SSRF.

    Resolvemos el hostname porque un dominio público puede apuntar a 127.0.0.1.
    Ante la duda (DNS que no resuelve), devolvemos False: negar es lo seguro.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


class BlockedRedirectError(urllib.error.URLError):
    """Un redirect (3xx) intentó llevarnos a la red interna: SSRF bloqueado."""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler que revalida cada URL destino contra is_public_url.

    urlopen sigue los 3xx automáticamente: sin esto, validar solo la URL
    original deja pasar un 302 hacia http://169.254.169.254/ o localhost.
    """

    def redirect_request(  # type: ignore[override]
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        if not is_public_url(newurl):
            raise BlockedRedirectError(
                f"redirect bloqueado hacia una dirección no pública: {newurl}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Opener único con el guardia anti-SSRF en los redirects. Se usa en TODOS los
# fetch de estas tools (read_page y la búsqueda de fetch_web_results): ambos
# descargan contenido remoto, así que ambos merecen el mismo cerrojo.
_SAFE_OPENER = urllib.request.build_opener(_SafeRedirectHandler)


def _http_get(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """GET con urllib. Devuelve (content_type, body_texto).

    Helper separado para que los tests lo parcheen sin tocar la red.
    Lanza OSError/urllib.error.URLError si la red falla, y
    BlockedRedirectError si un 3xx apunta a la red interna.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with _SAFE_OPENER.open(request, timeout=timeout) as response:  # noqa: S310
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
