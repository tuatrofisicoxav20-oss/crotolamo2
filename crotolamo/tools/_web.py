"""Fontanería HTTP/HTML de las tools de web (search.py).

Aquí vive lo que NO es una @tool: el GET con guardia anti-SSRF, los parsers de
HTML (resultados de DuckDuckGo, texto de página) y el filtro de URLs internas.
search.py queda solo con las funciones @tool que ve el LLM.

Todo con stdlib pura (urllib + html.parser).
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_MAX_DOWNLOAD_BYTES = 1_500_000  # ~1.5 MB: suficiente para cualquier artículo


def _ip_is_public(ip_str: str) -> bool:
    """False si la IP es de la red interna (loopback, LAN, link-local...)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _resolve_public_ip(host: str) -> str | None:
    """UNA resolución DNS: si TODAS las IPs del host son públicas, devuelve la
    primera (la IP a la que se conectará: pinning); si alguna es interna o el
    DNS no resuelve, None.

    Resolver y conectar en pasos separados abría un TOCTOU: un DNS malicioso
    con TTL 0 responde una IP pública en la validación y 127.0.0.1 al conectar
    (rebinding). Devolver la IP validada y conectar EXACTAMENTE a ella cierra
    esa ventana. Ante la duda, None: negar es lo seguro.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return None
    ips = [str(info[4][0]) for info in infos]
    if not ips or not all(_ip_is_public(ip) for ip in ips):
        return None
    return ips[0]


def is_public_url(url: str) -> bool:
    """False si la URL apunta a la red interna (loopback, LAN, link-local...).

    El LLM elige a qué URL llamar, y a veces la elige a partir de texto que vio en
    otra página. Sin este filtro podría pedir `http://localhost:11434` (el propio
    Ollama) o `http://169.254.169.254` (metadatos de nube): eso es SSRF.

    OJO: esto es el pre-check barato (lo usa read_page para responder en
    personaje sin descargar nada). La garantía fuerte contra rebinding vive en
    _http_get, que valida y PINNEA la misma resolución.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    return _resolve_public_ip(host) is not None


class BlockedRedirectError(urllib.error.URLError):
    """La URL (original o de un redirect) apunta a la red interna: SSRF bloqueado."""


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection que conecta a una IP ya validada (pinning anti-rebinding).

    El header Host sale de `host` (el hostname original), pero el TCP va a
    `pinned_ip`: así el servidor virtual correcto responde y el resolver ya no
    pinta nada entre la validación y la conexión.
    """

    def __init__(self, host: str, pinned_ip: str, port: int | None,
                 timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Como _PinnedHTTPConnection pero con TLS: el SNI y la validación del
    certificado usan el hostname original (server_hostname), no la IP."""

    def __init__(self, host: str, pinned_ip: str, port: int | None,
                 timeout: float) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip
        self._ssl_context = context

    def connect(self) -> None:
        raw = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw, server_hostname=self.host)


_MAX_REDIRECTS = 5


def _pinned_connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    """Valida la URL (gate + resolución pinneada) y devuelve (conexión, path)."""
    parts = urlparse(url)
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host:
        raise urllib.error.URLError(f"URL rara para un GET: {url!r}")
    # Gate a nivel URL (barato y parcheable en tests); la garantía real es el
    # pin de abajo: la conexión va a la MISMA IP que acaba de validarse.
    if not is_public_url(url):
        raise BlockedRedirectError(
            f"bloqueada una dirección no pública: {url}")
    ip = _resolve_public_ip(host)
    if ip is None:
        raise BlockedRedirectError(
            f"bloqueada una dirección no pública o que no resuelve: {url}")
    cls = _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
    conn = cls(host, ip, parts.port, timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return conn, path


def _http_get(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """GET con pinning de IP anti-rebinding. Devuelve (content_type, body_texto).

    Helper separado para que los tests lo parcheen sin tocar la red. Cada salto
    (URL original y cada redirect) se valida y se conecta a la IP de esa misma
    resolución. Lanza OSError/urllib.error.URLError si la red falla,
    urllib.error.HTTPError si el status es >= 400, y BlockedRedirectError si
    algún salto apunta a la red interna.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        conn, path = _pinned_connection(current, timeout)
        try:
            conn.request("GET", path, headers={"User-Agent": _USER_AGENT})
            response = conn.getresponse()

            location = response.getheader("Location")
            if 300 <= response.status < 400 and location:
                current = urljoin(current, location)
                continue
            if response.status >= 400:
                raise urllib.error.HTTPError(
                    current, response.status, response.reason, response.headers, None)

            content_type = (response.getheader("Content-Type") or "").lower()
            raw = response.read(_MAX_DOWNLOAD_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
        finally:
            conn.close()
        try:
            body = raw.decode(charset, errors="replace")
        except LookupError:  # charset inventado por el servidor
            body = raw.decode("utf-8", errors="replace")
        return content_type, body
    raise urllib.error.URLError(f"demasiados redirects (> {_MAX_REDIRECTS}): {url}")


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

    # Elementos vacíos de HTML: nunca tienen cierre, así que no cuentan para
    # el anidamiento. Contarlos dejaba al parser "dentro" del título/snippet
    # para siempre y se perdían TODOS los resultados siguientes al primer <br>.
    _VOID_TAGS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._mode: str | None = None  # "title" | "snippet" | None
        self._depth = 0  # anidamiento dentro del elemento capturado (b, span...)
        self._buffer: list[str] = []
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._mode is not None:
            if tag == "br":
                self._buffer.append(" ")  # un salto de línea separa palabras
            elif tag not in self._VOID_TAGS:
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

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # <br/> dentro de un título/snippet: ni abre ni cierra nada. El default
        # de HTMLParser (starttag + endtag) descuadraría el contador.
        if self._mode is None:
            super().handle_startendtag(tag, attrs)
        elif tag == "br":
            self._buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._mode is None or tag in self._VOID_TAGS:
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
