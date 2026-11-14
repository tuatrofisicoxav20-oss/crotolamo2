"""Fontanería HTTP/HTML de las tools de web (search.py).

Aquí vive lo que NO es una @tool: el GET con guardia anti-SSRF, los parsers de
HTML (resultados de DuckDuckGo, texto de página) y el filtro de URLs internas.
search.py queda solo con las funciones @tool que ve el LLM.

Todo con stdlib pura (urllib + html.parser).
"""

from __future__ import annotations

import codecs
import http.client
import ipaddress
import re
import socket
import ssl
import string
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urljoin, urlparse, urlsplit

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_MAX_DOWNLOAD_BYTES = 1_500_000  # ~1.5 MB: suficiente para cualquier artículo
_READ_CHUNK = 65_536  # tamaño de cada lectura del cuerpo (ver _read_body)
# Al percent-codificar una URL se deja intacta TODA la puntuación ASCII ("%"
# incluido: lo que ya venía codificado no se codifica dos veces). Solo cambian
# los espacios, los caracteres de control y lo que no es ASCII, que es justo lo
# que http.client rechaza en la línea de petición. Mismo criterio que urllib
# con sus redirects.
_URL_SAFE = string.punctuation


def _ip_is_public(ip_str: str) -> bool:
    """False si la IP es de la red interna (loopback, LAN, link-local...)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # is_global ADEMÁS de los chequeos explícitos: ipaddress no marca is_private
    # el espacio compartido 100.64.0.0/10 (CGNAT del ISP y TODOS los nodos de
    # Tailscale/Headscale), así que sin él read_page llegaba a los servicios de
    # la tailnet. is_site_local (solo IPv6) cubre fec0::/10, que tampoco sale
    # como privado.
    return ip.is_global and not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        or getattr(ip, "is_site_local", False))


def _resolve_public_ip(host: str) -> str | None:
    """UNA resolución DNS: si TODAS las IPs del host son públicas, devuelve la
    primera (la IP a la que se conectará: pinning); si alguna es interna, None.

    Resolver y conectar en pasos separados abría un TOCTOU: un DNS malicioso
    con TTL 0 responde una IP pública en la validación y 127.0.0.1 al conectar
    (rebinding). Devolver la IP validada y conectar EXACTAMENTE a ella cierra
    esa ventana. Ante la duda, None: negar es lo seguro.

    Si el DNS NO resuelve lanza URLError en vez de devolver None: sin wifi o
    con el dominio mal escrito no hay "red interna" que bloquear, hay un fallo
    de conexión, y mezclarlos hacía que el patrón oyera "esa dirección es de
    la red interna" de una página pública. Tampoco se conecta a nada.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError) as error:
        raise urllib.error.URLError(f"no pude resolver {host!r}: {error}") from error
    ips = [str(info[4][0]) for info in infos]
    if not ips:
        raise urllib.error.URLError(f"no pude resolver {host!r}: sin direcciones")
    if not all(_ip_is_public(ip) for ip in ips):
        return None
    return ips[0]


def is_public_url(url: str) -> bool:
    """False si la URL apunta a la red interna (loopback, LAN, link-local...).

    El LLM elige a qué URL llamar, y a veces la elige a partir de texto que vio en
    otra página. Sin este filtro podría pedir `http://localhost:11434` (el propio
    Ollama) o `http://169.254.169.254` (metadatos de nube): eso es SSRF.

    OJO: la garantía (y el bloqueo) de verdad vive en _http_get, que valida y
    PINNEA la misma resolución en cada salto. read_page ya no lo usa de
    pre-check (gastaba un getaddrinfo y llamaba "red interna" a un DNS que no
    resuelve); queda como consulta suelta y por compatibilidad (search lo
    re-exporta).
    """
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        return _resolve_public_ip(host) is not None
    except urllib.error.URLError:  # no resuelve: ante la duda, negar
        return False


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


_SSL_CONTEXT: ssl.SSLContext | None = None


def _ssl_context() -> ssl.SSLContext:
    """Contexto TLS compartido por todas las conexiones https del módulo.

    ssl.create_default_context() carga el bundle de CAs del sistema (~20 ms de
    CPU): crearlo por conexión lo repetía en cada salto de cada read_page. Un
    SSLContext es reutilizable y seguro entre hilos. Perezoso: se paga en el
    primer GET https, no al importar las tools.
    """
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Como _PinnedHTTPConnection pero con TLS: el SNI y la validación del
    certificado usan el hostname original (server_hostname), no la IP."""

    def __init__(self, host: str, pinned_ip: str, port: int | None,
                 timeout: float) -> None:
        context = _ssl_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip
        self._ssl_context = context

    def connect(self) -> None:
        raw = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw, server_hostname=self.host)


_MAX_REDIRECTS = 5


def _pinned_connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    """Valida la URL (resolución pinneada) y devuelve (conexión, path)."""
    # urlsplit y no urlparse: urlparse le corta el ";params" al último segmento
    # y aquí solo se re-unían path y query ("/doc;jsessionid=X" se pedía como
    # "/doc": otra página, o un 404).
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host:
        raise urllib.error.URLError(f"URL rara para un GET: {url!r}")
    # UNA resolución hace de gate y de pin: la conexión va a la MISMA IP que
    # acaba de validarse. Antes había además un is_public_url(url) delante que
    # repetía el getaddrinfo por cada salto sin validar nada distinto. Si el
    # DNS no resuelve, _resolve_public_ip lanza URLError (fallo de conexión,
    # no bloqueo SSRF).
    ip = _resolve_public_ip(host)
    if ip is None:
        raise BlockedRedirectError(
            f"bloqueada una dirección no pública: {url}")
    cls = _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
    # Puerto SIEMPRE explícito: con port=None http.client vuelve a partir el
    # host por el último ":" y a un IPv6 literal ("2606:4700:4700::1111") le
    # tomaba el último hexteto como puerto (o lanzaba InvalidURL con "::abcd").
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    conn = cls(host, ip, port, timeout)
    # La línea de petición tiene que ser ASCII sin espacios: "/wiki/México" o
    # "/wiki/Ciudad de México" (lo normal cuando la URL la escribe el LLM) hacían
    # fallar a http.client antes de conectar.
    path = quote(parts.path, safe=_URL_SAFE) or "/"
    if parts.query:
        path += "?" + quote(parts.query, safe=_URL_SAFE)
    return conn, path


def _is_text_mime(mime: str) -> bool:
    """True si el tipo es texto que las tools saben leer (o no viene declarado).

    Mismo criterio con el que read_page acepta o rechaza una respuesta."""
    return not mime or mime.startswith("text/") or mime == "application/xhtml+xml"


def _read_body(response: http.client.HTTPResponse, timeout: float) -> bytes:
    """Lee el cuerpo a trozos: hasta _MAX_DOWNLOAD_BYTES o `timeout` segundos EN
    TOTAL, lo que llegue antes. Devuelve lo que alcanzó a leer.

    El timeout del socket es por recv(): un servidor que gotea bytes (o un
    stream que no acaba nunca, tipo text/event-stream) no lo dispara jamás, y un
    único read(1.5 MB) se quedaba esperando con el hilo del Brain colgado.
    read1 devuelve lo que haya en cuanto llega, así que el plazo se revisa
    entre trozos; el recv() individual sigue acotado por el timeout del socket.
    """
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    remaining = _MAX_DOWNLOAD_BYTES
    while remaining > 0 and time.monotonic() < deadline:
        chunk = response.read1(min(_READ_CHUNK, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# <meta charset="x"> y <meta http-equiv="Content-Type" content="...; charset=x">.
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]*?[\s;"']charset\s*=\s*["']?\s*([\w.:-]+)""", re.IGNORECASE)
_META_PRESCAN_BYTES = 4096  # la declaración va al principio del <head>
_HTML_MIMES = ("", "text/html", "application/xhtml+xml")


def _sniff_charset(raw: bytes, mime: str) -> str | None:
    """Charset que declara el PROPIO documento (BOM o <meta>), o None."""
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"  # además se come el BOM
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if mime not in _HTML_MIMES:  # en text/plain un "<meta ...>" es texto, no una orden
        return None
    match = _META_CHARSET_RE.search(raw, 0, _META_PRESCAN_BYTES)
    if match is None:
        return None
    declared = match.group(1).decode("ascii")
    # Un <meta> que dice utf-16 miente: si se pudo leer como ASCII, no lo es
    # (los navegadores lo tratan como UTF-8).
    return "utf-8" if declared.lower().startswith("utf-16") else declared


def _decode_body(raw: bytes, header_charset: str | None, mime: str) -> str:
    """Bytes -> texto. Manda el charset del header HTTP; si calla, el que
    declare el documento; si nadie dice nada, UTF-8.

    Antes solo se miraba el header: una página ISO-8859-1 que lo declara en su
    <meta> (webs viejas en español) llegaba con cada acento y cada ñ
    convertidos en U+FFFD.
    """
    charset = header_charset or _sniff_charset(raw, mime) or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeError):  # charset inventado (o que no es de texto)
        return raw.decode("utf-8", errors="replace")


def _http_get(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """GET con pinning de IP anti-rebinding. Devuelve (content_type, body_texto).

    Helper separado para que los tests lo parcheen sin tocar la red. Cada salto
    (URL original y cada redirect) se valida y se conecta a la IP de esa misma
    resolución. Lanza OSError/urllib.error.URLError si la red falla (o el DNS
    no resuelve, o la respuesta HTTP llega rota), urllib.error.HTTPError si el
    status es >= 400, y BlockedRedirectError si algún salto apunta a la red
    interna.

    Si el Content-Type no es texto (PDF, zip, audio...) el cuerpo NO se
    descarga: body_texto llega vacío. `timeout` acota cada operación de red y,
    además, el total de la lectura del cuerpo (ver _read_body).
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            conn, path = _pinned_connection(current, timeout)
        except http.client.HTTPException as error:
            # El constructor de HTTPConnection rechaza un host con caracteres de
            # control (un NUL: getaddrinfo corta ahí y resuelve, http.client no)
            # con InvalidURL, que no es OSError: se escapaba como "reventó".
            # Nunca llegó a conectar: es una URL inválida, no un hueco SSRF.
            raise urllib.error.URLError(
                f"URL inválida para un GET ({error!r})") from error
        try:
            conn.request("GET", path, headers={"User-Agent": _USER_AGENT})
            response = conn.getresponse()

            location = response.getheader("Location")
            if 300 <= response.status < 400 and location:
                # http.client decodifica los headers como ISO-8859-1: se
                # recuperan los bytes originales y se percent-codifican (un
                # Location con UTF-8 crudo o espacios rompía el siguiente salto).
                current = urljoin(current, quote(
                    location, encoding="iso-8859-1", safe=_URL_SAFE))
                continue
            if response.status >= 400:
                raise urllib.error.HTTPError(
                    current, response.status, response.reason, response.headers, None)

            content_type = (response.getheader("Content-Type") or "").lower()
            mime = content_type.split(";")[0].strip()
            if not _is_text_mime(mime):
                # Quien llama lo va a rechazar por el tipo: bajarse antes 1.5 MB
                # de PDF (o minuto y medio de una radio por internet) era
                # trabajo tirado, con el hilo del Brain esperando.
                return content_type, ""
            raw = _read_body(response, timeout)
            charset = response.headers.get_content_charset()
        except http.client.HTTPException as error:
            # IncompleteRead (chunked cortado a medias), BadStatusLine (el
            # puerto no habla HTTP), InvalidURL...: NO heredan de OSError, así
            # que se le escapaban a quien llama (que espera lo que promete el
            # docstring) y la tool acababa en el "reventó" genérico.
            raise urllib.error.URLError(
                f"fallo de protocolo HTTP ({error!r})") from error
        finally:
            conn.close()
        return content_type, _decode_body(raw, charset, mime)
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


class _TextBudgetReached(Exception):
    """_PageTextExtractor ya juntó más texto del que le pidieron: deja de parsear."""


class _PageTextExtractor(HTMLParser):
    """Extrae texto legible de una página: párrafos, títulos, listas y tablas.

    Ignora por completo script/style/nav/header/footer/aside/noscript.

    Con `max_chars` deja de parsear en cuanto el texto ya extraído lo supera:
    quien solo va a conservar el principio (read_page se queda con unos miles
    de caracteres) no paga el HTMLParser en Python puro del documento entero.
    El texto devuelto es el mismo hasta ese punto y mide más que `max_chars`,
    así que quien llama sigue viendo que la página era más larga.
    """

    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
    _CONTENT_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "td", "th", "blockquote", "pre", "figcaption", "title",
    }
    # Etiquetas que el navegador pinta con salto de línea. DENTRO de un bloque
    # de contenido separan palabras: sin esto "<p>Av. Juárez 5<br>Col. Centro"
    # o "<blockquote><p>Uno.</p><p>Dos.</p>" llegaban pegados al LLM y al TTS
    # ("Av. Juárez 5Col. Centro"). Las etiquetas en línea (b, a, sub...) no.
    _BREAK_TAGS = _CONTENT_TAGS | {"br", "hr", "div"}

    def __init__(self, max_chars: int | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._content_depth = 0
        self._chunks: list[str] = []
        self._buffer: list[str] = []
        self._max_chars = max_chars
        self._length = 0  # len("\n".join(self._chunks)), llevado al día
        self._full = False

    def feed(self, data: str) -> None:
        if self._full:
            return
        # Se corta DESDE DENTRO del parseo (handle_endtag lanza al pasarse del
        # tope) y no dándole el HTML a trozos: el feed incremental de HTMLParser
        # deja datos sin procesar hasta close() según la versión de Python, y
        # así el camino sin tope y el con tope ven exactamente los mismos eventos.
        try:
            super().feed(data)
        except _TextBudgetReached:
            self._full = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0:
            if self._content_depth > 0 and tag in self._BREAK_TAGS:
                self._buffer.append(" ")
            if tag in self._CONTENT_TAGS:
                self._content_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0:
            if self._content_depth > 0 and tag in self._BREAK_TAGS:
                self._buffer.append(" ")
            if tag in self._CONTENT_TAGS:
                self._content_depth = max(0, self._content_depth - 1)
                if self._content_depth == 0:
                    self._flush()
                    if self._max_chars is not None and self._length > self._max_chars:
                        raise _TextBudgetReached

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._content_depth > 0:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer = []
        if text:
            self._length += len(text) + (1 if self._chunks else 0)
            self._chunks.append(text)

    def text(self) -> str:
        self._flush()
        return "\n".join(self._chunks)
