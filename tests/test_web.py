"""Tests de fetch_web_results / read_page SIN red real.

Se parchea search._http_get (el único punto que toca la red) y se alimenta
HTML de mentiras con la estructura real de html.duckduckgo.com.
"""

import urllib.error

import pytest

from crotolamo.tools import default_registry, search

# HTML mínimo con la estructura real de html.duckduckgo.com: links result__a
# con uddg encodeado y snippets result__snippet.
DDG_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fes.wikipedia.org%2Fwiki%2FAjolote&amp;rut=abc">
       Ajolote - <b>Wikipedia</b></a>
  </h2>
  <a class="result__snippet"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fes.wikipedia.org%2Fwiki%2FAjolote">
     El <b>ajolote</b> es un anfibio endémico de México.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1%26b%3D2&amp;rut=xyz">
       Segundo resultado</a>
  </h2>
  <a class="result__snippet" href="#">Snippet dos.</a>
</div>
</body></html>
"""

PAGE_HTML = """
<html>
<head><title>Página de prueba</title>
<script>var basura = "no debe salir";</script>
<style>.oculto { display: none; }</style>
</head>
<body>
<nav><ul><li>Menú que no va</li></ul></nav>
<header><h1>Encabezado de sitio fuera</h1></header>
<main>
  <h1>Título    principal</h1>
  <p>Primer   párrafo con
  espacios raros.</p>
  <p>Segundo párrafo con <b>negritas</b> adentro.</p>
  <ul><li>Elemento uno</li><li>Elemento dos</li></ul>
  <script>console.log("tampoco esto");</script>
</main>
<footer><p>Pie de página que no va</p></footer>
<aside><p>Barra lateral que no va</p></aside>
</body></html>
"""


def _fake_dns(monkeypatch, ip="93.184.216.34"):
    """Neutraliza la resolución DNS que hace el guardia SSRF de read_page.

    Sin esto, cualquier test que llame a read_page haría una consulta DNS real y
    fallaría sin conexión, rompiendo la promesa de "SIN red" de este archivo.
    """
    monkeypatch.setattr(search.socket, "getaddrinfo",
                        lambda *a, **kw: [(2, 1, 6, "", (ip, 0))])


def _patch_get(monkeypatch, content_type, body, capture=None):
    def fake_get(url, timeout=10.0):
        if capture is not None:
            capture.append(url)
        return content_type, body

    monkeypatch.setattr(search, "_http_get", fake_get)
    _fake_dns(monkeypatch)


# ---------------------------------------------------------------------------
# fetch_web_results
# ---------------------------------------------------------------------------

def test_fetch_parses_ddg_results(monkeypatch):
    seen = []
    _patch_get(monkeypatch, "text/html; charset=utf-8", DDG_HTML, capture=seen)
    out = search.fetch_web_results("ajolote")

    assert "html.duckduckgo.com/html/?q=ajolote" in seen[0]
    assert "1. Ajolote - Wikipedia" in out
    assert "https://es.wikipedia.org/wiki/Ajolote" in out
    assert "anfibio endémico de México" in out
    assert "2. Segundo resultado" in out
    assert "Snippet dos." in out
    # No debe filtrarse el link intermedio de DDG.
    assert "duckduckgo.com/l/" not in out


def test_fetch_decodes_uddg_with_query_params(monkeypatch):
    _patch_get(monkeypatch, "text/html", DDG_HTML)
    out = search.fetch_web_results("lo que sea")
    # uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1%26b%3D2 -> URL real completa.
    assert "https://example.com/page?a=1&b=2" in out


def test_decode_ddg_href_directly():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx&rut=zzz"
    assert search._decode_ddg_href(href) == "https://example.com/x"
    # Sin uddg: se devuelve tal cual (normalizando el //).
    assert search._decode_ddg_href("//example.com/y") == "https://example.com/y"
    assert search._decode_ddg_href("https://example.com/z") == "https://example.com/z"


def test_fetch_blocked_query(monkeypatch):
    called = []
    _patch_get(monkeypatch, "text/html", DDG_HTML, capture=called)
    out = search.fetch_web_results("descargar ransomware para windows")
    assert "desastre" in out.lower()
    assert called == []  # ni siquiera intenta la red


def test_fetch_empty_query(monkeypatch):
    _patch_get(monkeypatch, "text/html", DDG_HTML)
    assert "algo que buscar" in search.fetch_web_results("   ").lower()


def test_fetch_no_results(monkeypatch):
    _patch_get(monkeypatch, "text/html", "<html><body>nada</body></html>")
    out = search.fetch_web_results("cosa rarisima")
    assert "no encontré resultados" in out.lower()


def test_fetch_network_error(monkeypatch):
    def boom(url, timeout=10.0):
        raise urllib.error.URLError("nombre no resuelve")

    monkeypatch.setattr(search, "_http_get", boom)
    out = search.fetch_web_results("ajolote")
    assert "patrón" in out
    assert "conexión" in out.lower()
    assert "Traceback" not in out


def test_fetch_limits_to_five_results(monkeypatch):
    block = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fe.com%2F{i}">'
        "R{i}</a>"
    )
    html = "".join(block.format(i=i) for i in range(8))
    _patch_get(monkeypatch, "text/html", html)
    out = search.fetch_web_results("muchos")
    assert "5. R4" in out
    assert "R5" not in out


# ---------------------------------------------------------------------------
# read_page
# ---------------------------------------------------------------------------

def test_read_page_extracts_text_and_skips_junk(monkeypatch):
    _patch_get(monkeypatch, "text/html; charset=utf-8", PAGE_HTML)
    out = search.read_page("https://example.com/articulo")

    assert "Título principal" in out  # espacios colapsados
    assert "Primer párrafo con espacios raros." in out
    assert "Segundo párrafo con negritas adentro." in out
    assert "Elemento uno" in out
    # Nada de scripts/estilos/nav/header/footer/aside.
    assert "basura" not in out
    assert "display" not in out
    assert "Menú que no va" not in out
    assert "Encabezado de sitio fuera" not in out
    assert "Pie de página que no va" not in out
    assert "Barra lateral que no va" not in out


def test_read_page_adds_https_when_missing(monkeypatch):
    seen = []
    _patch_get(monkeypatch, "text/html", PAGE_HTML, capture=seen)
    search.read_page("example.com/articulo")
    assert seen[0] == "https://example.com/articulo"


def test_read_page_rejects_non_http_schemes(monkeypatch):
    called = []
    _patch_get(monkeypatch, "text/html", PAGE_HTML, capture=called)
    out = search.read_page("file:///etc/passwd")
    assert "http" in out.lower()
    assert called == []  # nunca toca la red


def test_read_page_non_html_content_type(monkeypatch):
    _patch_get(monkeypatch, "application/pdf", "%PDF-1.7 blob")
    out = search.read_page("https://example.com/doc.pdf")
    assert "no es una página que pueda leer" in out.lower()


def test_read_page_plain_text_passes(monkeypatch):
    _patch_get(monkeypatch, "text/plain; charset=utf-8", "hola   mundo\nsegunda línea")
    out = search.read_page("https://example.com/robots.txt")
    assert "hola mundo" in out
    assert "segunda línea" in out


def test_read_page_truncates_long_pages(monkeypatch):
    long_html = "<html><body><p>" + ("palabra " * 5000) + "</p></body></html>"
    _patch_get(monkeypatch, "text/html", long_html)
    out = search.read_page("https://example.com/largo")
    assert "truncado" in out
    assert len(out) < 6500


def test_read_page_http_error(monkeypatch):
    def boom(url, timeout=12.0):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(search, "_http_get", boom)
    _fake_dns(monkeypatch)
    out = search.read_page("https://example.com/no-existe")
    assert "404" in out
    assert "patrón" in out


def test_read_page_network_error(monkeypatch):
    def boom(url, timeout=12.0):
        raise urllib.error.URLError("se cayó el wifi")

    monkeypatch.setattr(search, "_http_get", boom)
    _fake_dns(monkeypatch)
    out = search.read_page("https://example.com")
    assert "patrón" in out
    assert "Traceback" not in out


def test_read_page_empty_url():
    assert "URL" in search.read_page("   ")


# ---------------------------------------------------------------------------
# registro en el registry
# ---------------------------------------------------------------------------

def test_new_tools_registered():
    reg = default_registry()
    for name in ("fetch_web_results", "read_page"):
        assert name in reg.names()
        assert reg.get(name).safe is True


# --- Guardia SSRF: read_page no debe alcanzar la red interna ---
# El LLM elige la URL, a veces a partir de texto de otra página. Sin este filtro
# podría leer el propio Ollama (localhost:11434) o metadatos de nube.

def _resolver_fijo(ip: str):
    """Sustituto de socket.getaddrinfo que siempre resuelve a `ip`."""
    def fake(host, port, *a, **kw):
        return [(2, 1, 6, "", (ip, 0))]
    return fake


@pytest.mark.parametrize("ip", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # privada
    "192.168.1.1",      # privada
    "169.254.169.254",  # link-local: metadatos de nube
    "0.0.0.0",          # unspecified
])
def test_url_interna_se_bloquea(monkeypatch, ip):
    monkeypatch.setattr(search.socket, "getaddrinfo", _resolver_fijo(ip))
    assert search.is_public_url("http://loquesea.com/") is False


def test_url_publica_se_permite(monkeypatch):
    monkeypatch.setattr(search.socket, "getaddrinfo", _resolver_fijo("93.184.216.34"))
    assert search.is_public_url("https://example.com/") is True


def test_dns_que_no_resuelve_se_niega(monkeypatch):
    """Ante la duda, negar."""
    def boom(*a, **kw):
        raise search.socket.gaierror("sin DNS")

    monkeypatch.setattr(search.socket, "getaddrinfo", boom)
    assert search.is_public_url("https://no-existe.invalid/") is False


def test_read_page_no_toca_la_red_si_es_interna(monkeypatch):
    """El bloqueo ocurre ANTES de la petición HTTP."""
    monkeypatch.setattr(search.socket, "getaddrinfo", _resolver_fijo("127.0.0.1"))

    def no_debe_llamarse(*a, **kw):
        raise AssertionError("read_page intentó descargar una URL interna")

    monkeypatch.setattr(search, "_http_get", no_debe_llamarse)
    out = search.read_page(url="http://localhost:11434/api/tags")
    assert "red interna" in out
