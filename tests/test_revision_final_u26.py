"""Revisión final, unidad 26: crotolamo/tools/projects.py y crotolamo/tools/search.py.

1. find_in_project con código que lleva corchetes o asteriscos. grep leía
   primero el patrón como regex (BRE) y solo lo buscaba tal cual si el regex
   no daba NADA. Pero `list[str]` como regex es "list" + una de {s,t,r}, que en
   cualquier código real casa con "lists": salían esas líneas falsas y las
   apariciones de verdad de `list[str]` quedaban escondidas.
2. fetch_web_results ante el captcha anti-bots de DuckDuckGo (HTTP 202, que
   _http_get da por bueno): el parser no veía resultados y la tool respondía
   "No encontré resultados", y el LLM le decía al patrón que no existe nada.
3. fetch_web_results con anuncios: html.duckduckgo.com pone los anuncios
   (div.result--ad, links a duckduckgo.com/y.js de kilobyte y medio) antes de
   los resultados orgánicos, y la tool se quedaba los 5 primeros sin filtrar.
4. read_page con texto que no es HTML (text/markdown, text/csv): todo lo que
   no era text/plain pasaba por el extractor de HTML, que solo guarda lo que
   va dentro de <p>, <li>...: "La página no tiene texto legible".
"""

from __future__ import annotations

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import _web, projects, search

# ---------------------------------------------------------------------------
# 1. find_in_project: el texto tal cual manda sobre la lectura como regex
# ---------------------------------------------------------------------------


@pytest.fixture
def proyecto(tmp_path, monkeypatch):
    proj = tmp_path / "p"
    proj.mkdir()
    monkeypatch.setattr(settings_mod.get_settings(), "projects", {"p": proj})
    return proj


@pytest.fixture
def greps(monkeypatch) -> list[list[str]]:
    """Deja correr grep de verdad y apunta cada argv lanzado."""
    lanzados: list[list[str]] = []
    real_run = projects.subprocess.run

    def espia(cmd, **kwargs):
        lanzados.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(projects.subprocess, "run", espia)
    return lanzados


@pytest.mark.parametrize("texto, codigo, ruido", [
    ("list[str]", "xs: list[str] = []", "def lists(): pass"),
    ("dict[str, Any]", "def f() -> dict[str, Any]: ...", "def dicts(): pass"),
    ("x[0]", "y = x[0] + 1", "x0 = 1"),
    ("n*2", "y = n*2", "z = 2"),  # como regex: cero o más "n" y un 2
], ids=["list-str", "dict-str-any", "subindice", "asterisco"])
def test_find_in_project_el_codigo_tal_cual_no_queda_tapado_por_el_regex(
    proyecto, greps, texto, codigo, ruido,
):
    """El caso que la propia corrección de la auditoría citaba: en un proyecto
    con la palabra "lists", `list[str]` leído como regex SÍ casa ("list" + s) y
    el literal nunca se probaba. Salía «Encontré» con la línea falsa sola."""
    (proyecto / "a.py").write_text(codigo + "\n", encoding="utf-8")
    (proyecto / "b.py").write_text(ruido + "\n", encoding="utf-8")
    out = projects.find_in_project("p", texto)
    assert out == f"Encontré «{texto}» en p, patrón:\na.py:1:{codigo}"
    # Encontrado tal cual a la primera: un solo grep, el literal.
    assert len(greps) == 1 and "-F" in greps[0]


def test_find_in_project_un_regex_de_verdad_se_sigue_encontrando(proyecto, greps):
    """Lo que el modelo pide como regex (anclas, clases) no existe tal cual en
    el código: el literal no da nada y entonces casa el regex. Dos greps como
    mucho, igual que antes en el caso contrario."""
    (proyecto / "agente.py").write_text("class ToolAgent:\n    pass\n", encoding="utf-8")
    out = projects.find_in_project("p", "^class Tool[A-Z]gent")
    assert out == "Encontré «^class Tool[A-Z]gent» en p, patrón:\nagente.py:1:class ToolAgent:"
    assert len(greps) == 2
    assert "-F" in greps[0] and "-F" not in greps[1]


@pytest.mark.parametrize("texto", ["no_esta[", "no_esta\\"])
def test_find_in_project_un_regex_invalido_que_no_esta_no_es_busqueda_incompleta(
    proyecto, greps, texto,
):
    """Con el regex de segundo, un patrón que ni es regex (`arr[`: grep sale con
    2 y "Unmatched [") no debe convertir el "no está" del literal en un "la
    búsqueda no fue completa: grep: Unmatched [..."."""
    (proyecto / "agente.py").write_text("class ToolAgent:\n    pass\n", encoding="utf-8")
    assert projects.find_in_project("p", texto) == f"No encontré «{texto}» en p, patrón."
    assert len(greps) == 2


def test_find_in_project_sin_metacaracteres_sigue_siendo_un_solo_grep(proyecto, greps):
    (proyecto / "agente.py").write_text("class ToolAgent:\n    pass\n", encoding="utf-8")
    out = projects.find_in_project("p", "toolagent")
    assert out == "Encontré «toolagent» en p, patrón:\nagente.py:1:class ToolAgent:"
    assert len(greps) == 1 and "-F" not in greps[0]


# ---------------------------------------------------------------------------
# Páginas de html.duckduckgo.com (recortadas de capturas reales)
# ---------------------------------------------------------------------------


def _patch_get(monkeypatch, content_type: str, body: str) -> None:
    monkeypatch.setattr(_web, "_http_get", lambda url, timeout=10.0: (content_type, body))


# Captura del 202 que DuckDuckGo le devuelve a veces a un cliente automático
# (14 KB; aquí lo que importa: el formulario de anomaly.js y el modal).
DDG_CAPTCHA = """<!DOCTYPE html>
<html lang="en">
<head><title> DuckDuckGo </title>
<link rel="stylesheet" href="//duckduckgo.com/dist/l.4db20359cb90b7c3b91a.css" type="text/css"/>
</head>
<body>
<center id="lite_wrapper">
<a class="header-url" href="/html/"><span class="header">DuckDuckGo</span></a>
<iframe name="ifr" width="0" height="0" border="0" class="hidden"></iframe>
<form id="img-form" action="//duckduckgo.com/anomaly.js?sv=html&cc=sre&ti=1794653202&gk=d4cd&q=python tutorial&r=use" target="ifr" method="POST"></form>
<form id="challenge-form" action="//duckduckgo.com/anomaly.js?sv=html&cc=sre&st=1794653202&gk=d4cd&q=python tutorial&r=use" method="POST">
  <div class="anomaly-modal__mask">
    <div class="anomaly-modal__modal  is-ie" data-testid="anomaly-modal">
      <div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>
      <div class="anomaly-modal__description">Please complete the following challenge to confirm this search was made by a human.</div>
      <div class="anomaly-modal__instructions">Select all squares containing a duck:</div>
      <div class="anomaly-modal__puzzle">
        <div class="anomaly-modal__box" data-index="0">
          <label class="" for="image-check_81c3" data-testid="anomaly-modal-tile-0">
            <input type="checkbox" class="anomaly-modal__check" name="image-check_81c3" id="image-check_81c3">
            <img class="anomaly-modal__image" alt=" " src="../assets/anomaly/images/challenge/81c3.jpg"></img>
          </label>
        </div>
      </div>
    </div>
  </div>
</form>
</center>
</body>
</html>
"""  # noqa: E501


def test_fetch_web_results_el_captcha_de_ddg_no_es_cero_resultados(monkeypatch):
    _patch_get(monkeypatch, "text/html; charset=utf-8", DDG_CAPTCHA)
    out = search.fetch_web_results("python tutorial")
    assert "No encontré resultados" not in out
    assert out.startswith("DuckDuckGo me frenó con un captcha")
    assert "'python tutorial'" in out and "search_web" in out


def test_fetch_web_results_una_pagina_sin_resultados_sigue_siendo_sin_resultados(monkeypatch):
    """La captura del captcha no se confunde con la página normal sin resultados."""
    sin_resultados = """<html><body><div class="serp__results">
      <div class="result results_links results_links_deep result--no-result">
        <div class="no-results">No  results.</div></div></div></body></html>"""
    _patch_get(monkeypatch, "text/html", sin_resultados)
    assert search.fetch_web_results("qwzxq") == "No encontré resultados para 'qwzxq', patrón."


# Anuncio tal cual lo sirve html.duckduckgo.com (URL de y.js recortada; la de
# verdad pasa de 1.5 KB con el aclick de bing dentro).
_ANUNCIO = """
<div class="result results_links results_links_deep result--ad {extra}">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3D{dominio}%26ad_provider%3Dbingv7aa%26ad_type%3Dtxad%26click_metadata%3DU1xZKhyu7Wwp0%26u3%3Dhttps%253A%252F%252Fwww.bing.com%252Faclick%253Fld%253De88uC2O84&amp;rut=313b">{titulo}</a>
      <div class="result__badge-wrap"><button class="badge--ad">Ad</button></div>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3D{dominio}">Anuncio de {dominio}.</a>
  </div>
</div>
"""  # noqa: E501

_ORGANICO = """
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.{dominio}%2F&amp;rut=6571">{titulo}</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.{dominio}%2F">Resumen de {dominio}.</a>
  </div>
</div>
"""  # noqa: E501

DDG_CON_ANUNCIOS = (
    '<html><body><div class="serp__results">'
    + _ANUNCIO.format(extra="", dominio="cheapoair.com", titulo="Book Cheap Flights Under $99")
    + _ANUNCIO.format(extra="result--ad--small", dominio="cheapflights.com",
                      titulo="Cheap Flights - Compare Prices")
    + "".join(_ORGANICO.format(dominio=d, titulo=f"Orgánico {i}") for i, d in enumerate(
        ["cheapflights.com", "skyscanner.com", "kayak.com", "google.com",
         "expedia.com", "priceline.com"], start=1))
    + "</div></body></html>"
)


def test_fetch_web_results_no_lista_anuncios(monkeypatch):
    _patch_get(monkeypatch, "text/html; charset=utf-8", DDG_CON_ANUNCIOS)
    out = search.fetch_web_results("cheap flights")
    assert "y.js" not in out and "bing.com" not in out and "Anuncio" not in out
    assert "Book Cheap Flights" not in out and "Compare Prices" not in out
    # Los cinco huecos son para resultados orgánicos, en su orden.
    for i, dominio in enumerate(
            ["cheapflights.com", "skyscanner.com", "kayak.com", "google.com",
             "expedia.com"], start=1):
        assert f"{i}. Orgánico {i}\n   URL: https://www.{dominio}/\n   Resumen de {dominio}." in out
    assert "Orgánico 6" not in out


def test_fetch_web_results_solo_anuncios_es_sin_resultados(monkeypatch):
    solo_anuncios = _ANUNCIO.format(extra="", dominio="x.com", titulo="Compra ya")
    _patch_get(monkeypatch, "text/html", solo_anuncios)
    assert search.fetch_web_results("compra") == "No encontré resultados para 'compra', patrón."


# ---------------------------------------------------------------------------
# 4. read_page: el texto que no es HTML se lee como texto
# ---------------------------------------------------------------------------

MARKDOWN = """# Workers

Cloudflare Workers   provides a **serverless** execution environment.

- Deploy in seconds
- Runs on the edge
"""


@pytest.mark.parametrize("content_type", ["text/markdown; charset=utf-8", "text/x-markdown"])
def test_read_page_lee_markdown(monkeypatch, content_type):
    _patch_get(monkeypatch, content_type, MARKDOWN)
    out = search.read_page("https://developers.cloudflare.com/workers/index.md")
    assert out == (
        "Contenido de https://developers.cloudflare.com/workers/index.md:\n"
        "# Workers\n"
        "Cloudflare Workers provides a **serverless** execution environment.\n"
        "- Deploy in seconds\n"
        "- Runs on the edge"
    )


def test_read_page_lee_csv(monkeypatch):
    _patch_get(monkeypatch, "text/csv", "ciudad,habitantes\nMonterrey,1142994\n")
    out = search.read_page("https://example.com/datos.csv")
    assert out == "Contenido de https://example.com/datos.csv:\nciudad,habitantes\nMonterrey,1142994"


@pytest.mark.parametrize("content_type", [
    "", "text/html", "application/xhtml+xml", "text/xml"])
def test_read_page_el_marcado_sigue_pasando_por_el_extractor(monkeypatch, content_type):
    """Lo que trae etiquetas (o no dice qué es) sigue yendo al extractor: ni
    scripts ni menús llegan al LLM."""
    html = ("<html><head><script>var basura = 1;</script></head><body><nav><li>Menú</li>"
            "</nav><p>Hola   <b>mundo</b></p></body></html>")
    _patch_get(monkeypatch, content_type, html)
    assert search.read_page("https://example.com/x") == "Contenido de https://example.com/x:\nHola mundo"
