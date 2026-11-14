"""Revisión final, unidad 18: crotolamo/safety (guard.py, paths.py).

1. Un URI file: SIN barra tras el esquema ("file:etc/passwd",
   "file:\\etc\\passwd", "file:../../etc/passwd", "file:%2Fetc/passwd") no se
   reconocía como URI: la regex pedía "file:/". El valor no empieza por "/", así
   que tampoco parecía una ruta, y pasaba sin clasificar bajo url/uri. Un parser
   WHATWG (Chromium, Node) lo abre desde la raíz (/etc/passwd) y urllib,
   relativo a su CWD: con un server MCP de solo lectura (el guard es su único
   corral) leía /etc/passwd sin bloqueo ni pregunta. Ahora se clasifican las
   dos lecturas y manda la más estricta.

2. Un server MCP con `cwd` en su config abre las rutas relativas contra ESE
   directorio, pero el guard las clasificaba contra el CWD de Crotolamo (el
   repo, zona libre, bajo el .service): read_file(path=".env") con el server en
   ~/proyectos/privado leía el .env sin preguntar. Una tool MCP que trae su
   base (`path_base`) se clasifica contra ella.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crotolamo.safety.guard import Guard
from crotolamo.tools.base import Tool


def _remota(path_base: Path | str | None = None, safe: bool = True) -> Tool:
    """Tool MCP (strict_args=False). `path_base` es el cwd de su server."""
    tool = Tool(name="mcp_x_leer", func=lambda **k: "ok", description="d", parameters={},
                safe=safe, strict_args=False)
    if path_base is not None:
        tool.path_base = path_base  # type: ignore[attr-defined]
    return tool


def _nativa() -> Tool:
    return Tool(name="t", func=lambda **k: "ok", description="d", parameters={})


@pytest.fixture
def casa(tmp_path, monkeypatch):
    """Home de mentira: Documentos es la zona libre, el resto del home pide
    confirmación y Crotolamo corre desde su repo (en Documentos), como el
    .service. Devuelve (home, repo, guard)."""
    home = (tmp_path / "home").resolve()
    repo = home / "Documentos" / "crotolamo2"
    repo.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(repo)
    return home, repo, Guard(allowed_roots=[home / "Documentos"], confirm_roots=[home])


# ---------------------------------------------------------------------------
# 1. URIs file: sin barra tras el esquema
# ---------------------------------------------------------------------------

_SIN_BARRA = [
    "file:etc/passwd",
    "file:\\etc\\passwd",
    "FILE:etc/passwd",
    " file:etc/passwd",
    "file:%2Fetc/passwd",
    "file:../../../../../../../../etc/passwd",
    "fi\tle:\netc/passwd",
]


@pytest.mark.parametrize("uri", _SIN_BARRA)
def test_un_uri_file_sin_barra_no_pasa_sin_clasificar(casa, uri):
    """Desde la zona libre (el CWD del .service) y con una tool MCP de solo
    lectura: antes, allowed=True sin confirmación bajo url y uri."""
    _home, _repo, guard = casa
    for tool in (_remota(), _nativa()):
        for nombre in ("url", "uri", "path"):
            decision = guard.check(tool, {nombre: uri})
            assert not decision.allowed, (tool.strict_args, nombre, uri, decision)
            assert "permitidas" in decision.reason


def test_un_uri_file_sin_barra_no_destructiva_nombra_el_archivo(casa):
    """Con una tool no-safe, el bloqueo no se convierte en la pregunta genérica
    de "puede ser destructiva" (que no nombra el archivo)."""
    _home, _repo, guard = casa
    decision = guard.check(_remota(safe=False), {"url": "file:etc/passwd"})
    assert not decision.allowed
    assert "'/etc/passwd'" in decision.reason


def test_un_uri_file_sin_barra_cuenta_tambien_relativo_al_cwd(casa, monkeypatch):
    """urllib abre "file:<resto>" relativo a su CWD; un navegador, desde la raíz.
    Si una de las dos cae fuera de la zona libre, se pregunta."""
    home, _repo, guard = casa
    libre = home / "Documentos"
    (home / "privado").mkdir()
    monkeypatch.chdir(home / "privado")
    uri = "file:" + str(libre / "x.txt").lstrip("/")
    decision = guard.check(_remota(), {"url": uri})
    # Lectura del navegador: <libre>/x.txt (libre). La de urllib:
    # <home>/privado/<libre>/x.txt (confirmación). Manda la más estricta.
    assert decision.allowed and decision.needs_confirmation, decision
    # Con el CWD en la zona libre, las dos lecturas son libres.
    monkeypatch.chdir(libre)
    limpia = guard.check(_remota(), {"url": uri})
    assert limpia.allowed and not limpia.needs_confirmation, limpia


@pytest.mark.parametrize("valor", ["file: x", "file:", "file:  ", "archivo file:x"])
def test_lo_que_no_es_un_uri_file_sigue_sin_serlo(casa, valor):
    """Un espacio tras "file:" sigue siendo texto ("file: x"), y "file:" a secas
    no es un URI."""
    _home, _repo, guard = casa
    decision = guard.check(_remota(), {"url": valor})
    assert decision.allowed and not decision.needs_confirmation, (valor, decision)


def test_los_uri_file_con_barra_se_clasifican_como_antes(casa):
    home, _repo, guard = casa
    libre = guard.check(_remota(), {"url": f"file://{home}/Documentos/a.txt"})
    assert libre.allowed and not libre.needs_confirmation
    pregunta = guard.check(_remota(), {"url": f"file://{home}/a.txt"})
    assert pregunta.allowed and pregunta.needs_confirmation
    assert not guard.check(_remota(), {"url": "file:///etc/passwd"}).allowed


# ---------------------------------------------------------------------------
# 2. Las relativas de un server MCP con cwd propio
# ---------------------------------------------------------------------------

def test_relativa_de_un_server_mcp_con_cwd_en_zona_de_confirmacion(casa):
    """Server con cwd = ~/proyectos/privado: path=".env" es SU .env, no el del
    repo de Crotolamo. Antes: allowed=True sin confirmación."""
    home, _repo, guard = casa
    privado = home / "proyectos" / "privado"
    privado.mkdir(parents=True)
    tool = _remota(path_base=privado)
    decision = guard.check(tool, {"path": ".env"})
    assert decision.allowed and decision.needs_confirmation, decision
    # Lo que oye el patrón es la ruta que el server va a abrir.
    assert decision.reason == f"fuera de la zona libre: {privado / '.env'}. ¿Lo hago?"
    # También bajo un nombre cualquiera y anidado.
    assert guard.check(tool, {"target": "./.env"}).needs_confirmation
    assert guard.check(tool, {"opciones": {"ruta": ".env"}}).needs_confirmation


def test_relativa_de_un_server_mcp_con_cwd_fuera_del_corral(casa):
    """Server con cwd = "/": path="etc/passwd" es /etc/passwd. Antes: libre."""
    _home, _repo, guard = casa
    tool = _remota(path_base=Path("/"))
    decision = guard.check(tool, {"path": "etc/passwd"})
    assert not decision.allowed
    assert "'/etc/passwd'" in decision.reason
    # Con cwd en el home, ".ssh/id_rsa" es ~/.ssh/id_rsa: se pregunta.
    home, _repo, _guard = casa
    assert guard.check(_remota(path_base=str(home)), {"path": ".ssh/id_rsa"}).needs_confirmation
    # Y la lectura relativa de un URI file: sin barra, también contra su cwd.
    assert not guard.check(tool, {"url": "file:etc/passwd"}).allowed


def test_server_mcp_con_cwd_en_la_zona_libre_corre_directo(casa, monkeypatch):
    """La base del server manda aunque Crotolamo corra desde fuera del corral;
    lo absoluto y lo de ~ no cambian."""
    home, repo, guard = casa
    monkeypatch.chdir("/")
    tool = _remota(path_base=repo)
    dentro = guard.check(tool, {"path": "src/main.py"})
    assert dentro.allowed and not dentro.needs_confirmation, dentro
    assert not guard.check(tool, {"path": "../../../../etc/passwd"}).allowed
    assert not guard.check(tool, {"path": "/etc/passwd"}).allowed
    assert guard.check(tool, {"path": "~/x.txt"}).needs_confirmation
    absoluta = guard.check(tool, {"path": str(repo / "a.txt")})
    assert absoluta.allowed and not absoluta.needs_confirmation


def test_sin_cwd_propio_un_server_mcp_sigue_resolviendo_contra_el_cwd(casa, monkeypatch):
    """Sin `cwd` en su config, el server hereda el CWD de Crotolamo."""
    home, repo, guard = casa
    dentro = guard.check(_remota(), {"path": ".env"})
    assert dentro.allowed and not dentro.needs_confirmation
    monkeypatch.chdir(home)
    assert guard.check(_remota(), {"path": ".env"}).needs_confirmation


def test_la_base_de_un_server_mcp_no_toca_a_las_tools_nativas(casa):
    """Las nativas se anclan a [paths].relative_base (o al CWD), nunca a una
    base que no les corresponde."""
    _home, _repo, guard = casa
    nativa = _nativa()
    nativa.path_base = Path("/")  # type: ignore[attr-defined]
    decision = guard.check(nativa, {"path": "a.txt"})
    assert decision.allowed and not decision.needs_confirmation
