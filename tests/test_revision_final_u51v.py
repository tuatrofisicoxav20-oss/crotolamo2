"""Revisión final, unidad 51v: scripts/crotolamo_doctor.py (la nube, Ollama y la caducidad).

1. Una key partida en dos líneas (el `source` de launch.sh conserva el salto
   dentro de las comillas) hacía reventar a http.client con un ValueError que
   trae la cabecera en repr ("Bearer sk-…\\n…"). _scrub solo tapaba el valor
   crudo, con un salto REAL: las dos mitades salían por pantalla bajo "Es un bug
   del doctor: repórtalo con este mensaje".
2. llm-key decía ✅ presente con una key que el runtime rechaza en cada turno
   (glm._key_cabe_en_header: comillas tipográficas, U+200B, un salto, un TAB) y
   llm-nube la probaba igual: o "el check reventó … bug del doctor", o ✅ con
   el TAB, que http.client sí manda.
3. La sonda de la nube seguía redirecciones (y reenviaba el Bearer al destino)
   mientras HTTPTransport no las sigue: llm-nube ✅ con cada turno fallando.
4. [llm].host con barra final: LLMClient la quita; el doctor pedía //api/tags,
   que el router de Ollama (gin) contesta con 404.
5. _as_date devolvía None con basura que no es texto (un int sin guiones, una
   lista): el aviso de caducidad desaparecía en silencio.

Todo SIN red externa: servidores efímeros en 127.0.0.1 y el `mundo` del doctor.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import date, time

import pytest

from crotolamo.core import glm
from crotolamo.core.llm import LLMClient
from scripts import crotolamo_doctor as doctor
from tests import test_doctor
from tests.test_doctor import _checks, _settings

# La fixture `mundo` del doctor, reutilizada tal cual (asignada, no importada por
# nombre, para que el linter no la tome por un import redefinido).
mundo = test_doctor.mundo

# El _urlopen de verdad, guardado al importar: `mundo` lo sustituye por uno falso
# y aquí hace falta el camino real (http.client) para ver qué hace con la key.
_URLOPEN_REAL = doctor._urlopen
_CONNECT_REAL = socket.create_connection

_Respuesta = tuple[int, dict[str, str], bytes]


class _Servidor(http.server.ThreadingHTTPServer):
    """Servidor efímero que apunta (ruta, Authorization) y contesta lo que diga
    `responder(ruta)`."""

    def __init__(self, responder: Callable[[str], _Respuesta]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.responder = responder
        self.vistas: list[tuple[str, str | None]] = []

    @property
    def raiz(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _Servidor)
        server.vistas.append((self.path, self.headers.get("Authorization")))
        status, headers, body = server.responder(self.path)
        self.send_response(status)
        for nombre, valor in {**headers, "Content-Length": str(len(body))}.items():
            self.send_header(nombre, valor)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def _nube_y_ollama(path: str) -> _Respuesta:
    """La nube (/v1/models con el modelo de `_settings`) y Ollama (/api/tags)."""
    if path.endswith("/models"):
        body = json.dumps({"data": [{"id": "glm-4.7-flash"}]}).encode()
    else:
        body = json.dumps({"models": [{"name": "qwen2.5-coder:7b"}]}).encode()
    return 200, {"Content-Type": "application/json"}, body


@pytest.fixture
def servir() -> Iterator[Callable[[Callable[[str], _Respuesta]], _Servidor]]:
    servidores: list[_Servidor] = []

    def arrancar(responder: Callable[[str], _Respuesta]) -> _Servidor:
        server = _Servidor(responder)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servidores.append(server)
        return server

    yield arrancar
    for server in servidores:
        server.shutdown()
        server.server_close()


@pytest.fixture
def nube_local(mundo, monkeypatch, servir) -> _Servidor:
    """El doctor con su red REAL apuntando a una nube y un Ollama en 127.0.0.1."""
    server = servir(_nube_y_ollama)
    mundo.settings.raw["llm"]["glm"]["base_url"] = f"{server.raiz}/v1"
    mundo.settings.raw["llm"]["host"] = server.raiz
    monkeypatch.setattr(doctor, "_urlopen", _URLOPEN_REAL)
    # `mundo` también finge socket.create_connection (el WebSocket de la música):
    # solo las conexiones a ESTE servidor van por el socket de verdad.
    falso = socket.create_connection

    def connect(addr, *args, **kwargs):
        if tuple(addr) == ("127.0.0.1", server.server_port):
            return _CONNECT_REAL(addr, *args, **kwargs)
        return falso(addr, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", connect)
    return server


# --- 1. la key partida en dos líneas no sale por pantalla ---------------------

PRIMERA, SEGUNDA = "PRIMERAMITAD", "SEGUNDAMITAD"
KEYS_PARTIDAS = (f"sk-{PRIMERA}\n{SEGUNDA}-xyz", f"sk-{PRIMERA}\r\n{SEGUNDA}-xyz")


@pytest.mark.parametrize("key", KEYS_PARTIDAS)
def test_una_key_partida_en_dos_lineas_no_sale_por_pantalla(nube_local, monkeypatch, capsys,
                                                             key):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", key)
    doctor.run_doctor()
    out = capsys.readouterr().out
    assert PRIMERA not in out and SEGUNDA not in out, out
    assert "reventó" not in out and "bug del doctor" not in out, out


@pytest.mark.parametrize("key", [*KEYS_PARTIDAS, f"sk-{PRIMERA}\n{SEGUNDA}-ñxyz"])
@pytest.mark.parametrize("forma", [
    repr,                                                      # repr del str
    lambda k: repr(f"Bearer {k}".encode("latin-1")),           # repr de los bytes (http.client)
    lambda k: k.splitlines()[-1],                              # solo una de las mitades
])
def test_scrub_tapa_tambien_la_key_escapada_y_sus_mitades(mundo, monkeypatch, capsys, key,
                                                          forma):
    """Última barrera: un check ajeno que revienta con la key en repr (con el salto
    escapado como \\n o los bytes como \\xf1) tampoco la deja salir."""
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", key)
    mundo.pw = RuntimeError(f"Invalid header value {forma(key)}")
    doctor.run_doctor()
    out = capsys.readouterr().out
    assert PRIMERA not in out and SEGUNDA not in out, out
    assert "«oculto»" in out


# --- 2. una key que el runtime rechaza no sale verde ni se prueba -------------

KEYS_RARAS = (
    "sk-AAAA-secreto-1234​",            # espacio invisible: strip() no lo quita
    "“sk-AAAA-secreto-1234”",      # comillas tipográficas
    "sk-AAAA\nsecreto-1234",                 # partida en dos líneas
    "sk-AAAA\tsecreto-1234",                 # TAB: http.client sí la manda
)


@pytest.mark.parametrize("key", KEYS_RARAS)
def test_una_key_que_el_runtime_rechaza_es_roja_y_no_se_manda(nube_local, monkeypatch, key):
    assert not glm._key_cabe_en_header(key), "premisa: el runtime la rechaza en cada turno"
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", key)
    checks = _checks()
    llm_key = checks["llm-key"]
    assert not llm_key.ok and llm_key.required
    assert "caracteres raros" in llm_key.detail and "CROTOLAMO_GLM_API_KEY" in llm_key.detail
    assert "~/.config/crotolamo/env" in llm_key.fix
    nube = checks["llm-nube"]
    assert not nube.ok and nube.required
    assert "reventó" not in nube.detail and "bug del doctor" not in nube.fix
    assert "caracteres raros" in nube.detail
    assert [ruta for ruta, _ in nube_local.vistas if ruta.endswith("/models")] == [], \
        "una key que Crotolamo no manda tampoco la manda el doctor"
    for texto in (llm_key.detail, llm_key.fix, nube.detail, nube.fix):
        assert "secreto" not in texto


def test_una_key_buena_sigue_verde_contra_la_nube_real(nube_local, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "sk-AAAA-secreto-1234")
    checks = _checks()
    assert checks["llm-key"].ok and checks["llm-nube"].ok and not checks["llm-nube"].warn
    assert ("/v1/models", "Bearer sk-AAAA-secreto-1234") in nube_local.vistas


# --- 3. la sonda no sigue redirecciones (el runtime tampoco) -------------------

@pytest.mark.parametrize("codigo", [301, 302, 307, 308])
def test_la_sonda_de_la_nube_no_sigue_redirecciones(servir, codigo):
    destino = servir(_nube_y_ollama)
    origen = servir(lambda path: (codigo, {"Location": f"{destino.raiz}{path}"}, b""))
    check = doctor._probe_cloud(f"{origen.raiz}/v1", "glm-4.7-flash", "sk-clave")
    assert not check.ok, check.detail
    assert "redirige" in check.detail and str(codigo) in check.detail
    assert f"{destino.raiz}/v1/models" in check.detail
    assert "[llm.glm].base_url" in check.fix
    assert origen.vistas == [("/v1/models", "Bearer sk-clave")]
    assert destino.vistas == [], "el Bearer no se reenvía al destino de la redirección"


# --- 4. el host de Ollama con barra final -------------------------------------

def test_ollama_con_barra_final_pide_api_tags_como_el_runtime(mundo, monkeypatch):
    mundo.settings.raw["llm"]["host"] = "http://localhost:11434/"
    mundo.ollama = json.dumps({"models": [{"name": "qwen2.5-coder:7b"}]}).encode()
    pedidas: list[str] = []
    falso = doctor._urlopen

    def urlopen(req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        pedidas.append(url)
        if "//api/" in url:  # el router de Ollama (gin) no colapsa la doble barra
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return falso(req, timeout=timeout)

    monkeypatch.setattr(doctor, "_urlopen", urlopen)
    checks = _checks()
    runtime = LLMClient.from_settings(mundo.settings).host
    assert f"{runtime}/api/tags" in pedidas and not any("//api/" in u for u in pedidas)
    assert checks["ollama"].ok and f"responde en {runtime} (" in checks["ollama"].detail
    assert checks["modelo"].ok


# --- 5. la caducidad con basura que no es texto -------------------------------

@pytest.mark.parametrize("raw", [20261129, ["2026-11-29"], True, 2026.5, time(7, 32)])
def test_caducidad_con_basura_que_no_es_texto_avisa(mundo, tmp_path, raw):
    mundo.settings = _settings(tmp_path, musica_caduca=raw)
    caduca = _checks()["musica-caduca"]
    assert not caduca.ok and not caduca.required
    assert "no es una fecha" in caduca.detail and repr(raw) in caduca.detail


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_caducidad_sin_fecha_sigue_omitida(raw):
    assert doctor._as_date(raw) is None


def test_as_date_acepta_fechas_toml_e_iso():
    assert doctor._as_date(date(2026, 11, 29)) == date(2026, 11, 29)
    assert doctor._as_date(" 2026-11-29 ") == date(2026, 11, 29)
