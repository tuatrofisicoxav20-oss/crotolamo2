"""Revisión final, unidad 23: crotolamo/tools/desktop.py y crotolamo/tools/facts.py.

1. Un comando vacío en [apps] (terminal = "" o []) era una lista vacía para
   open_app: run_detached([]) lanzaba IndexError desde Popen (no es OSError,
   el bucle no lo atrapaba) y "abre terminal" acababa en "La tool 'open_app'
   reventó" aunque terminal_exec, con la MISMA config, caía a la autodetección.
2. resolve_target guardaba un dominio pelón ("miblog.com") como búsqueda en
   Google, mientras open_url abría ese mismo texto como sitio: el atajo a "un
   sitio" abría resultados de Google en cada disparo.
3. _detect_terminal no conocía ptyxis, el terminal de fábrica de Fedora desde
   la 41, aunque terminal_exec ya sabía su sintaxis.
4. send_hotkey, si ydotool fallaba, mandaba a `systemctl --user start ydotoold`,
   una unidad que el paquete de Fedora no trae (trae la de sistema
   ydotool.service), y tiraba lo que ydotool explicaba en su salida.
5. remember_fact guardaba copias exactas: cada una ocupaba uno de los 30
   huecos de hechos fijados en el prompt y sacaba de él un hecho distinto. Lo
   arregla la capa de persistencia (unidad 17); aquí se fija desde la tool.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.tools import default_registry, desktop, shortcuts
from crotolamo.tools import facts as facts_tools

_INNER = ["bash", "-lc", "echo hola"]


@pytest.fixture
def apps_cfg(monkeypatch) -> dict:
    """[apps] del singleton de settings: vacío y rellenable por cada test."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    apps: dict = {}
    monkeypatch.setitem(real.raw, "apps", apps)
    return apps


@pytest.fixture
def db_tmp(monkeypatch, tmp_path) -> Path:
    """La sqlite de atajos y hechos sobre un fichero temporal (nunca ~/.crotolamo)."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    path = tmp_path / "u23.sqlite"
    monkeypatch.setitem(real.paths, "db", path)
    return path


@pytest.fixture
def lanzados(monkeypatch) -> list[list[str]]:
    """Captura lo que se lanzaría, sin lanzar nada."""
    args: list[list[str]] = []
    monkeypatch.setattr(desktop, "run_detached", args.append)
    monkeypatch.setattr(desktop, "_flatpak_app_missing", lambda _args: False)
    return args


def _solo_instalado(monkeypatch, *binarios: str) -> None:
    """shutil.which de mentiras: solo existen los `binarios` dados."""
    monkeypatch.setattr(desktop.shutil, "which",
                        lambda cmd, *a, **kw: f"/usr/bin/{cmd}" if cmd in binarios else None)


def _binario_falso(bindir: Path, nombre: str, cuerpo: str) -> Path:
    """Un ejecutable `nombre` en `bindir` que corre el script sh `cuerpo`."""
    bindir.mkdir(exist_ok=True)
    path = bindir / nombre
    path.write_text("#!/bin/sh\n" + cuerpo, encoding="utf-8")
    path.chmod(0o755)
    return path


def _espera(condicion, timeout: float = 15.0) -> bool:
    """Sondea hasta que se cumpla (margen largo: la máquina puede ir cargada)."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if condicion():
            return True
        time.sleep(0.02)
    return bool(condicion())


# ---------------------------------------------------------------------------
# 1. Comando vacío en [apps]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cableado", [[], "", "   "],
                         ids=["lista-vacia", "cadena-vacia", "solo-espacios"])
def test_open_app_terminal_vacio_en_apps_cae_a_la_autodeteccion(apps_cfg, monkeypatch,
                                                                lanzados, cableado):
    """La misma config con la que terminal_exec ya caía a la autodetección."""
    apps_cfg["terminal"] = cableado
    _solo_instalado(monkeypatch, "kitty")
    out = desktop.open_app("terminal")
    assert lanzados == [["kitty"]]  # y no run_detached([])
    assert "Abrí terminal" in out
    assert desktop.terminal_exec("echo hola") == ["kitty", *_INNER]


def test_open_app_terminal_vacio_de_punta_a_punta_no_revienta(apps_cfg, monkeypatch, tmp_path):
    """Por el registro real y con el Popen de verdad: antes IndexError desde
    Popen([]) -> "La tool 'open_app' reventó" y no se abría nada."""
    marca = tmp_path / "kitty-abierto"
    bindir = tmp_path / "bin"
    _binario_falso(bindir, "kitty", f'echo "$@" > "{marca}"\n')
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(desktop, "_scope_cache", [])  # sin systemd-run de por medio
    apps_cfg["terminal"] = []
    out = default_registry().run("open_app", {"name": "terminal"})
    assert "reventó" not in out
    assert "Abrí terminal" in out
    assert _espera(marca.exists), "el terminal autodetectado no se lanzó"


def test_app_vacia_en_apps_no_cuenta_como_app(apps_cfg, lanzados):
    """Una entrada vacía que no es el terminal no tiene a qué caer: es como si
    no estuviera (antes "Abrí vacia" tras lanzar una lista vacía)."""
    apps_cfg["vacia"] = ""
    apps_cfg["lista vacia"] = []
    apps_cfg["discord"] = ["discord"]
    out = desktop.open_app("vacia")
    assert lanzados == []
    assert out.startswith("No tengo registrada la app 'vacia'")
    conocidas = out.split("Conozco:")[1]
    assert "vacia" not in conocidas and "discord" in conocidas
    assert desktop.resolve_target("lista vacia")[0] == "search"
    assert desktop.resolve_target("discord") == ("app", {"value": "discord"})


def test_atajo_al_terminal_con_terminal_vacio_abre_el_detectado(apps_cfg, monkeypatch,
                                                                 lanzados, db_tmp):
    apps_cfg["terminal"] = []
    _solo_instalado(monkeypatch, "kitty")
    shortcuts.learn_shortcut("consola", "terminal")
    out = shortcuts.run_shortcut("consola")
    assert lanzados == [["kitty"]]
    assert "Abrí terminal" in out


# ---------------------------------------------------------------------------
# 2. Dominio pelón como destino de atajo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target, url", [
    ("miblog.com", "https://miblog.com"),
    ("www.miblog.com", "https://www.miblog.com"),
    ("reddit.com/r/fedora", "https://reddit.com/r/fedora"),
    (" miblog.com ", "https://miblog.com"),
], ids=["dominio", "www", "con-ruta", "con-espacios-alrededor"])
def test_resolve_target_dominio_pelon_es_url(apps_cfg, target, url):
    assert desktop.resolve_target(target) == ("url", {"value": url})


@pytest.fixture
def solo_xdg_open(monkeypatch, lanzados) -> list[list[str]]:
    """Sin flatpak: lo que open_url abre va derecho a xdg-open (capturado)."""
    _solo_instalado(monkeypatch, "xdg-open")
    return lanzados


def test_atajo_a_un_dominio_abre_el_sitio(apps_cfg, solo_xdg_open, db_tmp):
    """De punta a punta: el caso del hallazgo."""
    shortcuts.learn_shortcut("mi blog", "miblog.com")
    shortcuts.run_shortcut("mi blog")
    assert solo_xdg_open == [["xdg-open", "https://miblog.com"]]  # no google.com/search


@pytest.mark.parametrize("target", [
    "miblog.com", "reddit.com/r/fedora", "HTTPS://Example.com/Ruta", "gatos", "gatos persas",
])
def test_atajo_y_open_url_abren_lo_mismo(apps_cfg, solo_xdg_open, db_tmp, target):
    """Una sola regla para los dos: lo que abre el atajo es lo que abre open_url."""
    desktop.open_url(target)
    directo = list(solo_xdg_open)
    solo_xdg_open.clear()
    shortcuts.learn_shortcut("atajo", target)
    shortcuts.run_shortcut("atajo")
    assert solo_xdg_open == directo


def test_resolve_target_lo_que_no_es_dominio_no_cambia(apps_cfg):
    apps_cfg["code.oss"] = ["code-oss"]
    # Una app de [apps] con punto sigue siendo app (se mira antes que el dominio).
    assert desktop.resolve_target("code.oss") == ("app", {"value": "code.oss"})
    assert desktop.resolve_target("latin mafia")[0] == "search"
    assert desktop.resolve_target("mi blog.com nuevo")[0] == "search"  # con espacio: frase
    assert desktop.resolve_target("youtube") == ("url", {"value": "https://www.youtube.com"})


# ---------------------------------------------------------------------------
# 3. ptyxis, el terminal de fábrica de Fedora
# ---------------------------------------------------------------------------

def test_open_app_terminal_detecta_ptyxis(apps_cfg, monkeypatch, lanzados):
    _solo_instalado(monkeypatch, "ptyxis")
    out = desktop.open_app("terminal")
    assert lanzados == [["ptyxis"]]
    assert "Abrí terminal" in out
    assert desktop.terminal_exec("echo hola") == ["ptyxis", "--", *_INNER]


@pytest.mark.parametrize("instalados, esperado", [
    (("kitty", "ptyxis"), "kitty"),
    (("foot", "ptyxis"), "foot"),
    (("gnome-terminal", "ptyxis"), "gnome-terminal"),
    (("ptyxis", "konsole", "xterm"), "ptyxis"),
], ids=["kitty-gana", "foot-gana", "gnome-terminal-gana", "ptyxis-antes-que-konsole"])
def test_ptyxis_no_le_quita_la_prioridad_a_los_de_antes(apps_cfg, monkeypatch, instalados,
                                                        esperado):
    """Los de tiling siguen primero: ptyxis entra justo tras gnome-terminal."""
    _solo_instalado(monkeypatch, *instalados)
    assert desktop._detect_terminal() == [esperado]


# ---------------------------------------------------------------------------
# 4. send_hotkey: la pista de fallo y el diagnóstico de ydotool
# ---------------------------------------------------------------------------

# Lo que imprime el cliente de ydotool 1.0.x (Client/ydotool.c) cuando el socket
# del demonio es de root (la unidad de sistema de Fedora).
_YDOTOOL_EACCES = (
    "failed to connect socket `/tmp/.ydotool_socket': Permission denied\n"
    "Please check if ydotoold is running and you have sufficient permissions.\n"
)


@pytest.fixture
def ydotool_falso(monkeypatch, tmp_path):
    """Pone en el PATH un `ydotool` hecho con el script sh dado."""
    bindir = tmp_path / "bin"

    def hacer(cuerpo: str) -> None:
        _binario_falso(bindir, "ydotool", cuerpo)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    return hacer


def test_send_hotkey_fallo_no_manda_a_una_unidad_que_fedora_no_trae(ydotool_falso, caplog):
    ydotool_falso(f"cat <<'EOF'\n{_YDOTOOL_EACCES}EOF\nexit 2\n")
    with caplog.at_level(logging.DEBUG, logger="crotolamo.desktop"):
        out = desktop.send_hotkey("alt+tab")
    assert out.startswith("Falló ydotool, patrón.")
    assert "systemctl --user start ydotoold" not in out
    assert "ydotoold" in out  # sigue diciendo qué mirar
    # Lo que ydotool explicó queda en el log, que sin eso no hay qué diagnosticar.
    avisos = [r for r in caplog.records
              if r.name == "crotolamo.desktop" and r.levelno >= logging.WARNING]
    assert any("Permission denied" in r.getMessage() for r in avisos), caplog.text
    assert any("sufficient permissions" in r.getMessage() for r in avisos), caplog.text


def test_send_hotkey_fallo_por_stderr_tambien_se_loguea(ydotool_falso, caplog):
    ydotool_falso("echo 'ydotoold backend unavailable' >&2\nexit 1\n")
    with caplog.at_level(logging.DEBUG, logger="crotolamo.desktop"):
        out = desktop.send_hotkey("super")
    assert out.startswith("Falló ydotool, patrón.")
    assert "backend unavailable" in caplog.text


def test_send_hotkey_sin_ydotool(monkeypatch):
    _solo_instalado(monkeypatch)
    assert desktop.send_hotkey("alt+tab") == (
        "No tengo ydotool, patrón. Instálalo con `sudo dnf install ydotool`.")


def test_send_hotkey_timeout(monkeypatch, caplog):
    _solo_instalado(monkeypatch, "ydotool")

    def colgado(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 5))

    monkeypatch.setattr(desktop.subprocess, "run", colgado)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.desktop"):
        out = desktop.send_hotkey("alt+tab")
    assert out == "No pude enviar el atajo, patrón."
    assert "timed out" in caplog.text


def test_send_hotkey_exito_manda_los_keycodes(ydotool_falso, tmp_path):
    registro = tmp_path / "ydotool-args"
    ydotool_falso(f'echo "$@" > "{registro}"\n')
    assert desktop.send_hotkey("Alt + Tab") == "Envié Alt + Tab, patrón."
    assert registro.read_text(encoding="utf-8").split() == [
        "key", "56:1", "15:1", "15:0", "56:0"]


# ---------------------------------------------------------------------------
# 5. remember_fact sin copias. El arreglo vive en la capa de persistencia
# (facts.remember -> db.add_fact_unique, unidad 17), que comparten esta tool y el
# respaldo SQLite de recordar_de_mi; aquí se fija lo que ve la TOOL de fábrica.
# ---------------------------------------------------------------------------

def _textos(db_tmp: Path) -> list[tuple[str, str]]:
    from crotolamo.persistence import facts

    return [(r["texto"], r["categoria"]) for r in facts.recall(db_path=db_tmp)]


def test_remember_fact_repetido_deja_una_sola_fila(db_tmp):
    for _ in range(3):
        out = facts_tools.remember_fact("mi editor es geany", "preferencias")
        assert out == "Anotado, patrón. Lo recordaré: «mi editor es geany»."
    assert _textos(db_tmp) == [("mi editor es geany", "preferencias")]
    assert facts_tools.recall_facts().count("mi editor es geany") == 1


@pytest.mark.parametrize("variante", [
    "Mi editor es geany", "mi editor es geany ", "mi  editor es  geany", "mi editor es geany.",
    "MI EDITOR ES GEANY",
], ids=["mayuscula", "espacio-final", "espacios-dobles", "punto-final", "gritado"])
def test_remember_fact_variante_trivial_tambien_es_repetido(db_tmp, variante):
    facts_tools.remember_fact("mi editor es geany", "preferencias")
    facts_tools.remember_fact(variante, "preferencias")
    assert [t for t, _ in _textos(db_tmp)] == [variante.strip()]  # la redacción nueva


def test_remember_fact_repetido_ocupa_un_solo_hueco_fijado(db_tmp):
    from crotolamo.persistence import facts

    for _ in range(3):
        facts_tools.remember_fact("mi editor es geany", "preferencias")
    facts_tools.remember_fact("uso Fedora", "sistema")
    assert facts.facts_context(db_path=db_tmp) == (
        "- (preferencias) mi editor es geany\n- (sistema) uso Fedora")


def test_remember_fact_repetido_vuelve_a_ser_el_mas_reciente(db_tmp):
    """Con más de 30 hechos, re-anotar uno viejo lo devolvía a la ventana
    fijada (fila nueva). Saltárselo lo habría dejado fuera: se reemplaza."""
    from crotolamo.persistence import facts

    facts_tools.remember_fact("mi perro se llama Tletl", "datos")
    for i in range(35):
        facts_tools.remember_fact(f"hecho de relleno número {i}", "relleno")
    assert "Tletl" not in facts.facts_context(db_path=db_tmp)

    facts_tools.remember_fact("mi perro se llama Tletl", "datos")
    filas = facts.recall(db_path=db_tmp)
    assert [r["texto"] for r in filas].count("mi perro se llama Tletl") == 1
    assert filas[-1]["texto"] == "mi perro se llama Tletl"
    assert facts.facts_context(db_path=db_tmp).count("Tletl") == 1


def test_remember_fact_hechos_distintos_siguen_siendo_dos(db_tmp):
    facts_tools.remember_fact("mi editor es geany", "preferencias")
    assert facts_tools.remember_fact("mi editor es vim", "preferencias").startswith("Anotado")
    facts_tools.remember_fact("mi editor es geany", "proyectos")  # mismo texto: no suma
    assert _textos(db_tmp) == [("mi editor es vim", "preferencias"),
                               ("mi editor es geany", "proyectos")]
