"""Revisión final, unidad 27: crotolamo/tools/windows.py y crotolamo/tools/system.py.

1. Las tools de ventanas no encontraban el Hyprland que estaba corriendo.
   crotolamo.service arranca con default.target, ANTES de que Hyprland exista
   (o exporte su firma), y el proceso se queda con el entorno con el que nació:
   sin HYPRLAND_INSTANCE_SIGNATURE, o con la de un Hyprland que ya se cerró.
   _run solo corría `hyprctl ...` con ese entorno, así que list_windows,
   app_status, focus_window y close_window contestaban "hyprctl se quejó,
   patrón: HYPRLAND_INSTANCE_SIGNATURE not set!" tras CADA login, aunque la
   instancia viva estaba en $XDG_RUNTIME_DIR/hypr/<firma>/. Ahora, si la firma
   del entorno no es de una instancia viva, se busca la viva más reciente y se
   le pasa a hyprctl con `-i <firma>`.

2. system_status prometía "disco, RAM y carga" (es la descripción que ve el
   LLM, y las preguntas de CPU la eligen) pero no decía la carga.

3. Workspaces: una ventana en un workspace especial (scratchpad) se contaba
   como "workspace -98", que al patrón no le dice nada; y app_status ordenaba
   los ids como texto ("los workspaces 10, 2").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from crotolamo.tools import system, windows

# ---------------------------------------------------------------------------
# 1. hyprctl de mentiras, calcado de hyprctl/src/main.cpp: con `-i <firma>` usa
# esa firma; sin -i, la del entorno, y sin ella sale con 1 y la queja de
# siempre. Para conectarse necesita el socket de la firma Y que su Hyprland
# siga vivo (un socket que dejó un Hyprland que murió rechaza la conexión).
# ---------------------------------------------------------------------------

_SIN_FIRMA = "HYPRLAND_INSTANCE_SIGNATURE not set! (is hyprland running?)"

_FAKE_HYPRCTL = r'''
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["FAKE_HYPR_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
firma = None
if args[:1] in (["-i"], ["--instance"]):
    firma, args = args[1], args[2:]
if firma is None:
    firma = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if not firma:
        print("HYPRLAND_INSTANCE_SIGNATURE not set! (is hyprland running?)")
        sys.exit(1)
carpeta = Path(os.environ["XDG_RUNTIME_DIR"]) / "hypr" / firma
socket = carpeta / ".socket.sock"
vivo = socket.exists()
if vivo:
    try:
        os.kill(int((carpeta / "hyprland.lock").read_text().split()[0]), 0)
    except ProcessLookupError:
        vivo = False
    except (OSError, ValueError, IndexError):
        pass
if not vivo:
    print(f"Couldn't connect to {socket}. (4)")
    sys.exit(2)
if args[:2] == ["clients", "-j"]:
    print((carpeta / "clients.json").read_text())
elif args[:1] == ["dispatch"]:
    print("ok")
else:
    print("unknown request")
'''


def _pid_muerto() -> int:
    """PID de un proceso que ya terminó (y que ya se recogió)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class Hypr:
    """$XDG_RUNTIME_DIR/hypr de mentiras + hyprctl falso en el PATH."""

    def __init__(self, base: Path):
        self.runtime = base / "runtime"
        self.runtime.mkdir()
        self.log = base / "hyprctl.log"
        self.log.touch()

    def instancia(self, firma: str, *, pid: int | None = None, mtime: float = 1_000.0,
                  socket: bool = True, lock: bool = True,
                  ventanas: list[dict] | None = None) -> None:
        """Carpeta de una instancia como la deja Hyprland al arrancar."""
        carpeta = self.runtime / "hypr" / firma
        carpeta.mkdir(parents=True)
        if lock:
            (carpeta / "hyprland.lock").write_text(
                f"{os.getpid() if pid is None else pid}\nwayland-1\n")
        (carpeta / "clients.json").write_text(json.dumps(ventanas or [{
            "class": "Opera-GX", "title": "YouTube", "address": "0x1",
            "workspace": {"id": 2, "name": "2"},
        }]))
        if socket:
            sock = carpeta / ".socket.sock"
            sock.touch()
            os.utime(sock, (mtime, mtime))

    def llamadas(self) -> list[list[str]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def hypr(tmp_path, monkeypatch) -> Hypr:
    estado = Hypr(tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = tmp_path / "fake_hyprctl.py"
    script.write_text(_FAKE_HYPRCTL)
    hyprctl = bindir / "hyprctl"
    # Un sh que llama a ESTE intérprete: un shebang con la ruta del venv puede
    # pasarse del límite de longitud del kernel.
    hyprctl.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    hyprctl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(estado.runtime))
    monkeypatch.setenv("FAKE_HYPR_LOG", str(estado.log))
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    return estado


_UNA_VENTANA = "Tienes 1 ventana abierta, patrón:\nworkspace 2: Opera-GX — YouTube"


def test_servicio_sin_firma_encuentra_el_hyprland_que_esta_corriendo(hypr):
    """El caso de cada login: el servicio nació antes que Hyprland."""
    hypr.instancia("abc_123_456")
    assert windows.list_windows() == _UNA_VENTANA
    assert hypr.llamadas() == [["-i", "abc_123_456", "clients", "-j"]]


def test_firma_de_un_hyprland_que_se_cerro_usa_la_instancia_nueva(hypr, monkeypatch):
    """Hyprland se reinició: al salir borró su socket, la firma del entorno es vieja."""
    hypr.instancia("vieja_100_1", socket=False)
    hypr.instancia("nueva_200_2")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "vieja_100_1")
    assert windows.list_windows() == _UNA_VENTANA
    assert hypr.llamadas() == [["-i", "nueva_200_2", "clients", "-j"]]


def test_firma_de_un_hyprland_que_murio_usa_la_instancia_nueva(hypr, monkeypatch):
    """Hyprland se cayó: su socket sigue ahí, pero el proceso del lock ya no existe."""
    hypr.instancia("muerta_100_1", pid=_pid_muerto(), mtime=5_000.0)
    hypr.instancia("nueva_200_2", mtime=2_000.0)
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "muerta_100_1")
    assert windows.list_windows() == _UNA_VENTANA
    assert hypr.llamadas() == [["-i", "nueva_200_2", "clients", "-j"]]


def test_varias_instancias_vivas_usa_la_mas_reciente(hypr):
    hypr.instancia("antigua_100_1", mtime=1_000.0)
    hypr.instancia("reciente_300_3", mtime=3_000.0)
    hypr.instancia("intermedia_200_2", mtime=2_000.0)
    # Más nueva pero muerta: no cuenta.
    hypr.instancia("caida_400_4", pid=_pid_muerto(), mtime=4_000.0)
    # Sin '_' hyprctl leería `-i cache` como NÚMERO de instancia: no es una firma.
    hypr.instancia("cache", mtime=9_000.0)
    windows.list_windows()
    assert hypr.llamadas() == [["-i", "reciente_300_3", "clients", "-j"]]


def test_instancia_sin_lock_legible_cuenta_como_viva(hypr):
    """Sin el lock no hay cómo saber si murió: que lo intente hyprctl."""
    hypr.instancia("sinlock_100_1", lock=False)
    assert windows.list_windows() == _UNA_VENTANA
    assert hypr.llamadas() == [["-i", "sinlock_100_1", "clients", "-j"]]


def test_firma_del_entorno_viva_se_usa_como_siempre(hypr, monkeypatch):
    """Con la firma buena en el entorno no cambia nada: ni -i ni otra instancia."""
    hypr.instancia("buena_100_1", mtime=1_000.0)
    hypr.instancia("otra_200_2", mtime=2_000.0)
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "buena_100_1")
    assert windows.list_windows() == _UNA_VENTANA
    assert hypr.llamadas() == [["clients", "-j"]]


@pytest.mark.parametrize("hay_carpeta", [False, True], ids=["sin_hypr", "hypr_vacia"])
def test_sin_ninguna_instancia_queda_el_mensaje_de_siempre(hypr, hay_carpeta):
    if hay_carpeta:
        (hypr.runtime / "hypr").mkdir()
    assert windows.list_windows() == f"hyprctl se quejó, patrón: {_SIN_FIRMA}"
    assert hypr.llamadas() == [["clients", "-j"]]


def test_enfocar_y_cerrar_tambien_van_a_la_instancia_viva(hypr):
    hypr.instancia("abc_123_456")
    assert "enfocada Opera-GX" in windows.focus_window("opera")
    assert "Cerrada" in windows.close_window("opera")
    assert hypr.llamadas() == [
        ["-i", "abc_123_456", "clients", "-j"],
        ["-i", "abc_123_456", "dispatch", "focuswindow", "address:0x1"],
        ["-i", "abc_123_456", "clients", "-j"],
        ["-i", "abc_123_456", "dispatch", "closewindow", "address:0x1"],
    ]


def test_sin_xdg_runtime_dir_busca_donde_busca_hyprctl(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert windows._dir_instancias() == Path(f"/run/user/{os.getuid()}") / "hypr"


# ---------------------------------------------------------------------------
# 2. system_status: la carga que promete su descripción.
# ---------------------------------------------------------------------------


def test_system_status_dice_la_carga(monkeypatch):
    monkeypatch.setattr(system.os, "getloadavg", lambda: (0.52, 1.5, 2.25))
    monkeypatch.setattr(system.os, "cpu_count", lambda: 8)
    out = system.system_status()
    assert "Disco" in out and "RAM" in out
    assert "Carga: 0.52 el último minuto, 1.50 en 5 y 2.25 en 15, con 8 núcleos, patrón." in (
        out.splitlines())


def test_system_status_sin_numero_de_nucleos(monkeypatch):
    monkeypatch.setattr(system.os, "getloadavg", lambda: (3.0, 2.0, 1.0))
    monkeypatch.setattr(system.os, "cpu_count", lambda: None)
    out = system.system_status()
    assert "Carga: 3.00 el último minuto, 2.00 en 5 y 1.00 en 15, patrón." in out.splitlines()


def test_system_status_sin_carga_disponible_da_lo_demas(monkeypatch):
    def sin_carga():
        raise OSError("Load averages are unobtainable")

    monkeypatch.setattr(system.os, "getloadavg", sin_carga)
    out = system.system_status()
    assert "Disco" in out and "RAM" in out
    assert "Carga" not in out


def test_system_status_real_trae_la_carga():
    """Sin fakes: en Linux /proc/loadavg siempre está."""
    assert any(line.startswith("Carga: ") for line in system.system_status().splitlines())


# ---------------------------------------------------------------------------
# 3. Workspaces especiales y orden numérico.
# ---------------------------------------------------------------------------


def _kitty(titulo: str, address: str, workspace: dict) -> dict:
    return {"class": "kitty", "title": titulo, "address": address, "workspace": workspace}


@pytest.fixture
def con_scratchpad(monkeypatch):
    ventanas = [
        _kitty("a", "0xa", {"id": 10, "name": "10"}),
        _kitty("b", "0xb", {"id": 2, "name": "2"}),
        _kitty("c", "0xc", {"id": -98, "name": "special:magic"}),
    ]
    monkeypatch.setattr(windows, "_clients", lambda: ventanas)
    return ventanas


def test_app_status_scratchpad_por_nombre_y_workspaces_en_orden_numerico(con_scratchpad):
    assert windows.app_status("kitty") == (
        "Sí, patrón: 'kitty' está abierta con 3 ventanas en los workspaces 2, 10, special:magic.")


def test_app_status_ordena_los_workspaces_como_numeros(con_scratchpad):
    del con_scratchpad[2]
    assert windows.app_status("kitty") == (
        "Sí, patrón: 'kitty' está abierta con 2 ventanas en los workspaces 2, 10.")


def test_list_windows_scratchpad_por_nombre(con_scratchpad):
    assert windows.list_windows() == (
        "Tienes 3 ventanas abiertas, patrón:\n"
        "workspace 2: kitty — b\n"
        "workspace 10: kitty — a\n"
        "workspace special:magic: kitty — c")


def test_focus_window_en_scratchpad_dice_su_nombre(con_scratchpad, monkeypatch):
    monkeypatch.setattr(windows, "_dispatch", lambda args: subprocess.CompletedProcess(
        ["hyprctl", "dispatch", *args], 0, "ok\n", ""))
    con_scratchpad[:] = [con_scratchpad[2]]
    assert windows.focus_window("kitty") == (
        "Listo, patrón: enfocada kitty — c (workspace special:magic).")


def test_workspace_negativo_sin_nombre_conserva_el_id():
    assert windows._workspace_id(_kitty("x", "0x1", {"id": -1})) == -1
