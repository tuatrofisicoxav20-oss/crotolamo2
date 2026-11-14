"""Auditoría exhaustiva — grupo integración de escritorio (desktop/*.sh, units de
systemd, lanzadores .desktop, launch.sh).

Cada test fija un hallazgo confirmado de la auditoría: falla con el código de
antes y pasa con el arreglo. Los comentarios dicen QUÉ se rompía. Los scripts se
ejecutan DE VERDAD con bash, pero contra dobles de `pactl`/`systemctl`/`sleep`
puestos delante en el PATH y con un HOME temporal: ni PipeWire ni systemd ni el
home real se tocan.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

from scripts import crotolamo_doctor as doctor

RAIZ = Path(__file__).resolve().parent.parent
DESKTOP = RAIZ / "desktop"
AEC_SH = DESKTOP / "aec.sh"
AEC_CONF = DESKTOP / "pipewire-aec.conf"

# pactl de mentiras. Estado en $FAKE_DIR: `aec` = el source del AEC existe,
# `modulo` = hay un module-echo-cancel de prueba cargado. El source del AEC sale
# el PRIMERO y detrás van FAKE_SOURCES líneas más, cada una con su write(): lo
# que escribe después del match es justo lo que delataba el `grep -q`.
_PACTL = r"""#!/bin/sh
echo "$*" >> "$FAKE_DIR/pactl.log"
if [ -n "${FAKE_PACTL_ROTO:-}" ]; then
    echo "Connection failure: Connection refused" >&2
    exit 1
fi
case "$1" in
    list)
        [ -e "$FAKE_DIR/aec" ] && \
            printf '57\tcrotolamo_aec_source\tPipeWire\tfloat32le 1ch 48000Hz\tRUNNING\n'
        [ -e "$FAKE_DIR/modulo" ] && \
            printf '58\tcrotolamo_aec_test\tPipeWire\tfloat32le 1ch 48000Hz\tRUNNING\n'
        i=0
        while [ "$i" -lt "${FAKE_SOURCES:-3}" ]; do
            printf '%d\talsa_input.usb-mic-%d.analog-stereo\tPipeWire\ts16le 2ch 48000Hz\tIDLE\n' \
                "$i" "$i" || exit 141
            i=$((i + 1))
        done
        ;;
    get-default-source) echo "${FAKE_DEFAULT_SOURCE:-alsa_input.pci}" ;;
    get-default-sink) echo "${FAKE_DEFAULT_SINK:-alsa_output.pci}" ;;
    load-module)
        if [ -n "${FAKE_LOAD_FALLA:-}" ]; then
            echo "Failure: Module initialization failed" >&2
            exit 1
        fi
        : > "$FAKE_DIR/modulo"
        echo 42
        ;;
    unload-module) rm -f "$FAKE_DIR/modulo" ;;
esac
exit 0
"""

# systemctl de mentiras: "reiniciar pipewire" deja el source del AEC si su config
# está instalada en el HOME (de prueba) y lo quita si no, como el PipeWire real.
_SYSTEMCTL = r"""#!/bin/sh
echo "$*" >> "$FAKE_DIR/systemctl.log"
if [ -e "$HOME/.config/pipewire/pipewire.conf.d/crotolamo-aec.conf" ]; then
    : > "$FAKE_DIR/aec"
else
    rm -f "$FAKE_DIR/aec"
fi
exit 0
"""


class _Aec:
    """aec.sh de verdad contra el pactl/systemctl de mentiras."""

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "fake"
        self.home = tmp_path / "home"
        self.bin = tmp_path / "bin"
        for carpeta in (self.dir, self.home, self.bin):
            carpeta.mkdir()
        # `sleep` también: restart_pipewire espera 2 s a que PipeWire vuelva.
        for nombre, cuerpo in (("pactl", _PACTL), ("systemctl", _SYSTEMCTL),
                               ("sleep", "#!/bin/sh\nexit 0\n")):
            doble = self.bin / nombre
            doble.write_text(cuerpo, encoding="utf-8")
            doble.chmod(0o755)

    def activo(self) -> None:
        """Deja el source del AEC ya creado (como tras un `aec.sh on` anterior)."""
        (self.dir / "aec").touch()

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        entorno = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DIR": str(self.dir),
            **env,
        }
        return subprocess.run(["bash", str(AEC_SH), *args], env=entorno, capture_output=True,
                              text=True, encoding="utf-8", timeout=60)

    def pactl_log(self) -> list[str]:
        registro = self.dir / "pactl.log"
        return registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []


@pytest.fixture
def aec(tmp_path):
    return _Aec(tmp_path)


# ---------------------------------------------------------------------------
# aec.sh — have_source: `pactl … | grep -q` bajo `set -o pipefail`. grep -q sale
# en el primer match; si pactl sigue escribiendo muere de SIGPIPE (141), pipefail
# convierte eso en "falso" y el script decía "ausente" (y `on` salía con 1 y
# "el source no apareció") con el AEC funcionando. Que acierte no puede depender
# de cuánto ni cuándo escribe pactl.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sources", ["3", "3000"], ids=["pocos-sources", "muchos-sources"])
def test_status_ve_el_source_del_aec_escriba_pactl_lo_que_escriba(aec, sources):
    aec.activo()
    out = aec.run("status", FAKE_SOURCES=sources)
    assert out.returncode == 0, out.stderr
    assert "source 'crotolamo_aec_source': PRESENTE" in out.stdout


def test_status_sin_el_source_dice_ausente(aec):
    out = aec.run("status", FAKE_SOURCES="3000")
    assert out.returncode == 0, out.stderr
    assert "source 'crotolamo_aec_source': ausente" in out.stdout


def test_status_con_pactl_roto_dice_ausente_sin_reventar(aec):
    aec.activo()
    out = aec.run("status", FAKE_PACTL_ROTO="1")
    assert out.returncode == 0, out.stderr
    assert "source 'crotolamo_aec_source': ausente" in out.stdout
    assert "default source del sistema: ?" in out.stdout


def test_on_con_muchos_sources_no_da_el_falso_aviso(aec):
    out = aec.run("on", FAKE_SOURCES="3000")
    assert "no apareció" not in out.stdout
    assert out.returncode == 0, out.stderr
    assert "OK: el source 'crotolamo_aec_source' existe." in out.stdout
    # La config quedó instalada donde PipeWire la lee, idéntica a la del repo.
    instalada = aec.home / ".config/pipewire/pipewire.conf.d/crotolamo-aec.conf"
    assert instalada.read_bytes() == AEC_CONF.read_bytes()


def test_off_con_muchos_sources_y_el_modulo_cargado_a_mano_avisa(aec):
    """Sin config instalada pero con el source vivo: `off` debe decirlo, no "revertido"."""
    aec.activo()
    out = aec.run("off", FAKE_SOURCES="3000")
    assert out.returncode == 0, out.stderr
    assert "AVISO: el source sigue ahí" in out.stdout
    assert "OK: revertido." not in out.stdout


def test_on_y_off_de_punta_a_punta(aec):
    assert aec.run("on").returncode == 0
    out = aec.run("off")
    assert out.returncode == 0, out.stderr
    assert "OK: revertido." in out.stdout
    assert not (aec.home / ".config/pipewire/pipewire.conf.d/crotolamo-aec.conf").exists()


# ---------------------------------------------------------------------------
# aec.sh test — `MOD=$(pactl load-module …)` suelto bajo `set -e`: si pactl
# fallaba, el script moría en la asignación y la rama "FALLO: …" de debajo era
# inalcanzable (solo quedaba el stderr crudo de pactl).
# ---------------------------------------------------------------------------

def test_test_con_el_modulo_que_no_carga_dice_fallo(aec):
    out = aec.run("test", FAKE_LOAD_FALLA="1")
    assert out.returncode == 1
    assert "FALLO" in out.stdout
    assert "cargado" not in out.stdout
    assert not any(linea.startswith("unload-module") for linea in aec.pactl_log())


def test_test_carga_el_modulo_y_lo_descarga(aec):
    out = aec.run("test")
    assert out.returncode == 0, out.stderr
    assert "probando AEC sobre source=alsa_input.pci sink=alsa_output.pci" in out.stdout
    assert "cargado (id=42)" in out.stdout
    assert "crotolamo_aec_test" in out.stdout
    assert "descargado. Sistema como estaba." in out.stdout
    assert "FALLO" not in out.stdout
    assert aec.pactl_log()[-1] == "unload-module 42"
    assert not (aec.dir / "modulo").exists()


# ---------------------------------------------------------------------------
# AEC sin referencia — aec.sh y pipewire-aec.conf decían que no hacía falta tocar
# "el default del sistema" y solo pedían [voice].input_device. Pero el AEC resta
# únicamente lo que suena por crotolamo_aec_sink, y el TTS y la música salen por
# el sink por defecto: siguiendo el script al pie de la letra el source "limpio"
# seguía trayendo el eco, y el doctor marcaba en rojo justo el estado que dejaba
# `aec.sh on`. Los tres sitios tienen que decir lo mismo que el doctor exige.
# ---------------------------------------------------------------------------

_PASO_SINK = "pactl set-default-sink crotolamo_aec_sink"


def test_on_dice_que_el_sink_del_aec_tiene_que_ser_el_default(aec):
    out = aec.run("on")
    assert out.returncode == 0, out.stderr
    assert 'input_device = "crotolamo_aec_source"' in out.stdout
    assert _PASO_SINK in out.stdout


def test_on_con_el_sink_ya_por_defecto_no_pide_cambiarlo(aec):
    out = aec.run("on", FAKE_DEFAULT_SINK="crotolamo_aec_sink")
    assert out.returncode == 0, out.stderr
    assert 'input_device = "crotolamo_aec_source"' in out.stdout
    assert _PASO_SINK not in out.stdout


def test_status_avisa_si_el_sink_del_aec_no_es_el_default(aec):
    aec.activo()
    out = aec.run("status")
    assert out.returncode == 0, out.stderr
    assert "default sink del sistema: alsa_output.pci" in out.stdout
    assert "AVISO" in out.stdout and _PASO_SINK in out.stdout


def test_status_con_el_sink_del_aec_por_defecto_no_avisa(aec):
    aec.activo()
    out = aec.run("status", FAKE_DEFAULT_SINK="crotolamo_aec_sink")
    assert out.returncode == 0, out.stderr
    assert "default sink del sistema: crotolamo_aec_sink" in out.stdout
    assert "AVISO" not in out.stdout and _PASO_SINK not in out.stdout


def test_status_sin_aec_no_pide_cambiar_el_sink(aec):
    """AEC apagado: el sink por defecto se enseña, pero no hay nada que avisar."""
    out = aec.run("status")
    assert out.returncode == 0, out.stderr
    assert "default sink del sistema: alsa_output.pci" in out.stdout
    assert "AVISO" not in out.stdout and _PASO_SINK not in out.stdout


def test_la_config_del_aec_explica_el_paso_del_sink():
    assert _PASO_SINK in AEC_CONF.read_text(encoding="utf-8")


def test_los_nodos_del_aec_se_llaman_igual_en_config_script_y_doctor():
    """Un nombre que se desvíe en uno de los tres y el aviso/el doctor mienten."""
    conf = AEC_CONF.read_text(encoding="utf-8")
    script = AEC_SH.read_text(encoding="utf-8")
    nodos = re.findall(r'^\s*node\.name\s*=\s*"([^"]+)"', conf, re.MULTILINE)
    assert nodos == ["crotolamo_aec_source", "crotolamo_aec_sink"]
    en_script = dict(re.findall(r'^(AEC_SOURCE|AEC_SINK)="([^"]+)"$', script, re.MULTILINE))
    assert en_script == {"AEC_SOURCE": nodos[0], "AEC_SINK": nodos[1]}
    assert doctor._DOCTOR_DEFAULTS["aec_source"] == nodos[0]
    assert doctor._DOCTOR_DEFAULTS["aec_sink"] == nodos[1]


# ---------------------------------------------------------------------------
# install.sh (D7) — copiaba crotolamo.service TAL CUAL, con la ruta del clon de
# siempre (%h/Documentos/crotolamo2) quemada: clonado en otra carpeta, el
# servicio instalado no arrancaba (200/CHDIR, 203/EXEC) y systemd lo reintentaba
# cada 3 s mientras el script decía "encendido". crotolamo-hud.service (misma
# ruta quemada) no se instalaba nunca, y la ruta del panel iba sin comillas en el
# Exec= del lanzador: con un espacio se partía en dos argumentos. Se ejecuta el
# install.sh de verdad desde una copia del repo en otra carpeta, con HOME de
# prueba y systemctl de mentiras.
# ---------------------------------------------------------------------------

_SYSTEMCTL_INSTALL = r"""#!/bin/sh
echo "$*" >> "$FAKE_DIR/systemctl.log"
exit 0
"""

_PLANTILLAS = ("install.sh", "crotolamo.service", "crotolamo-hud.service",
               "crotolamo.desktop.in", "crotolamo.svg")


class _Instalador:
    """Copia de desktop/ en `raiz` (con un .venv de mentira) y su install.sh."""

    def __init__(self, tmp_path: Path, raiz: Path) -> None:
        self.home = tmp_path / "home"
        self.dir = tmp_path / "fake"
        self.bin = tmp_path / "bin"
        self.raiz = raiz
        for carpeta in (self.home, self.dir, self.bin, raiz / "desktop", raiz / ".venv/bin"):
            carpeta.mkdir(parents=True, exist_ok=True)
        for nombre in _PLANTILLAS:
            (raiz / "desktop" / nombre).write_bytes((DESKTOP / nombre).read_bytes())
        python = raiz / ".venv/bin/python"
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        for nombre, cuerpo in (("systemctl", _SYSTEMCTL_INSTALL),
                               ("gtk-update-icon-cache", "#!/bin/sh\nexit 0\n"),
                               ("update-desktop-database", "#!/bin/sh\nexit 0\n")):
            doble = self.bin / nombre
            doble.write_text(cuerpo, encoding="utf-8")
            doble.chmod(0o755)
        self.units = self.home / ".config/systemd/user"

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        entorno = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DIR": str(self.dir),
        }
        return subprocess.run(["bash", str(self.raiz / "desktop/install.sh"), *args],
                              env=entorno, capture_output=True, text=True, encoding="utf-8",
                              timeout=60)

    def unit(self, nombre: str) -> str:
        return (self.units / nombre).read_text(encoding="utf-8")

    def systemctl_log(self) -> list[str]:
        registro = self.dir / "systemctl.log"
        return registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []


def _clave(texto: str, clave: str) -> str:
    valores = [linea.split("=", 1)[1] for linea in texto.splitlines()
               if linea.startswith(f"{clave}=")]
    assert len(valores) == 1, (clave, valores)
    return valores[0]


def _especificadores(valor: str, home: Path) -> str:
    """Lo que systemd hace con %h y %% (los únicos especificadores que usan los units)."""
    def sustituye(m: re.Match[str]) -> str:
        assert m.group(1) in "h%", f"especificador inesperado %{m.group(1)} en {valor!r}"
        return str(home) if m.group(1) == "h" else "%"
    return re.sub(r"%(.)", sustituye, valor)


def _comando(valor: str, home: Path) -> list[str]:
    """Partir un ExecStart= como systemd: por espacios, respetando las comillas."""
    return shlex.split(_especificadores(valor, home))


# Rutas de clon: fuera del HOME con un espacio; dentro del HOME con espacios,
# acento, &, # y %; y la ruta de siempre.
_RAICES = {
    "fuera-del-home": ("fuera", "otra carpeta/crotolamo2"),
    "home-raro": ("home", "Código/mis proyectos/R&D #1 100%/crotolamo2"),
    "la-de-siempre": ("home", "Documentos/crotolamo2"),
}


@pytest.fixture(params=list(_RAICES), ids=list(_RAICES))
def instalador(request, tmp_path):
    base, relativa = _RAICES[request.param]
    raiz = (tmp_path / "home" if base == "home" else tmp_path / "fuera") / relativa
    return _Instalador(tmp_path, raiz)


def test_install_apunta_el_servicio_al_repo_real(instalador):
    out = instalador.run()
    assert out.returncode == 0, out.stdout + out.stderr
    unit = instalador.unit("crotolamo.service")
    home, raiz = instalador.home, instalador.raiz
    assert _especificadores(_clave(unit, "WorkingDirectory"), home) == str(raiz)
    assert _comando(_clave(unit, "ExecStart"), home) == [
        str(raiz / ".venv/bin/python"), "-m", "crotolamo", "listen", "$CROTOLAMO_LISTEN_ARGS"]
    doc = _especificadores(_clave(unit, "Documentation"), home)
    assert doc.startswith("file:") and doc.isascii() and " " not in doc
    assert unquote(doc.removeprefix("file:")) == str(raiz / "README.md")


def test_install_instala_y_habilita_el_hud_con_la_ruta_real(instalador):
    out = instalador.run()
    assert out.returncode == 0, out.stdout + out.stderr
    hud = instalador.unit("crotolamo-hud.service")
    assert _comando(_clave(hud, "ExecStart"), instalador.home) == [
        "/usr/bin/python3", str(instalador.raiz / "desktop/crotolamo_hud.py")]
    habilitados = [linea.split()[2:] for linea in instalador.systemctl_log()
                   if linea.startswith("--user enable")]
    assert any("crotolamo-hud.service" in unidades for unidades in habilitados)
    assert any("crotolamo.service" in unidades for unidades in habilitados)


def test_install_el_lanzador_lleva_la_ruta_del_panel_entera(instalador):
    out = instalador.run()
    assert out.returncode == 0, out.stdout + out.stderr
    lanzador = (instalador.home / ".local/share/applications/crotolamo.desktop").read_text(
        encoding="utf-8")
    # Spec de Desktop Entry: comillas dobles alrededor del argumento y %% = % literal;
    # cualquier otro %X es un código de campo que el lanzador sustituiría.
    def campo(m: re.Match[str]) -> str:
        assert m.group(1) == "%", f"código de campo %{m.group(1)} en el Exec="
        return "%"
    exec_ = shlex.split(re.sub(r"%(.)", campo, _clave(lanzador, "Exec")))
    assert exec_ == ["/usr/bin/python3", str(instalador.raiz / "desktop/crotolamo_panel.py")]
    # El resto del lanzador es la plantilla sin tocar.
    plantilla = (DESKTOP / "crotolamo.desktop.in").read_text(encoding="utf-8")
    quitar_exec = re.compile(r"^Exec=.*\n", re.MULTILINE)
    assert quitar_exec.sub("", lanzador) == quitar_exec.sub("", plantilla)


def test_install_en_la_ruta_de_siempre_deja_los_units_como_la_plantilla(tmp_path):
    """Con el clon de siempre los units instalados son byte a byte los del repo."""
    inst = _Instalador(tmp_path, tmp_path / "home/Documentos/crotolamo2")
    out = inst.run()
    assert out.returncode == 0, out.stdout + out.stderr
    for nombre in ("crotolamo.service", "crotolamo-hud.service"):
        assert (inst.units / nombre).read_bytes() == (DESKTOP / nombre).read_bytes(), nombre
    assert "bak-" not in " ".join(p.name for p in inst.units.iterdir())


def test_install_respalda_un_unit_editado_a_mano_antes_de_pisarlo(tmp_path):
    inst = _Instalador(tmp_path, tmp_path / "home/Documentos/crotolamo2")
    inst.units.mkdir(parents=True)
    editado = "[Service]\nExecStart=/usr/bin/python3 /mi/hud.py\nEnvironment=X=1\n"
    (inst.units / "crotolamo-hud.service").write_text(editado, encoding="utf-8")
    out = inst.run()
    assert out.returncode == 0, out.stdout + out.stderr
    respaldos = list(inst.units.glob("crotolamo-hud.service.bak-*"))
    assert len(respaldos) == 1
    assert respaldos[0].read_text(encoding="utf-8") == editado
    # Volver a instalar sin cambios no deja más respaldos.
    assert inst.run().returncode == 0
    assert len(list(inst.units.glob("*.bak-*"))) == 1


def test_install_start_reinicia_el_servicio_y_el_hud_solo_si_ya_corria(tmp_path):
    inst = _Instalador(tmp_path, tmp_path / "fuera/crotolamo2")
    out = inst.run("--start")
    assert out.returncode == 0, out.stdout + out.stderr
    log = inst.systemctl_log()
    assert "--user restart crotolamo.service" in log
    # El HUD necesita la sesión gráfica: desde una consola sin ella, arrancarlo lo
    # dejaría reintentando; try-restart solo lo reinicia si ya corría.
    assert "--user try-restart crotolamo-hud.service" in log
    assert not any(linea.startswith("--user restart crotolamo-hud") for linea in log)
    assert "Crotolamo encendido" in out.stdout


# ' incluido: systemd rechaza el ExecStart ("Executable name contains special
# characters") aunque vaya entre comillas.
@pytest.mark.parametrize("carpeta", ["con$dolar", 'con"comilla', "d'Ana", "con\\barra",
                                     "con`tilde"])
def test_install_rechaza_una_ruta_que_no_sabe_escribir_en_un_unit(tmp_path, carpeta):
    inst = _Instalador(tmp_path, tmp_path / "fuera" / carpeta / "crotolamo2")
    out = inst.run()
    assert out.returncode != 0
    assert "no sé escribirla" in out.stdout
    assert not inst.units.exists() or not any(inst.units.iterdir())
    assert inst.systemctl_log() == []


def test_el_servicio_es_type_exec():
    """Type=simple da el servicio por arrancado al hacer fork: con el .venv borrado o el
    repo movido, `systemctl restart` salía con 0 (y install.sh decía "encendido")
    aunque el ExecStart no existiera. Type=exec espera al execve y falla de verdad."""
    unit = (DESKTOP / "crotolamo.service").read_text(encoding="utf-8")
    assert _clave(unit, "Type") == "exec"


@pytest.mark.skipif(not shutil.which("systemd-analyze"), reason="sin systemd-analyze")
def test_systemd_acepta_los_units_instalados(instalador, tmp_path):
    """El parser REAL de systemd: ruta ejecutable, comillas, %% y URL de Documentation."""
    out = instalador.run()
    assert out.returncode == 0, out.stdout + out.stderr
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    # Testigo con un ExecStart que no existe: si systemd-analyze no se queja de él,
    # aquí no está verificando nada y el test no probaría nada.
    testigo = instalador.units / "testigo-roto.service"
    testigo.write_text("[Service]\nExecStart=/no/existe/python\n", encoding="utf-8")
    units = [testigo, instalador.units / "crotolamo.service"]
    if os.access("/usr/bin/python3", os.X_OK):
        units.append(instalador.units / "crotolamo-hud.service")
    verify = subprocess.run(
        ["systemd-analyze", "--user", "--man=no", "verify", *map(str, units)],
        env={**os.environ, "HOME": str(instalador.home), "XDG_RUNTIME_DIR": str(run_dir)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    salida = (verify.stdout + verify.stderr).splitlines()
    if not any("testigo-roto.service" in linea for linea in salida):
        pytest.skip("systemd-analyze no puede verificar units de usuario aquí")
    quejas = [linea for linea in salida
              if "crotolamo.service" in linea or "crotolamo-hud.service" in linea]
    assert quejas == []


# ---------------------------------------------------------------------------
# launch.sh — el menú corre con `set -euo pipefail` y llamaba cada opción a pelo
# dentro del `while true`: en cuanto el comando salía con != 0 (el doctor con un
# REQUERIDO en rojo, el listener sin voz, el smoke fallido) errexit mataba el
# launcher entero, sin "(Enter para volver al menú)", justo cuando hacía falta
# probar la siguiente opción. El modo directo (`./launch.sh doctor`) sí tiene que
# seguir devolviendo el código del comando. Se ejecuta el launch.sh de verdad,
# copiado junto a un .venv/bin/python de mentiras y con un HOME de prueba (sin el
# ~/.config/crotolamo/env real).
# ---------------------------------------------------------------------------

_PY_FALSO = r"""#!/bin/sh
echo "$*" >> "$FAKE_DIR/python.log"
exit "${FAKE_RC:-0}"
"""


class _Launcher:
    def __init__(self, tmp_path: Path) -> None:
        self.raiz = tmp_path / "repo"
        self.home = tmp_path / "home"
        self.dir = tmp_path / "fake"
        for carpeta in (self.raiz / ".venv/bin", self.home, self.dir):
            carpeta.mkdir(parents=True)
        (self.raiz / "launch.sh").write_bytes((RAIZ / "launch.sh").read_bytes())
        python = self.raiz / ".venv/bin/python"
        python.write_text(_PY_FALSO, encoding="utf-8")
        python.chmod(0o755)

    def run(self, *args: str, entrada: str = "", **env: str) -> subprocess.CompletedProcess[str]:
        entorno = {**os.environ, "HOME": str(self.home), "FAKE_DIR": str(self.dir), **env}
        return subprocess.run(["bash", str(self.raiz / "launch.sh"), *args], input=entrada,
                              env=entorno, capture_output=True, text=True, encoding="utf-8",
                              timeout=60)

    def python_log(self) -> list[str]:
        registro = self.dir / "python.log"
        return registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []


_OPCIONES = {
    "1": "-m crotolamo doctor",
    "2": "-m crotolamo shell --stream",
    "3": "-m crotolamo listen --no-barge-in",
    "4": "-m crotolamo listen --barge-in",
    "5": "-m crotolamo listen --simple",
    "6": "scripts/smoke_voz.py",
}


@pytest.mark.parametrize("opcion", list(_OPCIONES))
def test_launch_el_menu_sobrevive_a_un_comando_que_falla(tmp_path, opcion):
    launcher = _Launcher(tmp_path)
    # La opción falla, Enter, luego 7 (versión), Enter, y 0 (salir).
    out = launcher.run(entrada=f"{opcion}\n\n7\n\n0\n", FAKE_RC="1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert launcher.python_log() == [_OPCIONES[opcion], "-m crotolamo --version"]
    assert "(salió con código 1)" in out.stdout
    assert "Hasta luego, patrón." in out.stdout


def test_launch_el_menu_no_avisa_si_el_comando_sale_bien(tmp_path):
    launcher = _Launcher(tmp_path)
    out = launcher.run(entrada="1\n\n0\n")
    assert out.returncode == 0, out.stdout + out.stderr
    assert launcher.python_log() == ["-m crotolamo doctor"]
    assert "salió con código" not in out.stdout


def test_launch_el_menu_termina_si_se_acaba_la_entrada(tmp_path):
    """Ctrl-D en el prompt del menú lo cierra: el arreglo no puede dejarlo en bucle."""
    launcher = _Launcher(tmp_path)
    out = launcher.run(entrada="1\n", FAKE_RC="1")
    assert out.stdout.count("¿qué probamos, patrón?") == 2
    assert launcher.python_log() == ["-m crotolamo doctor"]


@pytest.mark.parametrize(("args", "rc"), [(("doctor",), 1), (("smoke",), 3),
                                          (("listen", "--simple"), 2)])
def test_launch_directo_devuelve_el_codigo_del_comando(tmp_path, args, rc):
    launcher = _Launcher(tmp_path)
    out = launcher.run(*args, FAKE_RC=str(rc))
    assert out.returncode == rc, out.stdout + out.stderr
    assert "salió con código" not in out.stdout


# ---------------------------------------------------------------------------
# soloist_key_to_env.sh — no era idempotente. Una 2ª corrida (para comprobar, o
# con --mode env cuando Soloist lo soporte, que es lo que promete la cabecera):
#  · sustituía por el placeholder la key NUEVA que el patrón ya había pegado en
#    ~/.config/soloist/env (o en api_key), sin respaldo de ninguno de los dos;
#  · en modo wrapper volvía a envolver el `sh -c` (comillas rotas: el unit no
#    arrancaba) y en modo file duplicaba --api-key-file;
#  · con una ayuda que anunciaba --key-file escribía --api-key-file igual;
#  · y una key entre comillas sin prefijo spak_ salía en claro en "ExecStart
#    actual (enmascarado)", contra el "NUNCA imprime" de la cabecera.
# Se ejecuta el script de verdad con un HOME de prueba y un soloist de mentiras.
# ---------------------------------------------------------------------------

SOLOIST_SH = DESKTOP / "soloist_key_to_env.sh"
_PLACEHOLDER = "PEGA_AQUI_LA_KEY_NUEVA"
_VIEJA = "spak_VIEJA0123456789"
_NUEVA = "spak_NUEVA_abcdef123456"
_FLAGS = "--device-name crotolamo --ws 127.0.0.1:9090"
_AYUDA_ENV = "  [env: SOLOIST_API_KEY=]   la key por entorno"

# soloist de mentiras: su ayuda + las líneas de $FAKE_HELP.
_SOLOIST = r"""#!/bin/sh
echo "Usage: soloist [OPTIONS]"
echo "  --api-key <KEY>   Spotify API key"
[ -n "${FAKE_HELP:-}" ] && echo "$FAKE_HELP"
exit 0
"""


class _Soloist:
    def __init__(self, tmp_path: Path,
                 exec_start: str = f"%h/.local/bin/soloist --api-key {_VIEJA} {_FLAGS}") -> None:
        self.home = tmp_path / "home"
        binario = self.home / ".local/bin/soloist"
        units = self.home / ".config/systemd/user"
        for carpeta in (binario.parent, units):
            carpeta.mkdir(parents=True)
        binario.write_text(_SOLOIST, encoding="utf-8")
        binario.chmod(0o755)
        self.service = units / "soloist.service"
        self.service.write_text(f"[Unit]\nDescription=Soloist\n\n[Service]\n"
                                f"ExecStart={exec_start}\nRestart=on-failure\n", encoding="utf-8")
        self.env = self.home / ".config/soloist/env"
        self.key_file = self.home / ".config/soloist/api_key"

    def run(self, *args: str, ayuda: str = "") -> subprocess.CompletedProcess[str]:
        entorno = {k: v for k, v in os.environ.items() if not k.startswith("SOLOIST_")}
        entorno.update(HOME=str(self.home), FAKE_HELP=ayuda)
        return subprocess.run(["bash", str(SOLOIST_SH), *args], env=entorno,
                              capture_output=True, text=True, encoding="utf-8", timeout=60)

    def exec_start(self) -> str:
        return _clave(self.service.read_text(encoding="utf-8"), "ExecStart")

    def pega(self, archivo: Path, key: str) -> None:
        """Lo que hace el patrón a mano: la key nueva en lugar del placeholder."""
        texto = archivo.read_text(encoding="utf-8")
        assert _PLACEHOLDER in texto
        archivo.write_text(texto.replace(_PLACEHOLDER, key), encoding="utf-8")


def test_soloist_wrapper_segunda_corrida_conserva_la_key_y_el_unit(tmp_path):
    s = _Soloist(tmp_path)
    out = s.run()
    assert out.returncode == 0, out.stdout + out.stderr
    wrapper = s.exec_start()
    assert shlex.split(wrapper) == [
        "/bin/sh", "-c",
        f'exec "%h/.local/bin/soloist" --api-key "$SOLOIST_API_KEY" {_FLAGS}']
    s.pega(s.env, _NUEVA)
    out = s.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.env.read_text(encoding="utf-8").splitlines() == [f"SOLOIST_API_KEY={_NUEVA}"]
    assert s.exec_start() == wrapper
    assert _NUEVA not in out.stdout + out.stderr


def test_soloist_de_wrapper_a_env_deshace_el_wrapper(tmp_path):
    """El camino que promete la cabecera: cuando Soloist acepte la key por entorno."""
    s = _Soloist(tmp_path)
    assert s.run().returncode == 0
    s.pega(s.env, _NUEVA)
    out = s.run(ayuda=_AYUDA_ENV)
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.exec_start() == f"%h/.local/bin/soloist {_FLAGS}"
    assert s.env.read_text(encoding="utf-8").splitlines() == [f"SOLOIST_API_KEY={_NUEVA}"]


def test_soloist_env_segunda_corrida_conserva_la_key(tmp_path):
    s = _Soloist(tmp_path)
    assert s.run(ayuda=_AYUDA_ENV).returncode == 0
    assert s.exec_start() == f"%h/.local/bin/soloist {_FLAGS}"
    s.env.write_text(f"OTRA=1\nexport SOLOIST_API_KEY=\"{_NUEVA}\"\n", encoding="utf-8")
    out = s.run(ayuda=_AYUDA_ENV)
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.env.read_text(encoding="utf-8") == f"OTRA=1\nexport SOLOIST_API_KEY=\"{_NUEVA}\"\n"
    assert s.exec_start() == f"%h/.local/bin/soloist {_FLAGS}"
    assert _NUEVA not in out.stdout + out.stderr


def test_soloist_la_key_vieja_del_unit_nunca_se_conserva(tmp_path):
    """La key que iba en ExecStart se da por comprometida: si alguien la copió al env,
    se sustituye por el placeholder (y las demás variables se quedan)."""
    s = _Soloist(tmp_path)
    s.env.parent.mkdir(parents=True)
    s.env.write_text(f"OTRA=1\nSOLOIST_API_KEY={_VIEJA}\n", encoding="utf-8")
    out = s.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.env.read_text(encoding="utf-8").splitlines() == [
        "OTRA=1", f"SOLOIST_API_KEY={_PLACEHOLDER}"]


@pytest.mark.parametrize("flag", ["--key-file", "--api-key-file"])
def test_soloist_file_usa_el_flag_de_la_ayuda_y_es_idempotente(tmp_path, flag):
    s = _Soloist(tmp_path)
    ayuda = f"  {flag} <PATH>   lee la key de un archivo"
    out = s.run(ayuda=ayuda)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"ARCHIVO ({flag})" in out.stdout
    esperado = f"%h/.local/bin/soloist {_FLAGS} {flag} %h/.config/soloist/api_key"
    assert s.exec_start() == esperado
    s.pega(s.key_file, _NUEVA)
    out = s.run(ayuda=ayuda)
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.exec_start() == esperado
    assert s.key_file.read_text(encoding="utf-8").strip() == _NUEVA
    assert _NUEVA not in out.stdout + out.stderr


@pytest.mark.parametrize(("arg", "secreto"), [
    ('--api-key "s3cr3t-KEY-sin-prefijo"', "s3cr3t"),
    ("--api-key 's3cr3t-KEY-sin-prefijo'", "s3cr3t"),
    ("--api-key=s3cr3t-KEY-sin-prefijo", "s3cr3t"),
    (f"--api-key {_VIEJA} --token spak_OTRA0123456789", "spak_OTRA"),
], ids=["comillas-dobles", "comillas-simples", "igual", "spak-en-otro-flag"])
def test_soloist_nunca_imprime_la_key(tmp_path, arg, secreto):
    s = _Soloist(tmp_path, exec_start=f"%h/.local/bin/soloist {arg} {_FLAGS}")
    out = s.run("--dry-run")
    assert out.returncode == 0, out.stdout + out.stderr
    assert secreto not in out.stdout + out.stderr


def test_soloist_la_ayuda_no_incluye_codigo():
    out = subprocess.run(["bash", str(SOLOIST_SH), "-h"], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0
    assert out.stdout.startswith("# Saca la API key")
    assert "set -euo pipefail" not in out.stdout


_KEY_FILE_FLAG = "--api-key-file %h/.config/soloist/api_key"


@pytest.mark.parametrize(("roto", "modo", "esperado"), [
    # Doble wrapper con las comillas rotas: lo que dejaba la 2ª corrida en modo wrapper.
    ("/bin/sh -c 'exec \"/bin/sh\" --api-key \"$SOLOIST_API_KEY\" -c 'exec "
     f"\"%h/.local/bin/soloist\" {_FLAGS}''",
     "wrapper",
     f"/bin/sh -c 'exec \"%h/.local/bin/soloist\" --api-key \"$SOLOIST_API_KEY\" {_FLAGS}'"),
    # --api-key-file duplicado: lo que dejaba la 2ª corrida en modo file.
    (f"%h/.local/bin/soloist {_FLAGS} {_KEY_FILE_FLAG} {_KEY_FILE_FLAG}",
     "file", f"%h/.local/bin/soloist {_FLAGS} {_KEY_FILE_FLAG}"),
], ids=["doble-wrapper", "key-file-duplicado"])
def test_soloist_arregla_el_unit_que_rompio_una_segunda_corrida(tmp_path, roto, modo, esperado):
    s = _Soloist(tmp_path, exec_start=roto)
    out = s.run("--mode", modo)
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.exec_start() == esperado


def test_soloist_no_imprime_la_key_que_ensena_la_ayuda(tmp_path):
    """clap enseña el valor actual de la variable (`[env: X=valor]`) si está exportada
    en la shell desde la que se corre el script: las líneas de la ayuda que se citan
    también pasan por el enmascarado."""
    s = _Soloist(tmp_path)
    out = s.run("--dry-run", ayuda="      [env: SOLOIST_API_KEY=spak_ENTORNO0123456]")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "spak_ENTORNO" not in out.stdout + out.stderr
    assert "[env: SOLOIST_API_KEY=" in out.stdout
