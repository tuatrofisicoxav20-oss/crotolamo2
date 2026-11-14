"""Tests de la auditoría para desktop/ (HUD y panel).

La lógica pura (hud.state, common.ipc, panel.config) se prueba directa. La que
vive en clases GTK (HUDWindow, HUDApp) se ejercita con un doble mínimo de
PyGObject y un reloj virtual para GLib: sin display, sin gi real y sin dormir.
"""

from __future__ import annotations

import collections
import heapq
import importlib
import itertools
import json
import os
import stat
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo.voice import state as voice_state
from interfaces import listener

# desktop/ no es paquete: se importa igual que en producción (sys.path)
DESKTOP_DIR = str(Path(__file__).resolve().parent.parent / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from common import ipc  # noqa: E402
from hud import state as hud_state  # noqa: E402
from panel import config, systemd  # noqa: E402


# ---------------------------------------------------------------------------
# Doble mínimo de PyGObject: solo lo que usa la lógica de desktop/hud
# ---------------------------------------------------------------------------

class _Reloj:
    """GLib.timeout_add / idle_add / source_remove sobre un reloj virtual (ms)."""

    def __init__(self) -> None:
        self.now = 0.0
        self._cola: list[tuple[float, int]] = []
        self._vivos: dict[int, tuple[float, object, tuple]] = {}
        self._ids = itertools.count(1)

    def timeout_add(self, ms, fn, *args):
        ident = next(self._ids)
        self._vivos[ident] = (ms, fn, args)
        heapq.heappush(self._cola, (self.now + ms, ident))
        return ident

    def idle_add(self, fn, *args):
        return self.timeout_add(0, fn, *args)

    def source_remove(self, ident):
        self._vivos.pop(ident, None)
        return True

    def avanzar(self, ms: float) -> None:
        """Corre los timers que vencen en los próximos `ms` (como el main loop)."""
        fin = self.now + ms
        while self._cola and self._cola[0][0] <= fin:
            cuando, ident = heapq.heappop(self._cola)
            if ident not in self._vivos:
                continue
            self.now = max(self.now, cuando)
            intervalo, fn, args = self._vivos[ident]
            if fn(*args):
                if ident in self._vivos:
                    heapq.heappush(self._cola, (self.now + max(intervalo, 1), ident))
            else:
                self._vivos.pop(ident, None)
        self.now = fin


class _Contexto:
    def __init__(self) -> None:
        self.clases: set[str] = set()

    def add_class(self, cls):
        self.clases.add(cls)

    def remove_class(self, cls):
        self.clases.discard(cls)


class _Widget:
    """Modela visible/opacidad/label; el resto de setters se anota y no hace nada."""

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.llamadas: dict[str, tuple] = {}
        self._ctx = _Contexto()
        self._visible = False
        self._opacity = 1.0
        self._label = kwargs.get("label", "")

    def get_style_context(self):
        return self._ctx

    def show(self):
        self._visible = True

    def show_all(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def set_visible(self, visible):
        self._visible = bool(visible)

    def get_visible(self):
        return self._visible

    def set_opacity(self, opacity):
        # GTK guarda la opacidad en 8 bits
        self._opacity = round(max(0.0, min(1.0, opacity)) * 255) / 255

    def get_opacity(self):
        return self._opacity

    def set_label(self, text):
        self._label = text

    def get_label(self):
        return self._label

    def get_screen(self):
        return SimpleNamespace(
            get_rgba_visual=lambda: None,
            get_monitor_geometry=lambda n: SimpleNamespace(x=0, y=0, width=1920, height=1080),
        )

    def get_size(self):
        return (400, 200)

    def __getattr__(self, nombre):
        if nombre.startswith("_"):
            raise AttributeError(nombre)

        def anotar(*args, **kwargs):
            self.llamadas[nombre] = args

        return anotar


class _Image(_Widget):
    @classmethod
    def new_from_icon_name(cls, name, size):
        return cls()


class _Application:
    """Gtk.Application: guarda los kwargs del constructor (id y flags)."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def do_startup(self):
        pass

    def hold(self):
        pass

    def add_window(self, win):
        pass

    def run(self, argv):
        return 0


class _Monitor:
    def connect(self, signal, handler):
        self.handler = handler


class _File:
    @staticmethod
    def new_for_path(path):
        return _File()

    def monitor_directory(self, flags, cancellable):
        return _Monitor()


class _GLibError(Exception):
    pass


def _gi_falso(reloj: _Reloj) -> tuple[types.ModuleType, types.ModuleType]:
    def require_version(namespace, version):
        # como el gi real: ValueError si el namespace no está instalado
        if namespace not in ("Gtk", "Gdk", "Pango"):
            raise ValueError(f"Namespace {namespace} not available")

    gi = types.ModuleType("gi")
    gi.require_version = require_version  # type: ignore[attr-defined]
    repo = types.ModuleType("gi.repository")
    repo.GLib = SimpleNamespace(  # type: ignore[attr-defined]
        timeout_add=reloj.timeout_add, idle_add=reloj.idle_add,
        source_remove=reloj.source_remove, Error=_GLibError)
    repo.Gio = SimpleNamespace(  # type: ignore[attr-defined]
        File=_File, FileMonitor=_Monitor,
        FileMonitorFlags=SimpleNamespace(WATCH_MOVES=1),
        # valores reales de GApplicationFlags
        ApplicationFlags=SimpleNamespace(FLAGS_NONE=0, DEFAULT_FLAGS=0, NON_UNIQUE=32))
    repo.Gdk = SimpleNamespace(  # type: ignore[attr-defined]
        WindowTypeHint=SimpleNamespace(NOTIFICATION=7))
    repo.Pango = SimpleNamespace(  # type: ignore[attr-defined]
        EllipsizeMode=SimpleNamespace(NONE=0, START=1, MIDDLE=2, END=3))
    repo.Gtk = SimpleNamespace(  # type: ignore[attr-defined]
        Window=_Widget, Box=_Widget, Label=_Widget, Spinner=_Widget, Image=_Image,
        Application=_Application,
        CssProvider=_Widget,
        StyleContext=SimpleNamespace(add_provider_for_screen=lambda *a: None),
        STYLE_PROVIDER_PRIORITY_APPLICATION=600,
        WindowType=SimpleNamespace(TOPLEVEL=0),
        Orientation=SimpleNamespace(HORIZONTAL=0, VERTICAL=1),
        Align=SimpleNamespace(CENTER=3),
        IconSize=SimpleNamespace(DIALOG=6),
        Justification=SimpleNamespace(CENTER=2))
    gi.repository = repo  # type: ignore[attr-defined]
    return gi, repo


# módulos de desktop/ que importan gi: se cargan contra el doble y se descargan
_MODULOS_GTK = ("common.gtk_util", "hud.window", "hud.app")


@pytest.fixture
def hud(tmp_path, monkeypatch):
    """hud.window y hud.app REALES cargados sobre el gi falso y el reloj virtual."""
    reloj = _Reloj()
    gi, repo = _gi_falso(reloj)
    nombres = ("gi", "gi.repository", *_MODULOS_GTK)
    guardados = {n: sys.modules.pop(n, None) for n in nombres}
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repo
    try:
        window = importlib.import_module("hud.window")
        app = importlib.import_module("hud.app")
        estado = tmp_path / "hud_state.json"
        monkeypatch.setattr(window, "HUD_STATE_FILE", estado)
        ventanas: list = []

        def ventana(demo: bool = False):
            win = window.HUDWindow(demo=demo)
            ventanas.append(win)
            reloj.avanzar(0)  # la lectura inicial (idle_add)
            return win

        def escribir(data: dict) -> None:
            estado.write_text(json.dumps(data), encoding="utf-8")

        def publicar(mode: str, text: str = "") -> None:
            """Lo que hace el loop (escribe el archivo) + el evento del FileMonitor."""
            escribir({"mode": mode, "text": text})
            for win in ventanas:
                win._read_state_once()

        yield SimpleNamespace(reloj=reloj, window=window, app=app, gio=repo.Gio,
                              pango=repo.Pango, ventana=ventana, escribir=escribir,
                              publicar=publicar)
    finally:
        for n in nombres:
            sys.modules.pop(n, None)
            if guardados[n] is not None:
                sys.modules[n] = guardados[n]
        # el paquete padre guarda el submódulo como atributo: que tampoco quede
        # ahí la versión enlazada al gi falso
        for n in _MODULOS_GTK:
            paquete, _, sub = n.partition(".")
            padre = sys.modules.get(paquete)
            if padre is None:
                continue
            if guardados[n] is not None:
                setattr(padre, sub, guardados[n])
            elif hasattr(padre, sub):
                delattr(padre, sub)


# ---------------------------------------------------------------------------
# D20: el contrato IPC está escrito en los dos lados (desktop/ corre con el
# python del sistema y no puede importar crotolamo.*). Estos tests son los que
# fijan que ambos lados digan lo mismo.
# ---------------------------------------------------------------------------

def test_ipc_rutas_iguales_en_desktop_y_en_el_loop():
    assert ipc.HUD_STATE_FILE == listener._HUD_STATE_PATH
    assert ipc.CONTROL_FILE == voice_state.CONTROL_PATH


def test_ipc_modos_del_hud_son_los_del_loop():
    assert hud_state.VALID_MODES == {m.name.lower() for m in voice_state.Mode}


def test_ipc_lo_que_publica_el_loop_lo_entienden_hud_y_panel(tmp_path, monkeypatch):
    path = tmp_path / "hud_state.json"
    monkeypatch.setattr(ipc, "HUD_STATE_FILE", path)
    estado = voice_state.SharedState(publisher=voice_state.make_file_publisher(path))
    estado.set_text("qué hora es")
    for modo in voice_state.Mode:
        estado.set_mode(modo)
        data = hud_state.read_hud_file(path)
        assert hud_state.extract_mode(data) == modo.name.lower()
        assert hud_state.extract_text(data) == "qué hora es"
    assert config.read_listening_enabled() is True
    estado.set_enabled(False)
    # Aquí se prueba el contrato, no la caché del panel (firma mtime+tamaño): en
    # un FS de timestamps gruesos dos escrituras del mismo tick con "ts" un
    # dígito más corto tienen la misma firma y el test fallaría al azar.
    config._MTIME_CACHE.pop(path, None)
    assert config.read_listening_enabled() is False


def test_ipc_la_pausa_que_escribe_el_panel_la_lee_el_loop(tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    monkeypatch.setattr(ipc, "CONTROL_FILE", control)
    assert voice_state.read_control_enabled(control) is True  # ausente => activa
    config.write_listening_enabled(False)
    assert voice_state.read_control_enabled(control) is False
    config.write_listening_enabled(True)
    assert voice_state.read_control_enabled(control) is True


# ---------------------------------------------------------------------------
# extract_mode: un "mode" que no es str (lista/objeto JSON) es inválido => idle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [["listening"], {"listening": 1}, 7, 1.5, True, None])
def test_extract_mode_con_valor_no_str_devuelve_idle(mode):
    data = hud_state.parse_hud_state(json.dumps({"mode": mode, "text": "hola"}))
    assert hud_state.extract_mode(data) == "idle"
    assert hud_state.extract_text(data) == "hola"


def test_hud_poll_sobrevive_a_un_mode_no_hashable(hud):
    win = hud.ventana()
    hud.publicar("speaking", "hola")
    hud.escribir({"mode": ["listening"], "text": "x"})
    # GLib descarta para siempre un timer cuyo callback lanza: el poll debe
    # devolver True y tratar el estado inválido como reposo.
    assert win._poll_tick() is True
    assert win._current_mode == "idle"


# ---------------------------------------------------------------------------
# Fade: una convocatoria justo cuando se arma el fade-out no puede ocultar el HUD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retraso_ms", [-10, 10, 30, 150, 400])
def test_hud_convocado_alrededor_del_fade_out_queda_visible(hud, retraso_ms):
    """+10 ms cae entre _start_fade_out() y su primer paso (opacidad aún 1.0)."""
    win = hud.ventana()
    hud.publicar("speaking", "respuesta")
    hud.reloj.avanzar(1000)
    assert win.get_visible() is True
    assert win.get_opacity() == 1.0

    hud.publicar("idle")
    hud.reloj.avanzar(hud.window.HIDE_AFTER_IDLE_MS + retraso_ms)
    hud.publicar("listening")  # el patrón vuelve a llamar justo ahora
    hud.reloj.avanzar(3000)

    assert win._current_mode == "listening"
    assert win.get_visible() is True
    assert win.get_opacity() == 1.0
    # visible => el poll de respaldo sigue vivo
    assert win._backup_timer_id is not None


def test_hud_se_oculta_tras_idle_si_nadie_lo_convoca(hud):
    win = hud.ventana()
    hud.publicar("speaking", "respuesta")
    hud.reloj.avanzar(1000)
    hud.publicar("idle")
    hud.reloj.avanzar(hud.window.HIDE_AFTER_IDLE_MS + 1000)
    assert win.get_visible() is False
    assert win.get_opacity() == 0.0
    assert win._backup_timer_id is None


# ---------------------------------------------------------------------------
# D14: --demo debe abrir SU ventana aunque el HUD del servicio ya esté corriendo
# ---------------------------------------------------------------------------

def test_hud_demo_no_negocia_instancia_unica(hud):
    demo = hud.app.HUDApp(demo=True)
    assert demo.kwargs.get("flags", 0) & hud.gio.ApplicationFlags.NON_UNIQUE


def test_hud_del_servicio_sigue_siendo_instancia_unica(hud):
    servicio = hud.app.HUDApp()
    assert servicio.kwargs["application_id"] == "org.crotolamo.HUD"
    assert not servicio.kwargs.get("flags", 0) & hud.gio.ApplicationFlags.NON_UNIQUE


# ---------------------------------------------------------------------------
# D15: el texto del HUD tiene tope de líneas (el loop publica la respuesta entera)
# ---------------------------------------------------------------------------

def test_hud_texto_con_tope_de_lineas_y_elipsis(hud):
    etiqueta = hud.ventana()._text_label
    # set_lines solo actúa en una etiqueta con wrap Y ellipsize
    assert etiqueta.llamadas.get("set_line_wrap") == (True,)
    assert etiqueta.llamadas.get("set_lines") == (3,)
    assert etiqueta.llamadas.get("set_ellipsize") == (hud.pango.EllipsizeMode.END,)


def test_extract_text_deja_un_solo_parrafo():
    # Pango aplica el tope de líneas POR PÁRRAFO: con saltos de línea no hay tope
    data = {"text": "  Uno.\n\nDos:\n- a\n- b\t fin  "}
    assert hud_state.extract_text(data) == "Uno. Dos: - a - b fin"


def test_hud_respuesta_con_parrafos_llega_a_la_etiqueta_en_uno(hud):
    win = hud.ventana()
    hud.publicar("speaking", "Uno.\n\nDos.\nTres.")
    assert win._text_label.get_label() == "Uno. Dos. Tres."
    # mismo modo, texto nuevo: también pasa por extract_text
    hud.publicar("speaking", "Otra\nrespuesta")
    assert win._text_label.get_label() == "Otra respuesta"


# ---------------------------------------------------------------------------
# D9: el loop murió sin su idle final (SIGKILL, OOM, SIGHUP, apagón) y
# hud_state.json se quedó en "speaking"/"thinking". El pid que el loop escribe
# en cada publicación dice que ya nadie habla: el HUD no puede quedarse encima.
# ---------------------------------------------------------------------------

@pytest.fixture
def pid_muerto():
    """Un pid que existió y ya no existe (hijo terminado y recogido)."""
    hijo = subprocess.Popen([sys.executable, "-c", "pass"])
    hijo.wait()
    return hijo.pid


@pytest.fixture
def loop_vivo():
    """Un proceso vivo que hace de listener; el test lo mata cuando quiere."""
    hijo = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield hijo
    finally:
        hijo.kill()
        hijo.wait()


def test_publicador_vivo_con_el_pid_propio():
    assert hud_state.publisher_alive({"mode": "speaking", "pid": os.getpid()}) is True


def test_publicador_muerto(pid_muerto):
    assert hud_state.publisher_alive({"mode": "speaking", "pid": pid_muerto}) is False


@pytest.mark.parametrize("pid", [None, "123", 1.5, True, 0, -1, 2**80])
def test_publicador_sin_pid_valido_se_da_por_vivo(pid):
    # esquema viejo o basura: no se puede saber y se respeta el modo publicado
    # (2**80 no cabe en un pid_t: os.kill lanzaría OverflowError en el poll)
    data = {"mode": "speaking"} if pid is None else {"mode": "speaking", "pid": pid}
    assert hud_state.publisher_alive(data) is True


def test_hud_se_oculta_si_el_loop_muere_sin_su_idle(hud, loop_vivo):
    win = hud.ventana()
    hud.escribir({"mode": "speaking", "text": "un cuento largo", "pid": loop_vivo.pid})
    win._read_state_once()  # evento del FileMonitor
    hud.reloj.avanzar(1000)
    assert win.get_visible() is True

    # muere sin su apagado limpio: el archivo sigue diciendo "speaking"
    loop_vivo.kill()
    loop_vivo.wait()
    # el poll de respaldo (vivo mientras el HUD se ve) lo nota sin ayuda
    w = hud.window
    hud.reloj.avanzar(w.BACKUP_POLL_MS + w.HIDE_AFTER_IDLE_MS + w.FADE_STEPS * w.FADE_STEP_MS
                      + 500)
    assert win._current_mode == "idle"
    assert win.get_visible() is False
    assert win._backup_timer_id is None
    assert win._text_label.get_label() == ""


def test_hud_al_arrancar_no_muestra_el_estado_de_un_loop_muerto(hud, pid_muerto):
    # apagón o kill -9: el archivo sobrevive al reinicio con el pid viejo
    hud.escribir({"mode": "thinking", "text": "x", "pid": pid_muerto})
    win = hud.ventana()  # lectura inicial
    hud.reloj.avanzar(3000)
    assert win._current_mode == "idle"
    assert win.get_visible() is False


def test_hud_con_el_loop_vivo_sigue_visible(hud):
    win = hud.ventana()
    hud.escribir({"mode": "speaking", "text": "hola", "pid": os.getpid()})
    win._read_state_once()
    hud.reloj.avanzar(10 * hud.window.BACKUP_POLL_MS)
    assert win._current_mode == "speaking"
    assert win.get_visible() is True
    assert win._text_label.get_label() == "hola"


# ---------------------------------------------------------------------------
# desktop#17: elegir modo en el panel reescribía listener.env DESDE CERO (el
# servicio carga el archivo entero con EnvironmentFile=: lo demás se perdía) y
# con truncate+write (un corte a medias lo dejaba vacío o cortado).
# ---------------------------------------------------------------------------

def test_write_mode_args_conserva_las_demas_lineas(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text("# modo de escucha de Crotolamo\n"
                   "CROTOLAMO_LISTEN_ARGS=--no-barge-in\n"
                   "CROTOLAMO_LOG=DEBUG\n"
                   "CROTOLAMO_WAKE_DEBUG=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--barge-in")
    assert env.read_text(encoding="utf-8") == ("# modo de escucha de Crotolamo\n"
                                               "CROTOLAMO_LISTEN_ARGS=--barge-in\n"
                                               "CROTOLAMO_LOG=DEBUG\n"
                                               "CROTOLAMO_WAKE_DEBUG=1\n")
    assert config.read_mode_args() == "--barge-in"


def test_write_mode_args_sin_la_linea_la_anade_al_final(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text("CROTOLAMO_LOG=DEBUG", encoding="utf-8")  # sin salto final
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--simple")
    assert env.read_text(encoding="utf-8") == ("CROTOLAMO_LOG=DEBUG\n"
                                               "CROTOLAMO_LISTEN_ARGS=--simple\n")


def test_write_mode_args_crea_el_archivo_y_su_carpeta(tmp_path, monkeypatch):
    env = tmp_path / "nueva" / "listener.env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--simple")
    assert env.read_text(encoding="utf-8") == "CROTOLAMO_LISTEN_ARGS=--simple\n"


def test_write_mode_args_clave_repetida_systemd_y_panel_coinciden(tmp_path, monkeypatch):
    # systemd se queda con la ÚLTIMA asignación y el panel lee la PRIMERA: todas
    # las líneas del modo llevan el valor elegido; un comentario no es una línea
    env = tmp_path / "listener.env"
    env.write_text("CROTOLAMO_LISTEN_ARGS=--simple\n"
                   "# CROTOLAMO_LISTEN_ARGS=--no-barge-in\n"
                   '  CROTOLAMO_LISTEN_ARGS="--simple"\r\n', encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--barge-in")
    # read_bytes: read_text traduciría el \r\n y no se vería si se conserva
    assert env.read_bytes().split(b"\n") == [
        b"CROTOLAMO_LISTEN_ARGS=--barge-in",
        b"# CROTOLAMO_LISTEN_ARGS=--no-barge-in",
        b"CROTOLAMO_LISTEN_ARGS=--barge-in\r",
        b"",
    ]
    assert config.read_mode_args() == "--barge-in"


def test_write_mode_args_respeta_bytes_que_no_son_utf8(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_bytes(b"# caf\xe9 (latin-1)\nCROTOLAMO_LISTEN_ARGS=--simple\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.write_mode_args("--barge-in")
    assert env.read_bytes() == b"# caf\xe9 (latin-1)\nCROTOLAMO_LISTEN_ARGS=--barge-in\n"


def test_write_mode_args_es_atomico_y_conserva_permisos(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    viejo = "CROTOLAMO_LISTEN_ARGS=--no-barge-in\nCROTOLAMO_LOG=DEBUG\n"
    env.write_text(viejo, encoding="utf-8")
    env.chmod(0o640)
    monkeypatch.setattr(config, "ENV_FILE", env)
    vistos = []
    real_replace = os.replace

    def espiando(src, dst):
        vistos.append((Path(src), Path(dst), Path(src).read_text(encoding="utf-8"),
                       Path(dst).read_text(encoding="utf-8")))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", espiando)
    config.write_mode_args("--simple")

    assert len(vistos) == 1
    src, dst, en_el_temporal, en_el_destino = vistos[0]
    assert dst == env
    assert src != env and src.parent == env.parent
    assert en_el_temporal == "CROTOLAMO_LISTEN_ARGS=--simple\nCROTOLAMO_LOG=DEBUG\n"
    assert en_el_destino == viejo  # intacto hasta el replace
    assert stat.S_IMODE(env.stat().st_mode) == 0o640  # mkstemp crea 0600: se copia
    assert sorted(p.name for p in tmp_path.iterdir()) == ["listener.env"]


def test_write_mode_args_si_falla_el_reemplazo_no_toca_nada(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    viejo = "CROTOLAMO_LISTEN_ARGS=--no-barge-in\nCROTOLAMO_LOG=DEBUG\n"
    env.write_text(viejo, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)

    def falla(src, dst):
        raise OSError(28, "No queda espacio en el dispositivo")

    monkeypatch.setattr(os, "replace", falla)
    config.write_mode_args("--simple")  # no lanza: solo se registra
    assert env.read_text(encoding="utf-8") == viejo
    assert sorted(p.name for p in tmp_path.iterdir()) == ["listener.env"]  # sin huérfanos


def test_write_listening_enabled_sin_temporal_huerfano_si_falla_el_reemplazo(
        tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    monkeypatch.setattr(ipc, "CONTROL_FILE", control)

    def falla(src, dst):
        raise OSError(28, "No queda espacio en el dispositivo")

    monkeypatch.setattr(os, "replace", falla)
    config.write_listening_enabled(False)
    assert list(tmp_path.iterdir()) == []


def test_write_mode_args_no_pisa_un_archivo_que_no_pudo_leer(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    viejo = "CROTOLAMO_LISTEN_ARGS=--no-barge-in\nSECRETO=1\n"
    env.write_text(viejo, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    real_read_bytes = Path.read_bytes

    def sin_permiso(self):
        if self == env:
            raise PermissionError(13, "Permiso denegado")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", sin_permiso)
    config.write_mode_args("--simple")
    assert env.read_text(encoding="utf-8") == viejo


# ---------------------------------------------------------------------------
# desktop#8 (D13): systemd.py — el estado que no se pudo consultar no es
# «apagado», y el resultado de una acción no se tira a /dev/null.
# ---------------------------------------------------------------------------

def _runner(stdout: str = "", returncode: int = 0, stderr: str = ""):
    def runner(*args: str) -> subprocess.CompletedProcess:
        runner.calls.append(args)
        return subprocess.CompletedProcess(list(args), returncode, stdout=stdout,
                                           stderr=stderr)
    runner.calls = []
    return runner


def test_query_state_pide_load_state():
    runner = _runner("LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
                     "UnitFileState=\n")
    st = systemd.query_state(runner=runner)
    assert "LoadState" in runner.calls[0]
    assert st["LoadState"] == "not-found"
    assert systemd.QUERY_ERROR not in st


@pytest.mark.parametrize("returncode, stderr, esperado", [
    (1, "Failed to connect to bus: No medium found\n",
     "Failed to connect to bus: No medium found"),
    (255, "Command '['systemctl']' timed out after 8.0 seconds",
     "Command '['systemctl']' timed out after 8.0 seconds"),
    (3, "", "systemctl salió con código 3"),
])
def test_query_state_marca_que_no_pudo_preguntar(returncode, stderr, esperado):
    st = systemd.query_state(runner=_runner("", returncode, stderr))
    assert st == {systemd.QUERY_ERROR: esperado}


def _esperar_do_async(monkeypatch, *args):
    """Corre systemd._do_async con un `run` falso; devuelve (cmd, resultado entregado)."""
    llamadas = []
    fallo = subprocess.CompletedProcess([], 5, stdout="",
                                        stderr="Failed to start crotolamo.service: "
                                               "Unit crotolamo.service not found.\n")

    def run_falso(cmd, timeout=8.0):
        llamadas.append((cmd, threading.current_thread()))
        return fallo

    monkeypatch.setattr(systemd, "run", run_falso)
    entregado = []
    listo = threading.Event()

    def on_done(res):
        entregado.append(res)
        listo.set()

    systemd._do_async(*args, on_done=on_done)
    assert listo.wait(5)
    (cmd, hilo), = llamadas
    assert hilo is not threading.current_thread()  # nunca en el hilo que llama (GTK)
    assert entregado == [fallo]
    return cmd


@pytest.mark.parametrize("verbo", ["start", "stop", "restart"])
def test_do_async_jobs_encolan_sin_bloquear_y_entregan_el_resultado(monkeypatch, verbo):
    cmd = _esperar_do_async(monkeypatch, verbo, systemd.SERVICE)
    assert cmd == ["systemctl", "--user", "--no-block", verbo, systemd.SERVICE]


@pytest.mark.parametrize("verbo", ["enable", "disable"])
def test_do_async_enable_disable_tal_cual(monkeypatch, verbo):
    cmd = _esperar_do_async(monkeypatch, verbo, systemd.SERVICE)
    assert cmd == ["systemctl", "--user", verbo, systemd.SERVICE]


def test_first_line():
    assert systemd._first_line("\n  Failed to x: y  \nmás\n") == "Failed to x: y"
    assert systemd._first_line("") == ""
    assert systemd._first_line(None) == ""


# ---------------------------------------------------------------------------
# Panel (desktop/panel/window.py) REAL sobre el gi falso: widgets que cuentan
# sus llamadas, para ver qué se pinta y cuántas veces.
# ---------------------------------------------------------------------------

class _ContextoContado(_Contexto):
    def __init__(self) -> None:
        super().__init__()
        self.cuenta: collections.Counter = collections.Counter()

    def add_class(self, cls):
        self.cuenta["add_class"] += 1
        super().add_class(cls)

    def remove_class(self, cls):
        self.cuenta["remove_class"] += 1
        super().remove_class(cls)


class _Contado(_Widget):
    """_Widget que cuenta cada llamada y modela sensible/activo/id activo."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cuenta: collections.Counter = collections.Counter()
        self._ctx = _ContextoContado()
        self._sensible = True
        self._activo = False
        self._active_id = None

    def set_label(self, text):
        self.cuenta["set_label"] += 1
        super().set_label(text)

    def set_sensitive(self, valor):
        self._sensible = bool(valor)

    def get_sensitive(self):
        return self._sensible

    def set_active(self, valor):
        self._activo = bool(valor)

    def get_active(self):
        return self._activo

    def set_active_id(self, ident):
        self._active_id = ident

    def get_active_id(self):
        return self._active_id

    def __getattr__(self, nombre):
        if nombre.startswith("_"):
            raise AttributeError(nombre)

        def anotar(*args, **kwargs):
            self.cuenta[nombre] += 1
            self.llamadas[nombre] = args

        return anotar


class _ImagenContada(_Contado):
    @classmethod
    def new_from_icon_name(cls, name, size):
        return cls()


_MODULOS_PANEL = ("common.gtk_util", "panel.window")


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """panel.window REAL sobre el gi falso; _do_async no lanza systemctl."""
    reloj = _Reloj()
    gi, repo = _gi_falso(reloj)
    for nombre in ("ApplicationWindow", "HeaderBar", "Box", "Label", "Button",
                   "Separator", "Switch", "ComboBoxText"):
        setattr(repo.Gtk, nombre, _Contado)
    repo.Gtk.Image = _ImagenContada
    nombres = ("gi", "gi.repository", *_MODULOS_PANEL)
    guardados = {n: sys.modules.pop(n, None) for n in nombres}
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repo
    try:
        window = importlib.import_module("panel.window")
        monkeypatch.setattr(ipc, "HUD_STATE_FILE", tmp_path / "hud_state.json")
        monkeypatch.setattr(config, "ENV_FILE", tmp_path / "listener.env")
        lanzados: list = []

        def lanzar(*args, on_done=None):
            lanzados.append((args, on_done))

        monkeypatch.setattr(window, "_do_async", lanzar)
        win = window.Panel(app=None)
        yield SimpleNamespace(reloj=reloj, window=window, win=win, lanzados=lanzados)
    finally:
        for n in nombres:
            sys.modules.pop(n, None)
            if guardados[n] is not None:
                sys.modules[n] = guardados[n]
        for n in _MODULOS_PANEL:
            paquete, _, sub = n.partition(".")
            padre = sys.modules.get(paquete)
            if padre is None:
                continue
            if guardados[n] is not None:
                setattr(padre, sub, guardados[n])
            elif hasattr(padre, sub):
                delattr(padre, sub)


_ACTIVO = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running",
           "UnitFileState": "enabled", "ActiveEnterTimestampMonotonic": "0"}
_APAGADO = {"LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead",
            "UnitFileState": "disabled", "ActiveEnterTimestampMonotonic": "0"}


def test_panel_unit_no_instalada_no_se_pinta_como_apagado(panel):
    win = panel.win
    win._on_state_ready({"LoadState": "not-found", "ActiveState": "inactive",
                         "SubState": "dead", "UnitFileState": ""})
    assert win.status.get_label() == "No instalado"
    assert "install.sh" in win.sub.get_label()
    assert win.big.get_sensitive() is False  # no hay nada que encender


def test_panel_sin_systemd_no_se_pinta_como_apagado(panel):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    assert win.autostart.get_active() is True
    win._on_state_ready({systemd.QUERY_ERROR: "Failed to connect to bus: No medium found"})
    assert win.status.get_label() not in ("Apagado", "Escuchando")
    assert "No medium found" in win.sub.get_label()
    assert win.big.get_sensitive() is False
    # sin respuesta de systemd no se adivina «no arranca con la sesión»
    assert win.autostart.get_active() is True


def test_panel_una_accion_que_falla_se_ve(panel):
    win = panel.win
    win._on_state_ready({"LoadState": "masked", "ActiveState": "inactive",
                         "SubState": "dead", "UnitFileState": "masked"})
    win.on_toggle(None)
    (args, on_done), = panel.lanzados
    assert args == ("start", systemd.SERVICE)
    assert win.big.get_sensitive() is False  # "Arrancando…"
    # systemctl responde desde el hilo de trabajo: el panel salta al hilo GTK
    on_done(subprocess.CompletedProcess([], 1, stdout="",
                                        stderr="Failed to start crotolamo.service: "
                                               "Unit crotolamo.service is masked.\n"))
    panel.reloj.avanzar(0)
    assert win.sub.get_label() == ("Failed to start crotolamo.service: "
                                   "Unit crotolamo.service is masked.")
    assert win.status.get_label() == "Apagado"  # no se queda en «Arrancando…»
    assert win.big.get_sensitive() is True
    # el siguiente tick con el mismo estado no borra el aviso...
    win._on_state_ready({"LoadState": "masked", "ActiveState": "inactive",
                         "SubState": "dead", "UnitFileState": "masked"})
    assert "is masked" in win.sub.get_label()
    # ...pero en cuanto el servicio cambia ya no aplica
    win._on_state_ready(dict(_ACTIVO))
    assert win.status.get_label() == "Escuchando"
    assert win.sub.get_label() == "activo"


def test_panel_accion_correcta_no_deja_aviso(panel):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    win.on_autostart(win.autostart, None)
    (args, on_done), = panel.lanzados
    assert args == ("disable", systemd.SERVICE)
    on_done(subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    panel.reloj.avanzar(0)
    assert win.sub.get_label() == "Crotolamo no está escuchando"


def test_panel_una_accion_nueva_borra_el_aviso_anterior(panel):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    win.on_toggle(None)
    panel.lanzados[-1][1](subprocess.CompletedProcess([], 1, stdout="", stderr="falló X\n"))
    panel.reloj.avanzar(0)
    assert win.sub.get_label() == "falló X"
    win.on_restart(None)
    win._on_state_ready(dict(_APAGADO))
    assert win.sub.get_label() == "Crotolamo no está escuchando"


def test_panel_ver_registro_sin_terminal_no_bloquea_el_hilo_gtk(panel, monkeypatch):
    win = panel.win
    monkeypatch.setattr(panel.window, "_find_terminal", lambda: None)
    hilos: list = []

    class _Hilo(threading.Thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            hilos.append(self)

    monkeypatch.setattr(panel.window, "threading", SimpleNamespace(Thread=_Hilo))
    soltar = threading.Event()
    corrio_en: list = []

    def journalctl_lento(cmd, timeout=8.0):
        corrio_en.append(threading.current_thread())
        soltar.wait(5)
        return subprocess.CompletedProcess(cmd, 0, stdout="línea 1\nlínea 2\n", stderr="")

    monkeypatch.setattr(panel.window, "run", journalctl_lento)
    mostrados: list = []
    monkeypatch.setattr(win, "_show_text", lambda titulo, texto: mostrados.append(texto))

    win.on_log(None)
    win.on_log(None)  # doble clic: un solo journalctl y un solo diálogo
    assert mostrados == []  # on_log volvió sin esperar a journalctl
    soltar.set()
    for hilo in hilos:
        hilo.join(5)
    assert len(corrio_en) == 1
    assert corrio_en[0] is not threading.current_thread()
    panel.reloj.avanzar(0)
    assert mostrados == ["línea 1\nlínea 2\n"]
    # terminado, otro clic vuelve a funcionar
    win.on_log(None)
    for hilo in hilos:
        hilo.join(5)
    panel.reloj.avanzar(0)
    assert len(mostrados) == 2


# ---------------------------------------------------------------------------
# desktop#15: cada tick (1.5 s) repintaba TODO aunque el estado fuera el mismo
# (swap_class reestiliza, set_from_icon_name recarga el icono, Button.set_label
# rehace la etiqueta hija y cada set_label pide relayout).
# ---------------------------------------------------------------------------

def test_panel_no_repinta_si_el_estado_no_cambio(panel):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    assert win.status.get_label() == "Escuchando"
    widgets = (win.icon, win.big, win.status, win.sub, win.halo)
    antes = [w.cuenta.copy() for w in widgets]
    ctx_antes = [w.get_style_context().cuenta.copy() for w in (win.halo, win.big)]
    for _ in range(20):
        win._on_state_ready(dict(_ACTIVO))
    assert [w.cuenta for w in widgets] == antes
    assert [w.get_style_context().cuenta for w in (win.halo, win.big)] == ctx_antes


def test_panel_repinta_lo_que_cambia(panel):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    win._on_state_ready(dict(_APAGADO))
    assert win.status.get_label() == "Apagado"
    assert win.big.get_label() == "⏻   ENCENDER"
    assert win.halo.get_style_context().clases == {"crot-halo", "state-off"}
    assert win.big.get_style_context().clases == {"crot-big", "go"}
    assert win.icon.llamadas["set_from_icon_name"][0] == "microphone-disabled-symbolic"
    win._on_state_ready(dict(_ACTIVO))
    assert win.status.get_label() == "Escuchando"
    assert win.big.get_label() == "⏻   APAGAR"
    assert win.halo.get_style_context().clases == {"crot-halo", "state-on"}
    assert win.icon.llamadas["set_from_icon_name"][0] == "microphone-sensitivity-high-symbolic"


def test_panel_tras_un_clic_el_tick_con_el_mismo_estado_repinta(panel):
    # el clic escribe «Arrancando…» y apaga el botón; si systemd sigue igual, el
    # siguiente tick tiene que devolver la verdad (la caché no puede taparlo)
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    win.on_toggle(None)
    assert win.status.get_label() == "Arrancando…"
    assert win.big.get_sensitive() is False
    win._on_state_ready(dict(_APAGADO))
    assert win.status.get_label() == "Apagado"
    assert win.big.get_sensitive() is True


def test_write_mode_args_escribe_a_traves_de_un_symlink(tmp_path, monkeypatch):
    # dotfiles con stow/chezmoi: listener.env es un enlace. write_text escribía en
    # el destino; os.replace sobre el enlace lo cambiaría por un archivo suelto.
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "listener.env"
    real.write_text("CROTOLAMO_LISTEN_ARGS=--no-barge-in\nCROTOLAMO_LOG=DEBUG\n",
                    encoding="utf-8")
    cfg = tmp_path / "config"
    cfg.mkdir()
    enlace = cfg / "listener.env"
    enlace.symlink_to(real)
    monkeypatch.setattr(config, "ENV_FILE", enlace)
    config.write_mode_args("--simple")
    assert enlace.is_symlink()
    assert real.read_text(encoding="utf-8") == ("CROTOLAMO_LISTEN_ARGS=--simple\n"
                                                "CROTOLAMO_LOG=DEBUG\n")
    assert sorted(p.name for p in cfg.iterdir()) == ["listener.env"]
    assert sorted(p.name for p in dotfiles.iterdir()) == ["listener.env"]


# ---------------------------------------------------------------------------
# desktop#5 / x-concurrency-lifecycle#12 (D5): el switch «Escuchar por voz» se
# revertía solo. on_listen_toggle escribe control.json, pero el loop tarda hasta
# ~0.3 s en sondearlo y republicar hud_state.json: un tick que caía antes (o una
# consulta ya en vuelo) repintaba el valor viejo y el switch rebotaba ~1.5 s.
# «Iniciar con la sesión» igual mientras el `systemctl enable` asíncrono corría.
# ---------------------------------------------------------------------------

@pytest.fixture
def escucha(panel, tmp_path, monkeypatch):
    """Panel con control.json en tmp y un reloj monotónico manual (segundos)."""
    control = tmp_path / "control.json"
    monkeypatch.setattr(ipc, "CONTROL_FILE", control)
    ahora = [1000.0]
    monkeypatch.setattr(panel.window, "time", SimpleNamespace(monotonic=lambda: ahora[0]))
    estado = tmp_path / "hud_state.json"

    def publicar(enabled: bool) -> None:
        """Lo que escribe el loop en hud_state.json al sondear control.json."""
        estado.write_text(json.dumps({"mode": "idle", "text": "", "enabled": enabled}),
                          encoding="utf-8")

    def avanzar(segundos: float) -> None:
        ahora[0] += segundos

    def clic_escucha(valor: bool) -> None:
        """El patrón mueve el switch: GTK cambia el valor y emite notify::active."""
        panel.win.listen_sw.set_active(valor)
        panel.win.on_listen_toggle(panel.win.listen_sw, None)

    def clic_autostart(valor: bool):
        panel.win.autostart.set_active(valor)
        panel.win.on_autostart(panel.win.autostart, None)
        return panel.lanzados[-1]

    publicar(True)
    return SimpleNamespace(control=control, publicar=publicar, avanzar=avanzar,
                           clic_escucha=clic_escucha, clic_autostart=clic_autostart)


def test_panel_switch_de_escucha_no_rebota_antes_de_que_el_loop_publique(panel, escucha):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is True
    escucha.clic_escucha(False)
    assert json.loads(escucha.control.read_text()) == {"listening_enabled": False}
    # tick ANTES de que el loop sondee control.json: hud_state.json sigue en true
    escucha.avanzar(0.2)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is False  # antes volvía a True ~1.5 s
    # el loop publica la pausa: el switch ya refleja la verdad
    escucha.publicar(False)
    escucha.avanzar(1.5)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is False
    # confirmado, lo que cambie después desde otro lado se ve al instante
    escucha.publicar(True)
    escucha.avanzar(0.1)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is True


def test_panel_switch_de_escucha_vuelve_a_la_verdad_si_el_loop_no_confirma(panel, escucha):
    # el loop no recogió la pausa (colgado, versión vieja): pasado el plazo el
    # switch dice lo que publica el loop, no lo que el panel quiso
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))
    escucha.clic_escucha(False)
    escucha.avanzar(1.5)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is False
    escucha.avanzar(2.0)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is True


def test_panel_switch_de_escucha_si_no_se_pudo_escribir_no_finge(panel, escucha, monkeypatch):
    win = panel.win
    win._on_state_ready(dict(_ACTIVO))

    def disco_lleno(path, data):
        raise OSError(28, "No queda espacio en el dispositivo")

    monkeypatch.setattr(config, "_replace_atomically", disco_lleno)
    escucha.clic_escucha(False)
    win._on_state_ready(dict(_ACTIVO))
    assert win.listen_sw.get_active() is True  # el loop sigue escuchando


def test_panel_autostart_no_rebota_mientras_systemctl_enable_corre(panel, escucha):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    assert win.autostart.get_active() is False
    args, on_done = escucha.clic_autostart(True)
    assert args == ("enable", systemd.SERVICE)
    # consulta que corrió antes de que `enable` terminara (aunque tarde segundos)
    escucha.avanzar(5.0)
    win._on_state_ready(dict(_APAGADO))
    assert win.autostart.get_active() is True  # antes volvía a False
    on_done(subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    panel.reloj.avanzar(0)
    win._on_state_ready(dict(_APAGADO))  # consulta lanzada justo antes de acabar
    assert win.autostart.get_active() is True
    win._on_state_ready(dict(_APAGADO, UnitFileState="enabled"))
    assert win.autostart.get_active() is True
    # confirmado, un `systemctl --user disable` desde una terminal se ve al instante
    win._on_state_ready(dict(_APAGADO))
    assert win.autostart.get_active() is False


def test_panel_autostart_sin_confirmar_vuelve_a_la_verdad_tras_el_plazo(panel, escucha):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    _args, on_done = escucha.clic_autostart(True)
    on_done(subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    panel.reloj.avanzar(0)
    escucha.avanzar(3.5)  # systemd nunca dijo «enabled» (p. ej. «linked»)
    win._on_state_ready(dict(_APAGADO, UnitFileState="linked"))
    assert win.autostart.get_active() is False


def test_panel_autostart_que_falla_vuelve_al_instante_con_el_motivo(panel, escucha):
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    _args, on_done = escucha.clic_autostart(True)
    on_done(subprocess.CompletedProcess(
        [], 1, stdout="",
        stderr="Failed to enable unit: Unit file crotolamo.service does not exist.\n"))
    panel.reloj.avanzar(0)
    assert win.autostart.get_active() is False
    assert "does not exist" in win.sub.get_label()


def test_panel_autostart_el_resultado_de_una_accion_vieja_no_pisa_la_nueva(panel, escucha):
    # ON y enseguida OFF: el `enable` (ya superado) falla después; el switch sigue
    # en lo último que pidió el patrón mientras su `disable` corre
    win = panel.win
    win._on_state_ready(dict(_APAGADO))
    _args, enable_done = escucha.clic_autostart(True)
    args, _disable_done = escucha.clic_autostart(False)
    assert args == ("disable", systemd.SERVICE)
    enable_done(subprocess.CompletedProcess([], 1, stdout="", stderr="enable falló\n"))
    panel.reloj.avanzar(0)
    win._on_state_ready(dict(_APAGADO, UnitFileState="enabled"))
    assert win.autostart.get_active() is False
