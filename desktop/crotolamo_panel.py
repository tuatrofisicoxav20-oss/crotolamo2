#!/usr/bin/env python3
"""Mini-ventana flotante para controlar Crotolamo desde el escritorio de Fedora.

Minimalista y nativa (GTK3 + tema Adwaita): muestra si Crotolamo está escuchando,
con un botón grande para encender/apagar, selector de modo de voz, un interruptor
para iniciarlo con la sesión y un acceso rápido al registro en vivo.

NO importa nada del núcleo de Crotolamo: lo controla a través del servicio de
usuario de systemd ('crotolamo.service'). Por eso se ejecuta con el python3 del
SISTEMA (el que trae gi/Gtk3), no con el venv del proyecto.

    /usr/bin/python3 desktop/crotolamo_panel.py

El estado se refresca solo cada 1.5 s, así que la ventana siempre refleja la
realidad aunque enciendas/apagues el servicio desde otro lado.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

# --- constantes de integración -------------------------------------------------

SERVICE = "crotolamo.service"
ENV_FILE = Path.home() / ".config" / "crotolamo" / "listener.env"

# Modo de escucha -> argumentos que recibe `python -m crotolamo listen`.
# El servicio los lee de ENV_FILE (variable CROTOLAMO_LISTEN_ARGS).
MODES: list[tuple[str, str, str]] = [
    # (clave, etiqueta visible, args)
    ("half", "Half-duplex (altavoces)", "--no-barge-in"),
    ("barge", "Barge-in (auriculares)", "--barge-in"),
    ("simple", "Simple (a prueba de fallos)", "--simple"),
]
ARGS_BY_KEY = {k: a for k, _, a in MODES}
KEY_BY_ARGS = {a: k for k, _, a in MODES}

POLL_MS = 1500

CSS = b"""
.crot-card { padding: 14px 18px 18px 18px; }
.crot-title {
    font-weight: 800;
    font-size: 15pt;
    letter-spacing: 3px;
}
.crot-status { font-size: 17pt; font-weight: 700; }
.crot-sub { font-size: 9pt; opacity: 0.65; }

/* halo del icono segun estado */
.crot-halo {
    border-radius: 999px;
    padding: 16px;
    background: alpha(@theme_fg_color, 0.06);
}
.state-on    image { color: #2ec27e; }   /* verde: escuchando */
.state-on    { background: alpha(#2ec27e, 0.14); }
.state-wait  image { color: #f5c211; }   /* ambar: arrancando */
.state-wait  { background: alpha(#f5c211, 0.14); }
.state-off   image { color: @theme_unfocused_fg_color; opacity: 0.7; }
.state-fail  image { color: #ed333b; }   /* rojo: error */
.state-fail  { background: alpha(#ed333b, 0.14); }

/* boton grande */
.crot-big {
    font-size: 13pt;
    font-weight: 800;
    letter-spacing: 1px;
    padding: 12px 0;
    border-radius: 12px;
}
.crot-big.go      { background: #2ec27e; color: white; }
.crot-big.go:hover{ background: #33d289; }
.crot-big.stop    { background: alpha(@theme_fg_color, 0.10); }
.crot-big.stop:hover { background: alpha(@theme_fg_color, 0.18); }

.crot-row { font-size: 10pt; }
.crot-mini { padding: 4px 10px; border-radius: 8px; font-size: 9.5pt; }
"""


def run(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess:
    """systemctl/journalctl sin reventar la UI: devuelve siempre un resultado."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception as exc:  # noqa: BLE001
        return subprocess.CompletedProcess(args, returncode=255, stdout="", stderr=str(exc))


def sc(*args: str) -> subprocess.CompletedProcess:
    return run(["systemctl", "--user", *args])


def query_state() -> dict[str, str]:
    """Una sola llamada barata para el polling: estado + si arranca con la sesion."""
    res = sc("show", SERVICE,
             "-p", "ActiveState", "-p", "SubState", "-p", "UnitFileState",
             "-p", "ActiveEnterTimestampMonotonic")
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            out[key] = val
    return out


class Panel(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="Crotolamo")
        self.set_resizable(False)
        self.set_default_size(340, 0)
        self.set_keep_above(True)
        try:
            self.set_icon_name("crotolamo")
        except Exception:  # noqa: BLE001
            pass

        self._syncing = False

        # --- barra de titulo minimalista (nativa GNOME) ---
        header = Gtk.HeaderBar(title="Crotolamo")
        header.set_show_close_button(True)
        header.set_subtitle("asistente local")
        self.set_titlebar(header)

        # --- cuerpo ---
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.get_style_context().add_class("crot-card")
        self.add(box)

        title = Gtk.Label(label="C R O T O L A M O")
        title.get_style_context().add_class("crot-title")
        box.pack_start(title, False, False, 0)

        # icono de estado dentro de un halo de color
        self.halo = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.halo.set_halign(Gtk.Align.CENTER)
        self.halo.get_style_context().add_class("crot-halo")
        self.icon = Gtk.Image.new_from_icon_name(
            "microphone-disabled-symbolic", Gtk.IconSize.DIALOG)
        self.icon.set_pixel_size(64)
        self.halo.add(self.icon)
        halo_wrap = Gtk.Box()
        halo_wrap.set_halign(Gtk.Align.CENTER)
        halo_wrap.pack_start(self.halo, False, False, 0)
        box.pack_start(halo_wrap, False, False, 4)

        self.status = Gtk.Label(label="…")
        self.status.get_style_context().add_class("crot-status")
        box.pack_start(self.status, False, False, 0)

        self.sub = Gtk.Label(label="")
        self.sub.get_style_context().add_class("crot-sub")
        box.pack_start(self.sub, False, False, 0)

        # boton grande encender/apagar
        self.big = Gtk.Button(label="…")
        self.big.get_style_context().add_class("crot-big")
        self.big.connect("clicked", self.on_toggle)
        box.pack_start(self.big, False, False, 6)

        box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL),
                       False, False, 2)

        # fila: modo de voz
        mode_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl_mode = Gtk.Label(label="Modo")
        lbl_mode.get_style_context().add_class("crot-row")
        lbl_mode.set_xalign(0)
        self.mode = Gtk.ComboBoxText()
        for key, label, _ in MODES:
            self.mode.append(key, label)
        self.mode.set_active_id("half")
        self.mode.connect("changed", self.on_mode_changed)
        mode_row.pack_start(lbl_mode, False, False, 0)
        mode_row.pack_end(self.mode, False, False, 0)
        box.pack_start(mode_row, False, False, 0)

        # fila: iniciar con la sesion
        auto_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl_auto = Gtk.Label(label="Iniciar con la sesión")
        lbl_auto.get_style_context().add_class("crot-row")
        lbl_auto.set_xalign(0)
        self.autostart = Gtk.Switch()
        self.autostart.set_valign(Gtk.Align.CENTER)
        self.autostart.connect("notify::active", self.on_autostart)
        auto_row.pack_start(lbl_auto, False, False, 0)
        auto_row.pack_end(self.autostart, False, False, 0)
        box.pack_start(auto_row, False, False, 0)

        # fila: acciones secundarias
        act_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        act_row.set_halign(Gtk.Align.CENTER)
        btn_log = Gtk.Button(label="Ver registro")
        btn_log.get_style_context().add_class("crot-mini")
        btn_log.connect("clicked", self.on_log)
        btn_restart = Gtk.Button(label="Reiniciar")
        btn_restart.get_style_context().add_class("crot-mini")
        btn_restart.connect("clicked", self.on_restart)
        act_row.pack_start(btn_log, False, False, 0)
        act_row.pack_start(btn_restart, False, False, 0)
        box.pack_start(act_row, False, False, 2)

        self._apply_css()
        self.refresh()
        GLib.timeout_add(POLL_MS, self._tick)
        self.show_all()

    # --- estilo ---------------------------------------------------------------
    def _apply_css(self) -> None:
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)

    # --- refresco de estado ---------------------------------------------------
    def _tick(self) -> bool:
        self.refresh()
        return True  # seguir llamando

    def _set_visual(self, halo_class: str, icon_name: str) -> None:
        ctx = self.halo.get_style_context()
        for c in ("state-on", "state-wait", "state-off", "state-fail"):
            ctx.remove_class(c)
        ctx.add_class(halo_class)
        self.icon.set_from_icon_name(icon_name, Gtk.IconSize.DIALOG)
        self.icon.set_pixel_size(64)

    def _set_big(self, label: str, cls: str) -> None:
        bctx = self.big.get_style_context()
        for c in ("go", "stop"):
            bctx.remove_class(c)
        bctx.add_class(cls)
        self.big.set_label(label)

    def refresh(self) -> None:
        st = query_state()
        active = st.get("ActiveState", "unknown")
        sub = st.get("SubState", "")
        unit_state = st.get("UnitFileState", "")

        # interruptor "iniciar con la sesion" sin disparar el callback
        self._syncing = True
        self.autostart.set_active(unit_state == "enabled")
        # reflejar el modo guardado en el env file
        cur_key = KEY_BY_ARGS.get(read_mode_args(), "half")
        if self.mode.get_active_id() != cur_key:
            self.mode.set_active_id(cur_key)
        self._syncing = False

        if active == "active":
            self._set_visual("state-on", "microphone-sensitivity-high-symbolic")
            self.status.set_label("Escuchando")
            self.sub.set_label(_uptime_text(st))
            self._set_big("⏻   APAGAR", "stop")
            self.big.set_sensitive(True)
        elif active in ("activating", "reloading", "deactivating"):
            self._set_visual("state-wait", "microphone-sensitivity-medium-symbolic")
            self.status.set_label("Arrancando…" if active == "activating" else "Cambiando…")
            self.sub.set_label("cargando modelos de voz")
            self._set_big("…", "stop")
            self.big.set_sensitive(False)
        elif active == "failed" or sub == "failed":
            self._set_visual("state-fail", "microphone-hardware-disabled-symbolic")
            self.status.set_label("Error")
            self.sub.set_label("revisa «Ver registro»")
            self._set_big("⏻   ENCENDER", "go")
            self.big.set_sensitive(True)
        else:  # inactive / dead / unknown
            self._set_visual("state-off", "microphone-disabled-symbolic")
            self.status.set_label("Apagado")
            self.sub.set_label("Crotolamo no está escuchando")
            self._set_big("⏻   ENCENDER", "go")
            self.big.set_sensitive(True)

    # --- acciones -------------------------------------------------------------
    def _do_async(self, *args: str) -> None:
        """Lanza systemctl sin congelar la UI; el polling refleja el resultado."""
        try:
            subprocess.Popen(["systemctl", "--user", *args],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass

    def _soon_refresh(self) -> None:
        GLib.timeout_add(600, self._refresh_once)

    def _refresh_once(self) -> bool:
        self.refresh()
        return False

    def on_toggle(self, _btn: Gtk.Button) -> None:
        st = query_state().get("ActiveState", "")
        if st == "active":
            self._do_async("stop", SERVICE)
            self.status.set_label("Apagando…")
        else:
            self._do_async("start", SERVICE)
            self.status.set_label("Arrancando…")
            self.big.set_sensitive(False)
        self._soon_refresh()

    def on_restart(self, _btn: Gtk.Button) -> None:
        self._do_async("restart", SERVICE)
        self.status.set_label("Reiniciando…")
        self.big.set_sensitive(False)
        self._soon_refresh()

    def on_mode_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._syncing:
            return
        key = combo.get_active_id()
        if not key:
            return
        write_mode_args(ARGS_BY_KEY[key])
        # si está escuchando, reiniciar para aplicar el nuevo modo
        if query_state().get("ActiveState") == "active":
            self._do_async("restart", SERVICE)
            self.status.set_label("Aplicando modo…")
            self._soon_refresh()

    def on_autostart(self, switch: Gtk.Switch, _param) -> None:
        if self._syncing:
            return
        if switch.get_active():
            self._do_async("enable", SERVICE)
        else:
            self._do_async("disable", SERVICE)

    def on_log(self, _btn: Gtk.Button) -> None:
        cmd = ["journalctl", "--user", "-u", SERVICE, "-n", "200", "-f"]
        term = _find_terminal()
        if term:
            try:
                subprocess.Popen(term + cmd)
                return
            except Exception:  # noqa: BLE001
                pass
        # sin terminal: mostrar las ultimas lineas en un dialogo
        res = run(["journalctl", "--user", "-u", SERVICE, "-n", "200", "--no-pager"])
        self._show_text("Registro de Crotolamo", res.stdout or res.stderr or "(vacío)")

    def _show_text(self, title: str, text: str) -> None:
        dlg = Gtk.Dialog(title=title, transient_for=self, modal=True)
        dlg.set_default_size(640, 460)
        dlg.add_button("Cerrar", Gtk.ResponseType.CLOSE)
        sw = Gtk.ScrolledWindow()
        sw.set_vexpand(True)
        sw.set_hexpand(True)
        tv = Gtk.TextView()
        tv.set_editable(False)
        tv.set_monospace(True)
        tv.get_buffer().set_text(text)
        sw.add(tv)
        dlg.get_content_area().add(sw)
        dlg.show_all()
        dlg.run()
        dlg.destroy()


# --- helpers del env file (modo de voz) ---------------------------------------

def read_mode_args() -> str:
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("CROTOLAMO_LISTEN_ARGS="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return "--no-barge-in"


def write_mode_args(args: str) -> None:
    try:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text(f"CROTOLAMO_LISTEN_ARGS={args}\n")
    except Exception:  # noqa: BLE001
        pass


def _uptime_text(st: dict[str, str]) -> str:
    try:
        now = int(GLib.get_monotonic_time())  # microsegundos
        started = int(st.get("ActiveEnterTimestampMonotonic", "0"))
        if started <= 0:
            return "activo"
        secs = max(0, (now - started) // 1_000_000)
        if secs < 60:
            return f"activo desde hace {secs}s"
        mins = secs // 60
        if mins < 60:
            return f"activo desde hace {mins} min"
        return f"activo desde hace {mins // 60} h {mins % 60} min"
    except Exception:  # noqa: BLE001
        return "activo"


def _find_terminal() -> list[str] | None:
    candidates = [
        ("kitty", ["kitty", "-e"]),
        ("ptyxis", ["ptyxis", "--"]),
        ("gnome-terminal", ["gnome-terminal", "--"]),
        ("konsole", ["konsole", "-e"]),
        ("foot", ["foot"]),
        ("alacritty", ["alacritty", "-e"]),
        ("xterm", ["xterm", "-e"]),
    ]
    for binary, prefix in candidates:
        if shutil.which(binary):
            return prefix
    return None


class App(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="org.crotolamo.Panel")
        self.win: Panel | None = None

    def do_activate(self) -> None:
        if self.win is None:
            self.win = Panel(self)
        self.win.present()


def main() -> int:
    app = App()
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
