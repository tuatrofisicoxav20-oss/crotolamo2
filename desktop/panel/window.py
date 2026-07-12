"""Ventana del panel de control (GTK3 + Adwaita).

Rendimiento: query_state() (subprocess systemctl, hasta 8 s de timeout) corre
en un hilo daemon y entrega el resultado con GLib.idle_add — el hilo GTK nunca
se bloquea. El tick periódico se suspende cuando la ventana está desmapeada
(señales map/unmap) y se reanuda al mostrarse.
"""

from __future__ import annotations

import logging
import subprocess
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

from common.gtk_util import apply_css, swap_class  # noqa: E402

from .config import (  # noqa: E402
    ARGS_BY_KEY,
    KEY_BY_ARGS,
    MODES,
    read_listening_enabled,
    read_mode_args,
    write_listening_enabled,
    write_mode_args,
)
from .systemd import SERVICE, _do_async, _uptime_text, query_state, run  # noqa: E402
from .terminal import _find_terminal  # noqa: E402

log = logging.getLogger("crotolamo.desktop.panel")

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

_HALO_CLASSES = ("state-on", "state-wait", "state-off", "state-fail")
_BIG_CLASSES = ("go", "stop")


class Panel(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="Crotolamo")
        self.set_resizable(False)
        self.set_default_size(340, 0)
        self.set_keep_above(True)
        try:
            self.set_icon_name("crotolamo")
        except Exception as exc:  # noqa: BLE001
            log.debug("Icono 'crotolamo' no disponible: %s", exc)

        self._syncing = False
        self._tick_id: int | None = None
        self._query_inflight = False
        # último estado conocido del servicio; los callbacks de botones lo usan
        # en vez de consultar systemctl de forma síncrona en el hilo GTK.
        self._last_state: dict[str, str] = {}

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

        # fila: pausar/reanudar la escucha por voz SIN apagar el servicio.
        # A diferencia del botón grande (que mata el proceso y recarga modelos al
        # volver, ~segundos), esto solo silencia la wake word: Crotolamo sigue
        # caliente y reacciona al instante en cuanto lo reactivas.
        listen_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl_listen = Gtk.Label(label="Escuchar por voz")
        lbl_listen.get_style_context().add_class("crot-row")
        lbl_listen.set_xalign(0)
        self.listen_sw = Gtk.Switch()
        self.listen_sw.set_valign(Gtk.Align.CENTER)
        self.listen_sw.connect("notify::active", self.on_listen_toggle)
        listen_row.pack_start(lbl_listen, False, False, 0)
        listen_row.pack_end(self.listen_sw, False, False, 0)
        box.pack_start(listen_row, False, False, 0)

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

        apply_css(self.get_screen(), CSS)

        # el tick vive atado a la visibilidad real de la ventana: mientras esté
        # desmapeada (cerrada/minimizada a nada) no se consulta systemctl.
        self.connect("map", self._on_map)
        self.connect("unmap", self._on_unmap)
        self.show_all()

    # --- ciclo de refresco (hilo de trabajo + idle_add) -------------------------
    def _on_map(self, *_a) -> None:
        """Ventana visible: refresco inmediato + tick periódico."""
        if self._tick_id is None:
            self._tick_id = GLib.timeout_add(POLL_MS, self._tick)
        self.refresh()

    def _on_unmap(self, *_a) -> None:
        """Ventana oculta: suspender el polling (nada que pintar)."""
        if self._tick_id is not None:
            GLib.source_remove(self._tick_id)
            self._tick_id = None

    def _tick(self) -> bool:
        self.refresh()
        return True  # seguir llamando

    def refresh(self) -> None:
        """Dispara una consulta de estado en un hilo; no bloquea el hilo GTK."""
        if self._query_inflight:
            return
        self._query_inflight = True
        threading.Thread(target=self._query_worker, daemon=True).start()

    def _query_worker(self) -> None:
        st = query_state()
        GLib.idle_add(self._on_state_ready, st)

    def _on_state_ready(self, st: dict[str, str]) -> bool:
        self._query_inflight = False
        self._last_state = st
        self._apply_state(st)
        return False  # idle-callback de un solo uso

    # --- pintado del estado -----------------------------------------------------
    def _set_visual(self, halo_class: str, icon_name: str) -> None:
        swap_class(self.halo.get_style_context(), _HALO_CLASSES, halo_class)
        self.icon.set_from_icon_name(icon_name, Gtk.IconSize.DIALOG)
        self.icon.set_pixel_size(64)

    def _set_big(self, label: str, cls: str) -> None:
        swap_class(self.big.get_style_context(), _BIG_CLASSES, cls)
        self.big.set_label(label)

    def _apply_state(self, st: dict[str, str]) -> None:
        active = st.get("ActiveState", "unknown")
        sub = st.get("SubState", "")
        unit_state = st.get("UnitFileState", "")

        # interruptor "iniciar con la sesion" sin disparar el callback
        self._syncing = True
        self.autostart.set_active(unit_state == "enabled")
        # reflejar el modo guardado en el env file (lectura cacheada por mtime)
        cur_key = KEY_BY_ARGS.get(read_mode_args(), "half")
        if self.mode.get_active_id() != cur_key:
            self.mode.set_active_id(cur_key)
        # reflejar el estado REAL de la escucha (lo que publica el loop), no lo
        # que el panel cree; así el switch no miente si cambia desde otro lado.
        self.listen_sw.set_active(read_listening_enabled())
        self._syncing = False
        # Pausar la escucha solo tiene sentido con el servicio corriendo.
        self.listen_sw.set_sensitive(active == "active")

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
    def _soon_refresh(self) -> None:
        GLib.timeout_add(600, self._refresh_once)

    def _refresh_once(self) -> bool:
        self.refresh()
        return False

    def on_toggle(self, _btn: Gtk.Button) -> None:
        # usa el último estado conocido: nada de systemctl síncrono en el clic
        if self._last_state.get("ActiveState", "") == "active":
            _do_async("stop", SERVICE)
            self.status.set_label("Apagando…")
        else:
            _do_async("start", SERVICE)
            self.status.set_label("Arrancando…")
            self.big.set_sensitive(False)
        self._soon_refresh()

    def on_restart(self, _btn: Gtk.Button) -> None:
        _do_async("restart", SERVICE)
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
        if self._last_state.get("ActiveState") == "active":
            _do_async("restart", SERVICE)
            self.status.set_label("Aplicando modo…")
            self._soon_refresh()

    def on_autostart(self, switch: Gtk.Switch, _param) -> None:
        if self._syncing:
            return
        if switch.get_active():
            _do_async("enable", SERVICE)
        else:
            _do_async("disable", SERVICE)

    def on_listen_toggle(self, switch: Gtk.Switch, _param) -> None:
        if self._syncing:
            return
        # Escribe el flag; el loop lo sondea (~3x/s) y el próximo refresh lee de
        # vuelta el estado real publicado, así que no forzamos nada aquí.
        write_listening_enabled(switch.get_active())

    def on_log(self, _btn: Gtk.Button) -> None:
        cmd = ["journalctl", "--user", "-u", SERVICE, "-n", "200", "-f"]
        term = _find_terminal()
        if term:
            try:
                subprocess.Popen(term + cmd)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("No se pudo abrir la terminal %s: %s", term, exc)
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
