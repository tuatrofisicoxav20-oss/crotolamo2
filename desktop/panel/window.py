"""Ventana del panel de control (GTK3 + Adwaita).

Rendimiento: query_state() (subprocess systemctl, hasta 8 s de timeout) corre
en UN hilo daemon de vida larga, dormido entre consultas, y entrega el resultado
con GLib.idle_add — el hilo GTK nunca se bloquea; las acciones
(systemd._do_async) y el journalctl de respaldo, en un hilo por clic.
El tick periódico vive entre las señales map/unmap (ocultar o destruir la
ventana lo para; minimizarla NO, GTK3 no desmapea al minimizar) y un tick con el
mismo estado no repinta nada.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

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
from .systemd import (  # noqa: E402
    QUERY_ERROR,
    SERVICE,
    _do_async,
    _first_line,
    _uptime_text,
    query_state,
    run,
    spawn_reaped,
)
from .terminal import _find_terminal  # noqa: E402

log = logging.getLogger("crotolamo.desktop.panel")

POLL_MS = 1500

# Lo que el patrón acaba de pedir con un switch se sostiene hasta que el estado
# publicado lo confirme o pasen HOLD_S segundos. El loop ve control.json por
# inotify (cada ~0.3 s sin él o en --simple) y `systemctl enable` tarda: un
# tick, o una consulta ya en vuelo, que caía antes repintaba el valor viejo y el
# switch rebotaba ~1.5 s.
# Pasado el plazo manda lo publicado: el switch nunca miente por mucho tiempo.
HOLD_S = 3.0

# (valor pedido, plazo monotónico; None = sin plazo: su systemctl aún corre)
_Hold = tuple[bool, float | None]

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


def _state_key(st: dict[str, str]) -> tuple[str, ...]:
    """Lo que identifica un estado del servicio (sin el uptime, que avanza solo)."""
    return tuple(st.get(k, "") for k in
                 ("LoadState", "ActiveState", "SubState", "UnitFileState", QUERY_ERROR))


def _resolve_hold(hold: _Hold | None, real: bool) -> tuple[bool, _Hold | None]:
    """(lo que muestra el switch, hold que sigue) dado el valor real publicado."""
    if hold is None:
        return real, None
    wanted, deadline = hold
    if deadline is None:
        # su systemctl aún corre: ni un estado que «ya coincide» vale (ON y OFF
        # seguidos: el `enable` en vuelo todavía puede cambiarlo)
        return wanted, hold
    if wanted == real or time.monotonic() >= deadline:
        return real, None
    return wanted, hold


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
        # Fallo de la última acción (start/enable…): (estado en que falló, motivo).
        # Se ve en el subtítulo hasta otra acción o hasta que el estado cambie.
        self._action_error: tuple[tuple[str, ...], str] | None = None
        # Lo último que el patrón pidió con cada switch, hasta que se confirme
        # (ver HOLD_S y _resolve_hold).
        self._listen_hold: _Hold | None = None
        self._autostart_hold: _Hold | None = None
        self._log_inflight = False
        # Lo último pintado en halo/icono y botón grande: repintar lo mismo cada
        # tick no es gratis (swap_class reestiliza, set_from_icon_name recarga el
        # icono y Button.set_label rehace su etiqueta hija; todo pide relayout).
        self._visual: tuple[str, str] | None = None
        self._big_look: tuple[str, str] | None = None
        # Un solo hilo de consulta para toda la vida de la ventana: refresh() lo
        # despierta con _wake (antes, un hilo nuevo por tick: 40 por minuto con
        # la ventana abierta). Nace en el primer refresh; destroy lo para.
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._closing = False

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
        # el motivo de un fallo de systemctl es una línea larga: que parta en
        # varias en vez de ensanchar la ventana
        self.sub.set_line_wrap(True)
        self.sub.set_max_width_chars(44)
        self.sub.set_justify(Gtk.Justification.CENTER)
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

        # el tick vive atado al mapeo de la ventana: oculta o destruida no se
        # consulta systemctl (minimizar no desmapea en GTK3: ahí sigue).
        self.connect("map", self._on_map)
        self.connect("unmap", self._on_unmap)
        self.connect("destroy", self._on_destroy)
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
        """Pide una consulta de estado al hilo de trabajo; no bloquea el hilo GTK.

        Como antes, un refresco que cae con la consulta anterior aún en vuelo
        se descarta: las mismas consultas, sin un hilo nuevo por cada una.
        """
        if self._query_inflight or self._closing:
            return
        self._query_inflight = True
        if self._worker is None:
            self._worker = threading.Thread(target=self._query_worker, daemon=True,
                                            name="panel-query")
            self._worker.start()
        self._wake.set()

    def _query_worker(self) -> None:
        """Hilo de consulta: duerme en _wake (sin reloj) hasta el próximo refresh."""
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._closing:
                return
            try:
                st = query_state()
            except Exception as exc:  # noqa: BLE001 — el hilo no puede morir
                # Antes la excepción mataba el hilo con _query_inflight en True y
                # el panel no volvía a refrescar. Se pinta como lo que es (no se
                # pudo preguntar a systemd), igual que un systemctl que revienta.
                log.exception("No pude consultar el estado del servicio: %s", exc)
                st = {QUERY_ERROR: str(exc) or type(exc).__name__}
            if self._closing:
                return  # ventana destruida mientras consultaba: nada que pintar
            GLib.idle_add(self._on_state_ready, st)

    def _on_destroy(self, *_a) -> None:
        """Ventana destruida: el hilo de consulta termina en cuanto lo ve.

        Sin join: el hilo GTK nunca espera a systemctl (la única espera larga
        del hilo es la de run(), acotada a 8 s) y es daemon.
        """
        self._closing = True
        self._wake.set()

    def _on_state_ready(self, st: dict[str, str]) -> bool:
        self._query_inflight = False
        self._last_state = st
        self._apply_state(st)
        return False  # idle-callback de un solo uso

    # --- pintado del estado -----------------------------------------------------
    def _set_visual(self, halo_class: str, icon_name: str) -> None:
        if self._visual == (halo_class, icon_name):
            return  # igual que el tick anterior (ver __init__)
        self._visual = (halo_class, icon_name)
        swap_class(self.halo.get_style_context(), _HALO_CLASSES, halo_class)
        self.icon.set_from_icon_name(icon_name, Gtk.IconSize.DIALOG)
        self.icon.set_pixel_size(64)

    def _set_big(self, label: str, cls: str) -> None:
        if self._big_look == (label, cls):
            return
        self._big_look = (label, cls)
        swap_class(self.big.get_style_context(), _BIG_CLASSES, cls)
        self.big.set_label(label)

    @staticmethod
    def _set_switch(switch: Gtk.Switch, active: bool) -> None:
        """set_active solo si cambia, comparando con el switch DE VERDAD (como
        _set_text): lo que movió un clic se corrige igual en el siguiente tick."""
        if switch.get_active() != active:
            switch.set_active(active)

    @staticmethod
    def _set_sensitive(widget: Gtk.Widget, sensitive: bool) -> None:
        """set_sensitive solo si cambia; contra el widget, no contra una caché:
        los clics apagan el botón grande por su cuenta."""
        if widget.get_sensitive() != sensitive:
            widget.set_sensitive(sensitive)

    @staticmethod
    def _set_text(label: Gtk.Label, text: str) -> None:
        """set_label solo si cambia: en GTK3 cada set_label rehace el layout.

        Compara con lo que la etiqueta muestra DE VERDAD (no con una caché): así
        lo que escriben los clics («Arrancando…») se corrige en el siguiente tick.
        """
        if label.get_label() != text:
            label.set_label(text)

    def _apply_state(self, st: dict[str, str]) -> None:
        active = st.get("ActiveState", "unknown")
        sub = st.get("SubState", "")
        unit_state = st.get("UnitFileState", "")
        query_error = st.get(QUERY_ERROR)

        # interruptor "iniciar con la sesion" sin disparar el callback
        self._syncing = True
        if query_error is None:
            # sin respuesta de systemd no se adivina «no arranca con la sesión»
            shown, self._autostart_hold = _resolve_hold(self._autostart_hold,
                                                        unit_state == "enabled")
            self._set_switch(self.autostart, shown)
        # reflejar el modo guardado en el env file (lectura cacheada por mtime)
        cur_key = KEY_BY_ARGS.get(read_mode_args(), "half")
        if self.mode.get_active_id() != cur_key:
            self.mode.set_active_id(cur_key)
        # reflejar el estado REAL de la escucha (lo que publica el loop), no lo
        # que el panel cree; así el switch no miente si cambia desde otro lado.
        # Salvo un clic recién hecho que el loop aún no recogió (HOLD_S).
        shown, self._listen_hold = _resolve_hold(self._listen_hold,
                                                 read_listening_enabled())
        self._set_switch(self.listen_sw, shown)
        self._syncing = False
        # Pausar la escucha solo tiene sentido con el servicio corriendo.
        self._set_sensitive(self.listen_sw, active == "active")

        if query_error is not None:
            # systemctl no contestó (sin bus de usuario, timeout): NO es «Apagado»
            self._set_visual("state-fail", "microphone-hardware-disabled-symbolic")
            status, subtitle = "Desconocido", f"no pude consultar systemd: {query_error}"
            self._set_big("…", "stop")
            big_ok = False
        elif active == "active":
            self._set_visual("state-on", "microphone-sensitivity-high-symbolic")
            status, subtitle = "Escuchando", _uptime_text(st)
            self._set_big("⏻   APAGAR", "stop")
            big_ok = True
        elif active in ("activating", "reloading", "deactivating"):
            self._set_visual("state-wait", "microphone-sensitivity-medium-symbolic")
            status = "Arrancando…" if active == "activating" else "Cambiando…"
            subtitle = "cargando modelos de voz"
            self._set_big("…", "stop")
            big_ok = False
        elif active == "failed" or sub == "failed":
            self._set_visual("state-fail", "microphone-hardware-disabled-symbolic")
            status, subtitle = "Error", "revisa «Ver registro»"
            self._set_big("⏻   ENCENDER", "go")
            big_ok = True
        elif st.get("LoadState") == "not-found":
            # systemd no conoce la unit (panel abierto antes de install.sh o tras
            # desinstalar): salía «Apagado» y ENCENDER no hacía nada
            self._set_visual("state-off", "microphone-disabled-symbolic")
            status, subtitle = "No instalado", "ejecuta desktop/install.sh"
            self._set_big("⏻   ENCENDER", "go")
            big_ok = False
        else:  # inactive / dead / unknown
            self._set_visual("state-off", "microphone-disabled-symbolic")
            status, subtitle = "Apagado", "Crotolamo no está escuchando"
            self._set_big("⏻   ENCENDER", "go")
            big_ok = True

        if self._action_error is not None:
            if self._action_error[0] == _state_key(st):
                subtitle = self._action_error[1]
            else:
                self._action_error = None  # el servicio cambió: el aviso ya no aplica
        self._set_text(self.status, status)
        self._set_text(self.sub, subtitle)
        self._set_sensitive(self.big, big_ok)

    # --- acciones -------------------------------------------------------------
    def _systemctl(self, *args: str) -> None:
        """Lanza `systemctl --user <args>` en un hilo; si falla, el panel lo dice."""
        self._action_error = None
        _do_async(*args,
                  on_done=lambda res: GLib.idle_add(self._on_action_done, args, res))

    def _on_action_done(self, args: tuple[str, ...],
                        res: subprocess.CompletedProcess) -> bool:
        hold = self._autostart_hold
        if (args[0] in ("enable", "disable") and hold is not None
                and hold[0] == (args[0] == "enable")):
            # el resultado de la acción vigente (no de una ya superada): si
            # falló, el switch vuelve ya a la verdad; si no, se sostiene hasta
            # que una consulta posterior lo confirme o venza el plazo
            self._autostart_hold = ((hold[0], time.monotonic() + HOLD_S)
                                    if res.returncode == 0 else None)
        if res.returncode != 0:
            reason = (_first_line(res.stderr)
                      or f"systemctl {args[0]} falló (código {res.returncode})")
            self._action_error = (_state_key(self._last_state), reason)
            # repintar ya: devuelve el estado real (no «Arrancando…») con el motivo
            self._apply_state(self._last_state)
        return False  # idle-callback de un solo uso

    def _soon_refresh(self) -> None:
        GLib.timeout_add(600, self._refresh_once)

    def _refresh_once(self) -> bool:
        self.refresh()
        return False

    def on_toggle(self, _btn: Gtk.Button) -> None:
        # usa el último estado conocido: nada de systemctl síncrono en el clic
        if self._last_state.get("ActiveState", "") == "active":
            self._systemctl("stop", SERVICE)
            self.status.set_label("Apagando…")
        else:
            self._systemctl("start", SERVICE)
            self.status.set_label("Arrancando…")
            self.big.set_sensitive(False)
        self._soon_refresh()

    def on_restart(self, _btn: Gtk.Button) -> None:
        self._systemctl("restart", SERVICE)
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
            self._systemctl("restart", SERVICE)
            self.status.set_label("Aplicando modo…")
            self._soon_refresh()

    def on_autostart(self, switch: Gtk.Switch, _param) -> None:
        if self._syncing:
            return
        # sin plazo mientras su systemctl corre (`enable` puede tardar segundos;
        # run() lo corta a los 8 s y _on_action_done llega siempre)
        self._autostart_hold = (switch.get_active(), None)
        if switch.get_active():
            self._systemctl("enable", SERVICE)
        else:
            self._systemctl("disable", SERVICE)

    def on_listen_toggle(self, switch: Gtk.Switch, _param) -> None:
        if self._syncing:
            return
        # Escribe el flag; el loop lo sondea (~3x/s) y publica el estado real,
        # que los refresh leen de vuelta. Hasta que llegue, el switch se queda
        # donde lo puso el patrón (si no se pudo escribir, nada que sostener).
        enabled = switch.get_active()
        self._listen_hold = ((enabled, time.monotonic() + HOLD_S)
                             if write_listening_enabled(enabled) else None)

    def on_log(self, _btn: Gtk.Button) -> None:
        cmd = ["journalctl", "--user", "-u", SERVICE, "-n", "200", "-f"]
        term = _find_terminal()
        if term:
            try:
                spawn_reaped(term + cmd)  # con quien la espere: cerrarla no deja zombi
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("No se pudo abrir la terminal %s: %s", term, exc)
        # sin terminal: las últimas líneas en un diálogo. journalctl (hasta 8 s de
        # timeout) va en un hilo: en el hilo GTK congelaba la ventana mientras.
        if self._log_inflight:
            return  # doble clic: un solo journalctl y un solo diálogo
        self._log_inflight = True
        threading.Thread(target=self._log_worker, daemon=True).start()

    def _log_worker(self) -> None:
        res = run(["journalctl", "--user", "-u", SERVICE, "-n", "200", "--no-pager"])
        GLib.idle_add(self._on_log_ready, res.stdout or res.stderr or "(vacío)")

    def _on_log_ready(self, text: str) -> bool:
        self._log_inflight = False
        self._show_text("Registro de Crotolamo", text)
        return False  # idle-callback de un solo uso

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
