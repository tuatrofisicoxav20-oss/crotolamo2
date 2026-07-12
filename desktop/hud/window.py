"""Ventana del HUD (GTK3 + gtk-layer-shell opcional).

El estado se sigue con Gio.FileMonitor (inotify) sobre el DIRECTORIO padre de
hud_state.json — el loop reemplaza el archivo por rename, y monitorear el
archivo directo puede perder eventos con reemplazo atómico. Se conserva un
poll de respaldo LENTO (BACKUP_POLL_MS) por robustez, que se pausa mientras el
HUD está oculto (el monitor sigue activo, así que no se pierde el despertar).
Si el monitor no puede crearse, se cae al polling clásico rápido y sin pausas.
"""

from __future__ import annotations

import logging

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # noqa: E402

from common.gtk_util import apply_css, swap_class  # noqa: E402
from common.ipc import HUD_STATE_FILE  # noqa: E402

from .state import extract_mode, extract_text, read_hud_file  # noqa: E402
from .style import CSS, MODE_ICONS, MODE_LABELS  # noqa: E402

log = logging.getLogger("crotolamo.desktop.hud")

# Intento de carga de gtk-layer-shell (opcional, Wayland/Hyprland)
_LAYER_SHELL_AVAILABLE = False
try:
    gi.require_version("GtkLayerShell", "0.1")
    from gi.repository import GtkLayerShell  # type: ignore[attr-defined]

    _LAYER_SHELL_AVAILABLE = True
except (ValueError, ImportError):
    GtkLayerShell = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constantes de UI
# ---------------------------------------------------------------------------

POLL_MS = 110          # polling clásico — SOLO como fallback si no hay monitor
BACKUP_POLL_MS = 2000  # poll de respaldo lento cuando el FileMonitor funciona
FADE_STEPS = 12        # pasos del fade-in/out
FADE_STEP_MS = 25      # ms entre pasos → fade total ~300 ms
HIDE_AFTER_IDLE_MS = 1500  # ms visibles tras volver a idle

DEMO_CYCLE_MODES = ["listening", "thinking", "speaking", "idle"]
DEMO_STEP_MS = 2500    # ms entre modos en --demo


class HUDWindow(Gtk.Window):
    """Overlay flotante estilo Jarvis para Crotolamo."""

    _ALL_MODE_CLASSES = frozenset(
        {"mode-idle", "mode-listening", "mode-thinking", "mode-speaking"})

    def __init__(self, demo: bool = False) -> None:
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        self._demo = demo
        self._current_mode = "idle"
        self._hide_timer_id: int | None = None
        self._fade_timer_id: int | None = None
        self._fade_opacity: float = 0.0
        self._fading_in: bool = False
        self._demo_mode_idx: int = 0
        self._monitor: Gio.FileMonitor | None = None
        self._backup_timer_id: int | None = None

        # --- propiedades de ventana (sin decoraciones, sin robar foco) ---
        self.set_title("Crotolamo HUD")
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_accept_focus(False)
        self.set_focus_on_map(False)
        self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)

        # translucencia RGBA (requiere compositor con soporte alpha)
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)
        self.set_app_paintable(True)

        # arrancar invisible
        self.set_opacity(0.0)

        # --- construir widgets ---
        self._build_ui()
        apply_css(self.get_screen(), CSS)

        # --- Wayland: gtk-layer-shell si disponible; fallback X11 ---
        if _LAYER_SHELL_AVAILABLE and GtkLayerShell is not None:
            self._setup_layer_shell()
        else:
            self.set_keep_above(True)

        self.show_all()

        # posicionar después de show_all (ya tenemos las dimensiones reales)
        if not (_LAYER_SHELL_AVAILABLE and GtkLayerShell is not None):
            self._position_window()

        # Arrancar OCULTO de verdad. En Wayland/Hyprland set_opacity sobre un
        # toplevel es no-op, así que la invisibilidad real la da hide() (unmap);
        # el opacity 0 queda para animar el fade en X11.
        self.set_opacity(0.0)
        self.hide()

        # --- seguimiento del estado (monitor + respaldo) o demo ---
        # En modo demo NO se lee el archivo (el demo lo ignora deliberadamente
        # para que no interfiera un hud_state.json inexistente o idle).
        if self._demo:
            GLib.timeout_add(DEMO_STEP_MS, self._demo_tick)
        else:
            self._setup_state_watch()

    # -----------------------------------------------------------------------
    # Construcción de la UI
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Crea la jerarquía de widgets."""
        # wrapper externo transparente (para margen de sombra)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        outer.get_style_context().add_class("hud-root")
        outer.set_margin_top(12)
        outer.set_margin_bottom(12)
        outer.set_margin_start(12)
        outer.set_margin_end(12)
        self.add(outer)

        # tarjeta interna: fondo oscuro translúcido + bordes redondeados
        self._card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._card.get_style_context().add_class("hud-card")
        self._card.get_style_context().add_class("mode-idle")
        outer.pack_start(self._card, True, True, 0)

        # marca / título
        brand = Gtk.Label(label="C  R  O  T  O  L  A  M  O")
        brand.get_style_context().add_class("hud-brand")
        brand.set_halign(Gtk.Align.CENTER)
        self._card.pack_start(brand, False, False, 0)

        # orbe central
        orb_wrap = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        orb_wrap.get_style_context().add_class("hud-orb-wrap")
        orb_wrap.set_halign(Gtk.Align.CENTER)
        self._card.pack_start(orb_wrap, False, False, 0)

        self._orb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._orb.get_style_context().add_class("hud-orb")
        self._orb.set_halign(Gtk.Align.CENTER)
        self._orb.set_valign(Gtk.Align.CENTER)
        orb_wrap.pack_start(self._orb, False, False, 0)

        # icono simbólico dentro del orbe
        self._icon = Gtk.Image.new_from_icon_name(
            MODE_ICONS["idle"], Gtk.IconSize.DIALOG)
        self._icon.set_pixel_size(52)
        self._orb.pack_start(self._icon, True, True, 0)

        # spinner GTK para modo "thinking"
        # (Gtk.Spinner es auto-animado y extremadamente liviano en RAM)
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(52, 52)
        self._spinner.get_style_context().add_class("hud-spinner")
        # set_no_show_all evita que show_all() lo haga visible;
        # lo controlamos manualmente en _apply_mode().
        self._spinner.set_no_show_all(True)
        self._orb.pack_start(self._spinner, True, True, 0)

        # etiqueta de modo (ESCUCHANDO / PENSANDO / etc.)
        self._mode_label = Gtk.Label(label=MODE_LABELS["idle"])
        self._mode_label.get_style_context().add_class("hud-mode-label")
        self._mode_label.set_halign(Gtk.Align.CENTER)
        self._card.pack_start(self._mode_label, False, False, 0)

        # separador decorativo
        sep = Gtk.Box()
        sep.get_style_context().add_class("hud-sep")
        sep.set_size_request(-1, 1)
        self._card.pack_start(sep, False, False, 4)

        # texto (frase reconocida del usuario o respuesta del asistente)
        self._text_label = Gtk.Label(label="")
        self._text_label.get_style_context().add_class("hud-text")
        self._text_label.set_halign(Gtk.Align.CENTER)
        self._text_label.set_line_wrap(True)
        self._text_label.set_max_width_chars(40)
        self._text_label.set_justify(Gtk.Justification.CENTER)
        self._card.pack_start(self._text_label, False, False, 0)

    # -----------------------------------------------------------------------
    # Posicionamiento (fallback X11 / XWayland)
    # -----------------------------------------------------------------------

    def _position_window(self) -> None:
        """Centra el HUD horizontalmente en la parte superior de la pantalla."""
        screen = self.get_screen()
        geo = screen.get_monitor_geometry(0)
        win_w, _win_h = self.get_size()
        x = geo.x + (geo.width - win_w) // 2
        y = geo.y + 48  # 48 px desde arriba (hueco para waybar/panel)
        self.move(x, y)

    # -----------------------------------------------------------------------
    # gtk-layer-shell (Wayland puro — Hyprland, Sway, etc.)
    # -----------------------------------------------------------------------

    def _setup_layer_shell(self) -> None:
        """Configura el HUD como overlay Wayland via gtk-layer-shell.

        Debe llamarse ANTES de show().  Si gtk-layer-shell no está instalado,
        este método nunca se llama (se usa el fallback).
        """
        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        # Anclar al borde superior, centrado (solo TOP activo)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.TOP, 48)
        # No excluir zona — no quita espacio a otras ventanas
        GtkLayerShell.set_exclusive_zone(self, 0)

    # -----------------------------------------------------------------------
    # Cambio de modo visual
    # -----------------------------------------------------------------------

    def _apply_mode(self, mode: str, text: str = "") -> None:
        """Actualiza el aspecto visual al modo indicado."""
        # intercambiar clases de modo en la tarjeta
        swap_class(self._card.get_style_context(),
                   self._ALL_MODE_CLASSES, f"mode-{mode}")

        # mostrar spinner para thinking, icono para el resto
        if mode == "thinking":
            self._icon.set_visible(False)
            self._spinner.set_visible(True)
            self._spinner.start()
        else:
            self._spinner.stop()
            self._spinner.set_visible(False)
            self._icon.set_visible(True)
            icon_name = MODE_ICONS.get(mode, MODE_ICONS["idle"])
            self._icon.set_from_icon_name(icon_name, Gtk.IconSize.DIALOG)
            self._icon.set_pixel_size(52)

        self._mode_label.set_label(MODE_LABELS.get(mode, mode.upper()))
        self._text_label.set_label(text)

    # -----------------------------------------------------------------------
    # Fade in / out mediante set_opacity()
    # -----------------------------------------------------------------------

    def _cancel_fade(self) -> None:
        if self._fade_timer_id is not None:
            GLib.source_remove(self._fade_timer_id)
            self._fade_timer_id = None

    def _start_fade_in(self) -> None:
        self._cancel_fade()
        # Lo ESENCIAL en Wayland es MAPEAR la ventana: show() la hace visible
        # (en X11, además, el opacity de abajo la animará suavemente).
        self.show()
        # visible: reanudar el poll de respaldo
        self._resume_backup_poll()
        self._fading_in = True
        self._fade_opacity = self.get_opacity()
        self._fade_timer_id = GLib.timeout_add(FADE_STEP_MS, self._fade_step)

    def _start_fade_out(self) -> None:
        self._cancel_fade()
        self._fading_in = False
        self._fade_opacity = self.get_opacity()
        self._fade_timer_id = GLib.timeout_add(FADE_STEP_MS, self._fade_step)

    def _fade_step(self) -> bool:
        step = 1.0 / FADE_STEPS
        if self._fading_in:
            self._fade_opacity = min(1.0, self._fade_opacity + step)
            self.set_opacity(self._fade_opacity)
            if self._fade_opacity >= 1.0:
                self._fade_timer_id = None
                return False
        else:
            self._fade_opacity = max(0.0, self._fade_opacity - step)
            self.set_opacity(self._fade_opacity)
            if self._fade_opacity <= 0.0:
                self._fade_timer_id = None
                # Unmap REAL: en Wayland es lo único que oculta de verdad.
                self.hide()
                # oculto: pausar el respaldo (el FileMonitor nos despierta)
                self._pause_backup_poll()
                return False
        return True

    # -----------------------------------------------------------------------
    # Timer de ocultamiento tras idle
    # -----------------------------------------------------------------------

    def _cancel_hide_timer(self) -> None:
        if self._hide_timer_id is not None:
            GLib.source_remove(self._hide_timer_id)
            self._hide_timer_id = None

    def _schedule_hide(self) -> None:
        self._cancel_hide_timer()
        self._hide_timer_id = GLib.timeout_add(HIDE_AFTER_IDLE_MS, self._do_hide)

    def _do_hide(self) -> bool:
        self._hide_timer_id = None
        self._start_fade_out()
        return False

    # -----------------------------------------------------------------------
    # Máquina de estados (transición de modo)
    # -----------------------------------------------------------------------

    def _transition_to(self, mode: str, text: str = "") -> None:
        """Cambia al modo indicado, con fade-in/out y timer de ocultamiento.

        La guarda cubre TODOS los modos (incluido idle) para que el timer de
        ocultamiento se programe UNA SOLA VEZ en la transición non-idle→idle y
        no sea cancelado en cada lectura subsiguiente de idle.
        """
        if mode == self._current_mode:
            # mismo modo: sólo actualizar texto si hay novedad (y no es idle)
            if text and mode != "idle":
                self._text_label.set_label(text)
            return

        self._current_mode = mode
        self._apply_mode(mode, text)

        if mode != "idle":
            # convocado: cancelar hide pendiente y aparecer con fade-in
            self._cancel_hide_timer()
            if self.get_opacity() < 0.99:
                self._start_fade_in()
        else:
            # volvió a reposo: programar ocultamiento en 1.5 s
            # (cancela cualquier hide previo para no acortar si ya estaba en idle)
            self._schedule_hide()

    # -----------------------------------------------------------------------
    # Seguimiento del archivo de estado: Gio.FileMonitor + poll de respaldo
    # -----------------------------------------------------------------------

    def _setup_state_watch(self) -> None:
        """Monitor inotify del directorio padre + respaldo lento pausable."""
        try:
            parent = Gio.File.new_for_path(str(HUD_STATE_FILE.parent))
            # Directorio (no archivo): el loop reemplaza hud_state.json por
            # rename y el monitor de archivo puede perder esos eventos.
            self._monitor = parent.monitor_directory(
                Gio.FileMonitorFlags.WATCH_MOVES, None)
            self._monitor.connect("changed", self._on_state_dir_event)
        except GLib.Error as exc:
            self._monitor = None
            log.warning("No se pudo crear el FileMonitor (%s); "
                        "uso polling clásico cada %d ms", exc, POLL_MS)

        if self._monitor is None:
            # sin inotify no es seguro pausar: polling rápido permanente
            self._backup_timer_id = GLib.timeout_add(POLL_MS, self._poll_tick)
        else:
            # arrancamos ocultos → respaldo pausado; una lectura inicial por
            # si ya había estado non-idle publicado antes de arrancar.
            GLib.idle_add(self._read_state_once)

    def _on_state_dir_event(self, _monitor, gfile, other_file, _event) -> None:
        """Evento inotify en ~/.crotolamo: filtra por nombre y relee."""
        names = {gfile.get_basename()}
        if other_file is not None:
            names.add(other_file.get_basename())
        if HUD_STATE_FILE.name not in names:
            return
        self._read_state_once()

    def _pause_backup_poll(self) -> None:
        """Detiene el respaldo mientras el HUD está oculto (solo con monitor)."""
        if self._demo or self._monitor is None:
            return
        if self._backup_timer_id is not None:
            GLib.source_remove(self._backup_timer_id)
            self._backup_timer_id = None

    def _resume_backup_poll(self) -> None:
        if self._demo or self._monitor is None:
            return
        if self._backup_timer_id is None:
            self._backup_timer_id = GLib.timeout_add(
                BACKUP_POLL_MS, self._poll_tick)

    def _read_state_once(self) -> bool:
        """Lee hud_state.json y aplica la transición; usable como idle-callback."""
        data = read_hud_file(HUD_STATE_FILE)
        self._transition_to(extract_mode(data), extract_text(data))
        return False  # no repetir (para GLib.idle_add)

    def _poll_tick(self) -> bool:
        """Poll periódico (respaldo lento o fallback clásico)."""
        self._read_state_once()
        return True  # seguir el timer

    # -----------------------------------------------------------------------
    # Modo demo: cicla entre los cuatro modos sin listener ni LLM
    # -----------------------------------------------------------------------

    def _demo_tick(self) -> bool:
        """Cicla entre modos en --demo."""
        mode = DEMO_CYCLE_MODES[self._demo_mode_idx % len(DEMO_CYCLE_MODES)]
        demo_texts: dict[str, str] = {
            "listening": "di tu comando…",
            "thinking":  "procesando…",
            "speaking":  "aquí va mi respuesta",
            "idle":      "",
        }
        self._transition_to(mode, demo_texts.get(mode, ""))
        self._demo_mode_idx += 1
        return True
