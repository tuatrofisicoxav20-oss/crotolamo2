#!/usr/bin/env python3
"""HUD estilo Jarvis para Crotolamo — overlay flotante sin bordes.

Aparece cuando el asistente sale del reposo y se oculta ~1.5 s después de volver
a él. Lee el estado en tiempo real de `~/.crotolamo/hud_state.json` publicado por
el subsistema de voz; en modo `--demo` cicla automáticamente entre los cuatro
modos sin necesitar el listener.

Arranque normal:
    /usr/bin/python3 desktop/crotolamo_hud.py

Arranque en modo demo (sin listener, sin LLM):
    /usr/bin/python3 desktop/crotolamo_hud.py --demo

Verificación de lógica headless (sin display, sin GTK):
    /usr/bin/python3 desktop/crotolamo_hud.py --check

Dependencias del sistema:
    python3-gobject  (siempre presente en Fedora con GTK3)
    gtk-layer-shell  (opcional, Wayland/Hyprland)
        sudo dnf install gtk-layer-shell

Si gtk-layer-shell no está disponible, el HUD cae a Gtk.Window normal con
set_keep_above(True) — funcional en X11 y en la mayoría de compositores Wayland
vía XWayland.  Para tener la posición y la capa de overlay perfectas en Hyprland
puro, instala gtk-layer-shell.

No importa nada de `crotolamo.*`: se ejecuta con /usr/bin/python3 del sistema,
sin el venv del proyecto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lógica pura (sin GTK) — testeable headless
# ---------------------------------------------------------------------------

HUD_STATE_FILE = Path.home() / ".crotolamo" / "hud_state.json"

# Modos reconocidos (igual que los valores JSON del contrato de integración)
VALID_MODES = frozenset({"idle", "listening", "thinking", "speaking"})


def parse_hud_state(raw: str) -> dict[str, Any]:
    """Parsea el JSON del contrato. Nunca lanza; devuelve {} si hay error."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, ValueError):
        return {}


def read_hud_file(path: Path) -> dict[str, Any]:
    """Lee y parsea hud_state.json de forma defensiva."""
    try:
        raw = path.read_text(encoding="utf-8")
        return parse_hud_state(raw)
    except FileNotFoundError:
        return {}
    except OSError:
        return {}


def extract_mode(data: dict[str, Any]) -> str:
    """Extrae y valida el campo 'mode'; devuelve 'idle' si no es válido."""
    mode = data.get("mode", "idle")
    if mode not in VALID_MODES:
        return "idle"
    return mode


def extract_text(data: dict[str, Any]) -> str:
    """Extrae el campo 'text' de forma segura."""
    text = data.get("text", "")
    if not isinstance(text, str):
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# Self-check headless (lógica pura, sin GTK ni display)
# ---------------------------------------------------------------------------

def _run_headless_checks() -> bool:
    """Verifica parse_hud_state / extract_mode / extract_text sin display.

    Devuelve True si todos los casos pasan, False si alguno falla.
    """
    ok = True
    cases = [
        # (descripción, raw_json, expected_mode, expected_text)
        ("JSON válido idle",
         '{"mode":"idle","turn_id":0,"text":"","ts":0.0,"pid":0}',
         "idle", ""),
        ("JSON válido listening",
         '{"mode":"listening","turn_id":1,"text":"qué tiempo hace","ts":1.0,"pid":123}',
         "listening", "qué tiempo hace"),
        ("JSON válido thinking",
         '{"mode":"thinking","turn_id":1,"text":"procesando","ts":2.0,"pid":123}',
         "thinking", "procesando"),
        ("JSON válido speaking",
         '{"mode":"speaking","turn_id":1,"text":"respuesta aquí","ts":3.0,"pid":123}',
         "speaking", "respuesta aquí"),
        ("String vacío (archivo vacío)",
         "",
         "idle", ""),
        ("JSON malformado",
         '{"mode": INVALID}',
         "idle", ""),
        ("Modo desconocido",
         '{"mode":"singing","text":"la la la"}',
         "idle", "la la la"),
        ("Campo text no es str",
         '{"mode":"listening","text":42}',
         "listening", ""),
        ("Sin campo mode",
         '{"text":"hola"}',
         "idle", "hola"),
        ("text con espacios al inicio/fin",
         '{"mode":"speaking","text":"  hola  "}',
         "speaking", "hola"),
        ("JSON es lista, no dict",
         '[1,2,3]',
         "idle", ""),
    ]

    print("── Headless self-check ──────────────────────────")
    for desc, raw, exp_mode, exp_text in cases:
        data = parse_hud_state(raw)
        got_mode = extract_mode(data)
        got_text = extract_text(data)
        pass_mode = got_mode == exp_mode
        pass_text = got_text == exp_text
        status = "OK" if (pass_mode and pass_text) else "FAIL"
        if not (pass_mode and pass_text):
            ok = False
        print(f"  [{status}] {desc}")
        if not pass_mode:
            print(f"        mode: esperado={exp_mode!r} obtenido={got_mode!r}")
        if not pass_text:
            print(f"        text: esperado={exp_text!r} obtenido={got_text!r}")

    print("── read_hud_file con archivo inexistente ────────")
    data = read_hud_file(Path("/tmp/hud_state_NOEXISTE_1234567.json"))
    if data != {}:
        print(f"  [FAIL] Esperaba {{}}, obtuvo {data!r}")
        ok = False
    else:
        print("  [OK] Archivo inexistente → {}")

    print("─────────────────────────────────────────────────")
    print("Resultado:", "TODOS OK" if ok else "HAY FALLOS")
    return ok


# ---------------------------------------------------------------------------
# Punto de entrada temprano para --check (antes de cargar GTK)
# ---------------------------------------------------------------------------
if "--check" in sys.argv:
    sys.exit(0 if _run_headless_checks() else 1)

# ---------------------------------------------------------------------------
# Importaciones GTK (sólo si no se sale antes con --check)
# ---------------------------------------------------------------------------
import gi  # noqa: E402

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

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

POLL_MS = 110          # intervalo de polling del archivo (~100-120 ms)
FADE_STEPS = 12        # pasos del fade-in/out
FADE_STEP_MS = 25      # ms entre pasos → fade total ~300 ms
HIDE_AFTER_IDLE_MS = 1500  # ms visibles tras volver a idle

DEMO_CYCLE_MODES = ["listening", "thinking", "speaking", "idle"]
DEMO_STEP_MS = 2500    # ms entre modos en --demo

# ---------------------------------------------------------------------------
# CSS — paleta cian/neón sobre fondo oscuro translúcido
# ---------------------------------------------------------------------------

CSS = """
/* ============================================================
   Crotolamo HUD  —  paleta Jarvis/Neon (GTK3)
   ============================================================ */

/* wrapper externo transparente (necesario para el visual RGBA) */
.hud-root {
    background: transparent;
}

/* tarjeta principal */
.hud-card {
    background: rgba(5, 10, 25, 0.82);
    border-radius: 20px;
    border: 1.5px solid rgba(0, 210, 255, 0.30);
    padding: 18px 24px 20px 24px;
    box-shadow: 0 0 32px rgba(0, 210, 255, 0.15),
                0 4px 20px rgba(0, 0, 0, 0.55);
}

/* marca / título */
.hud-brand {
    font-family: monospace;
    font-weight: 900;
    font-size: 11pt;
    letter-spacing: 5px;
    color: rgba(0, 210, 255, 0.75);
}

/* orbe central */
.hud-orb-wrap {
    padding: 6px 0 10px 0;
}

.hud-orb {
    border-radius: 999px;
    padding: 20px;
    border: 2px solid rgba(0, 210, 255, 0.20);
    background: rgba(0, 210, 255, 0.06);
    min-width: 80px;
    min-height: 80px;
}

/* ---- estados del orbe por modo ---- */

.mode-idle .hud-orb {
    border-color: rgba(100, 120, 140, 0.25);
    background: rgba(40, 50, 60, 0.30);
}
.mode-idle .hud-orb image {
    color: rgba(120, 140, 160, 0.60);
}

.mode-listening .hud-orb {
    border-color: rgba(0, 210, 255, 0.70);
    background: rgba(0, 210, 255, 0.12);
    box-shadow: 0 0 22px rgba(0, 210, 255, 0.35),
                inset 0 0 12px rgba(0, 210, 255, 0.10);
    animation: pulse-listen 1.2s ease-in-out infinite;
}
.mode-listening .hud-orb image {
    color: #00d2ff;
}

.mode-thinking .hud-orb {
    border-color: rgba(100, 80, 255, 0.70);
    background: rgba(90, 60, 220, 0.14);
    box-shadow: 0 0 26px rgba(100, 80, 255, 0.40),
                inset 0 0 14px rgba(100, 80, 255, 0.12);
    animation: pulse-think 1.6s ease-in-out infinite;
}
.mode-thinking .hud-orb image {
    color: #a07fff;
}

.mode-speaking .hud-orb {
    border-color: rgba(0, 255, 180, 0.70);
    background: rgba(0, 200, 140, 0.12);
    box-shadow: 0 0 26px rgba(0, 255, 180, 0.40),
                inset 0 0 14px rgba(0, 200, 140, 0.10);
    animation: pulse-speak 0.9s ease-in-out infinite;
}
.mode-speaking .hud-orb image {
    color: #00ffb4;
}

/* spinner GTK para "thinking" (Gtk.Spinner, auto-animado) */
.hud-spinner {
    color: #a07fff;
}

/* etiqueta de modo */
.hud-mode-label {
    font-family: monospace;
    font-size: 9pt;
    letter-spacing: 3px;
    color: rgba(0, 210, 255, 0.70);
    padding-top: 2px;
}

/* texto reconocido / respuesta del asistente */
.hud-text {
    font-size: 10.5pt;
    color: rgba(200, 230, 255, 0.90);
    padding-top: 8px;
}

/* separador decorativo */
.hud-sep {
    background: rgba(0, 210, 255, 0.12);
    min-height: 1px;
    margin: 6px 0;
}

/* ---- @keyframes (GTK3: box-shadow, opacity, color) ---- */

@keyframes pulse-listen {
    0%   { box-shadow: 0 0 10px rgba(0,210,255,0.20), inset 0 0 6px rgba(0,210,255,0.08); }
    50%  { box-shadow: 0 0 32px rgba(0,210,255,0.55), inset 0 0 16px rgba(0,210,255,0.18); }
    100% { box-shadow: 0 0 10px rgba(0,210,255,0.20), inset 0 0 6px rgba(0,210,255,0.08); }
}

@keyframes pulse-think {
    0%   { box-shadow: 0 0 12px rgba(100,80,255,0.25), inset 0 0 8px rgba(100,80,255,0.10); }
    50%  { box-shadow: 0 0 36px rgba(100,80,255,0.60), inset 0 0 18px rgba(100,80,255,0.20); }
    100% { box-shadow: 0 0 12px rgba(100,80,255,0.25), inset 0 0 8px rgba(100,80,255,0.10); }
}

@keyframes pulse-speak {
    0%   { box-shadow: 0 0 10px rgba(0,255,180,0.20), inset 0 0 6px rgba(0,200,140,0.08); }
    50%  { box-shadow: 0 0 34px rgba(0,255,180,0.55), inset 0 0 16px rgba(0,200,140,0.18); }
    100% { box-shadow: 0 0 10px rgba(0,255,180,0.20), inset 0 0 6px rgba(0,200,140,0.08); }
}
"""

# ---------------------------------------------------------------------------
# Labels e iconos por modo
# ---------------------------------------------------------------------------

MODE_LABELS: dict[str, str] = {
    "idle":      "  IDLE  ",
    "listening": "ESCUCHANDO",
    "thinking":  "PENSANDO",
    "speaking":  "HABLANDO",
}

MODE_ICONS: dict[str, str] = {
    "idle":      "microphone-sensitivity-low-symbolic",
    "listening": "microphone-sensitivity-high-symbolic",
    "thinking":  "system-search-symbolic",
    "speaking":  "audio-volume-high-symbolic",
}

# ---------------------------------------------------------------------------
# Ventana HUD
# ---------------------------------------------------------------------------


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
        self._apply_css()

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

        # --- timers de polling y demo ---
        # En modo demo NO se lee el archivo (el demo lo ignora deliberadamente
        # para que no interfiera un hud_state.json inexistente o idle).
        if self._demo:
            GLib.timeout_add(DEMO_STEP_MS, self._demo_tick)
        else:
            GLib.timeout_add(POLL_MS, self._poll_tick)

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
    # CSS
    # -----------------------------------------------------------------------

    def _apply_css(self) -> None:
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS.encode("utf-8"))
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)

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
        ctx = self._card.get_style_context()
        for cls in self._ALL_MODE_CLASSES:
            ctx.remove_class(cls)
        ctx.add_class(f"mode-{mode}")

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
        no sea cancelado en cada poll subsiguiente de idle (110ms < 1500ms).
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
    # Polling del archivo de estado (~100-120 ms)
    # -----------------------------------------------------------------------

    def _poll_tick(self) -> bool:
        """Llamado por GLib cada POLL_MS ms; lee hud_state.json."""
        data = read_hud_file(HUD_STATE_FILE)
        mode = extract_mode(data)
        text = extract_text(data)
        self._transition_to(mode, text)
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


# ---------------------------------------------------------------------------
# Aplicación GTK
# ---------------------------------------------------------------------------

class HUDApp(Gtk.Application):
    def __init__(self, demo: bool = False) -> None:
        super().__init__(application_id="org.crotolamo.HUD")
        self._demo = demo
        self._win: HUDWindow | None = None

    def do_startup(self) -> None:
        Gtk.Application.do_startup(self)
        # Mantener la app VIVA aunque la ventana esté oculta (unmapped). Sin esto,
        # Gtk.Application se cierra al quedarse sin ventanas visibles — y el HUD
        # pasa la mayor parte del tiempo oculto, esperando a ser convocado por voz.
        self.hold()

    def do_activate(self) -> None:
        if self._win is None:
            self._win = HUDWindow(demo=self._demo)
            self.add_window(self._win)
        # NO hacemos present(): el HUD arranca OCULTO y se muestra solo cuando se
        # le convoca (poll de hud_state.json) o, en --demo, en el primer ciclo.


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> int:
    demo = "--demo" in sys.argv
    app = HUDApp(demo=demo)
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
