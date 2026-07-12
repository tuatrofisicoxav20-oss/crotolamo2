"""Utilidades GTK compartidas por el HUD y el panel.

ÚNICO módulo de common/ que importa gi: no lo importes desde lógica pura.
"""

from __future__ import annotations

from collections.abc import Iterable

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402


def apply_css(screen, css: str | bytes) -> None:
    """Registra una hoja CSS a nivel de pantalla (prioridad app + 1)."""
    provider = Gtk.CssProvider()
    data = css if isinstance(css, bytes) else css.encode("utf-8")
    provider.load_from_data(data)
    Gtk.StyleContext.add_provider_for_screen(
        screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)


def swap_class(ctx, all_classes: Iterable[str], active: str) -> None:
    """Deja `active` como única clase del grupo `all_classes` en el contexto."""
    for cls in all_classes:
        ctx.remove_class(cls)
    ctx.add_class(active)
