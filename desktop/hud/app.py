"""Aplicación GTK del HUD y punto de entrada."""

from __future__ import annotations

import logging
import sys

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from .window import HUDWindow  # noqa: E402


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
        # le convoca (cambio en hud_state.json) o, en --demo, en el primer ciclo.


def main() -> int:
    # Los warnings (monitor caído, etc.) van a stderr → journal del servicio.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    demo = "--demo" in sys.argv
    app = HUDApp(demo=demo)
    return app.run(None)
