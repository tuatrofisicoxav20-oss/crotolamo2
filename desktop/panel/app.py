"""Aplicación GTK del panel y punto de entrada."""

from __future__ import annotations

import logging

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from .window import Panel  # noqa: E402


class App(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="org.crotolamo.Panel")
        self.win: Panel | None = None

    def do_activate(self) -> None:
        if self.win is None:
            self.win = Panel(self)
        self.win.present()


def main() -> int:
    # Los warnings (systemctl caído, E/S, etc.) van a stderr → journal.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    app = App()
    return app.run(None)
