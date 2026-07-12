#!/usr/bin/env python3
"""HUD estilo Jarvis para Crotolamo — shim de compatibilidad.

El código real vive en el paquete `desktop/hud/` (state, style, window, app).
Este archivo se conserva porque crotolamo-hud.service e install.sh apuntan a
esta ruta exacta; solo prepara sys.path y delega en hud.app.main().

Arranque normal:
    /usr/bin/python3 desktop/crotolamo_hud.py

Arranque en modo demo (sin listener, sin LLM):
    /usr/bin/python3 desktop/crotolamo_hud.py --demo

Dependencias del sistema:
    python3-gobject  (siempre presente en Fedora con GTK3)
    gtk-layer-shell  (opcional, Wayland/Hyprland)
        sudo dnf install gtk-layer-shell

Si gtk-layer-shell no está disponible, el HUD cae a Gtk.Window normal con
set_keep_above(True) — funcional en X11 y en la mayoría de compositores Wayland
vía XWayland.

No importa nada de `crotolamo.*`: se ejecuta con /usr/bin/python3 del sistema,
sin el venv del proyecto.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegura que desktop/ esté en sys.path aunque se importe desde otro cwd.
_DESKTOP_DIR = str(Path(__file__).resolve().parent)
if _DESKTOP_DIR not in sys.path:
    sys.path.insert(0, _DESKTOP_DIR)

from hud.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
