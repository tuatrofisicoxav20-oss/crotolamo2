#!/usr/bin/env python3
"""Mini-ventana flotante para controlar Crotolamo — shim de compatibilidad.

El código real vive en el paquete `desktop/panel/` (systemd, config, terminal,
window, app). Este archivo se conserva porque install.sh y el lanzador
.desktop apuntan a esta ruta exacta; solo prepara sys.path y delega en
panel.app.main().

NO importa nada del núcleo de Crotolamo: lo controla a través del servicio de
usuario de systemd ('crotolamo.service'). Por eso se ejecuta con el python3 del
SISTEMA (el que trae gi/Gtk3), no con el venv del proyecto.

    /usr/bin/python3 desktop/crotolamo_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegura que desktop/ esté en sys.path aunque se importe desde otro cwd.
_DESKTOP_DIR = str(Path(__file__).resolve().parent)
if _DESKTOP_DIR not in sys.path:
    sys.path.insert(0, _DESKTOP_DIR)

from panel.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
