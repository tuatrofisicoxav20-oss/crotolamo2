#!/usr/bin/env bash
# Instala la integración de escritorio de Crotolamo 2 en Fedora/GNOME:
#   - servicio de usuario systemd 'crotolamo.service' (apunta a este repo)
#   - migra/apaga el servicio viejo de Crotolamo 1 (con backup)
#   - lanzador en el menú de apps + icono
#   - modo de voz por defecto (half-duplex)
#
# Uso:
#   ./desktop/install.sh            instala (no arranca el servicio)
#   ./desktop/install.sh --start    instala y enciende Crotolamo
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../crotolamo2/desktop
ROOT="$(cd "$HERE/.." && pwd)"                          # .../crotolamo2
VENV_PY="$ROOT/.venv/bin/python"

SD_DIR="$HOME/.config/systemd/user"
APP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
CFG_DIR="$HOME/.config/crotolamo"

OLD="crotolamo-listener.service"
NEW="crotolamo.service"

echo "==> Integración de escritorio de Crotolamo 2"
echo "    repo: $ROOT"

# --- comprobaciones mínimas ---------------------------------------------------
if [ ! -x "$VENV_PY" ]; then
  echo "!! No encuentro el venv: $VENV_PY"
  echo "   Créalo:  cd '$ROOT' && python3 -m venv .venv && .venv/bin/pip install -e '.[voice]'"
  exit 1
fi

mkdir -p "$SD_DIR" "$APP_DIR" "$ICON_DIR" "$CFG_DIR"

# --- 1) migrar el servicio viejo de Crotolamo 1 (si existe) -------------------
if [ -f "$SD_DIR/$OLD" ]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  cp -f "$SD_DIR/$OLD" "$SD_DIR/$OLD.c1-backup-$ts"
  echo "==> Backup del servicio viejo  ->  $OLD.c1-backup-$ts"
  systemctl --user disable --now "$OLD" 2>/dev/null || true
  rm -f "$SD_DIR/$OLD"
  echo "==> Servicio de Crotolamo 1 apagado y deshabilitado"
fi

# --- 2) instalar el servicio nuevo (apunta a C2) -----------------------------
install -m 644 "$HERE/crotolamo.service" "$SD_DIR/$NEW"
echo "==> Servicio instalado: $SD_DIR/$NEW"

# --- 3) modo de voz por defecto (si no hay uno ya) ---------------------------
if [ ! -f "$CFG_DIR/listener.env" ]; then
  echo 'CROTOLAMO_LISTEN_ARGS=--no-barge-in' > "$CFG_DIR/listener.env"
  echo "==> Modo por defecto: half-duplex (altavoces)"
fi

# --- 4) icono -----------------------------------------------------------------
install -m 644 "$HERE/crotolamo.svg" "$ICON_DIR/crotolamo.svg"
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# --- 5) lanzador .desktop (con la ruta real del panel) -----------------------
sed "s#@PANEL@#$HERE/crotolamo_panel.py#g" "$HERE/crotolamo.desktop.in" \
    > "$APP_DIR/crotolamo.desktop"
chmod 644 "$APP_DIR/crotolamo.desktop"
update-desktop-database "$APP_DIR" 2>/dev/null || true
echo "==> Lanzador instalado: «Crotolamo» en el menú de apps"

# --- 6) recargar systemd y habilitar arranque con la sesión ------------------
systemctl --user daemon-reload
systemctl --user enable "$NEW" 2>/dev/null || true
echo "==> Habilitado para iniciar con la sesión"

if [ "${1:-}" = "--start" ]; then
  systemctl --user restart "$NEW"
  echo "==> Crotolamo encendido"
fi

echo
echo "Listo, patrón."
echo "  • Abre «Crotolamo» desde el menú de aplicaciones, o lanza:"
echo "      /usr/bin/python3 $HERE/crotolamo_panel.py"
echo "  • Encender/apagar a mano:  systemctl --user start|stop $NEW"
