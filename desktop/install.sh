#!/usr/bin/env bash
# Instala la integración de escritorio de Crotolamo 2 en Fedora/GNOME:
#   - servicio de usuario systemd 'crotolamo.service' (apunta a este repo)
#   - servicio del HUD 'crotolamo-hud.service' (el overlay al convocar por voz)
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
HUD="crotolamo-hud.service"

echo "==> Integración de escritorio de Crotolamo 2"
echo "    repo: $ROOT"

# --- comprobaciones mínimas ---------------------------------------------------
# La ruta del repo acaba dentro de los units y del lanzador. Con $ ` " o \ systemd
# y el .desktop la leerían como variables, escapes o comillas, y systemd no acepta
# un ejecutable cuya ruta lleve ' (ni entre comillas): mejor no instalar nada que
# dejar un servicio que no arranca.
if [[ "$ROOT" == *[\$\`\"\'\\]* || "$ROOT" == *$'\n'* ]]; then
  echo "!! La ruta del repo lleva \$, \`, \", ' o \\ y no sé escribirla en un unit de systemd:"
  echo "   $ROOT"
  echo "   Mueve el repo a una carpeta sin esos caracteres y vuelve a correr el instalador."
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  # Con el índice CPU de PyTorch, como la CI: silero-vad (extra [voice]) declara torch
  # y, solo con PyPI, pip baja la build CUDA (~3 GB de torch + nvidia-* + triton) que
  # en esta lap sin GPU no corre nunca. Con el índice queda torch+cpu y ningún nvidia-*.
  echo "!! No encuentro el venv: $VENV_PY"
  echo "   Créalo:  cd \"$ROOT\" && python3 -m venv .venv && \\"
  echo "              .venv/bin/pip install -e '.[voice]' \\"
  echo "              --extra-index-url https://download.pytorch.org/whl/cpu"
  exit 1
fi

mkdir -p "$SD_DIR" "$APP_DIR" "$ICON_DIR" "$CFG_DIR"

# --- los units llevan la ruta REAL del repo ----------------------------------
# Las plantillas de desktop/ están escritas para el clon de siempre. Copiadas tal
# cual, un clon en otra carpeta dejaba un servicio que no arranca (200/CHDIR,
# 203/EXEC) y que systemd reintenta cada 3 s.
TPL_ROOT="%h/Documentos/crotolamo2"

# La ruta como la lee systemd: relativa a %h si cuelga del HOME (con el clon de
# siempre el unit instalado es idéntico a la plantilla) y con % como %%, porque
# los especificadores (%h, %u…) se expanden en todas las claves que la llevan.
if [[ "$ROOT" == "$HOME"/* ]]; then
  UNIT_PRE="%h/"; UNIT_REL="${ROOT#"$HOME"/}"
else
  UNIT_PRE=""; UNIT_REL="$ROOT"
fi

# Documentation= es una URL: systemd ignora (con un aviso) la que lleva espacios
# o acentos, así que va en %XX (y cada % como %% para systemd).
url_path() {
  local LC_ALL=C s="$1" out="" c i
  for ((i = 0; i < ${#s}; i++)); do
    c="${s:i:1}"
    case "$c" in
      [A-Za-z0-9/._~-]) out+="$c" ;;
      *) printf -v c '%%%%%02X' "'$c"; out+="$c" ;;
    esac
  done
  printf '%s' "$out"
}

UNIT_ROOT="$UNIT_PRE${UNIT_REL//%/%%}"
DOC_ROOT="$UNIT_PRE$(url_path "$UNIT_REL")"

# Plantilla -> unit. En ExecStart la palabra entera (ruta + sufijo) va entre
# comillas si lleva espacios: systemd parte la línea por espacios.
render_unit() {  # $1 plantilla, $2 destino
  local line pre rest word
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == *"$TPL_ROOT"* ]]; then
      case "$line" in
        Documentation=*) line="${line//"$TPL_ROOT"/"$DOC_ROOT"}" ;;
        ExecStart=*)
          pre="${line%%"$TPL_ROOT"*}"
          rest="${line#*"$TPL_ROOT"}"
          word="$UNIT_ROOT${rest%%[[:space:]]*}"
          rest="${rest#"${rest%%[[:space:]]*}"}"
          if [[ "$word" == *[[:space:]]* ]]; then word="\"$word\""; fi
          line="$pre$word$rest" ;;
        *) line="${line//"$TPL_ROOT"/"$UNIT_ROOT"}" ;;
      esac
    fi
    printf '%s\n' "$line"
  done < "$1" > "$2"
}

# Si ya había un unit distinto (editado a mano), se respalda antes de pisarlo.
install_unit() {  # $1 nombre del unit (el mismo en desktop/ y en $SD_DIR)
  local nuevo="$SD_DIR/.$1.nuevo" ts
  render_unit "$HERE/$1" "$nuevo"
  chmod 644 "$nuevo"
  if [ -f "$SD_DIR/$1" ] && [[ "$(< "$SD_DIR/$1")" != "$(< "$nuevo")" ]]; then
    ts="$(date +%Y%m%d-%H%M%S)"
    cp -f "$SD_DIR/$1" "$SD_DIR/$1.bak-$ts"
    echo "==> Había otro $1: respaldo en $1.bak-$ts"
  fi
  mv -f "$nuevo" "$SD_DIR/$1"
  echo "==> Servicio instalado: $SD_DIR/$1"
}

# --- 1) migrar el servicio viejo de Crotolamo 1 (si existe) -------------------
if [ -f "$SD_DIR/$OLD" ]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  cp -f "$SD_DIR/$OLD" "$SD_DIR/$OLD.c1-backup-$ts"
  echo "==> Backup del servicio viejo  ->  $OLD.c1-backup-$ts"
  systemctl --user disable --now "$OLD" 2>/dev/null || true
  rm -f "$SD_DIR/$OLD"
  echo "==> Servicio de Crotolamo 1 apagado y deshabilitado"
fi

# --- 2) instalar los servicios nuevos (apuntan a C2) -------------------------
install_unit "$NEW"
install_unit "$HUD"

# --- 3) modo de voz por defecto (si no hay uno ya) ---------------------------
if [ ! -f "$CFG_DIR/listener.env" ]; then
  echo 'CROTOLAMO_LISTEN_ARGS=--no-barge-in' > "$CFG_DIR/listener.env"
  echo "==> Modo por defecto: half-duplex (altavoces)"
fi

# --- 4) icono -----------------------------------------------------------------
install -m 644 "$HERE/crotolamo.svg" "$ICON_DIR/crotolamo.svg"
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# --- 5) lanzador .desktop (con la ruta real del panel) -----------------------
# La plantilla la pone entre comillas (con un espacio se partía en dos
# argumentos); % va como %% (códigos de campo del Exec=). Sin sed: un & o un #
# en la ruta rompían la sustitución.
PANEL="$HERE/crotolamo_panel.py"
LANZADOR="$(< "$HERE/crotolamo.desktop.in")"
printf '%s\n' "${LANZADOR//@PANEL@/"${PANEL//%/%%}"}" > "$APP_DIR/crotolamo.desktop"
chmod 644 "$APP_DIR/crotolamo.desktop"
update-desktop-database "$APP_DIR" 2>/dev/null || true
echo "==> Lanzador instalado: «Crotolamo» en el menú de apps"

# --- 6) recargar systemd y habilitar arranque con la sesión ------------------
systemctl --user daemon-reload
systemctl --user enable "$NEW" "$HUD" 2>/dev/null || true
echo "==> Habilitado para iniciar con la sesión (escucha y HUD)"

if [ "${1:-}" = "--start" ]; then
  systemctl --user restart "$NEW"
  # El HUD necesita la sesión gráfica: arrancado desde una consola sin ella se
  # quedaría reintentando. Solo se reinicia si ya corría (para que use el unit
  # nuevo); si no, arranca con la próxima sesión.
  systemctl --user try-restart "$HUD" 2>/dev/null || true
  echo "==> Crotolamo encendido"
fi

echo
echo "Listo, patrón."
echo "  • Abre «Crotolamo» desde el menú de aplicaciones, o lanza:"
echo "      /usr/bin/python3 \"$HERE/crotolamo_panel.py\""
echo "  • Encender/apagar a mano:  systemctl --user start|stop $NEW"
echo "  • El HUD arranca con la sesión gráfica; para verlo ya:  systemctl --user start $HUD"
