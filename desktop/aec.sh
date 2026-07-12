#!/usr/bin/env bash
# Activa/desactiva la cancelación de eco (AEC) de PipeWire para Crotolamo.
#
#   ./aec.sh on      instala la config y reinicia pipewire
#   ./aec.sh off     la quita y reinicia pipewire (reversión total)
#   ./aec.sh status  dice si está activo y si el source virtual existe
#   ./aec.sh test    carga el AEC TEMPORALMENTE (sin instalar) y lo descarga
#
# Nada de esto toca el default source del sistema: Crotolamo apunta al source
# limpio con [voice].input_device. Discord, el navegador, etc. siguen igual.
set -euo pipefail

CONF_NAME="crotolamo-aec.conf"
SRC_CONF="$(dirname "$(readlink -f "$0")")/pipewire-aec.conf"
DEST_DIR="$HOME/.config/pipewire/pipewire.conf.d"
DEST="$DEST_DIR/$CONF_NAME"
AEC_SOURCE="crotolamo_aec_source"

have_source() { pactl list short sources 2>/dev/null | grep -q "$AEC_SOURCE"; }

restart_pipewire() {
    echo "  reiniciando pipewire…"
    systemctl --user restart pipewire pipewire-pulse wireplumber 2>/dev/null || true
    sleep 2
}

case "${1:-status}" in
  on)
    [[ -f "$SRC_CONF" ]] || { echo "No encuentro $SRC_CONF"; exit 1; }
    mkdir -p "$DEST_DIR"
    cp "$SRC_CONF" "$DEST"
    echo "config instalada en $DEST"
    restart_pipewire
    if have_source; then
        echo "OK: el source '$AEC_SOURCE' existe."
        echo
        echo "Siguiente paso — que Crotolamo lo use (sin tocar el default del sistema):"
        echo "    [voice]"
        echo "    input_device = \"$AEC_SOURCE\"     # en config/crotolamo.local.toml"
        echo
        echo "Y ANTES de poner use_oww=true, comprueba que el eco murió:"
        echo "    .venv/bin/python scripts/wake_debug.py --threshold 0.3"
    else
        echo "AVISO: el source no apareció. Revisa: journalctl --user -u pipewire -n 50"
        exit 1
    fi
    ;;

  off)
    if [[ -f "$DEST" ]]; then
        rm -f "$DEST"
        echo "config eliminada de $DEST"
        restart_pipewire
    else
        echo "no estaba instalada ($DEST no existe)"
    fi
    have_source && echo "AVISO: el source sigue ahí (¿módulo cargado a mano?)" || echo "OK: revertido."
    echo
    echo "Acuérdate de quitar [voice].input_device de crotolamo.local.toml si lo pusiste."
    ;;

  status)
    if [[ -f "$DEST" ]]; then echo "config: INSTALADA ($DEST)"; else echo "config: no instalada"; fi
    if have_source; then echo "source '$AEC_SOURCE': PRESENTE"; else echo "source '$AEC_SOURCE': ausente"; fi
    echo "default source del sistema: $(pactl get-default-source 2>/dev/null || echo '?')"
    ;;

  test)
    # Prueba temporal, sin instalar nada. Carga el módulo y lo descarga.
    SRC="$(pactl get-default-source)"
    SINK="$(pactl get-default-sink)"
    echo "probando AEC sobre source=$SRC sink=$SINK"
    MOD=$(pactl load-module module-echo-cancel aec_method=webrtc \
            source_master="$SRC" sink_master="$SINK" \
            source_name=crotolamo_aec_test sink_name=crotolamo_aec_test_sink)
    if [[ "$MOD" =~ ^[0-9]+$ ]]; then
        echo "cargado (id=$MOD). Sources virtuales:"
        pactl list short sources | grep -i crotolamo_aec_test || echo "  (ninguno)"
        pactl unload-module "$MOD"
        echo "descargado. Sistema como estaba."
    else
        echo "FALLO: $MOD"; exit 1
    fi
    ;;

  *)
    echo "uso: $0 {on|off|status|test}"; exit 2 ;;
esac
