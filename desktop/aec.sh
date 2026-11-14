#!/usr/bin/env bash
# Activa/desactiva la cancelación de eco (AEC) de PipeWire para Crotolamo.
#
#   ./aec.sh on      instala la config y reinicia pipewire
#   ./aec.sh off     la quita y reinicia pipewire (reversión total)
#   ./aec.sh status  dice si está activo, si el source virtual existe y si su sink es el default
#   ./aec.sh test    carga el AEC TEMPORALMENTE (sin instalar) y lo descarga
#
# Nada de esto toca el default source del sistema: Crotolamo apunta al source
# limpio con [voice].input_device. Discord, el navegador, etc. siguen igual.
#
# La SALIDA sí cuenta: el AEC solo resta lo que suena por crotolamo_aec_sink, y
# el TTS y la música salen por el sink por defecto. Tiene que ser ese (lo mismo
# que exige el doctor); el script no lo cambia, lo comprueba y dice cómo.
set -euo pipefail

CONF_NAME="crotolamo-aec.conf"
SRC_CONF="$(dirname "$(readlink -f "$0")")/pipewire-aec.conf"
DEST_DIR="$HOME/.config/pipewire/pipewire.conf.d"
DEST="$DEST_DIR/$CONF_NAME"
AEC_SOURCE="crotolamo_aec_source"
AEC_SINK="crotolamo_aec_sink"

# La salida de pactl se captura ENTERA antes de buscar: con `pactl | grep -q`
# bajo pipefail, grep sale en el primer match y, si pactl sigue escribiendo,
# muere de SIGPIPE y el source "no existe" aunque esté ahí.
have_source() {
    local out
    out="$(pactl list short sources 2>/dev/null)" || return 1
    [[ "$out" == *"$AEC_SOURCE"* ]]
}

default_sink() { pactl get-default-sink 2>/dev/null || echo '?'; }

# Sin el sink del AEC como default no hay referencia que restar: el source
# "limpio" trae el eco igual y Crotolamo se puede activar con su propia voz.
sink_aviso() {
    echo "AVISO: el AEC solo resta lo que suena por '$AEC_SINK' y el sink por defecto"
    echo "es otro: la voz de Crotolamo y la música no pasan por él (AEC sin referencia)."
    echo "Ponlo por defecto:"
    echo "    pactl set-default-sink $AEC_SINK"
}

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
        echo "Siguiente paso — que Crotolamo lo use (sin tocar el default source del sistema):"
        echo "    [voice]"
        echo "    input_device = \"$AEC_SOURCE\"     # en config/crotolamo.local.toml"
        echo
        if [[ "$(default_sink)" != "$AEC_SINK" ]]; then sink_aviso; echo; fi
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
    if have_source; then ESTADO="PRESENTE"; else ESTADO="ausente"; fi
    echo "source '$AEC_SOURCE': $ESTADO"
    echo "default source del sistema: $(pactl get-default-source 2>/dev/null || echo '?')"
    DEF_SINK="$(default_sink)"
    echo "default sink del sistema: $DEF_SINK"
    # Con el AEC apagado el sink por defecto da igual: nada que avisar.
    if [[ "$ESTADO" == PRESENTE && "$DEF_SINK" != "$AEC_SINK" ]]; then sink_aviso; fi
    ;;

  test)
    # Prueba temporal, sin instalar nada. Carga el módulo y lo descarga.
    SRC="$(pactl get-default-source)"
    SINK="$(pactl get-default-sink)"
    echo "probando AEC sobre source=$SRC sink=$SINK"
    # La asignación va DENTRO del if: suelta, con set -e un pactl que falla
    # abortaba el script ahí mismo y la rama FALLO de abajo era inalcanzable.
    if MOD=$(pactl load-module module-echo-cancel aec_method=webrtc \
            source_master="$SRC" sink_master="$SINK" \
            source_name=crotolamo_aec_test sink_name=crotolamo_aec_test_sink) \
       && [[ "$MOD" =~ ^[0-9]+$ ]]; then
        echo "cargado (id=$MOD). Sources virtuales:"
        pactl list short sources | grep -i crotolamo_aec_test || echo "  (ninguno)"
        pactl unload-module "$MOD"
        echo "descargado. Sistema como estaba."
    else
        # El motivo lo escribe pactl por stderr (justo encima); MOD suele venir vacío.
        echo "FALLO: pactl no cargó module-echo-cancel${MOD:+ ($MOD)}"; exit 1
    fi
    ;;

  *)
    echo "uso: $0 {on|off|status|test}"; exit 2 ;;
esac
