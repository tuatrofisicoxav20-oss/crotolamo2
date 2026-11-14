#!/usr/bin/env bash
# Saca la API key de Spotify Soloist de la línea de comando y del .service.
#
# Hoy: ExecStart=... --api-key spak_...  -> se ve con `ps`/`pgrep -fa` y queda en
# texto plano dentro de ~/.config/systemd/user/soloist.service.
#
# Este script (correrlo en la LAPTOP, no en CI):
#   1. Lee `soloist --help` y busca si el binario acepta la key por VARIABLE DE
#      ENTORNO o por ARCHIVO. No asume nada: decide con la ayuda real e imprime
#      las líneas de la ayuda en las que se basó.
#   2. Respalda el .service (soloist.service.bak-FECHA).
#   3. Deja la key SOLO en ~/.config/soloist/env (chmod 600) con un placeholder
#      para pegar la key NUEVA a mano (la vieja se considera comprometida).
#   4. Reescribe el .service:
#        modo env     : EnvironmentFile= y ExecStart SIN --api-key (no sale en ps)
#        modo file    : EnvironmentFile= y --api-key-file (no sale en ps)
#        modo wrapper : EnvironmentFile= y `sh -c` que pasa "$VAR" -> la key ya
#                       no está en el .service ni en el repo, pero SÍ SIGUE
#                       VISIBLE EN ps mientras Soloist no soporte otra forma.
#   5. Conserva TODOS los demás flags (--device-name crotolamo, --ws
#      127.0.0.1:9090, ...).
#
# NUNCA imprime, copia ni loguea el valor de la key: todo lo que muestra pasa
# por un enmascarado. Uso:
#   desktop/soloist_key_to_env.sh            # aplica
#   desktop/soloist_key_to_env.sh --dry-run  # solo muestra qué haría
#   desktop/soloist_key_to_env.sh --mode env|file|wrapper   # forzar el modo
set -euo pipefail

SERVICE="${SOLOIST_SERVICE:-$HOME/.config/systemd/user/soloist.service}"
ENV_FILE="${SOLOIST_ENV_FILE:-$HOME/.config/soloist/env}"
BIN="${SOLOIST_BIN:-$HOME/.local/bin/soloist}"
PLACEHOLDER="PEGA_AQUI_LA_KEY_NUEVA"
DRY_RUN=0
MODE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --mode=*) MODE="${arg#--mode=}" ;;
        --mode) shift_mode=1 ;;
        env|file|wrapper) if [[ "${shift_mode:-0}" == 1 ]]; then MODE="$arg"; shift_mode=0; fi ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "argumento desconocido: $arg" >&2; exit 2 ;;
    esac
done

# Enmascara cualquier cosa que parezca una key en lo que se imprime. El valor de
# --api-key se tapa también entre comillas (una key sin el prefijo spak_ salía en
# claro); "$VAR" no, que es la referencia del wrapper, no la key.
mask() { sed -E 's/spak_[A-Za-z0-9_-]+/spak_<oculto>/g; s/(--api-key(=|[[:space:]]+))("[^"$][^"]*"|'"'"'[^'"'"'$][^'"'"']*'"'"'|[^[:space:]"'"'"']+)/\1<oculto>/g; s/(API_KEY[A-Z_]*=)[^[:space:]]*/\1<oculto>/g'; }

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$SERVICE" ]] || die "no encuentro el servicio: $SERVICE"
[[ -x "$BIN" ]] || BIN="$(command -v soloist || true)"
[[ -n "$BIN" && -x "$BIN" ]] || die "no encuentro el binario de soloist (ni en ~/.local/bin ni en PATH)"

echo "== Servicio : $SERVICE"
echo "== Binario  : $BIN"
echo "== Env file : $ENV_FILE"
echo

# ---------------------------------------------------------------------------
# 1) ¿Qué soporta el binario? (ayuda real)
# ---------------------------------------------------------------------------
HELP="$("$BIN" --help 2>&1 || true)"
if [[ -z "$HELP" ]]; then
    HELP="$("$BIN" -h 2>&1 || true)"
fi
echo "== Líneas de 'soloist --help' relacionadas con la key:"
RELEVANT="$(printf '%s\n' "$HELP" | grep -i -E 'api[-_ ]?key|env|environment|key[-_ ]?file|from[-_ ]file|stdin' || true)"
# | mask: clap enseña `[env: X=valor]` si la variable está exportada en esta shell.
if [[ -n "$RELEVANT" ]]; then printf '%s\n' "$RELEVANT" | mask | sed 's/^/   | /'; else echo "   | (la ayuda no menciona api-key, env ni archivo)"; fi
echo

ENV_VAR=""
if [[ -z "$MODE" ]]; then
    # Variable de entorno: un nombre tipo SOLOIST_API_KEY / SPOTIFY_API_KEY en la ayuda.
    ENV_VAR="$(printf '%s\n' "$HELP" | grep -o -E '\b[A-Z][A-Z0-9_]*API_KEY\b' | head -n1 || true)"
    if [[ -n "$ENV_VAR" ]]; then
        MODE="env"
    elif printf '%s\n' "$HELP" | grep -q -i -E -- '--api-key-file|--key-file|api[-_]key[-_]file'; then
        MODE="file"
    else
        MODE="wrapper"
    fi
fi
[[ -z "$ENV_VAR" ]] && ENV_VAR="SOLOIST_API_KEY"
KEY_FILE="$HOME/.config/soloist/api_key"
# El flag de archivo tal como lo anuncia la ayuda: escribir --api-key-file a un
# binario que solo conoce --key-file deja el servicio sin arrancar.
FILE_FLAG="$(printf '%s\n' "$HELP" | grep -o -E -- '--(api[-_]key[-_]file|key[-_]file)' | head -n1 || true)"
[[ -z "$FILE_FLAG" ]] && FILE_FLAG="--api-key-file"

case "$MODE" in
    env)
        echo "== Resultado: Soloist SÍ acepta la key por la variable $ENV_VAR (según su ayuda)."
        echo "   ExecStart irá SIN --api-key: la key no aparecerá en ps." ;;
    file)
        echo "== Resultado: Soloist SÍ acepta la key desde un ARCHIVO ($FILE_FLAG)."
        echo "   La key vivirá en $KEY_FILE (600); ExecStart irá sin --api-key: no aparecerá en ps." ;;
    wrapper)
        echo "== Resultado: la ayuda NO muestra forma de pasar la key por entorno ni archivo."
        echo "   Se usa un wrapper sh -c que la lee de $ENV_VAR: la key deja de estar en el"
        echo "   .service y en el repo, pero SEGUIRÁ VIÉNDOSE EN ps mientras Soloist no soporte"
        echo "   otra forma. Esto NO es una solución completa." ;;
    *) die "modo desconocido: $MODE (env|file|wrapper)" ;;
esac
echo

# ---------------------------------------------------------------------------
# 2) ExecStart actual -> sin --api-key (y sin líneas Environment= con la key)
# ---------------------------------------------------------------------------
EXEC_LINE="$(grep -E '^ExecStart=' "$SERVICE" | head -n1 || true)"
[[ -n "$EXEC_LINE" ]] || die "el .service no tiene ExecStart="
EXEC_CMD="${EXEC_LINE#ExecStart=}"
EXEC_ACTUAL="$EXEC_CMD"
# ¿Es el wrapper `sh -c` de una corrida anterior? Se deshace antes de seguir:
# envolverlo otra vez metía un `sh -c '…'` dentro de otro (comillas rotas: el unit
# no arrancaba) y con --mode env|file la key seguiría yendo por la línea de comando.
# En bucle: así también se arregla el doble wrapper que ya dejó esa 2ª corrida.
WRAPPER_RE='^/bin/sh -c '\''exec "([^"]+)"(.*)'\''$'
DESENVUELTO=0
while [[ "$EXEC_CMD" =~ $WRAPPER_RE ]]; do
    EXEC_CMD="$(printf '%s' "${BASH_REMATCH[1]}${BASH_REMATCH[2]}" | sed -E 's/[[:space:]]+--api-key[[:space:]]+"\$[A-Za-z_][A-Za-z0-9_]*"//')"
    DESENVUELTO=1
done
if [[ "$DESENVUELTO" == 1 ]]; then
    echo "AVISO: ExecStart ya es el wrapper sh -c de una corrida anterior: lo deshago y rehago."
fi
# La key que lleva hoy el unit (comprometida): nunca se conserva en el env ni en el
# archivo de la key. Con el wrapper es "$VAR", que no coincide con ninguna key real.
OLD_KEY="$(printf '%s' "$EXEC_CMD" | sed -n -E 's/.*[[:space:]]--api-key(=|[[:space:]]+)("([^"]*)"|'"'"'([^'"'"']*)'"'"'|([^[:space:]"'"'"']+)).*/\3\4\5/p')"
# Quita --api-key <valor> y --api-key=<valor> (con o sin comillas), nada más.
EXEC_SIN_KEY="$(printf '%s' "$EXEC_CMD" | sed -E 's/[[:space:]]+--api-key(=|[[:space:]]+)("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:]]+)//g')"
if [[ "$EXEC_SIN_KEY" == "$EXEC_CMD" && "$DESENVUELTO" == 0 ]]; then
    echo "AVISO: ExecStart no llevaba --api-key (¿ya migrado?). Sigo igual, sin tocar los flags."
fi
# Y el flag de archivo con NUESTRA ruta que puso una corrida anterior en modo file:
# se volvía a añadir detrás (flag duplicado) y otro modo lo heredaba.
EXEC_SIN_KEY="$(printf '%s' "$EXEC_SIN_KEY" | sed -E ':a; s#[[:space:]]+--[A-Za-z_-]+(=|[[:space:]]+)%h/\.config/soloist/api_key([[:space:]]|$)#\2#; ta')"
echo "== ExecStart actual (enmascarado):"
printf '   %s\n' "$EXEC_ACTUAL" | mask
echo "== Flags que se conservan:"
printf '   %s\n' "$EXEC_SIN_KEY" | mask
echo

# Comprobación de que los flags críticos siguen ahí.
for flag in "--device-name crotolamo" "--ws 127.0.0.1:9090"; do
    if printf '%s' "$EXEC_CMD" | grep -q -F -- "$flag" && ! printf '%s' "$EXEC_SIN_KEY" | grep -q -F -- "$flag"; then
        die "perdí el flag '$flag' al quitar la key; no toco nada"
    fi
done

case "$MODE" in
    env)     NEW_EXEC="ExecStart=$EXEC_SIN_KEY" ;;
    file)    NEW_EXEC="ExecStart=$EXEC_SIN_KEY $FILE_FLAG %h/.config/soloist/api_key" ;;
    wrapper)
        # Comillas simples: sh expande $VAR desde el entorno que cargó systemd.
        BIN_PART="${EXEC_SIN_KEY%% *}"
        REST_PART="${EXEC_SIN_KEY#"$BIN_PART"}"
        NEW_EXEC="ExecStart=/bin/sh -c 'exec \"$BIN_PART\" --api-key \"\$$ENV_VAR\"$REST_PART'" ;;
esac

# Nuevo unit: mismas líneas, salvo ExecStart, EnvironmentFile y Environment= con key.
NEW_UNIT="$(awk -v new_exec="$NEW_EXEC" -v envfile="EnvironmentFile=-%h/.config/soloist/env" '
    /^ExecStart=/ && !done { print envfile; print new_exec; done=1; next }
    /^ExecStart=/ { next }
    /^EnvironmentFile=.*soloist\/env/ { next }
    /^Environment=.*API_KEY/ { next }
    { print }
' "$SERVICE")"

echo "== Nuevo $SERVICE (enmascarado):"
printf '%s\n' "$NEW_UNIT" | mask | sed 's/^/   /'
echo

# ---------------------------------------------------------------------------
# 3) Archivo de entorno con placeholder (la key vieja NO se conserva)
# ---------------------------------------------------------------------------
# Una key que ya está en el env o en el archivo de la key y no es ni el placeholder
# ni la vieja del unit es la NUEVA que pegó el patrón tras una corrida anterior: se
# conserva. Sustituirla por el placeholder la perdía (solo se respalda el .service).
es_key_nueva() { [[ -n "$1" && "$1" != "$PLACEHOLDER" && "$1" != "$OLD_KEY" ]]; }
CONSERVA_ENV=0
CONSERVA_KEY_FILE=0
if [[ -f "$ENV_FILE" ]] && es_key_nueva "$(grep -E "^(export[[:space:]]+)?$ENV_VAR=" "$ENV_FILE" | tail -n1 | cut -d= -f2- | tr -d "\"' \t\r" || true)"; then
    CONSERVA_ENV=1
fi
if [[ "$MODE" == file && -f "$KEY_FILE" ]] && es_key_nueva "$(tr -d "\"' \t\r\n" < "$KEY_FILE")"; then
    CONSERVA_KEY_FILE=1
fi
echo "== $ENV_FILE quedará así (enmascarado):"
NEW_ENV=""
if [[ "$CONSERVA_ENV" == 1 ]]; then
    NEW_ENV="$(cat "$ENV_FILE")"
elif [[ -f "$ENV_FILE" ]]; then
    # Conserva las demás variables; la de la key se sustituye por el placeholder.
    NEW_ENV="$(grep -v -E "^(export[[:space:]]+)?$ENV_VAR=" "$ENV_FILE" || true)"
fi
[[ "$CONSERVA_ENV" == 1 ]] || NEW_ENV="$(printf '%s\n%s=%s\n' "$NEW_ENV" "$ENV_VAR" "$PLACEHOLDER" | sed '/^$/d')"
printf '%s\n' "$NEW_ENV" | sed -E 's/=.*/=<oculto>/' | mask | sed 's/^/   /'
if [[ "$CONSERVA_ENV" == 1 ]]; then
    echo "   (la línea $ENV_VAR= ya tiene una key nueva: la conservo, el archivo no se toca)"
else
    echo "   (la línea $ENV_VAR= lleva el placeholder $PLACEHOLDER; pega ahí la key nueva)"
fi
[[ "$CONSERVA_KEY_FILE" == 1 ]] && echo "   ($KEY_FILE ya tiene una key nueva: la conservo)"
echo

if [[ "$DRY_RUN" == 1 ]]; then
    echo "== --dry-run: no escribí nada."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4) Escribir: respaldo, unit, env (600)
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$SERVICE.bak-$STAMP"
cp -p "$SERVICE" "$BACKUP"
chmod 600 "$BACKUP"
echo "== Respaldo: $BACKUP (contiene la key vieja: bórralo con 'shred -u' cuando compruebes que todo va)"

mkdir -p "$(dirname "$ENV_FILE")"
chmod 700 "$(dirname "$ENV_FILE")"
umask 077
[[ "$CONSERVA_ENV" == 1 ]] || printf '%s\n' "$NEW_ENV" > "$ENV_FILE"
chmod 600 "$ENV_FILE"
if [[ "$MODE" == "file" && "$CONSERVA_KEY_FILE" != 1 ]]; then
    printf '%s\n' "$PLACEHOLDER" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "== $KEY_FILE creado con el placeholder (chmod 600): pega ahí SOLO la key, sin nombre de variable."
elif [[ "$MODE" == "file" ]]; then
    chmod 600 "$KEY_FILE"
fi

printf '%s\n' "$NEW_UNIT" > "$SERVICE"
chmod 644 "$SERVICE"

# Verificación: la key ya no está en el unit nuevo.
if grep -q -E 'spak_[A-Za-z0-9_-]{6,}' "$SERVICE"; then
    die "el unit nuevo TODAVÍA contiene algo que parece una key; revisa a mano (no la muestro)"
fi
echo "== Escrito: $SERVICE y $ENV_FILE (600)."
echo
PEGAR=""
[[ "$CONSERVA_ENV" == 1 ]] || PEGAR="$ENV_FILE"
if [[ "$MODE" == file && "$CONSERVA_KEY_FILE" != 1 ]]; then PEGAR="${PEGAR:+$PEGAR y en }$KEY_FILE"; fi
if [[ -n "$PEGAR" ]]; then
    PASO1="Pega la key NUEVA en $PEGAR (sustituye $PLACEHOLDER)."
else
    PASO1="La key nueva ya está puesta (no la toqué)."
fi
cat <<EOF
== Siguientes pasos (a mano):
   1. $PASO1
   2. systemctl --user daemon-reload
   3. systemctl --user restart soloist.service
   4. journalctl --user -u soloist.service -n 50 --no-pager -f
   5. pgrep -fa soloist | sed -E 's/spak_[A-Za-z0-9_-]+/spak_<oculto>/g'
      (modo env/file: no debe aparecer --api-key; modo wrapper: seguirá apareciendo)
   6. Cuando todo funcione: shred -u "$BACKUP"
   Crotolamo no cambia: sigue hablando con Soloist por ws://127.0.0.1:9090 (device crotolamo).
EOF
