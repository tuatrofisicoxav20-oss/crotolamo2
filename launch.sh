#!/usr/bin/env bash
# Launcher de Crotolamo 2. Menú para probar el asistente sin recordar comandos.
# Uso:  ./launch.sh            (menú interactivo)
#       ./launch.sh doctor|shell|listen|smoke|version
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- elegir intérprete: venv si existe, si no python3 del sistema ---
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="python3"
    echo "⚠  No hay .venv; uso python3 del sistema."
    echo "   Para un entorno aislado: python3 -m venv .venv && .venv/bin/pip install -e '.[dev,voice]'"
    echo
fi

run() { echo "→ $PY -m crotolamo $*"; echo; "$PY" -m crotolamo "$@"; }

case "${1:-}" in
    version) run --version ;;
    doctor)  run doctor ;;
    shell)   run shell ;;
    listen)  shift; run listen "$@" ;;
    smoke)   echo "→ smoke de voz (TTS↔STT, sin micrófono)"; echo; "$PY" scripts/smoke_voz.py ;;
    "")
        while true; do
            echo "=========================================="
            echo "  Crotolamo 2  —  ¿qué probamos, patrón?"
            echo "=========================================="
            echo "  1) Doctor (chequeo de salud)"
            echo "  2) Shell de texto (conversar + tools)"
            echo "  3) Voz — half-duplex (seguro)"
            echo "  4) Voz — barge-in (con auriculares)"
            echo "  5) Voz — modo simple (fallback)"
            echo "  6) Smoke de voz (TTS↔STT, sin micrófono)"
            echo "  7) Versión"
            echo "  0) Salir"
            echo
            read -rp "Opción > " opt
            echo
            case "$opt" in
                1) run doctor ;;
                2) run shell ;;
                3) run listen --no-barge-in ;;
                4) run listen --barge-in ;;
                5) run listen --simple ;;
                6) "$PY" scripts/smoke_voz.py ;;
                7) run --version ;;
                0|q|salir) echo "Hasta luego, patrón."; exit 0 ;;
                *) echo "Opción no válida." ;;
            esac
            echo; read -rp "(Enter para volver al menú) " _ || true; echo
        done
        ;;
    *) echo "Uso: ./launch.sh [doctor|shell|listen|smoke|version]"; exit 2 ;;
esac
