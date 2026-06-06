#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.crotolamo_runtime import CrotolamoRuntime


def main() -> None:
    runtime = CrotolamoRuntime(ROOT)
    print("Crotolamo Runtime Shell v3. Escribe 'salir' para terminar. 'runtime' para diagnóstico.\n")
    while True:
        try:
            prompt = input("Patrón > ").strip()
        except KeyboardInterrupt:
            print("\nApagado, patrón.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"salir", "exit", "quit"}:
            print("Apagado, patrón.")
            break
        result = runtime.process_text(prompt)
        if result.get("kind") == "direct":
            print("\n" + str(result.get("text", "")))
            continue
        print("\nPlan:")
        print(result.get("explanation") or result.get("text") or "Sin explicación.")
        commands = result.get("commands") or []
        if commands:
            print("\nComandos:")
            for cmd in commands:
                print("  " + cmd)
            print(f"\nRiesgo: {result.get('risk')} | safe={result.get('safe')}")
            if result.get("safe") and input("¿Ejecutar? [y/N] ").strip().lower() == "y":
                for event in runtime.execute_commands(commands):
                    print(f"{event['label']}> {event['text']}")
            else:
                print("Cancelado, patrón.")
        else:
            print("Sin comandos.")


if __name__ == "__main__":
    main()
