#!/usr/bin/env python3
"""
Crotolamo Runtime Shell FIX.

Arregla el crash:
AttributeError: 'str' object has no attribute 'get'

Causa:
El runtime ahora puede devolver str o dict. El shell viejo asumía dict siempre.
Porque aparentemente pedirle a Python que no sea dramático era demasiado.
"""
from __future__ import annotations

from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _make_runtime():
    from core.crotolamo_runtime import CrotolamoRuntime
    return CrotolamoRuntime()


def _current_mode() -> str:
    try:
        from core.project_modes import get_current_mode
        mode = get_current_mode(ROOT)
        if isinstance(mode, str):
            return mode
        if hasattr(mode, "name"):
            return str(mode.name)
    except Exception:
        pass
    return "crotolamo"


def _render_result(result) -> str:
    """
    Soporta respuestas viejas tipo dict y nuevas tipo str.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        # Formatos posibles de versiones anteriores
        if "text" in result:
            return str(result.get("text", ""))
        if "response" in result:
            return str(result.get("response", ""))
        if "message" in result:
            return str(result.get("message", ""))
        if result.get("kind") == "direct":
            return str(result.get("output") or result.get("result") or result)
        if "output" in result:
            return str(result.get("output", ""))
        if "result" in result:
            return str(result.get("result", ""))
        return str(result)

    return str(result)


def main() -> int:
    try:
        runtime = _make_runtime()
    except Exception as e:
        print("No pude iniciar CrotolamoRuntime:")
        print(f"{type(e).__name__}: {e}")
        return 1

    print("Crotolamo Runtime Shell FIX. Escribe 'salir' para terminar.")
    print("Comandos útiles: inteligencia, memoria, contexto, proyectos, executor, test crotolamo")
    print()

    while True:
        try:
            prompt = input(f"{_current_mode()} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue

        if prompt.lower() in {"salir", "exit", "quit", "q"}:
            return 0

        try:
            result = runtime.process_text(prompt)
            rendered = _render_result(result)
            if rendered:
                print(rendered)
        except KeyboardInterrupt:
            print("Operación cancelada.")
        except Exception:
            print("Error en runtime shell:")
            traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
