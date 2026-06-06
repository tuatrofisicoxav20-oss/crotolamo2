#!/usr/bin/env python3
"""Diagnóstico rápido para Crotolamo Orbital UI.
No ejecuta comandos peligrosos. Solo importa módulos y revisa servicios básicos.
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_import(module: str, names: list[str] | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        if names:
            missing = [name for name in names if not callable(getattr(mod, name, None))]
            if missing:
                return False, f"faltan funciones: {', '.join(missing)}"
        return True, "OK"
    except Exception as error:
        return False, str(error)


def check_ollama() -> tuple[bool, str]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
        models = [m.get("name", "sin_nombre") for m in data.get("models", [])[:5]]
        return True, "OK" + (f" | modelos: {', '.join(models)}" if models else " | sin modelos listados")
    except Exception as error:
        return False, str(error)


def main() -> int:
    print("CROTOLAMO ORBITAL :: CHECK DE CONEXIONES")
    print("=" * 48)
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Sistema:  {platform.system()} {platform.release()}")
    print()

    checks = [
        ("core.chapi_shell", ["ask_ollama", "normalize_plan"]),
        ("core.skills", ["handle_direct_skill"]),
        ("core.voice_in", ["listen_once"]),
        ("core.voice_out", ["speak"]),
        ("ui.crotolamo_orbital_ui", ["main"]),
    ]

    ok_all = True
    for module, funcs in checks:
        ok, msg = check_import(module, funcs)
        ok_all = ok_all and ok
        print(f"[{ 'OK' if ok else 'NO' }] {module:<26} {msg}")

    ok, msg = check_ollama()
    ok_all = ok_all and ok
    print(f"[{ 'OK' if ok else 'NO' }] {'ollama api':<26} {msg}")
    print()

    if ok_all:
        print("Resultado: conectado. El monstruo vive, qué conveniente.")
        return 0

    print("Resultado: algo falta. Lee las líneas con [NO], porque ahí está el drama.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
