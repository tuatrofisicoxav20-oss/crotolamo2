"""Verificación por fase. `python scripts/verify_phase.py N`.

Cada fase corre sus checks. Sale 0 si todo pasa, !=0 si algo falla.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT).returncode


def _pytest(paths: list[str]) -> int:
    return _run([sys.executable, "-m", "pytest", "-q", *paths])


def verify(phase: int) -> int:
    py = sys.executable
    failures = 0

    if phase >= 0:
        print("== Fase 0: andamiaje ==")
        failures += _run([py, "-m", "crotolamo", "--version"]) != 0
        # El doctor puede salir 1 si Ollama está apagado; no lo contamos como
        # fallo duro de la Fase 0 (el andamiaje en sí no depende de Ollama vivo).
        _run([py, "scripts/crotolamo_doctor.py"])

    if phase >= 1:
        print("== Fase 1: núcleo conversacional ==")
        failures += _pytest(["tests/test_memory.py"]) != 0

    if phase >= 2:
        print("== Fase 2: tool-calling ==")
        failures += _pytest(
            ["tests/test_tools.py", "tests/test_safety.py", "tests/test_wake.py",
             "tests/test_agent.py"]
        ) != 0

    if phase >= 3:
        print("== Fase 3: tools que hacen cosas ==")
        failures += _pytest(
            ["tests/test_files.py", "tests/test_projects.py",
             "tests/test_system.py", "tests/test_shortcuts.py"]
        ) != 0

    if phase >= 4:
        print("== Fase 4: memoria persistente ==")
        failures += _pytest(["tests/test_facts.py"]) != 0

    if phase >= 5:
        print("== Fase 5: voz ==")
        failures += _pytest(["tests/test_voice.py"]) != 0

    if phase >= 6:
        print("== Fase 6: extensiones ==")
        failures += _pytest(["tests/test_fase6.py"]) != 0

    print("\n" + "=" * 40)
    if failures:
        print(f"Fase {phase}: {failures} bloque(s) en rojo, patrón.")
        return 1
    print(f"Fase {phase}: verde, patrón. ✅")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python scripts/verify_phase.py N", file=sys.stderr)
        return 2
    try:
        phase = int(sys.argv[1])
    except ValueError:
        print(f"Fase inválida: {sys.argv[1]!r}", file=sys.stderr)
        return 2
    return verify(phase)


if __name__ == "__main__":
    raise SystemExit(main())
