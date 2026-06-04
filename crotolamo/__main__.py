"""Punto de entrada: python -m crotolamo [comando].

Comandos:
  --version        imprime la versión
  shell            REPL de texto (Fase 1+)
  doctor           auditor de salud (Fase 0)
  listen           bucle de voz wake-word (Fase 5, stub por ahora)
"""

from __future__ import annotations

import sys

from crotolamo import __version__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"--version", "-v", "version"}:
        print(f"Crotolamo {__version__}")
        return 0

    command, rest = argv[0], argv[1:]

    if command == "shell":
        from interfaces.shell import run_shell
        return run_shell(rest)

    if command == "doctor":
        from scripts.crotolamo_doctor import run_doctor
        return run_doctor()

    if command == "listen":
        print("La voz llega en la Fase 5, patrón. Por ahora usa: python -m crotolamo shell")
        return 0

    print(f"Comando desconocido: {command!r}", file=sys.stderr)
    print("Usa: python -m crotolamo [--version|shell|doctor|listen]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
