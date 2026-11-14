"""Punto de entrada: python -m crotolamo [comando].

Comandos:
  --version        imprime la versión
  shell            REPL de texto (Fase 1+)
  doctor           auditor de salud (Fase 0)
  listen           bucle de voz wake-word (Fase 5)
  memoria          memoria semántica: migrar|listar|buscar|olvidar|calibrar
"""

from __future__ import annotations

import sys
from typing import Callable

from crotolamo import __version__


def _cargar_doctor() -> Callable[[], int] | None:
    """El `run_doctor` de scripts/crotolamo_doctor.py, o None si el script no está.

    `scripts` no se empaqueta (pyproject solo incluye crotolamo* e interfaces*):
    solo es importable con `python -m crotolamo` DESDE la raíz del repo. Con el
    comando `crotolamo` (el console-script) o desde otra carpeta, `import scripts`
    reventaba con un traceback; ahí el doctor se carga por su ruta en el repo
    (REPO_ROOT, la misma ancla con la que settings encuentra config/), que no
    depende de dónde estés. Mismo criterio que `memoria calibrar`.
    """
    try:
        from scripts.crotolamo_doctor import run_doctor
    except ImportError:
        # Si el fallo era un import roto DENTRO del doctor, la carga por ruta de
        # abajo lo vuelve a lanzar tal cual: no se disfraza de "no está".
        pass
    else:
        return run_doctor
    import importlib.util

    from crotolamo.settings import REPO_ROOT

    ruta = REPO_ROOT / "scripts" / "crotolamo_doctor.py"
    if not ruta.is_file():
        return None
    nombre = "crotolamo_doctor"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        return None
    modulo = importlib.util.module_from_spec(spec)
    # Registrado ANTES de ejecutarlo: el doctor define dataclasses con
    # anotaciones diferidas, y `dataclasses` las resuelve mirando
    # sys.modules[cls.__module__] (sin esto: AttributeError sobre None).
    sys.modules[nombre] = modulo
    try:
        spec.loader.exec_module(modulo)
    except BaseException:
        sys.modules.pop(nombre, None)  # nada a medio cargar para el siguiente import
        raise
    doctor: Callable[[], int] = modulo.run_doctor
    return doctor


def main(argv: list[str] | None = None) -> int:
    from crotolamo.logging_setup import setup_logging

    setup_logging()
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"--version", "-v", "version"}:
        print(f"Crotolamo {__version__}")
        return 0

    command, rest = argv[0], argv[1:]

    if command == "shell":
        from interfaces.shell import run_shell
        return run_shell(rest)

    if command == "doctor":
        run_doctor = _cargar_doctor()
        if run_doctor is None:
            print("No encuentro scripts/crotolamo_doctor.py junto al paquete: el doctor "
                  "necesita el checkout completo del repo (pip install -e .).", file=sys.stderr)
            return 1
        return run_doctor()

    if command == "listen":
        from interfaces.listener import run_listen
        return run_listen(rest)

    if command == "memoria":
        from crotolamo.core.memoria import run_cli
        return run_cli(rest)

    print(f"Comando desconocido: {command!r}", file=sys.stderr)
    print("Usa: python -m crotolamo [--version|shell|doctor|listen|memoria]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
