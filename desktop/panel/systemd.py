"""Interacción con systemd (systemctl --user) — sin gi, testeable headless.

Nada aquí toca la UI: query_state() se llama desde un hilo de trabajo del
panel para no bloquear el hilo GTK con subprocess síncronos.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable

log = logging.getLogger("crotolamo.desktop.panel.systemd")

SERVICE = "crotolamo.service"


def run(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess:
    """systemctl/journalctl sin reventar la UI: devuelve siempre un resultado."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Fallo ejecutando %s: %s", args, exc)
        return subprocess.CompletedProcess(args, returncode=255, stdout="", stderr=str(exc))


def sc(*args: str) -> subprocess.CompletedProcess:
    return run(["systemctl", "--user", *args])


def query_state(runner: Callable[..., subprocess.CompletedProcess] | None = None
                ) -> dict[str, str]:
    """Una sola llamada barata para el polling: estado + si arranca con la sesión.

    `runner` permite inyectar un doble de `sc` en los tests (sin systemctl real).
    """
    if runner is None:
        runner = sc
    res = runner("show", SERVICE,
                 "-p", "ActiveState", "-p", "SubState", "-p", "UnitFileState",
                 "-p", "ActiveEnterTimestampMonotonic")
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            out[key] = val
    return out


def _do_async(*args: str) -> None:
    """Lanza systemctl sin congelar la UI; el polling refleja el resultado."""
    try:
        subprocess.Popen(["systemctl", "--user", *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo lanzar systemctl --user %s: %s", " ".join(args), exc)


def _uptime_text(st: dict[str, str], now_us: int | None = None) -> str:
    """Texto humano de uptime a partir de ActiveEnterTimestampMonotonic.

    `now_us` (microsegundos monotónicos) es inyectable para los tests; por
    defecto usa el reloj monotónico del sistema (mismo que usa systemd).
    """
    try:
        if now_us is None:
            now_us = time.monotonic_ns() // 1_000  # microsegundos
        started = int(st.get("ActiveEnterTimestampMonotonic", "0"))
        if started <= 0:
            return "activo"
        secs = max(0, (now_us - started) // 1_000_000)
        if secs < 60:
            return f"activo desde hace {secs}s"
        mins = secs // 60
        if mins < 60:
            return f"activo desde hace {mins} min"
        return f"activo desde hace {mins // 60} h {mins % 60} min"
    except Exception as exc:  # noqa: BLE001
        log.debug("Uptime ilegible en %r: %s", st, exc)
        return "activo"
