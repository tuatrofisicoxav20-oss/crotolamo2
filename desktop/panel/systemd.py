"""Interacción con systemd (systemctl --user) — sin gi, testeable headless.

Nada aquí toca la UI: query_state() se llama desde un hilo de trabajo del
panel para no bloquear el hilo GTK con subprocess síncronos.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger("crotolamo.desktop.panel.systemd")

SERVICE = "crotolamo.service"

# Clave que query_state() añade cuando NO pudo preguntar a systemd (sin bus de
# usuario, timeout, systemctl ausente). Sin ella el panel pintaba «Apagado».
QUERY_ERROR = "QueryError"

# Verbos que encolan un job: con --no-block systemctl solo valida y encola (unit
# inexistente, enmascarada, sin bus: eso sí falla al momento) y vuelve; cómo
# acaba el job lo refleja el polling, como antes. Esperarlo haría que un stop
# lento (TimeoutStopSec=10) pasara el timeout de run() y saliera como fallo.
_JOB_VERBS = frozenset({"start", "stop", "restart"})


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


def _first_line(text: str | None) -> str:
    """Primera línea no vacía (sin espacios) de la salida de un comando."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def query_state(runner: Callable[..., subprocess.CompletedProcess] | None = None
                ) -> dict[str, str]:
    """Una sola llamada barata para el polling: estado + si arranca con la sesión.

    LoadState distingue una unit que NO EXISTE ("not-found": panel abierto antes
    de install.sh) de una apagada; las dos salen ActiveState=inactive. Si
    systemctl falla, el resultado lleva QUERY_ERROR con el motivo.

    `runner` permite inyectar un doble de `sc` en los tests (sin systemctl real).
    """
    if runner is None:
        runner = sc
    res = runner("show", SERVICE,
                 "-p", "LoadState", "-p", "ActiveState", "-p", "SubState",
                 "-p", "UnitFileState", "-p", "ActiveEnterTimestampMonotonic")
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            out[key] = val
    if res.returncode != 0:
        out[QUERY_ERROR] = (_first_line(res.stderr)
                            or f"systemctl salió con código {res.returncode}")
    return out


def _do_async(*args: str,
              on_done: Callable[[subprocess.CompletedProcess], object] | None = None
              ) -> None:
    """systemctl --user en un hilo: no congela la UI y su resultado no se pierde.

    Antes era un Popen suelto con la salida a /dev/null: si fallaba (unit no
    instalada, enmascarada, sin bus de usuario) nadie se enteraba y el botón
    simplemente no hacía nada. `on_done(resultado)` se llama DESDE EL HILO de
    trabajo: quien toque GTK ahí debe saltar al hilo GTK (GLib.idle_add).
    """
    cmd = ["--no-block", *args] if args and args[0] in _JOB_VERBS else list(args)

    def worker() -> None:
        res = sc(*cmd)
        if res.returncode != 0:
            log.warning("systemctl --user %s falló (%s): %s", " ".join(args),
                        res.returncode, _first_line(res.stderr))
        if on_done is not None:
            on_done(res)

    threading.Thread(target=worker, daemon=True, name="panel-systemctl").start()


def spawn_reaped(args: list[str], **kwargs: Any) -> subprocess.Popen:
    """Popen de un proceso que vive por su cuenta + un hilo que lo espera.

    Nadie más hace wait() de lo que el panel lanza suelto (la terminal con el
    `journalctl -f`): al cerrarlo quedaba <defunct> hasta que otro subprocess del
    panel disparaba la limpieza interna de subprocess, y con la ventana oculta
    no hay tick que lo haga. El OSError de Popen (terminal ausente) se propaga
    tal cual, antes de arrancar ningún hilo.
    """
    proc = subprocess.Popen(args, **kwargs)
    threading.Thread(target=proc.wait, daemon=True, name="panel-reap").start()
    return proc


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
