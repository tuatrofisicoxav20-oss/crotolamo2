"""Tools de estado del sistema y hardware. Migrado/ampliado de C1::system_status.

Todo read-only y sin bash generado por el LLM: cada tool corre lecturas fijas
(shutil, /proc, ps). El modelo solo decide CUÁL tool llamar.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from crotolamo.tools.base import tool

HOME = Path.home()


def _human(num_bytes: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}P"


@tool
def disk_usage() -> str:
    """Muestra el uso de disco de la partición del home."""
    usage = shutil.disk_usage(HOME)
    pct = usage.used / usage.total * 100 if usage.total else 0
    return (
        f"Disco ({HOME}): {_human(usage.used)} usados de {_human(usage.total)} "
        f"({pct:.0f}%), {_human(usage.free)} libres, patrón."
    )


@tool
def ram_usage() -> str:
    """Muestra el uso de memoria RAM."""
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            kb = rest.strip().split()[0]
            meminfo[key] = int(kb) * 1024
    except (OSError, ValueError, IndexError):
        return "No pude leer /proc/meminfo, patrón."

    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    used = total - available
    pct = used / total * 100 if total else 0
    return f"RAM: {_human(used)} usados de {_human(total)} ({pct:.0f}%), patrón."


@tool
def list_processes(limit: int = 8) -> str:
    """Lista los procesos que más memoria consumen (read-only).

    Args:
        limit: cuántos procesos mostrar.
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,comm,%mem,%cpu", "--sort=-%mem"],
            text=True, capture_output=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "No pude listar los procesos, patrón."

    lines = result.stdout.strip().splitlines()
    if not lines:
        return "No obtuve procesos, patrón."
    header, rows = lines[0], lines[1 : max(1, limit) + 1]
    return "Procesos que más comen, patrón:\n" + "\n".join([header, *rows])


@tool
def system_status() -> str:
    """Resumen del estado del sistema: disco, RAM y carga."""
    parts = [disk_usage(), ram_usage()]
    try:
        uptime = Path("/proc/uptime").read_text().split()[0]
        secs = float(uptime)
        hours, rem = divmod(int(secs), 3600)
        mins = rem // 60
        parts.append(f"Encendida hace {hours}h {mins}m, patrón.")
    except (OSError, ValueError, IndexError):
        pass
    return "Estado del sistema, patrón:\n" + "\n".join(parts)
