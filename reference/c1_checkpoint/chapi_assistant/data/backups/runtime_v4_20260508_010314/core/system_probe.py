"""Diagnóstico liviano para Crotolamo sin depender de psutil."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 3) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        output = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, output
    except Exception as error:
        return False, str(error)


def _mem_info() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {"available": False}
    data: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        nums = rest.strip().split()
        if nums and nums[0].isdigit():
            data[key] = int(nums[0]) * 1024
    total = data.get("MemTotal", 0)
    available = data.get("MemAvailable", 0)
    used = max(total - available, 0) if total else 0
    percent = round((used / total) * 100, 1) if total else None
    return {"available": True, "total": total, "used": used, "percent": percent}


def _battery() -> dict[str, Any]:
    power_root = Path("/sys/class/power_supply")
    if not power_root.exists():
        return {"available": False}
    for item in power_root.iterdir():
        if not item.name.upper().startswith("BAT"):
            continue
        cap = item / "capacity"
        status = item / "status"
        return {
            "available": True,
            "name": item.name,
            "capacity": cap.read_text().strip() if cap.exists() else None,
            "status": status.read_text().strip() if status.exists() else None,
        }
    return {"available": False}


def check_ollama_api(timeout: int = 2) -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        data = json.loads(raw or "{}")
        models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        return {"available": True, "models": models}
    except Exception as error:
        return {"available": False, "error": str(error)}


def system_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or Path.cwd()
    disk = shutil.disk_usage(Path.home())
    loadavg = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "home": str(Path.home()),
        "disk_home": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": round((disk.used / disk.total) * 100, 1) if disk.total else None,
        },
        "memory": _mem_info(),
        "battery": _battery(),
        "loadavg": loadavg,
        "executables": {
            "ollama": shutil.which("ollama"),
            "piper": shutil.which("piper"),
            "ffplay": shutil.which("ffplay"),
            "gnome-terminal": shutil.which("gnome-terminal"),
            "xdg-open": shutil.which("xdg-open"),
        },
        "ollama_api": check_ollama_api(),
    }


def snapshot_text(project_root: Path | None = None) -> str:
    snap = system_snapshot(project_root)
    lines = [
        f"Python: {snap['python']}",
        f"Sistema: {snap['platform']}",
        f"Proyecto: {snap['project_root']}",
        f"Disco HOME: {snap['disk_home']['percent']}% usado",
    ]
    mem = snap.get("memory", {})
    if mem.get("available"):
        lines.append(f"RAM: {mem.get('percent')}% usada")
    bat = snap.get("battery", {})
    if bat.get("available"):
        lines.append(f"Batería: {bat.get('capacity')}% ({bat.get('status')})")
    api = snap.get("ollama_api", {})
    if api.get("available"):
        models = api.get("models") or []
        lines.append("Ollama API: OK" + (f" | modelos: {', '.join(models[:5])}" if models else ""))
    else:
        lines.append(f"Ollama API: NO ({api.get('error')})")
    return "\n".join(lines)
