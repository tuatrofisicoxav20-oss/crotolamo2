"""Diagnóstico realista para Crotolamo v4 sin depender de psutil."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
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
    return {"available": True, "total": total, "used": used, "free": available, "percent": percent}


def _read_cpu_line() -> list[int] | None:
    path = Path("/proc/stat")
    if not path.exists():
        return None
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].split()
    if not first or first[0] != "cpu":
        return None
    vals: list[int] = []
    for item in first[1:]:
        try:
            vals.append(int(item))
        except ValueError:
            vals.append(0)
    return vals


def _cpu_percent(sample_delay: float = 0.12) -> dict[str, Any]:
    a = _read_cpu_line()
    if not a:
        return {"available": False}
    time.sleep(sample_delay)
    b = _read_cpu_line()
    if not b:
        return {"available": False}
    total_a = sum(a)
    total_b = sum(b)
    idle_a = a[3] + (a[4] if len(a) > 4 else 0)
    idle_b = b[3] + (b[4] if len(b) > 4 else 0)
    total_delta = max(total_b - total_a, 1)
    idle_delta = max(idle_b - idle_a, 0)
    used = round((1 - idle_delta / total_delta) * 100, 1)
    return {"available": True, "percent": max(0.0, min(100.0, used)), "cores": os.cpu_count()}


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


def _network() -> dict[str, Any]:
    root = Path("/sys/class/net")
    if not root.exists():
        return {"available": False}
    interfaces: list[dict[str, Any]] = []
    for iface in sorted(root.iterdir()):
        if iface.name == "lo":
            continue
        oper = iface / "operstate"
        state = oper.read_text().strip() if oper.exists() else "unknown"
        interfaces.append({"name": iface.name, "state": state})
    ok, route = _run(["bash", "-lc", "ip route get 1.1.1.1 2>/dev/null | head -n1"], timeout=2)
    return {"available": True, "interfaces": interfaces, "default_route": route if ok else "", "online_guess": any(i["state"] == "up" for i in interfaces)}


def _audio() -> dict[str, Any]:
    try:
        import sounddevice as sd  # type: ignore
        devices = sd.query_devices()
        input_count = 0
        output_count = 0
        names: list[str] = []
        for d in devices:
            try:
                if d.get("max_input_channels", 0) > 0:
                    input_count += 1
                if d.get("max_output_channels", 0) > 0:
                    output_count += 1
                if len(names) < 5:
                    names.append(str(d.get("name", "audio")))
            except AttributeError:
                pass
        return {"available": True, "inputs": input_count, "outputs": output_count, "sample": names}
    except Exception as error:
        return {"available": False, "error": str(error)}


def _top_processes() -> list[dict[str, Any]]:
    ok, out = _run(["bash", "-lc", "ps -eo pid,comm,%cpu,%mem --sort=-%mem | head -n 6"], timeout=2)
    if not ok or not out:
        return []
    lines = out.splitlines()[1:]
    items: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split(None, 3)
        if len(parts) == 4:
            pid, comm, cpu, mem = parts
            items.append({"pid": pid, "name": comm, "cpu": cpu, "mem": mem})
    return items


def _git_status(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"git": False}
    ok_branch, branch = _run(["git", "-C", str(path), "branch", "--show-current"], timeout=2)
    ok_status, status = _run(["git", "-C", str(path), "status", "--porcelain"], timeout=2)
    return {
        "git": True,
        "branch": branch if ok_branch else "?",
        "dirty": bool(status.strip()) if ok_status else None,
        "changes": len(status.splitlines()) if ok_status and status else 0,
    }


def project_snapshot(project_paths: dict[str, str] | None = None) -> dict[str, Any]:
    project_paths = project_paths or {}
    out: dict[str, Any] = {}
    for name, raw in project_paths.items():
        path = Path(str(raw)).expanduser()
        info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            info.update(_git_status(path))
        out[name] = info
    return out


def check_ollama_api(timeout: int = 2) -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        data = json.loads(raw or "{}")
        models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        ps_models: list[str] = []
        try:
            with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=timeout) as response:
                ps_raw = response.read().decode("utf-8", errors="ignore")
            ps_data = json.loads(ps_raw or "{}")
            ps_models = [m.get("name") or m.get("model") for m in ps_data.get("models", []) if isinstance(m, dict)]
        except Exception:
            ps_models = []
        return {"available": True, "models": models, "running": [m for m in ps_models if m]}
    except Exception as error:
        return {"available": False, "error": str(error)}


def system_snapshot(project_root: Path | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    project_root = project_root or Path.cwd()
    disk = shutil.disk_usage(Path.home())
    loadavg = os.getloadavg() if hasattr(os, "getloadavg") else None
    settings = settings or {}
    project_paths = settings.get("project_paths") if isinstance(settings.get("project_paths"), dict) else {}
    return {
        "version": "v4",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "home": str(Path.home()),
        "cpu": _cpu_percent(),
        "disk_home": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": round((disk.used / disk.total) * 100, 1) if disk.total else None,
        },
        "memory": _mem_info(),
        "battery": _battery(),
        "network": _network(),
        "audio": _audio(),
        "top_processes": _top_processes(),
        "projects": project_snapshot(project_paths),
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


def snapshot_text(project_root: Path | None = None, settings: dict[str, Any] | None = None) -> str:
    snap = system_snapshot(project_root, settings=settings)
    lines = [
        "Crotolamo System Probe v4",
        f"Python: {snap['python']}",
        f"Sistema: {snap['platform']}",
        f"Proyecto: {snap['project_root']}",
        f"Disco HOME: {snap['disk_home']['percent']}% usado",
    ]
    cpu = snap.get("cpu", {})
    if cpu.get("available"):
        lines.append(f"CPU: {cpu.get('percent')}% | núcleos: {cpu.get('cores')}")
    mem = snap.get("memory", {})
    if mem.get("available"):
        lines.append(f"RAM: {mem.get('percent')}% usada")
    bat = snap.get("battery", {})
    if bat.get("available"):
        lines.append(f"Batería: {bat.get('capacity')}% ({bat.get('status')})")
    net = snap.get("network", {})
    if net.get("available"):
        up = ", ".join(i["name"] for i in net.get("interfaces", []) if i.get("state") == "up") or "sin interfaz activa"
        lines.append(f"Red: {up}")
    audio = snap.get("audio", {})
    if audio.get("available"):
        lines.append(f"Audio: {audio.get('inputs')} entradas | {audio.get('outputs')} salidas")
    else:
        lines.append(f"Audio: NO ({audio.get('error')})")
    api = snap.get("ollama_api", {})
    if api.get("available"):
        models = api.get("models") or []
        running = api.get("running") or []
        lines.append("Ollama API: OK" + (f" | modelos: {', '.join(models[:5])}" if models else ""))
        if running:
            lines.append("Ollama en memoria: " + ", ".join(running[:3]))
    else:
        lines.append(f"Ollama API: NO ({api.get('error')})")
    projs = snap.get("projects", {})
    if projs:
        lines.append("\nProyectos:")
        for name, info in projs.items():
            if info.get("exists"):
                git = ""
                if info.get("git"):
                    git = f" | git:{info.get('branch') or '?'} | cambios:{info.get('changes')}"
                lines.append(f"- {name}: OK{git}")
            else:
                lines.append(f"- {name}: NO ({info.get('path')})")
    return "\n".join(lines)
