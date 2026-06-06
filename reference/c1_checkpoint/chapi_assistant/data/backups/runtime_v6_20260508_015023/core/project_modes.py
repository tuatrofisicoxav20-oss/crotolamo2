"""
Crotolamo Project Modes v5.

Gestiona modos de trabajo: Crotolamo, Huevonitis, Tletl, Fedora, Escuela y Laboratorio.
La idea es que Crotolamo no responda igual cuando estás arreglando Fedora que cuando estás
pensando en una tarea o un prototipo. Increíble: contexto. La humanidad lo redescubre.
"""

from __future__ import annotations

import json
import shlex
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def _q(path: Path | str) -> str:
    return shlex.quote(str(Path(str(path)).expanduser()))


@dataclass
class ProjectMode:
    key: str
    title: str
    path: str
    description: str
    color: str = "cyan"
    keywords: list[str] | None = None
    habits: list[str] | None = None
    diagnostic_commands: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModeManager:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_dir = self.project_root / "config"
        self.config_path = self.config_dir / "crotolamo_modes.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_or_create()

    def _defaults(self) -> dict[str, Any]:
        return {
            "active_mode": "crotolamo",
            "modes": {
                "crotolamo": ProjectMode(
                    key="crotolamo",
                    title="Crotolamo",
                    path=str(self.project_root),
                    description="Asistente local general: terminal, voz, UI, skills, seguridad y diagnóstico.",
                    color="cyan",
                    keywords=["crotolamo", "asistente", "runtime", "orbital", "ui"],
                    habits=[
                        "Prioriza estabilidad del runtime antes de estética.",
                        "Sugiere comandos seguros primero y pide confirmación si modifican archivos.",
                    ],
                    diagnostic_commands=["python tools/crotolamo_doctor.py", "python -m py_compile core/crotolamo_runtime.py ui/crotolamo_orbital_ui.py"],
                ).to_dict(),
                "huevonitis": ProjectMode(
                    key="huevonitis",
                    title="Huevonitis",
                    path="~/Documentos/huevonitis version 2.1",
                    description="Proyecto de escritura/handwriting: extractor, glyph bank, tipografía realista, UI y flujo de revisión.",
                    color="magenta",
                    keywords=["huevonitis", "extractor", "handwriting", "letra", "glyph", "tipografia", "ocr"],
                    habits=[
                        "No tocar todo el extractor a ciegas: aislar traceback, módulo y función.",
                        "Proteger muestras/glyph bank antes de limpiar archivos.",
                        "Preferir mejoras medibles: detección, calidad de glifos, fallback y revisión humana.",
                    ],
                    diagnostic_commands=["pwd", "find . -maxdepth 2 -type f | sort | sed -n '1,120p'", "python -m py_compile main.py"],
                ).to_dict(),
                "tletl": ProjectMode(
                    key="tletl",
                    title="Tletl",
                    path="~/Documentos/tletl_control_v4_1_ai_integrado",
                    description="Proyecto de gestos/holograma: MediaPipe, OpenCV, entrenamiento, Blender y prototipos de control.",
                    color="purple",
                    keywords=["tletl", "gestos", "mediapipe", "opencv", "camara", "blender", "holograma", "mano"],
                    habits=[
                        "Separar captura, features, entrenamiento y UI para no mezclar mundos.",
                        "Medir cobertura de datasets antes de entrenar como si rezar fuera pipeline.",
                        "Cuidar Fedora/Wayland/cámara antes de culpar al modelo.",
                    ],
                    diagnostic_commands=["pwd", "find . -maxdepth 2 -type f | sort | sed -n '1,120p'", "python -m py_compile main.py train_gesture_lab.py 2>/dev/null || true"],
                ).to_dict(),
                "fedora": ProjectMode(
                    key="fedora",
                    title="Fedora / Sistema",
                    path="~",
                    description="Diagnóstico y mantenimiento de Fedora, paquetes, audio, drivers, logs y rendimiento.",
                    color="blue",
                    keywords=["fedora", "linux", "dnf", "systemctl", "journalctl", "audio", "driver", "ram", "bateria"],
                    habits=[
                        "Primero diagnosticar, luego modificar. No disparar sudo como confeti.",
                        "Guardar salidas de error completas, no fotos borrosas de un traceback poseído.",
                    ],
                    diagnostic_commands=["uname -a", "cat /etc/fedora-release", "free -h", "df -h ~", "systemctl --failed --no-pager"],
                ).to_dict(),
                "escuela": ProjectMode(
                    key="escuela",
                    title="Escuela",
                    path="~/Documentos",
                    description="Tareas, estudio, química, matemáticas, olimpiada, resúmenes y planes realistas.",
                    color="green",
                    keywords=["escuela", "tarea", "quimica", "mate", "examen", "olimpiada", "estudiar"],
                    habits=[
                        "Separar lo urgente de lo importante.",
                        "Usar bloques pequeños y entregables concretos para que sí se termine.",
                    ],
                    diagnostic_commands=["pwd", "find . -maxdepth 2 -type d | sed -n '1,80p'"],
                ).to_dict(),
                "laboratorio": ProjectMode(
                    key="laboratorio",
                    title="Laboratorio",
                    path="~/Documentos",
                    description="Electrónica, prototipos, garra robótica, sensores, piezas, pruebas y bitácora de inventos.",
                    color="yellow",
                    keywords=["laboratorio", "electronica", "robot", "garra", "arduino", "sensor", "prototipo", "hardware"],
                    habits=[
                        "Empezar con prototipo mínimo medible antes de comprar medio AliExpress.",
                        "Separar mecánica, electrónica, control y software.",
                    ],
                    diagnostic_commands=["pwd", "find . -maxdepth 2 -type d | sed -n '1,80p'"],
                ).to_dict(),
            },
        }

    def _load_or_create(self) -> dict[str, Any]:
        defaults = self._defaults()
        if not self.config_path.exists():
            self.config_path.write_text(json.dumps(defaults, indent=2, ensure_ascii=False), encoding="utf-8")
            return defaults
        try:
            loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("config no es objeto")
        except Exception:
            loaded = {}
        merged = defaults
        # Merge suave: respeta campos personalizados, añade modos faltantes.
        loaded_modes = loaded.get("modes") if isinstance(loaded.get("modes"), dict) else {}
        for key, mode in loaded_modes.items():
            if isinstance(mode, dict):
                base = merged["modes"].get(key, {})
                base.update(mode)
                merged["modes"][key] = base
        active = loaded.get("active_mode")
        if isinstance(active, str) and active in merged["modes"]:
            merged["active_mode"] = active
        self.config_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        return merged

    def save(self) -> None:
        self.config_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_modes(self) -> list[dict[str, Any]]:
        modes = self.data.get("modes", {})
        return [dict(mode) for _, mode in sorted(modes.items()) if isinstance(mode, dict)]

    def mode_keys(self) -> list[str]:
        return [m["key"] for m in self.list_modes() if m.get("key")]

    def get_mode(self, key: str | None = None) -> dict[str, Any]:
        modes = self.data.get("modes", {})
        key = key or str(self.data.get("active_mode") or "crotolamo")
        return dict(modes.get(key) or modes.get("crotolamo") or {})

    def active_key(self) -> str:
        return str(self.get_mode().get("key") or "crotolamo")

    def set_active_mode(self, requested: str) -> tuple[bool, str]:
        norm = _norm(requested)
        modes = self.data.get("modes", {})
        for key, mode in modes.items():
            names = [key, mode.get("title", "")] + list(mode.get("keywords") or [])
            if norm in {_norm(n) for n in names if n}:
                self.data["active_mode"] = key
                self.save()
                return True, key
        return False, requested

    def detect_requested_mode(self, text: str) -> str | None:
        lower = _norm(text)
        # Cambio explícito: "modo huevonitis", "cambia a tletl", etc.
        triggers = ["modo", "cambia a", "cambiar a", "pon modo", "activar modo", "activa modo"]
        if not any(t in lower for t in triggers):
            return None
        for mode in self.list_modes():
            candidates = [mode.get("key", ""), mode.get("title", "")] + list(mode.get("keywords") or [])
            if any(_norm(c) and _norm(c) in lower for c in candidates):
                return str(mode.get("key"))
        return None

    def infer_mode_from_text(self, text: str) -> str | None:
        lower = _norm(text)
        best_key: str | None = None
        best_score = 0
        for mode in self.list_modes():
            score = 0
            for kw in [mode.get("key", ""), mode.get("title", "")] + list(mode.get("keywords") or []):
                n = _norm(kw)
                if n and n in lower:
                    score += 2 if n == mode.get("key") else 1
            if score > best_score:
                best_score = score
                best_key = str(mode.get("key"))
        return best_key if best_score >= 2 else None

    def mode_context_text(self) -> str:
        mode = self.get_mode()
        path = Path(str(mode.get("path") or "~")).expanduser()
        habits = "\n".join(f"- {h}" for h in (mode.get("habits") or []))
        exists = "sí" if path.exists() else "no"
        return (
            f"MODO ACTIVO: {mode.get('title')} ({mode.get('key')})\n"
            f"Ruta del modo: {path}\n"
            f"Existe la ruta: {exists}\n"
            f"Descripción: {mode.get('description')}\n"
            f"Reglas de trabajo:\n{habits or '- Sin reglas especiales.'}"
        )

    def summary_text(self) -> str:
        active = self.active_key()
        lines = ["Modos de Crotolamo v5", f"Activo: {active}", ""]
        for mode in self.list_modes():
            marker = "→" if mode.get("key") == active else " "
            path = Path(str(mode.get("path") or "~")).expanduser()
            exists = "OK" if path.exists() else "NO"
            lines.append(f"{marker} {mode.get('key'):<12} {mode.get('title')} | {exists} | {path}")
        lines.append("\nComandos útiles: 'modo huevonitis', 'modo tletl', 'proyectos', 'abrir modo', 'terminal modo', 'diagnóstico modo'.")
        return "\n".join(lines)

    def status_text(self) -> str:
        mode = self.get_mode()
        path = Path(str(mode.get("path") or "~")).expanduser()
        lines = [f"Modo activo: {mode.get('title')} ({mode.get('key')})", f"Ruta: {path}", f"Existe: {'sí' if path.exists() else 'no'}"]
        if path.exists() and (path / ".git").exists():
            lines.append("Git: detectado")
        lines.append(f"Descripción: {mode.get('description')}")
        return "\n".join(lines)

    def open_active_commands(self) -> list[str]:
        path = Path(str(self.get_mode().get("path") or "~")).expanduser()
        return [f"xdg-open {_q(path)}"]

    def terminal_active_commands(self) -> list[str]:
        path = Path(str(self.get_mode().get("path") or "~")).expanduser()
        return [f"gnome-terminal --working-directory={_q(path)}"]

    def diagnostics_commands(self) -> list[str]:
        mode = self.get_mode()
        path = Path(str(mode.get("path") or "~")).expanduser()
        commands = list(mode.get("diagnostic_commands") or [])
        if path.exists():
            return [f"cd {_q(path)} && {cmd}" for cmd in commands]
        msg = f"La ruta del modo no existe: {path}"
        return [f"printf '%s\n' {shlex.quote(msg)}"]

    def handle_mode_command(self, text: str) -> dict[str, Any] | None:
        lower = _norm(text)
        requested = self.detect_requested_mode(text)
        if requested:
            ok, key = self.set_active_mode(requested)
            if ok:
                return {"kind": "direct", "text": "Modo cambiado.\n\n" + self.status_text(), "meta": {"mode": key}}
        if lower in {"modo", "modos", "modo actual", "lista modos", "listar modos"}:
            return {"kind": "direct", "text": self.summary_text(), "meta": {"mode": self.active_key()}}
        if lower in {"proyectos", "estado proyectos", "rutas", "rutas proyectos"}:
            return {"kind": "direct", "text": self.summary_text(), "meta": {"mode": self.active_key()}}
        if lower in {"abrir modo", "abrir proyecto", "abrir carpeta", "abre modo", "abre proyecto"}:
            return {
                "kind": "plan",
                "explanation": "Voy a abrir la carpeta del modo activo. No es magia, es xdg-open con autoestima.",
                "commands": self.open_active_commands(),
                "safe": True,
            }
        if lower in {"terminal modo", "terminal proyecto", "abrir terminal modo", "abre terminal modo"}:
            return {
                "kind": "plan",
                "explanation": "Voy a abrir una terminal en la carpeta del modo activo.",
                "commands": self.terminal_active_commands(),
                "safe": True,
            }
        if lower in {"diagnostico modo", "diagnóstico modo", "doctor modo", "revisar modo", "revisa modo"}:
            return {
                "kind": "plan",
                "explanation": "Diagnóstico del modo activo. Primero mirar, luego tocar. Concepto revolucionario.",
                "commands": self.diagnostics_commands(),
                "safe": True,
            }
        return None
