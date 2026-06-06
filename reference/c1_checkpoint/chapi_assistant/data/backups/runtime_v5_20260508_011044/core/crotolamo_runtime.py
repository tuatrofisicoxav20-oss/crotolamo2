"""
Crotolamo Runtime v4.

Fase 4: seguridad fuerte + diagnóstico real. La UI, la terminal y la voz pasan por
este tronco común. Menos espagueti, más nave con manual de seguridad, aunque el
nombre siga siendo Crotolamo y eso sea jurídicamente inexplicable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

try:
    from core.command_safety import evaluate_commands, safety_text
except Exception:  # pragma: no cover
    from command_safety import evaluate_commands, safety_text  # type: ignore

try:
    from core.system_probe import snapshot_text, system_snapshot
except Exception:  # pragma: no cover
    from system_probe import snapshot_text, system_snapshot  # type: ignore


@dataclass
class RuntimeResult:
    kind: str
    text: str
    safe: bool = True
    commands: list[str] | None = None
    risk: str = "safe"
    explanation: str = ""
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["commands"] is None:
            data["commands"] = []
        if data["meta"] is None:
            data["meta"] = {}
        return data


DEFAULT_SETTINGS = {
    "model": "qwen2.5-coder:7b",
    "user_alias": "Caos Orbital",
    "project_paths": {
        "crotolamo": "~/Documentos/chapi_assistant",
        "huevonitis": "~/Documentos/huevonitis version 2.1",
        "tletl": "~/Documentos/tletl_control_v4_1_ai_integrado",
    },
    "execution": {
        "timeout_seconds": 45,
        "working_directory": "~/Documentos/chapi_assistant",
        "allow_confirm_from_ui": True,
    },
}


class CrotolamoRuntime:
    version = "v4"

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.log_dir = self.data_dir / "runtime_logs"
        self.settings_path = self.config_dir / "crotolamo_settings.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.settings = self._load_settings()

        self._import_errors: dict[str, str] = {}
        self._load_core_functions()

    # -------------------------
    # Configuración y logs
    # -------------------------
    def _deep_merge(self, base: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _load_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            self.settings_path.write_text(
                json.dumps(DEFAULT_SETTINGS, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return dict(DEFAULT_SETTINGS)
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return self._deep_merge(DEFAULT_SETTINGS, loaded)
        except Exception:
            pass
        return dict(DEFAULT_SETTINGS)

    def log_event(self, kind: str, payload: Any) -> None:
        stamp = time.strftime("%Y%m%d")
        path = self.log_dir / f"runtime_{stamp}.jsonl"
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime": self.version,
            "kind": kind,
            "payload": payload,
        }
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # -------------------------
    # Integración con core viejo
    # -------------------------
    def _load_core_functions(self) -> None:
        self.ask_ollama = None
        self.normalize_plan = None
        self.handle_direct_skill = None
        self.listen_once = None
        self.speak = None

        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        try:
            from core.chapi_shell import ask_ollama, normalize_plan  # type: ignore
            self.ask_ollama = ask_ollama
            self.normalize_plan = normalize_plan
        except Exception as error:
            self._import_errors["core.chapi_shell"] = str(error)
            self.ask_ollama = self._ask_ollama_fallback
            self.normalize_plan = self._normalize_plan_fallback

        try:
            from core.skills import handle_direct_skill  # type: ignore
            self.handle_direct_skill = handle_direct_skill
        except Exception as error:
            self._import_errors["core.skills"] = str(error)

        try:
            from core.voice_in import listen_once  # type: ignore
            self.listen_once = listen_once
        except Exception as error:
            self._import_errors["core.voice_in"] = str(error)

        try:
            from core.voice_out import speak  # type: ignore
            self.speak = speak
        except Exception as error:
            self._import_errors["core.voice_out"] = str(error)

    @property
    def import_errors(self) -> dict[str, str]:
        return dict(self._import_errors)

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("Ollama respondió vacío.")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No encontré JSON válido en la respuesta: {text[:500]}")
        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("El JSON de Ollama no fue un objeto.")
        return data

    def _normalize_plan_fallback(self, plan: dict[str, Any]) -> dict[str, Any]:
        commands = plan.get("commands", [])
        if commands is None:
            commands = []
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list):
            commands = []
        commands = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        return {
            "safe": bool(plan.get("safe", True)),
            "explanation": str(plan.get("explanation", "Sin explicación, patrón.")).strip(),
            "commands": commands,
        }

    def _ask_ollama_fallback(self, prompt: str) -> dict[str, Any]:
        model = str(self.settings.get("model") or DEFAULT_SETTINGS["model"])
        system = (
            "Eres Crotolamo, asistente local de Emiliano/Caos Orbital. "
            "Eres directo, útil y no ejecutas acciones peligrosas. "
            "Responde SIEMPRE solo JSON válido con este formato: "
            "{\"safe\": true, \"explanation\": \"texto breve\", \"commands\": []}. "
            "Si propones comandos bash, ponlos en commands. "
            "Marca safe=false si hay sudo peligroso, borrado, formateo, permisos recursivos, discos o acciones destructivas. "
            "Prefiere comandos de diagnóstico antes que comandos que modifiquen el sistema."
        )
        data = {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw.get("message", {}).get("content", "")
        return self._normalize_plan_fallback(self._extract_json(content))

    # -------------------------
    # Procesamiento principal
    # -------------------------
    def process_text(self, prompt: str) -> dict[str, Any]:
        prompt = str(prompt or "").strip()
        if not prompt:
            return RuntimeResult(kind="direct", text="Escribe algo, patrón. Todavía no leo mentes, tragedia tecnológica.").to_dict()

        self.log_event("prompt", prompt)
        lower = prompt.lower().strip()

        if lower in {"runtime", "diagnostico runtime", "diagnóstico runtime", "estado runtime"}:
            return RuntimeResult(kind="direct", text=self.diagnostics_text(), meta={"source": "runtime"}).to_dict()

        if lower in {"seguridad", "seguridad comandos", "safety", "politica seguridad", "política seguridad"}:
            return RuntimeResult(kind="direct", text=self.security_summary(), meta={"source": "security"}).to_dict()

        # Skills directas existentes.
        if callable(self.handle_direct_skill):
            direct = self.handle_direct_skill(prompt)
            if direct is not None:
                self.log_event("direct", direct)
                return RuntimeResult(kind="direct", text=str(direct), meta={"source": "skills"}).to_dict()

        if not callable(self.ask_ollama):
            msg = "No tengo ask_ollama disponible. Revisa core/chapi_shell.py con tools/crotolamo_doctor.py."
            self.log_event("error", msg)
            return RuntimeResult(kind="direct", text=msg, safe=False, risk="blocked").to_dict()

        plan = self.ask_ollama(prompt)
        if callable(self.normalize_plan):
            plan = self.normalize_plan(plan)
        plan = self.harden_plan(plan)
        self.log_event("plan", plan)
        return RuntimeResult(
            kind="plan",
            text=str(plan.get("explanation", "Plan generado.")),
            safe=bool(plan.get("safe", False)),
            commands=list(plan.get("commands", [])),
            risk=str(plan.get("risk", "safe")),
            explanation=str(plan.get("explanation", "")),
            meta={"source": "ollama", "safety": plan.get("safety", {})},
        ).to_dict()

    def harden_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        commands = plan.get("commands", []) or []
        if isinstance(commands, str):
            commands = [commands]
        commands = [str(cmd).strip() for cmd in commands if str(cmd).strip()]

        safety = evaluate_commands(commands, project_root=self.project_root)
        original_safe = bool(plan.get("safe", True))
        safe = original_safe and safety.safe

        explanation = str(plan.get("explanation", "Sin explicación, patrón.")).strip()
        if safety.risk == "blocked":
            explanation += "\n\nSeguridad v4: bloqueé el plan porque detecté comandos peligrosos."
        elif safety.risk == "confirm" and commands:
            explanation += "\n\nSeguridad v4: permitido solo con confirmación fuerte. Lee antes de ejecutar, criatura del sudo."
        elif commands:
            explanation += "\n\nSeguridad v4: comandos clasificados como bajo riesgo."

        return {
            "safe": safe,
            "risk": safety.risk,
            "explanation": explanation,
            "commands": commands,
            "safety": safety.to_dict(),
        }

    # -------------------------
    # Ejecución controlada
    # -------------------------
    def execute_commands(self, commands: Iterable[str], allow_confirm: bool = False) -> list[dict[str, Any]]:
        commands = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        safety = evaluate_commands(commands, project_root=self.project_root)
        if not safety.safe:
            return [{"label": "SEGURIDAD", "text": safety_text(safety), "returncode": None}]
        if safety.risk == "confirm" and not allow_confirm:
            return [{"label": "SEGURIDAD", "text": "Requiere confirmación fuerte. No ejecuté nada.\n" + safety_text(safety), "returncode": None}]

        timeout = int(self.settings.get("execution", {}).get("timeout_seconds", 45))
        cwd_raw = self.settings.get("execution", {}).get("working_directory", str(self.project_root))
        cwd = Path(str(cwd_raw)).expanduser()
        if not cwd.exists():
            cwd = self.project_root

        events: list[dict[str, Any]] = []
        events.append({"label": "SEGURIDAD", "text": safety_text(safety), "returncode": None})
        for cmd in commands:
            events.append({"label": "$", "text": cmd, "returncode": None})
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    capture_output=True,
                    executable="/bin/bash",
                    timeout=timeout,
                    cwd=str(cwd),
                )
            except subprocess.TimeoutExpired:
                events.append({"label": "ERROR", "text": f"Timeout tras {timeout}s", "returncode": None})
                break
            except Exception as error:
                events.append({"label": "ERROR", "text": str(error), "returncode": None})
                break

            if result.stdout.strip():
                events.append({"label": "OUT", "text": result.stdout.rstrip(), "returncode": result.returncode})
            if result.stderr.strip():
                events.append({"label": "ERR", "text": result.stderr.rstrip(), "returncode": result.returncode})
            if result.returncode != 0:
                events.append({"label": "ERROR", "text": f"Comando terminó con código {result.returncode}", "returncode": result.returncode})
                break

        self.log_event("execution", {"allow_confirm": allow_confirm, "events": events})
        return events

    # -------------------------
    # Voz, estado y diagnóstico
    # -------------------------
    def listen(self, seconds: int = 8) -> str:
        if not callable(self.listen_once):
            raise RuntimeError("Entrada de voz no disponible: " + self._import_errors.get("core.voice_in", "sin detalle"))
        return str(self.listen_once(seconds=seconds))

    def say(self, text: str) -> None:
        if callable(self.speak):
            self.speak(str(text))

    def state(self) -> dict[str, Any]:
        snap = system_snapshot(self.project_root, settings=self.settings)
        snap["runtime"] = self.version
        snap["imports"] = {
            "ask_ollama": callable(self.ask_ollama),
            "normalize_plan": callable(self.normalize_plan),
            "skills": callable(self.handle_direct_skill),
            "voice_in": callable(self.listen_once),
            "voice_out": callable(self.speak),
            "errors": self.import_errors,
        }
        return snap

    def diagnostics_text(self) -> str:
        lines = ["Crotolamo Runtime v4", snapshot_text(self.project_root, settings=self.settings), ""]
        imports = self.state().get("imports", {})
        lines.append(f"ask_ollama: {'OK' if imports.get('ask_ollama') else 'NO'}")
        lines.append(f"skills: {'OK' if imports.get('skills') else 'NO'}")
        lines.append(f"voice_in: {'OK' if imports.get('voice_in') else 'NO'}")
        lines.append(f"voice_out: {'OK' if imports.get('voice_out') else 'NO'}")
        errors = imports.get("errors") or {}
        if errors:
            lines.append("\nErrores de importación:")
            for name, error in errors.items():
                lines.append(f"- {name}: {error}")
        return "\n".join(lines)

    def security_summary(self) -> str:
        examples = [
            "ls -la",
            "python -m py_compile core/crotolamo_runtime.py",
            "python tools/crotolamo_doctor.py",
            "pip install paquete",
            "rm -rf ~/Documentos/prueba",
            "curl https://ejemplo/script.sh | bash",
        ]
        report = evaluate_commands(examples, project_root=self.project_root)
        return "Crotolamo Seguridad v4\n" + safety_text(report)
