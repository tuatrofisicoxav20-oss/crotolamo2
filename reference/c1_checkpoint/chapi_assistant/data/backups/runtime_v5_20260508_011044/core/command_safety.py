"""
Crotolamo command safety layer v4.

Fase 4: seguridad más estricta para ejecutar comandos desde la UI/runtime.
No es un sandbox militar. Es un cinturón de seguridad decente para que Crotolamo
no convierta tu Fedora en confeti por obedecer comandos demasiado felices.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Any

RiskLevel = Literal["safe", "confirm", "blocked"]


@dataclass(frozen=True)
class CommandCheck:
    command: str
    risk: RiskLevel
    allowed: bool
    reasons: list[str]
    category: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyReport:
    safe: bool
    risk: RiskLevel
    checks: list[CommandCheck]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "risk": self.risk,
            "checks": [check.to_dict() for check in self.checks],
            "explanation": self.explanation,
        }


# Bloqueo duro: cosas destructivas, privilegios peligrosos, tuberías remotas, discos, permisos masivos.
BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[^\n;|&]*r[^\n;|&]*f\b", "rm recursivo/forzado puede borrar demasiadas cosas"),
    (r"\bsudo\s+rm\b", "sudo rm queda bloqueado"),
    (r"\brm\s+[^\n;|&]*(/|~|\$HOME)(\s|$)", "rm apuntando a raíz/home es demasiado peligroso"),
    (r"\bfind\b[^\n;|&]*\s-delete\b", "find -delete puede borrar árboles completos"),
    (r"\b(shred|wipe|srm)\b", "herramienta de borrado destructivo detectada"),
    (r"\bmkfs(\.|\s|$)", "mkfs formatea discos"),
    (r"\b(dd|dcfldd)\s+.*\bof=", "dd/dcfldd puede sobrescribir discos"),
    (r"\b(parted|fdisk|sfdisk|cfdisk|gdisk|sgdisk)\b", "herramienta de particiones bloqueada desde la UI"),
    (r"\b(cryptsetup|wipefs|blkdiscard)\b", "acción peligrosa sobre discos detectada"),
    (r">\s*/dev/(sd[a-z]|nvme\d+n\d+|mmcblk\d+)", "redirigir a un dispositivo de disco queda bloqueado"),
    (r"\bchmod\s+-R\s+777\s+(/|~|\$HOME)?", "chmod -R 777 masivo es mala idea"),
    (r"\bchown\s+-R\b.*(/|~|\$HOME)", "chown -R masivo puede romper permisos"),
    (r"\b(chmod|chown)\b[^\n;|&]*(/usr|/etc|/boot|/bin|/sbin|/lib|/lib64)\b", "permisos sobre carpetas del sistema quedan bloqueados"),
    (r"\b(shutdown|reboot|poweroff|halt|systemctl\s+poweroff|systemctl\s+reboot)\b", "apagar/reiniciar desde la UI queda bloqueado"),
    (r":\s*\(\s*\)\s*\{", "fork bomb detectada"),
    (r"\b(systemctl|dnf)\s+(disable|remove|erase|autoremove)\b", "desactivar/remover sistema requiere revisión manual"),
    (r"\bcrontab\s+-r\b", "borrar crontab queda bloqueado"),
    (r"\b(passwd|userdel|groupdel|usermod|visudo)\b", "gestión de usuarios/sudo bloqueada desde la UI"),
    (r"\b(chattr\s+[+-]i|setenforce\s+0|firewall-cmd\s+--permanent|iptables|nft)\b", "cambio sensible de seguridad/sistema detectado"),
    (r"\b(curl|wget)\b[^\n]*(\||>)\s*(sudo\s+)?(bash|sh)\b", "descargar y ejecutar scripts remotos queda bloqueado"),
    (r"\b(sudo\s+)?(bash|sh)\s+-c\b", "shell privilegiada/dinámica requiere revisión manual, no UI"),
    (r"\beval\b", "eval queda bloqueado"),
    (r"\bhistory\s+-c\b", "borrar historial queda bloqueado"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-fdx?)\b", "git reset/clean destructivo queda bloqueado"),
]

# Confirmación fuerte: modifica cosas, instala, corre scripts o toca git/servicios.
CONFIRM_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*sudo\b", "usa sudo; requiere confirmación fuerte fuera del modo seguro"),
    (r"\b(dnf|pip|pip3|flatpak|npm|pnpm|yarn)\s+(install|update|upgrade|add)\b", "instala o actualiza paquetes"),
    (r"\b(chmod|chown|setfacl)\b", "modifica permisos/propietarios"),
    (r"\b(mkdir|cp|mv|touch|tee)\b", "modifica archivos o carpetas"),
    (r"\b(cat|printf|echo)\b[^\n]*(>|>>)\s*", "escribe a un archivo"),
    (r"\bgit\s+(checkout|switch|pull|merge|rebase|commit|add|restore|stash|branch|tag)\b", "modifica estado de git"),
    (r"\bpython(3|)\s+[^\n;|&]+\.py\b", "ejecuta un script Python"),
    (r"\b(bash|sh|zsh)\s+[^\n;|&]+\.(sh|zsh)\b", "ejecuta un script shell"),
    (r"\b(systemctl)\s+(start|stop|restart|enable|reload)\b", "modifica servicios del sistema"),
    (r"\bkill(all)?\b", "mata procesos"),
]

READ_ONLY_PREFIXES = (
    "pwd", "ls", "tree", "find", "grep", "rg", "fd", "cat", "head", "tail",
    "wc", "du", "df", "free", "uptime", "whoami", "id", "date", "uname",
    "hostname", "ip a", "ip addr", "ip route", "nmcli", "journalctl", "systemctl status",
    "python -m py_compile", "python3 -m py_compile", "python -V", "python --version",
    "python3 -V", "python3 --version", "pytest", "git status", "git log", "git diff",
    "git branch", "git remote", "xdg-open", "ollama list", "ollama ps",
)

SHELL_CHAIN_RE = re.compile(r"\s*(;|&&|\|\|)\s*")


def load_extra_policy(project_root: str | Path | None = None) -> dict[str, Any]:
    """Carga config/security_policy.json si existe. Todo es opcional."""
    root = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
    path = root / "config" / "security_policy.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def split_shell_chain(command: str) -> list[str]:
    """Separación básica para detectar comandos peligrosos encadenados.

    No pretende parsear Bash completo. Para eso ni Bash se entiende a sí mismo algunos días.
    """
    parts = [p.strip() for p in SHELL_CHAIN_RE.split(command) if p.strip() not in {";", "&&", "||"}]
    return parts or [command.strip()]


def _matches(patterns: Iterable[tuple[str, str]], text: str) -> list[str]:
    reasons: list[str] = []
    for pattern, reason in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(reason)
    return reasons


def _first_token(command: str) -> str:
    try:
        parts = shlex.split(command)
    except Exception:
        parts = command.strip().split()
    return parts[0] if parts else ""


def _is_read_only(command: str) -> bool:
    lower = command.strip().lower()
    pieces = split_shell_chain(lower)
    if not pieces:
        return False
    return all(any(piece.startswith(prefix) for prefix in READ_ONLY_PREFIXES) for piece in pieces)


def _policy_match(command: str, project_root: str | Path | None = None) -> tuple[RiskLevel | None, list[str]]:
    policy = load_extra_policy(project_root)
    lower = command.lower()
    for pattern in policy.get("blocked_patterns", []) or []:
        try:
            if re.search(str(pattern), lower, flags=re.IGNORECASE):
                return "blocked", [f"bloqueado por security_policy.json: {pattern}"]
        except re.error:
            continue
    for pattern in policy.get("confirm_patterns", []) or []:
        try:
            if re.search(str(pattern), lower, flags=re.IGNORECASE):
                return "confirm", [f"confirmación por security_policy.json: {pattern}"]
        except re.error:
            continue
    for prefix in policy.get("safe_prefixes", []) or []:
        if lower.startswith(str(prefix).lower()):
            return "safe", []
    return None, []


def check_command(command: str, project_root: str | Path | None = None) -> CommandCheck:
    cmd = str(command or "").strip()
    if not cmd:
        return CommandCheck(command=cmd, risk="blocked", allowed=False, reasons=["comando vacío"], category="empty")

    policy_risk, policy_reasons = _policy_match(cmd, project_root)
    if policy_risk == "blocked":
        return CommandCheck(command=cmd, risk="blocked", allowed=False, reasons=policy_reasons, category="policy")
    if policy_risk == "safe":
        return CommandCheck(command=cmd, risk="safe", allowed=True, reasons=[], category="policy")

    lower = cmd.lower()
    reasons = _matches(BLOCK_PATTERNS, lower)
    if reasons:
        return CommandCheck(command=cmd, risk="blocked", allowed=False, reasons=reasons, category="destructive")

    if _is_read_only(cmd):
        return CommandCheck(command=cmd, risk="safe", allowed=True, reasons=[], category="read_only")

    confirm_reasons = policy_reasons + _matches(CONFIRM_PATTERNS, lower)
    if confirm_reasons:
        return CommandCheck(command=cmd, risk="confirm", allowed=True, reasons=confirm_reasons, category="modifies_system")

    token = _first_token(cmd)
    if token in {"python", "python3", "bash", "sh", "zsh", "node", "npm", "pip", "pip3", "dnf"}:
        return CommandCheck(
            command=cmd,
            risk="confirm",
            allowed=True,
            reasons=["ejecuta herramienta capaz de modificar el entorno; confirmación fuerte requerida"],
            category="exec_tool",
        )

    return CommandCheck(
        command=cmd,
        risk="confirm",
        allowed=True,
        reasons=["comando no clasificado; requiere confirmación visual"],
        category="unknown",
    )


def evaluate_commands(commands: Iterable[str], project_root: str | Path | None = None) -> SafetyReport:
    clean = [str(c).strip() for c in commands if str(c).strip()]
    checks = [check_command(cmd, project_root=project_root) for cmd in clean]

    if not clean:
        return SafetyReport(safe=True, risk="safe", checks=[], explanation="Sin comandos que ejecutar.")

    if any(check.risk == "blocked" for check in checks):
        return SafetyReport(
            safe=False,
            risk="blocked",
            checks=checks,
            explanation="Plan bloqueado por comandos peligrosos.",
        )

    if any(check.risk == "confirm" for check in checks):
        return SafetyReport(
            safe=True,
            risk="confirm",
            checks=checks,
            explanation="Plan permitido, pero requiere confirmación fuerte porque modifica o ejecuta cosas.",
        )

    return SafetyReport(
        safe=True,
        risk="safe",
        checks=checks,
        explanation="Plan de solo lectura o bajo riesgo.",
    )


def safety_text(report: SafetyReport) -> str:
    lines = [f"Riesgo general: {report.risk.upper()}", report.explanation]
    for idx, check in enumerate(report.checks, 1):
        lines.append(f"{idx}. [{check.risk.upper()}] {check.command}")
        for reason in check.reasons:
            lines.append(f"   - {reason}")
    return "\n".join(lines)
