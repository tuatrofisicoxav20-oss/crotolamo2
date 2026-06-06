"""
Crotolamo command safety layer.

No pretende ser seguridad militar, porque esto sigue siendo Python corriendo en
una laptop humana, pero sí evita los desastres obvios antes de ejecutar comandos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable, Literal

RiskLevel = Literal["safe", "confirm", "blocked"]


@dataclass(frozen=True)
class CommandCheck:
    command: str
    risk: RiskLevel
    allowed: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SafetyReport:
    safe: bool
    risk: RiskLevel
    checks: list[CommandCheck]
    explanation: str

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "risk": self.risk,
            "checks": [check.to_dict() for check in self.checks],
            "explanation": self.explanation,
        }


BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[^\n;]*r[^\n;]*f\b", "rm recursivo/forzado puede borrar demasiadas cosas"),
    (r"\brm\s+[^\n;]*(/|~|\$HOME)(\s|$)", "rm apuntando a raíz/home es demasiado peligroso"),
    (r"\bsudo\s+rm\b", "sudo rm queda bloqueado"),
    (r"\bmkfs(\.|\s|$)", "mkfs formatea discos"),
    (r"\bdd\s+.*\bof=", "dd puede sobrescribir discos"),
    (r">\s*/dev/(sd[a-z]|nvme\d+n\d+|mmcblk\d+)", "redirigir a un dispositivo de disco queda bloqueado"),
    (r"\bchmod\s+-R\s+777\s+(/|~|\$HOME)?", "chmod -R 777 masivo es mala idea"),
    (r"\bchown\s+-R\b.*(/|~|\$HOME)", "chown -R masivo puede romper permisos"),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "apagar/reiniciar desde la UI queda bloqueado"),
    (r":\s*\(\s*\)\s*\{", "fork bomb detectada"),
    (r"\b(systemctl|dnf)\s+(disable|remove|erase|autoremove)\b", "desactivar/remover sistema requiere revisión manual"),
    (r"\bcrontab\s+-r\b", "borrar crontab queda bloqueado"),
]

CONFIRM_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*sudo\b", "usa sudo; necesita confirmación fuera del modo seguro"),
    (r"\b(dnf|pip|pip3|flatpak)\s+install\b", "instala paquetes"),
    (r"\b(chmod|chown)\b", "modifica permisos/propietarios"),
    (r"\b(mkdir|cp|mv)\b", "modifica archivos o carpetas"),
    (r"\bgit\s+(checkout|switch|reset|clean|pull|merge|rebase)\b", "modifica estado de git"),
    (r"\bpython(3|)\s+[^\n;]+\.py\b", "ejecuta un script Python"),
    (r"\bbash\s+[^\n;]+\.sh\b", "ejecuta un script shell"),
]

READ_ONLY_PREFIXES = (
    "pwd", "ls", "tree", "find", "grep", "rg", "fd", "cat", "head", "tail",
    "wc", "du", "df", "free", "uptime", "whoami", "id", "date", "uname",
    "python -m py_compile", "python3 -m py_compile", "pytest", "xdg-open",
)

SHELL_CHAIN_RE = re.compile(r"\s*(;|&&|\|\|)\s*")


def split_shell_chain(command: str) -> list[str]:
    """Separación básica para detectar comandos peligrosos encadenados."""
    parts = [p.strip() for p in SHELL_CHAIN_RE.split(command) if p.strip() not in {";", "&&", "||"}]
    return parts or [command.strip()]


def _matches(patterns: Iterable[tuple[str, str]], text: str) -> list[str]:
    reasons: list[str] = []
    for pattern, reason in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(reason)
    return reasons


def check_command(command: str) -> CommandCheck:
    cmd = str(command or "").strip()
    if not cmd:
        return CommandCheck(command=cmd, risk="blocked", allowed=False, reasons=["comando vacío"])

    lower = cmd.lower()
    reasons = _matches(BLOCK_PATTERNS, lower)
    if reasons:
        return CommandCheck(command=cmd, risk="blocked", allowed=False, reasons=reasons)

    confirm_reasons = _matches(CONFIRM_PATTERNS, lower)

    # Si todos los trozos encadenados son de lectura, lo tratamos como seguro.
    pieces = split_shell_chain(lower)
    if pieces and all(piece.startswith(READ_ONLY_PREFIXES) for piece in pieces):
        risk: RiskLevel = "safe"
        return CommandCheck(command=cmd, risk=risk, allowed=True, reasons=[])

    if confirm_reasons:
        return CommandCheck(command=cmd, risk="confirm", allowed=True, reasons=confirm_reasons)

    # Comandos no reconocidos: se permiten con confirmación, no como totalmente seguros.
    return CommandCheck(
        command=cmd,
        risk="confirm",
        allowed=True,
        reasons=["comando no clasificado; requiere confirmación visual"],
    )


def evaluate_commands(commands: Iterable[str]) -> SafetyReport:
    clean = [str(c).strip() for c in commands if str(c).strip()]
    checks = [check_command(cmd) for cmd in clean]

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
            explanation="Plan permitido, pero requiere confirmación porque modifica o ejecuta cosas.",
        )

    return SafetyReport(
        safe=True,
        risk="safe",
        checks=checks,
        explanation="Plan de solo lectura o bajo riesgo.",
    )
