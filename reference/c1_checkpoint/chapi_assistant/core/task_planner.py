"""
Crotolamo Task Planner v12.

Convierte objetivos en planes técnicos usando:
- modos/proyectos conocidos
- project_indexer v10
- project_inspector v11
- command_safety si existe

Por diseño NO ejecuta cambios. Solo planea.
Porque planear antes de romper cosas es una costumbre rara pero recomendable.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PlanStep:
    id: int
    title: str
    why: str
    commands: list[str]
    risk: str = "safe"
    expected_result: str = ""


@dataclass
class TaskPlan:
    title: str
    project: str
    objective: str
    created_at: str
    risk_summary: str
    assumptions: list[str]
    steps: list[PlanStep]
    validation: list[str]
    rollback: list[str]
    next_action: str


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _plans_dir(root: Path | None = None) -> Path:
    root = root or _root()
    p = root / "data" / "task_plans"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()).strip("_") or "plan"


def _load_known_projects(root: Path) -> dict[str, str]:
    try:
        from core.project_indexer import known_projects
        return known_projects(root)
    except Exception:
        try:
            from core.local_memory import load_memory
            memory = load_memory(root)
            paths = dict(memory.get("project_paths", {}) or {})
            paths.setdefault("crotolamo", str(root))
            return paths
        except Exception:
            return {"crotolamo": str(root)}


def _detect_project(text: str, root: Path) -> str:
    low = text.lower()
    projects = _load_known_projects(root)
    for name in projects:
        if name.lower() in low:
            return name
    if "huevo" in low or "extractor" in low or "glifo" in low or "handwriting" in low:
        return "huevonitis"
    if "tletl" in low or "gesto" in low or "cámara" in low or "camara" in low or "blender" in low:
        return "tletl"
    if "fedora" in low or "linux" in low or "audio" in low or "ollama" in low:
        return "crotolamo"
    try:
        from core.project_modes import get_current_mode
        mode = get_current_mode(root)
        if isinstance(mode, str) and mode in projects:
            return mode
        if hasattr(mode, "name") and str(mode.name) in projects:
            return str(mode.name)
    except Exception:
        pass
    return "crotolamo"


def _resolve_project_path(project: str, root: Path) -> Path:
    try:
        from core.project_indexer import resolve_project
        _, path = resolve_project(project, root)
        return path
    except Exception:
        projects = _load_known_projects(root)
        return Path(os.path.expanduser(str(projects.get(project, root)))).resolve()


def _inspect(project: str, root: Path) -> Any | None:
    try:
        from core.project_inspector import inspect_project
        return inspect_project(project, root)
    except Exception:
        return None


def _safe_quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def _classify_command(cmd: str) -> str:
    try:
        from core.command_safety import classify_command
        result = classify_command(cmd)
        # soporta dict, objeto o string, porque quién sabe cómo quedó tu módulo anterior
        if isinstance(result, str):
            return result.lower()
        if isinstance(result, dict):
            return str(result.get("level") or result.get("risk") or result.get("status") or "confirm").lower()
        for attr in ("level", "risk", "status"):
            if hasattr(result, attr):
                return str(getattr(result, attr)).lower()
    except Exception:
        pass

    low = cmd.lower()
    dangerous = ("rm -rf", "mkfs", "dd if=", "sudo rm", "chmod -r 777 /", "curl | bash", "wget | bash")
    if any(x in low for x in dangerous):
        return "blocked"
    if any(x in low for x in ("sudo ", "dnf install", "pip install", "chmod ", "mv ", "cp ", "git ")):
        return "confirm"
    return "safe"


def _step(id_: int, title: str, why: str, commands: list[str], expected: str = "") -> PlanStep:
    risks = [_classify_command(c) for c in commands]
    if any("block" in r for r in risks):
        risk = "blocked"
    elif any("confirm" in r or "medium" in r or "risk" in r for r in risks):
        risk = "confirm"
    else:
        risk = "safe"
    return PlanStep(id=id_, title=title, why=why, commands=commands, risk=risk, expected_result=expected)


def _entrypoint_commands(project_path: Path, report: Any | None) -> list[str]:
    cmds: list[str] = []
    entries = []
    if report is not None:
        entries = list(getattr(report, "likely_entrypoints", []) or [])
    if not entries:
        for candidate in ("main.py", "app.py", "run.py"):
            if (project_path / candidate).exists():
                entries.append(candidate)
    for rel in entries[:3]:
        cmds.append(f"python -m py_compile {_safe_quote(project_path / rel)}")
    return cmds


def _project_specific_steps(project: str, objective: str, project_path: Path, report: Any | None) -> list[PlanStep]:
    low = f"{project} {objective}".lower()
    steps: list[PlanStep] = []

    if "huevonitis" in low or "extractor" in low or "glifo" in low:
        steps.extend([
            _step(
                4,
                "Localizar extractor, UI y módulos críticos",
                "Huevonitis suele romperse cuando UI, extractor y bancos de glifos quedan desconectados.",
                [
                    "python tools/crotolamo_project_index_cli.py buscar archivo extractor en huevonitis",
                    "python tools/crotolamo_project_index_cli.py buscar archivo glyph en huevonitis",
                    "python tools/crotolamo_project_index_cli.py buscar texto class ExtractorApp en huevonitis",
                ],
                "Rutas exactas de extractor, glifos y clases principales."
            ),
            _step(
                5,
                "Compilar archivos críticos antes de tocar diseño",
                "Antes de mejorar estética o realismo, hay que garantizar que Python importe y compile.",
                _entrypoint_commands(project_path, report),
                "Sin errores de sintaxis en entrypoints principales."
            ),
        ])
    elif "tletl" in low or "gesto" in low or "camara" in low or "cámara" in low or "blender" in low:
        steps.extend([
            _step(
                4,
                "Ubicar gestos, cámara y datasets",
                "Tletl depende de que cámara, dataset y reconocimiento estén bien conectados.",
                [
                    "python tools/crotolamo_project_index_cli.py buscar archivo gesture en tletl",
                    "python tools/crotolamo_project_index_cli.py buscar archivo camera en tletl",
                    "python tools/crotolamo_project_index_cli.py buscar texto GESTO en tletl",
                ],
                "Archivos candidatos para gestos, cámara y datasets."
            ),
            _step(
                5,
                "Validar entrypoints probables",
                "Si los launchers no compilan, la precisión del sistema da igual porque ni arranca.",
                _entrypoint_commands(project_path, report),
                "Entry points compilados."
            ),
        ])
    else:
        steps.extend([
            _step(
                4,
                "Revisar módulos críticos del proyecto",
                "El objetivo parece general, así que conviene ubicar núcleo, launchers y plugins.",
                [
                    f"python tools/crotolamo_project_index_cli.py mapa {shlex.quote(project)}",
                    f"python tools/crotolamo_project_inspector_cli.py entradas {shlex.quote(project)}",
                    f"python tools/crotolamo_project_inspector_cli.py dependencias {shlex.quote(project)}",
                ],
                "Mapa técnico y puntos de entrada."
            ),
            _step(
                5,
                "Compilar entrypoints principales",
                "Esto atrapa errores tontos antes de perseguir fantasmas.",
                _entrypoint_commands(project_path, report),
                "Sin errores de sintaxis en entrypoints."
            ),
        ])
    return steps


def create_task_plan(objective: str, project: str | None = None, root: Path | None = None) -> TaskPlan:
    root = root or _root()
    objective = objective.strip() or "mejorar proyecto"
    project = project or _detect_project(objective, root)
    project_path = _resolve_project_path(project, root)
    report = _inspect(project, root)

    assumptions = [
        "El plan solo propone pasos; no modifica archivos por sí mismo.",
        "Se asume que las rutas guardadas en memoria local son correctas.",
        "Si el proyecto no existe o no está actualizado, primero hay que corregir la ruta.",
    ]

    if not project_path.exists():
        assumptions.append(f"La ruta del proyecto no existe: {project_path}")

    base_steps = [
        _step(
            1,
            "Confirmar ruta y estado del proyecto",
            "No se puede diagnosticar bien una carpeta que Crotolamo no puede leer.",
            [
                "python tools/crotolamo_project_index_cli.py proyectos",
                f"python tools/crotolamo_project_index_cli.py mapa {shlex.quote(project)}",
            ],
            "Ruta confirmada y mapa básico disponible."
        ),
        _step(
            2,
            "Reindexar proyecto",
            "El índice debe estar fresco antes de tomar decisiones.",
            [
                f"python tools/crotolamo_project_index_cli.py indexar {shlex.quote(project)}",
            ],
            "Índice actualizado en data/project_index/."
        ),
        _step(
            3,
            "Inspeccionar estructura y dependencias",
            "Esto detecta entrypoints, imports externos, TODOs y archivos grandes.",
            [
                f"python tools/crotolamo_project_inspector_cli.py inspeccionar {shlex.quote(project)}",
            ],
            "Reporte guardado en data/project_reports/."
        ),
    ]

    specific_steps = _project_specific_steps(project, objective, project_path, report)

    final_step_id = len(base_steps) + len(specific_steps) + 1
    final_steps = [
        _step(
            final_step_id,
            "Elegir cambio mínimo y crear checkpoint",
            "Antes de editar, hay que decidir el cambio más pequeño que mejora el proyecto y guardar respaldo.",
            [
                "git status --short",
                "git diff --stat",
            ],
            "Estado limpio o cambios identificados antes de modificar."
        ),
        _step(
            final_step_id + 1,
            "Aplicar cambios en una rama o backup",
            "Modificar directo sobre la carpeta principal sin control es invocar al payaso del caos.",
            [
                "git checkout -b mejora-crotolamo-planificada",
            ],
            "Rama de trabajo creada si el proyecto usa git."
        ),
    ]

    steps = base_steps + specific_steps + final_steps

    risk_summary = "Bajo/medio: el plan principalmente lee archivos. Los pasos con git o cambios requieren confirmación."
    if not project_path.exists():
        risk_summary = "Alto por incertidumbre: la ruta del proyecto no existe o no está accesible."

    validation = [
        "Correr el doctor correspondiente si existe.",
        "Compilar archivos Python principales con `python -m py_compile ...`.",
        "Abrir la UI o launcher principal después de cambios.",
        "Comparar reporte antes/después si se modificó estructura.",
    ]

    rollback = [
        "No aplicar comandos de modificación sin revisar `git status`.",
        "Si se creó rama git, volver con `git checkout main` o la rama anterior.",
        "Si hay backup en `backups/`, restaurar solo el archivo afectado.",
        "No usar `rm -rf` como rollback. Eso no es rollback, es crimen digital."
    ]

    plan = TaskPlan(
        title=f"Plan v12: {objective[:80]}",
        project=project,
        objective=objective,
        created_at=datetime.now().isoformat(timespec="seconds"),
        risk_summary=risk_summary,
        assumptions=assumptions,
        steps=steps,
        validation=validation,
        rollback=rollback,
        next_action=f"Ejecuta primero: python tools/crotolamo_project_inspector_cli.py inspeccionar {shlex.quote(project)}",
    )
    save_plan(plan, root)
    return plan


def save_plan(plan: TaskPlan, root: Path | None = None) -> Path:
    p = _plans_dir(root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(plan.project)}.json"
    p.write_text(json.dumps(asdict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def format_plan(plan: TaskPlan) -> str:
    lines = [
        plan.title,
        "=" * len(plan.title),
        f"Proyecto: {plan.project}",
        f"Objetivo: {plan.objective}",
        f"Creado: {plan.created_at}",
        f"Riesgo: {plan.risk_summary}",
        "",
        "Supuestos:",
    ]
    for item in plan.assumptions:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("Pasos:")
    for step in plan.steps:
        lines.append("")
        lines.append(f"{step.id}. {step.title} [{step.risk.upper()}]")
        lines.append(f"   Por qué: {step.why}")
        if step.commands:
            lines.append("   Comandos:")
            for cmd in step.commands:
                lines.append(f"   $ {cmd}")
        if step.expected_result:
            lines.append(f"   Resultado esperado: {step.expected_result}")

    lines.append("")
    lines.append("Validación:")
    for item in plan.validation:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("Rollback:")
    for item in plan.rollback:
        lines.append(f"- {item}")

    lines.append("")
    lines.append(f"Siguiente acción: {plan.next_action}")
    return "\n".join(lines)


def list_recent_plans(root: Path | None = None, limit: int = 10) -> str:
    d = _plans_dir(root)
    files = sorted(d.glob("*.json"), reverse=True)[:limit]
    if not files:
        return "No hay planes guardados todavía."
    lines = ["Planes recientes:"]
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            lines.append(f"- {f.name}: {data.get('title')} [{data.get('project')}]")
        except Exception:
            lines.append(f"- {f.name}")
    return "\n".join(lines)


def handle_task_planner_command(text: str, root: Path | None = None) -> str | None:
    raw = (text or "").strip()
    low = raw.lower()

    if low in {"planes", "ver planes", "planes recientes"}:
        return list_recent_plans(root)

    if low in {"plan", "planificar", "crear plan"}:
        return "Formato: plan <objetivo>\nEjemplo: plan mejorar extractor de huevonitis"

    prefixes = ("plan ", "planificar ", "crear plan ", "hacer plan ", "planea ", "planear ")
    for prefix in prefixes:
        if low.startswith(prefix):
            objective = raw[len(prefix):].strip()
            if not objective:
                return "Formato: plan <objetivo>"
            plan = create_task_plan(objective, root=root)
            return format_plan(plan)

    return None
