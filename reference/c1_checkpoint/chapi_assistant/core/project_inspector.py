"""
Crotolamo Project Inspector v11.

Usa el indexador v10 para hacer una inspección más inteligente de proyectos:
- entradas ejecutables probables
- archivos Python principales
- imports externos probables
- TODO/FIXME/HACK
- archivos grandes
- estructura de carpetas
- recomendaciones accionables

Por defecto solo lee. Porque ejecutar cosas a lo bruto es tradición humana, no arquitectura.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


PY_STDLIB_HINTS = {
    "os", "sys", "pathlib", "json", "re", "math", "time", "datetime", "typing",
    "dataclasses", "subprocess", "shutil", "logging", "threading", "queue",
    "urllib", "http", "socket", "sqlite3", "csv", "argparse", "functools",
    "itertools", "collections", "statistics", "random", "tempfile", "hashlib",
    "base64", "inspect", "importlib", "traceback", "platform", "tkinter",
    "wave", "audioop", "html", "xml", "email", "zipfile", "tarfile",
}

IMPORTANT_PATTERNS = [
    "main.py", "app.py", "run.py", "launch", "launcher",
    "window", "ui", "runtime", "plugin", "doctor", "extractor",
    "gesture", "camera", "dataset", "voice", "shell",
]

TODO_PATTERNS = ("TODO", "FIXME", "HACK", "XXX", "BUG", "PENDIENTE", "ARREGLAR")


@dataclass
class InspectionReport:
    project: str
    root: str
    created_at: str
    total_python_files: int
    likely_entrypoints: list[str]
    important_python_files: list[str]
    external_imports: list[str]
    todo_hits: list[dict[str, Any]]
    large_files: list[dict[str, Any]]
    top_dirs: list[dict[str, Any]]
    warnings: list[str]
    recommendations: list[str]


def _runtime_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reports_dir(root: Path | None = None) -> Path:
    root = root or _runtime_root()
    p = root / "data" / "project_reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_") or "proyecto"


def _get_index(project_name: str | None, root: Path | None = None) -> tuple[str, Path, dict[str, Any]]:
    root = root or _runtime_root()
    from core.project_indexer import resolve_project, load_index, scan_project

    name, project_path = resolve_project(project_name, root)
    idx = load_index(name, root)
    if idx is None:
        idx_obj = scan_project(name, root)
        idx = asdict(idx_obj)
    return name, project_path, idx


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".log"}


def _read_text_safe(path: Path, max_bytes: int = 600_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _python_files(idx: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in idx.get("indexed_files", []) if str(f.get("ext", "")).lower() == ".py"]


def detect_entrypoints(project_path: Path, py_files: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for f in py_files:
        rel = f.get("relpath", "")
        name = Path(rel).name.lower()
        lower_rel = rel.lower()
        path = project_path / rel

        score = 0
        if name in {"main.py", "app.py", "run.py"}:
            score += 5
        if any(p in lower_rel for p in ["launch", "launcher", "shell", "cli"]):
            score += 3

        text = _read_text_safe(path, max_bytes=500_000)
        if 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
            score += 4
        if "argparse" in text or "click." in text:
            score += 1

        if score:
            found.append((score, rel))

    found_sorted = [rel for score, rel in sorted(found, key=lambda x: (-x[0], x[1]))]
    return found_sorted[:30]


def detect_important_files(py_files: list[dict[str, Any]]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for f in py_files:
        rel = f.get("relpath", "")
        low = rel.lower()
        score = 0
        for pattern in IMPORTANT_PATTERNS:
            if pattern in low:
                score += 2
        if f.get("important"):
            score += 3
        if score:
            scored.append((score, rel))
    return [rel for score, rel in sorted(scored, key=lambda x: (-x[0], x[1]))[:50]]


def detect_external_imports(project_path: Path, py_files: list[dict[str, Any]]) -> list[str]:
    imports: set[str] = set()
    local_roots = {Path(f.get("relpath", "")).parts[0] for f in py_files if Path(f.get("relpath", "")).parts}

    for f in py_files[:800]:
        rel = f.get("relpath", "")
        path = project_path / rel
        text = _read_text_safe(path, max_bytes=400_000)
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except Exception:
            continue

        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod:
                        imports.add(mod)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    mod = node.module.split(".")[0]
                    imports.add(mod)

    external = []
    for mod in imports:
        if not mod:
            continue
        if mod in PY_STDLIB_HINTS:
            continue
        if mod in local_roots:
            continue
        if mod.startswith("_"):
            continue
        external.append(mod)

    return sorted(external)


def find_todos(project_path: Path, idx: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for f in idx.get("indexed_files", []):
        rel = f.get("relpath", "")
        path = project_path / rel
        if not _is_text_file(path):
            continue
        text = _read_text_safe(path, max_bytes=500_000)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(pat.lower() in line.lower() for pat in TODO_PATTERNS):
                hits.append({
                    "file": rel,
                    "line": lineno,
                    "text": line.strip()[:220],
                })
                if len(hits) >= limit:
                    return hits
    return hits


def find_large_files(idx: dict[str, Any], limit: int = 20, threshold: int = 500_000) -> list[dict[str, Any]]:
    large = []
    for f in idx.get("indexed_files", []):
        size = int(f.get("size", 0) or 0)
        if size >= threshold:
            large.append({
                "file": f.get("relpath"),
                "size": size,
                "kind": f.get("kind"),
            })
    return sorted(large, key=lambda x: -x["size"])[:limit]


def top_directories(idx: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for f in idx.get("indexed_files", []):
        rel = f.get("relpath", "")
        parts = Path(rel).parts
        top = parts[0] if len(parts) > 1 else "."
        counts[top] = counts.get(top, 0) + 1
    return [
        {"dir": k, "files": v}
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ]


def build_recommendations(report: InspectionReport) -> list[str]:
    recs: list[str] = []

    if not report.likely_entrypoints:
        recs.append("No detecté entrypoint claro. Conviene tener `main.py`, `app.py` o un launcher evidente.")
    else:
        recs.append(f"Entrypoint más probable: `{report.likely_entrypoints[0]}`. Úsalo como punto de prueba principal.")

    if report.total_python_files > 80:
        recs.append("Hay muchos archivos Python. Conviene separar núcleo, UI, plugins, tools y pruebas para no crear sopa modular.")

    if report.external_imports:
        recs.append("Genera o actualiza `requirements.txt` con las dependencias externas detectadas.")
    else:
        recs.append("No detecté muchas dependencias externas. Eso es bueno para portabilidad.")

    if report.todo_hits:
        recs.append("Hay TODO/FIXME/HACK pendientes. No todos son urgentes, pero conviene revisar los primeros antes de seguir metiendo features.")

    if report.large_files:
        recs.append("Hay archivos grandes. Revisa si son assets, logs o datasets y evita meterlos al flujo principal del código.")

    recs.append("Siguiente paso sensato: correr inspección, elegir entrypoint principal y luego compilar solo archivos críticos.")
    return recs


def inspect_project(project_name: str | None = None, root: Path | None = None) -> InspectionReport:
    root = root or _runtime_root()
    name, project_path, idx = _get_index(project_name, root)
    warnings: list[str] = []

    if idx.get("warnings"):
        warnings.extend(idx.get("warnings") or [])

    py_files = _python_files(idx)
    report = InspectionReport(
        project=name,
        root=str(project_path),
        created_at=datetime.now().isoformat(timespec="seconds"),
        total_python_files=len(py_files),
        likely_entrypoints=detect_entrypoints(project_path, py_files),
        important_python_files=detect_important_files(py_files),
        external_imports=detect_external_imports(project_path, py_files),
        todo_hits=find_todos(project_path, idx),
        large_files=find_large_files(idx),
        top_dirs=top_directories(idx),
        warnings=warnings,
        recommendations=[],
    )
    report.recommendations = build_recommendations(report)
    save_report(report, root)
    return report


def save_report(report: InspectionReport, root: Path | None = None) -> Path:
    p = _reports_dir(root) / f"{_slug(report.project)}_inspection.json"
    p.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def format_report(report: InspectionReport) -> str:
    lines = [
        f"Inspección de proyecto: {report.project}",
        f"Ruta: {report.root}",
        f"Fecha: {report.created_at}",
        f"Archivos Python: {report.total_python_files}",
    ]

    if report.warnings:
        lines.append("")
        lines.append("Advertencias:")
        for w in report.warnings:
            lines.append(f"- {w}")

    lines.append("")
    lines.append("Entrypoints probables:")
    if report.likely_entrypoints:
        for item in report.likely_entrypoints[:15]:
            lines.append(f"- {item}")
    else:
        lines.append("- No detecté entrypoints claros.")

    lines.append("")
    lines.append("Archivos Python importantes:")
    if report.important_python_files:
        for item in report.important_python_files[:25]:
            lines.append(f"- {item}")
    else:
        lines.append("- No detecté candidatos importantes por nombre.")

    lines.append("")
    lines.append("Dependencias externas probables:")
    if report.external_imports:
        lines.append(", ".join(report.external_imports[:80]))
    else:
        lines.append("No detecté dependencias externas claras.")

    lines.append("")
    lines.append("Carpetas principales:")
    for item in report.top_dirs[:12]:
        lines.append(f"- {item['dir']}: {item['files']} archivos")

    if report.todo_hits:
        lines.append("")
        lines.append("TODO/FIXME/HACK detectados:")
        for item in report.todo_hits[:20]:
            lines.append(f"- {item['file']}:{item['line']} → {item['text']}")

    if report.large_files:
        lines.append("")
        lines.append("Archivos grandes:")
        for item in report.large_files[:12]:
            lines.append(f"- {item['file']} ({item['size']} bytes, {item['kind']})")

    lines.append("")
    lines.append("Recomendaciones:")
    for rec in report.recommendations:
        lines.append(f"- {rec}")

    return "\n".join(lines)


def handle_project_inspector_command(text: str, root: Path | None = None) -> str | None:
    raw = (text or "").strip()
    low = raw.lower()

    triggers = ("inspeccionar", "auditar", "analizar proyecto")
    if low in {"inspeccionar", "auditar", "inspeccionar proyecto", "auditar proyecto"}:
        return format_report(inspect_project("crotolamo", root))

    for trigger in triggers:
        if low.startswith(trigger + " "):
            name = raw[len(trigger):].strip()
            return format_report(inspect_project(name, root))

    if low in {"inspeccionar huevonitis", "auditar huevonitis"}:
        return format_report(inspect_project("huevonitis", root))

    if low in {"inspeccionar tletl", "auditar tletl"}:
        return format_report(inspect_project("tletl", root))

    if low.startswith("dependencias "):
        name = raw[len("dependencias "):].strip()
        report = inspect_project(name, root)
        if report.external_imports:
            return "Dependencias externas probables:\n" + "\n".join(f"- {x}" for x in report.external_imports)
        return "No detecté dependencias externas claras."

    if low.startswith("entradas "):
        name = raw[len("entradas "):].strip()
        report = inspect_project(name, root)
        if report.likely_entrypoints:
            return "Entrypoints probables:\n" + "\n".join(f"- {x}" for x in report.likely_entrypoints)
        return "No detecté entrypoints claros."

    if low.startswith("pendientes "):
        name = raw[len("pendientes "):].strip()
        report = inspect_project(name, root)
        if report.todo_hits:
            return "Pendientes detectados:\n" + "\n".join(
                f"- {x['file']}:{x['line']} → {x['text']}" for x in report.todo_hits[:40]
            )
        return "No detecté TODO/FIXME/HACK."

    return None
