"""
Crotolamo Project Indexer v10.

Indexa proyectos locales sin dependencias raras:
- mapa de carpetas
- archivos principales
- búsqueda por nombre
- búsqueda ligera por texto
- resumen por extensiones
- detección simple de candidatos importantes

No analiza como humano iluminado, pero al menos no finge que sabe qué hay en una carpeta
sin haberla leído. Progreso civilizatorio.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", "dist", "build", ".tox",
    "backups", "backup", ".backup",
    ".idea", ".vscode",
    "data/history",
}

TEXT_EXTS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
    ".xml", ".csv", ".log", ".rst",
}

IMPORTANT_NAMES = {
    "main.py", "app.py", "run.py", "launch.py", "launcher.py",
    "requirements.txt", "pyproject.toml", "setup.py", "README.md", "readme.md",
    "config.json", "settings.json", "package.json",
}


@dataclass
class IndexedFile:
    relpath: str
    size: int
    ext: str
    mtime: str
    kind: str
    important: bool = False


@dataclass
class ProjectIndex:
    name: str
    root: str
    created_at: str
    total_files: int
    total_dirs: int
    indexed_files: list[IndexedFile]
    skipped_files: int
    ext_counts: dict[str, int]
    important_files: list[str]
    warnings: list[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _index_dir(root: Path | None = None) -> Path:
    root = root or _project_root()
    p = root / "data" / "project_index"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return cleaned or "proyecto"


def _load_memory(root: Path | None = None) -> dict[str, Any]:
    root = root or _project_root()
    try:
        from core.local_memory import load_memory
        return load_memory(root)
    except Exception:
        return {}


def known_projects(root: Path | None = None) -> dict[str, str]:
    root = root or _project_root()
    memory = _load_memory(root)
    paths = dict(memory.get("project_paths", {}) or {})
    paths.setdefault("crotolamo", str(root))
    return paths


def resolve_project(name: str | None = None, root: Path | None = None) -> tuple[str, Path]:
    root = root or _project_root()
    projects = known_projects(root)
    key = (name or "").strip().lower()

    if not key or key in {"actual", "current", "crotolamo", "raiz", "raíz"}:
        return "crotolamo", root

    # match exact
    for pname, ppath in projects.items():
        if pname.lower() == key:
            return pname, Path(os.path.expanduser(str(ppath))).resolve()

    # fuzzy contains
    for pname, ppath in projects.items():
        if key in pname.lower() or pname.lower() in key:
            return pname, Path(os.path.expanduser(str(ppath))).resolve()

    # direct path
    candidate = Path(os.path.expanduser(name or ""))
    if candidate.exists():
        return candidate.name, candidate.resolve()

    # fallback: project key not found
    return key, Path(os.path.expanduser(projects.get(key, str(root)))).resolve()


def _is_ignored_dir(path: Path, base: Path) -> bool:
    try:
        rel_parts = path.relative_to(base).parts
    except Exception:
        rel_parts = path.parts
    joined = "/".join(rel_parts)
    if joined in IGNORE_DIRS:
        return True
    return any(part in IGNORE_DIRS for part in rel_parts)


def _kind_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".md", ".txt", ".rst"}:
        return "docs"
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "asset"
    if ext in {".sh", ".bash", ".zsh"}:
        return "script"
    if ext in {".log"}:
        return "log"
    return ext[1:] if ext else "sin_ext"


def scan_project(
    project_name: str | None = None,
    root: Path | None = None,
    max_files: int = 2500,
    max_depth: int = 8,
) -> ProjectIndex:
    runtime_root = root or _project_root()
    name, project_path = resolve_project(project_name, runtime_root)
    warnings: list[str] = []

    if not project_path.exists():
        warnings.append(f"No existe la ruta: {project_path}")
        return ProjectIndex(
            name=name,
            root=str(project_path),
            created_at=datetime.now().isoformat(timespec="seconds"),
            total_files=0,
            total_dirs=0,
            indexed_files=[],
            skipped_files=0,
            ext_counts={},
            important_files=[],
            warnings=warnings,
        )

    indexed: list[IndexedFile] = []
    ext_counts: dict[str, int] = {}
    important_files: list[str] = []
    total_dirs = 0
    skipped = 0

    base_depth = len(project_path.parts)

    for dirpath, dirnames, filenames in os.walk(project_path):
        d = Path(dirpath)

        # prune ignored dirs
        dirnames[:] = [
            x for x in dirnames
            if not _is_ignored_dir(d / x, project_path)
        ]

        depth = len(d.parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        if _is_ignored_dir(d, project_path):
            continue

        total_dirs += 1

        for filename in filenames:
            if len(indexed) >= max_files:
                skipped += 1
                continue

            path = d / filename
            try:
                rel = str(path.relative_to(project_path))
                stat = path.stat()
            except Exception:
                skipped += 1
                continue

            if stat.st_size > 5_000_000:
                skipped += 1
                continue

            ext = path.suffix.lower() or "<sin_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            important = filename in IMPORTANT_NAMES or any(x in rel.lower() for x in [
                "main_window", "extractor", "gesture", "runtime", "plugin", "doctor", "launcher"
            ])
            if important:
                important_files.append(rel)

            indexed.append(IndexedFile(
                relpath=rel,
                size=int(stat.st_size),
                ext=ext,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                kind=_kind_for(path),
                important=important,
            ))

    idx = ProjectIndex(
        name=name,
        root=str(project_path),
        created_at=datetime.now().isoformat(timespec="seconds"),
        total_files=len(indexed),
        total_dirs=total_dirs,
        indexed_files=indexed,
        skipped_files=skipped,
        ext_counts=dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        important_files=important_files[:80],
        warnings=warnings,
    )
    save_index(idx, runtime_root)
    return idx


def save_index(index: ProjectIndex, root: Path | None = None) -> Path:
    d = _index_dir(root)
    path = d / f"{_slug(index.name)}.json"
    data = asdict(index)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_index(project_name: str | None = None, root: Path | None = None) -> dict[str, Any] | None:
    root = root or _project_root()
    name, _ = resolve_project(project_name, root)
    path = _index_dir(root) / f"{_slug(name)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_index(project_name: str | None = None, root: Path | None = None) -> dict[str, Any]:
    idx = load_index(project_name, root)
    if idx is None:
        idx_obj = scan_project(project_name, root)
        return asdict(idx_obj)
    return idx


def index_summary(project_name: str | None = None, root: Path | None = None) -> str:
    idx = _ensure_index(project_name, root)
    lines = [
        f"Mapa de proyecto: {idx.get('name')}",
        f"Ruta: {idx.get('root')}",
        f"Indexado: {idx.get('created_at')}",
        f"Archivos: {idx.get('total_files')} | Carpetas: {idx.get('total_dirs')} | Saltados: {idx.get('skipped_files')}",
    ]

    warnings = idx.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Advertencias:")
        for w in warnings:
            lines.append(f"- {w}")

    lines.append("")
    lines.append("Extensiones principales:")
    for ext, count in list((idx.get("ext_counts") or {}).items())[:12]:
        lines.append(f"- {ext}: {count}")

    important = idx.get("important_files") or []
    if important:
        lines.append("")
        lines.append("Archivos importantes/candidatos:")
        for rel in important[:25]:
            lines.append(f"- {rel}")

    return "\n".join(lines)


def search_files(query: str, project_name: str | None = None, root: Path | None = None, limit: int = 40) -> list[dict[str, Any]]:
    idx = _ensure_index(project_name, root)
    q = query.strip().lower()
    results = []
    for f in idx.get("indexed_files", []):
        rel = f.get("relpath", "")
        if q in rel.lower():
            results.append(f)
            if len(results) >= limit:
                break
    return results


def search_text(query: str, project_name: str | None = None, root: Path | None = None, limit: int = 30, max_file_bytes: int = 400_000) -> list[dict[str, Any]]:
    runtime_root = root or _project_root()
    name, project_path = resolve_project(project_name, runtime_root)
    idx = _ensure_index(name, runtime_root)
    q = query.strip()
    qlow = q.lower()
    results = []

    for f in idx.get("indexed_files", []):
        rel = f.get("relpath", "")
        path = project_path / rel
        ext = path.suffix.lower()
        if ext not in TEXT_EXTS:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            if qlow in line.lower():
                results.append({
                    "file": rel,
                    "line": lineno,
                    "preview": line.strip()[:220],
                })
                break

        if len(results) >= limit:
            break

    return results


def format_file_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No encontré archivos con ese criterio."
    lines = ["Archivos encontrados:"]
    for item in results:
        lines.append(f"- {item.get('relpath')}  ({item.get('kind')}, {item.get('size')} bytes)")
    return "\n".join(lines)


def format_text_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No encontré coincidencias de texto."
    lines = ["Coincidencias encontradas:"]
    for item in results:
        lines.append(f"- {item.get('file')}:{item.get('line')} → {item.get('preview')}")
    return "\n".join(lines)


def list_known_projects(root: Path | None = None) -> str:
    projects = known_projects(root)
    lines = ["Proyectos conocidos:"]
    for name, path in projects.items():
        expanded = Path(os.path.expanduser(str(path)))
        status = "OK" if expanded.exists() else "NO EXISTE"
        lines.append(f"- {name}: {path} [{status}]")
    return "\n".join(lines)


def handle_project_index_command(text: str, root: Path | None = None) -> str | None:
    raw = (text or "").strip()
    low = raw.lower()

    if low in {"proyectos", "rutas proyectos", "proyectos conocidos"}:
        return list_known_projects(root)

    if low in {"index", "índice", "indice", "indexar", "indexar proyecto"}:
        idx = scan_project("crotolamo", root)
        return index_summary(idx.name, root)

    if low.startswith("indexar "):
        name = raw.split(maxsplit=1)[1].strip()
        idx = scan_project(name, root)
        return index_summary(idx.name, root)

    if low in {"mapa", "mapa proyecto", "mapa crotolamo"}:
        return index_summary("crotolamo", root)

    if low.startswith("mapa "):
        name = raw.split(maxsplit=1)[1].strip()
        return index_summary(name, root)

    if low.startswith("buscar archivo "):
        body = raw[len("buscar archivo "):].strip()
        # formato opcional: buscar archivo extractor en huevonitis
        project = None
        query = body
        if " en " in body:
            query, project = body.rsplit(" en ", 1)
        return format_file_results(search_files(query, project, root))

    if low.startswith("buscar texto "):
        body = raw[len("buscar texto "):].strip()
        project = None
        query = body
        if " en " in body:
            query, project = body.rsplit(" en ", 1)
        return format_text_results(search_text(query, project, root))

    if low in {"mapa huevonitis", "indexar huevonitis"}:
        name = "huevonitis"
        if low.startswith("indexar"):
            scan_project(name, root)
        return index_summary(name, root)

    if low in {"mapa tletl", "indexar tletl"}:
        name = "tletl"
        if low.startswith("indexar"):
            scan_project(name, root)
        return index_summary(name, root)

    return None
