from __future__ import annotations
import difflib, json, os, re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

TEXT_EXTS={".py",".txt",".md",".json",".yaml",".yml",".toml",".ini",".cfg",".sh",".bash",".zsh",".html",".css",".js",".ts",".tsx",".jsx",".xml",".csv",".rst"}

@dataclass
class PatchChange:
    relpath:str
    old_text:str
    new_text:str
    description:str=""

@dataclass
class PatchProposal:
    id:str
    title:str
    project:str
    project_root:str
    created_at:str
    status:str
    objective:str
    changes:list[PatchChange]
    notes:list[str]
    diff:str

def _root(): return Path(__file__).resolve().parents[1]
def _patch_dir(root=None):
    p=(root or _root())/"data"/"patches"; p.mkdir(parents=True, exist_ok=True); return p
def _backup_dir(root=None):
    p=(root or _root())/"backups"/"patches"; p.mkdir(parents=True, exist_ok=True); return p
def _slug(s): return re.sub(r"[^a-zA-Z0-9_-]+","_",s.strip().lower()).strip("_")[:80] or "patch"

def _resolve_project(project=None, root=None):
    root=root or _root()
    try:
        from core.project_indexer import resolve_project
        return resolve_project(project or "crotolamo", root)
    except Exception:
        return project or "crotolamo", root

def _inside(base:Path,target:Path):
    base=base.resolve(); target=target.resolve()
    if base!=target and base not in target.parents:
        raise ValueError(f"Ruta fuera del proyecto: {target}")

def _read(path:Path):
    if path.suffix.lower() not in TEXT_EXTS:
        raise ValueError(f"Extensión no permitida: {path.suffix}")
    if path.exists() and path.stat().st_size>2_000_000:
        raise ValueError("Archivo demasiado grande para parche seguro.")
    return path.read_text(encoding="utf-8",errors="replace") if path.exists() else ""

def _diff(rel,old,new):
    return "".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=f"a/{rel}",tofile=f"b/{rel}",lineterm=""))

def save_proposal(prop:PatchProposal, root=None):
    p=_patch_dir(root)/f"{prop.id}.json"
    p.write_text(json.dumps(asdict(prop),indent=2,ensure_ascii=False),encoding="utf-8")
    return p

def load_proposal(patch_id=None, root=None):
    root=root or _root()
    if not patch_id or patch_id in {"latest","ultimo","último"}:
        files=sorted(_patch_dir(root).glob("*.json"), reverse=True)
        if not files: return None
        path=files[0]
    else:
        pid=patch_id.removesuffix(".json")
        path=_patch_dir(root)/f"{pid}.json"
        if not path.exists():
            ms=sorted(_patch_dir(root).glob(f"{pid}*.json"), reverse=True)
            if not ms: return None
            path=ms[0]
    data=json.loads(path.read_text(encoding="utf-8"))
    data["changes"]=[PatchChange(**c) for c in data.get("changes",[])]
    return PatchProposal(**data)

def list_patches(root=None,limit=15):
    files=sorted(_patch_dir(root).glob("*.json"), reverse=True)[:limit]
    if not files: return "No hay parches guardados todavía."
    out=["Parches recientes:"]
    for f in files:
        try:
            d=json.loads(f.read_text(encoding="utf-8"))
            out.append(f"- {d.get('id')} [{d.get('status')}] {d.get('title')} ({d.get('project')})")
        except Exception:
            out.append(f"- {f.name}")
    return "\n".join(out)

def create_replace_patch(relpath, old_fragment, new_fragment, project=None, objective=None, root=None):
    root=root or _root(); pname,proot=_resolve_project(project,root)
    target=(proot/relpath).resolve(); _inside(proot,target)
    original=_read(target)
    if old_fragment not in original:
        raise ValueError("No encontré el fragmento exacto en el archivo. No hago reemplazos a ciegas.")
    updated=original.replace(old_fragment,new_fragment,1)
    pid=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(Path(relpath).name)}"
    prop=PatchProposal(pid,f"Replace en {relpath}",pname,str(proot),datetime.now().isoformat(timespec="seconds"),"preview",objective or f"Reemplazar fragmento en {relpath}",[PatchChange(relpath,original,updated,"replace")],["Preview creado.","No se aplicó ningún cambio.","Revisa el diff antes de aplicar."],_diff(relpath,original,updated))
    save_proposal(prop,root); return prop

def create_empty_planning_patch(objective, project=None, root=None):
    root=root or _root(); pname,proot=_resolve_project(project,root)
    notes=["Propuesta sin cambios todavía.","Para cambio real usa: patch replace <archivo> ::: <actual> ::: <nuevo>"]
    try:
        from core.project_inspector import inspect_project
        r=inspect_project(pname,root)
        e=getattr(r,"likely_entrypoints",[]) or []
        imp=getattr(r,"important_python_files",[]) or []
        if e: notes.append("Entrypoints: "+", ".join(e[:5]))
        if imp: notes.append("Archivos candidatos: "+", ".join(imp[:8]))
    except Exception as e:
        notes.append(f"Inspector no disponible: {e}")
    pid=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(pname)}_plan"
    prop=PatchProposal(pid,f"Propuesta: {objective[:80]}",pname,str(proot),datetime.now().isoformat(timespec="seconds"),"preview",objective,[],notes,"")
    save_proposal(prop,root); return prop

def format_proposal(prop):
    lines=[f"Patch: {prop.id}",f"Título: {prop.title}",f"Proyecto: {prop.project}",f"Estado: {prop.status}",f"Objetivo: {prop.objective}",f"Ruta: {prop.project_root}","","Notas:"]
    lines += [f"- {n}" for n in prop.notes]
    lines += ["",f"Cambios: {len(prop.changes)}"]
    lines += [f"- {c.relpath}: {c.description or 'cambio textual'}" for c in prop.changes]
    lines += ["","Diff:", prop.diff[:12000] if prop.diff else "vacío. Este parche todavía no modifica archivos."]
    return "\n".join(lines)

def apply_patch(patch_id=None, root=None, confirm=False):
    root=root or _root()
    if not confirm: return "No apliqué nada. Para aplicar usa: aplicar parche EJECUTAR <id|latest>"
    prop=load_proposal(patch_id,root)
    if not prop: return "No encontré ese parche."
    if not prop.changes: return "El parche no tiene cambios aplicables. Es preview/planeación."
    proot=Path(prop.project_root).resolve()
    broot=_backup_dir(root)/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prop.id}"; broot.mkdir(parents=True,exist_ok=True)
    applied=[]
    for ch in prop.changes:
        target=(proot/ch.relpath).resolve(); _inside(proot,target)
        current=_read(target)
        if current!=ch.old_text:
            return f"ABORTADO: {ch.relpath} ya cambió respecto al parche. No aplico sobre archivo modificado."
        b=broot/ch.relpath; b.parent.mkdir(parents=True,exist_ok=True); b.write_text(current,encoding="utf-8")
        target.write_text(ch.new_text,encoding="utf-8"); applied.append(ch.relpath)
    prop.status="applied"; prop.notes.append(f"Aplicado con backup en: {broot}"); save_proposal(prop,root)
    return "Parche aplicado.\nBackup: {}\nArchivos:\n{}".format(broot,"\n".join(f"- {x}" for x in applied))

def handle_patch_command(text, root=None):
    raw=(text or "").strip(); low=raw.lower()
    if low in {"parches","ver parches","patches"}: return list_patches(root)
    if low in {"patch","parche"}:
        return "Comandos:\n- patch <objetivo>\n- patch replace <archivo> ::: <texto_actual> ::: <texto_nuevo>\n- ver parche latest\n- aplicar parche EJECUTAR latest"
    if low.startswith("patch replace "):
        parts=raw[len("patch replace "):].split(":::")
        if len(parts)!=3: return "Formato: patch replace <archivo> ::: <texto_actual> ::: <texto_nuevo>"
        try: return format_proposal(create_replace_patch(*(p.strip() for p in parts), root=root))
        except Exception as e: return f"No pude crear el parche: {type(e).__name__}: {e}"
    if low.startswith("patch "): return format_proposal(create_empty_planning_patch(raw[len("patch "):].strip(), root=root))
    if low.startswith("ver parche"):
        bits=raw.split(); pid=bits[-1] if len(bits)>=3 else "latest"
        p=load_proposal(pid,root); return format_proposal(p) if p else "No encontré ese parche."
    if low.startswith("aplicar parche"):
        bits=raw.split(); confirm="EJECUTAR" in bits
        pid="latest" if (not bits or bits[-1]=="EJECUTAR") else bits[-1]
        return apply_patch(pid,root,confirm)
    return None
