from __future__ import annotations
import json, subprocess, time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class TestResult:
    name:str; command:str; ok:bool; returncode:int; stdout:str; stderr:str; seconds:float
@dataclass
class TestReport:
    id:str; created_at:str; target:str; ok:bool; results:list[TestResult]

def _root(): return Path(__file__).resolve().parents[1]
def _dir(root=None):
    p=(root or _root())/"data"/"test_reports"; p.mkdir(parents=True, exist_ok=True); return p
def _save(rep,root=None):
    p=_dir(root)/f"{rep.id}.json"; p.write_text(json.dumps(asdict(rep),indent=2,ensure_ascii=False),encoding="utf-8"); return p

def run_cmd(cmd,cwd,timeout=45,name=None):
    t=time.time()
    try:
        pr=subprocess.run(cmd,cwd=str(cwd),text=True,capture_output=True,timeout=timeout)
        return TestResult(name or " ".join(cmd)," ".join(cmd),pr.returncode==0,pr.returncode,pr.stdout[-6000:],pr.stderr[-6000:],round(time.time()-t,3))
    except subprocess.TimeoutExpired:
        return TestResult(name or " ".join(cmd)," ".join(cmd),False,124,"",f"Timeout después de {timeout}s",round(time.time()-t,3))
    except Exception as e:
        return TestResult(name or " ".join(cmd)," ".join(cmd),False,1,"",f"{type(e).__name__}: {e}",round(time.time()-t,3))

def _project(name,root):
    try:
        from core.project_indexer import resolve_project
        return resolve_project(name or "crotolamo",root)
    except Exception:
        return name or "crotolamo",root

def _entrypoints(project,root):
    pname,ppath=_project(project,root); entries=[]
    try:
        from core.project_inspector import inspect_project
        entries=list(getattr(inspect_project(pname,root),"likely_entrypoints",[]) or [])
    except Exception: pass
    if not entries:
        for c in ("main.py","app.py","run.py","launch_runtime_shell.py","launch_orbital_ui.py"):
            if (ppath/c).exists(): entries.append(c)
    return pname,ppath,entries[:12]

def run_project_tests(project=None,root=None):
    root=root or _root(); pname,ppath,entries=_entrypoints(project or "crotolamo",root); res=[]
    if not ppath.exists():
        res.append(TestResult("ruta_proyecto",str(ppath),False,1,"",f"No existe: {ppath}",0))
    else:
        for rel in entries:
            f=ppath/rel
            if f.suffix==".py" and f.exists():
                res.append(run_cmd(["python","-m","py_compile",str(f)],ppath,name=f"py_compile {rel}"))
    if pname.lower() in {"crotolamo","actual","raiz","raíz"} or ppath==root:
        for d in ["tools/crotolamo_doctor.py","tools/crotolamo_doctor_v10_extra.py","tools/crotolamo_doctor_v11_extra.py","tools/crotolamo_doctor_v12_extra.py","tools/crotolamo_doctor_v13_v15_extra.py"]:
            if (root/d).exists(): res.append(run_cmd(["python",d],root,60,d))
    rep=TestReport(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pname}_tests",datetime.now().isoformat(timespec="seconds"),pname,all(r.ok for r in res) if res else False,res)
    _save(rep,root); return rep

def run_patch_tests(patch_id=None,root=None):
    root=root or _root(); res=[]; target=patch_id or "latest"
    try:
        from core.patch_builder import load_proposal
        prop=load_proposal(target,root)
    except Exception as e:
        prop=None; res.append(TestResult("load_patch",str(target),False,1,"",f"{type(e).__name__}: {e}",0))
    if prop:
        ppath=Path(prop.project_root)
        for ch in prop.changes:
            f=ppath/ch.relpath
            if f.suffix==".py" and f.exists():
                res.append(run_cmd(["python","-m","py_compile",str(f)],ppath,name=f"py_compile {ch.relpath}"))
        if not prop.changes: res.append(TestResult("patch_changes",prop.id,True,0,"Patch sin cambios aplicables.","",0))
    rep=TestReport(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_patch_tests",datetime.now().isoformat(timespec="seconds"),f"patch:{target}",all(r.ok for r in res) if res else False,res)
    _save(rep,root); return rep

def format_report(rep):
    lines=[f"Test Report: {rep.id}",f"Target: {rep.target}",f"OK: {rep.ok}",f"Creado: {rep.created_at}","","Resultados:"]
    for r in rep.results:
        lines += [f"- {'OK' if r.ok else 'NO'} {r.name} ({r.seconds}s)",f"  $ {r.command}"]
        if r.stdout.strip():
            lines.append("  stdout:"); lines += [f"    {x}" for x in r.stdout.strip().splitlines()[-12:]]
        if r.stderr.strip():
            lines.append("  stderr:"); lines += [f"    {x}" for x in r.stderr.strip().splitlines()[-12:]]
    return "\n".join(lines)

def handle_test_command(text,root=None):
    raw=(text or "").strip(); low=raw.lower()
    if low in {"test","tests","pruebas","probar"}: return format_report(run_project_tests("crotolamo",root))
    if low.startswith("test parche") or low.startswith("probar parche"):
        bits=raw.split(); return format_report(run_patch_tests(bits[-1] if len(bits)>=3 else "latest",root))
    for p in ("test ","tests ","probar ","pruebas "):
        if low.startswith(p):
            t=raw[len(p):].strip()
            if t.lower().startswith("parche"):
                bits=t.split(); return format_report(run_patch_tests(bits[-1] if len(bits)>1 else "latest",root))
            return format_report(run_project_tests(t,root))
    return None
