from __future__ import annotations
import json, subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

BLOCK=["rm -rf /","rm -rf ~","sudo rm","mkfs","dd if=","dd of=",":(){","chmod -r 777 /","chown -r","curl | bash","wget | bash","shutdown","reboot","poweroff","systemctl disable","dnf remove","parted","fdisk","wipefs"]
CONFIRM=["sudo ","dnf install","pip install","python -m pip install","chmod ","chown ","mv ","cp ","git checkout","git reset","git clean"]
SAFE=["ls","pwd","find","grep","rg","cat","head","tail","tree","python -m py_compile","python tools/","git status","git diff","git log","ollama ps"]

@dataclass
class ExecResult:
    command:str; cwd:str; level:str; ok:bool; returncode:int; stdout:str; stderr:str; created_at:str

def _root(): return Path(__file__).resolve().parents[1]
def _logs(root=None):
    p=(root or _root())/"data"/"exec_logs"; p.mkdir(parents=True,exist_ok=True); return p

def classify(cmd):
    low=cmd.strip().lower()
    if not low: return "blocked"
    if any(x in low for x in BLOCK): return "blocked"
    try:
        from core.command_safety import classify_command
        r=classify_command(cmd)
        v=(r if isinstance(r,str) else (r.get("level") if isinstance(r,dict) else getattr(r,"level",getattr(r,"risk",""))))
        v=str(v).lower()
        if "block" in v: return "blocked"
        if "confirm" in v or "medium" in v or "risk" in v: return "confirm"
        if "safe" in v: return "safe"
    except Exception: pass
    if any(low.startswith(x) for x in SAFE): return "safe"
    if any(x in low for x in CONFIRM): return "confirm"
    return "confirm"

def _save(res,root=None):
    p=_logs(root)/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_exec.json"
    p.write_text(json.dumps(asdict(res),indent=2,ensure_ascii=False),encoding="utf-8"); return p

def execute_command(cmd,root=None,confirm=False,timeout=60,cwd=None):
    root=root or _root(); cwd=cwd or root; level=classify(cmd)
    if level=="blocked":
        r=ExecResult(cmd,str(cwd),level,False,126,"","Comando bloqueado por seguridad.",datetime.now().isoformat(timespec="seconds")); _save(r,root); return r
    if level=="confirm" and not confirm:
        r=ExecResult(cmd,str(cwd),level,False,125,"","Requiere confirmación: ejecutar comando EJECUTAR -- <cmd>",datetime.now().isoformat(timespec="seconds")); _save(r,root); return r
    try:
        pr=subprocess.run(cmd,shell=True,cwd=str(cwd),text=True,capture_output=True,timeout=timeout)
        r=ExecResult(cmd,str(cwd),level,pr.returncode==0,pr.returncode,pr.stdout[-8000:],pr.stderr[-8000:],datetime.now().isoformat(timespec="seconds"))
    except subprocess.TimeoutExpired:
        r=ExecResult(cmd,str(cwd),level,False,124,"",f"Timeout después de {timeout}s",datetime.now().isoformat(timespec="seconds"))
    except Exception as e:
        r=ExecResult(cmd,str(cwd),level,False,1,"",f"{type(e).__name__}: {e}",datetime.now().isoformat(timespec="seconds"))
    _save(r,root); return r

def format_exec_result(r):
    out=[f"Exec [{r.level.upper()}] {'OK' if r.ok else 'NO'}",f"$ {r.command}",f"CWD: {r.cwd}",f"Return code: {r.returncode}"]
    if r.stdout.strip(): out += ["","stdout:",r.stdout.strip()]
    if r.stderr.strip(): out += ["","stderr:",r.stderr.strip()]
    return "\n".join(out)

def handle_executor_command(text,root=None):
    raw=(text or "").strip(); low=raw.lower()
    if low in {"executor","safe executor","ejecutor"}:
        return "Safe Executor v15:\n- comando <cmd>\n- ejecutar comando <cmd>\n- ejecutar comando EJECUTAR -- <cmd>\n- aplicar parche EJECUTAR latest"
    if low.startswith("comando "):
        cmd=raw[len("comando "):].strip(); return f"Clasificación: {classify(cmd).upper()}\n$ {cmd}"
    if low.startswith("ejecutar comando "):
        body=raw[len("ejecutar comando "):].strip(); confirm=False
        if body.startswith("EJECUTAR -- "): confirm=True; body=body[len("EJECUTAR -- "):].strip()
        return format_exec_result(execute_command(body,root,confirm))
    if low.startswith("aplicar parche"):
        try:
            from core.patch_builder import apply_patch
        except Exception as e: return f"No está disponible patch_builder: {e}"
        bits=raw.split(); confirm="EJECUTAR" in bits; pid="latest" if (not bits or bits[-1]=="EJECUTAR") else bits[-1]
        return apply_patch(pid,root,confirm)
    return None
