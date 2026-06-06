from pathlib import Path
from typing import Any
from core.safe_executor import handle_executor_command
PLUGIN_NAME="safe_executor"; PLUGIN_DESCRIPTION="Ejecutor seguro con bloqueo y confirmación."
ACTIONS={"executor.info":{"name":"executor.info","mode":"general","description":"Ayuda del ejecutor.","triggers":["executor"]}}
def can_handle(text:str)->bool:
    low=(text or "").strip().lower()
    return low in {"executor","safe executor","ejecutor"} or low.startswith(("comando ","ejecutar comando ","aplicar parche"))
def run(text:str, root:Path|None=None)->str: return handle_executor_command(text,root) or "No entendí ese comando del ejecutor."
def get_actions()->dict[str,dict[str,Any]]: return ACTIONS
def run_action(action_name:str, root:Path|None=None, **kwargs:Any)->str: return handle_executor_command("executor",root) or "Executor disponible."
