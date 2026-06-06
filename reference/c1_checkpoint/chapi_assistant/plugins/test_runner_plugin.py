from pathlib import Path
from typing import Any
from core.test_runner import handle_test_command, run_project_tests, format_report
PLUGIN_NAME="test_runner"; PLUGIN_DESCRIPTION="Pruebas seguras y py_compile."
ACTIONS={"test.run":{"name":"test.run","mode":"general","description":"Corre pruebas seguras.","triggers":["test","pruebas"]}}
def can_handle(text:str)->bool:
    low=(text or "").strip().lower()
    return low in {"test","tests","pruebas","probar"} or low.startswith(("test ","tests ","probar ","pruebas "))
def run(text:str, root:Path|None=None)->str: return handle_test_command(text,root) or "No entendí ese comando de pruebas."
def get_actions()->dict[str,dict[str,Any]]: return ACTIONS
def run_action(action_name:str, root:Path|None=None, **kwargs:Any)->str:
    return format_report(run_project_tests(kwargs.get("project") or "crotolamo",root)) if action_name=="test.run" else f"Acción test no reconocida: {action_name}"
