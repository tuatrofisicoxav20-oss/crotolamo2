from pathlib import Path
from typing import Any
from core.patch_builder import handle_patch_command, list_patches
PLUGIN_NAME="patch_builder"; PLUGIN_DESCRIPTION="Parches con diff, backup y preview."
ACTIONS={"patch.list":{"name":"patch.list","mode":"general","description":"Lista parches.","triggers":["parches"]}}
def can_handle(text:str)->bool:
    low=(text or "").strip().lower()
    return low in {"patch","parche","parches","ver parches","patches"} or low.startswith(("patch ","ver parche","aplicar parche"))
def run(text:str, root:Path|None=None)->str: return handle_patch_command(text,root) or "No entendí ese comando de parches."
def get_actions()->dict[str,dict[str,Any]]: return ACTIONS
def run_action(action_name:str, root:Path|None=None, **kwargs:Any)->str:
    return list_patches(root) if action_name=="patch.list" else f"Acción patch no reconocida: {action_name}"
