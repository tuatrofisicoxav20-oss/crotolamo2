"""Atajos aprendidos por el patrón. Migrado de C1::learn_shortcut, ahora en SQLite.

El patrón enseña un alias ("mi música" -> spotify) y luego lo dispara. Se resuelve
el destino a una acción (url/app/project/folder/search) y se ejecuta vía las tools
de desktop/search ya existentes.
"""

from __future__ import annotations

from crotolamo.persistence import db
from crotolamo.tools.base import normalize_key, tool
from crotolamo.tools.desktop import open_app, open_folder, open_url, resolve_target
from crotolamo.tools.search import search_web

# La clasificación del destino vive en desktop.resolve_target (dueño de los
# catálogos de apps/sitios/carpetas). Alias local por compat con código viejo.
_classify_target = resolve_target


@tool
def learn_shortcut(alias: str, target: str) -> str:
    """Aprende un atajo: asocia un alias a algo que abrir o buscar.

    Args:
        alias: la frase corta que el patrón dirá luego.
        target: qué abrir/buscar (un sitio, app, carpeta, URL o término).
    """
    alias_key = normalize_key(alias)
    if not alias_key or not target.strip():
        return "Me faltó el alias o el destino, patrón."
    action_type, payload = resolve_target(target)
    db.save_shortcut(alias_key, action_type, payload)
    return f"Aprendí el atajo '{alias}', patrón. Ya puedo dispararlo."


@tool
def run_shortcut(alias: str) -> str:
    """Ejecuta un atajo aprendido por su alias.

    Args:
        alias: el alias del atajo a ejecutar.
    """
    action = db.get_shortcut(normalize_key(alias))
    if action is None:
        return f"No tengo un atajo llamado '{alias}', patrón."

    kind = action.get("type")
    if kind == "url":
        return open_url(action.get("value", ""))
    if kind == "app":
        return open_app(action.get("value", ""))
    if kind == "folder":
        return open_folder(action.get("value", ""))
    if kind == "search":
        return search_web(action.get("query", ""), action.get("engine", "google"))
    return f"El atajo '{alias}' está mal guardado, patrón."


@tool
def list_shortcuts() -> str:
    """Lista los atajos que el patrón ha enseñado."""
    shortcuts = db.all_shortcuts()
    if not shortcuts:
        return "No tienes atajos guardados, patrón."
    lines = []
    for alias, action in shortcuts.items():
        detail = action.get("value") or action.get("query") or ""
        lines.append(f"- {alias} -> {action.get('type')}: {detail}")
    return "Tus atajos, patrón:\n" + "\n".join(lines)
