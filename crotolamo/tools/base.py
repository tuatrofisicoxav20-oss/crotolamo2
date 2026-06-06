"""Decorador @tool + inferencia automática de esquema JSON.

Una tool es una función Python tipada. El esquema que ve el LLM se genera desde
la firma (tipos) y el docstring (descripción). Así, agregar una capacidad nueva =
escribir una función con @tool, no un regex.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

# Mapeo de tipos Python -> tipos JSON-schema.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _split_doc(doc: str) -> tuple[str, dict[str, str]]:
    """Separa el docstring en (descripción, {param: descripción}).

    Convención: tras una línea 'Args:' (o 'Parámetros:'), cada línea
    'nombre: texto' documenta un parámetro.
    """
    if not doc:
        return "", {}

    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    arg_re = re.compile(r"^\s*([a-zA-Z_]\w*)\s*:\s*(.+)$")

    for ln in lines:
        if ln.strip().lower() in {"args:", "parámetros:", "parametros:"}:
            in_args = True
            continue
        if in_args:
            m = arg_re.match(ln)
            if m:
                params[m.group(1)] = m.group(2).strip()
            continue
        desc_lines.append(ln)

    return " ".join(d for d in desc_lines if d).strip(), params


@dataclass
class Tool:
    name: str
    func: Callable[..., str]
    description: str
    parameters: dict[str, Any]  # JSON-schema del objeto de argumentos
    # safe=True: ejecuta directo. safe=False: el guard pedirá confirmación.
    safe: bool = True

    def schema(self) -> dict[str, Any]:
        """Formato que espera Ollama en el campo `tools`."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict[str, Any]) -> str:
        result = self.func(**arguments)
        return result if isinstance(result, str) else str(result)


def _build_parameters(func: Callable[..., Any], param_docs: dict[str, str]) -> dict[str, Any]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        hint = hints.get(pname, str)
        json_type = _JSON_TYPES.get(hint, "string")
        prop: dict[str, Any] = {"type": json_type}
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {"type": "object", "properties": properties, "required": required}


class Registry:
    """Colecciona tools y expone esquemas + ejecución por nombre."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def copy(self) -> "Registry":
        """Registry NUEVO con las mismas tools (m4: aislar del GLOBAL mutable)."""
        new = Registry()
        new._tools = dict(self._tools)
        return new

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"No tengo una tool llamada '{name}', patrón."
        try:
            return tool.run(arguments)
        except TypeError as error:
            return f"Argumentos inválidos para '{name}', patrón: {error}"
        except Exception as error:  # noqa: BLE001 - una tool rota no debe matar el agente
            return f"La tool '{name}' reventó, patrón: {error}"


# Registry global por defecto: las tools se registran al importar sus módulos.
GLOBAL_REGISTRY = Registry()


def tool(_func: Callable | None = None, *, name: str | None = None, safe: bool = True):
    """Decorador. Registra la función como tool en el registry global.

    Args:
        name: nombre expuesto al LLM (por defecto, el de la función).
        safe: si False, el guard pedirá confirmación antes de ejecutarla.
    """

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        description, param_docs = _split_doc(func.__doc__ or "")
        parameters = _build_parameters(func, param_docs)
        t = Tool(
            name=name or func.__name__,
            func=func,
            description=description or func.__name__,
            parameters=parameters,
            safe=safe,
        )
        GLOBAL_REGISTRY.register(t)
        func._crotolamo_tool = t  # type: ignore[attr-defined]
        return func

    if _func is not None:
        return wrap(_func)
    return wrap
