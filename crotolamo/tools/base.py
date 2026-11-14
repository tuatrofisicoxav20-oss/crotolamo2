"""Decorador @tool + inferencia automática de esquema JSON.

Una tool es una función Python tipada. El esquema que ve el LLM se genera desde
la firma (tipos) y el docstring (descripción). Así, agregar una capacidad nueva =
escribir una función con @tool, no un regex.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

from crotolamo.logging_setup import get_logger

log = get_logger("tools")


# ---------------------------------------------------------------------------
# Helpers compartidos por las tools (una sola implementación, cero duplicados).
# OJO: base.py no debe importar de desktop/windows/etc. (evita ciclos).
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_key(text: str) -> str:
    """Normaliza una clave (app/carpeta/sitio/ventana) para emparejar robusto:
    minúsculas, sin acentos y espacios colapsados."""
    return re.sub(r"\s+", " ", _strip_accents(text.lower())).strip()


def run_cmd(args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    """subprocess.run con la configuración estándar de las tools: texto,
    salida capturada y timeout. Lanza OSError/TimeoutExpired si falla."""
    # errors="replace": text=True decodifica en estricto, y un solo byte que no
    # sea UTF-8 en la salida (el nombre de un proceso o de un archivo) lanzaba
    # UnicodeDecodeError, que ningún caller atrapa. La salida válida no cambia.
    return subprocess.run(
        args, text=True, errors="replace", capture_output=True, timeout=timeout,
    )


# Sustitutos UTF-16 sueltos (U+D800..U+DFFF): no son texto y ningún códec los
# acepta.
_LONE_SURROGATES = re.compile("[\ud800-\udfff]")


def replace_surrogates(text: str) -> str:
    """Cambia los sustitutos sueltos por U+FFFD para que el texto sea codificable.

    Python nombra con ellos lo que no pudo decodificar (os.listdir devuelve
    'Declaraci\\udca2n.pdf' para un nombre con bytes que no son UTF-8) y
    json.loads los deja pasar (un "\\ud83d" suelto, medio emoji). Un str así
    lanza UnicodeEncodeError en cuanto alguien lo codifica: al escribirlo en un
    archivo, al imprimirlo, al mandarlo por un socket. Lo demás no se toca.
    """
    # Corre sobre el resultado de CADA tool: el caso normal (texto bien formado)
    # se despacha con un encode, varias veces más barato que pasar la regex.
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return _LONE_SURROGATES.sub("\ufffd", text)
    return text


# Carpetas que no son del patr\u00f3n sino de sus herramientas (entornos virtuales,
# cach\u00e9s, el historial de git): los recorridos y las b\u00fasquedas no bajan a ellas.
SKIP_DIRS = frozenset({".venv", "venv", "__pycache__", ".git", "node_modules"})


# Tope de lectura para no inundar el contexto del LLM.
READ_CAP = 20_000


def truncate_for_context(text: str, cap: int = READ_CAP) -> str:
    """Recorta un texto largo para que quepa en el contexto del LLM,
    dejando una marca con el tamaño original."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[recortado, {len(text)} chars en total]"


def read_text_capped(path: Path, cap: int = READ_CAP) -> str:
    """Texto de un archivo, recortado a `cap` caracteres SIN leerlo entero.

    read_text() + truncate_for_context cargaba y decodificaba el archivo
    completo (un log de 300 MB, un .iso) para quedarse con los primeros
    20 000 caracteres: segundos de CPU y el doble de su tamaño en RAM. Aquí se
    lee solo lo que se devuelve; el total de la marca va en bytes porque
    contarlo en caracteres exigiría leerlo todo. Mismo decodificado que
    read_text(errors="replace"); lanza OSError igual que él.
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        text = fh.read(cap + 1)
        if len(text) <= cap:
            return text
        size = os.fstat(fh.fileno()).st_size
    return text[:cap] + f"\n...[recortado, {size} bytes en total]"


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
    'nombre: texto' documenta un parámetro, y las que le siguen con más
    sangría (o con la misma, si no abren otro parámetro) continúan su texto.
    """
    if not doc:
        return "", {}

    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    arg_re = re.compile(r"^\s*([a-zA-Z_]\w*)\s*:\s*(.+)$")
    # Parámetro abierto y su sangría. Las líneas de continuación se tiraban y el
    # LLM veía la descripción cortada a media frase (light_control.action sin su
    # "toggle"). Una línea en blanco o con menos sangría cierra el parámetro: lo
    # que venga después (otra sección del docstring) no es suyo.
    current: str | None = None
    current_indent = 0

    for ln in lines:
        if ln.strip().lower() in {"args:", "parámetros:", "parametros:"}:
            in_args = True
            continue
        if in_args:
            indent = len(ln) - len(ln.lstrip())
            m = arg_re.match(ln)
            if current is not None and ln.strip() and (
                indent > current_indent or (indent == current_indent and not m)
            ):
                params[current] += " " + ln.strip()
            elif m:
                current, current_indent = m.group(1), indent
                params[current] = m.group(2).strip()
            else:
                current = None
            continue
        # Sin la sangría del docstring: unidas tal cual, el esquema que se manda
        # al LLM en cada petición llevaba "contenido     existente".
        desc_lines.append(ln.strip())

    return " ".join(d for d in desc_lines if d).strip(), params


@dataclass
class Tool:
    name: str
    func: Callable[..., str]
    description: str
    parameters: dict[str, Any]  # JSON-schema del objeto de argumentos
    # safe=True: ejecuta directo. safe=False: el guard pedirá confirmación.
    safe: bool = True
    # direct=True: tool "presentacional" cuyo output YA es una frase lista para el
    # patrón. El agente puede devolverlo tal cual sin una 2ª llamada al LLM
    # (short-circuit). Ver ToolAgent.direct_tools. Default False (comportamiento
    # normal: el modelo redacta la respuesta tras ver el resultado de la tool).
    direct: bool = False
    # strict_args=True (default): el agente FILTRA los kwargs que no estén en
    # parameters["properties"] antes de ejecutar, porque el 3B alucina argumentos
    # ajenos. False: se pasan todos tal cual. Lo usan las tools MCP (M4): su
    # esquema es JSON arbitrario del server (anidado, additionalProperties...) y
    # filtrar por el primer nivel de `properties` mutilaría llamadas válidas.
    strict_args: bool = True

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
        # El resultado va al historial, y de ahí al LLM, a la consola y al log:
        # tiene que ser texto codificable. list_dir y compañía devolvían los
        # nombres de archivo tal como los da Python, sustitutos incluidos.
        return replace_surrogates(result if isinstance(result, str) else str(result))


def _build_parameters(func: Callable[..., Any], param_docs: dict[str, str]) -> dict[str, Any]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception as error:  # noqa: BLE001 — sin hints, degradamos a "string"
        log.debug("no pude resolver los type hints de %s: %s", func.__name__, error)
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


# Tipos Python que casan con cada tipo JSON-schema simple (inverso de _JSON_TYPES;
# un entero vale donde se pide "number").
_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _binding_error(func: Callable[..., Any], arguments: dict[str, Any]) -> str | None:
    """Por qué `arguments` no encajan en la firma de `func` (falta uno requerido,
    sobra uno); None si encajan o si la firma no se puede inspeccionar."""
    try:
        # follow_wrapped=False: la llamada real pasa por el envoltorio (si lo
        # hay), y es SU firma la que decide si los argumentos encajan.
        signature = inspect.signature(func, follow_wrapped=False)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind(**arguments)
    except TypeError as error:
        return str(error)
    return None


def _json_kind(value: Any) -> str:
    """Nombre JSON-schema del tipo de un valor (el que vería el modelo)."""
    if value is None:
        return "null"
    # bool antes que int: es subclase suya.
    for kind, py_type in (("boolean", bool), ("integer", int), ("number", float),
                          ("string", str), ("array", list), ("object", dict)):
        if isinstance(value, py_type):
            return kind
    return type(value).__name__


def _mistyped_arguments(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Qué argumentos no son del tipo simple que declara el esquema (limit="5"
    donde se pide integer), nombrados uno a uno; None si no hay ninguno. Lo que
    el esquema no tipa con un tipo simple (anidados, uniones de las tools MCP)
    no se juzga."""
    properties = tool.parameters.get("properties")
    if not isinstance(properties, dict):
        return None
    problems: list[str] = []
    for key, value in arguments.items():
        spec = properties.get(key)
        declared = spec.get("type") if isinstance(spec, dict) else None
        expected = _PY_TYPES.get(declared) if isinstance(declared, str) else None
        if expected is None:
            continue
        # bool es subclase de int, pero para el esquema no es un número.
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            problems.append(f"'{key}' tiene que ser {declared}, no {_json_kind(value)}")
    return "; ".join(problems) or None


def _bool_as_number(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Por qué un argumento no vale aunque la tool no vaya a atragantarse con él:
    true/false donde el esquema pide un número. None si no hay ninguno.

    Hay que verlo ANTES de llamar: bool es subclase de int y Python lo usa como
    1/0 sin quejarse, así que forget_fact(fact_id=True) borraba el hecho #1 en
    vez de fallar.
    """
    properties = tool.parameters.get("properties")
    if not isinstance(properties, dict):
        return None
    for key, value in arguments.items():
        spec = properties.get(key)
        declared = spec.get("type") if isinstance(spec, dict) else None
        if isinstance(value, bool) and declared in ("integer", "number"):
            return f"'{key}' tiene que ser un número, no {str(value).lower()}"
    return None


class Registry:
    """Colecciona tools y expone esquemas + ejecución por nombre."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Quita una tool por nombre. True si existía (M4: los servers MCP que se
        caen o se cuelgan retiran sus tools en caliente, sin reiniciar)."""
        return self._tools.pop(name, None) is not None

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
        # "Argumentos inválidos" le dice al modelo que corrija SUS argumentos, así
        # que solo vale cuando el fallo es suyo: no encajan en la firma (se ve
        # antes de llamar) o la tool se atraganta con uno de tipo distinto al
        # declarado. Antes lo era CUALQUIER TypeError, también un bug interno de
        # la tool con argumentos correctos: el modelo reintentaba con otros en
        # vez de informar del fallo, y no quedaba traza.
        problem = _binding_error(tool.func, arguments)
        # Solo las nativas: las MCP (strict_args=False) mandan el argumento a
        # su server, que valida contra su propio esquema.
        if problem is None and tool.strict_args:
            problem = _bool_as_number(tool, arguments)
        if problem is not None:
            return f"Argumentos inválidos para '{name}', patrón: {problem}"
        try:
            return tool.run(arguments)
        except Exception as error:  # noqa: BLE001 - una tool rota no debe matar el agente
            # AttributeError también: un None o un número donde se pide texto
            # revienta en el primer .strip()/.lower() de la tool, y eso es tan
            # culpa del argumento como un TypeError.
            # Se nombra el argumento, no el texto de Python: "'>' not supported
            # between 'str' and 'int'" no dice cuál corregir, y "data must be
            # str" habla de un parámetro interno (el de write_text) que el
            # modelo nunca mandó.
            if isinstance(error, (TypeError, AttributeError)) and (
                mistyped := _mistyped_arguments(tool, arguments)
            ):
                log.debug("la tool '%s' se atragantó con %s: %s", name, mistyped, error)
                return f"Argumentos inválidos para '{name}', patrón: {mistyped}"
            log.exception("la tool '%s' reventó", name)
            # El mensaje de la excepción puede citar una ruta o un argumento.
            return replace_surrogates(f"La tool '{name}' reventó, patrón: {error}")


# Registry global por defecto: las tools se registran al importar sus módulos.
GLOBAL_REGISTRY = Registry()


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    safe: bool = True,
    direct: bool = False,
):
    """Decorador. Registra la función como tool en el registry global.

    Args:
        name: nombre expuesto al LLM (por defecto, el de la función).
        safe: si False, el guard pedirá confirmación antes de ejecutarla.
        direct: si True, su output es presentacional (frase lista para el patrón)
            y el agente puede devolverlo sin una 2ª llamada al LLM (short-circuit).
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
            direct=direct,
        )
        GLOBAL_REGISTRY.register(t)
        func._crotolamo_tool = t  # type: ignore[attr-defined]
        return func

    if _func is not None:
        return wrap(_func)
    return wrap
