"""Revisión final, unidad x2cfg: la documentación de config y README dice lo que
hace el código de hoy.

Cada test ancla una frase de la doc a un hecho del código (una constante, una
ruta que se ejecuta): si el código vuelve a cambiar, el test cae y obliga a
actualizar la doc en el mismo cambio.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from crotolamo import settings as settings_mod
from crotolamo.mcp import bridge

_RAIZ = Path(__file__).resolve().parents[1]


def _seccion_md(texto: str, titulo: str) -> str:
    """Cuerpo de la sección `## titulo` (hasta el siguiente `## `), en una línea."""
    cuerpo = re.split(rf"^## {re.escape(titulo)}\s*$", texto, maxsplit=1, flags=re.M)[1]
    cuerpo = re.split(r"^## ", cuerpo, maxsplit=1, flags=re.M)[0]
    return " ".join(cuerpo.split())


def _seccion_toml(texto: str, tabla: str) -> str:
    """Texto crudo (comentarios incluidos) de la tabla `[tabla]` del toml."""
    cuerpo = re.split(rf"^\[{re.escape(tabla)}\]\s*$", texto, maxsplit=1, flags=re.M)[1]
    return re.split(r"^\[", cuerpo, maxsplit=1, flags=re.M)[0]


def _comentario_de(seccion: str, clave: str) -> str:
    """Bloque de comentarios pegado encima de `clave = ...`, en una línea."""
    lineas = seccion.splitlines()
    idx = next(i for i, linea in enumerate(lineas)
               if re.match(rf"^{re.escape(clave)}\s*=", linea))
    bloque: list[str] = []
    for linea in reversed(lineas[:idx]):
        if not linea.startswith("#"):
            break
        bloque.append(linea.lstrip("#").strip())
    return " ".join(reversed(bloque))


# ---------------------------------------------------------------------------
# README_M4: el inputSchema de una tool MCP se adelgaza pasado MCP_SCHEMA_CAP.
# El paso 7 del presupuesto de bytes metió el recorte y README_M4 seguía
# diciendo "se respeta tal cual": quien depurara por qué el modelo no ve el
# `examples` de un campo no encontraba nada en la doc del subsistema.
# ---------------------------------------------------------------------------

def _esquema_grande(n: int = 12) -> dict:
    return {
        "type": "object",
        "description": "Tool de prueba con muchos campos.",
        "properties": {
            f"campo{i}": {"type": "string", "title": f"Campo {i}",
                          "examples": [f"ejemplo {i}"], "description": "x" * 300}
            for i in range(n)
        },
        "required": ["campo0"],
    }


def test_readme_m4_documenta_el_tope_del_inputschema_como_lo_aplica_el_bridge(caplog):
    # Lo que hace el código (el hecho que la doc tiene que contar).
    pequeno = {"type": "object", "properties": {
        "q": {"type": "string", "title": "Q", "examples": ["hola"], "default": "x"}}}
    assert bridge.normalize_schema(pequeno, "chica") == pequeno  # bajo el tope: igual
    caplog.set_level(logging.INFO, logger="crotolamo.mcp.bridge")
    grande = _esquema_grande()
    assert len(json.dumps(grande, ensure_ascii=False).encode()) > bridge.MCP_SCHEMA_CAP
    fuera = bridge.normalize_schema(grande, "grande")
    campo = fuera["properties"]["campo0"]
    assert set(campo) == {"type", "description"}
    assert len(campo["description"]) == bridge.NESTED_DESCRIPTION_CAP + 1  # + "…"
    assert fuera["required"] == ["campo0"]
    assert "adelgazado" in caplog.text

    # Lo que dice README_M4, en la sección donde se habla del inputSchema.
    doc = _seccion_md((_RAIZ / "README_M4.md").read_text(encoding="utf-8"),
                      "Nombres de tools")
    assert "inputSchema" in doc
    tope = bridge.MCP_SCHEMA_CAP
    assert re.search(rf"\b{tope}\b|\b{tope // 1000}[ .]?000\b", doc), \
        "README_M4 no dice el tope de bytes del inputSchema (MCP_SCHEMA_CAP)"
    assert "MCP_SCHEMA_CAP" in doc
    for anotacion in sorted(bridge._SCHEMA_ANNOTATIONS):
        assert f"`{anotacion}`" in doc, f"README_M4 no dice que se quita `{anotacion}`"
    assert re.search(rf"\b{bridge.NESTED_DESCRIPTION_CAP}\b", doc), \
        "README_M4 no dice a cuánto se recortan las descripciones anidadas"
    for intacta in ("type", "properties", "required", "enum", "additionalProperties"):
        assert f"`{intacta}`" in doc
    assert "INFO" in doc, "README_M4 no dice que el recorte se anota en el log"
    # Y ya no promete el "tal cual" sin condición: el recorte va pegado a él.
    assert re.search(r"tal cual[^.]*\.[^.]*MCP_SCHEMA_CAP|tal cual.{0,400}MCP_SCHEMA_CAP",
                     doc), "README_M4 sigue diciendo 'tal cual' sin mencionar el tope"


# ---------------------------------------------------------------------------
# config/crotolamo.toml: [voice].vad_backend solo lo lee el loop SIMPLE. El
# oído concurrente (use_oww = true, el de fábrica) construye _SileroVad sin
# mirar la clave: con 'energy' el dueño no cambiaba nada y la doc no lo decía.
# ---------------------------------------------------------------------------

def test_vad_backend_no_cambia_el_vad_del_oido_concurrente(monkeypatch):
    """El hecho del código: VoiceLoop (concurrente) usa _SileroVad sin leer
    [voice].vad_backend; la clave no llega a él por ningún lado."""
    import inspect

    from crotolamo.voice import loop as loop_mod
    from crotolamo.voice.adapters import _SileroVad

    assert "vad_backend" not in inspect.getsource(loop_mod)
    assert "vad_backend" not in inspect.getsource(_SileroVad)
    firma = inspect.signature(loop_mod.VoiceLoop.__init__)
    assert "vad_backend" not in firma.parameters


def test_toml_dice_que_vad_backend_solo_aplica_al_loop_simple():
    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    voz = _seccion_toml(texto, "voice")
    comentario = _comentario_de(voz, "vad_backend")
    assert "simple" in comentario, comentario
    assert re.search(r"concurrente.{0,80}Silero siempre", comentario), comentario

    # La lista de claves de [voice] que solo aplican al modo simple, en el
    # comentario de [wake].use_oww, la incluye.
    wake = _comentario_de(_seccion_toml(texto, "wake"), "use_oww")
    lista = wake.split("solo aplican al modo simple", 1)[1]
    assert "vad_backend" in lista, lista
