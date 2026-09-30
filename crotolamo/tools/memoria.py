"""Tools de memoria semántica (mem0): recordar, buscar y olvidar hechos sobre el patrón.

No usan el decorador @tool a propósito: se registran SOLO cuando
[memoria].enabled = true (ver tools/__init__.py), y entonces sustituyen a las
tools de hechos SQLite para que el modelo vea UNA sola familia de "recordar".
"""

from __future__ import annotations

from typing import Callable

from crotolamo.core.memoria import (
    REGLA_CONTENIDO,
    MemoriaNoDisponible,
    SecretoRechazado,
    get_memoria,
)
from crotolamo.tools.base import Tool, _build_parameters, _split_doc


def recordar_de_mi(hecho: str) -> str:
    """Guarda un hecho duradero sobre el patrón para recordarlo entre sesiones
    ("acuérdate de que me gusta el café sin azúcar"). REGLA: {regla}

    Args:
        hecho: el hecho, en una frase corta y en español.
    """
    hecho = hecho.strip()
    if not hecho:
        return "¿Qué quieres que recuerde, patrón? No me diste nada."
    try:
        ids = get_memoria().recordar(hecho)
    except SecretoRechazado:
        return "Eso tiene pinta de contraseña o clave, patrón; eso no lo guardo ni a la mala."
    except MemoriaNoDisponible as error:
        return f"Ahorita no tengo memoria de largo plazo, patrón ({error})."
    if not ids:
        return "No pude guardarlo, patrón."
    return f"Ya quedó, patrón. Me acordaré: «{hecho}»."


def buscar_recuerdos(pregunta: str) -> str:
    """Busca en la memoria de largo plazo lo que el patrón le ha contado a Crotolamo
    ("¿qué sabes de mí?", "¿cómo se llama mi perro?"). Solo hechos personales.

    Args:
        pregunta: qué quieres recordar, en lenguaje natural.
    """
    pregunta = pregunta.strip()
    if not pregunta:
        return "¿Qué quieres que busque, patrón?"
    memoria = get_memoria()
    recuerdos = memoria.buscar(pregunta, top_k=5, timeout_s=10.0)
    if not recuerdos:
        if not memoria.enabled:
            return "Ahorita no tengo memoria de largo plazo, patrón."
        return f"No recuerdo nada sobre «{pregunta}», patrón."
    return "Esto recuerdo, patrón:\n" + "\n".join(f"- {r.texto}" for r in recuerdos)


def olvidar_recuerdo(descripcion: str) -> str:
    """Olvida (borra) el recuerdo que mejor encaja con la descripción ("olvida lo de
    mi perro"). Borra UNO por llamada y dice cuál fue.

    Args:
        descripcion: de qué trata el recuerdo a olvidar.
    """
    descripcion = descripcion.strip()
    if not descripcion:
        return "¿Qué quieres que olvide, patrón?"
    try:
        borrado = get_memoria().olvidar(descripcion)
    except MemoriaNoDisponible as error:
        return f"Ahorita no tengo memoria de largo plazo, patrón ({error})."
    if borrado is None:
        return f"No encontré ningún recuerdo sobre «{descripcion}», patrón."
    return f"Olvidado, patrón: «{borrado.texto}»."


def _make(func: Callable[..., str], safe: bool = True) -> Tool:
    doc = (func.__doc__ or "").replace("{regla}", REGLA_CONTENIDO)
    description, param_docs = _split_doc(doc)
    return Tool(
        name=func.__name__,
        func=func,
        description=description,
        parameters=_build_parameters(func, param_docs),
        safe=safe,
    )


def memoria_tools() -> list[Tool]:
    """Las tres tools, listas para registrar (el registro lo hace tools/__init__)."""
    return [
        _make(recordar_de_mi),
        _make(buscar_recuerdos),
        # Olvidar es destructivo pero acotado (un recuerdo, y dice cuál): sin
        # confirmación para que "olvida lo de X" por voz sea de un solo paso.
        _make(olvidar_recuerdo),
    ]


# Tools de hechos SQLite que se retiran cuando la memoria semántica está activa.
TOOLS_SQLITE_SUSTITUIDAS: tuple[str, ...] = (
    "remember_fact", "recall_facts", "search_facts", "forget_fact",
)
