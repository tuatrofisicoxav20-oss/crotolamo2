"""Tools de memoria semántica (mem0): recordar, buscar y olvidar hechos sobre el patrón.

No usan el decorador @tool a propósito: se registran SOLO cuando
[memoria].enabled = true (ver tools/__init__.py), y entonces sustituyen a las
tools de hechos SQLite para que el modelo vea UNA sola familia de "recordar".
"""

from __future__ import annotations

import re
from typing import Callable

from crotolamo.core.memoria import (
    REGLA_CONTENIDO,
    BusquedaFallida,
    MemoriaNoDisponible,
    SecretoRechazado,
    get_memoria,
    parece_secreto,
)
from crotolamo.persistence import facts as facts_sqlite
from crotolamo.tools.base import LANE_LIBRETA, Tool, _build_parameters, _split_doc, normalize_key


# Si la memoria semántica cae en caliente (mem0 roto, Chroma ilegible), estas
# tools usan los hechos SQLite de siempre: retirar las tools SQLite no puede
# dejar al asistente sin ninguna forma de recordar.
_AVISO_RESPALDO = " (memoria semántica caída: uso la libreta de hechos)"


def _no_pude_buscar(error: BusquedaFallida, coletilla: str) -> str:
    # La memoria está en pie pero ESTA búsqueda falló o tardó de más. Contestar
    # "no recuerdo nada" era mentira (el recuerdo puede estar guardado) y es lo
    # que el modelo le repetía al patrón. Tampoco se cae a la libreta: no está
    # caída, y olvidar borraría de ahí un hecho que sigue vivo en la memoria.
    return f"No pude consultar la memoria, patrón ({error}). {coletilla}"


def _recordar_sqlite(hecho: str) -> str:
    if parece_secreto(hecho):
        return "Eso tiene pinta de contraseña o clave, patrón; eso no lo guardo ni a la mala."
    facts_sqlite.remember(hecho)
    return f"Ya quedó, patrón. Me acordaré: «{hecho}».{_AVISO_RESPALDO}"


def _buscar_sqlite(pregunta: str) -> str:
    rows = facts_sqlite.search(pregunta)
    if not rows:
        return f"No recuerdo nada sobre «{pregunta}», patrón.{_AVISO_RESPALDO}"
    return ("Esto recuerdo, patrón:\n" + "\n".join(f"- {r['texto']}" for r in rows)
            + _AVISO_RESPALDO)


# Palabras que no dicen DE QUÉ trata un recuerdo: artículos, preposiciones,
# conjunciones, pronombres, posesivos, interrogativos ("dónde vive mi mamá" no
# trata de "dónde"), "patrón" (los hechos se guardan también en tercera persona:
# "el patrón toma...")... y las de la propia orden ("olvida lo de mi X"). En
# minúsculas, con y sin acento.
_PALABRAS_VACIAS = frozenset(
    "a al ante bajo con contra de del desde en entre hacia hasta para por según segun sin "
    "sobre tras el la los las lo un una unos unas y e o u ni que qué pero si sí como cómo "
    "porque yo tú tu él me mí mi mis te ti tus se su sus le les nos este esta esto estos "
    "estas ese esa eso esos esas aquel aquella aquello es son era fue ser estar está hay no "
    "ya muy más mas también tambien olvida olvidar olvídate olvidate borra borrar quita "
    "quitar elimina eliminar recuerdo recuerdos dónde donde cuál cual cuáles cuales cuándo "
    "cuando quién quien quiénes quienes cuánto cuanto cuánta cuanta cuántos cuantos cuántas "
    "cuantas patrón patron".split()
)

# Palabras que dicen QUÉ CLASE de hecho es, no de qué trata: "cómo se llama mi
# gato" trata del gato, no de "llama", que comparten el perro, el jefe y la
# hermana. No bastan para elegir el hecho a borrar; sí desempatan entre los que
# tratan del tema ("cómo se llama mi perro": el del nombre, no el de las
# croquetas) y son el tema si la descripción no trae otra cosa ("olvida dónde
# vivo"). Sin acentos: se comparan ya normalizadas.
_PALABRAS_DEBILES = frozenset(
    "llama llamo llamas llaman nombre nombres gusta gustan gusto encanta encantan tengo tiene "
    "tienes tienen vive vivo vives viven favorito favorita favoritos favoritas".split()
)


def _palabras_con_contenido(texto: str) -> set[str]:
    # Las vacías se descartan ANTES de quitar acentos ("té" no es "te"); después
    # se comparan sin acentos, que el STT y el modelo no siempre los ponen.
    palabras = set(re.findall(r"\w+", texto.lower())) - _PALABRAS_VACIAS
    return {normalize_key(palabra) for palabra in palabras}


def _olvidar_sqlite(descripcion: str) -> str:
    # facts.search ORDENA por parecido pero no descarta nada (dos frases
    # cualesquiera comparten alguna letra o un "de"): con su top-1 a ciegas,
    # "olvida lo de mi coche" BORRABA «me gusta el café de olla». Tampoco vale "el
    # primero del ranking que comparta UNA palabra": el ranking cuenta las vacías
    # y basta un verbo genérico, y "cómo se llama mi gato" borraba «mi jefe se
    # llama Ramiro». Cada hecho se puntúa por las palabras del TEMA que comparte
    # (y, a igualdad, por todas); se borra el único mejor. Si ninguno trata del
    # tema, o empatan hechos distintos, no se borra nada: olvidar por voz debe ser
    # conservador (como Memoria.olvidar), y un borrado es para siempre.
    claves = _palabras_con_contenido(descripcion)
    tema = (claves - _PALABRAS_DEBILES) or claves
    mejor: tuple[int, int] = (0, 0)
    empatados: list[dict] = []
    for row in (facts_sqlite.recall() if tema else []):
        palabras = _palabras_con_contenido(row["texto"])
        puntos = (len(tema & palabras), len(claves & palabras))
        if puntos > mejor:
            mejor, empatados = puntos, [row]
        elif puntos == mejor:
            empatados.append(row)
    if mejor[0] == 0:
        return f"No encontré ningún recuerdo sobre «{descripcion}», patrón.{_AVISO_RESPALDO}"
    # Dos copias del mismo hecho (recordar dos veces) son UN recuerdo, no un
    # empate: si no, "olvida lo del pozole" preguntaría "¿cuál?" para siempre.
    distintos: dict[str, str] = {}
    for row in empatados:
        distintos.setdefault(normalize_key(row["texto"]), row["texto"])
    if len(distintos) > 1:
        textos = list(distintos.values())
        lista = ", ".join(f"«{texto}»" for texto in textos[:3])
        if len(textos) > 3:
            lista += f" y {len(textos) - 3} más"
        return (f"Tengo varios recuerdos sobre «{descripcion}», patrón, y no borré "
                f"ninguno: {lista}. Dime cuál olvido.{_AVISO_RESPALDO}")
    row = empatados[0]
    if facts_sqlite.forget(int(row["id"])):
        return f"Olvidado, patrón: «{row['texto']}».{_AVISO_RESPALDO}"
    return f"No encontré ningún recuerdo sobre «{descripcion}», patrón.{_AVISO_RESPALDO}"


def recordar_de_mi(hecho: str) -> str:
    """Guarda un hecho duradero sobre el patrón para recordarlo entre sesiones
    ("acuérdate de que me gusta el café sin azúcar"). REGLA: {regla}

    Args:
        hecho: el hecho, en una frase corta y en español.
    """
    hecho = hecho.strip()
    if not hecho:
        return "¿Qué quieres que recuerde, patrón? No me diste nada."
    memoria = get_memoria()
    if not memoria.enabled:
        return _recordar_sqlite(hecho)
    try:
        ids = memoria.recordar(hecho)
    except SecretoRechazado:
        return "Eso tiene pinta de contraseña o clave, patrón; eso no lo guardo ni a la mala."
    except MemoriaNoDisponible:
        return _recordar_sqlite(hecho)
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
    if not memoria.enabled:
        return _buscar_sqlite(pregunta)
    try:
        recuerdos = memoria.buscar_estricto(pregunta, top_k=5, timeout_s=10.0)
    except MemoriaNoDisponible:  # se cayó durante esta búsqueda
        return _buscar_sqlite(pregunta)
    except BusquedaFallida as error:
        return _no_pude_buscar(error, f"No sé si tengo algo sobre «{pregunta}».")
    if not recuerdos:
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
    memoria = get_memoria()
    if not memoria.enabled:
        return _olvidar_sqlite(descripcion)
    try:
        borrado = memoria.olvidar(descripcion)
    except MemoriaNoDisponible:  # se cayó durante la búsqueda
        return _olvidar_sqlite(descripcion)
    except BusquedaFallida as error:
        return _no_pude_buscar(error, "No sé si ese recuerdo existe: no borré nada.")
    if borrado is None:
        if not memoria.enabled:  # se apagó entre la comprobación de arriba y olvidar()
            return _olvidar_sqlite(descripcion)
        return f"No encontré ningún recuerdo sobre «{descripcion}», patrón."
    return f"Olvidado, patrón: «{borrado.texto}»."


def _make(func: Callable[..., str], safe: bool = True, parallel_safe: bool = False,
          parallel_lane: str | None = None) -> Tool:
    doc = (func.__doc__ or "").replace("{regla}", REGLA_CONTENIDO)
    description, param_docs = _split_doc(doc)
    return Tool(
        name=func.__name__,
        func=func,
        description=description,
        parameters=_build_parameters(func, param_docs),
        safe=safe,
        parallel_safe=parallel_safe,
        parallel_lane=parallel_lane,
    )


def memoria_tools() -> list[Tool]:
    """Las tres tools, listas para registrar (el registro lo hace tools/__init__)."""
    return [
        _make(recordar_de_mi),
        # Solo lee: con otras lecturas del mismo mensaje corre a la vez; con las
        # de la libreta (y con otra búsqueda de recuerdos), una tras otra.
        _make(buscar_recuerdos, parallel_safe=True, parallel_lane=LANE_LIBRETA),
        # Olvidar es destructivo pero acotado (un recuerdo, y dice cuál): sin
        # confirmación para que "olvida lo de X" por voz sea de un solo paso.
        _make(olvidar_recuerdo),
    ]


# Tools de hechos SQLite que se retiran cuando la memoria semántica está activa.
TOOLS_SQLITE_SUSTITUIDAS: tuple[str, ...] = (
    "remember_fact", "recall_facts", "search_facts", "forget_fact",
)
