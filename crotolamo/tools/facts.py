"""Tools de memoria persistente: recordar y recuperar hechos sobre el patrón."""

from __future__ import annotations

from crotolamo.persistence import facts
from crotolamo.tools.base import tool


@tool
def remember_fact(texto: str, categoria: str = "general") -> str:
    """Guarda un hecho sobre el patrón para recordarlo entre sesiones.

    Args:
        texto: el hecho a recordar (ej: "mi proyecto principal es Huevonitis 4").
        categoria: etiqueta opcional (proyectos, preferencias, datos, ...).
    """
    if not texto.strip():
        return "¿Qué quieres que recuerde, patrón? No me diste nada."
    facts.remember(texto, categoria)
    return f"Anotado, patrón. Lo recordaré: «{texto.strip()}»."


@tool
def recall_facts(categoria: str = "") -> str:
    """Recupera los hechos recordados, opcionalmente filtrando por categoría.

    Args:
        categoria: si se da, solo devuelve los hechos de esa categoría.
    """
    rows = facts.recall(categoria.strip() or None)
    if not rows:
        return "No tengo hechos guardados, patrón." if not categoria else \
            f"No tengo hechos en la categoría '{categoria}', patrón."
    lines = [f"#{r['id']} ({r['categoria']}): {r['texto']}" for r in rows]
    return "Esto es lo que recuerdo, patrón:\n" + "\n".join(lines)


@tool
def search_facts(query: str) -> str:
    """Busca entre los hechos recordados los más relevantes a una consulta.

    Args:
        query: lo que quieres encontrar (ej: "proyecto principal").
    """
    rows = facts.search(query)
    if not rows:
        return f"No encontré nada parecido a «{query}» en mi memoria, patrón."
    lines = [f"#{r['id']} ({r['categoria']}): {r['texto']}" for r in rows]
    return "Lo más parecido que recuerdo, patrón:\n" + "\n".join(lines)


@tool
def forget_fact(fact_id: int) -> str:
    """Olvida (borra) un hecho por su número de id.

    Args:
        fact_id: el id del hecho a borrar (lo muestra recall_facts).
    """
    if facts.forget(fact_id):
        return f"Olvidado el hecho #{fact_id}, patrón."
    return f"No tengo un hecho #{fact_id}, patrón."
