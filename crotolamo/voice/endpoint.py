"""Endpointing inteligente: ¿la frase suena a pausa de PENSAR o ya terminó?

El VAD corta por silencio fijo (vad_silence_ms): si el patrón se detiene a
media oración a pensar ("mueve el archivo de... eh..."), el corte lo deja
con un comando trunco. Este módulo mira la TRANSCRIPCIÓN y decide si parece
incompleta; si sí, STT.listen_smart reabre la escucha unos segundos y
concatena la continuación.

Heurística barata (sin LLM, sin latencia): una oración en español casi nunca
termina en conjunción/preposición/artículo ("y", "de", "para", "la"...), ni
en coma o puntos suspensivos, ni es solo un verbo transitivo colgado
("abre", "mueve"). Es deliberadamente conservadora: un falso "incompleta"
solo cuesta la ventana corta de continuación; un falso "completa" deja el
comando como hoy.
"""

from __future__ import annotations

import unicodedata


def _norm(text: str) -> str:
    """minúsculas + sin acentos (mismo criterio que el router)."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Palabras que NO cierran una oración natural en español: si la frase termina
# aquí, el patrón seguía hablando. OJO con lo que NO está: "es" ("qué hora
# es"), "eso/esa" ("borra eso"), "ya", "pausa" — esas sí cierran órdenes.
_CONNECTORS = frozenset({
    # conjunciones
    "y", "e", "o", "u", "pero", "que", "ni", "aunque", "porque", "si",
    "como", "cuando", "donde", "mientras", "entonces", "luego",
    # preposiciones
    "a", "al", "de", "del", "en", "con", "para", "por", "sin", "sobre",
    "entre", "hacia", "hasta", "desde", "segun", "contra",
    # artículos y determinantes colgados
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "mi", "mis", "tu", "tus", "su", "sus", "este", "esta",
    # comparativos/cuantificadores colgados
    "mas", "menos", "muy", "tan", "cada", "todo", "toda", "todos", "todas",
})

# Verbos transitivos que, SOLOS, piden objeto: "abre" (¿qué?), "mueve" (¿qué?).
# Solo aplican cuando la transcripción es UNA palabra; "pausa" o "siguiente"
# solos sí son órdenes completas y no están aquí.
_DANGLING_VERBS = frozenset({
    "abre", "abreme", "abrime", "pon", "ponme", "mueve", "mueveme",
    "busca", "buscame", "manda", "mandame", "pasa", "pasame", "quita",
    "quitame", "cierra", "cierrame", "cambia", "toca", "reproduce",
    "dime", "dame", "lee", "leeme", "escribe", "apunta", "borra",
    "elimina", "crea", "creame", "guarda", "guardame", "enseñame",
    "ensename", "muestrame", "recuerda", "recuerdame",
})


def seems_incomplete(text: str) -> bool:
    """True si la transcripción parece una oración a medias (pausa de pensar)."""
    t = text.strip()
    if not t:
        return False
    # Whisper marca el trailing-off con coma o puntos suspensivos.
    if t.endswith(",") or t.endswith("...") or t.endswith("…"):
        return True
    words = _norm(t.rstrip(".!?¿¡ ")).split()
    if not words:
        return False
    if words[-1] in _CONNECTORS:
        return True
    return len(words) == 1 and words[0] in _DANGLING_VERBS
