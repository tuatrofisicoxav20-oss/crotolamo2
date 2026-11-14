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

# Formas TÓNICAS: llevan tilde justo para distinguirse del conector átono, y
# ESAS sí cierran oración — "sí" (afirmación) no es "si" (conjunción), "está"
# (verbo) no es "esta" (determinante), "él/mí/tú" (pronombres) no son
# "el/mi/tu". Como _norm quita los acentos, sin esta lista un "sí" a secas o un
# "¿dónde está?" contaban como frase a medias y reabrían la escucha antes de
# actuar. Se comparan CON acento, antes de normalizar. Los interrogativos
# tónicos son la misma clase: "¿por qué?", "¿a dónde?", "¿hasta cuándo?" o
# "dime cómo" terminan la pregunta, y sin tilde chocaban con los conectores
# "que/donde/cuando/como" (reabrían 2.5 s y pegaban a la orden lo que sonara).
_TONIC_CLOSERS = frozenset({"sí", "él", "mí", "tú", "está", "qué", "cómo", "cuándo", "dónde"})

# Verbos transitivos que, SOLOS, piden objeto: "abre" (¿qué?), "mueve" (¿qué?).
# Solo aplican cuando la transcripción es UNA palabra; "pausa" o "siguiente"
# solos sí son órdenes completas y no están aquí.
_DANGLING_VERBS = frozenset({
    "abre", "abreme", "abrime", "pon", "ponme", "mueve", "mueveme",
    "busca", "buscame", "manda", "mandame", "pasa", "pasame", "quita",
    "quitame", "cierra", "cierrame", "cambia", "toca", "reproduce",
    "dime", "dame", "lee", "leeme", "escribe", "apunta", "borra",
    "elimina", "crea", "creame", "guarda", "guardame",
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
    body = t.rstrip(".!?¿¡ ")
    words = _norm(body).split()
    if not words:
        return False
    # NFC: Whisper puede entregar la tilde como carácter combinante suelto.
    last = unicodedata.normalize("NFC", body.lower()).split()[-1]
    # Una palabra que ABRE con ¿/¡ es una pregunta o exclamación de una sola
    # palabra ("¿Y?", "¿Cómo?", "¡Abre!"): Whisper ya la abrió y la cerró. Se
    # dice a propósito; antes solo salía bien porque el "¿" pegado a la palabra
    # le impedía coincidir con un conector o un verbo colgado.
    if last[0] in "¿¡":
        return False
    if last in _TONIC_CLOSERS:
        return False
    if words[-1] in _CONNECTORS:
        return True
    return len(words) == 1 and words[0] in _DANGLING_VERBS
