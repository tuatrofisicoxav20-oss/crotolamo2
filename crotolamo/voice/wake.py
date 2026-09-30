"""Detección de wake word 'crotolamo'. Migrado ÍNTEGRO de C1::listener.py.

Esta lógica (SequenceMatcher + ventanas deslizantes + tabla de variantes) es
ingeniosa y aguanta bien los errores de Whisper en español. Se conserva tal cual,
solo adaptada a leer target/threshold/variants desde config.

La E/S de audio (faster-whisper, micrófono) llega en la Fase 5; aquí vive solo
la lógica pura, testeable sin micrófono.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

WAKE_TARGET = "crotolamo"

# Fallback MÍNIMO. La fuente real de variantes es la config ([wake].variants);
# WAKE_VARIANTS solo se usa si la config no está disponible (M5: una sola fuente).
WAKE_VARIANTS = ["crotolamo", "croto lamo", "coto y amo", "control amo"]

CONFIRM_VARIANTS = ["sí", "si", "confirmo", "ejecuta", "dale", "hazlo", "va", "correcto"]
CANCEL_VARIANTS = ["no", "cancela", "cancelar", "nel", "no lo hagas"]

DEFAULT_THRESHOLD = 0.67


def _default_variants() -> list[str]:
    """Variantes por defecto desde la config ([wake].variants); WAKE_VARIANTS de fallback."""
    try:
        from crotolamo.settings import get_settings

        variants = get_settings().wake.get("variants")
        if variants:
            return list(variants)
    except Exception:
        pass
    return WAKE_VARIANTS


def normalize_for_wake(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text: str) -> str:
    return normalize_for_wake(text).replace(" ", "")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def wake_score(text: str, variants: Iterable[str] | None = None) -> tuple[float, str]:
    """Devuelve qué tan parecido fue lo escuchado a 'crotolamo'."""
    variant_list = list(variants) if variants is not None else _default_variants()
    comp = compact(normalize_for_wake(text))
    candidates = [compact(WAKE_TARGET)] + [compact(v) for v in variant_list]

    best_score = 0.0
    best_variant = ""

    # Comparación completa.
    for candidate in candidates:
        score = similarity(comp, candidate)
        if score > best_score:
            best_score, best_variant = score, candidate

    # Comparación por ventanas: "oye crotolamo abre youtube".
    for candidate in candidates:
        n = len(candidate)
        if n == 0 or len(comp) < n:
            continue
        for i in range(0, len(comp) - n + 1):
            score = similarity(comp[i:i + n], candidate)
            if score > best_score:
                best_score, best_variant = score, candidate

    return best_score, best_variant


def is_wake_word(text: str, threshold: float = DEFAULT_THRESHOLD,
                 variants: Iterable[str] | None = None) -> bool:
    variant_list = list(variants) if variants is not None else _default_variants()
    norm = normalize_for_wake(text)
    for variant in variant_list:
        if normalize_for_wake(variant) in norm:
            return True
    score, _ = wake_score(text, variant_list)
    return score >= threshold


def strip_wake_word(text: str, variants: Iterable[str] | None = None) -> str:
    """Quita la palabra de activación para dejar solo la orden."""
    variant_list = list(variants) if variants is not None else _default_variants()
    original = text.strip()
    lower = normalize_for_wake(original)

    for wake in variant_list:
        wake_norm = normalize_for_wake(wake)
        if lower.startswith(wake_norm):
            count = len(wake_norm.split())
            return " ".join(original.split()[count:]).strip(" ,.:;-")

    words = original.split()
    for n in (3, 2, 1):
        if len(words) <= n:
            continue
        if is_wake_word(" ".join(words[:n]), variants=variant_list):
            return " ".join(words[n:]).strip(" ,.:;-")

    return original


def contains_any(text: str, variants: Iterable[str]) -> bool:
    """True si alguna variante aparece como PALABRA(S) COMPLETA(S) en el texto.

    Antes comparaba por subcadena y eso confirmaba acciones delicadas por
    accidente: "necesito pensarlo" contenía el "si" de nece-SI-to, "nueva"
    contenía "va" y "bueno" cancelaba por el "no". Ahora la variante debe
    coincidir token a token (las de varias palabras, como secuencia contigua),
    sobre el texto normalizado (minúsculas, sin acentos ni puntuación).
    """
    words = normalize_for_wake(text).split()
    for variant in variants:
        needle = normalize_for_wake(variant).split()
        n = len(needle)
        if n == 0 or n > len(words):
            continue
        if any(words[i:i + n] == needle for i in range(len(words) - n + 1)):
            return True
    return False


def confirmation_from_answer(answer: str) -> bool:
    """Decide una confirmación por voz. Cancelar GANA si aparece junto a un sí.

    Es la única regla de decisión para tools delicadas dichas por voz; vive
    aquí (pura) para poder probarla sin micrófono. Cualquier respuesta que no
    contenga una confirmación clara se trata como "no" (lo seguro).
    """
    if contains_any(answer, CANCEL_VARIANTS):
        return False
    return contains_any(answer, CONFIRM_VARIANTS)


def split_wake_command(
    heard: str,
    threshold: float = DEFAULT_THRESHOLD,
    variants: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """(¿se activó?, comando pegado) para una frase dicha DE CORRIDO.

    Antes el listener rechazaba de plano cualquier frase de 4+ palabras (guard
    anti-alucinación), así que "crotolamo ábreme la carpeta" NO activaba. La
    regla nueva conserva la protección pero bien puesta: el wake word debe
    venir AL INICIO (en las 3 primeras palabras) — la música/ruido transcrito
    casi nunca ARRANCA con algo que suene a "crotolamo", pero el patrón sí.

    - "crotolamo"                    -> (True, "")            [pide la orden]
    - "crotolamo pausa la música"    -> (True, "pausa la música")  [directo]
    - "abre la carpeta de descargas" -> (False, "")           [no es wake]
    """
    heard = (heard or "").strip()
    if not heard:
        return False, ""
    variant_list = list(variants) if variants is not None else _default_variants()
    head = " ".join(heard.split()[:3])
    if not is_wake_word(head, threshold, variant_list):
        return False, ""
    command = strip_wake_word(heard, variant_list)
    # strip devuelve el original si no encontró qué quitar (p.ej. el score fue
    # difuso sobre las 3 palabras juntas): en ese caso no hay orden separable.
    if normalize_for_wake(command) == normalize_for_wake(heard):
        return True, ""
    return True, command.strip()
