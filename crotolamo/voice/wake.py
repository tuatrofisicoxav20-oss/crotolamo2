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

WAKE_VARIANTS = [
    "crotolamo", "crótolamo", "coto y amo", "coto lamo", "coto amo",
    "croto lamo", "croto el amo", "corto lamo", "corto la mano", "crotolo amo",
    "control amo", "contro lamo", "cuatro lamo", "proto lamo", "troto lamo",
    "cróto lamo",
]

CONFIRM_VARIANTS = ["sí", "si", "confirmo", "ejecuta", "dale", "hazlo", "va", "correcto"]
CANCEL_VARIANTS = ["no", "cancela", "cancelar", "nel", "no lo hagas"]

DEFAULT_THRESHOLD = 0.67


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
    variant_list = list(variants) if variants is not None else WAKE_VARIANTS
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
    variant_list = list(variants) if variants is not None else WAKE_VARIANTS
    norm = normalize_for_wake(text)
    for variant in variant_list:
        if normalize_for_wake(variant) in norm:
            return True
    score, _ = wake_score(text, variant_list)
    return score >= threshold


def strip_wake_word(text: str, variants: Iterable[str] | None = None) -> str:
    """Quita la palabra de activación para dejar solo la orden."""
    variant_list = list(variants) if variants is not None else WAKE_VARIANTS
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
    lower = normalize_for_wake(text)
    return any(normalize_for_wake(v) in lower for v in variants)
