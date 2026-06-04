"""Correcciones de errores típicos de Whisper en español MX. Migrado de C1.

'Cinta adhesiva inteligente': arregla cómo Whisper destroza ciertas frases antes
de que el texto llegue al agente. Pura y testeable, sin dependencias de audio.
"""

from __future__ import annotations

import re

# Errores frecuentes de Whisper -> forma correcta.
_REPLACEMENTS = {
    "mis hit hoyo": "mi escritorio",
    "mis sitio": "mi escritorio",
    "mi escrit hoyo": "mi escritorio",
    "mi escrito yo": "mi escritorio",
    "mi escriptorio": "mi escritorio",
    "you tube": "youtube",
    "yutub": "youtube",
    "yutube": "youtube",
    "git hub": "github",
    "gijub": "github",
    "espoti fai": "spotify",
    "espotifai": "spotify",
    "guevonitis": "huevonitis",
    "tle tl": "tletl",
}


def normalize_text(text: str) -> str:
    """Corrige errores comunes de Whisper. Devuelve el texto en minúsculas y limpio."""
    text = re.sub(r"\s+", " ", text.strip())
    lower = text.lower()

    for wrong, right in _REPLACEMENTS.items():
        lower = lower.replace(wrong, right)

    # 'crear una carpeta' / 'qué es una carpeta' -> 'crea una carpeta'
    lower = re.sub(r"^(?:qu[eé] es|que se|es)\s+una\s+carpeta", "crea una carpeta", lower)
    lower = re.sub(r"^crear\s+una\s+carpeta", "crea una carpeta", lower)

    return lower.strip()
