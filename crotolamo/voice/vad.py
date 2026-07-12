"""Decisión de "esto sigue siendo voz" con histéresis. Función pura, testeable.

Vive aquí (y no duplicada) porque la usan DOS rutas distintas:
  - `loop.EarThread` (loop concurrente, activo con use_oww=true)
  - `stt._record_silero` (modo simple, la ruta activa hoy con use_oww=false)

Histéresis: el umbral para ENTRAR en voz es alto (no queremos despertar con
cualquier ruido), pero el umbral para MANTENERSE en ella debe ser más bajo. Sin
esto, una micro-pausa (respirar, pensar a media frase) cae bajo el umbral de
entrada y empieza a contar silencio -> el comando se corta a media frase.

Es el mismo principio que un termostato: no enciendes y apagas en el mismo grado,
porque oscilarías sin parar en la frontera.
"""

from __future__ import annotations


def resolve_neg_threshold(threshold: float, neg_threshold: float | None) -> float:
    """Umbral de mantenimiento efectivo.

    None (default) => igual al de entrada, es decir SIN histéresis: reproduce
    exactamente el comportamiento histórico. Nunca se permite que el umbral de
    salida sea MAYOR que el de entrada (eso invertiría la histéresis y cortaría
    aún antes); en ese caso se ignora y se cae a `threshold`.
    """
    if neg_threshold is None:
        return threshold
    if neg_threshold > threshold:
        return threshold
    return neg_threshold


def is_voice(prob: float, *, speaking: bool, threshold: float,
             neg_threshold: float) -> bool:
    """¿Este chunk cuenta como voz?

    - Si aún NO hablamos: exige el umbral alto (`threshold`) para entrar.
    - Si YA hablamos: basta el umbral bajo (`neg_threshold`) para seguir.
    """
    return prob >= (neg_threshold if speaking else threshold)
