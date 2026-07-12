"""Función pura de histéresis del VAD (crotolamo/voice/vad.py).

La comparten las DOS rutas de captura: el EarThread del loop concurrente y
`stt._record_silero` del modo simple. Testearla aquí cubre la decisión del modo
simple, que es la ruta activa mientras `use_oww = false` y que no se puede
ejercitar en CI (abre el micrófono real).
"""

from __future__ import annotations

import pytest

from crotolamo.voice.vad import is_voice, resolve_neg_threshold


# --- resolve_neg_threshold -------------------------------------------------

def test_none_significa_sin_histeresis():
    """El default debe reproducir EXACTAMENTE el comportamiento histórico."""
    assert resolve_neg_threshold(0.8, None) == 0.8


def test_valor_explicito_se_respeta():
    assert resolve_neg_threshold(0.8, 0.35) == 0.35


def test_neg_mayor_que_entrada_se_ignora():
    """Un neg_threshold > threshold invertiría la histéresis y cortaría ANTES.

    Es un error de configuración: se ignora y se cae a `threshold` (sin histéresis)
    en vez de empeorar el truncado silenciosamente.
    """
    assert resolve_neg_threshold(0.8, 0.95) == 0.8


def test_neg_igual_a_entrada_es_valido():
    assert resolve_neg_threshold(0.8, 0.8) == 0.8


# --- is_voice --------------------------------------------------------------

@pytest.mark.parametrize("prob,esperado", [(0.95, True), (0.8, True), (0.79, False), (0.5, False)])
def test_entrar_exige_umbral_alto(prob, esperado):
    """Sin hablar aún: hace falta superar el umbral de ENTRADA (0.8)."""
    assert is_voice(prob, speaking=False, threshold=0.8, neg_threshold=0.35) is esperado


@pytest.mark.parametrize("prob,esperado", [(0.95, True), (0.5, True), (0.35, True), (0.34, False), (0.05, False)])
def test_mantenerse_basta_umbral_bajo(prob, esperado):
    """Ya hablando: basta el umbral de MANTENIMIENTO (0.35)."""
    assert is_voice(prob, speaking=True, threshold=0.8, neg_threshold=0.35) is esperado


def test_la_micropausa_solo_sobrevive_con_histeresis():
    """El bug real: prob=0.5 es una micro-pausa (respirar a media frase).

    Sin histéresis cuenta como SILENCIO (y acaba truncando el comando).
    Con histéresis sigue contando como VOZ.
    """
    micro_pausa = 0.5
    sin_hist = resolve_neg_threshold(0.8, None)      # 0.8
    con_hist = resolve_neg_threshold(0.8, 0.35)      # 0.35

    assert not is_voice(micro_pausa, speaking=True, threshold=0.8, neg_threshold=sin_hist)
    assert is_voice(micro_pausa, speaking=True, threshold=0.8, neg_threshold=con_hist)


def test_silencio_real_corta_incluso_con_histeresis():
    """La histéresis no debe volver sordo al corte: 0.05 es silencio de verdad."""
    assert not is_voice(0.05, speaking=True, threshold=0.8, neg_threshold=0.35)
