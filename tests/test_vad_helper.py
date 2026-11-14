"""Función pura de histéresis del VAD (crotolamo/voice/vad.py).

La comparten las DOS rutas de captura: el EarThread del loop concurrente y
`stt._record_silero` del modo simple. Testearla aquí cubre la decisión del modo
simple, que es la ruta activa mientras `use_oww = false` y que no se puede
ejercitar en CI (abre el micrófono real).
"""

from __future__ import annotations

import random

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


# --- Endpointer: UN endpointing para las dos rutas de captura ----------------------
# Causa raíz voice-pipeline-forked-audio-owned-per-call, paso 6 del diseño.
# Referencias: copias LITERALES de las decisiones de antes (sin E/S). Los chunks son
# enteros (su índice) para comparar qué frames sale de cada ruta. Los grabadores
# REALES frente a estas mismas copias, en
# tests/test_perf_voice_pipeline_forked_audio_owned_per_call.py.

def _ref_record_silero(probs, *, threshold, neg, silence_chunks, max_chunks, start_chunks,
                       pre_chunks):
    """El bucle de STT._record_silero (por llamada): (frames, first, leídos)."""
    from collections import deque

    from crotolamo.voice.vad import is_voice

    pre_buffer: deque = deque(maxlen=pre_chunks)
    frames: list = []
    first = 0
    speaking = False
    silent_run = 0
    leidos = 0
    for i in range(max_chunks):
        block = i
        leidos += 1
        prob = probs[i]
        if not speaking:
            pre_buffer.append(block)
            if is_voice(prob, speaking=False, threshold=threshold, neg_threshold=neg):
                speaking = True
                first = i - len(pre_buffer) + 1
                frames.extend(pre_buffer)
                silent_run = 0
            elif i >= start_chunks:
                break
        else:
            frames.append(block)
            if is_voice(prob, speaking=True, threshold=threshold, neg_threshold=neg):
                silent_run = 0
            else:
                silent_run += 1
                if silent_run >= silence_chunks:
                    break
    return frames, first, leidos


def _ref_oido(preroll, voiced, probs, *, threshold, neg, silence_chunks, max_chunks,
              start_chunks, pre_chunks):
    """La rama LISTENING de EarThread._on_chunk (y _start_command/_finish_command):
    (frames si hubo voz o None, chunks consumidos)."""
    from crotolamo.voice.vad import is_voice

    cmd_frames = list(preroll or [])
    silent_run = cmd_count = 0
    for k, prob in enumerate(probs):
        chunk = 1000 + k
        cmd_frames.append(chunk)
        cmd_count += 1
        if is_voice(prob, speaking=voiced, threshold=threshold, neg_threshold=neg):
            silent_run = 0
            voiced = True
        elif voiced:
            silent_run += 1
        else:
            if len(cmd_frames) > pre_chunks:
                del cmd_frames[0]
            if cmd_count >= start_chunks:
                return (cmd_frames if voiced else None), k + 1
        if silent_run >= silence_chunks or cmd_count >= max_chunks:
            return (cmd_frames if voiced else None), k + 1
    return "sigue", len(probs)


def _probs_al_azar(rng: random.Random, n: int) -> list[float]:
    """Rachas de silencio, voz, micro-pausas y picos: lo que de verdad decide."""
    out: list[float] = []
    while len(out) < n:
        out.extend([rng.choice([0.0, 0.05, 0.3, 0.5, 0.85, 0.95])] * rng.randint(1, 12))
    return out[:n]


def _parametros(rng: random.Random) -> dict:
    return dict(threshold=0.8, neg=rng.choice([0.8, 0.35]), silence_chunks=rng.randint(1, 8),
                max_chunks=rng.randint(1, 70), start_chunks=rng.randint(0, 45),
                pre_chunks=rng.randint(1, 12))


def test_endpointer_tabla_de_decisiones():
    """Casos a mano: espera, entrada con pre-activación, histéresis y los tres finales."""
    from crotolamo.voice.vad import ESPERANDO, FIN, HABLANDO, SIN_VOZ, Endpointer

    def correr(probs, **kw):
        ep = Endpointer(threshold=0.8, neg_threshold=0.35, silence_chunks=2, max_chunks=10,
                        start_chunks=4, pre_chunks=3, **kw)
        return [ep.feed(i, p) for i, p in enumerate(probs)], ep

    estados, ep = correr([0.0, 0.0, 0.9, 0.5, 0.1, 0.1])
    assert estados == [ESPERANDO, ESPERANDO, HABLANDO, HABLANDO, HABLANDO, FIN]
    assert ep.frames == [0, 1, 2, 3, 4, 5] and ep.first == 0
    estados, ep = correr([0.0] * 4)
    assert estados == [ESPERANDO] * 3 + [SIN_VOZ] and not ep.speaking and ep.frames == []
    estados, ep = correr([0.0, 0.0, 0.0, 0.85, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    assert estados[-1] == FIN and ep.speaking  # el tope cierra aunque siga la voz
    assert ep.frames[:3] == [1, 2, 3] and ep.first == 1  # anillo de 3, el actual incluido
    estados, ep = correr([0.1, 0.1], armed=False, preroll=["b1", "b2"])
    assert estados == [HABLANDO, FIN] and ep.frames == ["b1", "b2", 0, 1]
    # Lo ya capturado alarga la ventana; restart() abre otra sin perder el anillo.
    ep = Endpointer(threshold=0.8, neg_threshold=0.8, silence_chunks=2, max_chunks=10,
                    start_chunks=2, pre_chunks=3, ahead=3)
    assert [ep.feed(i, 0.0) for i in range(5)] == [ESPERANDO] * 4 + [SIN_VOZ]
    ep.restart()
    assert ep.feed(5, 0.0) == ESPERANDO and ep.feed(6, 0.9) == HABLANDO
    assert ep.frames == [4, 5, 6] and ep.first == 4


def test_endpointer_decide_igual_que_record_silero_de_antes():
    """Con la convención del modo simple (start_chunks + 1: esperaba un chunk más
    que su plazo), el Endpointer da los MISMOS frames, first y lecturas."""
    from crotolamo.voice.vad import FIN, SIN_VOZ, Endpointer

    rng = random.Random(6)
    for _ in range(1500):
        p = _parametros(rng)
        probs = _probs_al_azar(rng, p["max_chunks"])
        ep = Endpointer(threshold=p["threshold"], neg_threshold=p["neg"],
                        silence_chunks=p["silence_chunks"], max_chunks=p["max_chunks"],
                        start_chunks=p["start_chunks"] + 1, pre_chunks=p["pre_chunks"])
        leidos = 0
        for i in range(p["max_chunks"]):
            leidos += 1
            if ep.feed(i, probs[i]) in (FIN, SIN_VOZ):
                break
        assert (ep.frames, ep.first, leidos) == _ref_record_silero(probs, **p), (p, probs)


def test_endpointer_decide_igual_que_el_oido_de_antes():
    """Con la convención del oído (pre_chunks + 1: guardaba pre_chunks ANTES del
    primero con voz, y el wake al principio), mismos frames y el mismo chunk final."""
    from crotolamo.voice.vad import FIN, SIN_VOZ, Endpointer

    rng = random.Random(16)
    for _ in range(1500):
        p = _parametros(rng)
        p["start_chunks"] = max(1, p["start_chunks"])
        voiced = rng.random() < 0.3
        preroll = ["b1", "b2", "b3"] if voiced else ["wake"]
        probs = _probs_al_azar(rng, 80)
        ep = Endpointer(threshold=p["threshold"], neg_threshold=p["neg"],
                        silence_chunks=p["silence_chunks"], max_chunks=p["max_chunks"],
                        start_chunks=p["start_chunks"], pre_chunks=p["pre_chunks"] + 1,
                        armed=not voiced, preroll=preroll)
        nuevo: tuple = ("sigue", len(probs))
        for k, prob in enumerate(probs):
            if ep.feed(1000 + k, prob) in (FIN, SIN_VOZ):
                nuevo = ((ep.frames if ep.speaking else None), k + 1)
                break
        assert nuevo == _ref_oido(preroll, voiced, probs, **p), (p, voiced, probs)
