"""Histéresis del VAD en el EarThread (contra el corte a media frase).

Sin histéresis, el umbral para MANTENERSE en voz es el mismo que para ENTRAR
(0.8): una micro-pausa (respirar, pensar) cae debajo y empieza a contar silencio,
truncando el comando. Con histéresis (neg_threshold ~0.35) la pausa sigue siendo
"voz" y el comando no se corta.

El default (vad_neg_threshold=None) debe ser IDÉNTICO al comportamiento previo.
"""

from __future__ import annotations

import queue
import threading

from crotolamo.voice.threads import EarThread
from crotolamo.voice.state import Mode, SharedState


class _ScriptedMic:
    """Devuelve una secuencia fija de chunks y luego None (se agota)."""

    def __init__(self, n: int) -> None:
        self._left = n
        # Se enciende cuando el oído vuelve a leer tras el último chunk: ya
        # procesó todo el guion (lee y procesa de a uno).
        self.agotado = threading.Event()

    def read(self):
        if self._left <= 0:
            self.agotado.set()
            return None
        self._left -= 1
        return [0.0]


def _run_ear(vad_probs: list[float], *, neg: float | None,
             silence_ms: int = 100, chunk_ms: float = 32.0):
    """Corre el EarThread en LISTENING sobre una secuencia de probabilidades VAD.

    Devuelve (se_termino_el_comando, chunks_consumidos).
    """
    state = SharedState()
    state.set_mode(Mode.LISTENING)
    shutdown = threading.Event()
    stt_q: queue.Queue = queue.Queue()

    it = iter(vad_probs)
    consumed = {"n": 0}

    def vad_fn(_chunk) -> float:
        consumed["n"] += 1
        try:
            return next(it)
        except StopIteration:
            return 1.0  # tras el guion, "voz" para no cerrar por accidente

    mic = _ScriptedMic(len(vad_probs))
    ear = EarThread(
        mic, lambda c: False, vad_fn, lambda f: "WAV", None,
        stt_q, queue.Queue(), state, shutdown,
        vad_threshold=0.8, silence_ms=silence_ms, chunk_ms=chunk_ms,
        vad_neg_threshold=neg,
    )
    ear._start_command()
    t = threading.Thread(target=ear.run)
    t.start()
    # Esperar a que el oído procese TODO el guion (un sleep fijo dejaba pasar
    # "no se cortó" aunque el oído no hubiera procesado nada).
    agotado = mic.agotado.wait(2.0)
    shutdown.set()
    t.join(timeout=2)
    assert agotado, "el oído no procesó el guion"
    # Si terminó el comando, encoló el WAV y pasó a THINKING.
    return (not stt_q.empty()), consumed["n"]


# silence_ms=100 y chunk_ms=32 -> silence_chunks = 3 chunks seguidos bajo umbral.
_MICRO_PAUSA = [0.95, 0.95, 0.5, 0.5, 0.5, 0.95, 0.95]


def test_sin_histeresis_la_micropausa_corta_el_comando():
    """Comportamiento HISTÓRICO: 3 chunks a 0.5 (<0.8) cierran el comando."""
    terminado, _ = _run_ear(_MICRO_PAUSA, neg=None)
    assert terminado, "sin histéresis, la micro-pausa DEBE cortar (comportamiento previo)"


def test_con_histeresis_la_micropausa_no_corta():
    """Con neg_threshold=0.35, los chunks a 0.5 siguen contando como voz."""
    terminado, consumidos = _run_ear(_MICRO_PAUSA, neg=0.35)
    assert not terminado, "con histéresis, la micro-pausa NO debe cortar el comando"
    assert consumidos == len(_MICRO_PAUSA)  # todo el guion se oyó EN la orden


def test_con_histeresis_el_silencio_real_si_corta():
    """La histéresis no debe impedir el corte cuando de verdad hay silencio."""
    silencio_real = [0.95, 0.95, 0.05, 0.05, 0.05, 0.05]
    terminado, _ = _run_ear(silencio_real, neg=0.35)
    assert terminado, "un silencio real (<0.35) SÍ debe cerrar el comando"


def test_default_none_equivale_a_sin_histeresis():
    """vad_neg_threshold=None => neg == vad_threshold (inerte)."""
    ear = EarThread(
        _ScriptedMic(0), lambda c: False, lambda c: 0.0, lambda f: "", None,
        queue.Queue(), queue.Queue(), SharedState(), threading.Event(),
        vad_threshold=0.8, vad_neg_threshold=None,
    )
    assert ear.vad_neg_threshold == 0.8


def test_neg_threshold_explicito_se_respeta():
    ear = EarThread(
        _ScriptedMic(0), lambda c: False, lambda c: 0.0, lambda f: "", None,
        queue.Queue(), queue.Queue(), SharedState(), threading.Event(),
        vad_threshold=0.8, vad_neg_threshold=0.35,
    )
    assert ear.vad_neg_threshold == 0.35
