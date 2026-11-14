"""Decisión de "esto sigue siendo voz" con histéresis, el endpointing de una orden
y los valores de fábrica de la captura. Puro y testeable (sin audio ni modelos).

Vive aquí (y no duplicado) porque lo usan DOS rutas distintas:
  - `threads.EarThread` (loop concurrente, activo con use_oww=true)
  - `stt._record_silero` (modo simple, la ruta activa hoy con use_oww=false)

Histéresis: el umbral para ENTRAR en voz es alto (no queremos despertar con
cualquier ruido), pero el umbral para MANTENERSE en ella debe ser más bajo. Sin
esto, una micro-pausa (respirar, pensar a media frase) cae bajo el umbral de
entrada y empieza a contar silencio -> el comando se corta a media frase.

Es el mismo principio que un termostato: no enciendes y apagas en el mismo grado,
porque oscilarías sin parar en la frontera.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceDefaults:
    """Valores de fábrica de la captura y del endpointing: lo que vale cuando falta
    la clave de [voice] (o no hay config: tests y scripts) y las constantes que no
    son clave. UNA tabla para los dos loops y todos los módulos (stt, threads,
    loop, el listener); antes cada uno escribía su literal y algunos ya no
    coincidían. Donde hoy los loops difieren se guardan LOS DOS valores, cada uno
    con su lector: unificarlos cambia un comportamiento (decisión K del dueño). Con
    la config de fábrica todas las claves de aquí están puestas: estos valores no
    se alcanzan.
    """

    sample_rate: int = 16000
    vad_backend: str = "energy"  # modo simple; el oído concurrente usa Silero siempre
    vad_threshold: float = 0.8
    vad_neg_threshold: float | None = None  # sin histéresis
    vad_preactivation_ms: int = 800
    # vad_silence_ms sin la clave: 800 en el listener (ListenerConfig, que se lo pasa
    # a los DOS loops); 640 en STT, VoiceLoop y EarThread construidos sin él.
    silence_ms_listener: int = 800
    silence_ms: int = 640
    # hotwords sin la clave: el loop simple no sesga (None); el concurrente sí.
    hotwords_concurrent: str = "Crotolamo, Tletl, Huevonitis"
    barge_in_grace_ms: int = 400
    barge_in_threshold_margin: float = 0.1
    barge_in_min_chunks: int = 5
    endpoint_rounds: int = 2
    endpoint_continue_s: float = 2.5
    # Constantes (no son clave): la espera de la orden tras el wake o en el
    # followup... (simple y concurrente), el tope de una orden y la gracia del oído
    # tras un turno (eco del TTS).
    command_start_timeout_s: float = 4.0
    max_command_s: float = 12.0
    post_speak_grace_s: float = 1.5


DEFAULTS = VoiceDefaults()

# Lo que devuelve Endpointer.feed().
ESPERANDO = "waiting"  # aún sin voz (en la ventana de espera)
HABLANDO = "speech"  # la orden está sonando
FIN = "done"  # la orden se cerró (silencio o tope); sin voz, `speaking` es False
SIN_VOZ = "timeout"  # se agotó la espera sin una sola voz


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


class Endpointer:
    """Dónde empieza y dónde acaba una orden, chunk a chunk: la ÚNICA implementación.

    Antes había dos copias que ya divergían en detalles: la grabación del modo
    simple (stt._record_silero, por llamada y sobre una fuente ya abierta) y la
    rama LISTENING del oído concurrente (threads.EarThread). Las dos son hoy un
    bucle fino sobre esto, con las MISMAS decisiones que tenían (lo comparan los
    tests contra copias literales del código anterior).

    armed=True: aún sin voz. Se guarda la pre-activación (anillo de `pre_chunks`
    chunks, el actual incluido, que empieza con `preroll`), y entrar en voz exige
    `threshold`; si en `start_chunks` chunks no hubo voz -> SIN_VOZ. Con voz, los
    frames empiezan con ese anillo y ya basta `neg_threshold` para seguir en voz
    (histéresis); `silence_chunks` seguidos sin voz -> FIN. armed=False: en voz
    desde el primer chunk, con `preroll` como principio de la orden (el barge-in).
    Llegar a `max_chunks` cierra la ventana -> FIN (con o sin voz).

    Los plazos cuentan por ventana: `ahead` chunks ya capturados antes de la ventana
    la alargan (la fuente abierta de capture.MicSession: lo que sonó mientras
    Whisper decodificaba) y restart() abre otra ventana de espera conservando la
    pre-activación. `first` es el índice, entre todos los chunks recibidos, del
    primero de `frames` (el más viejo de la pre-activación).
    """

    def __init__(self, *, threshold: float, neg_threshold: float, silence_chunks: int,
                 max_chunks: int, start_chunks: int, pre_chunks: int, armed: bool = True,
                 preroll: Iterable[Any] = (), ahead: int = 0) -> None:
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.silence_chunks = silence_chunks
        self.max_chunks = max_chunks
        self.start_chunks = start_chunks
        self.speaking = not armed
        self.frames: list[Any] = [] if armed else list(preroll)
        self._pre: deque[Any] = deque(preroll if armed else (), maxlen=pre_chunks)
        self.count = -ahead  # chunks de la ventana actual
        self.fed = 0  # chunks recibidos en total
        self.first = 0
        self.silent_run = 0

    def feed(self, chunk: Any, prob: float) -> str:
        """Un chunk y su probabilidad de voz -> ESPERANDO, HABLANDO, FIN o SIN_VOZ."""
        self.count += 1
        self.fed += 1
        if not self.speaking:
            self._pre.append(chunk)
            # Entrar exige el umbral ALTO (no despertar con cualquier ruido).
            if is_voice(prob, speaking=False, threshold=self.threshold,
                        neg_threshold=self.neg_threshold):
                self.speaking = True
                self.first = self.fed - len(self._pre)
                self.frames = list(self._pre)  # M2.2: anteponer la pre-activación
                self.silent_run = 0
            elif self.count >= self.start_chunks:
                return SIN_VOZ
        else:
            self.frames.append(chunk)
            # Mantenerse basta con el umbral BAJO (histéresis).
            if is_voice(prob, speaking=True, threshold=self.threshold,
                        neg_threshold=self.neg_threshold):
                self.silent_run = 0
            else:
                self.silent_run += 1
                if self.silent_run >= self.silence_chunks:
                    return FIN
        if self.count >= self.max_chunks:
            return FIN
        return HABLANDO if self.speaking else ESPERANDO

    def expired(self) -> bool:
        """La ventana ya llegó al tope (antes de leer el chunk siguiente)."""
        return self.count >= self.max_chunks

    def restart(self) -> None:
        """Otra ventana de espera tras SIN_VOZ: plazos desde aquí, sin perder la
        pre-activación (lo que una grabación nueva por llamada sí perdía)."""
        self.count = 0
