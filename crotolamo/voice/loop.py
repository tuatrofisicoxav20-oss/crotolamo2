"""Loop de voz concurrente con interrupción por turn_id (de GLaDOS, redo M3).

Arquitectura race-free: cuatro threads (Ear/Stt/Brain/Mouth) coordinados por colas
y un SharedState con turn_id monótono. Cada frase nace con su turn_id; el MouthThread
descarta las de turnos abortados. La interrupción es correcta POR CONSTRUCCIÓN: nada
de Events de "interrupción" que se limpian (races); toda invalidación pasa por
state.new_turn() + state.is_current().

Este módulo quedó como ORQUESTADOR (VoiceLoop, M3.7). Los threads viven en
threads.py y los adapters reales (mic/VAD) en adapters.py; se re-exportan aquí
por compatibilidad con los consumidores que importaban todo desde loop.
"""

from __future__ import annotations

import os
import queue
import threading

from crotolamo.logging_setup import get_logger
from crotolamo.voice.adapters import _RealMic, _SileroVad
from crotolamo.voice.state import SharedState, read_control_enabled
from crotolamo.voice.threads import (
    END,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
    _drain_queue,
)

# Re-exports de compatibilidad: los nombres históricos de loop.py siguen
# importables desde aquí aunque ahora vivan en threads.py / adapters.py.
__all__ = [
    "END",
    "Utterance",
    "MouthThread",
    "BrainThread",
    "SttThread",
    "EarThread",
    "VoiceLoop",
    "_drain_queue",
    "_RealMic",
    "_SileroVad",
]

log = get_logger("voice.loop")


class VoiceLoop:
    """Orquestador: arma los 4 threads y los apaga limpio (M3.7).

    Args:
        hud_publisher: callable(dict) -> None para publicar el estado del HUD en
                       tiempo real. Si es None (default), no se publica nada.
                       Usar ``make_file_publisher(path)`` de state.py para escritura
                       atómica a archivo.
    """

    def __init__(self, agent, stt, tts, wake_detector, *, allow_barge_in: bool = False,
                 silence_ms: int = 640, mic=None, wake_fn=None, vad_fn=None,
                 to_wav=None, hud_publisher=None, control_path=None) -> None:
        self.state = SharedState(publisher=hud_publisher)
        self.shutdown = threading.Event()
        # Canal de control inverso (panel -> loop). None = sin sondeo (tests).
        self.control_path = control_path
        # mtime del control.json la última vez que se leyó: el sondeo corre ~3
        # veces por segundo y releer+parsear el archivo en cada tick es E/S de
        # disco inútil; solo se relee cuando el mtime cambia.
        self._control_mtime: float | None = None
        self.stt_q: queue.Queue = queue.Queue()
        self.cmd_q: queue.Queue = queue.Queue()
        self.tts_q: queue.Queue = queue.Queue()
        self.tts = tts
        # M3.8: parámetros de mitigación de eco desde la config (config-first).
        try:
            from crotolamo.settings import get_settings

            vcfg = get_settings().voice
        except Exception as error:  # noqa: BLE001 — sin config, defaults
            log.warning("no pude cargar la config de voz (%s); uso defaults", error)
            vcfg = {}
        # Adapters reales por defecto; los tests inyectan fakes (no abre audio).
        # input_device: None (default) = micrófono default del sistema. Apúntalo a
        # "crotolamo_aec_source" tras activar el AEC (ver desktop/aec.sh).
        self._mic = mic or _RealMic(
            sample_rate=vcfg.get("sample_rate", 16000),
            device=vcfg.get("input_device"),
        )
        ear = EarThread(
            self._mic,
            wake_fn or wake_detector.feed,
            vad_fn or _SileroVad(),
            to_wav or stt._frames_to_wav,
            tts, self.stt_q, self.tts_q, self.state, self.shutdown,
            allow_barge_in=allow_barge_in, silence_ms=silence_ms,
            vad_threshold=vcfg.get("vad_threshold", 0.8),
            barge_in_grace_ms=vcfg.get("barge_in_grace_ms", 400),
            barge_in_threshold_margin=vcfg.get("barge_in_threshold_margin", 0.1),
            barge_in_min_chunks=vcfg.get("barge_in_min_chunks", 5),
            # None (default) = sin histéresis, comportamiento idéntico al previo.
            vad_neg_threshold=vcfg.get("vad_neg_threshold"),
        )
        self.threads = [
            ear,
            SttThread(stt, self.stt_q, self.cmd_q, self.state, self.shutdown,
                      hotwords=vcfg.get("hotwords",
                                        "Crotolamo, Tletl, Huevonitis")),
            BrainThread(agent, self.cmd_q, self.tts_q, self.state, self.shutdown),
            MouthThread(tts, self.tts_q, self.state, self.shutdown),
        ]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def _poll_control(self) -> None:
        """Sondea el canal de control (panel -> loop) SOLO si el archivo cambió.

        Se compara el st_mtime con el de la última lectura: si no cambió, no se
        toca el disco (antes se releía+parseaba el JSON ~3 veces por segundo).
        Archivo ausente => escucha activa (default-seguro), igual que antes.
        """
        try:
            mtime = os.stat(self.control_path).st_mtime
        except FileNotFoundError:
            # Sin archivo de control: default-seguro (escucha activa), como
            # hacía read_control_enabled. Si reaparece, el mtime nuevo forzará
            # la relectura.
            self._control_mtime = None
            self.state.set_enabled(True)
            return
        except OSError:
            # E/S rara (permisos, etc.): mismo default-seguro que la lectura.
            self.state.set_enabled(True)
            return
        if mtime == self._control_mtime:
            return
        self._control_mtime = mtime
        self.state.set_enabled(read_control_enabled(self.control_path))

    def run(self) -> None:
        self.start()
        # Estado inicial de la escucha desde el canal de control (default: activa).
        if self.control_path is not None:
            self._poll_control()
        try:
            while not self.shutdown.is_set():
                self.shutdown.wait(0.3)
                # Sondeo del canal de control (panel -> loop). set_enabled solo
                # publica si el valor cambia, y _poll_control solo relee el JSON
                # si el mtime cambió. Sin thread extra: reutilizamos esta espera.
                if self.control_path is not None:
                    self._poll_control()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Apagado ordenado con un límite conocido (B1): si el EarThread está
        BLOQUEADO dentro de mic.read() (PortAudio sin timeout), el join expira
        a los 2s y el thread queda vivo; al ser daemon, muere con el proceso
        (el atajo de teclado usa os._exit justamente por esto). Cerrar el mic
        antes del join suele desbloquear la lectura, pero no está garantizado.
        """
        self.shutdown.set()                       # 1) avisar a todos
        self.tts.stop()                           # 2) cortar audio en curso
        if hasattr(self._mic, "close"):
            self._mic.close()
        for t in self.threads:                    # 3) esperar (todos miran shutdown)
            t.join(timeout=2.0)
