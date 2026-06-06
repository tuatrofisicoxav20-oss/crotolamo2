"""Loop de voz concurrente con interrupción por turn_id (de GLaDOS, redo M3).

Arquitectura race-free: cuatro threads (Ear/Stt/Brain/Mouth) coordinados por colas
y un SharedState con turn_id monótono. Cada frase nace con su turn_id; el MouthThread
descarta las de turnos abortados. La interrupción es correcta POR CONSTRUCCIÓN: nada
de Events de "interrupción" que se limpian (races); toda invalidación pasa por
state.new_turn() + state.is_current().

Se construye por sub-fases (M3.3 Mouth, M3.4 Brain, M3.5 Stt, M3.6 Ear, M3.7 loop).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from crotolamo.logging_setup import get_logger
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.tts import split_sentences

log = get_logger("voice.loop")

END = object()  # sentinel: fin de la respuesta de un turno


@dataclass
class Utterance:
    text: str
    turn_id: int


class MouthThread(threading.Thread):
    """Habla las frases de la cola, descartando las de turnos viejos (M3.3)."""

    def __init__(self, tts, tts_queue: queue.Queue, state: SharedState,
                 shutdown: threading.Event) -> None:
        super().__init__(name="Mouth", daemon=True)
        self.tts = tts
        self.q = tts_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is END:
                # Terminó la respuesta de un turno; si nadie habló nuevo, a IDLE.
                if self.state.get_mode() == Mode.SPEAKING:
                    self.state.set_mode(Mode.IDLE)
                continue
            assert isinstance(item, Utterance)
            # Descarta frases de turnos abortados (barge-in o comando nuevo).
            if not self.state.is_current(item.turn_id):
                continue
            self.state.set_mode(Mode.SPEAKING)
            self.tts.speak(item.text)  # interrumpible (M3.1)


class BrainThread(threading.Thread):
    """Saca comandos, llama al agente y encola las frases con su turn_id (M3.4).

    Nota honesta: agent.handle_turn llama al LLM por HTTP de forma BLOQUEANTE; no se
    puede cancelar a media inferencia. Si hay barge-in mientras el LLM piensa, la
    inferencia vieja termina en background pero su salida se DESCARTA por el check de
    is_current(turn); el audio para al instante. Cancelar el LLM en vuelo es mejora
    futura (streaming con check de turno).
    """

    def __init__(self, agent, command_queue: queue.Queue, tts_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event) -> None:
        super().__init__(name="Brain", daemon=True)
        self.agent = agent
        self.cmd_q = command_queue
        self.tts_q = tts_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                command, turn = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # El turno ya cambió (barge-in mientras esperaba en cola): ignora.
            if not self.state.is_current(turn):
                continue
            self.state.set_mode(Mode.THINKING)
            try:
                reply = self.agent.handle_turn(command)
            except Exception as error:  # noqa: BLE001 - un turno roto no mata el loop
                log.warning("brain: %s", error)
                continue
            # Si interrumpieron mientras pensaba, NO encolar la respuesta vieja.
            if not self.state.is_current(turn):
                continue
            for sentence in split_sentences(reply):
                self.tts_q.put(Utterance(sentence, turn))
            self.tts_q.put(END)


class SttThread(threading.Thread):
    """Transcribe el audio de comandos, solo si el turno sigue vigente (M3.5)."""

    def __init__(self, stt, stt_queue: queue.Queue, command_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event) -> None:
        super().__init__(name="Stt", daemon=True)
        self.stt = stt
        self.in_q = stt_queue
        self.out_q = command_queue
        self.state = state
        self.shutdown = shutdown

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                audio_path, turn = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.state.is_current(turn):
                continue  # comando abortado antes de transcribir
            try:
                text = self.stt.transcribe(audio_path)
            except Exception as error:  # noqa: BLE001
                log.warning("stt: %s", error)
                text = ""
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            if text.strip() and self.state.is_current(turn):
                self.out_q.put((text, turn))


def _drain_queue(q: queue.Queue) -> None:
    """Saca todo de la cola sin bloquear."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


class EarThread(threading.Thread):
    """El oído: único thread que toca el micrófono y decide transiciones por audio
    (M3.6). Reúne wake word (M1), Silero VAD (M2) y barge-in. Dependencias inyectadas
    (mic/wake_fn/vad_fn/to_wav) para ser testeable con fakes sin micrófono real.

    Eco/barge-in: con auriculares es confiable. Con altavoces y SIN AEC, las capas de
    M3.8 reducen los auto-disparos pero no los eliminan; si molesta, usa --no-barge-in.
    AEC real (webrtc-audio-processing) es una mejora futura.
    """

    def __init__(self, mic, wake_fn, vad_fn, to_wav, tts,
                 stt_queue: queue.Queue, tts_queue: queue.Queue,
                 state: SharedState, shutdown: threading.Event, *,
                 vad_threshold: float = 0.8, silence_ms: int = 640,
                 max_command_s: float = 12.0, allow_barge_in: bool = False,
                 chunk_ms: float = 32.0, barge_in_grace_ms: float = 400,
                 barge_in_threshold_margin: float = 0.1,
                 barge_in_min_chunks: int = 5) -> None:
        super().__init__(name="Ear", daemon=True)
        self.mic = mic
        self.wake_fn = wake_fn
        self.vad_fn = vad_fn
        self.to_wav = to_wav
        self.tts = tts
        self.stt_q = stt_queue
        self.tts_q = tts_queue
        self.state = state
        self.shutdown = shutdown
        self.vad_threshold = vad_threshold
        self.allow_barge_in = allow_barge_in
        self._chunk_ms = chunk_ms
        self._silence_chunks = max(1, int(silence_ms / chunk_ms))
        self._max_chunks = int(max_command_s * 1000 / chunk_ms)
        # M3.8: mitigación de eco (gracia + margen de umbral + persistencia).
        self._grace_s = barge_in_grace_ms / 1000.0
        self._margin = barge_in_threshold_margin
        self._min_chunks = max(1, barge_in_min_chunks)
        self._turn = 0
        self._cmd_frames: list = []
        self._silent_run = 0
        self._cmd_count = 0
        self._prev_mode = Mode.IDLE
        self._speaking_since = 0.0
        self._barge_run = 0

    def _start_command(self, preroll: list | None = None) -> None:
        # M2: arrancar con el buffer de pre-activación para no perder el inicio.
        self._cmd_frames = list(preroll or [])
        self._silent_run = 0
        self._cmd_count = 0

    def _is_patron_voice(self, chunk, mode) -> bool:
        """Detecta voz del patrón con mitigación de eco (M3.8), 3 capas:
        1) ventana de gracia tras empezar a hablar, 2) umbral elevado en SPEAKING,
        3) persistencia (N chunks consecutivos), no un pico suelto.
        """
        # Capa 2: umbral más alto mientras suena el TTS (el eco infla el VAD).
        thr = self.vad_threshold + (self._margin if mode is Mode.SPEAKING else 0.0)
        is_voice = self.vad_fn(chunk) >= thr
        # Capa 1: gracia tras arrancar el habla (el eco del arranque es el peor).
        in_grace = (mode is Mode.SPEAKING
                    and (time.monotonic() - self._speaking_since) < self._grace_s)
        if is_voice and not in_grace:
            self._barge_run += 1
        else:
            self._barge_run = 0
        # Capa 3: exige racha sostenida.
        if self._barge_run >= self._min_chunks:
            self._barge_run = 0
            return True
        return False

    def _finish_command(self) -> None:
        wav = self.to_wav(self._cmd_frames)
        self.stt_q.put((wav, self._turn))
        self.state.set_mode(Mode.THINKING)

    def run(self) -> None:
        while not self.shutdown.is_set():
            chunk = self.mic.read()
            if chunk is None:
                time.sleep(0.005)  # sin audio ahora mismo: no hacer busy-spin
                continue
            mode = self.state.get_mode()
            # M3.8: marca el inicio del habla para la ventana de gracia.
            if mode is Mode.SPEAKING and self._prev_mode is not Mode.SPEAKING:
                self._speaking_since = time.monotonic()
                self._barge_run = 0
            self._prev_mode = mode

            if mode is Mode.IDLE:
                if self.wake_fn(chunk):
                    self._turn = self.state.new_turn()
                    self.state.set_mode(Mode.LISTENING)
                    self._start_command(preroll=[chunk])

            elif mode is Mode.LISTENING:
                self._cmd_frames.append(chunk)
                self._cmd_count += 1
                if self.vad_fn(chunk) >= self.vad_threshold:
                    self._silent_run = 0
                else:
                    self._silent_run += 1
                if self._silent_run >= self._silence_chunks or self._cmd_count >= self._max_chunks:
                    self._finish_command()

            elif mode in (Mode.THINKING, Mode.SPEAKING):
                if self.allow_barge_in and self._is_patron_voice(chunk, mode):
                    self.tts.stop()            # corta el audio YA
                    _drain_queue(self.tts_q)   # tira las frases viejas
                    self._turn = self.state.new_turn()  # invalida turnos viejos
                    self.state.set_mode(Mode.LISTENING)
                    self._start_command(preroll=[chunk])


# --- adapters reales (solo en hardware; los tests inyectan fakes) ---
class _RealMic:
    """Micrófono real: InputStream de sounddevice abierto perezosamente."""

    def __init__(self, sample_rate: int = 16000, frame: int = 512) -> None:
        self.sample_rate = sample_rate
        self.frame = frame
        self._stream = None

    def read(self):
        import numpy as np
        import sounddevice as sd

        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32"
            )
            self._stream.start()
        block, _ = self._stream.read(self.frame)
        return np.squeeze(block)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None


class _SileroVad:
    """Adapter de Silero VAD (M2) como vad_fn(chunk) -> prob."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._model: Any = None

    def __call__(self, chunk) -> float:
        import numpy as np
        import torch
        from silero_vad import load_silero_vad

        if self._model is None:
            self._model = load_silero_vad(onnx=True)
        arr = np.asarray(chunk, dtype=np.float32)
        return float(self._model(torch.from_numpy(arr.copy()), self.sample_rate))


class VoiceLoop:
    """Orquestador: arma los 4 threads y los apaga limpio (M3.7)."""

    def __init__(self, agent, stt, tts, wake_detector, *, allow_barge_in: bool = False,
                 silence_ms: int = 640, mic=None, wake_fn=None, vad_fn=None,
                 to_wav=None) -> None:
        self.state = SharedState()
        self.shutdown = threading.Event()
        self.stt_q: queue.Queue = queue.Queue()
        self.cmd_q: queue.Queue = queue.Queue()
        self.tts_q: queue.Queue = queue.Queue()
        self.tts = tts
        # M3.8: parámetros de mitigación de eco desde la config (config-first).
        try:
            from crotolamo.settings import get_settings

            vcfg = get_settings().voice
        except Exception:
            vcfg = {}
        # Adapters reales por defecto; los tests inyectan fakes (no abre audio).
        self._mic = mic or _RealMic()
        ear = EarThread(
            self._mic,
            wake_fn or wake_detector.feed,
            vad_fn or _SileroVad(),
            to_wav or stt._frames_to_wav,
            tts, self.stt_q, self.tts_q, self.state, self.shutdown,
            allow_barge_in=allow_barge_in, silence_ms=silence_ms,
            barge_in_grace_ms=vcfg.get("barge_in_grace_ms", 400),
            barge_in_threshold_margin=vcfg.get("barge_in_threshold_margin", 0.1),
            barge_in_min_chunks=vcfg.get("barge_in_min_chunks", 5),
        )
        self.threads = [
            ear,
            SttThread(stt, self.stt_q, self.cmd_q, self.state, self.shutdown),
            BrainThread(agent, self.cmd_q, self.tts_q, self.state, self.shutdown),
            MouthThread(tts, self.tts_q, self.state, self.shutdown),
        ]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def run(self) -> None:
        self.start()
        try:
            while not self.shutdown.is_set():
                self.shutdown.wait(0.3)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self.shutdown.set()                       # 1) avisar a todos
        self.tts.stop()                           # 2) cortar audio en curso
        if hasattr(self._mic, "close"):
            self._mic.close()
        for t in self.threads:                    # 3) esperar (todos miran shutdown)
            t.join(timeout=2.0)
