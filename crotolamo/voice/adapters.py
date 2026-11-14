"""Adapters reales de audio del loop de voz (extraídos de loop.py).

Solo se usan en hardware; los tests inyectan fakes. Imports pesados (numpy,
sounddevice) perezosos y CACHEADOS: el hot path corre ~31 veces por
segundo, así que resolver los módulos en cada frame (aunque el import cacheado
sea "barato") es costo puro sin motivo.
"""

from __future__ import annotations

import threading
from typing import Any

from crotolamo.logging_setup import get_logger

log = get_logger("voice.loop")


class _RealMic:
    """Micrófono real: InputStream de sounddevice abierto perezosamente.

    device: nombre o índice del dispositivo de entrada. None (default) = el default
    del sistema. Se usa para apuntar SOLO a Crotolamo al source con cancelación de
    eco ("crotolamo_aec_source") sin cambiar el default global del escritorio.
    """

    def __init__(self, sample_rate: int = 16000, frame: int = 512, device=None) -> None:
        self.sample_rate = sample_rate
        self.frame = frame
        self.device = device
        self._stream = None
        # close() es TERMINAL (V17): VoiceLoop.stop() lo llama desde el hilo
        # principal con el oído todavía en su bucle. Sin esta marca, el read()
        # siguiente veía _stream en None y abría un InputStream NUEVO que ya
        # nadie cerraba. El lock solo protege el relevo del stream (publicarlo
        # / retirarlo), nunca la lectura ni la apertura: close() no espera a
        # PortAudio.
        self._closed = False
        self._lock = threading.Lock()

    def read(self):
        """Lee un frame BLOQUEANTE (~32ms a 16kHz con frame=512).

        Devuelve None si el micrófono ya se cerró (el oído trata None como
        "sin audio" y vuelve a mirar shutdown).

        LÍMITE CONOCIDO (B1): PortAudio no ofrece read() con timeout; si el
        driver/dispositivo se atora, esta llamada puede colgar el EarThread y
        VoiceLoop.stop() no lo cierra por las buenas (por eso el thread es
        daemon y el apagado duro usa os._exit). No se cambió a modo no
        bloqueante a propósito: alteraría el timing del hot path (~31 fps) por
        un fallo que en la práctica no se ha observado; si algún día aparece,
        el arreglo es un callback de sounddevice + cola con timeout.
        """
        import numpy as np
        import sounddevice as sd

        if self._closed:
            return None
        try:
            stream = self._stream
            if stream is None:
                stream = self._open(sd)
                if stream is None:
                    return None
            block, _ = stream.read(self.frame)
        except Exception:
            # stop() cierra el stream desde OTRO hilo para desbloquear esta
            # lectura: que PortAudio la corte con un error es entonces el
            # apagado, no un fallo del micro (propagarlo tumbaba al oído con un
            # traceback). Sin close() de por medio, el error es real y sube.
            if self._closed:
                return None
            raise
        return np.squeeze(block)

    def _open(self, sd):
        """Abre y publica el stream; None si close() llegó mientras se abría."""
        stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=self.device,
        )
        with self._lock:
            closed = self._closed
            if not closed:
                self._stream = stream
        if closed:
            # close() corrió durante la apertura y no vio este stream: lo
            # cierra quien lo abrió, o quedaría huérfano.
            self._shut(stream)
            return None
        stream.start()
        return stream

    @staticmethod
    def _shut(stream) -> None:
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stream, self._stream = self._stream, None
        if stream is not None:
            self._shut(stream)


class _SileroVad:
    """Adapter de Silero VAD (M2) como vad_fn(chunk) -> prob.

    Si silero-vad u onnxruntime no están instalados, cae a un VAD por energía RMS
    que devuelve 0.0 o 1.0 (escalado para ser comparable con el umbral 0-1 del
    loop). El fallback se activa UNA sola vez (no reintenta el import cada chunk) y
    se reporta en el log UNA vez para no inundar.

    Rendimiento: numpy se resuelve UNA vez (lazy, en el primer frame) y se cachea
    en self._np; antes el import corría en cada chunk (~31/s). El modelo recibe el
    chunk como numpy (runtime.SileroOnnx): torch ya no se importa, y con él se
    fueron ~218 MB y ~950 módulos que el primer chunk cargaba EN el hilo del oído.
    """

    _ENERGY_FLOOR = 0.01  # RMS mínimo para considerar voz en modo fallback

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._model: Any = None
        self._fallback: bool = False  # True = usar energía RMS
        self._np: Any = None     # numpy cacheado (lazy, primer frame)

    def reset(self) -> None:
        """Limpia el estado oculto de la RNN. Silero acumula contexto entre chunks:
        eso ayuda DENTRO de una frase, pero arrastrado de un turno al siguiente
        degrada la precisión del VAD conforme avanza la sesión.
        """
        if self._model is not None:
            self._model.reset_states()

    def __call__(self, chunk) -> float:
        if self._fallback:
            return self._energy_vad(chunk)

        try:
            if self._model is None:
                # Lazy y UNA sola vez: imports pesados fuera del hot path.
                import numpy

                # Comparte la instancia cacheada de stt: antes cada uno cargaba su
                # propia copia del ONNX, duplicando el modelo en RAM sin motivo.
                from crotolamo.voice.stt import _get_silero_vad

                self._np = numpy
                self._model = _get_silero_vad()
            # ascontiguousarray evita la copia extra que hacía .copy(): solo copia
            # si el chunk no es ya un array float32 contiguo.
            arr = self._np.ascontiguousarray(chunk, dtype=self._np.float32)
            return float(self._model(arr, self.sample_rate))
        except (ImportError, ModuleNotFoundError) as exc:
            log.warning(
                "silero-vad no disponible (%s); cayendo a VAD por energía. "
                "Para mejor VAD: pip install silero-vad torch",
                exc,
            )
            self._fallback = True
            return self._energy_vad(chunk)
        except Exception as exc:  # noqa: BLE001 — error de carga o inferencia
            log.warning(
                "silero-vad falló (%s); cayendo a VAD por energía permanentemente.",
                exc,
            )
            self._fallback = True
            self._model = None
            return self._energy_vad(chunk)

    def _energy_vad(self, chunk) -> float:
        """VAD de energía RMS como fallback. Devuelve 1.0 si hay voz, 0.0 si no.

        El valor retornado es 0.0 o 1.0 para ser directamente comparable con el
        umbral vad_threshold (por defecto 0.8) del EarThread.
        """
        try:
            if self._np is None:
                import numpy

                self._np = numpy
            np = self._np
            arr = np.asarray(chunk, dtype=np.float32)
            rms = float(np.sqrt(np.mean(arr ** 2)) + 1e-9)
            return 1.0 if rms >= self._ENERGY_FLOOR else 0.0
        except Exception:  # noqa: BLE001
            return 0.0
