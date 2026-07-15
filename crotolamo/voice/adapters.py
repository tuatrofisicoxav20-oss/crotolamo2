"""Adapters reales de audio del loop de voz (extraídos de loop.py).

Solo se usan en hardware; los tests inyectan fakes. Imports pesados (numpy,
torch, sounddevice) perezosos y CACHEADOS: el hot path corre ~31 veces por
segundo, así que resolver los módulos en cada frame (aunque el import cacheado
sea "barato") es costo puro sin motivo.
"""

from __future__ import annotations

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

    def read(self):
        """Lee un frame BLOQUEANTE (~32ms a 16kHz con frame=512).

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

        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                device=self.device,
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
    """Adapter de Silero VAD (M2) como vad_fn(chunk) -> prob.

    Si silero-vad o torch no están instalados, cae a un VAD por energía RMS que
    devuelve 0.0 o 1.0 (escalado para ser comparable con el umbral 0-1 del loop).
    El fallback se activa UNA sola vez (no reintenta el import cada chunk) y se
    reporta en el log UNA vez para no inundar.

    Rendimiento: numpy/torch se resuelven UNA vez (lazy, en el primer frame) y se
    cachean en self._np/self._torch; antes el import corría en cada chunk (~31/s).
    """

    _ENERGY_FLOOR = 0.01  # RMS mínimo para considerar voz en modo fallback

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._model: Any = None
        self._fallback: bool = False  # True = usar energía RMS
        self._np: Any = None     # numpy cacheado (lazy, primer frame)
        self._torch: Any = None  # torch cacheado (lazy, primer frame)

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
                import torch

                # Comparte la instancia cacheada de stt: antes cada uno cargaba su
                # propia copia del ONNX, duplicando el modelo en RAM sin motivo.
                from crotolamo.voice.stt import _get_silero_vad

                self._np = numpy
                self._torch = torch
                self._model = _get_silero_vad()
            # ascontiguousarray evita la copia extra que hacía .copy(): solo copia
            # si el chunk no es ya un array float32 contiguo.
            arr = self._np.ascontiguousarray(chunk, dtype=self._np.float32)
            return float(self._model(self._torch.from_numpy(arr), self.sample_rate))
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
