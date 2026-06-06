"""Text-to-speech con Piper como librería PERSISTENTE.

Cambios de la auditoría (I1, M4 parcial):
- El modelo Piper (63 MB) se carga UNA sola vez (_get_voice cachea), no por frase.
- Ya NO se lanza `python -m piper` por subprocess ni se depende de ffplay: se
  reproduce el PCM con sounddevice.

API verificada contra piper-tts 1.4.2: voice.synthesize(text) -> Iterable[AudioChunk],
con AudioChunk.audio_int16_array (int16) y AudioChunk.sample_rate. (Versiones viejas
exponían synthesize_stream_raw; esta no, por eso usamos synthesize()/audio_int16_array.)

La ruta del .onnx viene de la config ([paths].voces + [voice].piper_voice); cero
hardcodeo. Los imports pesados (numpy, sounddevice, piper) son perezosos: el módulo
importa sin la extra [voice].
"""

from __future__ import annotations

import re
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    """Parte el texto en frases para hablarlas una por una (Fase 6, TTS por frases)."""
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?¿¡\n])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


class TTS:
    def __init__(self, voice_model: Path) -> None:
        self.voice_model = Path(voice_model)
        self._voice = None  # PiperVoice perezoso, cargado una sola vez

    @classmethod
    def from_settings(cls, settings) -> "TTS":
        voces = settings.paths.get("voces", Path.home() / "voices")
        piper_voice = settings.voice.get("piper_voice", "es_MX-ald-medium.onnx")
        return cls(voces / piper_voice)

    def available(self) -> bool:
        """Hay voz si existe el .onnx (la reproducción ya no depende de ffplay)."""
        return self.voice_model.exists()

    def _get_voice(self):
        """Carga PiperVoice una sola vez y la cachea."""
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(self.voice_model))
        return self._voice

    def synthesize_pcm(self, text: str):
        """Devuelve (audio_int16 ndarray, sample_rate) sintetizado por Piper.

        Reutilizable por el smoke test sin reproducir nada.
        """
        import numpy as np

        voice = self._get_voice()
        chunks = list(voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.int16), int(voice.config.sample_rate)
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        sample_rate = int(chunks[0].sample_rate)
        return audio, sample_rate

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self.voice_model.exists():
            print(f"[voz desactivada: no encuentro {self.voice_model}]")
            return

        try:
            import sounddevice as sd
        except ImportError:
            print("[voz desactivada: falta sounddevice. Instala con pip install -e '.[voice]']")
            return

        try:
            audio, sample_rate = self.synthesize_pcm(text)
        except Exception as error:  # noqa: BLE001 - una voz rota no debe matar el agente
            print(f"[error sintetizando con Piper: {error}]")
            return

        if audio.size == 0:
            return
        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as error:  # noqa: BLE001 - p.ej. sin dispositivo de audio en headless
            print(f"[no pude reproducir el audio, patrón: {error}]")

    def speak_sentences(self, text: str) -> None:
        """Habla el texto frase por frase. Ya NO recarga el modelo: _get_voice() lo cachea."""
        for sentence in split_sentences(text):
            self.speak(sentence)
