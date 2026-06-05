"""Text-to-speech con Piper. Migrado de C1::voice_out, SIN la ruta hardcodeada.

La voz .onnx se resuelve desde la config ([paths].voces + [voice].piper_voice),
no del /home/exitili quemado de C1.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
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

    @classmethod
    def from_settings(cls, settings) -> "TTS":
        voces = settings.paths.get("voces", Path.home() / "voices")
        piper_voice = settings.voice.get("piper_voice", "es_MX-ald-medium.onnx")
        return cls(voces / piper_voice)

    def available(self) -> bool:
        return self.voice_model.exists() and shutil.which("ffplay") is not None

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self.voice_model.exists():
            print(f"[voz desactivada: no encuentro {self.voice_model}]")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)
        try:
            subprocess.run(
                ["python", "-m", "piper", "-m", str(self.voice_model),
                 "-f", str(audio_path), "--", text],
                check=True, text=True, capture_output=True,
            )
            if shutil.which("ffplay"):
                subprocess.run(
                    ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", str(audio_path)],
                    check=False,
                )
            else:
                print("[voz: falta ffplay para reproducir]")
        except subprocess.CalledProcessError as error:
            print(f"[error en Piper: {error.stderr}]")
        except FileNotFoundError:
            print("[voz desactivada: falta piper. Instala con pip install -e '.[voice]']")
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    def speak_sentences(self, text: str) -> None:
        """Habla el texto frase por frase (menor latencia percibida al combinar
        con streaming: se puede empezar a hablar antes de tener todo el texto)."""
        for sentence in split_sentences(text):
            self.speak(sentence)
