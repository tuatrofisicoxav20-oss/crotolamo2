"""Interfaces (Protocols) de voz: la costura para enchufar otros backends (L4).

Hoy STT usa faster-whisper y TTS usa Piper directos. Definir estas interfaces deja
una costura: el día que quieras un backend Wyoming (unix:// / tcp:// a un home
server), implementas estos Protocols y lo enchufas sin tocar loop.py ni el listener.

NO se implementa Wyoming aquí; solo se define la costura. Las clases actuales (STT,
TTS) ya cumplen estos Protocols por tipado estructural.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechToText(Protocol):
    def listen_once(self, **kwargs) -> str: ...

    def transcribe(self, path: Path) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    def speak(self, text: str) -> None: ...

    def speak_sentences(self, text: str) -> None: ...

    def stop(self) -> None: ...
