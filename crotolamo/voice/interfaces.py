"""Interfaces (Protocols) de voz: la costura para enchufar otros backends (L4).

Hoy STT usa faster-whisper y TTS usa Piper directos. Definir estas interfaces deja
una costura: el día que quieras un backend Wyoming (unix:// / tcp:// a un home
server), implementas estos Protocols y lo enchufas sin tocar loop.py ni el listener.

NO se implementa Wyoming aquí; solo se define la costura. Las clases actuales (STT,
TTS) ya cumplen estos Protocols por tipado estructural.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpeechToText(Protocol):
    def listen_once(self, **kwargs) -> str: ...

    # La firma que el loop USA de verdad (V14): SttThread llama
    # transcribe(audio, hotwords=...) y `audio` es un ndarray float32 16 kHz EN
    # MEMORIA en el camino caliente, o la ruta de un WAV (Path/str). Antes decía
    # transcribe(path: Path): un backend escrito contra eso pasaba isinstance
    # (runtime_checkable solo mira nombres) y luego fallaba con TypeError en cada
    # comando, que SttThread traga -> todas las órdenes perdidas en silencio.
    def transcribe(self, audio: Any, hotwords: str | None = None) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    def speak(self, text: str) -> None: ...

    def speak_sentences(self, text: str) -> None: ...

    def stop(self) -> None: ...
