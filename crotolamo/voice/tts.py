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

import queue
import re
import threading
from pathlib import Path

from crotolamo.logging_setup import get_logger

log = get_logger("voice.tts")


# --- Segmentación unificada (split_sentences y StreamSpeaker) ---
# Un punto NO siempre cierra frase: "40. 5 GB" (decimal partido por espacio) y
# "Dr. López" (abreviatura) deben hablarse de corrido; cortarlos suena a dos
# frases truncadas en el TTS. No se busca un segmentador perfecto: solo no
# romper los casos frecuentes del español MX.

_CANDIDATO = re.compile(r"(?<=[.!?¿¡\n])\s+")
# Abreviaturas cortas comunes en las respuestas ("no" se trata aparte: como
# adverbio SÍ cierra frase; solo bloquea el corte si sigue un número: "No. 5").
_ABREVIATURAS = {"sr", "sra", "srta", "dr", "dra", "etc", "ej", "p.ej", "vs", "av"}
_PALABRA_ANTES = re.compile(r"([a-záéíóúüñ]+(?:\.[a-záéíóúüñ]+)*)\.$", re.IGNORECASE)


def _cierra_frase(text: str, start: int, end: int) -> bool:
    """True si el candidato (los espacios en text[start:end]) es fin de frase real."""
    if text[start - 1] != ".":
        return True  # ! ? ¡ ¿ y salto de línea cortan siempre
    despues = text[end] if end < len(text) else ""
    antes = text[:start]
    if len(antes) >= 2 and antes[-2].isdigit() and despues.isdigit():
        return False  # decimal partido: "40. 5"
    m = _PALABRA_ANTES.search(antes)
    if m:
        palabra = m.group(1).lower()
        if palabra in _ABREVIATURAS:
            return False
        if palabra == "no" and despues.isdigit():
            return False  # "No. 5" como número
    return True


def _split(text: str, *, final: bool) -> tuple[list[str], str]:
    """Divide en (frases_cerradas, resto).

    Con final=False (streaming) un '.' seguido solo de espacios al final del
    buffer se queda en el resto: sin ver el siguiente carácter no se puede
    distinguir un fin de frase de un decimal/abreviatura a medias.
    """
    pieces: list[str] = []
    start = 0
    for m in _CANDIDATO.finditer(text):
        if m.end() >= len(text) and not final and text[m.start() - 1] == ".":
            break  # falta contexto: esperar más tokens
        if not _cierra_frase(text, m.start(), m.end()):
            continue
        pieces.append(text[start:m.start()])
        start = m.end()
    return [p.strip() for p in pieces if p.strip()], text[start:]


def split_sentences(text: str) -> list[str]:
    """Parte el texto en frases para hablarlas una por una (Fase 6, TTS por frases)."""
    pieces, resto = _split(text.strip(), final=True)
    if resto.strip():
        pieces.append(resto.strip())
    return pieces


class TTS:
    def __init__(self, voice_model: Path) -> None:
        self.voice_model = Path(voice_model)
        self._voice = None  # PiperVoice perezoso, cargado una sola vez
        self._stop_flag = threading.Event()  # M3.1: señal de corte thread-safe

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

        import importlib.util

        if importlib.util.find_spec("sounddevice") is None:
            log.warning("falta sounddevice; voz desactivada")
            return

        try:
            self._speak_streaming(text)
        except Exception as error:  # noqa: BLE001 - una voz rota no debe matar el agente
            log.warning("error sintetizando/reproduciendo con Piper: %s", error)

    def _speak_streaming(self, text: str) -> bool:
        """Sintetiza y reproduce por CHUNKS conforme Piper los genera.

        Antes se sintetizaba la frase ENTERA (list + concatenate) y recién ahí
        sonaba: la latencia a la primera palabra era toda la síntesis. Ahora el
        primer chunk suena en cuanto existe. Se escribe en rebanadas cortas
        (~90ms) vigilando el stop, así el corte (barge-in) sigue siendo casi
        inmediato. Devuelve False si se cortó a media frase.
        """
        import numpy as np
        import sounddevice as sd

        voice = self._get_voice()
        self._stop_flag.clear()
        stream = None
        step = 2048  # muestras por escritura: corte perceptible en <100ms
        try:
            for chunk in voice.synthesize(text):
                if self._stop_flag.is_set():
                    return False
                audio = np.ascontiguousarray(chunk.audio_int16_array).reshape(-1)
                if audio.size == 0:
                    continue
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=int(chunk.sample_rate), channels=1, dtype="int16",
                    )
                    stream.start()
                for i in range(0, audio.size, step):
                    if self._stop_flag.is_set():
                        return False
                    stream.write(audio[i:i + step])
            return True
        finally:
            if stream is not None:
                try:
                    if self._stop_flag.is_set():
                        stream.abort()
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001 - cerrar audio es best-effort
                    pass

    def _play_interruptible(self, audio, sample_rate: int) -> bool:
        """Reproduce vigilando el stop en pasos cortos (sd.wait() no es interrumpible).

        Devuelve False si se cortó a media frase (M3.1).
        """
        import sounddevice as sd

        self._stop_flag.clear()
        sd.play(audio, samplerate=sample_rate)
        stream = sd.get_stream()
        while stream is not None and stream.active:
            if self._stop_flag.is_set():
                sd.stop()
                return False
            sd.sleep(20)  # ms
            stream = sd.get_stream()
        return not self._stop_flag.is_set()

    def speak_sentences(self, text: str) -> None:
        """Habla el texto frase por frase. Ya NO recarga el modelo: _get_voice() lo cachea."""
        for sentence in split_sentences(text):
            self.speak(sentence)

    def beep(self) -> None:
        """Bip corto NO bloqueante: acuse de "te escucho" instantáneo.

        Sustituye al "Te escucho, patrón" hablado (~1s de síntesis+reproducción
        bloqueante) cuando [voice].ack = "beep": suena mientras la grabación del
        comando YA está abierta, así el patrón puede hablar de inmediato. Un
        tono senoidal no es voz, así que el VAD (Silero) no lo confunde.
        """
        try:
            import numpy as np
            import sounddevice as sd

            rate = 22050
            t = np.linspace(0.0, 0.12, int(0.12 * rate), endpoint=False)
            tone = (0.2 * np.sin(2 * np.pi * 880.0 * t)).astype("float32")
            sd.play(tone, samplerate=rate)  # sin wait: no bloquea la escucha
        except Exception as error:  # noqa: BLE001 - sin audio, el bip es opcional
            log.warning("no pude sonar el bip: %s", error)

    def stop(self) -> None:
        """Corta cualquier reproducción en curso (thread-safe, M3.1)."""
        self._stop_flag.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001 - sin sounddevice/dispositivo, nada que cortar
            pass


class StreamSpeaker:
    """Habla frases COMPLETAS conforme el LLM las va generando (streaming).

    Antes: agent.handle_turn devolvía la respuesta ENTERA y recién ahí se
    hablaba — con el LLM en CPU, varios segundos de silencio percibido. Ahora:
    feed() se cablea como on_token del agente; en cuanto se cierra una frase
    (. ! ? o salto de línea) se encola y un hilo la va hablando MIENTRAS el
    modelo sigue generando las siguientes. La primera palabra suena en cuanto
    existe la primera frase, no al final del turno.

    Uso:
        speaker = StreamSpeaker(tts, on_first=...)
        reply = agent.handle_turn(cmd, on_token=speaker.feed)
        spoke = speaker.finish()   # habla la cola restante y espera; True si habló
        if not spoke: tts.speak_sentences(reply)  # fallback (p.ej. error del LLM)
    """

    def __init__(self, tts: TTS, on_first=None) -> None:
        self.tts = tts
        self._on_first = on_first
        self._buffer = ""
        self._spoke = False
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="StreamSpeaker", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            sentence = self._q.get()
            if sentence is None:
                return
            if not self._spoke:
                self._spoke = True
                if self._on_first is not None:
                    try:
                        self._on_first()
                    except Exception as error:  # noqa: BLE001
                        log.warning("on_first falló: %s", error)
            try:
                self.tts.speak(sentence)
            except Exception as error:  # noqa: BLE001 - una frase rota no mata la cola
                log.warning("StreamSpeaker: %s", error)

    def feed(self, token: str) -> None:
        """on_token del agente: acumula y encola las frases ya cerradas.

        Usa la MISMA segmentación que split_sentences (_split): en streaming
        solo se decide con lo que hay, así que un '.' al final del buffer se
        retiene hasta ver el siguiente carácter (¿decimal? ¿abreviatura?).
        """
        self._buffer += token
        pieces, self._buffer = _split(self._buffer, final=False)
        for piece in pieces:
            self._q.put(piece)

    def finish(self, timeout_s: float = 120.0) -> bool:
        """Habla lo que quede en el buffer, espera la cola y devuelve si habló."""
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._q.put(tail)
        self._q.put(None)
        self._thread.join(timeout=timeout_s)
        return self._spoke
