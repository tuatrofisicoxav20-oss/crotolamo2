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
import time
from collections.abc import Callable
from pathlib import Path

from crotolamo.logging_setup import get_logger
from crotolamo.voice import runtime
from crotolamo.voice.clean import limpiar_para_voz

log = get_logger("voice.tts")


# --- Segmentación unificada (split_sentences y StreamSpeaker) ---
# Un punto NO siempre cierra frase: "40. 5 GB" (decimal partido por espacio) y
# "Dr. López" (abreviatura) deben hablarse de corrido; cortarlos suena a dos
# frases truncadas en el TTS. No se busca un segmentador perfecto: solo no
# romper los casos frecuentes del español MX.

# El fin de frase puede venir seguido de un cierre de énfasis markdown
# ("**Listo.** Ahora..."): esos cierres se tragan con el espacio para que la
# frase se corte donde termina de verdad y no se arrastre hasta la siguiente.
_CANDIDATO = re.compile(r"(?<=[.!?¿¡\n])[*_~]*\s+")
# Abreviaturas cortas comunes en las respuestas ("no" se trata aparte: como
# adverbio SÍ cierra frase; solo bloquea el corte si sigue un número: "No. 5").
_ABREVIATURAS = {"sr", "sra", "srta", "dr", "dra", "etc", "ej", "p.ej", "vs", "av"}
_PALABRA_ANTES = re.compile(r"([a-záéíóúüñ]+(?:\.[a-záéíóúüñ]+)*)\.$", re.IGNORECASE)
# "1. Abre Spotify": el punto de un marcador de lista numerada al inicio de
# línea no cierra frase (si no, "uno." se hablaría como frase suelta). Se
# compara contra la línea actual entera (fullmatch).
_MARCADOR_LISTA = re.compile(r"[ \t]*\d{1,2}\.")


def _inicio_palabra(text: str, fin: int) -> int:
    """Inicio de la racha de letras y puntos que termina justo antes de `fin`.

    isalpha() es un superconjunto de la clase de _PALABRA_ANTES (también con
    IGNORECASE): ninguna coincidencia de esa regex que acabe en `fin` puede
    empezar antes de este índice, así que buscar desde aquí da lo mismo que
    buscar en todo el prefijo.
    """
    i = fin
    while i > 0 and (text[i - 1] == "." or text[i - 1].isalpha()):
        i -= 1
    return i


def _cierra_frase(text: str, start: int, end: int) -> bool:
    """True si el candidato (los espacios en text[start:end]) es fin de frase real.

    Solo mira la COLA antes del punto (la línea actual y la última palabra), con
    pos/endpos y sin copiar: antes se copiaba y escaneaba text[:start] entero por
    cada candidato, y segmentar era cuadrático en la longitud de la respuesta
    (cúbico por token en streaming, con una lista numerada sin punto final).
    """
    if text[start - 1] != ".":
        return True  # ! ? ¡ ¿ y salto de línea cortan siempre
    despues = text[end] if end < len(text) else ""
    if start >= 2 and text[start - 2].isdigit() and despues.isdigit():
        return False  # decimal partido: "40. 5"
    if _MARCADOR_LISTA.fullmatch(text, text.rfind("\n", 0, start) + 1, start):
        return False  # marcador de lista numerada, no fin de frase
    m = _PALABRA_ANTES.search(text, _inicio_palabra(text, start), start)
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


def _piper_arena_keep_phonemes(voice_cfg) -> int:
    """[voice].piper_arena_keep_phonemes: frases de hasta tantos phoneme ids
    conservan el arena de ONNX Runtime; las más largas devuelven su pico al terminar.

    Un valor inválido no impide hablar: se avisa y se usa el default de fábrica.
    """
    default = runtime.PIPER_KEEP_PHONEMES
    value = voice_cfg.get("piper_arena_keep_phonemes", default)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    log.warning("[voice].piper_arena_keep_phonemes=%r no es un entero >= 0; uso %d",
                value, default)
    return default


class TTS:
    def __init__(self, voice_model: Path, *,
                 arena_keep_phonemes: int = runtime.PIPER_KEEP_PHONEMES) -> None:
        self.voice_model = Path(voice_model)
        self._voice = None  # PiperVoice perezoso, cargado una sola vez
        self.arena_keep_phonemes = arena_keep_phonemes
        # El hilo WarmVoice del listener precalienta la voz mientras el hilo
        # principal ya habla ("Crotolamo escuchando"): sin lock, los dos veían
        # _voice en None y cargaban el modelo dos veces (CPU y RAM dobles).
        self._voice_lock = threading.Lock()
        self._stop_flag = threading.Event()  # M3.1: señal de corte thread-safe

    @classmethod
    def from_settings(cls, settings) -> "TTS":
        voces = settings.paths.get("voces", Path.home() / "voices")
        piper_voice = settings.voice.get("piper_voice", "es_MX-ald-medium.onnx")
        return cls(voces / piper_voice,
                   arena_keep_phonemes=_piper_arena_keep_phonemes(settings.voice))

    def available(self) -> bool:
        """Hay voz si existe el .onnx (la reproducción ya no depende de ffplay)."""
        return self.voice_model.exists()

    def _get_voice(self):
        """Carga PiperVoice una sola vez y la cachea (aunque la pidan dos hilos)."""
        voice = self._voice  # camino caliente sin lock: ya cargada
        if voice is None:
            with self._voice_lock:
                # Doble comprobación: otro hilo pudo cargarla mientras esperábamos.
                if self._voice is None:
                    from piper import PiperVoice

                    # PiperVoice.load crea la sesión con SessionOptions por defecto: el
                    # arena retenía de por vida el pico de la frase más larga dicha
                    # (+411 MB tras una de 23 s). Envuelta, las frases de más de
                    # arena_keep_phonemes ids lo devuelven al terminar; las típicas
                    # hacen la MISMA llamada que antes (encoger en cada run costaba
                    # +10-25 % de síntesis). Mismo audio. Se envuelve ANTES de
                    # publicarla: el camino caliente de arriba la lee sin lock.
                    self._voice = runtime.wrap_piper_session(
                        PiperVoice.load(str(self.voice_model)), self.arena_keep_phonemes)
                voice = self._voice
        return voice

    def warm(self) -> bool:
        """Precalienta la voz Piper: la misma carga que la primera frase, en el hilo
        de quien llama (si otro hilo ya la está cargando, espera a esa carga).

        Sin el .onnx no carga nada (sin voz no hay modelo que cargar). Nunca lanza:
        si la carga falla no queda nada cacheado y la primera frase la reintenta y
        falla como siempre. True si la voz quedó cargada.
        """
        if not self.available():
            return False
        try:
            self._get_voice()
        except Exception as error:  # noqa: BLE001 - lo decide y lo avisa el primer uso real
            log.debug("no precalenté la voz Piper (%s); lo decidirá su primer uso", error)
            return False
        return True

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

    def speak(self, text: str, *, should_abort: Callable[[], bool] | None = None) -> None:
        """Habla `text`. `should_abort` (opcional): se consulta junto a la bandera
        de stop antes de sintetizar y en cada rebanada de audio; True = cortar.

        Existe porque stop() SOLO no basta (V4): _speak_streaming limpia la
        bandera al entrar (si no, un corte mataría todas las frases siguientes),
        así que un stop() del barge-in que cae entre el is_current() de quien
        llama y ese clear() se pierde y la frase del turno ya abortado suena
        entera. Quien sabe de turnos (la Boca) pasa `lambda: not
        state.is_current(turno)`: así el corte no depende del instante del stop().
        """
        # ÚNICO punto de limpieza para voz: por aquí pasa todo lo que se habla
        # (StreamSpeaker, speak_sentences, MouthThread, say), sea cual sea el
        # motor. Markdown, emojis y URLs se quitan ANTES de sintetizar; el corte
        # (stop/barge-in) no se toca: es solo trabajo de texto previo al audio.
        text = limpiar_para_voz(text)
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
            self._speak_streaming(text, should_abort)
        except Exception as error:  # noqa: BLE001 - una voz rota no debe matar el agente
            log.warning("error sintetizando/reproduciendo con Piper: %s", error)

    def _speak_streaming(
        self, text: str, should_abort: Callable[[], bool] | None = None,
    ) -> bool:
        """Sintetiza y reproduce por CHUNKS conforme Piper los genera.

        Antes se sintetizaba la frase ENTERA (list + concatenate) y recién ahí
        sonaba: la latencia a la primera palabra era toda la síntesis. Ahora el
        primer chunk suena en cuanto existe. Se escribe en rebanadas cortas
        (~90ms) vigilando el stop (y `should_abort`), así el corte (barge-in)
        sigue siendo casi inmediato. Devuelve False si se cortó a media frase.
        """
        import numpy as np
        import sounddevice as sd

        def _cortar() -> bool:
            return self._stop_flag.is_set() or (should_abort is not None and should_abort())

        # Turno ya abortado antes de empezar: ni cargar/sintetizar ni abrir audio.
        if should_abort is not None and should_abort():
            return False
        voice = self._get_voice()
        self._stop_flag.clear()
        stream = None
        cortado = False
        step = 2048  # muestras por escritura: corte perceptible en <100ms
        try:
            for chunk in voice.synthesize(text):
                if _cortar():
                    cortado = True
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
                    if _cortar():
                        cortado = True
                        return False
                    stream.write(audio[i:i + step])
            return True
        finally:
            if stream is not None:
                try:
                    # abort() tira lo que queda en el búfer; stop() solo lo
                    # dejaría sonar hasta el final.
                    if cortado or self._stop_flag.is_set():
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
        if not spoke: tts.speak_sentences(reply)  # fallback (un stream que no emitió nada)

    El error de un LLM caído ya no necesita el fallback: el agente lo emite por
    on_token (tras un salto de línea si ya se había hablado algo). Ese salto
    suelto no corta frase aquí: el loop simple lo convierte en frase aparte
    (interfaces/listener.py, _AvisoEnFraseAparte).
    """

    def __init__(self, tts: TTS, on_first=None) -> None:
        self.tts = tts
        self._on_first = on_first
        self._buffer = ""
        self._spoke = False
        self._q: queue.Queue[str | None] = queue.Queue()
        # finish() se rindió (C6): el hilo no habla ni una frase más de la cola.
        self._cancelled = threading.Event()
        # Cuándo tomó el hilo su última frase: el plazo de finish() se mide
        # desde aquí (por frase), no desde que lo llamaron.
        self._avance = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker, name="StreamSpeaker", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            sentence = self._q.get()
            self._avance = time.monotonic()
            if sentence is None or self._cancelled.is_set():
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
        """Habla lo que quede en el buffer, espera la cola y devuelve si habló.

        `timeout_s` es el plazo POR FRASE (desde que el hilo la toma, o desde
        esta llamada si fue después), no un tope a la respuesta: una respuesta
        larga que avanza se espera entera; solo un TTS colgado (sink de audio
        trabado: PipeWire reiniciado, Bluetooth que se fue) lo agota (C6). Antes
        era un join total y, al vencer, el hilo seguía vivo: quien llama ya
        seguía (HUD idle, escucha del followup) y la cola sonaba ENCIMA de la
        grabación, que el STT tomaba por una orden. Ahora se cancela el hilo.
        """
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._q.put(tail)
        self._q.put(None)
        inicio = time.monotonic()
        while self._thread.is_alive():
            restante = max(self._avance, inicio) + timeout_s - time.monotonic()
            if restante <= 0:
                self._cancelar(timeout_s)
                break
            self._thread.join(timeout=restante)
        return self._spoke

    def _cancelar(self, timeout_s: float) -> None:
        """El TTS no avanzó en `timeout_s`: ni una frase más y cortar la que suena.

        stop() no desbloquea una escritura trabada en el dispositivo, pero deja
        el corte marcado: cuando el audio se destraba, _speak_streaming aborta en
        la siguiente rebanada y el hilo sale sin tocar el resto de la cola.
        """
        self._cancelled.set()
        log.warning("StreamSpeaker: el TTS no avanzó en %g s; corto la voz", timeout_s)
        try:
            self.tts.stop()
        except Exception as error:  # noqa: BLE001 - finish() corre en un finally
            log.warning("StreamSpeaker: no pude cortar el TTS: %s", error)
