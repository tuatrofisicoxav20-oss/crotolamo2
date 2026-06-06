"""Loop de voz concurrente con interrupción (barge-in). Inspirado en GLaDOS, en
versión MÍNIMA adaptada al estilo de Crotolamo (no se copia su complejidad entera).

Tres threads coordinados por colas:
  - escucha:  wake -> graba comando -> command_queue. Durante la reproducción,
              vigila la voz del patrón para interrumpir (barge-in).
  - proceso:  command_queue -> agent.handle_turn -> frases a tts_queue.
  - habla:    tts_queue -> tts.speak (TTS persistente de R1).

El barge-in corta la reproducción en curso (tts.stop() -> sd.stop()), vacía la
tts_queue y deja constancia en el historial de que la respuesta se cortó.

Diseñado para ser testeable con fakes (sin micrófono): barge_in() y el flujo
proceso->habla se ejercitan sin audio real.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue

from crotolamo.logging_setup import get_logger
from crotolamo.voice.interfaces import SpeechToText, TextToSpeech
from crotolamo.voice.tts import split_sentences

log = get_logger("voice.loop")


class VoiceLoop:
    # L4: depende de las interfaces SpeechToText/TextToSpeech, no de STT/TTS concretos.
    def __init__(self, agent, stt: SpeechToText, tts: TextToSpeech, wake_detector,
                 *, silence_ms: int = 640) -> None:
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.wake = wake_detector
        self.silence_ms = silence_ms

        self.command_queue: Queue = Queue()
        self.tts_queue: Queue = Queue()
        self.shutdown = threading.Event()
        self.speaking = threading.Event()  # set mientras el thread de habla reproduce
        self._threads: list[threading.Thread] = []

    # --- utilidades ---
    @staticmethod
    def _drain(q: Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    def barge_in(self) -> bool:
        """Interrumpe la reproducción en curso: para el TTS y vacía la cola de habla.

        Devuelve True si había algo que interrumpir. Apto para llamarse desde el
        thread de escucha al detectar voz del patrón.
        """
        if not self.speaking.is_set() and self.tts_queue.empty():
            return False
        self.tts.stop()
        self._drain(self.tts_queue)
        # Constancia para el LLM de que su última respuesta se cortó (estilo GLaDOS).
        try:
            self.agent.conversation.add_user("(te interrumpí, patrón habló encima)")
        except Exception:  # noqa: BLE001 - el agente puede no tener conversación
            pass
        return True

    # --- threads ---
    def _process_loop(self) -> None:
        while not self.shutdown.is_set():
            try:
                command = self.command_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                reply = self.agent.handle_turn(command)
            except Exception as error:  # noqa: BLE001 - un turno roto no mata el loop
                reply = f"Tuve un error pensando, patrón: {error}"
            for sentence in split_sentences(reply):
                if self.shutdown.is_set():
                    break
                self.tts_queue.put(sentence)

    def _speak_loop(self) -> None:
        while not self.shutdown.is_set():
            try:
                sentence = self.tts_queue.get(timeout=0.2)
            except Empty:
                continue
            self.speaking.set()
            try:
                self.tts.speak(sentence)
            finally:
                self.speaking.clear()

    def _listen_loop(self) -> None:
        while not self.shutdown.is_set():
            try:
                # Mientras Crotolamo habla, vigila barge-in en vez de esperar wake.
                if self.speaking.is_set():
                    if self.wake.available() and self.wake.listen_for_wake(timeout_s=0.5):
                        self.barge_in()
                    continue

                if not self.wake.listen_for_wake(timeout_s=1.0):
                    continue
                command = self.stt.listen_once(silence_ms=self.silence_ms)
                if command.strip():
                    self.command_queue.put(command)
            except Exception as error:  # noqa: BLE001 - un fallo de mic no mata el loop
                log.warning("escucha: %s", error)

    # --- ciclo de vida ---
    def start(self) -> None:
        targets = (self._listen_loop, self._process_loop, self._speak_loop)
        self._threads = [threading.Thread(target=t, daemon=True) for t in targets]
        for th in self._threads:
            th.start()

    def stop(self) -> None:
        self.shutdown.set()
        self.tts.stop()
        for th in self._threads:
            th.join(timeout=2.0)

    def run_forever(self) -> None:
        """Arranca y bloquea hasta Ctrl-C (uso desde el listener)."""
        self.start()
        try:
            while not self.shutdown.is_set():
                threading.Event().wait(0.3)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
