"""Sesión de captura: UN dueño del micrófono para toda una sesión de escucha.

Los grabadores del modo simple (stt._record_silero / _record_energy,
wakeword.listen_for_wake) abren cada uno su sd.InputStream, graban UNA frase y
lo cierran: el dispositivo vive lo que una grabación, y lo que suena entre dos
(Whisper decodificando el wake, la base antes de una continuación de
smart_endpoint, el atenuado de la música) no lo oye nadie. MicSession abre el
micrófono una vez y un hilo lector lo lee sin parar a una cola acotada; quien
graba consume de ella (el parámetro `source` de esos grabadores) en vez de abrir
el dispositivo, y lo que sonó mientras nadie leía está ahí, en orden.

Contrato de una fuente (lo que esperan los grabadores con `source`):
  read(timeout=1.0, frames=FRAME): el siguiente bloque float32 1-D de `frames`
      muestras, o None si en `timeout` s no llegó audio suficiente (micrófono
      reintentando, sesión suspendida o cerrada). Lanza MicFailed si el
      micrófono no volvió (give_up_s).
  pending(frames=FRAME): bloques completos ya capturados y sin consumir.

La sesión parte y junta sus chunks de 512 al tamaño que pida cada uno (480 la
energía, 1280 openWakeWord): nadie se queda con muestras que le tocaban al
siguiente, así que no hay que devolver nada a la cola.

El hilo lector NO corre el VAD: la instancia de Silero es única y con estado
(stt._get_silero_vad), y la usa un solo consumidor a la vez.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from crotolamo.logging_setup import get_logger
from crotolamo.voice.stt import _require

log = get_logger("voice.capture")


class MicFailed(RuntimeError):
    """El micrófono de la sesión no volvió tras give_up_s de fallos seguidos."""


class MicSession:
    """Micrófono abierto toda la sesión, leído por un hilo a una cola acotada.

    start() abre el dispositivo en el hilo de quien llama y LANZA si no puede:
    quien la usa cae entonces a un stream por grabación (lo de siempre). Después,
    solo el hilo lector toca el stream: lo lee, lo cierra si falla, lo reabre y
    lo cierra al salir. close() nunca lo cierra con el lector dentro de read()
    (el límite B1 de PortAudio, que no tiene read() con timeout).

    Fallos del dispositivo: se cierra el stream, se avisa UNA vez por mensaje
    distinto, se espera con backoff exponencial (1, 2, 4... hasta backoff_max_s;
    close() corta la espera) y se reabre. Tras give_up_s segundos de fallos
    seguidos (0 = nunca) la sesión se rinde: read() lanza MicFailed.
    """

    FRAME = 512  # 32 ms a 16 kHz: el chunk que exige Silero
    _ESPERA_MIN_S = 1.0
    _JOIN_S = 0.5

    def __init__(self, sample_rate: int = 16000, device: Any = None, *,
                 backlog_s: float = 15.0, give_up_s: float = 0.0,
                 backoff_max_s: float = 30.0) -> None:
        self.sample_rate = sample_rate
        # None = default del sistema; "crotolamo_aec_source" tras el AEC.
        self.device = device
        self.give_up_s = give_up_s
        self.backoff_max_s = backoff_max_s
        # Tope de la cola: 15 s de float32 a 16 kHz son ~1 MB. Lo que nadie
        # consume más allá se tira empezando por lo más viejo (y se cuenta).
        self._max_muestras = max(self.FRAME, int(backlog_s * sample_rate))
        self._cola: deque[Any] = deque()
        self._muestras = 0
        self._cond = threading.Condition()
        # Cuántas muestras pide cada read() que espera (con el lock): el lector
        # solo despierta a alguien cuando ya tiene su bloque entero.
        self._piden: list[int] = []
        self._stop = threading.Event()
        # Puesto = capturando; borrado = suspendida (dispositivo cerrado).
        self._activa = threading.Event()
        self._activa.set()
        self._pausada_en = 0.0  # monotonic del suspend() que abrió la pausa: ver _leer
        self._cerrada = False
        self._fatal: str | None = None
        self._stream: Any = None
        self._sd: Any = None
        self._np: Any = None
        self._hilo: threading.Thread | None = None
        self.dropped = 0  # chunks tirados por la cola llena
        self.opens = 0    # aperturas del dispositivo (la de start incluida)

    @property
    def failed(self) -> bool:
        """True si la sesión se rindió (read() lanza MicFailed)."""
        return self._fatal is not None

    @property
    def closed(self) -> bool:
        """True tras close(): no volverá a entregar audio (read() da None ya)."""
        return self._cerrada

    # --- ciclo de vida ---
    def start(self) -> None:
        """Abre el micrófono y arranca el lector. Lanza si no puede abrirlo."""
        if self._hilo is not None:
            return
        self._np = _require("numpy")
        self._sd = _require("sounddevice")
        self._stream = self._abrir()
        hilo = threading.Thread(target=self._run, name="MicSession", daemon=True)
        try:
            hilo.start()
        except Exception:
            stream, self._stream = self._stream, None
            self._cerrar(stream)
            raise
        self._hilo = hilo

    def close(self) -> None:
        """Para el lector y espera a que salga (acotado) antes de dar el
        micrófono por cerrado. Idempotente; read() devuelve None desde aquí."""
        with self._cond:
            self._cerrada = True
            self._cola.clear()
            self._muestras = 0
            self._cond.notify_all()
        self._stop.set()
        self._activa.set()  # un lector suspendido despierta para salir
        hilo = self._hilo
        if hilo is None or hilo is threading.current_thread():
            return
        hilo.join(self._JOIN_S)
        if hilo.is_alive():
            # Cerrar el stream desde aquí con el lector dentro de read() es justo
            # el riesgo B1 que esta clase quita: lo cierra él al volver (daemon).
            log.warning("el lector del micrófono sigue dentro de read() tras %.1f s; "
                        "cerrará el dispositivo al volver", self._JOIN_S)

    def suspend(self) -> None:
        """Escucha pausada: el lector CIERRA el dispositivo hasta resume() y lo
        pendiente se tira (es de antes de la pausa). Idempotente: la pausa empieza
        en el PRIMER suspend() (ver _leer)."""
        # El loop simple llama a suspend() en cada vuelta de la pausa (cada 0,3 s):
        # si cada llamada movía el inicio, al reanudar solo se descontaban los
        # últimos 0,3 s y una pausa larga con el micro caído contaba para
        # give_up_s (la sesión se rendía al primer reintento tras reanudar).
        with self._cond:
            if self._activa.is_set():
                self._pausada_en = time.monotonic()
                self._activa.clear()
        self.flush()

    def resume(self) -> None:
        self._activa.set()

    # --- consumo ---
    def read(self, timeout: float = 1.0, frames: int | None = None) -> Any:
        """El siguiente bloque de `frames` muestras (FRAME por defecto), en orden;
        None si en `timeout` s no llegó audio suficiente o la sesión se cerró."""
        n = self.FRAME if frames is None else int(frames)
        limite = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._fatal is not None:
                    raise MicFailed(f"el micrófono no volvió: {self._fatal}")
                if self._cerrada:
                    return None
                if self._muestras >= n:
                    return self._tomar(n)
                resta = limite - time.monotonic()
                if resta <= 0:
                    return None
                self._piden.append(n)
                try:
                    self._cond.wait(resta)
                finally:
                    self._piden.remove(n)

    def pending(self, frames: int | None = None) -> int:
        """Bloques completos de `frames` muestras capturados y sin consumir."""
        n = self.FRAME if frames is None else int(frames)
        with self._cond:
            return self._muestras // n

    def flush(self) -> int:
        """Tira lo capturado y sin consumir; devuelve cuántas muestras."""
        with self._cond:
            tiradas = self._muestras
            self._cola.clear()
            self._muestras = 0
            return tiradas

    def _tomar(self, n: int) -> Any:
        """Saca `n` muestras de la cola (con el lock). Un chunk del tamaño justo
        sale tal cual; si no, se junta y el resto vuelve a la cabeza."""
        primero = self._cola[0]
        if len(primero) == n:
            self._cola.popleft()
            self._muestras -= n
            return primero
        partes = []
        falta = n
        while falta:
            bloque = self._cola.popleft()
            if len(bloque) > falta:
                partes.append(bloque[:falta])
                self._cola.appendleft(bloque[falta:])
                falta = 0
            else:
                partes.append(bloque)
                falta -= len(bloque)
        self._muestras -= n
        return self._np.concatenate(partes)

    # --- hilo lector ---
    def _abrir(self) -> Any:
        stream = self._sd.InputStream(samplerate=self.sample_rate, channels=1,
                                      dtype="float32", device=self.device)
        self.opens += 1
        try:
            stream.start()
        except Exception:
            self._cerrar(stream)
            raise
        return stream

    @staticmethod
    def _cerrar(stream: Any) -> None:
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as error:  # noqa: BLE001 - un stream roto ya no importa
            log.debug("cerrando el micrófono: %s", error)

    def _soltar_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._cerrar(stream)

    def _pausa(self, segundos: float) -> bool:
        """Espera del backoff; True si close() la cortó (hay que salir)."""
        return self._stop.wait(segundos)

    def _run(self) -> None:
        try:
            self._leer()
        except Exception as error:  # noqa: BLE001 - el lector no muere en silencio
            log.exception("el lector del micrófono se cayó: %s", error)
            self._rendirse(f"el lector se cayó ({error})")
        finally:
            self._soltar_stream()
            with self._cond:
                self._cond.notify_all()

    def _leer(self) -> None:
        fallos = 0
        desde = 0.0
        espera = self._ESPERA_MIN_S
        avisados: set[str] = set()
        np = self._np
        while not self._stop.is_set():
            if not self._activa.is_set():
                self._soltar_stream()
                self._activa.wait()
                if fallos:
                    # La escucha en pausa no es el micrófono fallando: give_up_s
                    # cuenta solo fallos con la escucha activa. Sin esto, reanudar
                    # con el micro aún caído tras una pausa larga rendía la sesión
                    # al primer reintento. Desde el suspend() (que puede caer en
                    # plena espera del backoff), o desde el primer fallo si vino
                    # después; fallos y avisados siguen: es el mismo incidente.
                    desde += time.monotonic() - max(self._pausada_en, desde)
                continue
            try:
                if self._stream is None:
                    self._stream = self._abrir()
                bloque, _ = self._stream.read(self.FRAME)
            except Exception as error:  # noqa: BLE001 - el dispositivo falló: se reintenta
                self._soltar_stream()
                ahora = time.monotonic()
                if not fallos:
                    desde = ahora
                fallos += 1
                motivo = f"{type(error).__name__}: {error}"
                if self.give_up_s > 0 and ahora - desde >= self.give_up_s:
                    log.error("micrófono: %d fallos seguidos en %.0f s (%s); me rindo",
                              fallos, ahora - desde, motivo)
                    self._rendirse(motivo)
                    return
                if motivo not in avisados:
                    avisados.add(motivo)
                    log.warning("micrófono: %s; lo reabro con espera creciente (%.0f-%.0f s)",
                                motivo, self._ESPERA_MIN_S, self.backoff_max_s)
                if self._pausa(espera):
                    return
                espera = min(espera * 2, self.backoff_max_s)
                continue
            if fallos:
                log.info("micrófono de vuelta tras %d fallo(s)", fallos)
                fallos, espera, avisados = 0, self._ESPERA_MIN_S, set()
            self._guardar(np.squeeze(np.asarray(bloque, dtype=np.float32)))

    def _guardar(self, bloque: Any) -> None:
        with self._cond:
            # Suspendida o cerrada mientras leía: ese chunk es de antes de la pausa.
            if self._cerrada or not self._activa.is_set():
                return
            self._cola.append(bloque)
            self._muestras += len(bloque)
            while self._muestras > self._max_muestras and len(self._cola) > 1:
                viejo = self._cola.popleft()
                self._muestras -= len(viejo)
                self.dropped += 1
            # Solo si alguien ya tiene su bloque: avisar en cada chunk de 512 a quien
            # pide 1280 (openWakeWord) eran 2,5 despertares por bloque, 1,5 en balde.
            # close(), _rendirse() y la salida del lector avisan siempre.
            if self._piden and self._muestras >= min(self._piden):
                self._cond.notify_all()

    def _rendirse(self, motivo: str) -> None:
        with self._cond:
            self._fatal = motivo
            self._cond.notify_all()
