"""Wake consciente de la música: umbral dinámico y ducking del reproductor.

POR QUÉ: con Spotify o YouTube sonando, el wake difuso (Whisper) transcribe
letras de canciones y de vez en cuando algo suena a "crotolamo" (falsos
despertares); y una vez despierto, la música tapa la orden del patrón y
enturbia la transcripción. Dos defensas, opcionales y desacopladas del oído:

1) MediaMonitor: un hilo sondea en segundo plano si hay algún reproductor
   MPRIS en estado Playing (playerctl) y CACHEA la respuesta. El oído consulta
   el caché (threshold() / is_playing()), nunca un subprocess: el camino
   caliente del audio jamás espera a playerctl.
2) Ducker: al activarse el wake baja el volumen de lo que suena a una fracción
   y al terminar lo restaura EXACTO (guarda el string crudo que imprime el
   backend, sin pasar por float y volver). Para el loop concurrente hay un
   worker con cola: las transiciones de modo llegan desde los hilos de audio
   (Ear/Mouth/Stt vía el publisher de SharedState) y un subprocess ahí
   bloquearía al oído.

La voz propia de Crotolamo (Piper por sounddevice) NO registra reproductor
MPRIS hoy; aun así, cualquier reproductor cuyo nombre contenga "crotolamo" se
descarta en todos los niveles, por si algún día la voz sale por un reproductor
propio: hablar no es "música sonando".

Solo stdlib (threading, queue, subprocess vía run_cmd de tools/base.py).
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from crotolamo.logging_setup import get_logger
from crotolamo.tools.base import run_cmd

log = get_logger("voice.media_aware")

# Marca de la voz propia: un reproductor con este texto en el nombre nunca cuenta
# como música (ni para subir el umbral ni para el ducking).
_OWN_VOICE_MARK = "crotolamo"


def is_own_voice(player: str) -> bool:
    """True si el nombre del reproductor es la voz propia de Crotolamo."""
    return _OWN_VOICE_MARK in player.lower()


@runtime_checkable
class MediaBackend(Protocol):
    """Fuente de verdad sobre qué suena y su volumen (playerctl hoy; costura
    para otros backends). Ninguna implementación debe lanzar: ante cualquier
    fallo devuelve [] / None / False."""

    def playing_players(self) -> list[str]:
        """Nombres de los reproductores en estado Playing."""
        ...

    def get_volume(self, player: str) -> str | None:
        """Volumen CRUDO tal cual lo imprime el backend (para restaurarlo exacto)."""
        ...

    def set_volume(self, player: str, value: str) -> bool:
        """Fija el volumen; True si el backend lo aceptó."""
        ...


class PlayerctlBackend:
    """Backend MPRIS vía playerctl (el mismo binario que usa tools/media.py).

    Todo fallo (playerctl ausente, OSError, TimeoutExpired, returncode != 0,
    salida rara) degrada a "no suena nada" / None / False: este código corre
    en un hilo de sondeo y en el camino del wake, y una excepción aquí no puede
    tumbar la voz.
    """

    def __init__(self, timeout_s: float = 2.0) -> None:
        # 2s: playerctl responde en milisegundos; si tarda más, algo está mal en
        # el bus y preferimos "no suena" a bloquear el hilo de sondeo.
        self.timeout_s = timeout_s

    def _run(self, args: list[str]) -> subprocess.CompletedProcess | None:
        """playerctl <args>; None si no está instalado o si falla al ejecutarse."""
        if shutil.which("playerctl") is None:
            return None
        try:
            return run_cmd(["playerctl", *args], timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as error:
            log.debug("playerctl %s falló: %s", " ".join(args), error)
        except Exception as error:  # noqa: BLE001 - nunca propagar desde aquí
            log.debug("playerctl %s reventó: %s", " ".join(args), error)
        return None

    def playing_players(self) -> list[str]:
        result = self._run(["-a", "metadata", "--format", "{{playerName}}\t{{status}}"])
        if result is None:
            return []
        stdout = result.stdout or ""
        # returncode != 0 con stdout vacío es el "No players found" de
        # playerctl. Pero también sale != 0 si UN reproductor no tiene metadata
        # (una pestaña sin pista) aunque los demás sí imprimieron su línea: en
        # ese caso lo que hay en stdout vale.
        if result.returncode != 0 and not stdout.strip():
            return []
        return _parse_playing(stdout)

    def get_volume(self, player: str) -> str | None:
        if not player.strip():
            return None
        result = self._run(["-p", player, "volume"])
        if result is None or result.returncode != 0:
            return None
        value = (result.stdout or "").strip()
        return value or None

    def set_volume(self, player: str, value: str) -> bool:
        if not player.strip() or not value.strip():
            return False
        result = self._run(["-p", player, "volume", value])
        return result is not None and result.returncode == 0


def sane_media_threshold(normal: float, media: float | None, where: str) -> float | None:
    """El umbral con música nunca puede ser MENOR que el normal: con
    [wake].threshold = 0.9 y el default wake_threshold_media = 0.85, la música
    RELAJARÍA el wake (efecto inverso). Se sube al normal con aviso."""
    if media is None:
        return None
    if media < normal:
        log.warning("%s: umbral con música %.2f < umbral normal %.2f; uso %.2f",
                    where, media, normal, normal)
        return normal
    return media


def _parse_playing(stdout: str) -> list[str]:
    """Parsea líneas "nombre<TAB>estado" y devuelve los reproductores en Playing.

    Tolerante: una línea sin tabulador o vacía se ignora (no invalida el
    resto). Se deduplica conservando el orden: dos instancias del mismo
    reproductor (p.ej. dos pestañas de Firefox) salen con el mismo playerName,
    y `playerctl -p nombre` actúa sobre la primera; ducking dos veces el mismo
    nombre pisaría el volumen original guardado con el ya atenuado.
    """
    playing: list[str] = []
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        name, status = line.split("\t", 1)
        name = name.strip()
        if not name or status.strip().lower() != "playing":
            continue
        if is_own_voice(name) or name in playing:
            continue
        playing.append(name)
    return playing


class MediaMonitor:
    """Sondea en segundo plano si suena música y cachea el resultado.

    El oído (wake difuso en el loop simple, WakeWordDetector.feed en el
    concurrente) pregunta threshold() / is_playing() por chunk: leen el caché
    bajo un lock, nunca tocan subprocess. Un backend que lanza o devuelve
    basura cuenta como "no suena" y el hilo sigue vivo pase lo que pase.

    Solo loguea a INFO al CAMBIAR de modo (entrar/salir de "modo música"):
    nada por sondeo, nada por chunk.
    """

    def __init__(self, backends: Sequence[MediaBackend], poll_s: float = 1.5, *,
                 threshold_normal: float = 0.72, threshold_media: float = 0.85) -> None:
        self._backends = list(backends)
        # Piso de 0.1s: un poll_s de 0 en la config no debe convertir el hilo en
        # un busy-loop de subprocess.
        self.poll_s = max(0.1, float(poll_s))
        self.threshold_normal = float(threshold_normal)
        self.threshold_media = float(threshold_media)
        self._lock = threading.Lock()
        self._playing: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- consulta (camino caliente: solo el caché) ---
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._playing)

    def playing_players(self) -> list[str]:
        with self._lock:
            return list(self._playing)

    def threshold(self) -> float:
        """Umbral vigente del wake: el de música si suena algo, el normal si no."""
        return self.threshold_media if self.is_playing() else self.threshold_normal

    # --- sondeo ---
    def _poll_backends(self) -> list[str]:
        playing: list[str] = []
        for backend in self._backends:
            try:
                names = backend.playing_players()
            except Exception as error:  # noqa: BLE001 - un backend roto = silencio
                log.debug("media: el backend %s falló (%s); lo cuento como silencio",
                          type(backend).__name__, error)
                continue
            # Basura (None, un str suelto, un número): no es una lista de nombres.
            # OJO: un str es iterable y daría "reproductores" de una letra.
            if not isinstance(names, (list, tuple)):
                log.debug("media: el backend %s devolvió basura (%r); lo ignoro",
                          type(backend).__name__, names)
                continue
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()
                if is_own_voice(name) or name in playing:
                    continue
                playing.append(name)
        return playing

    def refresh_now(self) -> bool:
        """Un sondeo SÍNCRONO a los backends; devuelve si suena algo.

        Es lo que corre el hilo en cada vuelta; expuesto para tests y para
        quien quiera un dato fresco sin esperar al siguiente tick.
        """
        playing = self._poll_backends()
        with self._lock:
            was_playing = bool(self._playing)
            self._playing = playing
        now_playing = bool(playing)
        if now_playing != was_playing:
            if now_playing:
                log.info("wake en modo música (umbral %.2f): suena %s",
                         self.threshold_media, ", ".join(playing))
            else:
                log.info("wake en modo normal (umbral %.2f)", self.threshold_normal)
        return now_playing

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_now()
            except Exception as error:  # noqa: BLE001 - el hilo no muere jamás
                log.debug("media: sondeo falló: %s", error)
            self._stop.wait(self.poll_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="MediaMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Para el hilo con un join CORTO: si está a medio subprocess, el timeout
        de playerctl lo termina solo y, al ser daemon, muere con el proceso."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None


class Ducker:
    """Atenúa la música mientras Crotolamo escucha y responde, y la restaura.

    Guarda el volumen CRUDO original (el string que imprime el backend) y lo
    repone tal cual: sin redondeos acumulados turno a turno. duck() estando ya
    en duck es NO-OP: jamás sobreescribir el original guardado con el volumen
    ya atenuado (eso dejaría la música cada vez más baja). restore() limpia el
    estado pase lo que pase: la música nunca queda atorada.

    Dos formas de uso:
    - Síncrona (loop simple): duck() / restore() o el context manager ducked().
    - Por modo (loop concurrente): on_mode(nombre) desde el publisher del
      SharedState. Como se invoca desde los hilos de audio, las acciones van a
      UN worker interno con cola FIFO (orden preservado: el duck de un turno
      siempre corre antes que su restore) y el oído nunca espera un subprocess.
    """

    def __init__(self, backends: Sequence[MediaBackend], factor: float = 0.2,
                 enabled: bool = True) -> None:
        self._backends = list(backends)
        # Fracción del volumen actual; fuera de [0, 1] no tiene sentido.
        self.factor = min(max(float(factor), 0.0), 1.0)
        self.enabled = enabled
        # Lock de estado: se sostiene durante TODO el duck/restore (subprocess
        # incluido) para que dos hilos no intercalen un duck a medias con un
        # restore. Las llamadas son cortas (timeout de 2s por playerctl).
        self._lock = threading.Lock()
        # reproductor -> (backend, volumen crudo original)
        self._saved: dict[str, tuple[MediaBackend, str]] = {}
        # Worker de on_mode (loop concurrente).
        self._q: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._closed = False
        # Se activa cuando close() ya esperó al worker: a partir de ahí ningún
        # duck (tardío o encolado por error) puede pisar el restore final.
        self._no_more_duck = False
        # Último "deseo" recibido por on_mode: el publisher dispara en CADA
        # cambio de estado (texto, turno, enabled) con el modo vigente; solo se
        # encola trabajo cuando cambia idle <-> no-idle. Su lock hace atómico el
        # leer-decidir-encolar: on_mode llega desde varios hilos (Ear/Mouth/Stt)
        # y sin él dos publicaciones casi simultáneas podrían perder un restore.
        self._want_duck: bool | None = None
        self._mode_lock = threading.Lock()
        # Reproductores que ya avisaron que no exponen volumen (Firefox por
        # MPRIS, p.ej.): el WARNING va una vez; después, DEBUG en cada wake.
        self._warned: set[str] = set()

    def _warn_once(self, name: str, message: str, *args: object) -> None:
        if name in self._warned:
            log.debug(message, *args)
            return
        self._warned.add(name)
        log.warning(message, *args)

    # --- API síncrona ---
    def is_ducked(self) -> bool:
        with self._lock:
            return bool(self._saved)

    def duck(self) -> None:
        """Baja el volumen de TODO lo que suena (sondeo fresco) a volumen*factor.

        Una vez que close() terminó de esperar al worker, es no-op: si el worker
        seguía atascado (D-Bus colgado) con un duck en cola, ese duck se
        aplicaría DESPUÉS del restore final del apagado y la música quedaría
        atorada baja. Lo encolado ANTES de close sí se procesa, en orden.
        """
        if not self.enabled or self._no_more_duck:
            return
        with self._lock:
            if self._saved:
                return  # ya en duck: no pisar el original guardado
            for backend in self._backends:
                self._duck_backend(backend)

    def _duck_backend(self, backend: MediaBackend) -> None:
        """Con self._lock tomado. Un reproductor que falla no impide los demás."""
        try:
            players = backend.playing_players()
        except Exception as error:  # noqa: BLE001
            log.warning("duck: no pude listar reproductores en %s: %s",
                        type(backend).__name__, error)
            return
        if not isinstance(players, (list, tuple)):
            log.warning("duck: %s devolvió basura al listar reproductores: %r",
                        type(backend).__name__, players)
            return
        for name in players:
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if is_own_voice(name) or name in self._saved:
                continue
            try:
                raw = backend.get_volume(name)
            except Exception as error:  # noqa: BLE001
                log.warning("duck: no pude leer el volumen de %s: %s", name, error)
                continue
            if raw is None:
                self._warn_once(name, "duck: no pude leer el volumen de %s; lo dejo como está",
                                name)
                continue
            try:
                current = float(raw)
            except (TypeError, ValueError):
                log.warning("duck: volumen raro de %s (%r); lo dejo como está", name, raw)
                continue
            # Guardar el original ANTES de escribir: si la escritura "falla"
            # (p.ej. timeout) pero playerctl sí la aplicó, restore() lo repone
            # igual. Restaurar de más es inocuo; de menos deja la música atorada.
            self._saved[name] = (backend, raw)
            target = f"{current * self.factor:.3f}"
            try:
                ok = backend.set_volume(name, target)
            except Exception as error:  # noqa: BLE001
                log.warning("duck: no pude bajar el volumen de %s: %s", name, error)
                continue
            if not ok:
                self._warn_once(name, "duck: %s rechazó el volumen %s", name, target)
                continue
            log.debug("duck: %s %s -> %s", name, raw, target)

    def restore(self) -> None:
        """Repone cada volumen guardado con su string CRUDO original. Idempotente."""
        with self._lock:
            saved, self._saved = self._saved, {}
            for name, (backend, raw) in saved.items():
                self._restore_one(backend, name, raw)

    @staticmethod
    def _restore_one(backend: MediaBackend, name: str, raw: str) -> None:
        try:
            ok = backend.set_volume(name, raw)
        except Exception as error:  # noqa: BLE001
            log.warning("restore: no pude devolver el volumen de %s a %s: %s",
                        name, raw, error)
            return
        if not ok:
            log.warning("restore: %s rechazó su volumen original %s", name, raw)
        else:
            log.debug("restore: %s -> %s", name, raw)

    def emergency_restore(self, lock_timeout_s: float = 1.0) -> None:
        """Restauración de último recurso para el manejador de señal del listener.

        Ahí el proceso termina con os._exit (se salta todos los finally) y el
        propio hilo principal puede tener el lock tomado (a medio duck), así
        que se espera el lock un instante y, si no llega, se restaura SIN él:
        con el proceso muriendo, dejar Spotify bajito es peor que una carrera.
        """
        acquired = self._lock.acquire(timeout=lock_timeout_s)
        try:
            saved, self._saved = self._saved, {}
            for name, (backend, raw) in saved.items():
                self._restore_one(backend, name, raw)
        finally:
            if acquired:
                self._lock.release()

    @contextmanager
    def ducked(self) -> Iterator[None]:
        """`with ducker.ducked(): ...` — restaura aunque el cuerpo lance."""
        try:
            self.duck()
            yield
        finally:
            self.restore()

    # --- API por modo (loop concurrente) ---
    def on_mode(self, mode_name: str) -> None:
        """Hook del publisher de SharedState: cualquier modo distinto de "idle"
        atenúa; "idle" restaura. Solo ENCOLA (no bloquea al hilo que llama)."""
        if not self.enabled:
            return
        want_duck = mode_name != "idle"
        with self._mode_lock:
            if want_duck == self._want_duck:
                return
            self._want_duck = want_duck
            self._submit(self.duck if want_duck else self.restore)

    def _submit(self, action: Callable[[], None]) -> bool:
        with self._worker_lock:
            if self._closed:
                log.debug("ducker: cerrado; ignoro la acción encolada")
                return False
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain, name="Ducker", daemon=True)
                self._worker.start()
        self._q.put(action)
        return True

    def _drain(self) -> None:
        while True:
            action = self._q.get()
            if action is None:
                return
            try:
                action()
            except Exception as error:  # noqa: BLE001 - una acción rota no mata el worker
                log.warning("ducker: %s", error)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Espera a que el worker procese todo lo encolado hasta ahora (tests y
        apagado). True si terminó dentro del plazo."""
        done = threading.Event()
        if not self._submit(done.set):
            return True  # cerrado: no hay cola viva que esperar
        return done.wait(timeout_s)

    def close(self, timeout_s: float = 5.0) -> None:
        """Para el worker. Lo ya encolado (p.ej. un restore) SÍ se procesa, en
        orden; lo que llegue después se ignora. Idempotente."""
        with self._worker_lock:
            self._closed = True
            worker = self._worker
        try:
            if worker is None or not worker.is_alive():
                return
            self._q.put(None)
            worker.join(timeout=timeout_s)
        finally:
            self._no_more_duck = True
