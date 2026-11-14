"""Wake consciente de la música: umbral dinámico y ducking del reproductor.

POR QUÉ: con Spotify o YouTube sonando, el wake difuso (Whisper) transcribe
letras de canciones y de vez en cuando algo suena a "crotolamo" (falsos
despertares); y una vez despierto, la música tapa la orden del patrón y
enturbia la transcripción. Dos defensas, opcionales y desacopladas del oído:

1) MediaMonitor: un hilo sabe en segundo plano si hay algún reproductor
   MPRIS en estado Playing (playerctl) y CACHEA la respuesta. El oído consulta
   el caché (threshold() / is_playing()), nunca un subprocess: el camino
   caliente del audio jamás espera a playerctl. Se entera por EVENTOS (un
   `playerctl --follow` vivo) y solo cae al sondeo periódico si eso no está.
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

Solo stdlib (threading, queue, select, subprocess vía run_cmd de tools/base.py).
"""

from __future__ import annotations

import atexit
import math
import os
import queue
import select
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

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
        # Si playerctl respondió a la última playing_players() de CADA hilo (ver
        # answered()). Por hilo: el Ducker comparte el backend con el MediaMonitor
        # y pregunta desde su propio hilo.
        self._local = threading.local()

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
        self._local.answered = result is not None
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

    def answered(self) -> bool:
        """¿Respondió playerctl a la última playing_players() de ESTE hilo? False
        si no se le pudo preguntar (colgado: TimeoutExpired; OSError; ausente): ese
        [] no es "no suena nada", es "no lo sé". El MediaMonitor con --follow lo
        necesita para volver a preguntar: el seguidor no repite la línea de un
        reproductor que sigue sonando."""
        return bool(getattr(self._local, "answered", True))

    # Mismo formato que la foto (playing_players): cada línea del seguidor se
    # puede comparar con el caché. Solo es un DISPARADOR: la verdad sigue siendo
    # la foto, con el parser de siempre.
    FOLLOW_ARGS = ("-a", "metadata", "--format", "{{playerName}}\t{{status}}", "--follow")

    def available(self) -> bool:
        return shutil.which("playerctl") is not None

    def follow(self) -> subprocess.Popen[bytes] | None:
        """UN `playerctl --follow` vivo: escribe una línea "nombre<TAB>estado" por
        cada cambio de cualquier reproductor (el estado actual al arrancar; si uno
        desaparece, la del que queda arriba, o una vacía si no queda ninguno) y no
        repite una línea idéntica a la anterior. En reposo no lanza ni un proceso.

        Se queda en el grupo de procesos y el cgroup del servicio: un Ctrl-C en la
        terminal o `systemctl stop` lo matan junto con el listener. None si no se
        pudo lanzar.
        """
        if not self.available():
            return None
        try:
            return subprocess.Popen(
                ["playerctl", *self.FOLLOW_ARGS],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        except (OSError, ValueError) as error:
            log.debug("playerctl --follow no arrancó: %s", error)
            return None


# Cómo se entera el MediaMonitor de que cambió lo que suena ([voice].media_watch):
# "follow" = eventos de `playerctl --follow` (con el sondeo como respaldo);
# "poll" = el sondeo de antes, tal cual (un playerctl cada media_poll_s).
MEDIA_WATCH_MODES = ("follow", "poll")


def _media_watch(value: object) -> str:
    mode = str(value).strip().lower()
    if mode in MEDIA_WATCH_MODES:
        return mode
    log.warning("[voice].media_watch = %r no existe (follow | poll); uso 'follow'", value)
    return "follow"


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


def _answered(backend: object) -> bool:
    """answered() del backend si lo tiene (PlayerctlBackend); sin él, toda
    respuesta cuenta como respuesta, lo de siempre."""
    respondio = getattr(backend, "answered", None)
    if not callable(respondio):
        return True
    try:
        return bool(respondio())
    except Exception:  # noqa: BLE001 - roto = no se sabe
        return False


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
    """Sabe en segundo plano si suena música y cachea el resultado.

    El oído (wake difuso en el loop simple, WakeWordDetector.feed en el
    concurrente) pregunta threshold() / is_playing() por chunk: leen el caché
    bajo un lock, nunca tocan subprocess. Un backend que lanza o devuelve
    basura cuenta como "no suena" y el hilo sigue vivo pase lo que pase.

    Cómo se entera (watch, de [voice].media_watch):
    - "follow" (default) con un solo backend que sabe seguir (PlayerctlBackend):
      un `playerctl --follow` vivo avisa de cada cambio y una línea que puede
      invalidar el caché pide UNA foto (refresh_now, la verdad de siempre). Sin
      música, cero procesos y cero despertares salvo una foto de seguridad cada
      SAFETY_REFRESH_S; con música, además, la foto de cada poll_s de antes (ver
      PLAYING_REFRESH_S); y una foto que no respondió (playerctl colgado) se
      repite cada poll_s hasta que responda. Si el seguidor no arranca o muere,
      vuelve el sondeo de siempre y se reintenta; sin playerctl instalado, se
      re-chequea cada ABSENT_RECHECK_S sin lanzar nada.
    - "poll", backends sin follow() (los fakes de los tests) o varios backends:
      el sondeo de siempre, una foto cada poll_s.

    Solo loguea a INFO al CAMBIAR de modo (entrar/salir de "modo música"):
    nada por sondeo, nada por chunk.
    """

    # Con el seguidor vivo y sin música, una foto de seguridad cada tanto por si
    # alguna vez pierde un evento sin morir: 288 procesos al día en vez de 57.600.
    SAFETY_REFRESH_S = 300.0
    # Con música, una foto cada tanto (None = poll_s, lo mismo que el sondeo de
    # antes). playerctl 2.4.1 -a --follow NO imprime nada cuando se cierra un
    # reproductor que sonaba pero no era el último que cambió (comprobado con el
    # binario real): sin esta foto el caché se quedaría en "modo música" hasta la
    # de seguridad. El caso contrario (algo empieza a sonar) siempre imprime.
    PLAYING_REFRESH_S: float | None = None
    # Seguidor caído: el sondeo de siempre durante este plazo y reintento. El
    # plazo se duplica con cada caída rápida hasta el tope; un seguidor que vivió
    # más que el plazo cuenta como sano y lo devuelve al inicio.
    FOLLOW_RETRY_S = 30.0
    FOLLOW_RETRY_MAX_S = 600.0
    # Sin playerctl no hay MPRIS que mirar: re-chequeo espaciado por si lo instalan.
    ABSENT_RECHECK_S = 60.0
    # Una "línea" del seguidor sin salto más larga que esto es basura.
    _MAX_LINE = 64 * 1024
    # Lo que terminate_children() (manejador de señal) espera al cerrojo del
    # seguidor: un Popen tarda milisegundos; esto solo acota el apagado.
    _TERMINATE_WAIT_S = 0.5
    # Las señales con las que se apaga el servicio (las que atiende el manejador
    # del listener). systemctl stop las manda a TODO el cgroup y Ctrl-C/SIGHUP a
    # todo el grupo de la terminal: el seguidor muere con ellas a la vez que el
    # listener, antes de que su manejador alcance a marcar el stop.
    _SHUTDOWN_SIGNALS = frozenset({signal.SIGTERM, signal.SIGINT, signal.SIGHUP})

    def __init__(self, backends: Sequence[MediaBackend], poll_s: float = 1.5, *,
                 threshold_normal: float = 0.72, threshold_media: float = 0.85,
                 watch: object = "follow") -> None:
        self._backends = list(backends)
        # Piso de 0.1s: un poll_s de 0 en la config no debe convertir el hilo en
        # un busy-loop de subprocess.
        self.poll_s = max(0.1, float(poll_s))
        self.threshold_normal = float(threshold_normal)
        self.threshold_media = float(threshold_media)
        self.watch = _media_watch(watch)
        self._lock = threading.Lock()
        self._playing: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # El `playerctl --follow` vivo, si lo hay. RLock: terminate_children()
        # corre en el manejador de señal, en el hilo principal, que pudo quedar
        # interrumpido DENTRO de stop() con el lock tomado.
        self._follower: subprocess.Popen[bytes] | None = None
        self._follower_lock = threading.RLock()
        self._follow_warned = False
        # Última foto del hilo (cualquiera) y última pedida por un evento (como
        # mucho una por poll_s); reloj monotónico.
        self._last_refresh = float("-inf")
        self._last_event_refresh = float("-inf")
        # Cuándo repetir una foto que NO respondió (None = la última respondió).
        self._retry_at: float | None = None

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
    def _poll_backends(self) -> tuple[list[str], bool]:
        """Los reproductores en Playing y si TODOS los backends respondieron.

        Un backend que lanza, devuelve basura o dice que no respondió (answered(),
        p.ej. playerctl colgado) cuenta como silencio, como siempre; pero eso no
        es una respuesta, y con --follow hay que volver a preguntar (ver
        _refresh_logged)."""
        playing: list[str] = []
        answered = True
        for backend in self._backends:
            try:
                names = backend.playing_players()
            except Exception as error:  # noqa: BLE001 - un backend roto = silencio
                log.debug("media: el backend %s falló (%s); lo cuento como silencio",
                          type(backend).__name__, error)
                answered = False
                continue
            if not _answered(backend):
                answered = False  # el motivo ya quedó en el log de DEBUG
            # Basura (None, un str suelto, un número): no es una lista de nombres.
            # OJO: un str es iterable y daría "reproductores" de una letra.
            if not isinstance(names, (list, tuple)):
                log.debug("media: el backend %s devolvió basura (%r); lo ignoro",
                          type(backend).__name__, names)
                answered = False
                continue
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()
                if is_own_voice(name) or name in playing:
                    continue
                playing.append(name)
        return playing, answered

    def refresh_now(self) -> bool:
        """Un sondeo SÍNCRONO a los backends; devuelve si suena algo.

        Es lo que corre el hilo en cada vuelta; expuesto para tests y para
        quien quiera un dato fresco sin esperar al siguiente tick.
        """
        return self._refresh()[0]

    def _refresh(self) -> tuple[bool, bool]:
        """refresh_now() y, además, si todos los backends respondieron."""
        playing, answered = self._poll_backends()
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
        return now_playing, answered

    def _refresh_logged(self) -> None:
        self._last_refresh = time.monotonic()
        answered = False
        try:
            answered = self._refresh()[1]
        except Exception as error:  # noqa: BLE001 - el hilo no muere jamás
            log.debug("media: sondeo falló: %s", error)
        # Una foto que no respondió deja el caché en "no suena", como el sondeo,
        # pero el seguidor no repite la línea de un reproductor que sigue sonando:
        # nadie más avisaría y el caché quedaría mal hasta la foto de seguridad.
        # Con --follow se repite poll_s después de ESTA (el ritmo del sondeo)
        # mientras no responda.
        self._retry_at = None if answered else time.monotonic() + self.poll_s

    def _poll(self, rounds: int | None) -> bool:
        """El sondeo de siempre: foto y espera poll_s, `rounds` veces (None = hasta
        el stop). Devuelve True si hay que salir."""
        while rounds is None or rounds > 0:
            if self._stop.is_set():
                return True
            self._refresh_logged()
            if self._stop.wait(self.poll_s):
                return True
            if rounds is not None:
                rounds -= 1
        return self._stop.is_set()

    def _follow_backend(self) -> Any | None:
        """El backend a seguir por eventos, o None si toca el sondeo de siempre."""
        if self.watch != "follow" or len(self._backends) != 1:
            return None
        backend = self._backends[0]
        return backend if callable(getattr(backend, "follow", None)) else None

    def _run(self) -> None:
        backend = self._follow_backend()
        if backend is not None:
            try:
                self._run_follow(backend)
                return
            except Exception as error:  # noqa: BLE001 - el hilo no muere jamás
                log.warning("media: el seguimiento por eventos falló (%s); vuelvo al sondeo",
                            error)
                with self._follower_lock:
                    proc = self._follower
                if proc is not None:
                    self._end_follower(proc, close=True)
        self._poll(None)

    def _run_follow(self, backend: Any) -> None:
        retry = self.FOLLOW_RETRY_S
        while not self._stop.is_set():
            available = getattr(backend, "available", None)
            if callable(available) and not available():
                # Sin playerctl ningún reproductor MPRIS puede contar como sonando:
                # la foto (que da "no suena", sin lanzar nada) y un re-chequeo
                # espaciado en vez de uno cada poll_s.
                self._refresh_logged()
                if self._stop.wait(max(self.poll_s, self.ABSENT_RECHECK_S)):
                    return
                continue
            self._refresh_logged()  # la verdad de partida
            proc = self._start_follower(backend)
            if proc is None:
                if self._stop.is_set():
                    return
                self._follow_down("no arrancó", retry)
            else:
                started = time.monotonic()
                try:
                    self._follow(proc)
                finally:
                    code = self._end_follower(proc, close=True)
                if self._stop.is_set():
                    return
                # Muerto por una señal de apagado: casi siempre es el mismo stop
                # que está por llegarle al listener. Se le da poll_s a su manejador
                # para marcarlo; sin esto, el journal se llevaba un WARNING de una
                # caída que no hubo y salía una foto de playerctl con el proceso
                # muriendo. Si no llega, era una caída de verdad (un pkill): el
                # respaldo empieza poll_s más tarde, lo que tardaba el sondeo de
                # antes en ver un cambio.
                if (code is not None and -code in self._SHUTDOWN_SIGNALS
                        and self._stop.wait(self.poll_s)):
                    return
                if time.monotonic() - started >= self.FOLLOW_RETRY_S:
                    retry = self.FOLLOW_RETRY_S  # vivió sano: una caída aislada
                self._follow_down(f"terminó (código {code})", retry)
            if self._poll(max(1, math.ceil(retry / self.poll_s))):
                return
            retry = min(retry * 2, self.FOLLOW_RETRY_MAX_S)

    def _follow_down(self, what: str, retry: float) -> None:
        """El aviso va una vez (WARNING); las caídas siguientes, a DEBUG."""
        message = "media: playerctl --follow %s; sondeo cada %.1fs y reintento en %.0fs"
        if self._follow_warned:
            log.debug(message, what, self.poll_s, retry)
            return
        self._follow_warned = True
        log.warning(message, what, self.poll_s, retry)

    def _start_follower(self, backend: Any) -> subprocess.Popen[bytes] | None:
        # Lanzar y registrar bajo el MISMO cerrojo. stop() y terminate_children()
        # marcan _stop y luego miran _follower con el cerrojo: o ven al seguidor
        # ya registrado y lo terminan ellos, o aquí se ve el stop y ni se lanza.
        # Antes el Popen corría fuera: una señal a medio Popen no veía al recién
        # nacido, os._exit llegaba antes de registrarlo y quedaba un playerctl
        # huérfano (ppid 1) para siempre.
        with self._follower_lock:
            if self._stop.is_set():
                return None
            try:
                proc = backend.follow()
            except Exception as error:  # noqa: BLE001 - un backend roto = sin eventos
                log.debug("media: follow() de %s reventó: %s", type(backend).__name__, error)
                return None
            if proc is None:
                return None
            self._follower = proc
        if getattr(proc, "stdout", None) is None or self._stop.is_set():
            self._end_follower(proc, close=True)
            return None
        log.debug("media: sigo los reproductores con playerctl --follow (pid %s)", proc.pid)
        return proc

    def _end_follower(self, proc: subprocess.Popen[bytes], *, close: bool) -> int | None:
        """Termina y RECOGE al seguidor (wait: sin zombis). close=True solo desde
        el hilo del monitor, el único que lee su stdout."""
        code: int | None = None
        try:
            # SIGTERM ANTES de sacarlo del registro: un terminate_children() (señal
            # y os._exit) que mire en medio lo ve todavía registrado y lo termina
            # él. Al revés, podía verlo ya fuera y sin terminar, y no lo terminaba
            # nadie.
            if proc.poll() is None:
                proc.terminate()
        except Exception as error:  # noqa: BLE001 - best-effort
            log.debug("media: no pude terminar playerctl --follow: %s", error)
        with self._follower_lock:
            if self._follower is proc:
                self._follower = None
        try:
            try:
                code = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait(timeout=1.0)
        except Exception as error:  # noqa: BLE001 - best-effort
            log.debug("media: no pude cerrar playerctl --follow: %s", error)
        finally:
            if close and proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
        return code

    def _follow(self, proc: subprocess.Popen[bytes]) -> None:
        """Bloquea en el stdout del seguidor hasta su EOF o el stop.

        Cada línea que puede invalidar el caché pide UNA foto, como mucho una
        por poll_s (ver _event_budget_from): una ráfaga (un reproductor que
        parpadea) no cuesta más procesos que el sondeo de antes, y la foto
        pendiente llega a más tardar poll_s después: nunca más tarde que antes.
        Sin eventos ni música, ningún despertar salvo la foto de seguridad.
        """
        stdout = proc.stdout
        assert stdout is not None
        fd = stdout.fileno()
        pending = b""
        dirty = False
        while not self._stop.is_set():
            if dirty:
                deadline = self._event_budget_from() + self.poll_s
            elif self.is_playing():
                playing_s = self.PLAYING_REFRESH_S
                deadline = self._last_refresh + (self.poll_s if playing_s is None else playing_s)
            else:
                deadline = self._last_refresh + self.SAFETY_REFRESH_S
            if self._retry_at is not None:
                deadline = min(deadline, self._retry_at)  # la foto que no respondió
            try:
                ready, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
            except (OSError, ValueError) as error:
                log.debug("media: no puedo esperar a playerctl --follow: %s", error)
                return
            if not ready:
                # Vence el plazo: la foto pendiente de una ráfaga, la de la música,
                # la que repite una que no respondió o la de seguridad.
                if dirty:
                    self._event_refresh()
                    dirty = False
                else:
                    self._refresh_logged()
                continue
            try:
                data = os.read(fd, 4096)
            except OSError as error:
                log.debug("media: no puedo leer a playerctl --follow: %s", error)
                return
            if not data:
                return  # EOF: el seguidor murió (o stop() lo terminó)
            pending += data
            *lines, pending = pending.split(b"\n")
            if len(pending) > self._MAX_LINE:
                lines.append(pending)  # basura sin salto: se mira la verdad
                pending = b""
            if dirty or not any(self._needs_refresh(line) for line in lines):
                continue
            if time.monotonic() - self._event_budget_from() >= self.poll_s:
                self._event_refresh()
            else:
                dirty = True

    def _event_budget_from(self) -> float:
        """Desde cuándo cuenta el tope de una foto por poll_s de los eventos.

        Con música en el caché ya corre la foto de cada poll_s (PLAYING_REFRESH_S)
        y el evento comparte SU ritmo: desde la última foto, sea cual sea. Contadas
        aparte, las líneas de otro reproductor (pausa, parado) sumaban hasta una
        foto más por poll_s: ~1.7 veces los procesos del sondeo de antes. Así, con
        música, el cambio se ve como mucho poll_s después, igual que antes. Sin
        música, desde la última foto pedida por un evento: lo que empieza a sonar
        se ve en el acto."""
        return self._last_refresh if self.is_playing() else self._last_event_refresh

    def _event_refresh(self) -> None:
        self._last_event_refresh = time.monotonic()
        self._refresh_logged()

    def _needs_refresh(self, raw: bytes) -> bool:
        """¿Esta línea del seguidor puede querer decir que el caché ya no vale?

        playerctl imprime la línea del reproductor que acaba de cambiar (o, si
        uno desaparece, la del que queda arriba; una vacía si no queda ninguno),
        y no repite una línea idéntica a la anterior.
        """
        line = raw.decode("utf-8", "replace").rstrip("\r")
        if not line.strip():
            with self._lock:  # vacía = ya no queda ningún reproductor
                return bool(self._playing)
        if "\t" not in line:
            return True  # basura: se mira la verdad
        name, status = line.split("\t", 1)
        name = name.strip()
        if not name or is_own_voice(name):
            return False  # lo mismo que descarta _parse_playing
        with self._lock:
            if status.strip().lower() == "playing":
                return name not in self._playing
            # Uno que no suena: cambia algo si estaba en el caché, o si el caché
            # tiene alguno (al cerrarse el que sonaba, playerctl imprime la línea
            # del que queda, no una vacía). Sin música en el caché, nada que hacer.
            return bool(self._playing)

    def terminate_children(self) -> None:
        """Para el apagado (manejador de señal del listener, atexit): da el monitor
        por parado, para que no lance otro seguidor, y manda SIGTERM al vivo SIN
        esperar a nadie. os._exit se salta stop() y todos los finally."""
        self._stop.set()
        # Espera (corta) a un Popen en curso del seguidor: lo ve nacer y lo termina.
        acquired = self._follower_lock.acquire(timeout=self._TERMINATE_WAIT_S)
        try:
            proc = self._follower
        finally:
            if acquired:
                self._follower_lock.release()
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort
                pass

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        if self._follow_backend() is not None:
            # Respaldo de stop(): un return temprano o una excepción que salga del
            # listener terminan el intérprete sin pasar por stop(), y el seguidor
            # (un hijo de larga vida) quedaría huérfano.
            atexit.register(self.terminate_children)
        self._thread = threading.Thread(target=self._run, name="MediaMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Para el hilo con un join CORTO: si está a medio subprocess, el timeout
        de playerctl lo termina solo y, al ser daemon, muere con el proceso. El
        seguidor se termina y se recoge aquí: su EOF desbloquea al hilo."""
        self._stop.set()
        with self._follower_lock:
            proc = self._follower
        if proc is not None:
            self._end_follower(proc, close=False)
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        atexit.unregister(self.terminate_children)


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
