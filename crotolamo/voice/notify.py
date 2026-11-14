"""Fuentes de despertar para los hilos del loop de voz (solo stdlib).

POR QUÉ: los hilos del loop concurrente esperaban "un ítem O el apagado"
haciendo expirar lo primero (queue.get(timeout=0.2)) solo para releer un flag,
y el hilo principal hacía stat() de control.json cada 0.3 s para enterarse de
una pausa del panel. En reposo eran ~1100 despertares por minuto para nada.
Aquí viven las dos piezas que cambian ese "preguntar por reloj" por "avisar":

- Wakeable: un threading.Event cuyo set() además ejecuta "despertadores"
  registrados (meter un centinela en una cola, escribir en un pipe). Un hilo
  puede bloquear SIN timeout en su cola y aun así enterarse del apagado.
- DirWatch: inotify(7) sobre un DIRECTORIO (vía ctypes; Linux) para saber que
  un archivo cambió en vez de preguntarlo con stat(). Se vigila el directorio y
  no el archivo porque el panel lo reemplaza por rename (temporal + os.replace):
  un watch sobre el archivo seguiría al inodo viejo. Si inotify no está o no se
  puede usar, open() lanza OSError y create() devuelve None: quien llama
  conserva su sondeo de siempre.
"""

from __future__ import annotations

import ctypes
import os
import select
import struct
import threading
import time
from collections.abc import Callable, Collection

from crotolamo.logging_setup import get_logger

log = get_logger("voice.notify")


class Wakeable(threading.Event):
    """Event que, al activarse, también despierta a quien espera en OTRA fuente
    (su cola, un select). Sigue siendo un threading.Event: is_set(), wait() y un
    set() a secas funcionan igual que siempre."""

    # Los hilos lo miran para decidir si pueden bloquear sin timeout (ver
    # threads._get_timeout): con un Event normal nadie les avisaría.
    wakes_waiters = True

    def __init__(self) -> None:
        super().__init__()
        self._wakers: list[Callable[[], None]] = []
        self._wakers_lock = threading.Lock()

    def add_waker(self, fn: Callable[[], None]) -> None:
        """Registra `fn` para cada activación. Si ya está activado, avisa en el
        acto: un despertador que llega tarde no se pierde el apagado."""
        with self._wakers_lock:
            self._wakers.append(fn)
            ya_activado = self.is_set()
        if ya_activado:
            _call(fn)

    def set(self) -> None:
        # Los despertadores corren una vez por ACTIVACIÓN (de no activado a
        # activado), no por llamada: el apagado se pide desde varios sitios
        # (stop(), el atajo, los tests) y cada STOP de más quedaría en su cola.
        # Corren fuera del cerrojo: un despertador puede tardar o volver a
        # entrar a otro Wakeable sin colgar a nadie.
        with self._wakers_lock:
            primera = not self.is_set()
            super().set()
            wakers = list(self._wakers) if primera else []
        for fn in wakers:
            _call(fn)


def _call(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - un despertador roto no impide el resto
        log.warning("despertador falló: %s", error)


# --- inotify -------------------------------------------------------------------
# Constantes de <sys/inotify.h>. IN_NONBLOCK e IN_CLOEXEC valen lo mismo que
# O_NONBLOCK y O_CLOEXEC, que cambian según la arquitectura: salen de os.
_IN_CLOSE_WRITE = 0x00000008   # escritor que cierra (escritura en el sitio)
_IN_MOVED_FROM = 0x00000040    # renombrado hacia fuera (el archivo ya no está)
_IN_MOVED_TO = 0x00000080      # renombrado hacia dentro (os.replace del panel)
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400   # el directorio vigilado se borró...
_IN_MOVE_SELF = 0x00000800     # ...o se movió
_IN_Q_OVERFLOW = 0x00004000    # el kernel tiró eventos: hay que volver a mirar
_IN_IGNORED = 0x00008000       # el watch dejó de existir
_MASK = (_IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_MOVED_FROM | _IN_CREATE | _IN_DELETE
         | _IN_DELETE_SELF | _IN_MOVE_SELF)
_ROTO = _IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF
_EVENT = struct.Struct("iIII")  # struct inotify_event: wd, mask, cookie, len (+ nombre)


def _libc_inotify() -> tuple[Callable[..., int], Callable[..., int]]:
    """inotify_init1 e inotify_add_watch de la libc del proceso.

    CDLL(None) son los símbolos ya cargados en el proceso (la libc de Python):
    ctypes.util.find_library lanzaría un ldconfig o un gcc para encontrarla.
    AttributeError si la libc no los tiene (no es Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    init1 = libc.inotify_init1
    init1.argtypes = [ctypes.c_int]
    init1.restype = ctypes.c_int
    add_watch = libc.inotify_add_watch
    add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    add_watch.restype = ctypes.c_int
    return init1, add_watch


def _oserror_de_errno(que: str, ruta: str | None = None) -> OSError:
    numero = ctypes.get_errno()
    return OSError(numero, f"{que}: {os.strerror(numero)}", ruta)


class DirWatch:
    """inotify de UN directorio más un pipe de despertar.

    Un solo hilo llama wait(); wake() y close() pueden llegar desde otros (el
    apagado). close() se hace con wait() ya vuelto (el hilo que espera lo
    cierra al salir): cerrar un fd que otro hilo tiene en select() no lo
    despierta en Linux.
    """

    def __init__(self, fd: int, wake_r: int, wake_w: int) -> None:
        self._fd = fd
        self._wake_r = wake_r
        self._wake_w = wake_w
        # True cuando el directorio desapareció (borrado, movido) o select/read
        # fallaron: quien llama vuelve a su sondeo.
        self.broken = False
        # wake() puede llegar desde otro hilo DESPUÉS de close(): sin este
        # cerrojo escribiría en un número de fd ya reutilizado por otra cosa.
        self._fd_lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(cls, directory: str | os.PathLike[str]) -> DirWatch:
        """Abre la vigilancia o lanza OSError con el motivo (sin inotify, sin el
        directorio, límite de instancias o de watches del usuario...)."""
        try:
            init1, add_watch = _libc_inotify()
        except (OSError, AttributeError) as error:
            raise OSError(f"inotify no disponible: {error}") from error
        ruta = os.fspath(directory)
        fd = init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise _oserror_de_errno("inotify_init1")
        try:
            if add_watch(fd, os.fsencode(ruta), _MASK) < 0:
                raise _oserror_de_errno("inotify_add_watch", ruta)
            wake_r, wake_w = os.pipe()  # ya nacen con O_CLOEXEC (PEP 446)
        except BaseException:
            os.close(fd)
            raise
        os.set_blocking(wake_r, False)
        os.set_blocking(wake_w, False)
        return cls(fd, wake_r, wake_w)

    @classmethod
    def create(cls, directory: str | os.PathLike[str]) -> DirWatch | None:
        """Como open(), pero None si no se puede (quien llama sigue sondeando)."""
        try:
            return cls.open(directory)
        except OSError as error:
            log.debug("sin inotify para %s: %s", directory, error)
            return None

    @property
    def wake_fd(self) -> int:
        """Extremo de escritura del pipe de despertar (no bloqueante), para
        signal.set_wakeup_fd: así cualquier señal despierta a wait(). Quien lo
        registre lo devuelve ANTES de close(), o el manejador de C escribiría
        en un fd cerrado o ya reutilizado."""
        return self._wake_w

    def wake(self) -> None:
        """Despierta a wait() desde otro hilo. Idempotente: con un despertar ya
        pendiente (pipe lleno, EAGAIN) no hace falta otro."""
        with self._fd_lock:
            if self._closed:
                return
            try:
                os.write(self._wake_w, b"\0")
            except BlockingIOError:
                pass
            except OSError as error:
                log.debug("DirWatch.wake: %s", error)

    def wait(self, names: Collection[str], timeout: float | None) -> bool | None:
        """Bloquea hasta un evento que nombre alguno de `names`, un wake() o el
        timeout (segundos; None = sin límite).

        True: cambió algo de `names` (o el kernel desbordó la cola: por si
        acaso). False: venció el timeout. None: lo despertaron, o la vigilancia
        se rompió (mirar `broken`). Los eventos de otros archivos del directorio
        se leen y se descartan sin volver: no alargan el plazo.
        """
        if self._closed or self.broken:
            return None
        limite = None if timeout is None else time.monotonic() + timeout
        while True:
            restante = None if limite is None else max(0.0, limite - time.monotonic())
            try:
                listos, _, _ = select.select([self._fd, self._wake_r], [], [], restante)
            except (OSError, ValueError) as error:  # ValueError: fd >= FD_SETSIZE
                self._romper(f"select: {error}")
                return None
            if not listos:
                return False
            if self._wake_r in listos:
                self._vaciar_pipe()
                return None
            try:
                buf = os.read(self._fd, 65536)
            except BlockingIOError:
                continue
            except OSError as error:
                self._romper(f"read: {error}")
                return None
            cambio = False
            i = 0
            while i + _EVENT.size <= len(buf):
                _wd, mask, _cookie, largo = _EVENT.unpack_from(buf, i)
                nombre = buf[i + _EVENT.size:i + _EVENT.size + largo].split(b"\0", 1)[0]
                i += _EVENT.size + largo
                if mask & _ROTO:
                    self.broken = True
                elif mask & _IN_Q_OVERFLOW or os.fsdecode(nombre) in names:
                    cambio = True
            if self.broken:
                return None
            if cambio:
                return True

    def _romper(self, motivo: str) -> None:
        log.debug("DirWatch roto: %s", motivo)
        self.broken = True

    def _vaciar_pipe(self) -> None:
        try:
            while os.read(self._wake_r, 512):
                pass
        except BlockingIOError:
            pass
        except OSError as error:
            log.debug("DirWatch: no pude vaciar el pipe: %s", error)

    def close(self) -> None:
        with self._fd_lock:
            if self._closed:
                return
            self._closed = True
            for fd in (self._fd, self._wake_r, self._wake_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
