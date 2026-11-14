"""Revisión final, unidad 32v: crotolamo/voice/media_aware.py.

1. El Ducker bajaba y restauraba la PRIMERA instancia de un nombre, no la que
   suena. PlayerctlBackend listaba con `{{playerName}}` y luego mandaba
   `playerctl -p <nombre> volume`. Pero playerctl 2.4.1 casa `-p X` con la
   instancia X y con toda "X.<algo>" y, sin -a, la orden se queda en la
   primera que la acepta. Dos VLC o mpv (registran "vlc" y luego
   "vlc.instance<pid>") con el primero en pausa: el duck escribía 0.200 en el
   pausado, la música que sonaba seguía a todo volumen el turno entero y el
   restore solo reponía al pausado; con los dos sonando, solo se atenuaba uno
   (la deduplicación por nombre). Ahora la foto y el seguidor listan por
   `{{playerInstance}}` (lo mismo que tools/media.py en la unidad 25), y
   _needs_refresh compara la misma clase de nombre que guarda el caché.

2. La línea INFO "wake en modo música (umbral 0.85)" anunciaba los umbrales
   del wake DIFUSO ([wake].threshold / [voice].wake_threshold_media), los
   únicos que conoce el MediaMonitor. Con la config de fábrica (use_oww=true)
   el wake de verdad es openWakeWord, que decide con oww_threshold /
   oww_threshold_media (0.5 / 0.7): el journal decía un número que nada usa y
   que contradecía la propia línea del detector ("umbral=0.70"). El monitor
   ya no dice umbral: cada wake loguea el suyo al decidir.

3. emergency_restore no cerraba el Ducker. El manejador de señal del listener
   restaura y luego sigue hasta ~3 s (MCP, memoria, idle final) con el oído,
   el cerebro, la boca y el worker del Ducker vivos: un duck encolado o una
   transición idle -> listening en esa ventana volvía a bajar la música y
   os._exit la dejaba al 20 %. Ahora emergency_restore activa el mismo cerrojo
   que close() (_no_more_duck) ANTES de restaurar, duck() lo vuelve a mirar
   con el lock tomado (un worker que pasó el chequeo sin lock justo antes de
   la señal) y no escribe un volumen bajo si llega mientras tanto.
"""

from __future__ import annotations

import logging
import subprocess
import threading

import pytest

from crotolamo.voice import media_aware
from crotolamo.voice.media_aware import Ducker, MediaMonitor, PlayerctlBackend

_LOGGER = "crotolamo.voice.media_aware"


# --- 1. el Ducker va a la instancia que suena ---------------------------------


class _PlayerctlInstancias:
    """playerctl de mentiras con las reglas de playerctl 2.4.1 sobre instancias
    (leídas de su código: playerctl-common.c y playerctl-cli.c):

    - sin playerctld, los reproductores van ordenados por la instancia
      (g_strcmp0): "vlc" antes que "vlc.instance4242";
    - el nombre (`{{playerName}}`) es la instancia hasta el primer '.';
    - `-p X` elige la instancia X y toda "X.<algo>";
    - sin -a, la orden va al primero de los elegidos que la acepta y ahí acaba.
    """

    def __init__(self, *reproductores: tuple[str, str, str]) -> None:
        # instancia -> [estado, volumen crudo]
        self.reps = {inst: [estado, vol] for inst, estado, vol in reproductores}
        self.argvs: list[list[str]] = []

    def volumenes(self) -> dict[str, str]:
        return {inst: rep[1] for inst, rep in self.reps.items()}

    def formatea(self, formato: str, instancia: str) -> str:
        linea = formato
        for clave, valor in (("playerName", instancia.split(".", 1)[0]),
                             ("playerInstance", instancia),
                             ("status", self.reps[instancia][0])):
            linea = linea.replace("{{%s}}" % clave, valor)
        return linea

    def __call__(self, args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
        assert args[0] == "playerctl"
        resto = list(args[1:])
        self.argvs.append(list(resto))
        todos = "-a" in resto
        if todos:
            resto.remove("-a")
        elegidos = sorted(self.reps)
        if "-p" in resto:
            i = resto.index("-p")
            p = resto[i + 1]
            del resto[i:i + 2]
            elegidos = [r for r in elegidos
                        if r == p or (r.startswith(p) and r[len(p):].startswith("."))]
        formato = ""
        if "--format" in resto:
            i = resto.index("--format")
            formato = resto[i + 1]
            del resto[i:i + 2]
        if not elegidos:
            return subprocess.CompletedProcess(args, 1, "", "No players found\n")
        salida = ""
        for inst in elegidos:
            if resto == ["metadata"]:
                salida += self.formatea(formato, inst) + "\n"
            elif resto == ["volume"]:
                salida += self.reps[inst][1] + "\n"
            elif resto[0] == "volume":
                self.reps[inst][1] = f"{float(resto[1]):f}"
            if not todos:
                break
        return subprocess.CompletedProcess(args, 0, salida, "")


@pytest.fixture
def playerctl(monkeypatch):
    def instala(*reproductores: tuple[str, str, str]) -> _PlayerctlInstancias:
        falso = _PlayerctlInstancias(*reproductores)
        monkeypatch.setattr(media_aware, "run_cmd", falso)
        monkeypatch.setattr(media_aware.shutil, "which", lambda _cmd: "/usr/bin/playerctl")
        return falso

    return instala


def test_duck_con_dos_instancias_baja_la_que_suena_y_la_restaura(playerctl):
    """Dos VLC: el primero ("vlc") en pausa a 1.0 y suena el segundo a 0.8."""
    falso = playerctl(("vlc", "Paused", "1.000000"),
                      ("vlc.instance4242", "Playing", "0.800000"))
    ducker = Ducker([PlayerctlBackend()], factor=0.2)
    ducker.duck()
    assert falso.volumenes() == {"vlc": "1.000000", "vlc.instance4242": "0.160000"}
    ducker.restore()
    assert falso.volumenes() == {"vlc": "1.000000", "vlc.instance4242": "0.800000"}
    assert not ducker.is_ducked()


def test_duck_con_dos_instancias_sonando_baja_las_dos(playerctl):
    falso = playerctl(("vlc", "Playing", "1.000000"),
                      ("vlc.instance4242", "Playing", "0.800000"))
    ducker = Ducker([PlayerctlBackend()], factor=0.2)
    ducker.duck()
    assert falso.volumenes() == {"vlc": "0.200000", "vlc.instance4242": "0.160000"}
    ducker.restore()
    assert falso.volumenes() == {"vlc": "1.000000", "vlc.instance4242": "0.800000"}


def test_reproductor_de_una_sola_instancia_sigue_igual(playerctl):
    """Para Spotify la instancia ES el nombre: nada cambia."""
    falso = playerctl(("spotify", "Playing", "0.750000"))
    ducker = Ducker([PlayerctlBackend()], factor=0.2)
    ducker.duck()
    assert falso.volumenes() == {"spotify": "0.150000"}
    assert ["-p", "spotify", "volume", "0.150"] in falso.argvs
    ducker.restore()
    assert falso.volumenes() == {"spotify": "0.750000"}


def test_el_monitor_cuenta_cada_instancia_y_el_seguidor_habla_igual_que_la_foto(playerctl):
    """La línea que imprimiría `playerctl FOLLOW_ARGS` para la instancia que
    sigue sonando no invalida el caché que dejó la foto: los dos formatos dan
    la misma clase de nombre (antes la foto guardaba "vlc" y el seguidor
    decía "vlc" por las dos instancias)."""
    falso = playerctl(("vlc", "Paused", "1.000000"),
                      ("vlc.instance4242", "Playing", "0.800000"))
    mon = MediaMonitor([PlayerctlBackend()], poll_s=0.1)
    assert mon.refresh_now() is True
    assert mon.playing_players() == ["vlc.instance4242"]
    args = PlayerctlBackend.FOLLOW_ARGS
    formato = args[args.index("--format") + 1]
    snapshot = falso.argvs[-1]
    assert formato == snapshot[snapshot.index("--format") + 1]
    linea = falso.formatea(formato, "vlc.instance4242").encode()
    assert mon._needs_refresh(linea) is False
    # El pausado que cambia algo sí invalida (el caché tiene música).
    assert mon._needs_refresh(falso.formatea(formato, "vlc").encode()) is True


# --- 2. el log de transición no anuncia un umbral que quizá nadie usa ----------


def test_el_log_de_modo_no_anuncia_el_umbral_del_wake_difuso(caplog):
    """Los umbrales del MediaMonitor son los del wake difuso; con openWakeWord
    (el de fábrica) rigen otros. La línea de transición no da número."""

    class _Backend:
        playing: list[str] = ["spotify"]

        def playing_players(self) -> list[str]:
            return list(self.playing)

        def get_volume(self, player: str) -> str | None:
            return None

        def set_volume(self, player: str, value: str) -> bool:
            return False

    backend = _Backend()
    mon = MediaMonitor([backend], poll_s=0.1, threshold_normal=0.72, threshold_media=0.85)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        mon.refresh_now()
        backend.playing = []
        mon.refresh_now()
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 2, infos
    assert "modo música" in infos[0] and "spotify" in infos[0]
    assert "modo normal" in infos[1]
    for mensaje in infos:
        assert "umbral" not in mensaje and "0.85" not in mensaje and "0.72" not in mensaje
    # El umbral del difuso sigue ahí para quien lo lee (el loop simple).
    assert mon.threshold() == pytest.approx(0.72)


# --- 3. emergency_restore cierra el Ducker ------------------------------------


class _Spotify:
    def __init__(self) -> None:
        self.vol = {"spotify": "0.800000"}
        self.listados = 0

    def playing_players(self) -> list[str]:
        self.listados += 1
        return ["spotify"]

    def get_volume(self, player: str) -> str | None:
        return self.vol[player]

    def set_volume(self, player: str, value: str) -> bool:
        self.vol[player] = value
        return True


def test_tras_emergency_restore_una_transicion_tardia_no_vuelve_a_bajar_la_musica():
    """La señal llega en pleno turno; mientras el manejador cierra MCP y
    memoria, el turno acaba y entra un wake (idle -> listening)."""
    backend = _Spotify()
    ducker = Ducker([backend], factor=0.2)
    try:
        ducker.on_mode("listening")
        assert ducker.flush()
        assert backend.vol == {"spotify": "0.160"}
        ducker.emergency_restore()
        assert backend.vol == {"spotify": "0.800000"}
        ducker.on_mode("idle")
        ducker.on_mode("listening")
        assert ducker.flush()
        assert backend.vol == {"spotify": "0.800000"}  # lo que ve os._exit
        assert not ducker.is_ducked()
    finally:
        ducker.close()


def test_duck_que_paso_el_chequeo_sin_lock_antes_de_la_senal_no_baja_la_musica():
    """El worker pasó el chequeo sin lock y espera el lock (tomado por quien
    sea) cuando llega la señal: al conseguirlo, ya no atenúa, y ni pregunta a
    playerctl (nada de procesos ni D-Bus con el servicio saliendo)."""

    class _CerrojoEspiado:
        def __init__(self) -> None:
            self.real = threading.Lock()
            self.esperando = threading.Event()

        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            self.esperando.set()
            return self.real.acquire(blocking, timeout)

        def release(self) -> None:
            self.real.release()

        def __enter__(self) -> _CerrojoEspiado:
            self.acquire()
            return self

        def __exit__(self, *exc: object) -> None:
            self.release()

    backend = _Spotify()
    ducker = Ducker([backend], factor=0.2)
    cerrojo = _CerrojoEspiado()
    ducker._lock = cerrojo  # type: ignore[assignment]
    cerrojo.real.acquire()  # alguien (un restore lento, el hilo principal) lo tiene
    hilo = threading.Thread(target=ducker.duck)
    try:
        hilo.start()
        assert cerrojo.esperando.wait(5)  # el duck ya pasó el chequeo sin lock
        ducker.emergency_restore(lock_timeout_s=0.05)
    finally:
        cerrojo.real.release()
    hilo.join(5)
    assert not hilo.is_alive()
    assert backend.vol == {"spotify": "0.800000"}
    assert not ducker.is_ducked()
    assert backend.listados == 0


def test_duck_a_medias_cuando_emergency_restore_no_consigue_el_lock_no_escribe():
    """D-Bus colgado: el worker tiene el lock esperando la lectura del volumen
    más que lock_timeout_s; emergency_restore sigue sin él. Cuando la lectura
    vuelve, el duck ya no escribe el volumen bajo."""

    class _Lento(_Spotify):
        def __init__(self) -> None:
            super().__init__()
            self.leyendo = threading.Event()
            self.soltar = threading.Event()

        def get_volume(self, player: str) -> str | None:
            self.leyendo.set()
            self.soltar.wait(5)
            return super().get_volume(player)

    backend = _Lento()
    ducker = Ducker([backend], factor=0.2)
    hilo = threading.Thread(target=ducker.duck)
    hilo.start()
    try:
        assert backend.leyendo.wait(5)
        ducker.emergency_restore(lock_timeout_s=0.05)
    finally:
        backend.soltar.set()
    hilo.join(5)
    assert not hilo.is_alive()
    assert backend.vol == {"spotify": "0.800000"}
    assert not ducker.is_ducked()
