"""Rendimiento: estado que se volvía a preguntar en vez de sostenerlo y avisar.

Paso 1: todo hijo suelto tiene quien lo recoja. El panel lanzaba la terminal
del registro (`journalctl -f`) con un Popen que nadie esperaba: al cerrarla
quedaba <defunct> hasta que otro subprocess del panel disparaba la limpieza
interna de subprocess (con la ventana oculta no hay tick: zombi hasta cerrar el
panel). run_detached y _do_async ya recogían a sus hijos (arreglos de la
auditoría); aquí se fija que lo sigan haciendo.

Paso 2: el MediaMonitor deja de lanzar `playerctl -a metadata` cada 1.5 s para
siempre (57.600 procesos al día en reposo). Un solo `playerctl --follow` vivo
avisa de cada cambio; la verdad sigue siendo la foto de siempre (refresh_now),
tomada solo cuando una línea contradice el caché. Si el seguidor no arranca,
muere o no hay playerctl, vuelve el sondeo de siempre; [voice].media_watch =
"poll" devuelve exactamente el comportamiento de antes.

Paso 3: el apagado y el canal de control DESPIERTAN en vez de preguntarse por
reloj. Mouth/Brain/Stt hacían queue.get(timeout=0.2) solo para releer el flag
de apagado (900 despertares por minuto en reposo) y el hilo principal hacía
stat() de control.json cada 0.3 s (200 por minuto). Ahora el apagado mete un
STOP en cada cola, control.json se vigila con inotify y un hilo que termina
avisa al bucle principal. Sin inotify (o si deja de funcionar) vuelve el sondeo
de siempre, con un solo aviso.

Paso 4: un wake (o un barge-in) es UNA publicación: new_turn(mode=LISTENING)
abre el turno y pasa a escuchar de una vez, en vez de publicar "idle" con el
turno nuevo y enseguida "listening".

Paso 6: el panel sostiene su hilo de consulta y lo que ya pintó. Cada tick
(1.5 s, con la ventana abierta) arrancaba un hilo nuevo para el `systemctl show`
(40 por minuto) y reescribía los dos switches y las dos sensibilidades con el
mismo valor. Ahora un solo hilo duerme en un Event entre consultas (la ventana
lo para al destruirse) y un tick sin cambios no escribe ningún widget.
"""

from __future__ import annotations

import collections
import errno
import importlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo.voice import loop as voice_loop
from crotolamo.voice import notify
from crotolamo.voice.loop import VoiceLoop
from crotolamo.voice.media_aware import Ducker, MediaMonitor, PlayerctlBackend
from crotolamo.voice.notify import DirWatch, Wakeable
from crotolamo.voice.state import Mode, SharedState
from crotolamo.voice.threads import (
    STOP,
    BrainThread,
    EarThread,
    MouthThread,
    SttThread,
    Utterance,
    _drain_queue,
)
from interfaces import listener
from tests.test_auditoria_desktop_py import _Contado, _gi_falso, _ImagenContada, _Reloj

REPO = Path(__file__).resolve().parent.parent
DESKTOP_DIR = str(REPO / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from panel import systemd  # noqa: E402


def _espera(condicion, timeout: float = 10.0) -> bool:
    """Sondea hasta que se cumpla; el margen largo es por la máquina cargada
    (solo se agota cuando el test va a fallar)."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if condicion():
            return True
        time.sleep(0.01)
    return bool(condicion())


def _estado_proc(pid: int) -> str | None:
    """Estado del proceso según /proc ('Z' = zombi), o None si ya no existe."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[0]


def _vivo(pid: int) -> bool:
    """Corriendo de verdad: ni recogido ni zombi (muerto sin recoger)."""
    return _estado_proc(pid) not in (None, "Z")


@pytest.fixture
def pids_lanzados(monkeypatch):
    """Popen de verdad que anota el pid de cada hijo (solo el pid: guardar el
    objeto taparía su __del__ y cambiaría lo que se mide)."""
    pids: list[int] = []
    llamadas: list[tuple[list[str], dict]] = []
    real = subprocess.Popen

    def espia(args, *a, **kw):
        proc = real(args, *a, **kw)
        pids.append(proc.pid)
        llamadas.append((list(args), dict(kw)))
        return proc

    monkeypatch.setattr(subprocess, "Popen", espia)
    return SimpleNamespace(pids=pids, llamadas=llamadas)


# ===========================================================================
# Paso 1: nadie deja hijos <defunct>
# ===========================================================================

def test_spawn_reaped_recoge_al_hijo_que_termina(pids_lanzados):
    proc = systemd.spawn_reaped(["true"])
    assert proc.pid == pids_lanzados.pids[-1]  # devuelve el Popen, como antes
    pid = pids_lanzados.pids[-1]
    del proc  # nadie más lo espera: el hilo de spawn_reaped es su único dueño
    assert _espera(lambda: _estado_proc(pid) is None), "quedó <defunct>"


def test_spawn_reaped_mismos_argumentos_que_popen(pids_lanzados):
    systemd.spawn_reaped(["true", "x"], stdout=subprocess.DEVNULL, start_new_session=True)
    assert pids_lanzados.llamadas == [
        (["true", "x"], {"stdout": subprocess.DEVNULL, "start_new_session": True})]


def test_spawn_reaped_oserror_se_propaga_sin_arrancar_hilos(monkeypatch):
    hilos: list = []
    monkeypatch.setattr(systemd, "threading",
                        SimpleNamespace(Thread=lambda *a, **kw: hilos.append(kw)))
    with pytest.raises(OSError):
        systemd.spawn_reaped(["crotolamo-binario-que-no-existe"])
    assert hilos == []


def test_do_async_no_deja_zombis(monkeypatch, pids_lanzados):
    """systemctl cambiado por /bin/true: tres acciones del panel, cero zombis."""
    espia = subprocess.Popen

    def popen_sin_systemctl(args, *a, **kw):
        args = ["true"] if args and str(args[0]) == "systemctl" else args
        return espia(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen_sin_systemctl)
    hechos = threading.Semaphore(0)
    for verbo in ("stop", "start", "restart"):
        systemd._do_async(verbo, systemd.SERVICE, on_done=lambda res: hechos.release())
    for _ in range(3):
        assert hechos.acquire(timeout=10)
    assert len(pids_lanzados.pids) == 3
    assert all(_espera(lambda p=p: _estado_proc(p) is None) for p in pids_lanzados.pids)


_MODULOS_PANEL = ("common.gtk_util", "panel.window")


@pytest.fixture
def panel_gtk(monkeypatch, tmp_path):
    """panel.window REAL sobre el gi falso de test_auditoria_desktop_py."""
    from common import ipc
    from panel import config

    reloj = _Reloj()
    gi, repo = _gi_falso(reloj)
    for nombre in ("ApplicationWindow", "HeaderBar", "Box", "Label", "Button",
                   "Separator", "Switch", "ComboBoxText"):
        setattr(repo.Gtk, nombre, _Contado)
    repo.Gtk.Image = _ImagenContada
    nombres = ("gi", "gi.repository", *_MODULOS_PANEL)
    guardados = {n: sys.modules.pop(n, None) for n in nombres}
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repo
    try:
        window = importlib.import_module("panel.window")
        monkeypatch.setattr(ipc, "HUD_STATE_FILE", tmp_path / "hud_state.json")
        monkeypatch.setattr(config, "ENV_FILE", tmp_path / "listener.env")
        monkeypatch.setattr(window, "_do_async", lambda *a, on_done=None: None)
        yield SimpleNamespace(reloj=reloj, window=window, win=window.Panel(app=None))
    finally:
        for n in nombres:
            sys.modules.pop(n, None)
            if guardados[n] is not None:
                sys.modules[n] = guardados[n]
        for n in _MODULOS_PANEL:
            paquete, _, sub = n.partition(".")
            padre = sys.modules.get(paquete)
            if padre is None:
                continue
            if guardados[n] is not None:
                setattr(padre, sub, guardados[n])
            elif hasattr(padre, sub):
                delattr(padre, sub)


def test_panel_ver_registro_en_terminal_no_deja_zombi(panel_gtk, monkeypatch, pids_lanzados):
    """La terminal del registro se cierra (aquí: sale al instante) con el panel
    oculto y sin ningún otro subprocess: antes quedaba <defunct>."""
    terminal = ["sh", "-c", "exit 0", "terminal"]
    monkeypatch.setattr(panel_gtk.window, "_find_terminal", lambda: list(terminal))
    panel_gtk.win.on_log(None)
    (argv, kwargs), = pids_lanzados.llamadas
    # mismo argv y mismos kwargs que el Popen de antes
    assert argv == [*terminal, "journalctl", "--user", "-u", systemd.SERVICE, "-n", "200", "-f"]
    assert kwargs == {}
    assert panel_gtk.win._log_inflight is False  # la terminal abrió: sin diálogo
    pid = pids_lanzados.pids[-1]
    assert _espera(lambda: _estado_proc(pid) is None), "la terminal quedó <defunct>"


def test_panel_terminal_que_no_arranca_cae_al_dialogo(panel_gtk, monkeypatch, caplog):
    win = panel_gtk.win
    monkeypatch.setattr(panel_gtk.window, "_find_terminal",
                        lambda: ["crotolamo-terminal-que-no-existe", "-e"])
    leido = threading.Event()

    def journalctl(cmd, timeout=8.0):
        leido.set()
        return subprocess.CompletedProcess(cmd, 0, stdout="línea\n", stderr="")

    monkeypatch.setattr(panel_gtk.window, "run", journalctl)
    with caplog.at_level(logging.WARNING, logger="crotolamo.desktop.panel"):
        win.on_log(None)
    assert "No se pudo abrir la terminal" in caplog.text
    assert leido.wait(10)  # el OSError de Popen sigue llevando al diálogo


# ===========================================================================
# Paso 2: MediaMonitor por eventos (`playerctl --follow`) con el sondeo de
# siempre como respaldo
# ===========================================================================

class _Seguidor:
    """Popen de mentira con un pipe DE VERDAD: select() y os.read() funcionan."""

    def __init__(self) -> None:
        r, w = os.pipe()
        self.stdout = open(r, "rb", buffering=0)
        self._w: int | None = w
        self.returncode: int | None = None
        self.pid = -1
        self.terminados = 0
        self.esperas = 0

    def emitir(self, *lineas: str) -> None:
        assert self._w is not None, "el seguidor ya murió"
        os.write(self._w, "".join(f"{linea}\n" for linea in lineas).encode())

    def morir(self, codigo: int = 1) -> None:
        if self.returncode is None:
            self.returncode = codigo
        if self._w is not None:
            os.close(self._w)
            self._w = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminados += 1
        self.morir(-15)

    def kill(self):
        self.morir(-9)

    def wait(self, timeout=None):
        self.esperas += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("playerctl", timeout)
        return self.returncode


class _BackendSeguible:
    """Backend en memoria con follow(): cuenta fotos y seguidores lanzados."""

    def __init__(self, playing=(), *, instalado=True, arranca=True) -> None:
        self.playing = list(playing)
        self.instalado = instalado
        self.arranca = arranca
        self.fotos = 0
        self.seguidores: list[_Seguidor] = []
        self.follow_llamadas = 0

    def playing_players(self):
        self.fotos += 1
        return list(self.playing)

    def get_volume(self, player):
        return "1.0"

    def set_volume(self, player, value):
        return True

    def available(self) -> bool:
        return self.instalado

    def follow(self):
        self.follow_llamadas += 1
        if not self.arranca:
            return None
        seguidor = _Seguidor()
        self.seguidores.append(seguidor)
        return seguidor

    @property
    def vivo(self) -> _Seguidor:
        return self.seguidores[-1]


@pytest.fixture
def monitores():
    """MediaMonitor con tiempos de test; todos se paran al final, pase o falle."""
    creados: list[MediaMonitor] = []

    def crear(*backends, poll_s=0.05, watch="follow", **plazos) -> MediaMonitor:
        mon = MediaMonitor(list(backends), poll_s=poll_s, watch=watch,
                           threshold_normal=0.72, threshold_media=0.85)
        for nombre, valor in plazos.items():
            setattr(mon, nombre, valor)
        creados.append(mon)
        return mon

    yield crear
    for mon in creados:
        mon.stop()


def _siguiendo(mon: MediaMonitor, backend: _BackendSeguible, n: int = 1) -> bool:
    return _espera(lambda: len(backend.seguidores) >= n and mon._follower is backend.vivo)


def test_seguidor_en_reposo_no_toma_fotos(monitores):
    """Antes: una foto (un proceso playerctl) cada poll_s para siempre."""
    backend = _BackendSeguible()
    mon = monitores(backend)
    mon.start()
    assert _siguiendo(mon, backend)
    time.sleep(0.5)  # 10 sondeos del sondeo de siempre
    assert backend.fotos == 1  # solo la foto inicial
    assert backend.follow_llamadas == 1
    assert mon.is_playing() is False


def test_linea_que_cambia_algo_una_foto_la_que_confirma_ninguna(monitores):
    # PLAYING_REFRESH_S alto: aquí solo cuentan las fotos que piden los eventos.
    backend = _BackendSeguible()
    mon = monitores(backend, PLAYING_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    # Sin música, un reproductor en pausa (o que ya no queda ninguno) no cambia
    # nada: cero fotos.
    backend.vivo.emitir("firefox\tPaused", "vlc\tStopped", "")
    time.sleep(0.3)
    assert backend.fotos == 1
    backend.playing = ["spotify"]
    backend.vivo.emitir("spotify\tPlaying")
    assert _espera(mon.is_playing)
    assert backend.fotos == 2
    assert mon.threshold() == pytest.approx(0.85)
    # Lo que confirma el caché no cuesta nada: el mismo que suena, la voz propia.
    backend.vivo.emitir("spotify\tPlaying", "Crotolamo TTS\tPlaying")
    time.sleep(0.3)
    assert backend.fotos == 2
    backend.playing = []
    backend.vivo.emitir("spotify\tPaused")
    assert _espera(lambda: not mon.is_playing())
    assert backend.fotos == 3
    assert mon.threshold() == pytest.approx(0.72)


def test_se_cierra_el_que_sonaba_y_queda_otro_en_pausa(monitores):
    """playerctl real: al cerrarse el que sonaba imprime la línea del que queda
    ("alpha<TAB>Paused"), no una vacía. Con música en el caché, eso pide foto."""
    backend = _BackendSeguible(playing=["beta"])
    mon = monitores(backend, PLAYING_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    assert mon.playing_players() == ["beta"]
    backend.playing = []
    backend.vivo.emitir("alpha\tPaused")
    assert _espera(lambda: not mon.is_playing())
    assert backend.fotos == 2


def test_cierre_silencioso_del_que_sonaba_se_ve_como_con_el_sondeo(monitores):
    """playerctl real: si se cierra uno que sonaba pero no era el último que
    cambió, --follow no imprime NADA. Con música en el caché sigue la foto de
    cada poll_s de antes, así que eso se ve igual de rápido que con el sondeo;
    sin música, vuelve el reposo sin fotos."""
    backend = _BackendSeguible(playing=["alpha"])
    mon = monitores(backend, poll_s=0.2)
    mon.start()
    assert _siguiendo(mon, backend)
    assert mon.is_playing()
    backend.playing = []  # se cierra sin que el seguidor diga nada
    t0 = time.monotonic()
    assert _espera(lambda: not mon.is_playing())
    assert time.monotonic() - t0 < 1.0
    fotos = backend.fotos
    time.sleep(0.6)
    assert backend.fotos == fotos  # sin música: cero fotos otra vez


def test_linea_vacia_reproductor_que_desaparece_dispara_una_foto(monitores):
    backend = _BackendSeguible(playing=["spotify"])
    mon = monitores(backend, PLAYING_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    assert mon.is_playing()
    backend.playing = []
    backend.vivo.emitir("")
    assert _espera(lambda: not mon.is_playing())
    assert backend.fotos == 2


def test_linea_basura_vuelve_a_mirar_la_verdad_sin_romper_nada(monitores):
    backend = _BackendSeguible()
    mon = monitores(backend, PLAYING_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    backend.playing = ["vlc"]
    backend.vivo.emitir("\x00\xff basura sin tabulador")
    assert _espera(mon.is_playing)  # la foto manda; el hilo sigue vivo
    assert mon.playing_players() == ["vlc"]
    backend.vivo.emitir("\tPlaying")  # sin nombre: lo mismo que ignora el parser
    time.sleep(0.2)
    assert backend.fotos == 2
    assert mon._thread is not None and mon._thread.is_alive()


def test_rafaga_de_eventos_no_supera_el_ritmo_del_sondeo(monitores):
    """Un reproductor que parpadea mil veces por segundo no dispara mil
    procesos: como mucho una foto por poll_s (lo que costaba el sondeo), y la
    última foto llega a más tardar poll_s después: nunca más tarde que antes."""
    backend = _BackendSeguible()
    mon = monitores(backend, poll_s=0.5, PLAYING_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    # La primera línea llega SOLA y su foto inmediata se toma antes de la
    # ráfaga. Si el hilo leyera todo de una vez (máquina cargada), esa foto ya
    # vería el estado final y no quedaría ninguna pendiente que esperar.
    backend.playing = ["spotify"]
    backend.vivo.emitir("spotify\tPlaying")
    assert _espera(lambda: backend.fotos >= 2 and mon.is_playing())
    t0 = time.monotonic()
    for i in range(200):
        backend.playing = [] if i % 2 == 0 else ["spotify"]
        backend.vivo.emitir("spotify\tPaused" if i % 2 == 0 else "spotify\tPlaying")
    backend.playing = ["spotify"]
    backend.vivo.emitir("spotify\tPlaying")
    assert _espera(lambda: backend.fotos >= 3)
    assert time.monotonic() - t0 < 2.0
    time.sleep(0.3)
    assert backend.fotos <= 3  # inicial + inmediata + la pendiente de la ráfaga
    assert mon.is_playing() and mon.playing_players() == ["spotify"]


# --- Una foto que no respondió no es "no suena nada" -------------------------
# playerctl colgado (TimeoutExpired, un reproductor congelado en el bus) u OSError:
# PlayerctlBackend.playing_players() da [] y el caché pasa a "sin música", igual
# que con el sondeo. Pero el seguidor no repite la línea de un reproductor que
# sigue sonando, así que nadie más avisaba: el caché quedaba en modo normal (umbral
# 0.72 con música sonando) hasta la foto de seguridad, 300 s, en vez de poll_s.

class _BackendQueFalla(_BackendSeguible):
    """Como PlayerctlBackend: cuando playerctl no respondió, playing_players() da
    [] (lo de siempre) y answered() lo dice."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.fallos = 0
        self._respondio = True

    def playing_players(self):
        self.fotos += 1
        self._respondio = not self.fallos
        if self.fallos:
            self.fallos -= 1
            return []
        return list(self.playing)

    def answered(self) -> bool:
        return self._respondio


@pytest.mark.parametrize("fallos", [1, 3])
def test_foto_que_no_responde_con_musica_se_reintenta_como_el_sondeo(monitores, fallos):
    vuelta = {}
    for watch in ("poll", "follow"):
        backend = _BackendQueFalla(playing=["spotify"])
        mon = monitores(backend, poll_s=0.1, watch=watch, SAFETY_REFRESH_S=60.0)
        mon.start()
        assert _espera(mon.is_playing)
        if watch == "follow":
            assert _siguiendo(mon, backend)
        backend.fallos = fallos  # la foto siguiente (y las que sigan) no responden
        assert _espera(lambda: backend.fallos == 0), watch
        t0 = time.monotonic()
        assert _espera(mon.is_playing, timeout=5), f"{watch}: sigue en modo normal"
        vuelta[watch] = time.monotonic() - t0
        mon.stop()
    # Lo mismo que el sondeo: la foto siguiente, poll_s después (margen por carga).
    assert vuelta["follow"] < 1.0, vuelta


def test_foto_de_un_evento_que_no_responde_se_reintenta(monitores):
    """Arranca la música y la foto que pidió su línea no responde: playerctl no
    vuelve a imprimir "spotify<TAB>Playing" mientras siga sonando."""
    backend = _BackendQueFalla()
    mon = monitores(backend, poll_s=0.1, SAFETY_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    backend.fallos = 2
    backend.playing = ["spotify"]
    backend.vivo.emitir("spotify\tPlaying")
    assert _espera(lambda: backend.fallos == 0)
    t0 = time.monotonic()
    assert _espera(mon.is_playing, timeout=5)
    assert time.monotonic() - t0 < 1.0


def test_playerctl_que_no_responde_se_reintenta_al_ritmo_del_sondeo_y_nada_mas(monitores):
    """Mientras no responda, una foto cada poll_s (lo del sondeo: ni más rápido ni
    nunca); en cuanto responde, otra vez el reposo sin fotos."""
    backend = _BackendQueFalla()
    backend.fallos = 10**6
    mon = monitores(backend, poll_s=0.1, SAFETY_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    fotos, t0 = backend.fotos, time.monotonic()
    assert _espera(lambda: backend.fotos >= fotos + 3)
    time.sleep(0.3)
    hechas, lapso = backend.fotos - fotos, time.monotonic() - t0
    assert hechas <= lapso / 0.1 + 1, (hechas, lapso)
    backend.fallos = 0
    assert _espera(lambda: backend.fotos >= fotos + hechas + 1)
    time.sleep(0.15)  # la que respondió ya pasó
    quietas = backend.fotos
    time.sleep(0.5)
    assert backend.fotos == quietas


def test_playerctl_answered_distingue_no_suena_de_no_respondio(monkeypatch):
    """playing_players() no cambia (todo fallo sigue siendo [], mismo argv);
    answered() dice además si playerctl respondió, por hilo (el Ducker comparte
    el backend desde el suyo)."""
    from crotolamo.voice import media_aware

    argv = ["playerctl", "-a", "metadata", "--format", "{{playerName}}\t{{status}}"]
    estado: dict = {"which": "/usr/bin/playerctl", "raise": None, "result": None}
    llamadas: list[list[str]] = []

    def run_cmd(args, timeout=10):
        llamadas.append(list(args))
        assert timeout == 2.0
        if estado["raise"] is not None:
            raise estado["raise"]
        return estado["result"]

    monkeypatch.setattr(media_aware.shutil, "which", lambda name: estado["which"])
    monkeypatch.setattr(media_aware, "run_cmd", run_cmd)
    backend = PlayerctlBackend()
    assert backend.answered() is True  # nada preguntado aún
    estado["result"] = subprocess.CompletedProcess(argv, 0, "spotify\tPlaying\nvlc\tPaused\n", "")
    assert backend.playing_players() == ["spotify"] and backend.answered() is True
    # "No players found" es una respuesta: no suena nada.
    estado["result"] = subprocess.CompletedProcess(argv, 1, "", "No players found\n")
    assert backend.playing_players() == [] and backend.answered() is True
    for error in (subprocess.TimeoutExpired("playerctl", 2.0), OSError(11, "EAGAIN")):
        estado["raise"] = error
        assert backend.playing_players() == []  # lo de siempre...
        assert backend.answered() is False  # ...pero no se sabe
        # Otro hilo (el Ducker) no ve el fallo de este.
        otro: list[bool] = []
        hilo = threading.Thread(target=lambda: otro.append(backend.answered()))
        hilo.start()
        hilo.join(5)
        assert otro == [True]
    estado["raise"] = None
    assert backend.playing_players() == [] and backend.answered() is True
    assert llamadas and all(a == argv for a in llamadas)
    # Sin playerctl: [] sin ejecutar nada, como siempre.
    estado["which"] = None
    antes = len(llamadas)
    assert backend.playing_players() == [] and backend.answered() is False
    assert len(llamadas) == antes


def test_seguidor_que_muere_cae_al_sondeo_y_reintenta(monitores):
    backend = _BackendSeguible(playing=["spotify"])
    mon = monitores(backend, FOLLOW_RETRY_S=0.4)
    mon.start()
    assert _siguiendo(mon, backend)
    primero = backend.vivo
    primero.morir(1)
    # El muerto se recoge (terminate/wait) y vuelve el sondeo de siempre...
    assert _espera(lambda: primero.esperas >= 1)
    fotos = backend.fotos
    backend.playing = []
    assert _espera(lambda: not mon.is_playing())
    assert _espera(lambda: backend.fotos >= fotos + 3)
    # ...hasta que pasa el plazo y arranca otro seguidor.
    assert _siguiendo(mon, backend, n=2)
    assert backend.follow_llamadas == 2
    fotos = backend.fotos
    time.sleep(0.3)
    assert backend.fotos == fotos  # con el nuevo seguidor, otra vez cero fotos


def test_seguidor_que_no_arranca_sondea_y_reintenta_cada_vez_mas_espaciado(monitores):
    backend = _BackendSeguible(arranca=False)
    mon = monitores(backend, FOLLOW_RETRY_S=0.2, FOLLOW_RETRY_MAX_S=0.4)
    mon.start()
    assert _espera(lambda: backend.follow_llamadas >= 1)
    backend.playing = ["spotify"]
    assert _espera(mon.is_playing)  # el sondeo de siempre sigue funcionando
    assert _espera(lambda: backend.follow_llamadas >= 3, timeout=10)
    t0 = time.monotonic()
    llamadas = backend.follow_llamadas
    assert _espera(lambda: backend.follow_llamadas > llamadas, timeout=10)
    assert time.monotonic() - t0 >= 0.3  # ya en el tope (0.4 s), no cada 0.2 s


def test_seguidor_que_no_arranca_avisa_una_sola_vez(monitores, caplog):
    backend = _BackendSeguible(arranca=False)
    mon = monitores(backend, FOLLOW_RETRY_S=0.1, FOLLOW_RETRY_MAX_S=0.1)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.media_aware"):
        mon.start()
        assert _espera(lambda: backend.follow_llamadas >= 3)
        mon.stop()
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1, [r.getMessage() for r in avisos]
    assert "playerctl --follow" in avisos[0].getMessage()


def test_stop_termina_y_recoge_al_seguidor(monitores):
    backend = _BackendSeguible()
    mon = monitores(backend)
    mon.start()
    assert _siguiendo(mon, backend)
    hilo = mon._thread
    mon.stop()
    assert backend.vivo.terminados >= 1 and backend.vivo.returncode is not None
    assert backend.vivo.esperas >= 1
    assert hilo is not None and not hilo.is_alive()
    assert mon._follower is None
    assert backend.vivo.stdout.closed  # el hilo cerró su pipe al salir


def test_sin_playerctl_no_lanza_nada_y_rechequea_espaciado(monitores):
    backend = _BackendSeguible(instalado=False)
    mon = monitores(backend, ABSENT_RECHECK_S=0.3)
    mon.start()
    assert _espera(lambda: backend.fotos >= 1)
    time.sleep(0.5)
    assert backend.follow_llamadas == 0
    assert backend.fotos <= 3  # cada 0.3 s, no cada poll_s (0.1 s: serían 5 o 6)
    backend.instalado = True  # lo instalan con el asistente corriendo
    assert _siguiendo(mon, backend)


def test_media_watch_poll_es_el_sondeo_de_siempre(monitores):
    backend = _BackendSeguible()
    mon = monitores(backend, watch="poll")
    mon.start()
    assert _espera(lambda: backend.fotos >= 5)
    assert backend.follow_llamadas == 0
    assert mon._follower is None


def test_backends_sin_follow_o_varios_usan_el_sondeo_de_siempre(monitores):
    uno, otro = _BackendSeguible(), _BackendSeguible()
    mon = monitores(uno, otro)
    mon.start()
    assert _espera(lambda: uno.fotos >= 5 and otro.fotos >= 5)
    assert uno.follow_llamadas == otro.follow_llamadas == 0

    class _SinFollow:  # como el FakeBackend de test_media_aware
        fotos = 0

        def playing_players(self):
            self.fotos += 1
            return ["vlc"]

    solo = _SinFollow()
    mon = monitores(solo)
    mon.start()
    assert _espera(lambda: solo.fotos >= 5 and mon.is_playing())
    assert mon._follower is None


@pytest.mark.parametrize("valor, esperado", [
    ("follow", "follow"), ("poll", "poll"), (" POLL ", "poll"), ("polling", "follow"),
    (None, "follow"), (3, "follow"),
])
def test_media_watch_valores(valor, esperado, caplog):
    with caplog.at_level(logging.WARNING, logger="crotolamo.voice.media_aware"):
        mon = MediaMonitor([], watch=valor)
    assert mon.watch == esperado
    assert ("media_watch" in caplog.text) is (str(valor).strip().lower() != esperado)


def test_terminate_children_manda_sigterm_y_no_lanza_otro(monitores):
    backend = _BackendSeguible()
    mon = monitores(backend, FOLLOW_RETRY_S=0.1)
    mon.start()
    assert _siguiendo(mon, backend)
    mon.terminate_children()
    assert backend.vivo.terminados == 1
    time.sleep(0.4)  # más que el plazo de reintento
    assert backend.follow_llamadas == 1


class _BackendQueTardaEnSeguir(_BackendSeguible):
    """follow() se queda a medio Popen (el proceso ya existe, aún no volvió)
    hasta que el test lo suelta."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.dentro = threading.Event()
        self.soltar = threading.Event()

    def follow(self):
        seguidor = super().follow()
        self.dentro.set()
        assert self.soltar.wait(10)
        return seguidor


def test_senal_a_medio_lanzar_el_seguidor_no_lo_deja_huerfano(monitores):
    """El manejador de señal corre terminate_children() y enseguida os._exit: lo
    que no termine él no lo termina nadie. Antes, una señal que llegaba con el
    hilo del monitor dentro del Popen no veía al seguidor (aún sin registrar), no
    mandaba nada y el playerctl quedaba huérfano (ppid 1) para siempre."""
    backend = _BackendQueTardaEnSeguir()
    mon = monitores(backend, _TERMINATE_WAIT_S=10.0)
    mon.start()
    assert backend.dentro.wait(10)
    hecho_antes_de_salir: list[int] = []

    def manejador() -> None:
        mon.terminate_children()
        hecho_antes_de_salir.append(backend.vivo.terminados)  # aquí iría os._exit

    hilo = threading.Thread(target=manejador)
    hilo.start()
    time.sleep(0.1)  # la señal llegó con el Popen en curso
    backend.soltar.set()
    hilo.join(10)
    assert hecho_antes_de_salir and hecho_antes_de_salir[0] >= 1
    assert backend.follow_llamadas == 1


def test_stop_durante_la_foto_de_partida_no_lanza_ningun_seguidor(monitores):
    """stop() con el hilo a medio sondeo (playerctl tarda): su join corto vence y
    el hilo, al volver de la foto, ya no lanza un seguidor que nadie esperaría."""
    backend = _BackendSeguible()
    en_la_foto, soltar = threading.Event(), threading.Event()
    original = backend.playing_players

    def foto_lenta():
        en_la_foto.set()
        assert soltar.wait(10)
        return original()

    backend.playing_players = foto_lenta
    mon = monitores(backend)
    mon.start()
    assert en_la_foto.wait(10)
    hilo = mon._thread
    parar = threading.Thread(target=mon.stop)
    parar.start()
    assert _espera(lambda: mon._stop.is_set())
    soltar.set()
    parar.join(10)
    assert hilo is not None
    hilo.join(10)
    assert not hilo.is_alive()
    assert backend.follow_llamadas == 0


def test_apagado_por_senal_termina_al_seguidor(monitores, monkeypatch):
    """os._exit se salta stop() y los finally: el manejador de señal tiene que
    mandarle SIGTERM al seguidor él mismo (como a los servers MCP)."""
    backend = _BackendSeguible()
    mon = monitores(backend)
    mon.start()
    assert _siguiendo(mon, backend)

    class _Salida(Exception):
        pass

    def _exit(code):
        raise _Salida(code)

    monkeypatch.setattr(listener.os, "_exit", _exit)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_flush_memoria_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_final_idle_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_ACTIVE_MEDIA", mon, raising=False)
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    assert backend.vivo.terminados == 1


def test_listener_config_lee_media_watch():
    from interfaces.listener import ListenerConfig

    class _S:
        voice: dict = {}
        wake: dict = {}

    assert ListenerConfig.from_settings(_S()).media_watch == "follow"
    _S.voice = {"media_watch": "poll"}
    assert ListenerConfig.from_settings(_S()).media_watch == "poll"


def test_config_de_fabrica_documenta_media_watch():
    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    voice = tomllib.loads(texto)["voice"]
    assert voice["media_watch"] == "follow"
    assert voice["media_poll_s"] == pytest.approx(1.5)
    bloque = texto[texto.index("# --- Wake con música sonando ---"):texto.index("[home]")]
    assert "media_watch" in bloque and '"poll"' in bloque


def test_config_de_fabrica_no_promete_de_mas_con_follow():
    """Lo que lee el patrón dice lo que cuesta "follow": sin música queda una
    foto de seguridad cada SAFETY_REFRESH_S (no "ningún proceso") y, mientras
    suena música, la de cada media_poll_s, igual que "poll"."""
    from crotolamo.settings import DEFAULT_CONFIG

    texto = DEFAULT_CONFIG.read_text(encoding="utf-8")
    bloque = texto[texto.index("# --- Wake con música sonando ---"):texto.index("[home]")]
    assert "ningún proceso" not in bloque
    assert f"cada {int(MediaMonitor.SAFETY_REFRESH_S // 60)} min" in bloque
    assert MediaMonitor.PLAYING_REFRESH_S is None  # con música: la cadencia del sondeo
    linea = next(x for x in bloque.splitlines() if x.startswith("media_poll_s"))
    assert "mientras suena música" in linea


# --- Con un `playerctl` falso de verdad en el PATH ---------------------------
# Procesos reales: el formato de --follow es el del playerctl 2.4.1 real
# (`nombre<TAB>estado` por cambio, el estado actual al arrancar y una línea vacía
# cuando un reproductor desaparece). Las líneas que "emite" el seguidor las mete
# el test por un FIFO; el estado de los reproductores vive en state.json.

_PLAYERCTL_FALSO = '''\
#!{python} -S
import json, os, sys
D = os.environ["FAKE_PLAYERCTL_DIR"]
argv = sys.argv[1:]
with open(os.path.join(D, "calls.log"), "a") as fh:
    fh.write(json.dumps({{"pid": os.getpid(), "argv": argv}}) + "\\n")
with open(os.path.join(D, "state.json")) as fh:
    st = json.load(fh)
if "--follow" in argv:
    with open(os.path.join(D, "follower.pid"), "a") as fh:
        fh.write(f"{{os.getpid()}}\\n")
    for name, status in st["players"]:
        sys.stdout.write(f"{{name}}\\t{{status}}\\n")
    sys.stdout.flush()
    with open(os.path.join(D, "events"), "r") as ev:
        for line in ev:
            if line.startswith("__EXIT__"):
                sys.exit(int(line.split()[1]))
            sys.stdout.write(line)
            sys.stdout.flush()
    sys.exit(0)
if "metadata" in argv and "-a" in argv:
    out = "".join(f"{{n}}\\t{{s}}\\n" for n, s in st["players"])
    if not out:
        sys.stderr.write("No players found\\n")
        sys.exit(1)
    sys.stdout.write(out)
    sys.exit(0)
if "-p" in argv and "volume" in argv:
    player = argv[argv.index("-p") + 1]
    rest = argv[argv.index("volume") + 1:]
    if rest:
        st["volumes"][player] = rest[0]
        with open(os.path.join(D, "state.json"), "w") as fh:
            json.dump(st, fh)
    else:
        print(st["volumes"].get(player, "1.000000"))
    sys.exit(0)
sys.exit(1)
'''


class _MundoPlayerctl:
    """El `playerctl` falso y sus reproductores."""

    def __init__(self, directorio: Path) -> None:
        self.dir = directorio
        self.players: list[tuple[str, str]] = []
        self.volumes: dict[str, str] = {}
        self._guardar()
        os.mkfifo(directorio / "events")
        # O_RDWR: el FIFO siempre tiene escritor (el seguidor no ve EOF por esto)
        # y abrirlo no bloquea a nadie.
        self._fifo = os.open(directorio / "events", os.O_RDWR)

    def _guardar(self) -> None:
        (self.dir / "state.json").write_text(
            json.dumps({"players": self.players, "volumes": self.volumes}), encoding="utf-8")

    def poner(self, players: list[tuple[str, str]], *eventos: str) -> None:
        """Cambia el estado y emite por --follow lo que emitiría el real."""
        self.players = list(players)
        self._guardar()
        for evento in eventos:
            os.write(self._fifo, f"{evento}\n".encode())

    def llamadas(self) -> list[dict]:
        log = self.dir / "calls.log"
        if not log.exists():
            return []
        return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]

    def fotos(self) -> int:
        return sum(1 for c in self.llamadas() if "--follow" not in c["argv"]
                   and "metadata" in c["argv"])

    def seguidores(self) -> list[int]:
        archivo = self.dir / "follower.pid"
        if not archivo.exists():
            return []
        return [int(x) for x in archivo.read_text(encoding="utf-8").split()]

    def cerrar(self) -> None:
        os.close(self._fifo)


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "playerctl"
    exe.write_text(_PLAYERCTL_FALSO.format(python=sys.executable), encoding="utf-8")
    exe.chmod(0o755)
    datos = tmp_path / "datos"
    datos.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_PLAYERCTL_DIR", str(datos))
    m = _MundoPlayerctl(datos)
    yield m
    for pid in m.seguidores():  # ningún test deja un playerctl vivo, pase o falle
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    m.cerrar()


def test_playerctl_falso_mismas_decisiones_que_el_sondeo(mundo, monitores):
    """La misma secuencia de estados con el seguidor y con el sondeo de siempre:
    mismo umbral del wake, mismos reproductores y el Ducker baja y repone
    exactamente lo mismo. Incluye: uno que aparece, pausa, dos a la vez, la voz
    propia, basura, uno que desaparece y el seguidor que muere."""
    seguidor = monitores(PlayerctlBackend(), poll_s=0.1, FOLLOW_RETRY_S=0.5)
    sondeo = monitores(PlayerctlBackend(), poll_s=0.1, watch="poll")
    seguidor.start()
    assert _espera(lambda: len(mundo.seguidores()) == 1 and seguidor._follower is not None)
    pid = mundo.seguidores()[0]
    assert seguidor._follower.pid == pid

    # Las líneas son las que imprime el playerctl 2.4.1 real en cada caso
    # (capturadas con reproductores MPRIS falsos en un bus de sesión privado).
    tts = ("Crotolamo TTS", "Playing")
    pasos: list[tuple[list[tuple[str, str]], tuple[str, ...]]] = [
        ([("spotify", "Playing")], ("spotify\tPlaying",)),                    # aparece
        ([("spotify", "Paused")], ("spotify\tPaused",)),                      # pausa
        ([("spotify", "Playing"), ("firefox", "Playing")],                    # dos a la vez
         ("spotify\tPlaying", "firefox\tPlaying")),
        ([("spotify", "Playing"), ("firefox", "Playing"), tts],
         ("Crotolamo TTS\tPlaying",)),                                        # voz propia
        ([("spotify", "Paused"), ("firefox", "Playing"), tts],
         ("basura sin tabulador", "spotify\tPaused")),                         # basura
        ([("spotify", "Paused"), tts], ()),          # se cierra firefox, que no era el último
        ([("spotify", "Paused"), ("firefox", "Playing"), tts], ("firefox\tPlaying",)),
        ([("spotify", "Paused"), tts], ("spotify\tPaused",)),  # se cierra el último que cambió
        ([tts], ("Crotolamo TTS\tPlaying",)),
        ([], ("",)),                                                           # no queda ninguno
    ]
    for players, eventos in pasos:
        mundo.poner(players, *eventos)
        esperado = sondeo.refresh_now()
        assert _espera(lambda: seguidor.playing_players() == sondeo.playing_players()), players
        assert seguidor.is_playing() is esperado
        assert seguidor.threshold() == sondeo.threshold()
    assert mundo.seguidores() == [pid]  # todo con el mismo seguidor vivo
    time.sleep(0.3)  # que termine lo que haya pendiente
    fotos = mundo.fotos()
    time.sleep(0.5)  # 5 vueltas del sondeo de antes (poll_s = 0.1)
    assert mundo.fotos() == fotos  # sin música ni eventos: ni un proceso

    # El seguidor muere: vuelve el sondeo y el estado sigue al día.
    mundo.poner([("spotify", "Paused")], "__EXIT__ 1")
    assert _espera(lambda: _estado_proc(pid) is None)  # muerto Y recogido
    mundo.poner([("vlc", "Playing")])  # sin evento: nadie lo avisa
    assert _espera(lambda: seguidor.playing_players() == ["vlc"])
    assert _espera(lambda: len(mundo.seguidores()) == 2, timeout=10)  # y reintenta

    # El Ducker no cambia: baja y repone lo mismo que con el sondeo.
    mundo.volumes = {"vlc": "0.800000"}
    mundo.poner([("vlc", "Playing"), ("Crotolamo TTS", "Playing")])
    ducker = Ducker([PlayerctlBackend()], factor=0.2)
    ducker.duck()
    assert json.loads((mundo.dir / "state.json").read_text())["volumes"]["vlc"] == "0.160"
    ducker.restore()
    assert json.loads((mundo.dir / "state.json").read_text())["volumes"]["vlc"] == "0.800000"

    nuevo = mundo.seguidores()[-1]
    seguidor.stop()
    assert _estado_proc(nuevo) is None  # stop() lo termina y lo recoge: ni zombi


def test_playerctl_falso_que_no_responde_no_deja_el_cache_viejo(mundo, monitores, monkeypatch):
    """PlayerctlBackend de verdad con procesos reales: la foto expira
    (TimeoutExpired: playerctl > 2 s con la máquina cargada o un reproductor
    colgado) y el seguidor no tiene nada nuevo que decir. Con música sonando, y
    con la música que arranca justo cuando falla la foto de su evento."""
    from crotolamo.voice import media_aware

    real = media_aware.run_cmd
    fallar = {"n": 0}

    def run_cmd(args, timeout=10):
        if fallar["n"] and "metadata" in args and "--follow" not in args:
            fallar["n"] -= 1
            raise subprocess.TimeoutExpired(args, timeout)
        return real(args, timeout=timeout)

    monkeypatch.setattr(media_aware, "run_cmd", run_cmd)
    mon = monitores(PlayerctlBackend(), poll_s=0.2, SAFETY_REFRESH_S=60.0)
    mundo.poner([("spotify", "Playing")])
    mon.start()
    assert _espera(lambda: mon._follower is not None and mon.is_playing())
    fallar["n"] = 2  # dos fotos seguidas sin respuesta
    assert _espera(lambda: fallar["n"] == 0)
    t0 = time.monotonic()
    assert _espera(mon.is_playing, timeout=5), "sigue en modo normal con música sonando"
    assert time.monotonic() - t0 < 2.0

    mundo.poner([("spotify", "Paused")], "spotify\tPaused")
    assert _espera(lambda: not mon.is_playing())
    fallar["n"] = 1
    mundo.poner([("spotify", "Playing")], "spotify\tPlaying")
    assert _espera(lambda: fallar["n"] == 0)
    t0 = time.monotonic()
    assert _espera(mon.is_playing, timeout=5), "la música que arrancó no se vio"
    assert time.monotonic() - t0 < 2.0


_SCRIPT_LISTENER = '''\
import sys, time
sys.path.insert(0, {repo!r})
from crotolamo.voice.media_aware import MediaMonitor, PlayerctlBackend
from interfaces import listener
mon = MediaMonitor([PlayerctlBackend()], poll_s=0.1)
listener._ACTIVE_MEDIA = mon
listener._install_signal_handlers()
mon.start()
for _ in range(1000):
    if mon._follower is not None:
        break
    time.sleep(0.01)
print("listo", mon._follower.pid, flush=True)
if {modo!r} == "senal":
    time.sleep(60)
# "fin": sale sin stop(), como un return temprano o una excepción de run_listen
sys.stdin.readline()
'''


@pytest.mark.parametrize("modo", ["senal", "fin"])
def test_el_listener_no_deja_playerctl_huerfano(mundo, tmp_path, modo):
    """De punta a punta, con procesos reales: SIGTERM (systemctl stop) y una
    salida normal sin stop() no dejan el `playerctl --follow` corriendo."""
    script = tmp_path / "listener_falso.py"
    script.write_text(_SCRIPT_LISTENER.format(repo=str(REPO), modo=modo), encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path), CROTOLAMO_LOG="CRITICAL")
    proc = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, cwd=str(tmp_path))
    try:
        linea = proc.stdout.readline()
        assert linea.startswith("listo"), proc.stderr.read()
        pid = int(linea.split()[1])
        assert _vivo(pid)
        if modo == "senal":
            proc.send_signal(signal.SIGTERM)
        else:
            proc.stdin.write("\n")
            proc.stdin.flush()
        assert proc.wait(timeout=20) == 0
        assert _espera(lambda: not _vivo(pid)), "playerctl --follow quedó huérfano"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# --- El apagado normal no es una caída del seguidor ---------------------------
# systemctl stop (KillMode=control-group) manda SIGTERM a todo el cgroup a la vez
# y Ctrl-C/SIGHUP llegan a todo el grupo de la terminal: el `playerctl --follow`
# muere en el mismo instante que el listener. El hilo del monitor veía su EOF
# antes de que el manejador de señal marcara el stop y escribía en el journal un
# WARNING falso ("playerctl --follow terminó (código -15)...") y lanzaba un
# playerctl más (la foto del sondeo de respaldo) con el proceso saliendo.

def _avisos_del_seguidor(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "playerctl --follow" in r.getMessage()]


@pytest.mark.parametrize("senal", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_seguidor_muerto_por_la_senal_del_apagado_no_avisa_ni_sondea(monitores, caplog,
                                                                       senal):
    backend = _BackendSeguible()
    mon = monitores(backend, poll_s=0.5)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.media_aware"):
        mon.start()
        assert _siguiendo(mon, backend)
        fotos, hilo = backend.fotos, mon._thread
        backend.vivo.morir(-senal)  # la misma señal le llega al seguidor...
        assert _espera(lambda: backend.vivo.esperas >= 1)  # (el hilo ya vio su EOF)
        time.sleep(0.1)
        mon.terminate_children()  # ...y un instante después corre el manejador
        assert hilo is not None
        hilo.join(5)
        assert not hilo.is_alive()
    assert _avisos_del_seguidor(caplog) == []
    assert backend.fotos == fotos  # ni un playerctl más saliendo
    assert backend.follow_llamadas == 1


def test_apagado_con_la_musica_lenta_de_reponer_tampoco_avisa(monitores, caplog, monkeypatch):
    """El manejador repone primero la música atenuada: con un reproductor colgado
    en D-Bus eso tarda (1 s de candado + 2 s por reproductor). Marcar el monitor
    como parado va ANTES, para que ese rato no cuente como caída del seguidor."""
    backend = _BackendSeguible()
    mon = monitores(backend, poll_s=0.2)

    class _Salida(Exception):
        pass

    def _exit(code):
        raise _Salida(code)

    monkeypatch.setattr(listener.os, "_exit", _exit)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: time.sleep(0.8))
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_flush_memoria_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_final_idle_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_ACTIVE_MEDIA", mon, raising=False)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.media_aware"):
        mon.start()
        assert _siguiendo(mon, backend)
        fotos, hilo = backend.fotos, mon._thread
        backend.vivo.morir(-signal.SIGTERM)  # el SIGTERM del cgroup, a la vez
        with pytest.raises(_Salida):
            listener._graceful_exit(signal.SIGTERM, None)
        assert hilo is not None
        hilo.join(5)
    assert _avisos_del_seguidor(caplog) == []
    assert backend.fotos == fotos


@pytest.mark.parametrize("codigo", [1, -signal.SIGKILL, -signal.SIGSEGV])
def test_seguidor_que_se_cae_solo_avisa_y_sondea_enseguida(monitores, caplog, codigo):
    """Una caída de verdad (sale con error, o la mata otra señal que no es la del
    apagado) se avisa y cae al sondeo en el acto, como siempre: no espera nada."""
    backend = _BackendSeguible()
    mon = monitores(backend, poll_s=5.0)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.media_aware"):
        mon.start()
        assert _siguiendo(mon, backend)
        fotos = backend.fotos
        t0 = time.monotonic()
        backend.vivo.morir(codigo)
        assert _espera(lambda: backend.fotos > fotos and _avisos_del_seguidor(caplog))
        assert time.monotonic() - t0 < 3.0  # bastante menos que poll_s
    assert f"código {codigo}" in _avisos_del_seguidor(caplog)[0]


def test_seguidor_terminado_desde_fuera_sin_apagado_igual_cae_al_sondeo(monitores, caplog):
    """Un SIGTERM que solo recibe el seguidor (un `pkill playerctl`) no es el
    apagado: pasado el margen, aviso y sondeo de respaldo como cualquier caída."""
    backend = _BackendSeguible(playing=["spotify"])
    mon = monitores(backend, poll_s=0.2, FOLLOW_RETRY_S=0.4)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.media_aware"):
        mon.start()
        assert _siguiendo(mon, backend)
        backend.vivo.morir(-signal.SIGTERM)
        backend.playing = []
        assert _espera(lambda: not mon.is_playing())
        assert _espera(lambda: bool(_avisos_del_seguidor(caplog)))
        assert _siguiendo(mon, backend, n=2)  # y reintenta, como siempre
    assert "código -15" in _avisos_del_seguidor(caplog)[0]


_SCRIPT_APAGADO = '''\
import sys, time
sys.path.insert(0, {repo!r})
from crotolamo.logging_setup import setup_logging
setup_logging("INFO")
from crotolamo.voice.media_aware import Ducker, MediaMonitor, PlayerctlBackend
from interfaces import listener
mon = MediaMonitor([PlayerctlBackend()], poll_s=1.5)
backend = PlayerctlBackend()
ducker = Ducker([backend], factor=0.2)
# A media orden con la música atenuada: el manejador la repone (un playerctl que
# tarda lo que tarda en arrancar). El caché del monitor, sin música: ninguna foto
# periódica legítima puede caer cerca de la señal.
ducker._saved = {{"spotify": (backend, "0.800000")}}
listener._ACTIVE_MEDIA = mon
listener._ACTIVE_DUCKER = ducker
listener._install_signal_handlers()
mon.start()
for _ in range(1000):
    if mon._follower is not None:
        break
    time.sleep(0.01)
print("listo", mon._follower.pid, flush=True)
time.sleep(60)
'''


@pytest.mark.parametrize("senal", [signal.SIGTERM, signal.SIGINT])
def test_apagado_del_grupo_entero_no_deja_aviso_falso_ni_playerctl_de_mas(
        mundo, tmp_path, senal):
    """De punta a punta, como systemd (SIGTERM a todo el cgroup) o Ctrl-C (SIGINT
    a todo el grupo): la música se repone y nada más; ni WARNING de una caída
    que no hubo ni foto de playerctl con el proceso saliendo."""
    mundo.volumes = {"spotify": "0.160"}
    mundo.poner([])
    script = tmp_path / "listener_apagado.py"
    script.write_text(_SCRIPT_APAGADO.format(repo=str(REPO)), encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path))
    env.pop("CROTOLAMO_LOG", None)
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path),
                            start_new_session=True)
    try:
        linea = proc.stdout.readline()
        assert linea.startswith("listo"), proc.stderr.read()
        pid = int(linea.split()[1])
        time.sleep(0.3)
        fotos = mundo.fotos()
        os.killpg(proc.pid, senal)
        _, err = proc.communicate(timeout=20)
        assert proc.returncode == 0, err
        time.sleep(0.3)
        assert mundo.fotos() == fotos, err
        assert "playerctl --follow" not in err, err
        estado = json.loads((mundo.dir / "state.json").read_text(encoding="utf-8"))
        assert estado["volumes"]["spotify"] == "0.800000"  # la música, repuesta
        assert _espera(lambda: not _vivo(pid))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


# --- Con música, los eventos no suman fotos a las del sondeo ------------------

def test_con_musica_eventos_y_foto_periodica_no_superan_el_ritmo_del_sondeo(monitores):
    """Con música en el caché corre la foto de cada poll_s (PLAYING_REFRESH_S de
    fábrica) y cada línea no-Playing de OTRO reproductor pide foto. Contaban por
    separado: con una línea cada 1.2 x poll_s salían ~1.7 fotos por cada una del
    sondeo de antes. Ahora comparten el mismo ritmo: nunca más que el sondeo, y
    la música que para se sigue viendo como mucho poll_s después, como antes."""
    assert MediaMonitor.PLAYING_REFRESH_S is None
    backend = _BackendSeguible(playing=["spotify"])
    mon = monitores(backend, poll_s=0.2, SAFETY_REFRESH_S=60.0)
    mon.start()
    assert _siguiendo(mon, backend)
    assert _espera(mon.is_playing)
    fotos, t0 = backend.fotos, time.monotonic()
    for i in range(15):
        # Pausa/parado alternos: playerctl no suprime líneas distintas.
        backend.vivo.emitir("firefox\tPaused" if i % 2 == 0 else "firefox\tStopped")
        time.sleep(0.24)
    hechas, lapso = backend.fotos - fotos, time.monotonic() - t0
    assert hechas <= lapso / 0.2 + 1, (hechas, lapso)
    assert mon.is_playing() and mon.playing_players() == ["spotify"]
    backend.playing = []
    t0 = time.monotonic()
    backend.vivo.emitir("spotify\tPaused")
    assert _espera(lambda: not mon.is_playing())
    assert time.monotonic() - t0 < 1.0


def test_sin_musica_el_evento_sigue_siendo_inmediato(monitores):
    """Lo que no cambia: sin música en el caché, la línea de algo que empieza a
    sonar pide su foto en el acto, aunque la última foto sea de hace nada."""
    backend = _BackendSeguible()
    mon = monitores(backend, poll_s=5.0)
    mon.start()
    assert _siguiendo(mon, backend)
    backend.playing = ["spotify"]
    t0 = time.monotonic()
    backend.vivo.emitir("spotify\tPlaying")
    assert _espera(mon.is_playing)
    assert time.monotonic() - t0 < 2.0  # no espera poll_s (5 s)


# ===========================================================================
# Paso 3: el apagado y el canal de control despiertan; nadie pregunta por reloj
# ===========================================================================

def _reemplazo_atomico(ruta: Path, texto: str) -> None:
    """Lo que hace el panel (desktop/panel/config.py): temporal + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=ruta.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(texto)
    os.replace(tmp, ruta)


_PAUSA = '{"listening_enabled": false}'
_ACTIVA = '{"listening_enabled": true}'


# --- Wakeable ----------------------------------------------------------------
def test_wakeable_es_un_event_y_avisa_una_vez_por_activacion():
    w = Wakeable()
    assert isinstance(w, threading.Event) and w.wakes_waiters is True
    llamadas: list[str] = []

    def roto() -> None:
        raise RuntimeError("despertador roto")

    w.add_waker(lambda: llamadas.append("a"))
    w.add_waker(roto)
    w.add_waker(lambda: llamadas.append("b"))
    w.set()
    w.set()  # ya activado: nada nuevo que avisar
    assert llamadas == ["a", "b"], "un despertador roto impidió el resto, o se repitió"
    w.add_waker(lambda: llamadas.append("tarde"))  # llega ya activado: avisa en el acto
    assert llamadas == ["a", "b", "tarde"]
    assert w.wait(0) is True
    w.clear()
    w.set()
    assert llamadas == ["a", "b", "tarde", "a", "b", "tarde"]


# --- STOP en las colas ---------------------------------------------------------
class _ColaQueAnota(queue.Queue):
    """Cola que anota el timeout de cada get() bloqueante del hilo que la lee."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    def get(self, block=True, timeout=None):
        if block:
            self.timeouts.append(timeout)
        return super().get(block, timeout)


def _hilo_de_cola(clase, cola: queue.Queue, shutdown: threading.Event):
    state = SharedState()
    if clase is MouthThread:
        return MouthThread(SimpleNamespace(speak=lambda t: None), cola, state, shutdown)
    if clase is BrainThread:
        return BrainThread(SimpleNamespace(handle_turn=lambda c: "ok"), cola, queue.Queue(),
                           state, shutdown)
    return SttThread(SimpleNamespace(transcribe=lambda a, hotwords=None: ""), cola,
                     queue.Queue(), state, shutdown)


@pytest.mark.parametrize("clase", [MouthThread, BrainThread, SttThread])
def test_hilos_con_wakeable_esperan_sin_reloj_y_el_apagado_los_saca(clase):
    """Con el apagado del loop (Wakeable) la cola se espera SIN timeout: en
    reposo el hilo no despierta nunca (antes, 5 veces por segundo), y set() lo
    saca al instante con el STOP que mete en la cola."""
    cola = _ColaQueAnota()
    shutdown = Wakeable()
    shutdown.add_waker(lambda: cola.put(STOP))
    hilo = _hilo_de_cola(clase, cola, shutdown)
    hilo.start()
    time.sleep(0.5)
    assert cola.timeouts == [None], f"despertó por reloj en reposo: {cola.timeouts}"
    inicio = time.monotonic()
    shutdown.set()
    hilo.join(timeout=2.0)
    assert not hilo.is_alive()
    assert time.monotonic() - inicio < 1.0


@pytest.mark.parametrize("clase", [MouthThread, BrainThread, SttThread])
def test_hilos_con_event_normal_conservan_el_sondeo(clase):
    """Hilos sueltos con un threading.Event (los tests de siempre): el sondeo
    de 0.2 s sigue ahí, porque nadie mete un STOP en su cola."""
    cola = _ColaQueAnota()
    shutdown = threading.Event()
    hilo = _hilo_de_cola(clase, cola, shutdown)
    hilo.start()
    assert _espera(lambda: len(cola.timeouts) >= 2)
    shutdown.set()
    hilo.join(timeout=2.0)
    assert not hilo.is_alive()
    assert set(cola.timeouts) == {0.2}


def test_el_apagado_del_loop_mete_stop_en_cada_cola():
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"),
        SimpleNamespace(transcribe=lambda a, hotwords=None: ""),
        SimpleNamespace(speak=lambda t: None, stop=lambda: None), None,
        mic=SimpleNamespace(read=lambda: None), wake_fn=lambda c: False,
        vad_fn=lambda c: 0.0, to_wav=lambda f: "WAV",
    )
    assert isinstance(vl.shutdown, threading.Event)  # contrato de siempre
    vl.shutdown.set()
    for cola in (vl.stt_q, vl.cmd_q, vl.tts_q):
        assert cola.get_nowait() is STOP
        assert cola.empty()


def test_drain_queue_conserva_el_stop():
    """Un barge-in que coincide con el apagado tira las frases, no el STOP: si
    no, la Boca, que espera sin timeout, se quedaría bloqueada en la cola."""
    cola: queue.Queue = queue.Queue()
    cola.put(Utterance("vieja", 1))
    cola.put(STOP)
    cola.put(Utterance("otra", 1))
    _drain_queue(cola)
    assert cola.get_nowait() is STOP
    assert cola.empty()
    _drain_queue(cola)  # sin STOP: vacía como siempre
    assert cola.empty()


# --- DirWatch ------------------------------------------------------------------
@pytest.fixture
def vigia(tmp_path):
    creados: list[DirWatch] = []

    def crear(directorio: Path = tmp_path) -> DirWatch:
        w = DirWatch.create(directorio)
        assert w is not None, "inotify no disponible en esta máquina"
        creados.append(w)
        return w

    yield crear
    for w in creados:
        w.close()


def test_dirwatch_ve_el_reemplazo_atomico_y_no_otros_nombres(vigia, tmp_path):
    w = vigia()
    control = tmp_path / "control.json"
    _reemplazo_atomico(tmp_path / "hud_state.json", '{"mode": "idle"}')  # otro archivo
    assert w.wait({"control.json"}, 0.2) is False
    _reemplazo_atomico(control, _PAUSA)
    assert w.wait({"control.json"}, 2.0) is True
    control.write_text(_ACTIVA, encoding="utf-8")  # escritor no atómico (un editor)
    assert w.wait({"control.json"}, 2.0) is True
    control.unlink()
    assert w.wait({"control.json"}, 2.0) is True
    assert not w.broken


def test_dirwatch_despierta_desde_otro_hilo_y_wake_tras_close_no_toca_nada(vigia):
    w = vigia()
    threading.Timer(0.1, w.wake).start()
    inicio = time.monotonic()
    assert w.wait({"control.json"}, 5.0) is None
    assert time.monotonic() - inicio < 2.0
    assert w.wait({"control.json"}, 0.05) is False  # el despertar se consumió
    w.wake()
    w.wake()  # varios despertares pendientes: una sola vuelta
    assert w.wait({"control.json"}, 1.0) is None
    assert w.wait({"control.json"}, 0.05) is False
    w.close()
    w.close()
    w.wake()  # tras close no escribe en un descriptor que ya puede ser de otro
    assert w.wait({"control.json"}, 0.05) is None


def test_dirwatch_directorio_borrado_queda_roto(vigia, tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    w = vigia(d)
    d.rmdir()
    assert w.wait({"control.json"}, 2.0) is None
    assert w.broken


def test_dirwatch_sin_inotify_o_sin_directorio_no_hay_vigia(tmp_path, monkeypatch):
    assert DirWatch.create(tmp_path / "no-existe") is None
    with pytest.raises(OSError):
        DirWatch.open(tmp_path / "no-existe")

    def sin_inotify():
        raise AttributeError("inotify_init1")

    monkeypatch.setattr(notify, "_libc_inotify", sin_inotify)
    assert DirWatch.create(tmp_path) is None


def test_dirwatch_no_deja_descriptores_abiertos(tmp_path):
    antes = len(os.listdir("/proc/self/fd"))
    for _ in range(5):
        w = DirWatch.create(tmp_path)
        assert w is not None
        w.wake()
        w.close()
    assert DirWatch.create(tmp_path / "no-existe") is None
    assert len(os.listdir("/proc/self/fd")) == antes


# --- VoiceLoop: control.json por inotify, hilos que avisan al terminar ------------
class _MicQuieto:
    """Micro sin audio: read() bloquea un frame, como PortAudio, y no entrega
    nada (el oído no cuenta: lo que se mide es el resto del loop)."""

    def __init__(self) -> None:
        self._cerrado = threading.Event()

    def read(self):
        self._cerrado.wait(0.05)
        return None

    def close(self) -> None:
        self._cerrado.set()


class _MicMuerto:
    def read(self):
        raise OSError("PortAudioError: Device unavailable [PaErrorCode -9985]")


def _loop_de_control(control_path, mic=None, publicado=None) -> VoiceLoop:
    return VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"),
        SimpleNamespace(transcribe=lambda a, hotwords=None: ""),
        SimpleNamespace(speak=lambda t: None, stop=lambda: None), None,
        mic=mic or _MicQuieto(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda f: "WAV", control_path=control_path,
        hud_publisher=None if publicado is None else publicado.append,
    )


@pytest.fixture
def correr_loop():
    """Corre vl.run() en un hilo; al final apaga y comprueba que no quede nada."""
    corriendo: list[tuple[VoiceLoop, threading.Thread, list]] = []

    def correr(vl: VoiceLoop):
        resultado: list = []
        hilo = threading.Thread(target=lambda: resultado.append(vl.run()), name="Main",
                                daemon=True)
        hilo.start()
        corriendo.append((vl, hilo, resultado))
        return hilo, resultado

    yield correr
    for vl, hilo, _ in corriendo:
        vl.shutdown.set()
        hilo.join(timeout=5.0)
        assert not hilo.is_alive()
        assert all(not t.is_alive() for t in vl.threads)


def _contar_stats(monkeypatch, ruta: Path) -> list[float]:
    stats: list[float] = []
    real = os.stat

    def contando(p, *a, **k):
        if isinstance(p, (str, os.PathLike)) and os.fspath(p) == os.fspath(ruta):
            stats.append(time.monotonic())
        return real(p, *a, **k)

    monkeypatch.setattr(os, "stat", contando)
    return stats


def test_loop_aplica_la_pausa_al_instante_y_en_reposo_no_mira_el_archivo(
        tmp_path, monkeypatch, correr_loop):
    """Antes: stat() de control.json cada 0.3 s para siempre y la pausa del
    panel se veía hasta 0.3 s tarde. Ahora: una lectura al arrancar, otra por
    cambio, y la pausa se aplica en cuanto el panel reemplaza el archivo."""
    control = tmp_path / "control.json"
    _reemplazo_atomico(control, _ACTIVA)
    stats = _contar_stats(monkeypatch, control)
    publicado: list[dict] = []
    vl = _loop_de_control(control, publicado=publicado)
    hilo, _ = correr_loop(vl)
    assert _espera(lambda: bool(publicado))
    time.sleep(1.0)
    assert len(stats) == 1, f"miró control.json en reposo: {len(stats)} stat()"
    inicio = time.monotonic()
    _reemplazo_atomico(control, _PAUSA)
    assert _espera(lambda: not vl.state.is_enabled())
    assert time.monotonic() - inicio < 1.0
    assert publicado[-1]["enabled"] is False
    assert len(stats) == 2
    inicio = time.monotonic()
    vl.shutdown.set()
    hilo.join(timeout=5.0)
    assert not hilo.is_alive()
    assert all(not t.is_alive() for t in vl.threads)
    assert time.monotonic() - inicio < 3.0


def test_loop_con_canal_de_control_ve_al_instante_un_hilo_muerto(tmp_path, correr_loop):
    """El vigía de los hilos (V2) no se queda esperando el re-chequeo de 30 s:
    el hilo que termina despierta al bucle principal."""
    vl = _loop_de_control(tmp_path / "control.json", mic=_MicMuerto())
    vl._ear._PAUSA_TRAS_FALLO_S = 0.01
    inicio = time.monotonic()
    hilo, resultado = correr_loop(vl)
    hilo.join(timeout=10.0)
    assert resultado == [1]
    assert time.monotonic() - inicio < 3.0, "el hilo muerto se vio por reloj, no al terminar"
    assert vl.shutdown.is_set()


def test_loop_sin_canal_de_control_ve_al_instante_un_hilo_muerto(correr_loop):
    vl = _loop_de_control(None, mic=_MicMuerto())
    vl._ear._PAUSA_TRAS_FALLO_S = 0.01
    hilo, resultado = correr_loop(vl)
    hilo.join(timeout=10.0)
    assert resultado == [1]


def test_loop_sin_inotify_vuelve_al_sondeo_con_un_solo_aviso(
        tmp_path, monkeypatch, caplog, correr_loop):
    def sin_inotify(directorio):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(DirWatch, "open", staticmethod(sin_inotify))
    control = tmp_path / "control.json"
    stats = _contar_stats(monkeypatch, control)
    vl = _loop_de_control(control)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.loop"):
        correr_loop(vl)
        _reemplazo_atomico(control, _PAUSA)
        assert _espera(lambda: not vl.state.is_enabled())
        time.sleep(0.7)
    assert len(stats) >= 3, "sin inotify tenía que seguir el sondeo de siempre"
    avisos = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1, [r.getMessage() for r in avisos]
    assert "inotify" in avisos[0].getMessage()


def test_loop_con_vigia_roto_vuelve_al_sondeo_con_un_solo_aviso(
        tmp_path, monkeypatch, caplog, correr_loop):
    roto = threading.Event()

    def se_rompe(self, names, timeout):
        self.broken = True  # p. ej. el directorio se borró o se movió
        roto.set()
        return None

    control = tmp_path / "control.json"
    vl = _loop_de_control(control)
    monkeypatch.setattr(DirWatch, "wait", se_rompe)
    with caplog.at_level(logging.DEBUG, logger="crotolamo.voice.loop"):
        correr_loop(vl)
        assert roto.wait(5.0)
        _reemplazo_atomico(control, _PAUSA)
        assert _espera(lambda: not vl.state.is_enabled())
        time.sleep(0.4)
    avisos = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1, [r.getMessage() for r in avisos]


# --- Las señales despiertan al hilo principal --------------------------------
# CPython corre los manejadores de Python solo en el hilo principal, y solo
# cuando ejecuta bytecode o recibe EINTR. El kernel entrega la señal del proceso
# a OTRO hilo si el principal ya tiene una pendiente (el SIGCHLD de un playerctl,
# por ejemplo). Con el principal dormido en el select del vigía (hasta 30 s) el
# SIGTERM de systemd esperaba hasta 30 s (antes: a lo sumo 0.3 s) y, con
# TimeoutStopSec=10, llegaba el SIGKILL sin apagado limpio.

_SCRIPT_LOOP_SENAL = '''\
import os, signal, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {repo!r})
from crotolamo.voice.loop import VoiceLoop
from interfaces import listener
home = Path(os.environ["HOME"])
listener._HUD_STATE_PATH = home / "hud.json"


class Mic:
    def read(self):
        time.sleep(0.032)
        return None

    def close(self):
        pass


vl = VoiceLoop(
    SimpleNamespace(handle_turn=lambda c: "ok"),
    SimpleNamespace(transcribe=lambda a, hotwords=None: ""),
    SimpleNamespace(speak=lambda t: None, stop=lambda: None), None,
    mic=Mic(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0, to_wav=lambda f: "WAV",
    control_path=home / ".crotolamo" / "control.json",
)
listener._install_signal_handlers()


def disparar():
    sys.stdin.readline()
    # Lo que pasa cuando el kernel le da el SIGTERM del proceso a otro hilo.
    oido = next(t for t in vl.threads if t.name == "Ear")
    signal.pthread_kill(oido.ident, signal.SIGTERM)


threading.Thread(target=disparar, daemon=True).start()
print("listo", flush=True)
sys.exit(vl.run())
'''


def test_sigterm_que_recibe_otro_hilo_apaga_el_listener_al_instante(tmp_path):
    script = tmp_path / "listener_loop.py"
    script.write_text(_SCRIPT_LOOP_SENAL.format(repo=str(REPO)), encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path), CROTOLAMO_LOG="CRITICAL")
    proc = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, cwd=str(tmp_path))
    try:
        assert proc.stdout.readline().startswith("listo"), proc.stderr.read()
        time.sleep(1.0)  # el hilo principal ya duerme en el select del vigía
        t0 = time.monotonic()
        proc.stdin.write("\n")
        proc.stdin.flush()
        assert proc.wait(timeout=15) == 0  # _graceful_exit: os._exit(0)
        assert time.monotonic() - t0 < 5.0
        assert "apagado" in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.fixture
def fd_de_senales(monkeypatch):
    """Anota (en orden) cada signal.set_wakeup_fd del loop y cada DirWatch.close,
    y comprueba al final que el fd de señales quedó como estaba."""
    real = signal.set_wakeup_fd
    original = real(-1)
    real(original)
    eventos: list[tuple[str, int]] = []

    def set_wakeup_fd(fd, *a, **kw):
        eventos.append(("senal", fd))
        return real(fd, *a, **kw)

    cerrar = DirWatch.close

    def close(self):
        eventos.append(("close", self.wake_fd))
        cerrar(self)

    monkeypatch.setattr(voice_loop.signal, "set_wakeup_fd", set_wakeup_fd)
    monkeypatch.setattr(DirWatch, "close", close)
    yield eventos
    actual = real(original)
    assert actual == original


def _correr_en_el_hilo_principal(vl: VoiceLoop, guion) -> int:
    assert threading.current_thread() is threading.main_thread()
    hilo = threading.Thread(target=guion, daemon=True)
    hilo.start()
    try:
        return vl.run()
    finally:
        vl.shutdown.set()
        hilo.join(5.0)


def test_loop_en_el_hilo_principal_enruta_las_senales_al_vigia_y_lo_devuelve(
        tmp_path, fd_de_senales):
    vl = _loop_de_control(tmp_path / "control.json")

    def guion():
        assert _espera(lambda: len(fd_de_senales) >= 1)
        time.sleep(0.1)
        vl.shutdown.set()

    assert _correr_en_el_hilo_principal(vl, guion) == 0
    (_, fd), (_, previo), (cierre, cerrado) = fd_de_senales
    assert fd >= 0 and previo != fd
    assert (cierre, cerrado) == ("close", fd)  # devuelto ANTES de cerrar el pipe


def test_vigia_que_se_rompe_devuelve_el_fd_de_senales_antes_de_cerrarse(
        tmp_path, fd_de_senales):
    """Si se cerrara antes, el manejador de C escribiría en un fd cerrado (o ya
    reutilizado por otra cosa: un socket, el micro...)."""
    directorio = tmp_path / "ctl"
    directorio.mkdir()
    vl = _loop_de_control(directorio / "control.json")

    def guion():
        assert _espera(lambda: len(fd_de_senales) >= 1)
        os.rmdir(directorio)  # el vigía se rompe: el loop vuelve al sondeo
        assert _espera(lambda: any(e == "close" for e, _ in fd_de_senales))
        time.sleep(0.4)  # una vuelta del sondeo de siempre
        vl.shutdown.set()

    assert _correr_en_el_hilo_principal(vl, guion) == 0
    (_, fd), (_, previo), (cierre, cerrado) = fd_de_senales  # y nada después
    assert previo != fd and (cierre, cerrado) == ("close", fd)


def test_loop_fuera_del_hilo_principal_no_toca_las_senales(tmp_path, fd_de_senales,
                                                           correr_loop):
    vl = _loop_de_control(tmp_path / "control.json")
    hilo, resultado = correr_loop(vl)
    time.sleep(0.2)
    vl.shutdown.set()
    hilo.join(5.0)
    assert resultado == [0]
    assert [e for e, _ in fd_de_senales] == ["close"]


def test_control_reescrito_en_el_mismo_tic_del_reloj_se_relee(tmp_path):
    """Con inotify el loop lee control.json justo al crearse, antes de que el
    escritor termine; si crear y escribir caen en el mismo tic del reloj del
    sistema de archivos el mtime no cambia, y comparando solo el mtime la pausa
    se perdía para siempre. La firma lleva también tamaño e inodo."""
    control = tmp_path / "control.json"
    vl = _loop_de_control(control)
    ns = time.time_ns()
    control.write_text("", encoding="utf-8")  # recién creado, aún sin escribir
    os.utime(control, ns=(ns, ns))
    vl._poll_control()
    assert vl.state.is_enabled() is True
    control.write_text(_PAUSA, encoding="utf-8")
    os.utime(control, ns=(ns, ns))  # mismo mtime: mismo tic
    vl._poll_control()
    assert vl.state.is_enabled() is False


def test_control_sin_cambios_no_se_relee(tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    control.write_text(_PAUSA, encoding="utf-8")
    lecturas: list[Path] = []
    real = voice_loop.read_control_enabled
    monkeypatch.setattr(voice_loop, "read_control_enabled",
                        lambda p: lecturas.append(Path(p)) or real(p))
    vl = _loop_de_control(control)
    for _ in range(5):
        vl._poll_control()
    assert lecturas == [control]
    assert vl.state.is_enabled() is False


def test_control_json_enlazado_se_sigue_viendo(tmp_path, correr_loop):
    """control.json como enlace (dotfiles con stow/chezmoi): el panel reemplaza
    el DESTINO, en otro directorio, e inotify sobre ~/.crotolamo no lo ve. Ahí
    el loop mira el archivo como siempre (cada 0.3 s)."""
    real_dir = tmp_path / "dotfiles"
    real_dir.mkdir()
    real = real_dir / "control.json"
    _reemplazo_atomico(real, _ACTIVA)
    home = tmp_path / "home"
    home.mkdir()
    control = home / "control.json"
    control.symlink_to(real)
    vl = _loop_de_control(control)
    correr_loop(vl)
    time.sleep(0.2)
    _reemplazo_atomico(real, _PAUSA)  # lo que hace el panel con un enlace
    assert _espera(lambda: not vl.state.is_enabled(), timeout=3.0)


def test_loop_crea_el_directorio_del_canal_si_falta(tmp_path, correr_loop):
    control = tmp_path / "nuevo" / "control.json"
    vl = _loop_de_control(control)
    correr_loop(vl)
    assert _espera(lambda: control.parent.is_dir())
    _reemplazo_atomico(control, _PAUSA)
    assert _espera(lambda: not vl.state.is_enabled(), timeout=3.0)


class _MicConWakeYa:
    """El primer chunk ya es la wake word; luego, frames sin audio."""

    def __init__(self) -> None:
        self.chunks = ["wake"]
        self._cerrado = threading.Event()

    def read(self):
        if self.chunks:
            return self.chunks.pop(0)
        self._cerrado.wait(0.05)
        return None

    def close(self) -> None:
        self._cerrado.set()


@pytest.mark.parametrize("vuelta", range(3))
def test_con_la_escucha_pausada_el_primer_chunk_no_abre_turno(tmp_path, correr_loop, vuelta):
    """Los hilos arrancaban ANTES de leer control.json: un wake en el primer
    chunk se aceptaba con la escucha pausada en el panel. Ahora se lee el
    canal y se publica el estado inicial, y recién entonces arranca el oído."""
    control = tmp_path / "control.json"
    _reemplazo_atomico(control, _PAUSA)
    publicado: list[dict] = []
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"),
        SimpleNamespace(transcribe=lambda a, hotwords=None: ""),
        SimpleNamespace(speak=lambda t: None, stop=lambda: None), None,
        mic=_MicConWakeYa(), wake_fn=lambda c: c == "wake", vad_fn=lambda c: 0.0,
        to_wav=lambda f: "WAV", control_path=control, hud_publisher=publicado.append,
    )
    correr_loop(vl)
    assert _espera(lambda: not vl._ear.mic.chunks)
    time.sleep(0.1)
    assert vl.state.turn_id == 0, "aceptó un wake con la escucha pausada"
    assert (publicado[0]["mode"], publicado[0]["enabled"]) == ("idle", False)


# ===========================================================================
# Paso 4: un wake es una publicación
# ===========================================================================

def _pubs_de(state_pubs: list[dict]) -> list[tuple]:
    return [(p["mode"], p["turn_id"], p["text"]) for p in state_pubs]


def test_new_turn_con_modo_es_una_sola_publicacion():
    publicado: list[dict] = []
    state = SharedState(publisher=publicado.append)
    state.set_text("respuesta anterior")
    publicado.clear()
    assert state.new_turn(mode=Mode.LISTENING) == 1
    assert _pubs_de(publicado) == [("listening", 1, "")]
    assert state.get_mode() is Mode.LISTENING
    assert state.new_turn() == 2  # sin modo: como siempre, publica el turno nuevo
    assert _pubs_de(publicado) == [("listening", 1, ""), ("listening", 2, "")]


class _MicDeGuion:
    """Entrega el guion y, agotado, apaga al oído (todo en el hilo del test)."""

    def __init__(self, shutdown: threading.Event, chunks) -> None:
        self.shutdown = shutdown
        self.chunks = list(chunks)

    def read(self):
        if not self.chunks:
            self.shutdown.set()
            return None
        return self.chunks.pop(0)


def _oido(state, chunks, **kw):
    shutdown = threading.Event()
    tts = SimpleNamespace(stop=lambda: None)
    return EarThread(_MicDeGuion(shutdown, chunks), wake_fn=lambda c: c == "wake",
                     vad_fn=lambda c: 0.9 if c.startswith("voz") else 0.0,
                     to_wav=lambda f: "WAV", tts=tts, stt_queue=queue.Queue(),
                     tts_queue=queue.Queue(), state=state, shutdown=shutdown, **kw)


def test_wake_del_oido_es_una_publicacion():
    """Antes: new_turn() publicaba "idle" con el turno nuevo y set_mode()
    enseguida "listening": dos escrituras de hud_state.json (y dos despertares
    del HUD) para una sola transición."""
    publicado: list[dict] = []
    state = SharedState(publisher=publicado.append)
    _oido(state, ["wake"]).run()
    assert _pubs_de(publicado) == [("listening", 1, "")]


def test_barge_in_del_oido_es_una_publicacion():
    publicado: list[dict] = []
    state = SharedState(publisher=publicado.append)
    state.new_turn()
    state.set_text("Uno. Dos.")
    state.set_mode(Mode.SPEAKING)
    publicado.clear()
    oido = _oido(state, ["voz0", "voz1"], allow_barge_in=True, barge_in_grace_ms=0,
                 barge_in_threshold_margin=0.0, barge_in_min_chunks=2)
    oido.run()
    assert _pubs_de(publicado) == [("listening", 2, "")]


def test_ducker_ve_las_mismas_transiciones_con_un_wake_de_una_publicacion():
    """El Ducker solo actúa al cruzar idle <-> no idle: el "idle" que publicaba
    el wake con el turno nuevo nunca fue una transición (ya estaba en idle)."""
    llamadas: list[str] = []
    backend = SimpleNamespace(
        playing_players=lambda: llamadas.append("list") or ["spotify"],
        get_volume=lambda p: llamadas.append("get") or "0.750000",
        set_volume=lambda p, v: llamadas.append("set") or True,
    )
    ducker = Ducker([backend], factor=0.2)
    state = SharedState(publisher=lambda d: ducker.on_mode(d["mode"]))
    _oido(state, ["wake"]).run()
    state.set_mode(Mode.IDLE)
    assert ducker.flush()
    ducker.close()
    assert llamadas == ["list", "get", "set", "set"]


def test_setters_concurrentes_dejan_publicado_el_estado_real():
    """Varios hilos cambiando modo, texto y turno a la vez: lo último publicado
    es el estado real (C8), también con new_turn(mode=...)."""
    publicado: list[dict] = []

    def publisher(d: dict) -> None:
        time.sleep(0.0002)  # suelta el GIL a media publicación, como el disco
        publicado.append(d)

    state = SharedState(publisher=publisher)
    modos = [Mode.LISTENING, Mode.THINKING, Mode.SPEAKING, Mode.IDLE]

    def martillo(i: int) -> None:
        for n in range(40):
            if n % 3 == 0:
                state.new_turn(mode=modos[(i + n) % 4])
            elif n % 3 == 1:
                state.set_mode(modos[(i + n) % 4])
            else:
                state.set_text(f"{i}-{n}")

    hilos = [threading.Thread(target=martillo, args=(i,)) for i in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    sin_ts = {k: v for k, v in state.current_snapshot().items() if k != "ts"}
    assert {k: v for k, v in publicado[-1].items() if k != "ts"} == sin_ts


def test_loop_simple_un_wake_es_una_publicacion(monkeypatch):
    from tests.test_auditoria_voice_listener import _RecStt, _run_simple, _ScriptedWake

    publicado: list[dict] = []
    rc, agent = _run_simple(monkeypatch, wake_stt=_ScriptedWake(["crotolamo"]),
                            stt=_RecStt(["qué hora es"]),
                            hud=SharedState(publisher=publicado.append))
    assert rc == 0 and agent.commands == ["qué hora es"]
    del_turno = [p for p in publicado if p["turn_id"] == 1]
    assert (del_turno[0]["mode"], del_turno[0]["text"]) == ("listening", "")
    assert not any(p["mode"] == "idle" and p["text"] == "" for p in del_turno[:1])


# ===========================================================================
# Paso 6: el panel sostiene su hilo de consulta y lo que ya pintó
# ===========================================================================

def _hilos_contados(monkeypatch, window) -> list[threading.Thread]:
    """Cuenta los hilos que arranca panel.window (los demás módulos, intactos)."""
    hilos: list[threading.Thread] = []

    class _Hilo(threading.Thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            hilos.append(self)

    monkeypatch.setattr(window, "threading",
                        SimpleNamespace(Thread=_Hilo, Event=threading.Event))
    return hilos


def _entregar(panel_gtk) -> bool:
    """Corre el main loop virtual hasta que el resultado de la consulta en vuelo
    llegue al hilo GTK (el hilo de trabajo lo entrega con idle_add)."""
    win = panel_gtk.win

    def entregado() -> bool:
        panel_gtk.reloj.avanzar(0)
        return not win._query_inflight

    return _espera(entregado)


def test_panel_un_solo_hilo_de_consulta_para_todos_los_ticks(panel_gtk, monkeypatch):
    """Antes cada tick (1.5 s) arrancaba un hilo nuevo: 40 por minuto abierto."""
    from tests.test_auditoria_desktop_py import _ACTIVO

    window, win = panel_gtk.window, panel_gtk.win
    hilos = _hilos_contados(monkeypatch, window)
    consultas: list[threading.Thread] = []

    def consulta():
        consultas.append(threading.current_thread())
        return dict(_ACTIVO)

    monkeypatch.setattr(window, "query_state", consulta)
    win._on_map()
    assert _entregar(panel_gtk)
    for _ in range(40):  # un minuto de ticks con la ventana abierta
        assert win._tick() is True
        assert _entregar(panel_gtk)
    assert len(consultas) == 41  # una consulta por refresco, como antes
    assert len(hilos) == 1
    assert set(consultas) == {hilos[0]}  # nunca en el hilo GTK
    assert win.status.get_label() == "Escuchando"
    win._on_destroy()
    hilos[0].join(5)
    assert not hilos[0].is_alive()


def test_panel_refresco_con_consulta_en_vuelo_se_descarta_como_antes(panel_gtk, monkeypatch):
    """Mismo número de consultas que con un hilo por tick: un tick que cae con
    la consulta anterior aún en vuelo no encola otra."""
    from tests.test_auditoria_desktop_py import _APAGADO

    window, win = panel_gtk.window, panel_gtk.win
    _hilos_contados(monkeypatch, window)
    soltar = threading.Event()
    empezadas = threading.Semaphore(0)
    consultas: list[int] = []

    def consulta_lenta():
        consultas.append(1)
        empezadas.release()
        soltar.wait(10)
        return dict(_APAGADO)

    monkeypatch.setattr(window, "query_state", consulta_lenta)
    win.refresh()
    assert empezadas.acquire(timeout=10)
    win.refresh()
    win._tick()
    soltar.set()
    assert _entregar(panel_gtk)
    assert win.status.get_label() == "Apagado"
    time.sleep(0.05)  # una consulta de más se vería aquí
    assert consultas == [1]
    win.refresh()  # entregada, el siguiente refresco consulta otra vez
    assert empezadas.acquire(timeout=10)
    assert _entregar(panel_gtk)
    assert consultas == [1, 1]
    win._on_destroy()


def test_panel_hilo_de_consulta_sobrevive_a_un_fallo_y_lo_dice(panel_gtk, monkeypatch, caplog):
    """Una excepción en la consulta mataba su hilo con _query_inflight en True:
    el panel no volvía a refrescar nunca. Ahora se ve como «no pude consultar
    systemd» (lo mismo que un systemctl que revienta) y el siguiente tick sigue."""
    from tests.test_auditoria_desktop_py import _ACTIVO

    window, win = panel_gtk.window, panel_gtk.win
    hilos = _hilos_contados(monkeypatch, window)
    respuestas: list = [RuntimeError("bus roto"), dict(_ACTIVO)]

    def consulta():
        r = respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(window, "query_state", consulta)
    with caplog.at_level(logging.WARNING, logger="crotolamo.desktop.panel"):
        win.refresh()
        assert _entregar(panel_gtk)
    assert "bus roto" in caplog.text
    assert win.status.get_label() == "Desconocido"
    assert "bus roto" in win.sub.get_label()
    win._tick()
    assert _entregar(panel_gtk)
    assert win.status.get_label() == "Escuchando"
    assert len(hilos) == 1 and hilos[0].is_alive()
    win._on_destroy()
    hilos[0].join(5)
    assert not hilos[0].is_alive()


def test_panel_destruido_para_su_hilo_y_no_entrega_nada(panel_gtk, monkeypatch):
    """El dueño del hilo es la ventana: al destruirse lo despierta y el hilo
    termina; una consulta que acaba después ya no toca widgets destruidos."""
    from tests.test_auditoria_desktop_py import _ACTIVO

    window, win = panel_gtk.window, panel_gtk.win
    hilos = _hilos_contados(monkeypatch, window)
    soltar = threading.Event()
    empezada = threading.Event()

    def consulta_lenta():
        empezada.set()
        soltar.wait(10)
        return dict(_ACTIVO)

    monkeypatch.setattr(window, "query_state", consulta_lenta)
    entregas: list = []
    monkeypatch.setattr(win, "_on_state_ready", lambda st: entregas.append(st))
    win.refresh()
    assert empezada.wait(10)
    win._on_destroy()
    soltar.set()
    hilos[0].join(5)
    assert not hilos[0].is_alive()
    panel_gtk.reloj.avanzar(0)
    assert entregas == []
    # destruida sin haber consultado nunca: no hay hilo que parar
    otra = window.Panel(app=None)
    otra._on_destroy()
    assert len(hilos) == 1


def test_panel_destroy_conectado_a_parar_el_hilo(panel_gtk, monkeypatch):
    """La señal destroy de la ventana es la que para el hilo de consulta."""
    conectadas: list = []
    monkeypatch.setattr(_Contado, "connect",
                        lambda self, senal, fn, *a: conectadas.append((self, senal, fn)),
                        raising=False)
    win = panel_gtk.window.Panel(app=None)
    assert (win, "destroy", win._on_destroy) in conectadas


def _contar_escrituras(win) -> collections.Counter:
    """Envuelve set_active/set_sensitive de switches y botón grande para contar
    las escrituras (el doble de test_auditoria_desktop_py no las cuenta)."""
    cuenta: collections.Counter = collections.Counter()
    for nombre in ("autostart", "listen_sw", "big"):
        widget = getattr(win, nombre)
        for metodo in ("set_active", "set_sensitive"):
            real = getattr(widget, metodo)

            def contar(valor, _real=real, _clave=f"{nombre}.{metodo}"):
                cuenta[_clave] += 1
                _real(valor)

            setattr(widget, metodo, contar)
    return cuenta


def test_panel_tick_sin_cambios_no_escribe_ningun_widget(panel_gtk):
    """Lo que quedaba de cada tick: los dos switches y las dos sensibilidades
    se reescribían con el mismo valor (y un set_active corta la animación del
    switch que el patrón acaba de mover)."""
    from tests.test_auditoria_desktop_py import _ACTIVO, _APAGADO

    win = panel_gtk.win
    win._on_state_ready(dict(_ACTIVO))
    cuenta = _contar_escrituras(win)
    for _ in range(20):
        win._on_state_ready(dict(_ACTIVO))
    assert cuenta == collections.Counter()
    # lo que cambia se sigue escribiendo, una vez
    win._on_state_ready(dict(_APAGADO))
    assert cuenta == collections.Counter({"autostart.set_active": 1,
                                          "listen_sw.set_sensitive": 1})
    assert win.autostart.get_active() is False
    assert win.listen_sw.get_sensitive() is False
    assert win.big.get_sensitive() is True


def test_panel_misma_pintura_que_reescribiendo_siempre(panel_gtk):
    """Viejo contra nuevo: el panel de antes reescribía cada switch y cada
    sensibilidad en cada tick. Para cualquier secuencia de estados, y también
    cuando algo cambió los widgets por fuera (un clic: el switch movido, el
    botón grande apagado por on_toggle), los dos terminan pintando lo mismo."""
    from tests.test_auditoria_desktop_py import _ACTIVO, _APAGADO

    from panel import systemd as sd

    nuevo = panel_gtk.win
    viejo = panel_gtk.window.Panel(app=None)
    viejo._set_switch = lambda sw, valor: sw.set_active(valor)
    viejo._set_sensitive = lambda widget, valor: widget.set_sensitive(valor)
    estados = [dict(_ACTIVO), dict(_ACTIVO), dict(_APAGADO),
               {sd.QUERY_ERROR: "sin bus"}, dict(_APAGADO, UnitFileState="enabled"),
               {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead"},
               dict(_ACTIVO, ActiveState="activating"), dict(_ACTIVO, ActiveState="failed"),
               {sd.QUERY_ERROR: "sin bus"}, dict(_ACTIVO)]

    def foto(win):
        return (win.autostart.get_active(), win.listen_sw.get_active(),
                win.listen_sw.get_sensitive(), win.big.get_sensitive(),
                win.status.get_label(), win.sub.get_label(), win.big.get_label())

    def tocado_por_fuera(win):
        for sw in (win.autostart, win.listen_sw):
            sw.set_active(not sw.get_active())
        for widget in (win.listen_sw, win.big):
            widget.set_sensitive(not widget.get_sensitive())

    for st in estados:
        for tocar in (False, True):
            for win in (nuevo, viejo):
                if tocar:
                    tocado_por_fuera(win)
                win._on_state_ready(dict(st))
            assert foto(nuevo) == foto(viejo), (st, tocar)
