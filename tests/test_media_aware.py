"""Tests del wake consciente de la música (crotolamo/voice/media_aware.py).

Todo con fakes: ni subprocess ni playerctl ni audio real. Cubre:
- MediaMonitor: umbral normal sin música / alto con música, backends rotos o
  con basura, la voz propia no cuenta, log INFO solo al cambiar de modo.
- PlayerctlBackend: parseo de `playerctl -a metadata`, playerctl ausente,
  timeout y returncode != 0, con run_cmd y shutil.which parcheados.
- Ducker: baja al factor y restaura el string EXACTO, restaura aunque la
  interacción lance, no pisa el original, fallos con WARNING, enabled=False y
  el worker de on_mode del loop concurrente.
- WakeWordDetector y VoiceLoop con monitor/ducker enganchados.
- _run_simple_loop con fakes: umbral alto con música y restore aunque el
  agente reviente / la orden venga vacía.
- ListenerConfig.from_settings lee los knobs nuevos con sus defaults.
"""

from __future__ import annotations

import logging
import subprocess
import time
import tomllib
from types import SimpleNamespace

import pytest

from crotolamo.voice import media_aware
from crotolamo.voice.media_aware import Ducker, MediaMonitor, PlayerctlBackend, is_own_voice

_LOGGER = "crotolamo.voice.media_aware"


# --- fakes ------------------------------------------------------------------
class FakeBackend:
    """Backend en memoria: reproductores en Playing y sus volúmenes crudos."""

    def __init__(self, playing=(), volumes=None, *, fail_set=False):
        self.playing = list(playing)
        self.volumes = dict(volumes or {})
        self.fail_set = fail_set
        self.list_calls = 0
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str]] = []

    def playing_players(self):
        self.list_calls += 1
        return list(self.playing)

    def get_volume(self, player):
        self.get_calls.append(player)
        return self.volumes.get(player)

    def set_volume(self, player, value):
        self.set_calls.append((player, value))
        if self.fail_set:
            return False
        self.volumes[player] = value
        return True


class BrokenBackend:
    """Backend que revienta en todo (bus D-Bus caído)."""

    def playing_players(self):
        raise RuntimeError("dbus caído")

    def get_volume(self, player):
        raise RuntimeError("dbus caído")

    def set_volume(self, player, value):
        raise RuntimeError("dbus caído")


class GarbageBackend:
    """Backend que devuelve cualquier cosa menos una lista de nombres."""

    def __init__(self, value):
        self.value = value

    def playing_players(self):
        return self.value

    def get_volume(self, player):
        return None

    def set_volume(self, player, value):
        return False


def _monitor(*backends, normal=0.72, media=0.85) -> MediaMonitor:
    return MediaMonitor(list(backends), poll_s=0.05,
                        threshold_normal=normal, threshold_media=media)


def _ducker(backend=None, **kw) -> tuple[Ducker, FakeBackend]:
    backend = backend or FakeBackend(playing=["spotify"], volumes={"spotify": "0.750000"})
    return Ducker([backend], factor=0.2, **kw), backend


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# --- MediaMonitor ------------------------------------------------------------
def test_umbral_normal_sin_musica_y_alto_con_musica():
    backend = FakeBackend()
    mon = _monitor(backend)
    assert mon.refresh_now() is False
    assert mon.is_playing() is False
    assert mon.threshold() == pytest.approx(0.72)

    backend.playing = ["spotify"]
    assert mon.refresh_now() is True
    assert mon.is_playing() is True
    assert mon.threshold() == pytest.approx(0.85)
    assert mon.playing_players() == ["spotify"]

    backend.playing = []
    mon.refresh_now()
    assert mon.threshold() == pytest.approx(0.72)


def test_las_consultas_leen_el_cache_sin_tocar_el_backend():
    """is_playing/threshold/playing_players jamás sondean: el oído no espera."""
    backend = FakeBackend(playing=["spotify"])
    mon = _monitor(backend)
    for _ in range(50):
        mon.is_playing()
        mon.threshold()
        mon.playing_players()
    assert backend.list_calls == 0


@pytest.mark.parametrize("backend", [
    BrokenBackend(),
    GarbageBackend(None),
    GarbageBackend("spotify"),   # un str es iterable: NO debe dar "s","p","o"...
    GarbageBackend(42),
    GarbageBackend([None, "", 3, "   "]),
])
def test_backend_roto_o_con_basura_cuenta_como_silencio(backend):
    mon = _monitor(backend)
    assert mon.refresh_now() is False  # y no lanza
    assert mon.threshold() == pytest.approx(0.72)
    assert mon.playing_players() == []


def test_backend_roto_no_tapa_a_uno_sano():
    mon = _monitor(BrokenBackend(), FakeBackend(playing=["vlc"]))
    assert mon.refresh_now() is True
    assert mon.playing_players() == ["vlc"]


@pytest.mark.parametrize("name", ["crotolamo", "Crotolamo TTS", "org.crotolamo.voz"])
def test_la_voz_propia_no_cuenta_como_musica(name):
    assert is_own_voice(name)
    mon = _monitor(FakeBackend(playing=[name]))
    assert mon.refresh_now() is False
    assert mon.threshold() == pytest.approx(0.72)
    # Mezclada con música de verdad, se descarta SOLO la voz propia.
    mon2 = _monitor(FakeBackend(playing=[name, "spotify"]))
    assert mon2.refresh_now() is True
    assert mon2.playing_players() == ["spotify"]


def test_log_info_solo_al_cambiar_de_modo(caplog):
    backend = FakeBackend(playing=["spotify"])
    mon = _monitor(backend)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(3):
            mon.refresh_now()
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1, infos
        assert "modo música" in infos[0] and "0.85" in infos[0] and "spotify" in infos[0]

        backend.playing = []
        for _ in range(3):
            mon.refresh_now()
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 2, infos
        assert "modo normal" in infos[1] and "0.72" in infos[1]


def test_el_hilo_de_sondeo_sobrevive_a_un_backend_que_lanza():
    backend = FakeBackend(playing=["spotify"])
    mon = _monitor(BrokenBackend(), backend)
    mon.stop()  # sin start: no-op seguro
    mon.start()
    try:
        assert _wait(mon.is_playing)
        thread = mon._thread
        assert thread is not None and thread.is_alive()
        backend.playing = []
        assert _wait(lambda: not mon.is_playing())
    finally:
        mon.stop()
    assert not thread.is_alive()


# --- PlayerctlBackend (run_cmd y shutil.which parcheados) --------------------
def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["playerctl"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


@pytest.fixture
def playerctl(monkeypatch):
    """Parchea shutil.which y run_cmd tal como los ve media_aware; registra llamadas."""
    calls: list[tuple[list[str], float]] = []
    state: dict = {"which": "/usr/bin/playerctl", "result": _completed(), "raise": None}

    def fake_run_cmd(args, timeout=10):
        calls.append((list(args), timeout))
        if state["raise"] is not None:
            raise state["raise"]
        return state["result"]

    monkeypatch.setattr(media_aware.shutil, "which", lambda name: state["which"])
    monkeypatch.setattr(media_aware, "run_cmd", fake_run_cmd)
    return SimpleNamespace(calls=calls, state=state)


def test_playerctl_parsea_los_reproductores_en_playing(playerctl):
    playerctl.state["result"] = _completed("spotify\tPlaying\nfirefox\tPaused\nvlc\tStopped\n")
    assert PlayerctlBackend().playing_players() == ["spotify"]
    args, timeout = playerctl.calls[0]
    assert args == ["playerctl", "-a", "metadata", "--format", "{{playerName}}\t{{status}}"]
    assert timeout == 2.0


def test_playerctl_playing_insensible_a_mayusculas_tolerante_y_sin_duplicados(playerctl):
    playerctl.state["result"] = _completed(
        "firefox\tPLAYING\nfirefox\tplaying\nlinea rara sin tab\n\tPlaying\nspotify\t\n"
    )
    assert PlayerctlBackend().playing_players() == ["firefox"]


def test_playerctl_excluye_la_voz_propia(playerctl):
    playerctl.state["result"] = _completed("Crotolamo TTS\tPlaying\nspotify\tPlaying\n")
    assert PlayerctlBackend().playing_players() == ["spotify"]
    playerctl.state["result"] = _completed("crotolamo\tPlaying\n")
    assert PlayerctlBackend().playing_players() == []


def test_playerctl_ausente_devuelve_vacio_sin_ejecutar_nada(playerctl):
    playerctl.state["which"] = None
    backend = PlayerctlBackend()
    assert backend.playing_players() == []
    assert backend.get_volume("spotify") is None
    assert backend.set_volume("spotify", "0.5") is False
    assert playerctl.calls == []


def test_playerctl_timeout_devuelve_vacio(playerctl):
    playerctl.state["raise"] = subprocess.TimeoutExpired(cmd="playerctl", timeout=2)
    backend = PlayerctlBackend()
    assert backend.playing_players() == []
    assert backend.get_volume("spotify") is None
    assert backend.set_volume("spotify", "0.5") is False


def test_playerctl_oserror_devuelve_vacio(playerctl):
    playerctl.state["raise"] = OSError("no se pudo ejecutar")
    assert PlayerctlBackend().playing_players() == []


def test_playerctl_returncode_distinto_de_cero(playerctl):
    playerctl.state["result"] = _completed("", returncode=1, stderr="No players found")
    backend = PlayerctlBackend()
    assert backend.playing_players() == []
    assert backend.get_volume("spotify") is None
    assert backend.set_volume("spotify", "0.5") is False


def test_playerctl_volumen_crudo_y_set(playerctl):
    playerctl.state["result"] = _completed("0.750000\n")
    backend = PlayerctlBackend()
    assert backend.get_volume("spotify") == "0.750000"  # tal cual: sin pasar por float
    assert playerctl.calls[-1][0] == ["playerctl", "-p", "spotify", "volume"]
    assert backend.set_volume("spotify", "0.150") is True
    assert playerctl.calls[-1][0] == ["playerctl", "-p", "spotify", "volume", "0.150"]
    playerctl.state["result"] = _completed("\n")
    assert backend.get_volume("spotify") is None  # salida vacía = no sé el volumen


def test_playerctl_cumple_el_protocol():
    assert isinstance(PlayerctlBackend(), media_aware.MediaBackend)
    assert isinstance(FakeBackend(), media_aware.MediaBackend)


# --- Ducker -------------------------------------------------------------------
def test_ducker_baja_al_factor_y_restaura_el_string_exacto():
    ducker, backend = _ducker()
    ducker.duck()
    assert ducker.is_ducked()
    assert backend.set_calls == [("spotify", "0.150")]
    ducker.restore()
    assert not ducker.is_ducked()
    assert backend.set_calls[-1] == ("spotify", "0.750000")  # el string original, exacto
    assert backend.volumes["spotify"] == "0.750000"


def test_ducker_varios_reproductores():
    backend = FakeBackend(playing=["spotify", "firefox"],
                          volumes={"spotify": "1.000000", "firefox": "0.5"})
    ducker = Ducker([backend], factor=0.5)
    ducker.duck()
    assert backend.volumes == {"spotify": "0.500", "firefox": "0.250"}
    ducker.restore()
    assert backend.volumes == {"spotify": "1.000000", "firefox": "0.5"}


def test_ducked_restaura_aunque_la_interaccion_lance():
    ducker, backend = _ducker()
    with pytest.raises(RuntimeError):
        with ducker.ducked():
            assert backend.volumes["spotify"] == "0.150"
            raise RuntimeError("el LLM reventó")
    assert backend.volumes["spotify"] == "0.750000"
    assert not ducker.is_ducked()


def test_duck_dos_veces_no_pisa_el_original():
    ducker, backend = _ducker()
    ducker.duck()
    ducker.duck()  # NO-OP: ni relee (ya vale 0.150) ni reescribe
    assert backend.get_calls == ["spotify"]
    assert backend.set_calls == [("spotify", "0.150")]
    ducker.restore()
    assert backend.volumes["spotify"] == "0.750000"


def test_restore_es_idempotente():
    ducker, backend = _ducker()
    ducker.restore()  # sin duck previo: nada que hacer
    assert backend.set_calls == []
    ducker.duck()
    ducker.restore()
    ducker.restore()
    assert backend.set_calls == [("spotify", "0.150"), ("spotify", "0.750000")]


def test_fallo_de_set_volume_avisa_y_no_propaga(caplog):
    backend = FakeBackend(playing=["spotify"], volumes={"spotify": "0.75"}, fail_set=True)
    ducker = Ducker([backend])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        ducker.duck()
        ducker.restore()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "spotify" in warnings[0]
    assert not ducker.is_ducked()  # el estado queda limpio pase lo que pase


def test_lectura_fallida_o_volumen_raro_se_saltan_solo_ese_reproductor(caplog):
    backend = FakeBackend(playing=["spotify", "vlc", "firefox"],
                          volumes={"spotify": "abc", "firefox": "0.5"})  # vlc: sin volumen
    ducker = Ducker([backend], factor=0.2)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        ducker.duck()
    assert backend.set_calls == [("firefox", "0.100")]  # solo el sano
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
    ducker.restore()
    assert backend.volumes["firefox"] == "0.5"


def test_backend_que_lanza_en_duck_no_propaga(caplog):
    ducker = Ducker([BrokenBackend()])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        ducker.duck()
        ducker.restore()
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not ducker.is_ducked()


def test_enabled_false_cero_llamadas():
    ducker, backend = _ducker(enabled=False)
    ducker.duck()
    ducker.restore()
    with ducker.ducked():
        pass
    ducker.on_mode("listening")
    ducker.on_mode("idle")
    assert ducker.flush()
    ducker.close()
    assert backend.list_calls == 0 and backend.get_calls == [] and backend.set_calls == []


def test_on_mode_pasa_por_el_worker_y_conserva_el_orden():
    ducker, backend = _ducker()
    try:
        ducker.on_mode("listening")
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.150"
        ducker.on_mode("thinking")
        ducker.on_mode("speaking")
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.150"
        assert backend.set_calls == [("spotify", "0.150")]  # sigue en duck, sin re-duck
        ducker.on_mode("idle")
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.750000"
        assert not ducker.is_ducked()
    finally:
        ducker.close()


def test_on_mode_no_bloquea_al_hilo_que_publica():
    """El oído publica el modo; un backend lento no debe frenarlo (va a la cola)."""

    class SlowBackend(FakeBackend):
        def get_volume(self, player):
            time.sleep(0.3)
            return super().get_volume(player)

    backend = SlowBackend(playing=["spotify"], volumes={"spotify": "1.0"})
    ducker = Ducker([backend])
    try:
        t0 = time.monotonic()
        ducker.on_mode("listening")
        assert time.monotonic() - t0 < 0.5  # solo encola; el backend lento tarda más
        assert ducker.flush(timeout_s=3.0)
        assert backend.volumes["spotify"] == "0.200"
    finally:
        ducker.close()


def test_close_procesa_lo_encolado_y_luego_ignora():
    ducker, backend = _ducker()
    ducker.on_mode("listening")
    ducker.on_mode("idle")
    ducker.close()  # procesa duck y restore, en orden, antes de parar
    assert backend.set_calls == [("spotify", "0.150"), ("spotify", "0.750000")]
    ducker.on_mode("listening")  # tras close: ignorado
    assert ducker.flush()
    assert backend.set_calls[-1] == ("spotify", "0.750000")
    ducker.close()  # idempotente


def test_emergency_restore_restaura_aunque_el_lock_este_tomado():
    ducker, backend = _ducker()
    ducker.duck()
    ducker._lock.acquire()  # simula el hilo principal a medio duck al llegar SIGINT
    try:
        ducker.emergency_restore(lock_timeout_s=0.05)
    finally:
        ducker._lock.release()
    assert backend.volumes["spotify"] == "0.750000"
    assert not ducker.is_ducked()


# --- WakeWordDetector con monitor ---------------------------------------------
class _FakeOwwModel:
    def __init__(self, value):
        self.value = value

    def predict(self, arr):
        return {"crotolamo": self.value}


def test_wakeword_usa_el_umbral_de_musica_con_el_monitor(caplog):
    np = pytest.importorskip("numpy")
    from crotolamo.voice.wakeword import WakeWordDetector

    backend = FakeBackend()
    mon = _monitor(backend, normal=0.5, media=0.7)
    det = WakeWordDetector(model_name="fake", threshold=0.5, threshold_media=0.7)
    det._model = _FakeOwwModel(0.6)
    det.attach_media(mon)
    chunk = np.zeros(1280, dtype=np.int16)

    assert det.feed(chunk) is True  # sin música: 0.6 >= 0.5
    backend.playing = ["spotify"]
    mon.refresh_now()
    with caplog.at_level(logging.INFO, logger="crotolamo.voice.wakeword"):
        assert det.feed(chunk) is False  # con música: 0.6 < 0.7
    assert any("umbral=0.70" in r.getMessage() for r in caplog.records)  # umbral EFECTIVO
    det.attach_media(None)
    assert det.feed(chunk) is True  # sin monitor: idéntico a siempre


def test_wakeword_from_settings_lee_oww_threshold_media():
    from crotolamo.voice.wakeword import WakeWordDetector

    class _S:
        wake = {"oww_threshold": 0.5, "oww_threshold_media": 0.66}

    assert WakeWordDetector.from_settings(_S()).threshold_media == pytest.approx(0.66)

    class _S2:
        wake = {}

    assert WakeWordDetector.from_settings(_S2()).threshold_media == pytest.approx(0.7)


# --- VoiceLoop con ducker (fakes, sin audio) ----------------------------------
class _FakeCmdStt:
    def __init__(self, text="abre youtube"):
        self.text = text

    def listen_once(self, **kwargs):
        return self.text

    def listen_smart(self, **kwargs):
        return self.text

    def transcribe(self, audio, hotwords=None):
        return self.text


class _FakeTts:
    def __init__(self):
        self.spoken: list[str] = []
        self.beeps = 0

    def speak(self, text):
        self.spoken.append(text)

    def speak_sentences(self, text):
        self.spoken.append(text)

    def beep(self):
        self.beeps += 1

    def stop(self):
        pass


class _FakeMic:
    def read(self):
        return None


def test_voice_loop_engancha_el_ducker_al_publisher_y_lo_cierra_en_stop():
    from crotolamo.voice.loop import VoiceLoop
    from crotolamo.voice.state import Mode

    ducker, backend = _ducker()
    received: list[dict] = []
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _FakeCmdStt(), _FakeTts(), None,
        mic=_FakeMic(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda f: "WAV", hud_publisher=received.append, ducker=ducker,
    )
    vl.start()
    try:
        vl.state.set_mode(Mode.LISTENING)
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.150"
        assert received[-1]["mode"] == "listening"  # el HUD sigue recibiendo el dict
        vl.state.set_mode(Mode.IDLE)
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.750000"
        # Apagado a media respuesta: stop() restaura y cierra el worker.
        vl.state.set_mode(Mode.SPEAKING)
        assert ducker.flush()
        assert backend.volumes["spotify"] == "0.150"
    finally:
        vl.stop()
    assert backend.volumes["spotify"] == "0.750000"
    assert not ducker.is_ducked()
    assert all(not t.is_alive() for t in vl.threads)


def test_voice_loop_sin_ducker_no_cambia_nada():
    from crotolamo.voice.loop import VoiceLoop

    received: list[dict] = []
    publisher = received.append
    vl = VoiceLoop(
        SimpleNamespace(handle_turn=lambda c: "ok"), _FakeCmdStt(), _FakeTts(), None,
        mic=_FakeMic(), wake_fn=lambda c: False, vad_fn=lambda c: 0.0,
        to_wav=lambda f: "WAV", hud_publisher=publisher,
    )
    assert vl.ducker is None
    assert vl.state._publisher is publisher  # el publisher va tal cual, sin envolver


# --- _run_simple_loop con fakes ------------------------------------------------
class _ScriptedWakeStt:
    """wake_stt falso: devuelve las frases del guion y, agotado, Ctrl-C (sale del loop)."""

    def __init__(self, heard):
        self.heard = list(heard)

    def listen_once(self, **kwargs):
        if self.heard:
            return self.heard.pop(0)
        raise KeyboardInterrupt


class _FakeAgent:
    def __init__(self, reply="hecho, patrón.", fail=False, probe=None):
        self.reply = reply
        self.fail = fail
        self.probe = probe  # se evalúa DURANTE el turno (p.ej. leer el volumen)
        self.commands: list[str] = []
        self.seen: list = []

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        if self.probe is not None:
            self.seen.append(self.probe())
        if self.fail:
            raise RuntimeError("LLM caído")
        return self.reply


def _run_simple(monkeypatch, *, heard, media=None, ducker=None, agent=None,
                command_text="abre youtube", threshold=0.72):
    from crotolamo.voice.state import SharedState
    from interfaces import listener

    monkeypatch.setattr(listener, "read_control_enabled", lambda: True)
    # Sin esperas reales: el loop duerme 0.5s por turno y 1s tras un error.
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=time.time))
    cfg = listener.ListenerConfig(
        threshold=threshold, variants=["crotolamo", "croto lamo"], followup_s=0,
        ack="off", stream_speak=False, smart_endpoint=False,
    )
    agent = agent or _FakeAgent()
    hablado: list[str] = []
    rc = listener._run_simple_loop(
        agent=agent, stt=_FakeCmdStt(command_text), wake_stt=_ScriptedWakeStt(heard),
        tts=_FakeTts(), wake_detector=None, use_oww=False, cfg=cfg,
        hud_state=SharedState(), say=hablado.append, media=media, ducker=ducker,
    )
    return rc, agent, hablado


def test_loop_simple_con_musica_usa_el_umbral_alto(monkeypatch):
    # "crotomalo": score difuso ~0.78 -> activa con 0.72 (sin música) y NO con 0.85 (con música).
    backend = FakeBackend(playing=["spotify"])
    mon = _monitor(backend, normal=0.72, media=0.85)
    mon.refresh_now()
    rc, agent, _ = _run_simple(monkeypatch, heard=["crotomalo"], media=mon)
    assert rc == 0 and agent.commands == []

    backend.playing = []
    mon.refresh_now()
    rc, agent, _ = _run_simple(monkeypatch, heard=["crotomalo"], media=mon)
    assert rc == 0 and agent.commands == ["abre youtube"]

    # Sin monitor (media=None): umbral fijo de cfg, exactamente como antes.
    rc, agent, _ = _run_simple(monkeypatch, heard=["crotomalo"])
    assert agent.commands == ["abre youtube"]


def test_loop_simple_restaura_la_musica_aunque_el_agente_reviente(monkeypatch):
    ducker, backend = _ducker()
    agent = _FakeAgent(fail=True, probe=lambda: backend.volumes["spotify"])
    rc, agent, _ = _run_simple(monkeypatch, heard=["crotolamo"], ducker=ducker, agent=agent)
    assert rc == 0
    assert agent.seen == ["0.150"]  # mientras pensaba, la música estaba bajita
    assert backend.volumes["spotify"] == "0.750000"  # y volvió aunque el turno reventó
    assert not ducker.is_ducked()
    assert backend.set_calls == [("spotify", "0.150"), ("spotify", "0.750000")]


def test_loop_simple_restaura_con_orden_vacia_y_con_orden_pegada(monkeypatch):
    # Orden vacía ("No te escuché claro"): el continue pasa por el finally.
    ducker, backend = _ducker()
    rc, agent, hablado = _run_simple(monkeypatch, heard=["crotolamo"], ducker=ducker,
                                     command_text="")
    assert agent.commands == [] and any("No te escuché" in h for h in hablado)
    assert backend.set_calls == [("spotify", "0.150"), ("spotify", "0.750000")]

    # Orden pegada al wake: se atiende directo, también con duck + restore.
    ducker, backend = _ducker()
    rc, agent, _ = _run_simple(monkeypatch, heard=["crotolamo pausa la música"], ducker=ducker)
    assert agent.commands == ["pausa la música"]
    assert backend.set_calls == [("spotify", "0.150"), ("spotify", "0.750000")]


# --- ListenerConfig y config de fábrica -------------------------------------
def test_listener_config_lee_los_knobs_de_musica_con_defaults():
    from interfaces.listener import ListenerConfig

    class _S:
        voice: dict = {}
        wake: dict = {}

    cfg = ListenerConfig.from_settings(_S())
    assert cfg.threshold_media == pytest.approx(0.85)
    assert cfg.media_poll_s == pytest.approx(1.5)
    assert cfg.duck is True
    assert cfg.duck_volume == pytest.approx(0.2)

    class _S2:
        voice = {"wake_threshold_media": 0.9, "media_poll_s": 3.0,
                 "wake_duck": False, "wake_duck_volume": 0.35}
        wake = {"threshold": 0.7}

    cfg = ListenerConfig.from_settings(_S2())
    assert (cfg.threshold_media, cfg.media_poll_s, cfg.duck, cfg.duck_volume) == (
        0.9, 3.0, False, 0.35
    )
    assert cfg.threshold == pytest.approx(0.7)


def test_config_de_fabrica_documenta_los_knobs():
    from crotolamo.settings import DEFAULT_CONFIG

    with DEFAULT_CONFIG.open("rb") as fh:
        raw = tomllib.load(fh)
    voice, wake = raw["voice"], raw["wake"]
    assert voice["wake_threshold_media"] == pytest.approx(0.85)
    assert voice["media_poll_s"] == pytest.approx(1.5)
    assert voice["wake_duck"] is True
    assert voice["wake_duck_volume"] == pytest.approx(0.2)
    assert wake["oww_threshold_media"] == pytest.approx(0.7)
