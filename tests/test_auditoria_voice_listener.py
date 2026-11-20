"""Auditoría: hallazgos del grupo voice-listener (interfaces/listener.py).

- Hotwords: el loop simple con smart_endpoint apagado transcribía la orden sin
  el sesgo de [voice].hotwords (la rama smart y el loop concurrente sí lo pasan).
- Pausa del panel: el loop simple solo miraba control.json TRAS un wake, así
  que el switch "Escuchar por voz" se revertía solo y Whisper seguía
  transcribiendo el ambiente con la escucha "pausada".
- Apagado: el idle final debe ser lo ÚLTIMO que se publica (una publicación de
  otro hilo podía pisarlo y dejar el HUD en "HABLANDO"), y los temporales de
  una publicación interrumpida se barren al arrancar.
- Contrato: rutas y esquema del estado, iguales en listener, state y desktop.
- HUD del loop simple: cada wake (y cada followup) arranca turno con el texto
  limpio, y con stream_speak la respuesta se publica antes de terminar de hablarla.
- V9: en el loop concurrente, el aviso de una tool delicada suena con el modo en
  SPEAKING (no THINKING) y el de un turno ya interrumpido no se dice.
- C5: en el modo simple, la respuesta a "confirmo o cancela" la transcribe el
  modelo de la orden, no el 'tiny' del wake.

Todo con fakes: sin micrófono, sin modelos, sin tocar el HOME real.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo.voice import state as voice_state
from crotolamo.voice.state import SharedState, make_file_publisher

from interfaces import listener

# desktop/ no es paquete: se importa igual que en producción (sys.path).
DESKTOP_DIR = str(Path(__file__).resolve().parent.parent / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from common import ipc  # noqa: E402

HOTWORDS = "Crotolamo, Tletl, Huevonitis"


# --- fakes ---------------------------------------------------------------------
class _RecStt:
    """STT de la orden: registra los kwargs y devuelve las frases del guion."""

    def __init__(self, texts=("abre youtube",)):
        self.texts = list(texts)
        self.calls: list[tuple[str, dict]] = []

    def _next(self) -> str:
        return self.texts.pop(0) if self.texts else ""

    def listen_once(self, **kwargs):
        self.calls.append(("listen_once", kwargs))
        return self._next()

    def listen_smart(self, **kwargs):
        self.calls.append(("listen_smart", kwargs))
        return self._next()


class _ScriptedWake:
    """wake_stt falso: frases del guion y, agotado, Ctrl-C (sale del loop)."""

    def __init__(self, heard, on_listen=None):
        self.heard = list(heard)
        self.calls: list[dict] = []
        self.on_listen = on_listen

    def listen_once(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_listen is not None:
            self.on_listen(len(self.calls))
        if self.heard:
            return self.heard.pop(0)
        raise KeyboardInterrupt


class _FakeTts:
    def available(self):
        return False

    def speak(self, text):
        pass

    def speak_sentences(self, text):
        pass

    def beep(self):
        pass

    def stop(self):
        pass


class _FakeAgent:
    def __init__(self):
        self.commands: list[str] = []

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        return "hecho, patrón."


def _cfg(**overrides):
    base = dict(threshold=0.72, variants=["crotolamo"], followup_s=0, ack="off",
                stream_speak=False, smart_endpoint=False)
    base.update(overrides)
    return listener.ListenerConfig(**base)


def _run_simple(monkeypatch, *, wake_stt, stt=None, cfg=None, hud=None, agent=None,
                sleep=lambda s: None, flag=None):
    """Corre _run_simple_loop con fakes. `flag` = dict mutable con el flag del panel."""
    flag = flag if flag is not None else {"enabled": True}
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: flag["enabled"])
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=sleep, time=time.time))
    agent = agent or _FakeAgent()
    rc = listener._run_simple_loop(
        agent=agent, stt=stt or _RecStt(), wake_stt=wake_stt, tts=_FakeTts(),
        wake_detector=None, use_oww=False, cfg=cfg or _cfg(),
        hud_state=hud or SharedState(), say=lambda text: None,
    )
    return rc, agent


def _wait_until(cond, timeout=5.0) -> bool:
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def _mode(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["mode"]


def _estado(mode: str, **extra) -> dict:
    base = {"mode": mode, "turn_id": 1, "text": "", "enabled": True,
            "ts": time.time(), "pid": os.getpid()}
    base.update(extra)
    return base


# --- hotwords en el loop simple --------------------------------------------------
@pytest.mark.parametrize("smart", [False, True])
def test_loop_simple_pasa_hotwords_a_la_orden_en_las_dos_ramas(monkeypatch, smart):
    """ListenerConfig: "Hotwords solo para COMANDOS (post-wake), igual que el loop
    concurrente". Con smart_endpoint = false la orden salía SIN hotwords."""
    stt = _RecStt(["abre youtube", ""])  # la orden + una ventana de followup vacía
    wake_stt = _ScriptedWake(["crotolamo"])
    rc, agent = _run_simple(
        monkeypatch, wake_stt=wake_stt, stt=stt,
        cfg=_cfg(smart_endpoint=smart, hotwords=HOTWORDS, followup_s=6.0),
    )
    assert rc == 0 and agent.commands == ["abre youtube"]
    metodo = "listen_smart" if smart else "listen_once"
    # La orden y la ventana de followup, las dos por la misma rama y con hotwords.
    assert [m for m, _ in stt.calls] == [metodo, metodo]
    assert [kw.get("hotwords") for _, kw in stt.calls] == [HOTWORDS, HOTWORDS]
    # La ruta del WAKE jamás lleva hotwords: un sesgo con "crotolamo" sobre
    # ruido/silencio aumentaría los falsos despertares (ver STT.transcribe).
    assert wake_stt.calls and all("hotwords" not in kw for kw in wake_stt.calls)


# --- pausa del panel en el loop simple ------------------------------------------
def test_loop_simple_publica_la_pausa_sin_esperar_a_un_wake(monkeypatch):
    """El panel pausa durante la espera ambiente. Antes: 0 lecturas del flag
    hasta el siguiente wake, hud_state.json seguía con enabled=true (el switch
    del panel se revertía) y Whisper seguía transcribiendo el ambiente."""
    flag = {"enabled": True}
    escuchas_en_pausa: list[int] = []
    dormidas: list[float] = []

    def on_listen(n):
        if not flag["enabled"]:
            escuchas_en_pausa.append(n)
        if n == 1:
            flag["enabled"] = False  # el panel escribe control.json = false
        if n >= 2:
            raise KeyboardInterrupt  # el código viejo volvía a escuchar: cortar

    def sleep(s):
        dormidas.append(s)
        if len(dormidas) >= 5:
            raise KeyboardInterrupt  # 5 vueltas en pausa bastan

    pubs: list[dict] = []
    wake_stt = _ScriptedWake(["", "", ""], on_listen=on_listen)
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt, flag=flag, sleep=sleep,
                            hud=SharedState(publisher=pubs.append))
    assert rc == 0 and agent.commands == []
    assert escuchas_en_pausa == [], "siguió abriendo el micrófono con la escucha pausada"
    assert len(wake_stt.calls) == 1
    # La pausa quedó publicada para el panel, sin que nadie dijera "crotolamo".
    assert [p["enabled"] for p in pubs] == [False]
    assert pubs[0]["mode"] == "idle"
    # En pausa el loop duerme al ritmo del sondeo (no gira en vacío).
    assert dormidas == [listener._CONTROL_POLL_S] * 5


def test_loop_simple_pausado_al_arrancar_no_escucha_y_reanuda(monkeypatch):
    flag = {"enabled": False}
    escuchas_en_pausa: list[int] = []
    dormidas: list[float] = []

    def on_listen(n):
        if not flag["enabled"]:
            escuchas_en_pausa.append(n)

    def sleep(s):
        dormidas.append(s)
        if len(dormidas) == 3:
            flag["enabled"] = True  # el patrón reactiva la escucha en el panel
        if len(dormidas) > 50:
            raise KeyboardInterrupt

    pubs: list[dict] = []
    wake_stt = _ScriptedWake(["crotolamo"], on_listen=on_listen)
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt, flag=flag, sleep=sleep,
                            hud=SharedState(publisher=pubs.append))
    assert rc == 0
    assert escuchas_en_pausa == []
    # Reanudada, el wake vuelve a atenderse con normalidad.
    assert agent.commands == ["abre youtube"]
    enabled = [p["enabled"] for p in pubs]
    assert enabled[0] is False and enabled[-1] is True


def test_loop_simple_sigue_ignorando_el_wake_si_se_pausa_durante_la_espera(monkeypatch):
    """La pausa llega con el wake ya grabándose: ese wake se ignora (como antes)."""
    flag = {"enabled": True}

    def on_listen(n):
        if n == 1:
            flag["enabled"] = False

    dormidas: list[float] = []

    def sleep(s):
        dormidas.append(s)
        if len(dormidas) >= 3:
            raise KeyboardInterrupt

    hud = SharedState()
    wake_stt = _ScriptedWake(["crotolamo"], on_listen=on_listen)
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt, flag=flag, sleep=sleep, hud=hud)
    assert rc == 0 and agent.commands == []
    assert hud.is_enabled() is False


def test_vigia_del_control_publica_la_pausa_y_solo_relee_si_cambia_el_archivo(
        tmp_path, monkeypatch):
    """Con el wake bloqueado (openWakeWord con timeout None) el loop no puede
    sondear: el vigía publica la pausa mientras tanto. Como en el loop
    concurrente, el JSON solo se relee cuando cambia el mtime."""
    control = tmp_path / "control.json"
    lecturas: list[Path] = []
    real_read = voice_state.read_control_enabled

    def contando(path):
        lecturas.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(listener, "read_control_enabled", contando)
    pubs: list[dict] = []
    hud = SharedState(publisher=pubs.append)
    stop = listener._start_control_watcher(hud, control, poll_s=0.01)
    try:
        time.sleep(0.1)  # sin archivo: escucha activa, nada que publicar ni leer
        assert hud.is_enabled() is True and pubs == [] and lecturas == []

        control.write_text('{"listening_enabled": false}', encoding="utf-8")
        assert _wait_until(lambda: not hud.is_enabled())
        assert pubs[-1]["enabled"] is False
        time.sleep(0.15)  # ~15 sondeos con el archivo intacto
        assert lecturas == [control], "releyó el JSON sin que cambiara el mtime"

        control.write_text('{"listening_enabled": true}', encoding="utf-8")
        os.utime(control, (time.time() + 5, time.time() + 5))  # mtime distinto seguro
        assert _wait_until(hud.is_enabled)
        assert pubs[-1]["enabled"] is True

        control.write_text('{"listening_enabled": false}', encoding="utf-8")
        os.utime(control, (time.time() + 10, time.time() + 10))
        assert _wait_until(lambda: not hud.is_enabled())
        control.unlink()  # archivo ausente => default-seguro: escucha activa
        assert _wait_until(hud.is_enabled)
    finally:
        stop()
    assert not [t for t in threading.enumerate() if t.name == "ControlWatch"]
    n = len(pubs)
    control.write_text('{"listening_enabled": false}', encoding="utf-8")
    time.sleep(0.1)
    assert len(pubs) == n, "el vigía siguió vivo tras pararlo"


# --- arnés de run_listen (modo simple con openWakeWord) --------------------------
class _FakeSttCls:
    model_size = "tiny"

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def _get_model(self):
        return None

    def listen_once(self, **kwargs):
        return ""

    def listen_smart(self, **kwargs):
        return ""


class _FakeTtsCls(_FakeTts):
    @classmethod
    def from_settings(cls, settings):
        return cls()


class _FakeBackend:
    def playing_players(self):
        return []

    def get_volume(self, player):
        return None

    def set_volume(self, player, value):
        return True


class _BlockingDetector:
    """openWakeWord falso: listen_for_wake BLOQUEA (como timeout_s=None en un
    cuarto en silencio) hasta que el test lo suelta; entonces, Ctrl-C."""

    threshold = 0.5
    threshold_media = None

    def __init__(self):
        self.esperando = threading.Event()
        self.soltar = threading.Event()
        self.llamadas = 0

    def available(self):
        return True

    def attach_media(self, monitor):
        pass

    def listen_for_wake(self, timeout_s=None):
        self.llamadas += 1
        self.esperando.set()
        self.soltar.wait(20)
        raise KeyboardInterrupt


def _arnes_run_listen(monkeypatch, tmp_path, detector):
    """Parchea todo lo pesado de run_listen (modelos, audio, agente, LLM) y
    apunta el HUD y el canal de control a tmp_path."""
    hud_path = tmp_path / "hud_state.json"
    control = tmp_path / "control.json"
    real_read = voice_state.read_control_enabled
    monkeypatch.setattr(listener, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(listener, "get_settings",
                        lambda: SimpleNamespace(voice={"wake_duck": False}, wake={}))
    monkeypatch.setattr(listener, "STT", _FakeSttCls)
    monkeypatch.setattr(listener, "TTS", _FakeTtsCls)
    monkeypatch.setattr(listener, "WakeWordDetector",
                        SimpleNamespace(from_settings=lambda settings: detector))
    monkeypatch.setattr(listener, "PlayerctlBackend", _FakeBackend)
    monkeypatch.setattr(listener, "build_agent", lambda confirm_fn=None: (_FakeAgent(), None))
    monkeypatch.setattr(listener, "_HUD_STATE_PATH", hud_path)
    monkeypatch.setattr(listener, "CONTROL_PATH", control)
    monkeypatch.setattr(listener, "read_control_enabled",
                        lambda path=control: real_read(path))
    monkeypatch.setattr(listener, "_ACTIVE_DUCKER", None)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", None, raising=False)
    from crotolamo.core import engine

    monkeypatch.setattr(engine, "build_primary_llm",
                        lambda settings: SimpleNamespace(chat=lambda messages: "ok"))
    return hud_path, control


def _join_warmers():
    for hilo in threading.enumerate():
        if hilo.name in ("WarmVoice", "WarmLLM"):
            hilo.join(timeout=5.0)


def test_run_listen_simple_publica_la_pausa_con_el_wake_bloqueado(monkeypatch, tmp_path):
    """De punta a punta: modo --simple con openWakeWord esperando el wake para
    siempre. El panel pausa: hud_state.json debe reflejarlo en el acto (antes:
    nunca, hasta que alguien dijera la palabra de activación)."""
    detector = _BlockingDetector()
    hud_path, control = _arnes_run_listen(monkeypatch, tmp_path, detector)
    rc: list[int] = []
    hilo = threading.Thread(target=lambda: rc.append(listener.run_listen(["--simple"])),
                            daemon=True)
    hilo.start()
    try:
        assert detector.esperando.wait(10), "el loop no llegó a esperar el wake"
        assert _wait_until(hud_path.exists)
        assert json.loads(hud_path.read_text(encoding="utf-8"))["enabled"] is True

        control.write_text('{"listening_enabled": false}', encoding="utf-8")
        reflejado = _wait_until(
            lambda: json.loads(hud_path.read_text(encoding="utf-8"))["enabled"] is False,
            timeout=5.0,
        )
        assert detector.llamadas == 1  # seguía bloqueado: nadie dijo "crotolamo"
        assert reflejado, "la pausa del panel no llegó a hud_state.json sin un wake"
    finally:
        detector.soltar.set()
        hilo.join(timeout=10.0)
        _join_warmers()
    assert rc == [0]
    final = json.loads(hud_path.read_text(encoding="utf-8"))
    assert final["mode"] == "idle" and final["enabled"] is False
    assert not [t for t in threading.enumerate() if t.name == "ControlWatch"]


# --- apagado: el idle final es lo último ------------------------------------------
class _Salida(BaseException):
    """Sustituto de os._exit en los tests del manejador de señal."""


def _parchear_salida(monkeypatch, destino):
    def _exit(code):
        raise _Salida(code)

    real_idle = listener._write_idle_hud
    monkeypatch.setattr(listener.os, "_exit", _exit)
    monkeypatch.setattr(listener, "_write_idle_hud", lambda path=destino: real_idle(path))
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", None, raising=False)


def test_apagado_por_senal_escribe_el_idle_despues_de_cerrar_mcp(tmp_path, monkeypatch):
    """systemctl stop con la Boca hablando: mientras el manejador cierra los
    servers MCP, otro hilo publica "speaking". El idle se escribía ANTES, así
    que el HUD se quedaba en "HABLANDO" hasta el siguiente arranque."""
    destino = tmp_path / "hud_state.json"
    _parchear_salida(monkeypatch, destino)
    boca = make_file_publisher(destino)
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort",
                        lambda: boca(_estado("speaking")))
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    assert _mode(destino) == "idle"


def test_compuerta_del_hud_descarta_lo_que_llega_tras_el_idle_final(tmp_path, monkeypatch):
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    destino = tmp_path / "hud_state.json"
    gate = listener._HudGate(destino)
    gate.publish(_estado("speaking"))
    assert _mode(destino) == "speaking"  # abierta: publica tal cual
    gate.close_with_idle()
    assert _mode(destino) == "idle"
    gate.publish(_estado("speaking"))  # un hilo rezagado tras el apagado
    assert _mode(destino) == "idle"
    assert list(tmp_path.glob("*.tmp")) == []


def test_el_idle_final_espera_a_la_publicacion_en_curso(tmp_path, monkeypatch):
    """Otro hilo está ENTRE mkstemp y os.replace cuando llega el apagado: sin
    compuerta, su replace caía después del idle (HUD en "HABLANDO") o, con
    os._exit, dejaba el temporal huérfano."""
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    destino = tmp_path / "hud_state.json"
    gate = listener._HudGate(destino)
    dentro, soltar = threading.Event(), threading.Event()
    real_replace = os.replace

    def replace_lento(src, dst):
        if threading.current_thread().name == "Boca":
            dentro.set()
            soltar.wait(5)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_lento)
    boca = threading.Thread(target=lambda: gate.publish(_estado("speaking")), name="Boca")
    boca.start()
    assert dentro.wait(5)
    threading.Timer(0.1, soltar.set).start()
    gate.close_with_idle(lock_timeout_s=5.0)
    boca.join(5)
    assert _mode(destino) == "idle"
    assert list(tmp_path.glob("*.tmp")) == []


def test_el_idle_final_no_se_bloquea_si_el_hilo_principal_estaba_publicando(
        tmp_path, monkeypatch):
    """El manejador de señal corre en el hilo principal, que puede haber sido
    interrumpido A MEDIA publicación (loop simple): el cierre no puede esperar
    a un candado que tiene ese mismo hilo."""
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    destino = tmp_path / "hud_state.json"
    gate = listener._HudGate(destino)
    real_replace = os.replace
    tardo: list[float] = []

    def replace_interrumpido(src, dst):
        if not tardo:  # la señal cae justo aquí, dentro de la publicación
            t0 = time.monotonic()
            tardo.append(0.0)
            gate.close_with_idle(lock_timeout_s=5.0)
            tardo[0] = time.monotonic() - t0
            raise _Salida(0)  # os._exit: la publicación interrumpida no continúa
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_interrumpido)
    with pytest.raises(_Salida):
        gate.publish(_estado("speaking"))
    assert tardo[0] < 2.0, "el cierre esperó el candado de su propio hilo"
    assert _mode(destino) == "idle"


def test_apagado_por_senal_cierra_la_compuerta_activa(tmp_path, monkeypatch):
    destino = tmp_path / "hud_state.json"
    _parchear_salida(monkeypatch, tmp_path / "no_debe_usarse.json")
    orden: list[str] = []
    gate = listener._HudGate(destino)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", gate)
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort",
                        lambda: (orden.append("mcp"), gate.publish(_estado("speaking"))))
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    assert orden == ["mcp"]
    assert _mode(destino) == "idle"
    gate.publish(_estado("thinking"))  # hilo rezagado entre el idle y os._exit
    assert _mode(destino) == "idle"
    assert not (tmp_path / "no_debe_usarse.json").exists()


def test_run_listen_concurrente_un_hilo_rezagado_no_pisa_el_idle_final(
        monkeypatch, tmp_path):
    """VoiceLoop.stop() hace join con timeout: un hilo que sigue vivo (la Boca a
    media frase) puede publicar DESPUÉS del idle final del finally."""
    detector = _BlockingDetector()
    hud_path, _control = _arnes_run_listen(monkeypatch, tmp_path, detector)
    capturado: dict = {}

    class _FakeLoop:
        def __init__(self, agent, stt, tts, wake_detector, **kwargs):
            capturado.update(kwargs)

        def run(self):
            capturado["hud_publisher"](_estado("speaking"))

    from crotolamo.voice import loop as voice_loop

    monkeypatch.setattr(voice_loop, "VoiceLoop", _FakeLoop)
    try:
        assert listener.run_listen([]) == 0
    finally:
        _join_warmers()
    assert capturado["control_path"] == tmp_path / "control.json"
    assert _mode(hud_path) == "idle"
    capturado["hud_publisher"](_estado("speaking"))  # el hilo rezagado
    assert _mode(hud_path) == "idle"


# --- temporales huérfanos ---------------------------------------------------------
def test_barrido_de_temporales_huerfanos_al_arrancar(tmp_path):
    """os._exit a media publicación deja tmpXXXX.tmp en ~/.crotolamo y nada los
    barría. Solo se borran los VIEJOS con el patrón de mkstemp: uno reciente
    puede ser una escritura en curso (del panel o de otro proceso)."""
    viejo = tmp_path / "tmpabc12345.tmp"
    reciente = tmp_path / "tmpzzz99999.tmp"
    ajenos = [tmp_path / "hud_state.json", tmp_path / "control.json",
              tmp_path / "crotolamo.sqlite", tmp_path / "notas.tmp"]
    for archivo in (viejo, reciente, *ajenos):
        archivo.write_text("{}", encoding="utf-8")
    hace_rato = time.time() - 3600
    for archivo in (viejo, *ajenos):
        os.utime(archivo, (hace_rato, hace_rato))
    carpeta = tmp_path / "tmpdirectorio.tmp"
    carpeta.mkdir()
    os.utime(carpeta, (hace_rato, hace_rato))

    assert listener._sweep_stale_tmp(tmp_path) == 1
    assert not viejo.exists()
    assert reciente.exists() and carpeta.is_dir()
    assert all(archivo.exists() for archivo in ajenos)
    # Carpeta inexistente: no lanza.
    assert listener._sweep_stale_tmp(tmp_path / "no_existe") == 0


def test_run_listen_barre_los_temporales_viejos(monkeypatch, tmp_path):
    detector = _BlockingDetector()
    detector.soltar.set()  # Ctrl-C en la primera espera: arranca y sale
    hud_path, _control = _arnes_run_listen(monkeypatch, tmp_path, detector)
    huerfano = tmp_path / "tmpdeadbeef.tmp"
    huerfano.write_text('{"mode": "speaking"}', encoding="utf-8")
    hace_rato = time.time() - 3600
    os.utime(huerfano, (hace_rato, hace_rato))
    try:
        assert listener.run_listen(["--simple"]) == 0
    finally:
        _join_warmers()
    assert not huerfano.exists()
    assert _mode(hud_path) == "idle"


# --- contrato de rutas y esquema --------------------------------------------------
def test_rutas_del_contrato_iguales_en_listener_state_y_desktop():
    """El loop escribe hud_state.json y lee control.json; HUD y panel hacen lo
    inverso desde desktop/ (que no importa crotolamo). Tres definiciones: si
    una se mueve, el HUD o el switch del panel se desconectan en silencio."""
    assert listener._HUD_STATE_PATH == ipc.HUD_STATE_FILE
    assert listener.CONTROL_PATH is voice_state.CONTROL_PATH
    assert voice_state.CONTROL_PATH == ipc.CONTROL_FILE
    # read_control_enabled() sin argumentos lee esa misma ruta.
    assert voice_state.read_control_enabled.__defaults__ == (ipc.CONTROL_FILE,)


def test_idle_final_tiene_el_mismo_esquema_que_las_publicaciones(tmp_path, monkeypatch):
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    publicado: list[dict] = []
    estado = SharedState(publisher=publicado.append)
    estado.publish_now()
    destino = tmp_path / "hud_state.json"
    listener._write_idle_hud(destino)
    final = json.loads(destino.read_text(encoding="utf-8"))
    assert set(final) == set(publicado[0]) == set(estado.current_snapshot())
    for clave in ("mode", "turn_id", "text", "enabled"):
        assert final[clave] == publicado[0][clave]
    assert {type(final[c]) for c in ("ts", "pid")} == {float, int}


# --- SIGHUP: cerrar la terminal apaga igual que un stop -----------------------------
# Hijo real con los manejadores de verdad: un ducker falso deja constancia de la
# restauración y la compuerta del HUD arranca en "speaking" (a media respuesta).
_HIJO_SENALES = r"""
import signal, sys, time
from pathlib import Path
from interfaces import listener

home = Path(sys.argv[1])
if sys.argv[2] == "nohup":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # lo que hace `nohup`
else:
    signal.signal(signal.SIGHUP, signal.SIG_DFL)  # aunque pytest corra bajo nohup


class Ducker:
    def emergency_restore(self):
        (home / "musica_restaurada").write_text("ok", encoding="utf-8")


listener._ACTIVE_DUCKER = Ducker()
gate = listener._HudGate(home / "hud_state.json")
gate.publish({"mode": "speaking", "turn_id": 3, "text": "", "enabled": True,
              "ts": 0.0, "pid": 0})
listener._ACTIVE_HUD_GATE = gate
listener._install_signal_handlers()
print("listo", flush=True)
# En tramos y no signal.pause(): la señal que llega entre el último chequeo del
# intérprete y la syscall pause() la apunta el manejador C, pero pause() ya no
# despierta (1 de 3000 bajo carga: el hijo no salía y el test agotaba sus 15 s).
while True:
    time.sleep(0.05)
"""


def _hijo_con_senales(tmp_path, modo):
    import subprocess

    raiz = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": raiz}
    hijo = subprocess.Popen(
        [sys.executable, "-c", _HIJO_SENALES, str(tmp_path), modo],
        env=env, cwd=raiz, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert hijo.stdout is not None
    assert hijo.stdout.readline().strip() == "listo", hijo.stderr.read() if hijo.stderr else ""
    return hijo


def test_sighup_apaga_limpio_como_sigterm(tmp_path):
    """`kitty --hold -e launch.sh` y el patrón cierra la ventana a media
    respuesta: la terminal manda SIGHUP. Sin manejador, la acción por defecto
    mataba el proceso sin pasar por _graceful_exit: Spotify se quedaba al 20%,
    hud_state.json en "speaking" y los servers MCP (otra sesión) huérfanos."""
    hijo = _hijo_con_senales(tmp_path, "normal")
    try:
        hijo.send_signal(signal.SIGHUP)
        rc = hijo.wait(timeout=15)
    finally:
        if hijo.poll() is None:
            hijo.kill()
            hijo.wait()
    assert rc == 0, "SIGHUP mató el proceso sin el apagado limpio"
    assert (tmp_path / "musica_restaurada").exists()
    assert _mode(tmp_path / "hud_state.json") == "idle"


def test_sighup_ignorado_por_nohup_se_respeta(tmp_path):
    """Quien lanza `nohup crotolamo listen` pidió sobrevivir al cierre de la
    terminal: el manejador nuevo no puede pisar ese SIG_IGN heredado."""
    hijo = _hijo_con_senales(tmp_path, "nohup")
    try:
        hijo.send_signal(signal.SIGHUP)
        time.sleep(0.5)
        assert hijo.poll() is None, "SIGHUP tumbó un proceso lanzado con nohup"
        assert _mode(tmp_path / "hud_state.json") == "speaking"
        hijo.send_signal(signal.SIGTERM)  # el stop normal sigue funcionando
        assert hijo.wait(timeout=15) == 0
    finally:
        if hijo.poll() is None:
            hijo.kill()
            hijo.wait()
    assert _mode(tmp_path / "hud_state.json") == "idle"


# --- espera del wake: se anuncia una vez, no en cada ventana vacía ------------------
_ESPERANDO = "Esperando palabra de activación..."


def test_loop_simple_anuncia_la_espera_una_vez_por_periodo_de_espera(monkeypatch, capsys):
    """Wake difuso: cada ventana de 6 s sin activación volvía al tope del bucle
    y reimprimía el aviso (dos líneas a stdout/journald cada 6 s de silencio,
    ~28.800 al día, y la última respuesta se iba de la pantalla). Con
    openWakeWord el aviso sale una vez por espera: el difuso, igual."""
    wake_stt = _ScriptedWake(["", "", "hola qué tal", "crotolamo", "", "", ""])
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt)
    assert rc == 0 and agent.commands == ["abre youtube"]
    assert len(wake_stt.calls) == 8  # siete ventanas del guion + el Ctrl-C
    salida = capsys.readouterr().out
    # Una al arrancar y otra tras atender la orden: no una por ventana.
    assert salida.count(_ESPERANDO) == 2
    assert salida.index("Orden: abre youtube") < salida.rindex(_ESPERANDO)


def test_loop_simple_reanuncia_la_espera_tras_pausa_y_tras_error(monkeypatch, capsys):
    flag = {"enabled": True}

    def on_listen(n):
        if n == 2:
            raise RuntimeError("micrófono desconectado")  # "Error en el listener"

    dormidas: list[float] = []

    def sleep(s):
        dormidas.append(s)
        if s == listener._CONTROL_POLL_S and len(dormidas) >= 3:
            flag["enabled"] = True  # el patrón reactiva la escucha

    wake_stt = _ScriptedWake(["", "", "", "", ""], on_listen=on_listen)
    real_listen = wake_stt.listen_once

    def listen_y_pausa(**kwargs):
        texto = real_listen(**kwargs)
        if len(wake_stt.calls) == 3:
            flag["enabled"] = False  # pausa desde el panel tras la tercera ventana
        return texto

    wake_stt.listen_once = listen_y_pausa
    rc, agent = _run_simple(monkeypatch, wake_stt=wake_stt, flag=flag, sleep=sleep)
    assert rc == 0 and agent.commands == []
    salida = capsys.readouterr().out
    # Al arrancar, tras el error y al reanudar tras la pausa.
    assert salida.count(_ESPERANDO) == 3
    assert "Error en el listener: micrófono desconectado" in salida


# --- HUD del loop simple: cada interacción arranca con el texto limpio --------------
class _GuionAgent:
    """Agente falso con una respuesta por orden; con on_token la va soltando a
    pedazos (como el streaming del LLM) antes de devolverla entera."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.commands: list[str] = []

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        reply = self.replies.pop(0)
        if on_token is not None:
            for palabra in reply.split(" "):
                on_token(palabra + " ")
        return reply


def _publicaciones_en_escucha(pubs: list[dict]) -> list[tuple[str, int]]:
    return [(p["text"], p["turn_id"]) for p in pubs if p["mode"] == "listening"]


def test_loop_simple_no_muestra_la_respuesta_anterior_bajo_escuchando(monkeypatch):
    """Segundo wake del loop simple: set_mode(LISTENING) republicaba el texto
    que quedaba (la respuesta anterior) y el HUD pintaba "ESCUCHANDO" con ella
    debajo hasta transcribir la orden nueva. El loop concurrente arranca cada
    wake con new_turn(), que limpia el texto; el simple nunca lo llamaba (y su
    turn_id se quedaba en 0 para siempre)."""
    pubs: list[dict] = []
    agent = _GuionAgent(["Son las cinco, patrón.", "Listo, ya la pausé."])
    rc, _ = _run_simple(
        monkeypatch, wake_stt=_ScriptedWake(["crotolamo", "crotolamo"]),
        stt=_RecStt(["qué hora es", "pausa la música"]), agent=agent,
        hud=SharedState(publisher=pubs.append),
    )
    assert rc == 0 and agent.commands == ["qué hora es", "pausa la música"]
    # Al aparecer el HUD, vacío; luego la orden reconocida. Un turno por wake.
    assert _publicaciones_en_escucha(pubs) == [
        ("", 1), ("qué hora es", 1), ("", 2), ("pausa la música", 2),
    ]


def test_loop_simple_followup_no_muestra_la_respuesta_bajo_escuchando(monkeypatch):
    """La ventana de followup ("sigo escuchando") es una orden nueva sin wake:
    tampoco debe pintar la respuesta recién hablada bajo "ESCUCHANDO"."""
    pubs: list[dict] = []
    agent = _GuionAgent(["Son las cinco, patrón.", "Mañana llueve, patrón."])
    rc, _ = _run_simple(
        monkeypatch, wake_stt=_ScriptedWake(["crotolamo"]),
        stt=_RecStt(["qué hora es", "y mañana", ""]), agent=agent,
        cfg=_cfg(followup_s=6.0), hud=SharedState(publisher=pubs.append),
    )
    assert rc == 0 and agent.commands == ["qué hora es", "y mañana"]
    assert _publicaciones_en_escucha(pubs) == [
        ("", 1), ("qué hora es", 1), ("", 2), ("y mañana", 2), ("", 3),
    ]


class _TtsQueAnota(_FakeTts):
    """TTS falso que anota cada frase hablada en la misma línea de tiempo que
    las publicaciones del HUD (speak corre en el hilo del StreamSpeaker)."""

    def __init__(self, eventos: list[tuple]):
        self.eventos = eventos

    def speak(self, text):
        self.eventos.append(("habla", text))


def test_loop_simple_stream_speak_publica_la_respuesta_antes_de_terminar_de_hablarla(
        monkeypatch):
    """Con stream_speak, set_text(reply) iba DESPUÉS de speaker.finish(), que
    espera a que se hable la cola entera: mientras hablaba, el HUD mostraba la
    orden del patrón bajo "HABLANDO" y la respuesta aparecía ya dicha (la rama
    sin streaming y el loop concurrente la publican antes de hablarla)."""
    eventos: list[tuple] = []
    respuesta = "Son las cinco, patrón. Y mañana llueve."
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=time.time))
    rc = listener._run_simple_loop(
        agent=_GuionAgent([respuesta]), stt=_RecStt(["qué hora es"]),
        wake_stt=_ScriptedWake(["crotolamo"]), tts=_TtsQueAnota(eventos),
        wake_detector=None, use_oww=False, cfg=_cfg(stream_speak=True),
        hud_state=SharedState(publisher=lambda d: eventos.append(("hud", d["mode"], d["text"]))),
        say=lambda text: None,
    )
    assert rc == 0
    habladas = [e[1] for e in eventos if e[0] == "habla"]
    assert habladas == ["Son las cinco, patrón.", "Y mañana llueve."]  # el streaming sigue
    publicada = [i for i, e in enumerate(eventos) if e[0] == "hud" and e[2] == respuesta]
    ultima_frase = max(i for i, e in enumerate(eventos) if e[0] == "habla")
    assert publicada and publicada[0] < ultima_frase, (
        "la respuesta llegó al HUD cuando ya se había terminado de hablar")


# --- loop simple: la respuesta llega a la consola antes de terminar de hablarla ------
def test_loop_simple_stream_speak_imprime_la_respuesta_antes_de_terminar_de_hablarla(
        monkeypatch):
    """El HUD ya recibía la respuesta antes de finish(), pero el print(reply)
    seguía después: la terminal (kitty, journald) mostraba la respuesta cuando
    ya se había terminado de decir. La rama sin streaming imprime antes de hablar."""
    eventos: list[tuple] = []
    respuesta = "Son las cinco, patrón. Y mañana llueve."
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=time.time))
    monkeypatch.setattr(listener, "print",
                        lambda *a, **k: eventos.append(("print", " ".join(map(str, a)))),
                        raising=False)
    rc = listener._run_simple_loop(
        agent=_GuionAgent([respuesta]), stt=_RecStt(["qué hora es"]),
        wake_stt=_ScriptedWake(["crotolamo"]), tts=_TtsQueAnota(eventos),
        wake_detector=None, use_oww=False, cfg=_cfg(stream_speak=True),
        hud_state=SharedState(), say=lambda text: None,
    )
    assert rc == 0
    impresa = [i for i, e in enumerate(eventos) if e == ("print", respuesta)]
    ultima_frase = max(i for i, e in enumerate(eventos) if e[0] == "habla")
    assert len(impresa) == 1 and impresa[0] < ultima_frase, (
        "la respuesta salió a la consola cuando ya se había terminado de hablar")


# --- loop simple: un turno que revienta se dice, no solo se imprime -----------------
# La misma frase que dice el BrainThread del loop concurrente en el mismo caso.
_CABLES = "Se me trabaron los cables, patrón. Vuelve a intentarlo."


class _TtsQueAnotaTodo(_TtsQueAnota):
    """Además de speak (el StreamSpeaker), anota el camino clásico por frases."""

    def speak_sentences(self, text):
        self.eventos.append(("frases", text))


class _AgenteQueRevienta:
    """handle_turn revienta con algo que NO es LLMError (un hook, sqlite, la
    memoria), opcionalmente tras soltar texto por on_token."""

    def __init__(self, antes: str = ""):
        self.antes = antes
        self.commands: list[str] = []

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        if on_token is not None and self.antes:
            on_token(self.antes)
        raise RuntimeError("database is locked")


def _correr_turno(monkeypatch, agent, *, stream: bool, followup_s: float = 0):
    eventos: list[tuple] = []
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=lambda s: None, time=time.time))
    stt = _RecStt(["guarda esto", "y lo otro", ""])
    rc = listener._run_simple_loop(
        agent=agent, stt=stt, wake_stt=_ScriptedWake(["crotolamo"]),
        tts=_TtsQueAnotaTodo(eventos), wake_detector=None, use_oww=False,
        cfg=_cfg(stream_speak=stream, followup_s=followup_s),
        hud_state=SharedState(publisher=lambda d: eventos.append(("hud", d["mode"], d["text"]))),
        say=lambda text: eventos.append(("say", text)),
    )
    return rc, eventos


@pytest.mark.parametrize("stream", [True, False])
def test_loop_simple_turno_que_revienta_se_dice_en_personaje(monkeypatch, stream):
    """Antes el reventón caía al `except` de afuera, que solo imprimía "Error en
    el listener": el patrón daba la orden y oía silencio (el HUD se iba a idle
    sin decir nada). El BrainThread del loop concurrente, en el mismo caso,
    dice "Se me trabaron los cables": el simple, igual."""
    rc, eventos = _correr_turno(monkeypatch, _AgenteQueRevienta(), stream=stream)
    assert rc == 0
    dicho = [e[1] for e in eventos if e[0] in ("habla", "frases")]
    assert dicho == [_CABLES]
    # En el HUD, bajo "HABLANDO", el aviso (no la orden del patrón).
    assert ("hud", "speaking", _CABLES) in eventos
    # El cierre de siempre: idle al final, y Ctrl-C sigue apagando.
    assert [e for e in eventos if e[0] == "hud"][-1][1] == "idle"
    assert eventos[-1] == ("say", "Crotolamo apagado, patrón.")


def test_loop_simple_turno_que_revienta_tras_hablar_a_medias_tambien_avisa(monkeypatch):
    """stream_speak y el reventón DESPUÉS de soltar media respuesta: finish()
    devuelve spoke=True y el fallback `if not spoke` se saltaba el aviso; se
    oía la frase cortada y nada más. El aviso va aparte, después de lo dicho."""
    agent = _AgenteQueRevienta(antes="Claro, patrón. Primero ")
    rc, eventos = _correr_turno(monkeypatch, agent, stream=True)
    assert rc == 0
    dicho = [e[1] for e in eventos if e[0] in ("habla", "frases")]
    assert dicho == ["Claro, patrón.", "Primero", _CABLES]


def test_loop_simple_turno_que_revienta_no_corta_la_conversacion(monkeypatch):
    """Tras el aviso el turno termina como cualquier otro: con followup, la
    ventana de "sigo escuchando" se abre igual (antes se perdía: el reventón
    sacaba del bucle de la interacción entera)."""
    agent = _AgenteQueRevienta()
    rc, eventos = _correr_turno(monkeypatch, agent, stream=False, followup_s=6.0)
    assert rc == 0
    assert agent.commands == ["guarda esto", "y lo otro"]


# --- C2: el motor se cae a media frase en streaming (extremo a extremo) -------------
def test_loop_simple_llm_que_cae_a_media_frase_dice_el_error_una_vez(monkeypatch):
    """C2 de la auditoría, por el camino real: ToolAgent + StreamSpeaker + loop
    simple. El motor suelta media respuesta y se cae: el error en personaje se
    tiene que OÍR (antes: frase truncada y silencio; solo salía impreso), y una
    sola vez (el fallback `if not spoke` no debe repetirlo)."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.llm import TransientLLMError
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry

    error = "GLM no respondió a tiempo, patrón."

    class _SeCaeAMedias:
        model = "x"

        def chat_stream(self, messages, tools=None, on_token=None):
            on_token("Claro, patrón. ")
            on_token("La capital de Francia es ")
            raise TransientLLMError(error)

    agent = ToolAgent(_SeCaeAMedias(), Conversation("SYS"), registry=Registry(),
                      guard=Guard(allowed_roots=[]), fastpath=False)
    rc, eventos = _correr_turno(monkeypatch, agent, stream=True)
    assert rc == 0
    dicho = [e[1] for e in eventos if e[0] in ("habla", "frases")]
    assert dicho[0] == "Claro, patrón."
    assert sum(error in d for d in dicho) == 1, dicho


# --- V9: el aviso de una tool delicada (concurrente) se dice en SPEAKING ------------
# El confirm_fn del loop concurrente avisa DESDE el Cerebro, dentro de
# agent.handle_turn. Con el modo en THINKING el HUD decía "pensando" mientras
# sonaba y, con --barge-in, el oído juzgaba el eco con las reglas de THINKING
# (sin gracia ni margen de umbral): el eco del propio aviso lo cortaba y abría un
# turno fantasma.
_AVISO_V9 = "Voy a borrar la carpeta Descargas."
_RESPUESTA_V9 = "Cancelado, patrón."


def _modo(state) -> str:
    # El nombre que publica el HUD ("idle", "speaking"...).
    return state.get_mode().name.lower()


class _MicConEco:
    """Micrófono falso: wake + orden + silencio. Después capta la voz del patrón
    si está hablando y, si no, el ECO del TTS mientras suena (prob. VAD 0.85: por
    encima del umbral de THINKING, 0.8, y por debajo del de SPEAKING, 0.8 + 0.1)."""

    def __init__(self, sonando: threading.Event, patron: threading.Event):
        self.sonando = sonando
        self.patron = patron
        self.guion = iter(["wake"] + ["voz"] * 8 + ["sil"] * 30)

    def read(self):
        time.sleep(0.002)
        try:
            return next(self.guion)
        except StopIteration:
            if self.patron.is_set():
                return "voz"
            return "eco" if self.sonando.is_set() else "sil"


class _TtsConEco(_FakeTts):
    """TTS falso: anota (texto, modo del loop, hilo) y "suena" 0.6 s (más que la
    gracia de 400 ms del barge-in), interrumpible por stop() como el real."""

    def __init__(self):
        self.sonando = threading.Event()
        self.corte = threading.Event()
        self.dichos: list[tuple[str, str | None, str]] = []
        self.cortados: list[str] = []
        self.loop = None

    def speak(self, text, should_abort=None):
        modo = _modo(self.loop.state) if self.loop is not None else None
        self.dichos.append((text, modo, threading.current_thread().name))
        self.corte.clear()
        self.sonando.set()
        cortado = self.corte.wait(0.6)
        self.sonando.clear()
        if cortado:
            self.cortados.append(text)

    def stop(self):
        self.corte.set()


def _correr_concurrente(monkeypatch, tmp_path, argv, handle_turn, terminado):
    """run_listen concurrente con el VoiceLoop REAL sobre fakes (mic, VAD, STT,
    TTS). `handle_turn(agente, comando)` hace de ToolAgent: recibe el confirm_fn
    que run_listen le pasó a build_agent. Corre hasta `terminado(tts, loop)`."""
    from crotolamo import settings as crotolamo_settings
    from crotolamo.voice import loop as voice_loop

    _arnes_run_listen(monkeypatch, tmp_path, _BlockingDetector())
    monkeypatch.setattr(crotolamo_settings, "get_settings", lambda: SimpleNamespace(voice={
        "vad_threshold": 0.8, "barge_in_grace_ms": 400,
        "barge_in_threshold_margin": 0.1, "barge_in_min_chunks": 5,
    }))
    tts = _TtsConEco()
    patron = threading.Event()
    mic = _MicConEco(tts.sonando, patron)
    monkeypatch.setattr(listener, "TTS", SimpleNamespace(from_settings=lambda s: tts))
    ordenes = ["borra la carpeta descargas"]

    class _Stt(_FakeSttCls):
        def transcribe(self, audio, hotwords=None):
            return ordenes.pop(0) if ordenes else ""

    monkeypatch.setattr(listener, "STT", _Stt)

    loops: list = []

    class _Agente:
        def __init__(self, confirm_fn):
            self.confirm_fn = confirm_fn
            self.confirmaciones: list[bool] = []
            self.patron = patron
            self.loops = loops

        def handle_turn(self, command, on_token=None):
            return handle_turn(self, command)

    agentes: list = []

    def _build_agent(confirm_fn=None):
        agentes.append(_Agente(confirm_fn))
        return agentes[-1], None

    monkeypatch.setattr(listener, "build_agent", _build_agent)
    modos: list[str] = []
    real_loop = voice_loop.VoiceLoop

    def _loop(agent, stt, tts_, wake_detector, **kwargs):
        publicar = kwargs["hud_publisher"]

        def _anota(state_dict):
            modos.append(state_dict["mode"])
            publicar(state_dict)

        kwargs["hud_publisher"] = _anota
        vl = real_loop(agent, stt, tts_, wake_detector, mic=mic,
                       wake_fn=lambda chunk: chunk == "wake",
                       vad_fn=lambda chunk: {"voz": 0.95, "eco": 0.85}.get(chunk, 0.0),
                       to_wav=lambda frames: frames, **kwargs)
        tts.loop = vl
        loops.append(vl)
        return vl

    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop)
    rc: list[int] = []
    hilo = threading.Thread(target=lambda: rc.append(listener.run_listen(argv)), daemon=True)
    hilo.start()
    try:
        assert _wait_until(lambda: bool(loops), timeout=10.0), "el loop no arrancó"
        listo = _wait_until(lambda: terminado(tts, loops[0]), timeout=15.0)
    finally:
        if loops:
            loops[0].shutdown.set()
        hilo.join(timeout=10.0)
        _join_warmers()
    assert listo, f"el turno no terminó: {tts.dichos} / {modos}"
    assert rc == [0]
    # Sin repeticiones seguidas (set_mode solo publica si cambia; new_turn sí).
    secuencia = [m for i, m in enumerate(modos) if i == 0 or m != modos[i - 1]]
    return tts, secuencia, loops[0], agentes[0]


def _respondio(tts, vl):
    return (any(texto == _RESPUESTA_V9 for texto, _, _ in tts.dichos)
            and _modo(vl.state) == "idle")


@pytest.mark.parametrize("argv", [[], ["--barge-in"]])
def test_concurrente_aviso_de_tool_delicada_suena_en_speaking(monkeypatch, tmp_path, argv):
    """El aviso suena con el modo en SPEAKING (HUD "hablando"; el oído aplica la
    gracia y el margen de umbral al eco) y al terminar se vuelve a THINKING: el
    LLM sigue con el turno. Con --barge-in, el eco del aviso ya no lo corta ni
    abre un turno fantasma."""

    def handle_turn(agente, command):
        agente.confirmaciones.append(agente.confirm_fn(_AVISO_V9))  # ToolAgent._execute_call
        return _RESPUESTA_V9

    tts, secuencia, vl, agente = _correr_concurrente(
        monkeypatch, tmp_path, argv, handle_turn, _respondio)
    assert agente.confirmaciones == [False]  # se sigue denegando, como antes
    assert [modo for texto, modo, _ in tts.dichos if _AVISO_V9 in texto] == ["speaking"]
    assert tts.cortados == [], "el eco del propio aviso lo cortó"
    assert vl.state.turn_id == 1, "el eco del aviso abrió un turno fantasma"
    assert (_RESPUESTA_V9, "speaking", "Mouth") in tts.dichos
    assert secuencia == ["idle", "listening", "thinking", "speaking", "thinking",
                         "speaking", "idle"]


def test_concurrente_aviso_de_un_turno_abortado_no_suena_sobre_la_orden_nueva(
        monkeypatch, tmp_path):
    """Barge-in mientras el LLM pensaba: el patrón ya dicta otra orden (LISTENING)
    cuando llega el aviso del turno viejo. Lo de un turno abortado se descarta
    (su respuesta tampoco se dice): hablar encima metía el aviso en la grabación
    de la orden nueva. La tool se sigue denegando."""

    def handle_turn(agente, command):
        agente.patron.set()  # el patrón habla encima: barge-in real del oído
        st = agente.loops[0].state
        assert _wait_until(lambda: st.turn_id == 2 and _modo(st) == "listening")
        agente.confirmaciones.append(agente.confirm_fn(_AVISO_V9))
        agente.modo_tras_aviso = _modo(st)
        agente.patron.clear()  # deja de hablar: la orden nueva se cierra
        return _RESPUESTA_V9

    tts, secuencia, vl, agente = _correr_concurrente(
        monkeypatch, tmp_path, ["--barge-in"], handle_turn,
        lambda tts, vl: vl.state.turn_id == 2 and _modo(vl.state) == "idle")
    assert agente.confirmaciones == [False]
    assert [t for t, _, _ in tts.dichos if _AVISO_V9 in t] == []
    assert agente.modo_tras_aviso == "listening"  # el aviso no pisó la orden nueva
    assert "speaking" not in secuencia


# --- C5: la respuesta a "confirmo o cancela" la transcribe el modelo de la orden ----
# voice_confirm (modo simple) transcribía la respuesta con el 'tiny' del wake, el
# modelo "solo para la palabra de activación": un mishear vale "no" (lo seguro) y
# el patrón tenía que repetir la orden entera.
def _clase_stt_por_rol(instancias: list, respuesta: str):
    """STT falso que distingue el de la orden (from_settings, con pista) del del
    wake (initial_prompt=None) y anota qué se le pide a cada uno."""

    class _SttPorRol:
        def __init__(self, model_size="small", initial_prompt="pista", **kwargs):
            self.model_size = model_size
            self.rol = "wake" if initial_prompt is None else "orden"
            self.escuchas: list[dict] = []
            instancias.append(self)

        @classmethod
        def from_settings(cls, settings):
            return cls(model_size="base")

        def _get_model(self):
            return None

        def listen_once(self, **kwargs):
            self.escuchas.append(kwargs)
            return respuesta

    return _SttPorRol


def _confirmar_por_voz(monkeypatch, tmp_path, respuesta: str):
    """run_listen --simple (openWakeWord esperando el wake) y, con el loop vivo,
    el confirm_fn que recibió build_agent: lo que llamaría ToolAgent._execute_call."""
    detector = _BlockingDetector()
    _arnes_run_listen(monkeypatch, tmp_path, detector)
    instancias: list = []
    monkeypatch.setattr(listener, "STT", _clase_stt_por_rol(instancias, respuesta))
    confirms: list = []

    def _build_agent(confirm_fn=None):
        confirms.append(confirm_fn)
        return _FakeAgent(), None

    monkeypatch.setattr(listener, "build_agent", _build_agent)
    rc: list[int] = []
    hilo = threading.Thread(target=lambda: rc.append(listener.run_listen(["--simple"])),
                            daemon=True)
    hilo.start()
    try:
        assert detector.esperando.wait(10), "el loop no llegó a esperar el wake"
        confirmado = confirms[0]("Voy a borrar el archivo.")
    finally:
        detector.soltar.set()
        hilo.join(timeout=10.0)
        _join_warmers()
    assert rc == [0]
    roles = {s.rol: s for s in instancias}
    return confirmado, roles["orden"], roles["wake"]


def test_confirmacion_por_voz_la_transcribe_el_modelo_de_la_orden(monkeypatch, tmp_path):
    confirmado, orden, wake_stt = _confirmar_por_voz(monkeypatch, tmp_path, "Sí, confirmo.")
    assert confirmado is True
    assert wake_stt.escuchas == [], "la respuesta la transcribió el modelo del wake"
    assert len(orden.escuchas) == 1
    pedido = orden.escuchas[0]
    # Mismas ventanas que antes (respuesta de una palabra) y SIN hotwords: no es
    # un comando. min_audio_s: si nadie contestó, "" sin pasar por Whisper.
    assert pedido == {"silence_ms": 400, "max_seconds": 5, "min_audio_s": 0.3}


@pytest.mark.parametrize("respuesta", ["", "cancela", "¿se va a borrar todo?"])
def test_confirmacion_por_voz_sin_respuesta_clara_sigue_siendo_no(
        monkeypatch, tmp_path, respuesta):
    confirmado, orden, wake_stt = _confirmar_por_voz(monkeypatch, tmp_path, respuesta)
    assert confirmado is False
    assert wake_stt.escuchas == [] and len(orden.escuchas) == 1
