"""Integración de la auditoría: cambios que un grupo pidió en archivos de otro.

Cada bloque cita el pedido (idx de la lista de integración y el id del hallazgo)
y prueba el comportamiento que el arreglo cambia; todos fallaban antes.
"""

from __future__ import annotations

import builtins
import http.client
import logging
import re
import runpy
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.agent import ToolAgent
from crotolamo.core.llm import TransientLLMError
from crotolamo.core.memoria import Memoria, MemoriaConfig
from crotolamo.tools import _web, media, search
from crotolamo.tools.base import Registry, Tool
from crotolamo.voice import state as voice_state
from crotolamo.voice import wake
from crotolamo.voice.stt import VoiceUnavailable, normalize_text
from crotolamo.voice.tts import StreamSpeaker

from interfaces import listener

REPO = Path(__file__).resolve().parents[1]


# --- fakes para run_listen (sin modelos, sin audio, sin LLM) ------------------------
class _FakeStt:
    model_size = "tiny"

    def __init__(self, **kwargs):
        self.escuchas = 0

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def _get_model(self):
        return None

    def listen_once(self, **kwargs):
        # Agotada la "escucha", Ctrl-C: el loop simple sale con 0.
        self.escuchas += 1
        raise KeyboardInterrupt

    def listen_smart(self, **kwargs):
        return ""


class _FakeTts:
    @classmethod
    def from_settings(cls, settings):
        return cls()

    def available(self):
        return False

    def speak(self, text, **kwargs):
        pass

    def speak_sentences(self, text):
        pass

    def beep(self):
        pass

    def stop(self):
        pass


class _FakeBackend:
    def playing_players(self):
        return []

    def get_volume(self, player):
        return None

    def set_volume(self, player, value):
        return True


class _Detector:
    """openWakeWord falso con carga de modelo; listen_for_wake sale con Ctrl-C."""

    threshold = 0.5
    threshold_media = None

    def __init__(self, carga_falla: Exception | None = None):
        self.carga_falla = carga_falla
        self.cargas = 0
        self.esperas = 0

    def available(self):
        return True

    def attach_media(self, monitor):
        pass

    def _get_model(self):
        self.cargas += 1
        if self.carga_falla is not None:
            raise self.carga_falla
        return object()

    def can_load(self):
        # Como WakeWordDetector.can_load: la carga de _get_model, sin lanzar.
        try:
            self._get_model()
        except VoiceUnavailable as error:
            self.load_error = str(error)
            return False
        return True

    def listen_for_wake(self, timeout_s=None):
        self.esperas += 1
        raise KeyboardInterrupt

    def feed(self, chunk):
        return False


class _FakeAgent:
    def handle_turn(self, command, on_token=None):
        return "hecho, patrón."


def _arnes(monkeypatch, tmp_path, detector, confirm_fns=None, stt=None):
    """run_listen con todo lo pesado parcheado y el HUD/control en tmp_path."""
    control = tmp_path / "control.json"
    real_read = voice_state.read_control_enabled
    stt = stt or _FakeStt()
    monkeypatch.setattr(listener, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(listener, "get_settings",
                        lambda: SimpleNamespace(voice={"wake_duck": False}, wake={}))
    # STT.from_settings (la orden) y STT(...) (el wake): el mismo doble.
    monkeypatch.setattr(listener, "STT", _SttFactory(stt))
    monkeypatch.setattr(listener, "TTS", _FakeTts)
    monkeypatch.setattr(listener, "WakeWordDetector",
                        SimpleNamespace(from_settings=lambda settings: detector))
    monkeypatch.setattr(listener, "PlayerctlBackend", _FakeBackend)

    def _build_agent(confirm_fn=None):
        if confirm_fns is not None:
            confirm_fns.append(confirm_fn)
        return _FakeAgent(), None

    monkeypatch.setattr(listener, "build_agent", _build_agent)
    monkeypatch.setattr(listener, "_HUD_STATE_PATH", tmp_path / "hud_state.json")
    monkeypatch.setattr(listener, "CONTROL_PATH", control)
    monkeypatch.setattr(listener, "read_control_enabled",
                        lambda path=control: real_read(path))
    monkeypatch.setattr(listener, "_ACTIVE_DUCKER", None)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", None, raising=False)
    from crotolamo.core import engine

    monkeypatch.setattr(engine, "build_primary_llm",
                        lambda settings: SimpleNamespace(chat=lambda messages: "ok"))
    return stt


class _SttFactory:
    """listener.STT: `STT.from_settings(...)` (la orden) y `STT(...)` (el wake)
    devuelven el mismo doble, para poder contar sus escuchas."""

    model_size = "tiny"

    def __init__(self, stt):
        self._stt = stt

    def from_settings(self, settings):
        return self._stt

    def __call__(self, **kwargs):
        return self._stt


def _join_warmers():
    for hilo in threading.enumerate():
        if hilo.name in ("WarmVoice", "WarmLLM"):
            hilo.join(timeout=5.0)


# --- lead -1: el código de salida de VoiceLoop.run() llega al proceso ---------------
# VoiceLoop.run() devuelve != 0 cuando un hilo del loop murió (el oído que se rinde
# ante un micro que no vuelve). run_listen devolvía 0 siempre en el modo
# concurrente: el servicio quedaba "inactive" y Restart=on-failure nunca actuaba.

def _loop_que_devuelve(codigo):
    class _Loop:
        def __init__(self, agent, stt, tts, wake_detector, **kwargs):
            self.state = None

        def run(self):
            return codigo

    return _Loop


def test_run_listen_devuelve_el_codigo_del_voice_loop(monkeypatch, tmp_path):
    from crotolamo.voice import loop as voice_loop

    _arnes(monkeypatch, tmp_path, _Detector())
    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop_que_devuelve(3))
    try:
        assert listener.run_listen([]) == 3
    finally:
        _join_warmers()


def test_main_listen_sale_con_el_codigo_del_voice_loop(monkeypatch, tmp_path):
    from crotolamo import __main__ as main_mod
    from crotolamo import logging_setup
    from crotolamo.voice import loop as voice_loop

    _arnes(monkeypatch, tmp_path, _Detector())
    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop_que_devuelve(3))
    # Sin tocar el logging raíz del proceso de pytest.
    monkeypatch.setattr(logging_setup, "setup_logging", lambda level=None: None)
    try:
        # `python -m crotolamo listen` hace `raise SystemExit(main())`.
        assert main_mod.main(["listen"]) == 3
    finally:
        _join_warmers()


def test_run_listen_apagado_normal_sigue_saliendo_con_cero(monkeypatch, tmp_path):
    from crotolamo.voice import loop as voice_loop

    _arnes(monkeypatch, tmp_path, _Detector())
    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop_que_devuelve(0))
    try:
        assert listener.run_listen([]) == 0
    finally:
        _join_warmers()


# --- idx 63 (voice-capture#18): un oww_model que no carga no deja sordo al loop ------
# El modelo de openWakeWord se cargaba en el primer chunk, DENTRO del hilo del
# oído: con un oww_model malo el asistente decía "Crotolamo escuchando" y el
# EarThread moría en su primer chunk. Ahora se carga antes de elegir el loop y,
# si no carga, se cae al wake difuso (Whisper), que no lo necesita.

def test_si_openwakeword_no_carga_se_usa_el_wake_difuso(monkeypatch, tmp_path, capsys):
    from crotolamo.voice import loop as voice_loop

    class _LoopSordo:
        def __init__(self, *a, **kw):
            raise AssertionError("eligió el loop concurrente sin modelo de wake")

    detector = _Detector(carga_falla=VoiceUnavailable("No pude cargar el wake word 'x'"))
    confirm_fns: list = []
    stt = _arnes(monkeypatch, tmp_path, detector, confirm_fns=confirm_fns)
    monkeypatch.setattr(voice_loop, "VoiceLoop", _LoopSordo)
    try:
        assert listener.run_listen([]) == 0
    finally:
        _join_warmers()
    assert detector.cargas == 1
    assert detector.esperas == 0, "siguió esperando a openWakeWord"
    assert stt.escuchas == 1, "el wake difuso (Whisper) no llegó a escuchar"
    # Modo simple: la confirmación por voz, no la negación del concurrente.
    assert confirm_fns[0].__name__ == "voice_confirm"
    assert "No pude cargar el wake word" in capsys.readouterr().out


def test_si_openwakeword_carga_el_loop_concurrente_sigue_igual(monkeypatch, tmp_path):
    from crotolamo.voice import loop as voice_loop

    detector = _Detector()
    _arnes(monkeypatch, tmp_path, detector)
    monkeypatch.setattr(voice_loop, "VoiceLoop", _loop_que_devuelve(0))
    try:
        assert listener.run_listen([]) == 0
    finally:
        _join_warmers()
    assert detector.cargas == 1


# --- idx 61 (entry-scripts-ci#9): `python interfaces/listener.py --simple` -----------
# El bloque __main__ llamaba run_listen() a secas: --simple y --barge-in se
# ignoraban, y sin setup_logging los INFO se perdían.

def test_listener_como_script_respeta_sus_argumentos_y_configura_el_logging(
        monkeypatch, tmp_path):
    from crotolamo import logging_setup
    from crotolamo.voice import media_aware, stt as stt_mod, tts as tts_mod, wakeword
    import interfaces.shell as shell_mod

    confirm_fns: list = []
    logging_llamado: list[int] = []
    # El script corre como módulo NUEVO: los dobles entran por donde los importa.
    monkeypatch.setattr(signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(voice_state, "HUD_STATE_PATH", tmp_path / "hud_state.json")
    # Sin ducking y con el mismo umbral con música: no arranca el sondeo de fondo
    # (run_listen sale antes de llegar a pararlo).
    monkeypatch.setattr(settings_mod, "get_settings",
                        lambda: SimpleNamespace(voice={"wake_duck": False,
                                                       "wake_threshold_media": 0.67},
                                                wake={}))
    monkeypatch.setattr(stt_mod, "STT", _SttFactory(_FakeStt()))
    monkeypatch.setattr(tts_mod, "TTS", _FakeTts)
    monkeypatch.setattr(wakeword, "WakeWordDetector",
                        SimpleNamespace(from_settings=lambda settings: _Detector()))
    monkeypatch.setattr(media_aware, "PlayerctlBackend", _FakeBackend)

    def _build_agent(confirm_fn=None):
        confirm_fns.append(confirm_fn)
        raise RuntimeError("basta: ya se vio con qué modo arrancó")

    monkeypatch.setattr(shell_mod, "build_agent", _build_agent)
    monkeypatch.setattr(logging_setup, "setup_logging",
                        lambda level=None: logging_llamado.append(1))
    monkeypatch.setattr(sys, "argv", ["interfaces/listener.py", "--simple"])
    try:
        with pytest.raises(SystemExit) as fin:
            runpy.run_path(str(REPO / "interfaces" / "listener.py"), run_name="__main__")
    finally:
        _join_warmers()
    assert fin.value.code == 1  # build_agent "falló": No pude armar el agente
    # Con --simple el agente se arma con la confirmación POR VOZ; sin argumentos
    # (lo que pasaba) sería el aviso del loop concurrente.
    assert confirm_fns and confirm_fns[0].__name__ == "voice_confirm"
    assert logging_llamado == [1]


# --- idx 38 / 59 (x-concurrency-lifecycle#22, memoria-persistence#15) ---------------
# El apagado por señal acaba en os._exit, que se salta el vaciado atexit de la
# Memoria: el hecho que el patrón contó en el último turno se perdía al parar
# el servicio.

class _Salida(BaseException):
    """Sustituto de os._exit en los tests del manejador de señal."""


class _BackendLento:
    def __init__(self):
        self.extraidos: list[tuple[str, str]] = []

    def extraer(self, usuario, respuesta):
        time.sleep(0.2)  # una extracción de verdad pasa por el LLM
        self.extraidos.append((usuario, respuesta))
        return [usuario]


def _parchear_salida(monkeypatch, tmp_path, orden):
    def _exit(code):
        orden.append("exit")
        raise _Salida(code)

    monkeypatch.setattr(listener.os, "_exit", _exit)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: None)
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: orden.append("mcp"))
    monkeypatch.setattr(listener, "_final_idle_best_effort", lambda: orden.append("idle"))


def test_apagado_por_senal_guarda_la_extraccion_pendiente(monkeypatch, tmp_path):
    orden: list[str] = []
    _parchear_salida(monkeypatch, tmp_path, orden)
    backend = _BackendLento()
    memoria = Memoria(MemoriaConfig(enabled=True), backend=backend)
    monkeypatch.setattr(memoria_mod, "_MEMORIA", memoria)
    assert memoria.extraer_en_fondo("mi hermana Ana vive en Monterrey", "Anotado, patrón.")
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    assert backend.extraidos == [("mi hermana Ana vive en Monterrey", "Anotado, patrón.")]
    # El idle sigue siendo lo último antes de salir.
    assert orden == ["mcp", "idle", "exit"]


def test_apagado_por_senal_no_construye_una_memoria(monkeypatch, tmp_path):
    orden: list[str] = []
    _parchear_salida(monkeypatch, tmp_path, orden)
    monkeypatch.setattr(memoria_mod, "_MEMORIA", None)

    def no_debe_llamarse():
        raise AssertionError("el apagado no debe armar una Memoria que nadie usó")

    monkeypatch.setattr(memoria_mod, "get_memoria", no_debe_llamarse)
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    assert orden == ["mcp", "idle", "exit"]


def test_apagado_por_senal_no_espera_de_mas_a_la_memoria(monkeypatch, tmp_path):
    """El manejador corre en el hilo principal, que pudo quedar interrumpido
    DENTRO de un put() a la cola de la Memoria (con su candado tomado): la espera
    tiene que estar acotada aunque flush() no vuelva nunca."""
    orden: list[str] = []
    _parchear_salida(monkeypatch, tmp_path, orden)
    soltar = threading.Event()
    monkeypatch.setattr(memoria_mod, "_MEMORIA",
                        SimpleNamespace(flush=lambda timeout_s=5.0: soltar.wait(60)))
    t0 = time.monotonic()
    try:
        with pytest.raises(_Salida):
            listener._graceful_exit(signal.SIGTERM, None)
    finally:
        soltar.set()
    assert time.monotonic() - t0 < 5.0
    assert orden == ["mcp", "idle", "exit"]


# --- idx 49 (desktop#18): una sola definición de la ruta del HUD en crotolamo/ -------

def test_la_ruta_del_hud_vive_en_voice_state():
    assert voice_state.HUD_STATE_PATH == Path.home() / ".crotolamo" / "hud_state.json"
    assert listener._HUD_STATE_PATH == voice_state.HUD_STATE_PATH


# --- lead -2 / idx 40 (voice-runtime#11): el aviso del LLM caído, frase aparte -----
# El agente emite el error de un LLM que se cae a media respuesta como su último
# token, on_token("\n" + aviso). StreamSpeaker no corta en un salto de línea
# suelto (cómo suenan los saltos EN GENERAL es decisión del dueño, voice-runtime#18,
# y no se toca), así que el aviso se hablaba pegado a la frase cortada: "La capital
# de Francia es GLM no respondió...". Lo arregla el loop simple, que es quien cablea
# el agente al StreamSpeaker; la segmentación de cualquier otra respuesta no cambia.

class _TtsDeTurno:
    """TTS falso del loop simple: anota lo hablado (StreamSpeaker y fallback)."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, **kwargs):
        self.spoken.append(text)

    def speak_sentences(self, text):
        self.spoken.append(text)

    def beep(self):
        pass

    def stop(self):
        pass


class _WakeUnaVez:
    """wake_stt falso: un "crotolamo" y, después, Ctrl-C (el loop sale con 0)."""

    def __init__(self):
        self.heard = ["crotolamo"]

    def listen_once(self, **kwargs):
        if self.heard:
            return self.heard.pop(0)
        raise KeyboardInterrupt


class _OrdenUnaVez:
    def listen_once(self, **kwargs):
        return "cuál es la capital de francia"


def _hablar_un_turno(monkeypatch, agent) -> list[str]:
    """Un turno del loop simple con stream_speak y el StreamSpeaker REAL."""
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time",
                        SimpleNamespace(sleep=lambda s: None, time=time.time))
    tts = _TtsDeTurno()
    cfg = listener.ListenerConfig(threshold=0.72, variants=["crotolamo"], followup_s=0,
                                  ack="off", stream_speak=True, smart_endpoint=False)
    rc = listener._run_simple_loop(
        agent=agent, stt=_OrdenUnaVez(), wake_stt=_WakeUnaVez(), tts=tts,
        wake_detector=None, use_oww=False, cfg=cfg, hud_state=voice_state.SharedState(),
        say=lambda text: None,
    )
    assert rc == 0
    return tts.spoken


class _LlmQueSeCaeAMedias:
    model = "x"

    def __init__(self, tokens, error):
        self.tokens = tokens
        self.error = error

    def chat_stream(self, messages, tools=None, on_token=None):
        for token in self.tokens:
            on_token(token)
        raise TransientLLMError(self.error)


def _agente_real(llm):
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard

    return ToolAgent(llm, Conversation("SYS"), registry=Registry(),
                     guard=Guard(allowed_roots=[]), fastpath=False)


@pytest.mark.parametrize("corte", ["La capital de Francia es", "La capital de Francia es "])
def test_el_aviso_del_llm_caido_se_habla_aparte_de_la_frase_cortada(monkeypatch, corte):
    llm = _LlmQueSeCaeAMedias(["Claro, patrón. ", corte],
                              "GLM no respondió a tiempo, patrón. (timed out)")
    dicho = _hablar_un_turno(monkeypatch, _agente_real(llm))
    assert dicho == ["Claro, patrón.", "La capital de Francia es",
                     "GLM no respondió a tiempo, patrón.", "(timed out)"]


def test_el_aviso_con_frases_dentro_tampoco_se_pega(monkeypatch):
    # El ". " DENTRO del aviso cerraba frase al alimentarlo: la frase cortada y
    # el principio del aviso ya iban juntos a la cola antes de acabar el turno.
    llm = _LlmQueSeCaeAMedias(
        ["La capital", " de Francia es"],
        "GLM me cortó la respuesta a medias, patrón. (IncompleteRead(0 bytes read))")
    dicho = _hablar_un_turno(monkeypatch, _agente_real(llm))
    assert dicho == ["La capital de Francia es",
                     "GLM me cortó la respuesta a medias, patrón.",
                     "(IncompleteRead(0 bytes read))"]


class _AgenteQueStreamea:
    """Agente falso: suelta `tokens` por on_token y devuelve su unión."""

    def __init__(self, tokens):
        self.tokens = tokens

    def handle_turn(self, command, on_token=None):
        for token in self.tokens:
            on_token(token)
        return "".join(self.tokens)


@pytest.mark.parametrize("tokens", [
    ["Tienes tres pendientes:", "\nComprar leche", "\nLlamar a mamá", "\nPagar la luz"],
    ["Pasos:", "\n", "1. Abre", " Spotify.", "\n", "2. Pon", " música."],
    ["Hola, patrón.", "\nAdiós."],
    ["Mira:", "\n\n", "Primero esto.", " Luego", "\n"],
    ["\nEmpieza con salto."],
])
def test_una_respuesta_normal_suena_igual_que_antes(monkeypatch, tokens):
    # Referencia: los mismos tokens directo al StreamSpeaker, como antes.
    directo = _TtsDeTurno()
    speaker = StreamSpeaker(directo)
    for token in tokens:
        speaker.feed(token)
    speaker.finish()
    assert _hablar_un_turno(monkeypatch, _AgenteQueStreamea(tokens)) == directo.spoken


# --- idx 13 / 14 / 39 (tools-net-desktop#1, #13; x-tool-surface-wiring#14) ----------
# Sin DNS (offline o dominio mal escrito) read_page decía "esa dirección es de la
# red interna": su pre-check is_public_url trataba "no resuelve" como "interna".

def test_read_page_sin_dns_dice_que_no_pudo_conectar(monkeypatch):
    def sin_dns(*a, **kw):
        raise socket.gaierror(-3, "Temporary failure in name resolution")

    monkeypatch.setattr(_web.socket, "getaddrinfo", sin_dns)
    out = search.read_page("https://es.wikipedia.org/wiki/Ajolote")
    assert "No pude conectarme" in out
    assert "red interna" not in out


def test_read_page_resuelve_el_dns_una_sola_vez(monkeypatch):
    resoluciones: list[str] = []

    def resolver(host, *a, **kw):
        resoluciones.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    def sin_red(*a, **kw):
        raise OSError("sin red en los tests")

    monkeypatch.setattr(_web.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(_web.socket, "create_connection", sin_red)
    out = search.read_page("https://ejemplo.com/pagina")
    assert "No pude conectarme" in out
    assert resoluciones == ["ejemplo.com"]


# --- idx 19 (tools-net-desktop#12): read_page no parsea más de lo que conserva -------

def test_read_page_pide_el_extractor_con_tope(monkeypatch):
    topes: list = []

    class _Extractor(_web._PageTextExtractor):
        def __init__(self, max_chars=None):
            topes.append(max_chars)
            super().__init__(max_chars=max_chars)

    html = "<html><body>" + "<p>Una frase de relleno para la página.</p>" * 400 + "</body></html>"
    monkeypatch.setattr(_web, "_http_get", lambda url, timeout=10.0: ("text/html", html))
    monkeypatch.setattr(_web, "_PageTextExtractor", _Extractor)
    out = search.read_page("https://ejemplo.com/larga")
    assert topes == [search._MAX_PAGE_CHARS]
    assert out.endswith("[... texto truncado, la página sigue, patrón]")


# --- idx 24 (tools-fs-safety#8): un host con caracteres de control no es "reventó" ---

def test_http_get_con_un_nul_en_el_host_es_un_fallo_de_conexion(monkeypatch):
    monkeypatch.setattr(
        _web.socket, "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))])

    def no_debe_conectar(*a, **kw):
        raise AssertionError("no debe llegar a conectar")

    monkeypatch.setattr(_web.socket, "create_connection", no_debe_conectar)
    with pytest.raises(_web.urllib.error.URLError) as error:
        _web._http_get("http://8.8.8.8\x00/", timeout=1.0)
    assert isinstance(error.value.__cause__, http.client.InvalidURL)
    from crotolamo.tools import default_registry

    out = default_registry().run("read_page", {"url": "http://8.8.8.8\x00/"})
    assert "reventó" not in out
    assert "No pude conectarme" in out


# --- idx 43 (mcp#11): Registry.schemas() aguanta que el bridge quite una tool --------
# El bridge MCP retira tools desde un hilo de fondo cuando un server muere solo; un
# pop en plena comprensión de schemas() lanzaba "dictionary changed size during
# iteration" en el turno de GLM.

def test_schemas_aguanta_que_se_quite_una_tool_a_media_iteracion():
    reg = Registry()

    class _ToolQueRetiraOtra(Tool):
        def schema(self):
            reg.unregister("b")  # lo que haría el bridge desde su hilo
            return super().schema()

    vacio = {"type": "object", "properties": {}}
    reg.register(_ToolQueRetiraOtra("a", lambda: "a", "A", vacio))
    reg.register(Tool("b", lambda: "b", "B", vacio))
    nombres = [s["function"]["name"] for s in reg.schemas()]
    assert nombres == ["a", "b"]
    assert reg.names() == ["a"]


# --- idx 65 (x-tool-surface-wiring#28): music_control(player=null) ------------------

def test_music_control_con_player_null_actua_sobre_el_activo(monkeypatch):
    from crotolamo.tools import default_registry

    ordenes: list[list[str]] = []

    def _run(args):
        ordenes.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/playerctl")
    monkeypatch.setattr(media, "_run", _run)
    monkeypatch.setattr(media, "_players", lambda: [("spotify", "playing", "A — B")])
    out = default_registry().run("music_control", {"action": "pause", "player": None})
    assert out == "Pausado, patrón."
    assert ordenes == [["-p", "spotify", "pause"]]


# --- idx 41 (voice-runtime#4): la confirmación por voz exige un sí claro ------------
# Frases largas con el "si" condicional o con un "va" dentro confirmaban la tool
# delicada (borrar, mover...). Un "si" sin tilde pegado a otra palabra es la
# conjunción, y el sí se dice primero (dentro de las dos primeras palabras).

@pytest.mark.parametrize("answer", [
    "mmm, si eso borra todo mejor espera",
    "para, para, eso va mal",
    "y si llueve mañana, el partido se va a suspender",
    "el paquete va en camino",
    "si llueve mañana",
    "a ver si",
])
def test_frases_con_si_o_va_dentro_no_confirman(answer):
    assert wake.confirmation_from_answer(normalize_text(answer)) is False, answer


@pytest.mark.parametrize("answer", [
    "Sí.", "si", "confirmo", "hazlo ya", "va que va", "correcto", "bueno, dale",
    "Va.", "Va, hazlo.", "Sí, va a quedar bien.", "¡Sí, confirmo!",
    "Si, confirmo", "Sí, dale", "órale, confirmo", "sí por favor",
])
def test_los_si_claros_siguen_confirmando(answer):
    assert wake.confirmation_from_answer(normalize_text(answer)) is True, answer


# --- idx 16 (tools-fs-safety#25): la suite no corre contra el local.toml del dueño ---

def test_la_suite_no_fusiona_el_local_toml_del_repo():
    assert settings_mod.LOCAL_CONFIG != settings_mod.CONFIG_DIR / "crotolamo.local.toml"
    assert not settings_mod.LOCAL_CONFIG.exists()


# --- idx 7 (memoria-persistence#4): el doctor dice lo que hará la memoria al arrancar --
# Con llm_provider = "ollama" mem0 necesita el paquete `ollama` (sin él no arranca y
# la memoria ENTERA queda fuera) y el doctor no lo miraba. Sin API key para openai o
# groq la memoria arranca pero sin extracción, y el doctor daba un ✅ liso.

_ENVS_KEY_MEMORIA = ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY",
                     "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY")


def _check_memoria_del_doctor(monkeypatch, tmp_path, faltan=(), env=None, llm=None,
                              **memoria):
    from scripts import crotolamo_doctor as doctor

    for nombre in _ENVS_KEY_MEMORIA:
        monkeypatch.delenv(nombre, raising=False)
    for nombre, valor in (env or {}).items():
        monkeypatch.setenv(nombre, valor)
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    # mem0 & cía no están en el venv de la suite: se dan por importables salvo `faltan`.
    monkeypatch.setattr(doctor, "_import_error",
                        lambda m: f"ModuleNotFoundError: {m}" if m in faltan else None)
    raw = {"memoria": {"enabled": True, "ruta": str(tmp_path / "mem"), **memoria}}
    if llm is not None:
        raw["llm"] = llm
    settings = settings_mod.Settings(raw=raw, user="test", home=tmp_path,
                                     paths={"home": tmp_path})
    ctx = doctor._Ctx(settings=settings, doctor={}, backend="glm", today=date(2026, 9, 30))
    [check] = doctor._check_memoria(ctx)
    return check


def test_doctor_memoria_con_ollama_sin_su_paquete_es_rojo(monkeypatch, tmp_path):
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, faltan=("ollama",),
                                      llm_provider="ollama")
    assert not check.ok and "ollama" in check.detail
    assert "pip install ollama" in check.fix


def test_doctor_memoria_con_ollama_instalado_no_pide_key(monkeypatch, tmp_path):
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, llm_provider="ollama")
    assert check.ok and not check.warn


@pytest.mark.parametrize("provider", ["auto", "openai", "groq"])
def test_doctor_memoria_sin_key_avisa_que_no_extrae(monkeypatch, tmp_path, provider):
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, llm_provider=provider)
    assert check.ok and check.warn
    assert "sin API key: la extracción automática queda apagada" in check.detail
    assert "CROTOLAMO_GLM_API_KEY" in check.fix


@pytest.mark.parametrize("env, llm", [
    ({"CROTOLAMO_GLM_API_KEY": "clave-de-zai"}, None),
    # El SDK toma OPENAI_API_KEY solo y la manda al base_url de la nube: con la de
    # OpenAI el runtime extrae (con la de Z.ai no: ver test_revision_final_u09.py).
    ({"OPENAI_API_KEY": "clave-del-sdk"}, {"glm": {"base_url": "https://api.openai.com/v1"}}),
])
def test_doctor_memoria_con_key_es_un_ok_liso(monkeypatch, tmp_path, env, llm):
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, env=env, llm=llm)
    assert check.ok and not check.warn and "sin API key" not in check.detail


def test_doctor_memoria_sin_key_y_con_extraer_false_no_avisa(monkeypatch, tmp_path):
    # Sin extracción configurada no hay nada que se apague: el runtime tampoco avisa.
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, extraer=False)
    assert check.ok and not check.warn


# --- idx 21 (tools-fs-safety#2): un turno que revienta no mata el shell -------------
# run_shell llamaba a handle_turn sin try: una excepción imprevista (un hook, un
# print de un texto que no se puede codificar) cerraba la sesión entera, historial
# incluido. Ahora va al log con su traceback y el patrón oye una frase en personaje.

class _AgenteQueRevienta:
    def handle_turn(self, text, on_token=None, should_abort=None):
        if text == "rompe":
            raise RuntimeError("bug imprevisto")
        return f"eco: {text}"


class _ToolAgentQueRevienta(ToolAgent):
    def __init__(self):  # sin LLM ni registry: solo el contrato que usa el shell
        pass

    def handle_turn(self, text, on_token=None, should_abort=None):
        assert on_token is not None
        if text == "rompe":
            on_token("Empiezo a cont")
            raise RuntimeError("bug imprevisto")
        on_token(f"eco: {text}")
        return f"eco: {text}"


@pytest.mark.parametrize("argv, agente", [
    ([], _AgenteQueRevienta()),
    (["--stream"], _ToolAgentQueRevienta()),
])
def test_un_turno_que_revienta_no_cierra_el_shell(monkeypatch, capsys, caplog, argv, agente):
    import interfaces.shell as shell

    monkeypatch.setattr(shell, "build_agent", lambda confirm_fn=None: (agente, None))
    entradas = iter(["rompe", "sigue", "salir"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(entradas))

    with caplog.at_level(logging.ERROR, logger="crotolamo.shell"):
        assert shell.run_shell(argv) == 0
    out = capsys.readouterr().out
    assert "Se me trabaron los cables, patrón. Vuelve a intentarlo." in out
    assert "eco: sigue" in out  # el turno siguiente se atendió
    assert out.rstrip().endswith("Crotolamo apagado, patrón.")
    [registro] = [r for r in caplog.records if r.exc_info]
    assert "bug imprevisto" in str(registro.exc_info[1])


def test_ctrl_c_en_un_turno_sigue_sin_tragarse(monkeypatch):
    """El except es de Exception: un KeyboardInterrupt sigue saliendo como antes."""
    import interfaces.shell as shell

    class _Interrumpe:
        def handle_turn(self, text, on_token=None, should_abort=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(shell, "build_agent", lambda confirm_fn=None: (_Interrumpe(), None))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "hola")
    with pytest.raises(KeyboardInterrupt):
        shell.run_shell([])


# --- revisión final u18: el guard aprende el cwd del server MCP (Tool.path_base) ------
# El guard ya clasificaba las relativas de una tool MCP contra `tool.path_base`,
# pero nadie lo ponía: con un server de verdad seguía resolviendo contra el CWD.

def test_build_tools_lleva_el_cwd_del_server_a_path_base():
    from crotolamo.mcp.bridge import MCPServerConfig, _build_tools

    con_cwd = MCPServerConfig(name="s", command=["x"], cwd="/srv")
    [tool] = _build_tools(Registry(), con_cwd, [{"name": "leer"}])
    assert tool.path_base == Path("/srv")
    [tool] = _build_tools(Registry(), MCPServerConfig(name="s", command=["x"]), [{"name": "leer"}])
    assert tool.path_base is None
    # Una tool nativa no trae base: la suya es [paths].relative_base (o el CWD).
    assert Tool(name="t", func=lambda: "", description="d", parameters={}).path_base is None


def test_relativa_a_un_server_mcp_con_cwd_en_zona_de_confirmacion_pregunta(tmp_path, monkeypatch):
    """De punta a punta, con el fake server por stdio: Crotolamo corre desde su
    repo (zona libre, como el .service) y el server desde ~/proyectos/privado.
    path=".env" es el .env del SERVER: antes pasaba sin preguntar."""
    from crotolamo.core import router
    from crotolamo.core.memory import Conversation
    from crotolamo.mcp import bridge
    from crotolamo.safety.guard import Guard
    from crotolamo.settings import Settings

    home = (tmp_path / "home").resolve()
    libre = home / "Documentos"
    repo = libre / "crotolamo2"
    privado = home / "proyectos" / "privado"
    repo.mkdir(parents=True)
    privado.mkdir(parents=True)
    monkeypatch.chdir(repo)
    servers = {"git": {"command": [sys.executable, str(REPO / "tests" / "fake_mcp_server.py")],
                       "cwd": str(privado)}}
    settings = Settings(raw={"mcp": {"enabled": True, "timeout_s": 5, "startup_timeout_s": 5,
                                     "servers": servers}}, user="test", home=home)
    registry = Registry()
    bridge.close_all()
    try:
        bridge.register_mcp_tools(registry, settings)
        tool = registry.get("mcp_git_echo")  # solo lectura: sin confirmación de por sí
        assert tool is not None and tool.safe
        guard = Guard(allowed_roots=[libre], confirm_roots=[home])
        decision = guard.check(tool, {"text": "x", "path": ".env"})
        assert decision.allowed and decision.needs_confirmation, decision
        assert decision.reason == f"fuera de la zona libre: {privado / '.env'}. ¿Lo hago?"
        preguntas: list[str] = []
        agente = ToolAgent(llm=None, conversation=Conversation("SYS"), registry=registry,
                           guard=guard, confirm_fn=lambda r: preguntas.append(r) or False)
        assert agente._execute_call("mcp_git_echo", {"text": "x", "path": ".env"}) \
            == "Cancelado por el patrón."
        assert preguntas == [decision.reason]
    finally:
        bridge.close_all()
        for name in router.dynamic_groups():
            router.unregister_group(name)


# --- revisión final u23: re-anotar un hecho sin categoría no le quita la suya --------

def _hechos(dbp: Path) -> list[tuple[str, str]]:
    from crotolamo.persistence import facts

    return [(f["texto"], f["categoria"]) for f in facts.recall(db_path=dbp)]


def test_reanotar_un_hecho_sin_categoria_conserva_su_etiqueta(tmp_path):
    """remember_fact trae categoria="general" por defecto: el modelo que volvía a
    guardar '(preferencias) mi editor es geany' sin categoría lo dejaba en
    '(general)' y recall_facts('preferencias') ya no lo encontraba."""
    from crotolamo.persistence import facts

    dbp = tmp_path / "libreta.sqlite"
    facts.remember("mi editor es geany", "preferencias", db_path=dbp)
    facts.remember("Mi editor es geany.", db_path=dbp)
    facts.remember("mi editor es geany", " General ", db_path=dbp)
    facts.remember("mi editor es geany", None, db_path=dbp)  # type: ignore[arg-type]
    assert _hechos(dbp) == [("mi editor es geany", "preferencias")]
    assert [f["texto"] for f in facts.recall("preferencias", db_path=dbp)] == [
        "mi editor es geany"]


def test_reanotar_con_otra_categoria_la_sigue_cambiando(tmp_path):
    """Solo la categoría por defecto hereda: una explícita manda, y un hecho que
    nunca tuvo etiqueta sigue en general."""
    from crotolamo.persistence import db, facts

    dbp = tmp_path / "libreta.sqlite"
    facts.remember("uso Fedora", "preferencias", db_path=dbp)
    facts.remember("uso Fedora", "sistema", db_path=dbp)
    facts.remember("tomo café", db_path=dbp)
    facts.remember("Tomo café.", db_path=dbp)
    assert _hechos(dbp) == [("uso Fedora", "sistema"), ("Tomo café.", "general")]
    # Una libreta de antes, con copias: hereda la etiqueta de la más reciente que tenía.
    db.add_fact("mi perro se llama Tletl", "mascotas", db_path=dbp)
    db.add_fact("Mi perro se llama Tletl.", "familia", db_path=dbp)
    db.add_fact("mi perro se llama tletl", db_path=dbp)
    facts.remember("mi perro se llama Tletl", db_path=dbp)
    assert _hechos(dbp)[-1] == ("mi perro se llama Tletl", "familia")
    assert len(_hechos(dbp)) == 3


# --- idx 0 (final-u30v): un bloque de código no se lee línea a línea ----------------
# La segmentación (split_sentences, SentenceFeeder y la sonda de "sentence" de
# LiveStreamer) no corta dentro de un bloque ```: llega ENTERO a limpiar_para_voz, que
# lo descarta (los tests de la unidad están en test_revision_final_u30v.py). Aquí: la
# cota de una cerca que nunca cierra, la sonda y el coste en streaming.

# Una cerca sin cerrar no abre bloque: lo que la sigue se corta como siempre (las
# frases esperadas son las de antes del arreglo, iguales). `antes_del_flush` es lo
# que el streaming ya soltó cuando llega la cerca: el resto sale en flush.
_SIN_CERRAR = {
    "corte dentro del codigo": (
        "Así se lee:\n```python\nimport json\n\ndatos = json.load(f)\nprint(datos)",
        ["Así se lee:\n```python\nimport json", "datos = json.load(f)\nprint(datos)"], 0),
    "prosa tras la cerca": (
        "Prueba esto.\n\n```bash\nsudo dnf install x\n\nY luego reinicia. Listo.",
        ["Prueba esto.", "```bash\nsudo dnf install x", "Y luego reinicia.", "Listo."], 1),
}


@pytest.mark.parametrize("paso", [1, 3, 7])
@pytest.mark.parametrize("texto,esperado,antes_del_flush", _SIN_CERRAR.values(),
                         ids=_SIN_CERRAR.keys())
def test_una_cerca_que_nunca_cierra_se_corta_como_antes(texto, esperado, antes_del_flush,
                                                       paso):
    """Antes de la cota, todo lo que seguía a la cerca era UNA pieza: el aviso del
    LLM caído pegado al código y una frase de minutos que el plazo por frase de
    StreamSpeaker.finish() cortaba. Sale en flush (no más tarde que la cola de
    siempre) y en sus frases de siempre."""
    from crotolamo.voice.tts import SentenceFeeder, split_sentences

    assert split_sentences(texto) == esperado
    frases: list[str] = []
    feeder = SentenceFeeder(frases.append)
    for i in range(0, len(texto), paso):
        feeder.feed(texto[i:i + paso])
    assert frases == esperado[:antes_del_flush]
    feeder.flush()
    assert frases == esperado
    assert feeder.fed == len(esperado)


def test_el_aviso_del_llm_caido_dentro_de_un_bloque_va_en_frase_aparte(monkeypatch):
    """El LLM se cae con el bloque ``` abierto: el aviso no sale pegado al código."""
    llm = _LlmQueSeCaeAMedias(["Así se lee:\n```python\nimport json\n\n", "datos = json.load(f)"],
                              "GLM no respondió a tiempo, patrón. (timed out)")
    dicho = _hablar_un_turno(monkeypatch, _agente_real(llm))
    assert dicho == ["Así se lee:\n```python\nimport json", "datos = json.load(f)",
                     "GLM no respondió a tiempo, patrón.", "(timed out)"]


def test_la_sonda_de_sentence_no_cierra_frase_dentro_del_codigo():
    """release="sentence" suelta cuando la 1ª frase cerró y le sigue texto: la
    sangría de un bloque no es fin de frase (en lo que se habla tampoco). Antes
    soltaba en la 'p' de 'print'; ahora, tras el bloque, en la 'L' de 'Listo'."""
    from crotolamo.core.streaming import LiveStreamer

    texto = "Así:\n```python\nfor x in lista:\n    print(x)\n```\n\nListo. Ya quedó."
    dichos: list[str] = []
    s = LiveStreamer(dichos.append, hold_until_done=True, release="sentence")
    soltado = [i for i, c in enumerate(texto) if s.feed(c)]
    assert texto[:soltado[0] + 1].endswith("```\n\nL"), texto[:soltado[0] + 1]
    assert "".join(dichos) == texto


def test_el_feeder_no_resegmenta_un_bloque_largo_en_cada_token(monkeypatch):
    """Retener el bloque hasta su cierre no puede volver cuadrático el streaming:
    re-segmentar el buffer entero en cada token costaba 2 s de CPU por un bloque
    de 19 KB en trozos de 4 caracteres. Cada carácter se escanea pocas veces."""
    from crotolamo.voice import tts

    codigo = "".join(f"def f{i}(x):\n    # Paso {i}.\n    return x + {i}\n\n"
                     for i in range(300))
    texto = "Así queda:\n\n```python\n" + codigo + "```\n\nListo, patrón."
    escaneado: list[int] = []
    original = tts._split

    def contado(text, *, final, lista=None):
        escaneado.append(len(text))
        return original(text, final=final, lista=lista)

    monkeypatch.setattr(tts, "_split", contado)
    frases: list[str] = []
    feeder = tts.SentenceFeeder(frases.append)
    for i in range(0, len(texto), 4):
        feeder.feed(texto[i:i + 4])
    feeder.flush()
    assert frases == ["Así queda:", "```python\n" + codigo + "```", "Listo, patrón."]
    assert sum(escaneado) <= 3 * len(texto), sum(escaneado)


_PIEZAS_CON_CERCAS = ["```", "```python\n", "``", "`", "````", "Hola", "x", "Dr", "12", ".",
                      ". ", "!", "?", "\n", "\n\n", "    ", " ", "**", "1."]


def _feeder_de_referencia(trozos: list[str]) -> list[str]:
    """SentenceFeeder sin atajos: el buffer entero por `_split` en cada trozo
    (con la memoria de lista de la respuesta, que el buffer reiniciado no ve)."""
    from crotolamo.voice import tts

    buffer, frases = "", []
    lista = tts._Lista()
    for trozo in trozos:
        buffer += trozo
        piezas, buffer = tts._split(buffer, final=False, lista=lista)
        frases += piezas
    if buffer.count(tts._CERCA) % 2:
        piezas, buffer = tts._split(buffer, final=True, lista=lista)
        frases += piezas
    if buffer.strip():
        frases.append(buffer.strip())
    return frases


def test_el_feeder_que_espera_la_cerca_corta_igual_que_resegmentar_todo():
    """Esperar el cierre del bloque buscando la cerca solo en lo nuevo es una
    optimización pura: mismas frases que re-segmentar todo en cada trozo,
    también con cercas partidas entre trozos y rachas de 4 acentos graves."""
    import random

    from crotolamo.voice import tts

    rnd = random.Random(20261114)
    for _ in range(3000):
        texto = "".join(rnd.choice(_PIEZAS_CON_CERCAS) for _ in range(rnd.randint(0, 30)))
        trozos, i = [], 0
        while i < len(texto):
            n = rnd.randint(1, 6)
            trozos.append(texto[i:i + n])
            i += n
        frases: list[str] = []
        feeder = tts.SentenceFeeder(frases.append)
        for trozo in trozos:
            feeder.feed(trozo)
        feeder.flush()
        assert frases == _feeder_de_referencia(trozos), trozos


# --- idx 1 (final-u50): smoke_voz no arranca la pre-síntesis -------------------------

def test_smoke_circular_no_arranca_el_hilo_de_presintesis(monkeypatch, tmp_path):
    """Con [voice].tts_presynth = true circular() solo usa available() y
    synthesize_pcm(), que sintetiza directo con Piper: el hilo TtsPresintesis
    (que arrancan prepare() y speak()) no llega a existir y no hay nada que
    cerrar. Si el smoke llegara a hablar, necesitaría un dueño (close())."""
    import numpy as np

    from crotolamo.voice.tts import TTS
    from scripts import smoke_voz

    class _Chunk:
        audio_int16_array = np.full(2205, 1000, dtype=np.int16)
        sample_rate = 22050

    class _Voz:
        def synthesize(self, text):
            return [_Chunk()]

    class _Stt:
        def transcribe(self, audio, hotwords=None):
            return "crotolamo prueba uno dos tres"

    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo, presynth=True)
    tts._voice = _Voz()
    monkeypatch.setattr(smoke_voz, "get_settings", lambda: object())
    monkeypatch.setattr(smoke_voz, "TTS", SimpleNamespace(from_settings=lambda s: tts))
    monkeypatch.setattr(smoke_voz, "STT", SimpleNamespace(from_settings=lambda s: _Stt()))
    # Solo los hilos nuevos: otro test pudo dejar el suyo (daemon) vivo.
    antes = set(threading.enumerate())
    assert smoke_voz.circular() == 0
    assert tts._presint_hilo is None
    assert [t.name for t in set(threading.enumerate()) - antes] == []


# --- idx 0 (final-x7conexion-1): el followup del loop simple también pre-conecta -----
# El aviso del wake (WakePreconnect) solo viajaba con la orden que sigue al wake. El
# keep-alive de un stream se reutiliza hasta 15 s después de su fin
# (HTTPTransport._REUSE_MAX_IDLE_S) y, con stream_release = "final" (fábrica), la
# respuesta ENTERA se habla después de ese fin: una respuesta de 8-13 s más la orden
# del followup pasan de 15 s y el handshake TCP+TLS con la nube se pagaba dentro del
# turno. Ahora la grabación del followup lleva el mismo aviso y su 1ª voz pre-conecta.

class _AgenteDeUnStream:
    """Lo justo de un agente con motor de nube: el turno es UN chat_stream."""

    def __init__(self, llm):
        self.llm = llm
        self.ordenes: list[str] = []

    def handle_turn(self, command, on_token=None):
        self.ordenes.append(command)
        return self.llm.chat_stream([{"role": "user", "content": command}],
                                    on_token=on_token).content


class _OrdenesConReloj:
    """STT falso que mueve el reloj del transporte. Por escucha, (segundos hasta la
    1ª voz, texto): con voz llama a on_voice, como el STT real, espera a que la
    pre-conexión acabe (el patrón tarda más en decir la orden que un handshake) y
    suma lo que duran la orden, el silencio de corte y Whisper."""

    _ORDEN_S = 2.0 + 0.6 + 1.7

    def __init__(self, reloj, guion, aviso):
        self.reloj = reloj
        self.guion = list(guion)
        self.aviso = aviso
        self.avisos: list[bool] = []    # por escucha: ¿llegó con on_voice?

    def listen_once(self, on_voice=None, **kwargs):
        antes, texto = self.guion.pop(0)
        self.avisos.append(on_voice is not None)
        self.reloj["t"] += antes
        if not texto:
            return ""                   # nadie habló: la ventana venció
        if on_voice is not None:
            on_voice()
        if self.aviso is not None:
            self.aviso.join(5)
        self.reloj["t"] += self._ORDEN_S
        return texto

    listen_smart = listen_once


@pytest.mark.parametrize("respuesta_s, sin_aviso", [(10.5, 2), (1.5, 1)],
                         ids=["respuesta_larga", "respuesta_corta"])
def test_el_followup_preconecta_y_su_turno_no_paga_el_handshake(monkeypatch, respuesta_s,
                                                                sin_aviso):
    """Loop simple con GLMClient en streaming y sockets de verdad, el aviso que le
    da _listen_with_warmer y el reloj del transporte a mano: una orden tras el wake,
    un followup y una ventana que acaba en silencio.

    respuesta_larga: la petición del followup sale 16.3 s después del fin del stream
    anterior. El keep-alive ya no se reutiliza y antes conectaba el hilo del turno
    ({"Preconnect": 1, "MainThread": 1}). respuesta_corta (7.3 s): el keep-alive
    seguía valiendo y ahora se abre un repuesto igual, en segundo plano (el precio
    que ya pagan las órdenes tras un wake). En los dos casos el turno no hace ningún
    handshake, la ventana en silencio no abre nada y lo que se pide y se oye es lo
    mismo que sin aviso."""
    from crotolamo.core import engine
    from crotolamo.voice.threads import WakePreconnect
    from tests.test_perf_llm_turn_is_one_atomic_blocking_call import (
        _cliente,
        _conexiones_estables,
        _contando_conexiones,
        _Server,
    )

    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time",
                        SimpleNamespace(sleep=lambda s: None, time=time.time))
    reloj = {"t": 1000.0}
    monkeypatch.setattr(engine, "_reloj", lambda: reloj["t"])
    cfg = listener.ListenerConfig(threshold=0.72, variants=["crotolamo"], followup_s=6.0,
                                  ack="off", stream_speak=True, smart_endpoint=False)
    # Hasta la 1ª voz del followup: la respuesta hablada (entera tras el fin del
    # stream), la guarda tras responder y lo que el patrón tarda en reaccionar.
    guion = [(1.0, "abre youtube"), (respuesta_s + 0.5 + 1.0, "y ahora pausa la música"),
             (6.0, "")]

    vistos = {}
    for con_aviso in (False, True):
        srv = _Server()
        try:
            client = _cliente("glm", srv)
            por_hilo = _contando_conexiones(monkeypatch, client._transport)
            agente = _AgenteDeUnStream(client)
            aviso = WakePreconnect.del_agente(agente) if con_aviso else None
            stt = _OrdenesConReloj(reloj, guion, aviso)
            tts = _TtsDeTurno()
            extra = {} if aviso is None else {"on_wake": aviso}
            assert listener._run_simple_loop(
                agent=agente, stt=stt, wake_stt=_WakeUnaVez(), tts=tts, wake_detector=None,
                use_oww=False, cfg=cfg, hud_state=voice_state.SharedState(),
                say=lambda texto: None, **extra) == 0
            if aviso is not None:
                aviso.join(5)
            vistos[con_aviso] = (agente.ordenes, tts.spoken, list(srv.peticiones))
            if con_aviso:
                assert por_hilo == {"Preconnect": 2}
                assert _conexiones_estables(srv) == 2
            else:
                assert por_hilo == {"MainThread": sin_aviso}
            # El aviso viaja con las tres grabaciones; solo sale con las dos con voz.
            assert stt.avisos == [con_aviso] * 3
        finally:
            srv.parar()

    assert vistos[True] == vistos[False] == (
        ["abre youtube", "y ahora pausa la música"], ["Hola, patrón."] * 2,
        ["/api/paas/v4/chat/completions"] * 2)


@pytest.mark.parametrize("followup_con_voz", [True, False], ids=["con_voz", "en_silencio"])
def test_la_ventana_de_followup_en_silencio_no_abre_ninguna_conexion(monkeypatch,
                                                                     followup_con_voz):
    """Loop simple de fábrica (smart_endpoint, acuse con bip, followup de 6 s) con
    el STT real sobre un micrófono de guion, el aviso de _listen_with_warmer y
    GLMClient con sockets de verdad. El aviso del followup sale con su 1ª voz
    (on_voice de la grabación), no al abrir la ventana: la que acaba en silencio no
    abre nada (una conexión, la de la orden) y nada llega a Whisper. Con voz, la
    misma grabación abre una más, en segundo plano."""
    from crotolamo.voice import stt as stt_mod
    from crotolamo.voice.threads import WakePreconnect
    from tests.test_media_aware import _FakeTts as _TtsConBip
    from tests.test_perf_llm_turn_is_one_atomic_blocking_call import (
        _AgenteDeUnaLlamada,
        _cliente,
        _conexiones_estables,
        _contando_conexiones,
        _MicDeOrdenes,
        _Server,
        _SileroDeNivel,
        _WakeDeGuion,
    )

    turnos = 2 if followup_con_voz else 1
    # Un guion por grabación (un nivel por chunk de 32 ms); agotados, silencio.
    monkeypatch.setattr(_MicDeOrdenes, "ordenes", [[0] * 3 + [1] * 15 for _ in range(turnos)])
    monkeypatch.setattr(_MicDeOrdenes, "listo", None)
    monkeypatch.setitem(sys.modules, "sounddevice",
                        SimpleNamespace(InputStream=_MicDeOrdenes))
    monkeypatch.setattr(stt_mod, "_voice_cfg", lambda: {"vad_backend": "silero"})
    monkeypatch.setattr(stt_mod, "_silero_load_error", None)
    monkeypatch.setattr(stt_mod, "_get_silero_vad", lambda **k: _SileroDeNivel())
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "time",
                        SimpleNamespace(sleep=lambda s: None, time=time.time))
    cfg = listener.ListenerConfig(threshold=0.72, variants=["crotolamo"], stream_speak=False)
    assert (cfg.smart_endpoint, cfg.ack, cfg.followup_s) == (True, "beep", 6.0)  # fábrica

    srv = _Server("entero")
    try:
        client = _cliente("glm", srv)
        por_hilo = _contando_conexiones(monkeypatch, client._transport)
        agente = _AgenteDeUnaLlamada(client)
        aviso = WakePreconnect.del_agente(agente)
        assert aviso is not None
        textos = ["abre youtube", "y ahora pausa la música"]
        stt = stt_mod.STT(model_size="fake")

        def transcribir(audio, hotwords=None):
            aviso.join(5)               # Whisper tarda más que un handshake
            return textos.pop(0)

        monkeypatch.setattr(stt, "transcribe", transcribir)
        wake_stt = _WakeDeGuion(["crotolamo"])
        tts = _TtsConBip()
        assert listener._run_simple_loop(
            agent=agente, stt=stt, wake_stt=wake_stt, tts=tts, wake_detector=wake_stt,
            use_oww=False, cfg=cfg, hud_state=voice_state.SharedState(),
            say=lambda texto: None, on_wake=aviso) == 0
        aviso.join(5)

        assert agente.ordenes == ["abre youtube", "y ahora pausa la música"][:turnos]
        assert len(textos) == 2 - turnos            # la ventana en silencio: sin Whisper
        assert por_hilo == {"Preconnect": turnos}   # el turno nunca conecta
        assert _conexiones_estables(srv) == turnos
        assert srv.peticiones == ["/api/paas/v4/chat/completions"] * turnos
        assert tts.spoken == ["Hola, patrón."] * turnos
    finally:
        srv.parar()


# --- idx 1 (final-x6paquete-1): las pistas de instalación de Silero no nombran torch --
# silero-vad declara torch como dependencia dura y, solo con PyPI, pip resuelve la
# build CUDA (~3 GB entre torch, nvidia-* y triton) que una lap sin GPU no usa nunca;
# el código ni importa torch (voice/runtime.py corre el ONNX directo). install.sh,
# launch.sh y pyproject ya dan el índice CPU de PyTorch, el de la CI; el doctor y el
# aviso del oído seguían diciendo "pip install silero-vad torch".

def _indice_cpu_de_la_ci() -> str:
    """El índice con el que instala la CI (la instalación que sí se prueba)."""
    (url,) = re.findall(r"--extra-index-url\s+(\S+)",
                        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert url.rstrip("/").endswith("/whl/cpu"), url
    return url


def _pip_install_de(pista: str) -> list[str]:
    """Los argumentos del `pip install ...` que sugiere `pista`, como los recibiría
    pip (lo que va entre paréntesis tras el comando es comentario para el patrón)."""
    comando = pista.split("pip install ", 1)[1].split("  (", 1)[0]
    return shlex.split(comando)


def test_doctor_sin_silero_sugiere_instalarlo_con_el_indice_cpu(monkeypatch):
    from scripts import crotolamo_doctor as doctor

    monkeypatch.setitem(sys.modules, "silero_vad", None)
    (check,) = doctor._check_silero(None)
    assert check.ok is False
    # Ni `torch` suelto (PyPI: la build CUDA) ni otro índice que el de la CI.
    assert _pip_install_de(check.fix) == [
        "silero-vad", "--extra-index-url", _indice_cpu_de_la_ci()]
    assert check.fix.endswith("(opcional, mejora la detección de voz)")


def test_el_aviso_del_oido_sin_silero_sugiere_instalarlo_con_el_indice_cpu(monkeypatch,
                                                                            caplog):
    pytest.importorskip("numpy")
    from crotolamo.voice import stt as stt_mod
    from crotolamo.voice.adapters import _SileroVad

    def _sin_silero(**kwargs):
        raise ModuleNotFoundError("No module named 'silero_vad'")

    monkeypatch.setattr(stt_mod, "_get_silero_vad", _sin_silero)
    vad = _SileroVad()
    with caplog.at_level(logging.WARNING):
        assert vad([0.5] * 512) == 1.0      # cae a energía y sigue oyendo
        assert vad([0.0] * 512) == 0.0
    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1 and avisos[0].startswith("silero-vad no disponible")
    assert _pip_install_de(avisos[0]) == [
        "silero-vad", "--extra-index-url", _indice_cpu_de_la_ci()]
