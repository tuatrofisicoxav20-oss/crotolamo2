"""Integración de la auditoría: cambios que un grupo pidió en archivos de otro.

Cada bloque cita el pedido (idx de la lista de integración y el id del hallazgo)
y prueba el comportamiento que el arreglo cambia; todos fallaban antes.
"""

from __future__ import annotations

import builtins
import http.client
import logging
import runpy
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


def _check_memoria_del_doctor(monkeypatch, tmp_path, faltan=(), env=None, **memoria):
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


@pytest.mark.parametrize("env", [
    {"CROTOLAMO_GLM_API_KEY": "clave-de-zai"},
    {"OPENAI_API_KEY": "clave-del-sdk"},  # el SDK la toma solo: el runtime extrae
])
def test_doctor_memoria_con_key_es_un_ok_liso(monkeypatch, tmp_path, env):
    check = _check_memoria_del_doctor(monkeypatch, tmp_path, env=env)
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
