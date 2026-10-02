"""Tests del doctor: audita el stack real con TODO mockeado (sin red, sin binarios).

La fixture `mundo` deja un entorno "vacío" y controlable (sin key, sin nube, sin
PipeWire, sin systemd, sin módulos de voz); cada test rompe o arregla UNA cosa,
así el mensaje que se comprueba es el de ese check y no ruido de otro.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import tomllib
import types
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pytest

from crotolamo.core.glm import API_KEY_ENVS
from crotolamo.settings import DEFAULT_CONFIG, Settings
from scripts import crotolamo_doctor as doctor

HOY = date(2026, 9, 30)
KEY = "sk-test-SECRETO-123"
MODULOS_VOZ = ("faster_whisper", "piper", "sounddevice", "openwakeword", "silero_vad", "onnx_asr")


def _settings(tmp_path: Path, **doctor_cfg) -> Settings:
    """Settings de mentiras: backend glm, Piper como TTS y la caducidad del toml."""
    voces = tmp_path / "voices"
    voces.mkdir(exist_ok=True)
    raw = {
        "llm": {
            "backend": "glm",
            "host": "http://localhost:11434",
            "model": "qwen2.5-coder:7b",
            "glm": {"base_url": "https://nube.test/v1", "model": "glm-4.7-flash"},
        },
        "voice": {"piper_voice": "es_MX-ald-medium.onnx"},
        "doctor": {"musica_caduca": date(2026, 11, 29), **doctor_cfg},
    }
    return Settings(raw=raw, user="test", home=tmp_path,
                    paths={"home": tmp_path, "voces": voces})


class _Resp:
    """Respuesta HTTP de mentiras con el contrato mínimo que usa urlopen."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args) -> bool:
        return False


class _Mundo:
    """Perillas del entorno falso. Cada test ajusta las que necesita."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.nube: BaseException | bytes = urllib.error.URLError("sin red")
        self.ollama: BaseException | bytes = urllib.error.URLError("sin ollama")
        self.pw: BaseException | list = []
        self.systemctl: BaseException | str = "inactive"
        self.ws_ok = False
        self.listeners: list[int] = []
        self.peticiones: list[urllib.request.Request] = []
        self.pw_dump_real = doctor._pw_dump


@pytest.fixture
def mundo(monkeypatch, tmp_path):
    m = _Mundo(_settings(tmp_path))
    for env in (*API_KEY_ENVS, "SPOTIFY_CLIENT_ID", "CROTOLAMO_LLM_BACKEND"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(doctor, "load_settings", lambda: m.settings)

    def urlopen(req, timeout=None):
        assert timeout is not None and timeout <= 5, "toda la red lleva timeout corto"
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        if isinstance(req, urllib.request.Request):
            m.peticiones.append(req)
        resultado = m.nube if "/models" in url else m.ollama
        if isinstance(resultado, BaseException):
            raise resultado
        return _Resp(resultado)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def pw_dump():
        if isinstance(m.pw, BaseException):
            raise m.pw
        return m.pw

    monkeypatch.setattr(doctor, "_pw_dump", pw_dump)

    def run(cmd, **kwargs):
        assert kwargs.get("timeout", 99) <= 5, "todo subprocess lleva timeout corto"
        if cmd[0] == "pw-dump":
            raise FileNotFoundError("pw-dump")
        if cmd[0] == "systemctl":
            if isinstance(m.systemctl, BaseException):
                raise m.systemctl
            rc = 0 if m.systemctl == "active" else 3
            return subprocess.CompletedProcess(cmd, rc, m.systemctl + "\n", "")
        raise AssertionError(f"subprocess inesperado: {cmd}")

    monkeypatch.setattr(subprocess, "run", run)

    def connect(addr, timeout=None, **kwargs):
        assert timeout is not None and timeout <= 5
        if not m.ws_ok:
            raise ConnectionRefusedError("rechazada")
        return types.SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(doctor, "_listener_pids", lambda: list(m.listeners))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    # Módulos de voz: ninguno instalado salvo que el test diga otra cosa.
    for mod in MODULOS_VOZ:
        monkeypatch.setitem(sys.modules, mod, None)
    return m


def _checks(today: date = HOY) -> dict[str, doctor.Check]:
    out: dict[str, doctor.Check] = {}
    for c in doctor.collect_checks(today=today):
        assert c.name not in out, f"check duplicado: {c.name}"
        out[c.name] = c
    return out


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://nube.test/v1/models", code, "err", None, None)


# --- la key de la nube -------------------------------------------------------

def test_sin_key_es_requerido_y_avisa_de_la_caida_a_ollama(mundo):
    """El bug original: sin key, Crotolamo caía a Ollama y el doctor salía verde."""
    checks = _checks()
    key = checks["llm-key"]
    assert not key.ok and key.required
    assert "caer a Ollama" in key.detail
    assert "~/.config/crotolamo/env" in key.fix and "sin export" in key.fix
    nube = checks["llm-nube"]
    assert not nube.ok and nube.required and "sin key" in nube.detail
    assert not mundo.peticiones, "sin key no se toca la red"


def test_con_key_dice_presente_sin_revelar_nada(mundo, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", KEY)
    key = _checks()["llm-key"]
    assert key.ok and "presente en ZAI_API_KEY" in key.detail
    assert KEY not in key.detail and str(len(KEY)) not in key.detail


def test_key_en_blanco_cuenta_como_ausente(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "   ")
    assert not _checks()["llm-key"].ok


def test_con_backend_ollama_la_nube_pasa_a_opcional_y_ollama_a_requerido(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "ollama")
    checks = _checks()
    assert not checks["llm-key"].required and "la nube no se usa" in checks["llm-key"].detail
    assert not checks["llm-nube"].required and "la nube no se usa" in checks["llm-nube"].detail
    assert checks["ollama"].required and "motor principal" in checks["ollama"].detail


# --- la nube -----------------------------------------------------------------

def test_nube_401_es_key_invalida(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = _http_error(401)
    nube = _checks()["llm-nube"]
    assert not nube.ok and nube.required and "key inválida" in nube.detail
    (req,) = mundo.peticiones
    assert req.full_url == "https://nube.test/v1/models"
    assert req.get_header("Authorization") == f"Bearer {KEY}"


def test_nube_429_es_rate_limit(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = _http_error(429)
    assert "rate limit" in _checks()["llm-nube"].detail


def test_nube_sin_red(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    assert "sin internet" in _checks()["llm-nube"].detail


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),                      # salta leyendo
    urllib.error.URLError(TimeoutError("timed out")),  # salta conectando
])
def test_nube_timeout(mundo, monkeypatch, error):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = error
    assert "no respondió en 5 s" in _checks()["llm-nube"].detail


def test_nube_otro_status(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = _http_error(500)
    assert "respondió 500" in _checks()["llm-nube"].detail


def test_nube_responde_y_lista_el_modelo(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = json.dumps({"data": [{"id": "glm-4.7-flash"}, {"id": "glm-5.2"}]}).encode()
    nube = _checks()["llm-nube"]
    assert nube.ok and not nube.warn and nube.required
    assert "responde" in nube.detail and "glm-4.7-flash" in nube.detail


def test_nube_responde_pero_el_modelo_no_esta_en_la_lista(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = json.dumps({"data": [{"id": "otro-modelo"}]}).encode()
    nube = _checks()["llm-nube"]
    assert nube.ok and nube.warn and "no aparece en /models" in nube.detail


def test_nube_responde_sin_lista_de_modelos(mundo, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = b"<html>no json</html>"
    nube = _checks()["llm-nube"]
    assert nube.ok and not nube.warn and "modelo configurado: glm-4.7-flash" in nube.detail


# --- AEC de PipeWire ---------------------------------------------------------

def _dump(default_source="crotolamo_aec_source", default_sink="crotolamo_aec_sink",
          mute: bool | None = False, con_nodos: bool = True) -> list[dict]:
    """Grafo pw-dump mínimo: los dos nodos AEC y el metadata `default`."""
    objetos: list[dict] = []
    if con_nodos:
        props = {"Props": [{"mute": mute}]} if mute is not None else {}
        objetos += [
            {"id": 61, "type": "PipeWire:Interface:Node",
             "info": {"props": {"node.name": "crotolamo_aec_source"}, "params": props}},
            {"id": 62, "type": "PipeWire:Interface:Node",
             "info": {"props": {"node.name": "crotolamo_aec_sink"}, "params": {}}},
        ]
    objetos += [
        {"id": 33, "type": "PipeWire:Interface:Node",
         "info": {"props": {"node.name": "alsa_output.pci"}, "params": {}}},
        {"id": 34, "type": "PipeWire:Interface:Metadata", "props": {"metadata.name": "default"},
         "metadata": [
             {"key": "default.audio.source", "value": {"name": default_source}},
             {"key": "default.audio.sink", "value": {"name": default_sink}},
         ]},
    ]
    return objetos


def test_aec_todo_en_orden(mundo):
    mundo.pw = _dump()
    aec = _checks()["aec"]
    assert aec.ok and not aec.warn and aec.required


def test_aec_sink_por_defecto_incorrecto_avisa_de_la_referencia(mundo):
    mundo.pw = _dump(default_sink="alsa_output.pci")
    aec = _checks()["aec"]
    assert not aec.ok and "no tiene referencia" in aec.detail
    assert "wpctl set-default 62" in aec.fix


def test_aec_source_en_mute(mundo):
    mundo.pw = _dump(mute=True)
    aec = _checks()["aec"]
    assert not aec.ok and "micrófono AEC en mute" in aec.detail
    assert "wpctl set-mute 61 0" in aec.fix


def test_aec_sin_dato_de_mute_asume_no_mute_con_aviso(mundo):
    mundo.pw = _dump(mute=None)
    aec = _checks()["aec"]
    assert aec.ok and aec.warn and "mute" in aec.detail


def test_aec_source_por_defecto_incorrecto(mundo):
    mundo.pw = _dump(default_source="alsa_input.pci")
    aec = _checks()["aec"]
    assert not aec.ok and "trae eco" in aec.detail


def test_aec_source_no_default_solo_avisa_si_crotolamo_lo_abre_explicito(mundo):
    """Con [voice].input_device apuntando al AEC, el default global no le afecta."""
    mundo.settings.raw["voice"]["input_device"] = "crotolamo_aec_source"
    mundo.pw = _dump(default_source="alsa_input.pci")
    aec = _checks()["aec"]
    assert aec.ok and aec.warn and "input_device" in aec.detail


def test_aec_nodos_ausentes(mundo):
    mundo.pw = _dump(con_nodos=False)
    aec = _checks()["aec"]
    assert not aec.ok and "faltan los nodos" in aec.detail
    assert "aec.sh on" in aec.fix


def test_aec_sin_pw_dump(mundo, monkeypatch):
    """Con el _pw_dump real y sin binario: ❌ claro, sin traceback."""
    monkeypatch.setattr(doctor, "_pw_dump", mundo.pw_dump_real)
    aec = _checks()["aec"]
    assert not aec.ok and aec.required and "no tengo pw-dump" in aec.detail


def test_aec_pw_dump_colgado(mundo):
    mundo.pw = subprocess.TimeoutExpired("pw-dump", 5)
    assert "no respondió" in _checks()["aec"].detail


def test_aec_se_omite_sin_nodos_configurados(mundo, tmp_path):
    mundo.settings = _settings(tmp_path, aec_source="", aec_sink="")
    assert "aec" not in _checks()


# --- música (Soloist) y Spotify ----------------------------------------------

def test_soloist_caido_y_websocket_rechazado(mundo):
    checks = _checks()
    servicio = checks["musica-servicio"]
    assert not servicio.ok and not servicio.required and "inactive" in servicio.detail
    assert "systemctl --user start soloist.service" in servicio.fix
    ws = checks["musica-ws"]
    assert not ws.ok and not ws.required and "127.0.0.1:9090" in ws.detail


def test_soloist_activo_y_websocket_acepta(mundo):
    mundo.systemctl = "active"
    mundo.ws_ok = True
    checks = _checks()
    assert checks["musica-servicio"].ok and checks["musica-ws"].ok


def test_sin_systemctl(mundo):
    mundo.systemctl = FileNotFoundError("systemctl")
    servicio = _checks()["musica-servicio"]
    assert not servicio.ok and "no tengo systemctl" in servicio.detail


def test_spotify_env(mundo, monkeypatch):
    spotify = _checks()["spotify"]
    assert not spotify.ok and not spotify.required and "falta SPOTIFY_CLIENT_ID" in spotify.detail

    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id-cliente-secreto")
    spotify = _checks()["spotify"]
    assert spotify.ok and "id-cliente-secreto" not in spotify.detail


def test_caducidad_lejana_es_verde(mundo):
    caduca = _checks(today=HOY)["musica-caduca"]
    assert caduca.ok and not caduca.warn and "60 días" in caduca.detail


def test_caducidad_a_diez_dias_avisa(mundo):
    caduca = _checks(today=date(2026, 11, 19))["musica-caduca"]
    assert caduca.ok and caduca.warn and "caduca en 10 días" in caduca.detail


def test_caducidad_pasada_es_roja(mundo):
    caduca = _checks(today=date(2026, 12, 1))["musica-caduca"]
    assert not caduca.ok and not caduca.required and "caducó el 2026-11-29" in caduca.detail


def test_caducidad_se_omite_sin_fecha(mundo, tmp_path):
    mundo.settings = _settings(tmp_path, musica_caduca="")
    assert "musica-caduca" not in _checks()


# --- listeners ---------------------------------------------------------------

def test_dos_listeners_avisan_que_se_pisan_el_micro(mundo):
    mundo.listeners = [4242, 4343]
    lst = _checks()["listeners"]
    assert lst.ok and lst.warn and "se pisan el micrófono" in lst.detail and "4242" in lst.detail


def test_un_listener_es_normal(mundo):
    mundo.listeners = [4242]
    lst = _checks()["listeners"]
    assert lst.ok and not lst.warn


def test_listener_pids_ignora_envoltorios_y_al_propio_proceso(monkeypatch, tmp_path):
    """launch.sh (bash) envolviendo al python real es UN listener, no dos."""
    proc = tmp_path / "proc"

    def pid(n: int, argv: list[str], ppid: int) -> None:
        d = proc / str(n)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "stat").write_text(f"{n} (proc x) S {ppid} 1 1 0", encoding="utf-8")

    pid(100, ["bash", "/home/p/crotolamo2/launch.sh", "listen"], 1)   # envoltorio
    pid(101, ["python", "-m", "crotolamo", "listen"], 100)             # el real
    pid(102, ["python", "-m", "crotolamo", "listen", "--simple"], 1)   # otro manual
    pid(103, ["grep", "listen", "crotolamo"], 1)                       # 'listen' suelto… y crotolamo
    pid(104, ["python", "-m", "crotolamo", "doctor"], 1)               # no es listener
    pid(105, ["cat", "/home/p/.config/crotolamo/listener.env"], 1)     # 'listener' no cuenta
    monkeypatch.setattr(doctor.os, "getpid", lambda: 103)              # el 103 somos nosotros
    monkeypatch.setattr(doctor, "Path", lambda p: proc if p == "/proc" else Path(p))
    assert doctor._listener_pids() == [101, 102]


# --- STT / TTS ---------------------------------------------------------------

def test_stt_importable(mundo, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    stt = _checks()["stt"]
    assert stt.ok and stt.required and "faster_whisper importable" in stt.detail


def test_stt_no_importable(mundo):
    stt = _checks()["stt"]
    assert not stt.ok and stt.required and "no puedo importar faster_whisper" in stt.detail


def test_stt_parakeet_con_modelo_en_disco(mundo, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnx_asr", types.ModuleType("onnx_asr"))
    modelo = tmp_path / "parakeet"
    mundo.settings = _settings(tmp_path, stt_module="onnx_asr", stt_model=str(modelo))
    assert not _checks()["stt"].ok and "falta el modelo" in _checks()["stt"].detail
    modelo.mkdir()
    assert _checks()["stt"].ok


def test_tts_piper_ausente_presente_y_sin_json(mundo, tmp_path):
    assert not _checks()["tts"].ok and "falta la voz Piper" in _checks()["tts"].detail
    onnx = tmp_path / "voices" / "es_MX-ald-medium.onnx"
    onnx.write_bytes(b"onnx")
    tts = _checks()["tts"]
    assert tts.ok and tts.warn and tts.required and ".onnx.json" in tts.detail
    onnx.with_name(onnx.name + ".json").write_text("{}", encoding="utf-8")
    tts = _checks()["tts"]
    assert tts.ok and not tts.warn


def test_tts_kokoro_con_voz_en_npz(mundo, tmp_path):
    np = pytest.importorskip("numpy")
    modelo = tmp_path / "kokoro-v1.0.onnx"
    modelo.write_bytes(b"onnx")
    voces = tmp_path / "voices-v1.0.bin"
    with voces.open("wb") as fh:  # con nombre de archivo, savez añadiría .npz
        np.savez(fh, em_alex=np.zeros(3), af_bella=np.ones(2))
    mundo.settings = _settings(tmp_path, tts_model=str(modelo), tts_voices=str(voces),
                               tts_voice="em_alex")
    tts = _checks()["tts"]
    assert tts.ok and not tts.warn and "voz em_alex OK" in tts.detail

    mundo.settings = _settings(tmp_path, tts_model=str(modelo), tts_voices=str(voces),
                               tts_voice="no_existe")
    tts = _checks()["tts"]
    assert not tts.ok and "'no_existe' no está en voices-v1.0.bin" in tts.detail


def test_tts_kokoro_sin_numpy_avisa_que_no_verifico_la_voz(mundo, monkeypatch, tmp_path):
    modelo = tmp_path / "kokoro-v1.0.onnx"
    modelo.write_bytes(b"onnx")
    voces = tmp_path / "voices-v1.0.bin"
    voces.write_bytes(b"lo que sea")
    mundo.settings = _settings(tmp_path, tts_model=str(modelo), tts_voices=str(voces),
                               tts_voice="em_alex")
    monkeypatch.setitem(sys.modules, "numpy", None)
    tts = _checks()["tts"]
    assert tts.ok and tts.warn and "no pude verificar la voz" in tts.detail


def test_tts_voces_en_json(mundo, tmp_path):
    modelo = tmp_path / "modelo.onnx"
    modelo.write_bytes(b"onnx")
    voces = tmp_path / "voices.json"
    voces.write_text(json.dumps({"em_alex": [0.1]}), encoding="utf-8")
    mundo.settings = _settings(tmp_path, tts_model=str(modelo), tts_voices=str(voces),
                               tts_voice="em_alex")
    assert _checks()["tts"].ok


def test_fallback_voz_revisa_piper_y_whisper_cuando_el_principal_es_otro(mundo, monkeypatch,
                                                                       tmp_path):
    modelo = tmp_path / "kokoro-v1.0.onnx"
    modelo.write_bytes(b"onnx")
    mundo.settings = _settings(tmp_path, tts_model=str(modelo), stt_module="onnx_asr")
    fb = _checks()["fallback-voz"]
    assert not fb.ok and not fb.required
    assert "piper no importa" in fb.detail and "falta la voz Piper" in fb.detail
    assert "faster_whisper no importa" in fb.detail

    monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    onnx = tmp_path / "voices" / "es_MX-ald-medium.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_name(onnx.name + ".json").write_text("{}", encoding="utf-8")
    assert _checks()["fallback-voz"].ok


# --- Ollama de respaldo ------------------------------------------------------

def test_ollama_respaldo_con_modelo(mundo):
    mundo.ollama = json.dumps({"models": [{"name": "qwen2.5-coder:7b"}]}).encode()
    checks = _checks()
    assert checks["ollama"].ok and not checks["ollama"].required
    assert "respaldo local" in checks["ollama"].detail
    assert checks["modelo"].ok and not checks["modelo"].required


def test_ollama_caido_no_bloquea_con_la_nube(mundo):
    ollama = _checks()["ollama"]
    assert not ollama.ok and not ollama.required and "modelo" not in _checks()


# --- robustez y salida -------------------------------------------------------

def test_un_check_que_revienta_se_reporta_y_el_doctor_sigue(mundo):
    mundo.pw = RuntimeError("kaboom")
    checks = _checks()
    aec = checks["aec"]
    assert not aec.ok and aec.required and "kaboom" in aec.detail
    assert {"llm-key", "stt", "tts", "navegador"} <= set(checks)


def test_config_rota_devuelve_solo_ese_check(mundo, monkeypatch):
    def rota():
        raise FileNotFoundError("No encuentro la config")

    monkeypatch.setattr(doctor, "load_settings", rota)
    (config,) = doctor.collect_checks(today=HOY)
    assert config.name == "config" and not config.ok and config.required
    assert doctor.run_doctor() == 1


def test_la_salida_nunca_contiene_la_key(mundo, monkeypatch, capsys, caplog):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = json.dumps({"data": [{"id": "glm-4.7-flash"}]}).encode()
    # Un check ajeno que revienta con la key en el mensaje: ni así debe salir.
    mundo.pw = RuntimeError(f"pw-dump explotó con {KEY} dentro")
    with caplog.at_level(logging.DEBUG, logger="crotolamo.doctor"):
        doctor.run_doctor()
    out = capsys.readouterr()
    todo = out.out + out.err + caplog.text
    assert "SECRETO" not in todo and KEY not in todo
    assert "presente en CROTOLAMO_GLM_API_KEY" in out.out
    assert "«oculto»" in out.out


def test_run_doctor_devuelve_1_solo_si_falla_un_requerido(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "collect_checks", lambda today=None: [
        doctor.Check("config", True, "ok", required=True),
        doctor.Check("aec", False, "roto", "arregla", required=True),
        doctor.Check("ollama", False, "caído", required=False),
    ])
    assert doctor.run_doctor() == 1
    out = capsys.readouterr().out
    assert "Hay fallos REQUERIDOS que arreglar, patrón." in out
    assert out.index("--- REQUERIDOS ---") < out.index("❌ aec") < out.index("--- OPCIONALES ---")
    assert "↳ fix: arregla" in out


def test_run_doctor_devuelve_0_con_solo_opcionales_en_rojo(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "collect_checks", lambda today=None: [
        doctor.Check("config", True, "ok", required=True),
        doctor.Check("musica-caduca", True, "caduca en 3 días", "renueva", warn=True),
        doctor.Check("ollama", False, "caído", required=False),
    ])
    assert doctor.run_doctor() == 0
    out = capsys.readouterr().out
    assert "Todo en orden, patrón. (Los ❌/⚠️ opcionales no bloquean.)" in out
    assert "⚠️ musica-caduca" in out and "↳ fix: renueva" in out
    assert "❌ ollama" in out


def test_marcas_de_check():
    assert doctor.Check("x", True, "d").mark == "✅"
    assert doctor.Check("x", True, "d", warn=True).mark == "⚠️"
    assert doctor.Check("x", False, "d").mark == "❌"


# --- la config real ----------------------------------------------------------

def test_el_toml_trae_la_seccion_doctor_con_los_defaults_del_repo():
    """Se parsea el archivo versionado a pelo (sin el merge del local.toml)."""
    with DEFAULT_CONFIG.open("rb") as fh:
        data = tomllib.load(fh)
    secciones = list(data)
    assert secciones.index("doctor") == secciones.index("logging") + 1
    cfg = data["doctor"]
    assert cfg["stt_module"] == "faster_whisper" and cfg["tts_model"] == ""
    assert cfg["aec_source"] == "crotolamo_aec_source"
    assert cfg["aec_sink"] == "crotolamo_aec_sink"
    assert cfg["musica_service"] == "soloist.service"
    assert (cfg["musica_host"], cfg["musica_port"]) == ("127.0.0.1", 9090)
    assert cfg["musica_caduca"] == date(2026, 11, 29) and cfg["musica_aviso_dias"] == 14
    assert cfg["spotify_client_id_env"] == "SPOTIFY_CLIENT_ID"
    # Las claves del toml y los defaults del código son el mismo conjunto: si
    # alguien añade una perilla, que la documente en los dos sitios.
    assert set(cfg) == set(doctor._DOCTOR_DEFAULTS)
