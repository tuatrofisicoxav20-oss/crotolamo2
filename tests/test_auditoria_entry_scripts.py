"""Auditoría exhaustiva — grupo scripts de entrada (doctor, smoke, harness, wake_debug).

Cada test fija un hallazgo confirmado de la auditoría: falla con el código de
antes y pasa con el arreglo. Los comentarios dicen QUÉ se rompía. Todo SIN red
externa: servidores efímeros en 127.0.0.1 y el `mundo` de mentiras del doctor.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import types
import urllib.request

import pytest

from scripts import crotolamo_doctor as doctor
from tests import test_doctor
from tests.test_doctor import KEY, _checks, _dump, _settings

# La fixture `mundo` del doctor (su entorno de mentiras), reutilizada tal cual.
# Se asigna en vez de importarla por nombre para que el linter no la tome por un
# import "redefinido" por el parámetro de cada test.
mundo = test_doctor.mundo


# --- el doctor mide el MISMO camino de red que el runtime (sin proxy) ---------

class _Servidor(http.server.HTTPServer):
    """HTTPServer efímero que apunta (ruta, cabecera Authorization) de cada GET."""

    def __init__(self, handler) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.vistas: list[tuple[str, str | None]] = []


def _apuntar(handler: http.server.BaseHTTPRequestHandler) -> None:
    assert isinstance(handler.server, _Servidor)
    handler.server.vistas.append((handler.path, handler.headers.get("Authorization")))


class _Destino(http.server.BaseHTTPRequestHandler):
    """La nube y Ollama de mentiras: /v1/models y /api/tags."""

    def do_GET(self) -> None:
        _apuntar(self)
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "glm-de-prueba"}]}).encode()
        else:
            body = json.dumps({"models": [{"name": "qwen2.5-coder:7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class _Proxy(http.server.BaseHTTPRequestHandler):
    """Proxy de mentiras: apunta lo que le llega y contesta 502."""

    def do_GET(self) -> None:
        _apuntar(self)
        self.send_response(502)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


def _servir(handler) -> _Servidor:
    server = _Servidor(handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def tras_proxy(monkeypatch):
    """Un destino vivo en 127.0.0.1 y un entorno con *_proxy apuntando a otro lado.

    Es la situación del hallazgo: el runtime (HTTPTransport, http.client) ignora
    las variables de proxy y llega directo; el doctor sondeaba con urlopen, que
    las obedece, y daba un veredicto de OTRO camino de red.
    """
    destino, proxy = _servir(_Destino), _servir(_Proxy)
    for env in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(env, raising=False)
    for env in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy",
                "ALL_PROXY"):
        monkeypatch.setenv(env, f"http://127.0.0.1:{proxy.server_port}")
    # El opener global de urllib se construye UNA vez, con el entorno de ese
    # momento: se suelta para que quien use urlopen lo rehaga bajo ESTE entorno.
    monkeypatch.setattr(urllib.request, "_opener", None)
    yield destino, proxy
    for server in (destino, proxy):
        server.shutdown()
        server.server_close()


def test_la_sonda_de_la_nube_no_pasa_por_el_proxy_del_entorno(tras_proxy):
    """Con *_proxy en el entorno el doctor decía `llm-nube` ❌ ("respondió 502")
    aunque Crotolamo conectaba bien: sondeaba por el proxy y el runtime va directo.
    De paso, la key viajaba en claro al proxy cuando la base_url es http://."""
    destino, proxy = tras_proxy
    base = f"http://127.0.0.1:{destino.server_port}/v1"
    check = doctor._probe_cloud(base, "glm-de-prueba", KEY)
    assert check.ok and not check.warn, check.detail
    assert destino.vistas == [("/v1/models", f"Bearer {KEY}")]
    assert proxy.vistas == [], "la sonda (y la key) no deben salir hacia el proxy"


def test_la_sonda_de_ollama_no_pasa_por_el_proxy_del_entorno(tras_proxy):
    """Mismo caso con Ollama en localhost: un proxy que no enruta localhost lo
    pintaba como caído (`_ollama_tags` devolvía None) con Ollama respondiendo."""
    destino, proxy = tras_proxy
    tags = doctor._ollama_tags(f"http://127.0.0.1:{destino.server_port}")
    assert tags == ["qwen2.5-coder:7b"]
    assert [ruta for ruta, _ in destino.vistas] == ["/api/tags"]
    assert proxy.vistas == []


# --- `tts` requerido: con Piper de principal, el módulo también debe importar --

def _voz_piper_en_disco(tmp_path) -> None:
    onnx = tmp_path / "voices" / "es_MX-ald-medium.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_name(onnx.name + ".json").write_text("{}", encoding="utf-8")


def test_tts_piper_con_la_voz_en_disco_pero_sin_el_modulo_es_rojo(mundo, tmp_path):
    """Falso OK: con el .onnx descargado y sin piper-tts, `tts` salía ✅ mientras
    TTS.speak fallaba en CADA frase ("error sintetizando/reproduciendo con Piper")
    y el asistente quedaba mudo. El STT requerido sí comprobaba su módulo."""
    _voz_piper_en_disco(tmp_path)  # `mundo` deja piper como no importable
    tts = _checks()["tts"]
    assert not tts.ok and tts.required
    assert "no puedo importar piper" in tts.detail
    assert "[voice]" in tts.fix


def test_tts_piper_sin_modulo_y_sin_json_da_los_dos_arreglos(mundo, tmp_path):
    """El aviso del .onnx.json que falta no se pierde por el rojo del módulo: el
    detalle y el fix traen las dos cosas, para arreglarlas de una pasada."""
    (tmp_path / "voices" / "es_MX-ald-medium.onnx").write_bytes(b"onnx")
    tts = _checks()["tts"]
    assert not tts.ok
    assert ".onnx.json" in tts.detail and "no puedo importar piper" in tts.detail
    assert "Descarga el .onnx.json" in tts.fix and "piper-tts" in tts.fix


def test_doctor_sale_con_1_si_el_tts_principal_no_puede_hablar(mundo, monkeypatch, tmp_path,
                                                              capsys):
    """De punta a punta: todo lo requerido en verde salvo el módulo piper. Antes:
    "Todo en orden, patrón" y exit 0; ahora el requerido en rojo devuelve 1."""
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", KEY)
    mundo.nube = json.dumps({"data": [{"id": "glm-4.7-flash"}]}).encode()
    mundo.pw = _dump()
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    _voz_piper_en_disco(tmp_path)

    assert doctor.run_doctor() == 1
    out = capsys.readouterr().out
    assert "❌ tts" in out and "Hay fallos REQUERIDOS" in out

    # Con el módulo instalado, el mismo mundo es un doctor verde: el rojo de
    # arriba era SOLO por piper, no ruido de otro check.
    monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
    assert doctor.run_doctor() == 0
    assert "✅ tts" in capsys.readouterr().out


def test_tts_con_otro_motor_de_principal_no_exige_piper(mundo, tmp_path):
    """Con [doctor].tts_model (p.ej. Kokoro) Piper es solo el respaldo: que falte
    el módulo lo cuenta `fallback-voz` (opcional), no el `tts` requerido."""
    modelo = tmp_path / "kokoro-v1.0.onnx"
    modelo.write_bytes(b"onnx")
    mundo.settings = _settings(tmp_path, tts_model=str(modelo))
    checks = _checks()
    assert checks["tts"].ok and checks["tts"].required
    assert not checks["fallback-voz"].ok and "piper no importa" in checks["fallback-voz"].detail


# --- `modelo` de Ollama: nombre COMPLETO, con su tag --------------------------

@pytest.mark.parametrize(("configurado", "instalados", "esta"), [
    # El hallazgo: mismo nombre base, otro tag. /api/chat responde 404 y cada
    # turno sale "¿Existe el modelo...?", pero el doctor decía "instalado".
    ("qwen2.5-coder:7b", ["qwen2.5-coder:1.5b"], False),
    ("qwen2.5-coder:7b", ["qwen2.5-coder:7b-instruct-q4_K_M"], False),
    # Sin tag, Ollama resuelve `:latest`: otro tag instalado tampoco sirve.
    ("qwen2.5-coder", ["qwen2.5-coder:7b"], False),
    ("qwen2.5-coder", ["qwen2.5-coder:latest"], True),
    ("qwen2.5-coder:latest", ["qwen2.5-coder:latest"], True),
    ("qwen2.5-coder:7b", ["llama3.2:3b", "qwen2.5-coder:7b"], True),
    # Los ':' de un puerto de registry no son un tag.
    ("localhost:5000/patron/modelo", ["localhost:5000/patron/modelo:latest"], True),
    ("localhost:5000/patron/modelo", ["localhost:5000/patron/modelo:q4"], False),
])
def test_modelo_de_ollama_se_compara_con_su_tag(mundo, monkeypatch, configurado, instalados,
                                                esta):
    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "ollama")  # Ollama = motor: `modelo` requerido
    mundo.settings.raw["llm"]["model"] = configurado
    mundo.ollama = json.dumps({"models": [{"name": n} for n in instalados]}).encode()
    modelo = _checks()["modelo"]
    assert modelo.required
    assert modelo.ok is esta, modelo.detail
    if not esta:
        # El fallo dice qué falta, qué hay (para ver el tag equivocado) y cómo arreglarlo.
        assert f"falta '{configurado}'" in modelo.detail
        assert all(n in modelo.detail for n in instalados)
        assert f"ollama pull {configurado}" in modelo.fix


# --- migrate_shortcuts: solo migra atajos que run_shortcut sabe ejecutar -----

@pytest.fixture
def db_temporal(tmp_path, monkeypatch):
    """La SQLite de los atajos en tmp: el script NUNCA toca la DB real."""
    from crotolamo.persistence import db

    monkeypatch.setattr(db, "_default_db_path", lambda: tmp_path / "atajos.sqlite")
    return db


def _json_viejo(tmp_path, data) -> object:
    source = tmp_path / "learned_commands.json"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return source


def test_migrar_atajo_de_proyecto_de_c1_queda_ejecutable(db_temporal, tmp_path, monkeypatch,
                                                         capsys):
    """C1 guardaba {"type": "project"}; run_shortcut de C2 no tiene esa rama, así
    que el atajo migrado contaba en "Migré N" y luego respondía "está mal
    guardado". C2 decidió guardar el nombre de un proyecto como búsqueda
    (docstring de tools/shortcuts.py): la migración lo re-resuelve igual que
    learn_shortcut hoy, y el atajo funciona."""
    from crotolamo.tools import shortcuts
    from scripts import migrate_shortcuts

    source = _json_viejo(tmp_path, {
        "mi juego": {"type": "project", "value": "huevonitis"},
        "mi musica": {"type": "url", "value": "https://open.spotify.com"},
    })
    assert migrate_shortcuts.migrate(source) == 0
    assert "Migré 2 atajo(s)" in capsys.readouterr().out

    guardado = db_temporal.get_shortcut("mi juego")
    assert guardado == {"type": "search", "engine": "google", "query": "huevonitis"}
    assert db_temporal.get_shortcut("mi musica") == {"type": "url",
                                                     "value": "https://open.spotify.com"}

    buscado: list[tuple[str, str]] = []

    def _buscar(query: str, engine: str = "google") -> str:
        buscado.append((query, engine))
        return "Busqué, patrón."

    monkeypatch.setattr(shortcuts, "search_web", _buscar)
    assert shortcuts.run_shortcut("mi juego") == "Busqué, patrón."
    assert buscado == [("huevonitis", "google")]


def test_migrar_salta_tipos_que_run_shortcut_no_ejecuta_y_lo_dice(db_temporal, tmp_path,
                                                                  capsys):
    """Un tipo desconocido no se cuenta como migrado (sería un atajo muerto): se
    salta y se nombra, igual que una entrada que no es un objeto."""
    from scripts import migrate_shortcuts

    source = _json_viejo(tmp_path, {
        "macro": {"type": "macro", "value": "algo"},
        "basura": "no soy un objeto",
        "descargas": {"type": "folder", "value": "descargas"},
    })
    assert migrate_shortcuts.migrate(source) == 0
    out = capsys.readouterr().out
    assert "Migré 1 atajo(s)" in out
    assert "'macro'" in out and "'basura'" in out
    assert set(db_temporal.all_shortcuts()) == {"descargas"}


@pytest.mark.parametrize("data", [[{"alias": "x"}], "texto", 3, None])
def test_migrar_json_que_no_es_un_objeto_da_mensaje_y_no_traceback(db_temporal, tmp_path,
                                                                   capsys, data):
    """`data.items()` sobre una lista reventaba con AttributeError (traceback) en
    vez del mensaje en personaje de los demás errores de lectura."""
    from scripts import migrate_shortcuts

    assert migrate_shortcuts.migrate(_json_viejo(tmp_path, data)) == 1
    out = capsys.readouterr().out
    assert "No pude leer" in out and "patrón" in out
    assert db_temporal.all_shortcuts() == {}


# --- stt_harness: el bench transcribe con el decoder de producción -----------

class _WhisperQueApunta:
    """Doble de faster_whisper.WhisperModel: apunta los kwargs de cada transcribe."""

    def __init__(self) -> None:
        self.llamadas: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.llamadas.append(kwargs)
        seg = types.SimpleNamespace(text="abre spotify", no_speech_prob=0.0, avg_logprob=-0.1)
        return [seg], None


def _wav_silencio(path) -> None:
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)


@pytest.mark.parametrize(("hotwords", "esperados"), [
    ("Crotolamo, Tletl, Huevonitis", "Crotolamo, Tletl, Huevonitis"),
    # Igual que el listener (`voice.get("hotwords") or None`): vacío = sin sesgo.
    ("", None),
    (None, None),
])
def test_bench_transcribe_con_hotwords_y_beam_de_la_config(monkeypatch, tmp_path, hotwords,
                                                           esperados):
    """El listener pasa hotwords=[voice].hotwords y beam_size=[voice].beam_size en
    cada comando; el bench construía STT(...) a mano y transcribía SIN hotwords,
    así que su veredicto (qué modelo da >=90% de intents) medía OTRO decoder,
    justo en los nombres propios que los hotwords existen para arreglar."""
    from crotolamo import settings as settings_mod
    from crotolamo.voice import stt as stt_mod
    from scripts import stt_harness

    voice: dict = {"whisper_model": "small", "beam_size": 2, "sample_rate": 16000}
    if hotwords is not None:
        voice["hotwords"] = hotwords
    # llm/memoria/mcp: el bench enruta con el registry y [llm].max_tools reales.
    monkeypatch.setattr(settings_mod, "get_settings",
                        lambda: types.SimpleNamespace(voice=voice, llm={}, memoria={}, mcp={}))
    elegido, trampa = _WhisperQueApunta(), _WhisperQueApunta()
    # Solo cambia el modelo: el bench pide "tiny" aunque la config diga "small".
    monkeypatch.setitem(stt_mod._models, "tiny", elegido)
    monkeypatch.setitem(stt_mod._models, "small", trampa)
    _wav_silencio(tmp_path / "cmd_01.wav")

    result = stt_harness._bench_model(
        "tiny", [{"wav": "cmd_01.wav", "expected_tool": None}], tmp_path)

    assert result["model"] == "tiny" and trampa.llamadas == []
    assert len(elegido.llamadas) == 2  # warmup + la medición
    for kwargs in elegido.llamadas:
        assert kwargs["hotwords"] == esperados
        assert kwargs["beam_size"] == 2


# --- smoke_voz / wake_debug: "FALLO: ..." en vez de traceback ----------------

class _PortAudioError(Exception):
    """Como sounddevice.PortAudioError: hereda de Exception, NO de OSError."""


def _sounddevice_sin_mic() -> types.ModuleType:
    """Un sounddevice que importa bien pero sin micrófono (PortAudioError al abrir)."""
    sd = types.ModuleType("sounddevice")

    def _input_stream(*args, **kwargs):
        raise _PortAudioError("Error querying device -1")

    sd.PortAudioError = _PortAudioError  # type: ignore[attr-defined]
    sd.InputStream = _input_stream  # type: ignore[attr-defined]
    return sd


class _SinPortAudio:
    """Finder que hace que `import sounddevice` falle como sin libportaudio."""

    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return None


def _sounddevice_que_falla(monkeypatch, como: str) -> None:
    if como == "sin_modulo":
        monkeypatch.setitem(sys.modules, "sounddevice", None)  # ModuleNotFoundError
    elif como == "sin_portaudio":
        monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_SinPortAudio(), *sys.meta_path])
    else:
        monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice_sin_mic())


class _DetectorFalso:
    model_name = "crotolamo"
    threshold = 0.5

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def _get_model(self):
        return object()

    def score(self, audio) -> float:
        return 0.0


@pytest.mark.parametrize("como", ["sin_modulo", "sin_portaudio", "sin_mic"])
def test_wake_debug_sin_microfono_dice_fallo_y_sale_1(monkeypatch, capsys, como):
    """Solo se atrapaba VoiceUnavailable: sin sounddevice (ModuleNotFoundError),
    sin libportaudio (OSError) o sin micrófono (PortAudioError al abrir el
    InputStream) el script soltaba un traceback en vez de "FALLO: ..."."""
    from scripts import wake_debug

    settings = types.SimpleNamespace(voice={}, wake={})
    monkeypatch.setattr(wake_debug, "get_settings", lambda: settings)
    monkeypatch.setattr(wake_debug, "WakeWordDetector", _DetectorFalso)
    _sounddevice_que_falla(monkeypatch, como)

    assert wake_debug.run(None, False) == 1
    out = capsys.readouterr().out
    assert "FALLO" in out
    assert "RESUMEN" not in out  # sin audio no hay medición que resumir


class _TTSFalso:
    voice_model = "voz.onnx"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def available(self) -> bool:
        return True

    def synthesize_pcm(self, text: str):
        import numpy as np

        if self.error is not None:
            raise self.error
        return (np.full(2205, 1000, dtype=np.int16), 22050)


class _STTFalso:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def transcribe(self, audio, hotwords=None) -> str:
        raise self.error

    def listen_once(self, **kwargs) -> str:
        raise self.error


def _smoke_con(monkeypatch, tts, stt):
    from scripts import smoke_voz

    monkeypatch.setattr(smoke_voz, "get_settings", lambda: object())
    monkeypatch.setattr(smoke_voz, "TTS", types.SimpleNamespace(from_settings=lambda s: tts))
    monkeypatch.setattr(smoke_voz, "STT", types.SimpleNamespace(from_settings=lambda s: stt))
    return smoke_voz


def test_smoke_circular_sin_piper_dice_fallo(monkeypatch, capsys):
    """circular() no atrapaba nada: sin piper-tts el ImportError salía crudo."""
    from crotolamo.voice.stt import VoiceUnavailable

    smoke = _smoke_con(monkeypatch, _TTSFalso(ImportError("No module named 'piper'")),
                       _STTFalso(VoiceUnavailable("no debería llegar al STT")))
    assert smoke.circular() == 1
    out = capsys.readouterr().out
    assert "FALLO" in out and "piper" in out


def test_smoke_circular_con_stt_roto_dice_fallo_y_no_deja_el_wav(monkeypatch, tmp_path,
                                                                 capsys):
    """Sin faster-whisper el STT lanza VoiceUnavailable: antes traceback y el WAV
    temporal quedaba huérfano en /tmp (el unlink iba DESPUÉS del transcribe)."""
    import tempfile

    from crotolamo.voice.stt import VoiceUnavailable

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    smoke = _smoke_con(monkeypatch, _TTSFalso(),
                       _STTFalso(VoiceUnavailable("Falta 'faster_whisper', patrón.")))
    assert smoke.circular() == 1
    assert "FALLO: Falta 'faster_whisper'" in capsys.readouterr().out
    assert list(tmp_path.glob("*.wav")) == []


@pytest.mark.parametrize("como", ["sin_modulo", "sin_portaudio", "sin_mic"])
def test_smoke_mic_sin_microfono_dice_fallo(monkeypatch, capsys, como):
    """--mic solo atrapaba VoiceUnavailable: PortAudioError (sin micrófono) y el
    OSError de sounddevice sin libportaudio escapaban como traceback."""
    from crotolamo.voice.stt import VoiceUnavailable

    _sounddevice_que_falla(monkeypatch, como)
    # Lo que suelta el STT real al grabar en cada caso: _require convierte el
    # ImportError en VoiceUnavailable (eso ya se atrapaba), pero el OSError de
    # libportaudio y el PortAudioError de abrir el mic no.
    error = {
        "sin_modulo": VoiceUnavailable("Falta 'sounddevice', patrón."),
        "sin_portaudio": OSError("PortAudio library not found"),
        "sin_mic": _PortAudioError("Error querying device -1"),
    }[como]
    smoke = _smoke_con(monkeypatch, _TTSFalso(), _STTFalso(error))
    assert smoke.with_mic() == 1
    assert "FALLO" in capsys.readouterr().out


# --- scripts de voz sueltos: mismo fallback de sys.path que el doctor --------

def _correr_suelto(tmp_path, script: str, *args: str):
    """Corre el script como lo hace launch.sh sin .venv: `python3 scripts/x.py`,
    con un intérprete que NO tiene crotolamo instalado (-S: sin site-packages)."""
    import os
    import subprocess
    from pathlib import Path

    raiz = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-S", str(raiz / "scripts" / script), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("script", ["smoke_voz.py", "wake_debug.py"])
def test_script_de_voz_suelto_encuentra_el_paquete(tmp_path, script):
    """`./launch.sh smoke` con python3 del sistema moría con ModuleNotFoundError:
    sys.path[0] es scripts/, no la raíz. El doctor y migrate_shortcuts ya caían a
    la raíz del repo; smoke_voz y wake_debug no."""
    proc = _correr_suelto(tmp_path, script, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout and "Traceback" not in proc.stderr


def test_stt_harness_suelto_encuentra_el_paquete(tmp_path):
    """stt_harness importa crotolamo dentro de cada comando: --help andaba, pero
    `bench`/`record` reventaban con ModuleNotFoundError en vez de su "FALLO"."""
    proc = _correr_suelto(tmp_path, "stt_harness.py", "bench", "--dir", str(tmp_path))
    assert proc.returncode == 1, proc.stderr
    assert "FALLO: no encuentro" in proc.stdout and "Traceback" not in proc.stderr
