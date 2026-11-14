"""Revisión final u53v — scripts/stt_harness.py y scripts/wake_debug.py.

- bench transcribe el audio EN MEMORIA (float32), como producción, no la ruta del
  WAV: con av 19 (Python >= 3.12) faster-whisper revienta al abrir el archivo.
- bench enruta como producción: con el registry real y [llm].max_tools.
- bench: un modelo que falla o un manifest roto dan "FALLO: ...", no traceback, y
  los modelos ya medidos conservan su veredicto.
- record: sin micrófono o con [voice].input_device inexistente, "FALLO: ...".
- la regla del veredicto (>=90% intents, media < 3 s, gana el más chico).
- wake_debug: un input_device con NOMBRE que no existe (ValueError) es "FALLO".
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import wave

import numpy as np
import pytest


# --- dobles -----------------------------------------------------------------

class _WhisperComoAv19:
    """Doble de faster_whisper.WhisperModel. Con una RUTA falla como faster-whisper
    1.2 sobre av 19 (decode_audio -> av.open(metadata_errors=...)); con un ndarray
    transcribe `texto` y apunta lo que recibió."""

    def __init__(self, texto: str = "abre spotify") -> None:
        self.texto = texto
        self.audios: list = []

    def transcribe(self, audio, **kwargs):
        if isinstance(audio, str):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        self.audios.append(audio)
        seg = types.SimpleNamespace(text=self.texto, no_speech_prob=0.0, avg_logprob=-0.1)
        return [seg], None


def _settings(voice: dict | None = None, llm: dict | None = None):
    """Settings mínimos: sin memoria semántica ni MCP (lo de este equipo)."""
    return types.SimpleNamespace(
        voice={"beam_size": 1, "sample_rate": 16000, **(voice or {})},
        llm=llm if llm is not None else {"max_tools": 8},
        memoria={}, mcp={}, wake={},
    )


def _con_settings(monkeypatch, settings) -> None:
    from crotolamo import settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_settings", lambda: settings)


def _wav(path, muestras: np.ndarray, rate: int = 16000, canales: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(canales)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(muestras.astype("<i2").tobytes())


def _muestras(n: int = 1600, semilla: int = 3) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.integers(-20000, 20000, size=n).astype(np.int16)


def _bench_dir(tmp_path, entradas: list[tuple[str, str | None]]):
    """Un wav por entrada + manifest.json, como lo deja `record`."""
    manifest = []
    for i, (_, esperada) in enumerate(entradas, 1):
        nombre = f"cmd_{i:02d}.wav"
        _wav(tmp_path / nombre, _muestras(semilla=i))
        manifest.append({"wav": nombre, "expected_tool": esperada})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _modelos(monkeypatch, **modelos) -> None:
    """Modelos falsos ya "cargados": el bench nunca toca faster-whisper real."""
    from crotolamo.voice import stt as stt_mod

    for nombre, modelo in modelos.items():
        monkeypatch.setitem(stt_mod._models, nombre, modelo)


def _bench(tmp_path, models: str) -> int:
    from scripts import stt_harness

    return stt_harness.cmd_bench(argparse.Namespace(dir=str(tmp_path), models=models))


# --- 1. el audio va en memoria, como en producción -----------------------------

def test_bench_transcribe_el_audio_en_memoria_no_la_ruta(monkeypatch, tmp_path, capsys):
    """El bench pasaba la RUTA del wav a stt.transcribe: faster-whisper la abre con
    PyAV, y con av 19 (lo que pip resuelve en Python >= 3.12) eso es TypeError en
    el warmup del PRIMER modelo; no se medía ninguno. Producción nunca decodifica
    un archivo: transcribe el float32 en memoria (threads.SttThread)."""
    _con_settings(monkeypatch, _settings())
    modelo = _WhisperComoAv19()
    _modelos(monkeypatch, tiny=modelo)
    _bench_dir(tmp_path, [("abre spotify", None)])

    assert _bench(tmp_path, "tiny") == 0
    out = capsys.readouterr().out
    assert "FALLO" not in out and "Sugerido" not in out  # sin esperada: no hay intents

    assert len(modelo.audios) == 2  # warmup + la medición, ambos en memoria
    with wave.open(str(tmp_path / "cmd_01.wav"), "rb") as w:
        crudo = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    for audio in modelo.audios:
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert audio.ndim == 1
        np.testing.assert_allclose(audio, crudo / 32768.0, rtol=0, atol=1e-7)


def test_bench_model_directo_tambien_lee_el_wav_a_memoria(monkeypatch, tmp_path):
    """_bench_model sin audios precargados (quien lo llama suelto) los lee él."""
    from scripts import stt_harness

    _con_settings(monkeypatch, _settings())
    modelo = _WhisperComoAv19()
    _modelos(monkeypatch, tiny=modelo)
    _bench_dir(tmp_path, [("abre spotify", None)])

    result = stt_harness._bench_model(
        "tiny", [{"wav": "cmd_01.wav", "expected_tool": None}], tmp_path)
    assert result["rows"][0]["text"] == "abre spotify"
    assert all(isinstance(a, np.ndarray) for a in modelo.audios)


@pytest.mark.parametrize("como", ["estereo", "48k", "basura"])
def test_bench_wav_ilegible_es_fallo_antes_de_cargar_modelos(monkeypatch, tmp_path, capsys,
                                                             como):
    """Un wav que no es el mono 16 kHz int16 que graba `record` (o no es wav) se
    dice ANTES de cargar ningún modelo: Whisper en memoria asume 16 kHz mono."""
    _con_settings(monkeypatch, _settings())
    modelo = _WhisperComoAv19()
    _modelos(monkeypatch, tiny=modelo)
    _bench_dir(tmp_path, [("abre spotify", None)])
    wav = tmp_path / "cmd_01.wav"
    if como == "estereo":
        _wav(wav, _muestras(3200), canales=2)
    elif como == "48k":
        _wav(wav, _muestras(), rate=48000)
    else:
        wav.write_bytes(b"esto no es un wav")

    assert _bench(tmp_path, "tiny") == 1
    out = capsys.readouterr().out
    assert "FALLO" in out and "cmd_01.wav" in out
    assert modelo.audios == []


# --- 2. el routing es el de producción -------------------------------------------

def test_bench_enruta_con_el_registry_como_produccion(monkeypatch, tmp_path, capsys):
    """El bench llamaba select_tool_names(cmd) sin available=: las tools de la
    memoria semántica (sin mem0 no están registradas) se comían huecos del tope y
    dejaban fuera tools reales. Producción ofrece music_control para esta orden."""
    from crotolamo.core.router import route_schemas
    from crotolamo.tools import default_registry

    texto = "recuerdas que te dije que abriera spotify"
    _con_settings(monkeypatch, _settings())
    _modelos(monkeypatch, tiny=_WhisperComoAv19(texto))
    _bench_dir(tmp_path, [(texto, "music_control")])

    produccion = [s["function"]["name"] for s in route_schemas(default_registry(), texto, 8)]
    assert "music_control" in produccion  # la premisa: producción sí la ofrece

    assert _bench(tmp_path, "tiny") == 0
    out = capsys.readouterr().out
    assert "intents OK 1/1" in out and "Sugerido: 'tiny'" in out


@pytest.mark.parametrize(("max_tools", "esperado"), [(2, "intents OK 0/1"),
                                                     (8, "intents OK 1/1")])
def test_bench_respeta_max_tools_de_la_config(monkeypatch, tmp_path, capsys, max_tools,
                                             esperado):
    """El tope era el default del router, no [llm].max_tools: con un tope chico en
    la config el bench daba OK a una tool que producción nunca ofrece."""
    from crotolamo.core.router import route_schemas
    from crotolamo.tools import default_registry

    texto = "pon musica y abre firefox"
    _con_settings(monkeypatch, _settings(llm={"max_tools": max_tools}))
    _modelos(monkeypatch, tiny=_WhisperComoAv19(texto))
    _bench_dir(tmp_path, [(texto, "open_app")])

    produccion = [s["function"]["name"]
                  for s in route_schemas(default_registry(), texto, max_tools)]
    assert ("open_app" in produccion) == (max_tools == 8)

    _bench(tmp_path, "tiny")
    assert esperado in capsys.readouterr().out


def test_record_lista_solo_las_tools_registradas(monkeypatch, tmp_path, capsys):
    """`record` listaba como "Tools válidas" las de la memoria semántica aunque no
    estén registradas (sin mem0): elegir una daba un OK que producción no da."""
    from scripts import stt_harness

    class _Alto(Exception):
        """Corta en el primer prompt: aquí solo importa la lista impresa."""

    def _input(*args):
        raise _Alto

    _con_settings(monkeypatch, _settings())
    monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice_que_lanza(_PortAudioError))
    monkeypatch.setattr("builtins.input", _input)

    with pytest.raises(_Alto):
        stt_harness.cmd_record(argparse.Namespace(dir=str(tmp_path / "b"), n=1,
                                                  max_seconds=1.0, force=False))
    lista = capsys.readouterr().out.split("Tools válidas", 1)[1].splitlines()[1]
    nombres = [n.strip() for n in lista.split(",")]
    assert "music_control" in nombres and "remember_fact" in nombres
    assert not {"recordar_de_mi", "buscar_recuerdos", "olvidar_recuerdo"} & set(nombres)


# --- 3. bench: errores en personaje, sin perder lo medido -----------------------

def test_bench_modelo_que_falla_no_pierde_los_ya_medidos(monkeypatch, tmp_path, capsys):
    """Un typo en --models ('smal'), o un modelo sin caché y sin red, reventaba con
    traceback DESPUÉS de minutos de CPU: sin veredicto ni líneas de bitácora."""
    from crotolamo.voice import stt as stt_mod

    def _carga_que_falla(fw, size, **kwargs):
        raise ValueError(f"Invalid model size '{size}', expected one of: tiny, base")

    monkeypatch.setattr(stt_mod.runtime, "load_whisper", _carga_que_falla)
    monkeypatch.delitem(stt_mod._models, "smal", raising=False)
    _con_settings(monkeypatch, _settings())
    _modelos(monkeypatch, tiny=_WhisperComoAv19("pon musica"))
    _bench_dir(tmp_path, [("pon musica", "music_control")])

    assert _bench(tmp_path, "tiny,smal") == 1  # hubo un FALLO
    out = capsys.readouterr().out
    assert "FALLO: modelo 'smal': ValueError: Invalid model size 'smal'" in out
    assert "=== veredicto ===" in out and "Sugerido: 'tiny'" in out
    bitacora = out.split("Para la bitácora:", 1)[1]
    assert "tiny:" in bitacora and "smal" not in bitacora


def test_bench_ningun_modelo_medible_es_fallo(monkeypatch, tmp_path, capsys):
    from crotolamo.voice import stt as stt_mod

    def _carga_que_falla(fw, size, **kwargs):
        raise OSError("sin red y sin caché")

    monkeypatch.setattr(stt_mod.runtime, "load_whisper", _carga_que_falla)
    monkeypatch.delitem(stt_mod._models, "smal", raising=False)
    _con_settings(monkeypatch, _settings())
    _bench_dir(tmp_path, [("pon musica", "music_control")])

    assert _bench(tmp_path, "smal") == 1
    out = capsys.readouterr().out
    assert "FALLO: modelo 'smal': OSError: sin red y sin caché" in out
    assert "Sugerido" not in out and "Para la bitácora" not in out


@pytest.mark.parametrize("contenido", [
    b"[{\"wav\": \"cmd_01.wav\",}",                    # JSON roto a mano
    b"\xff\xfe no es utf-8",                            # bytes que no son texto
    b"{\"wav\": \"cmd_01.wav\"}",                       # objeto, no lista
    b"[\"cmd_01.wav\"]",                                # lista de cadenas
    b"[{\"expected_tool\": \"music_control\"}]",        # entrada sin 'wav'
    b"[{\"wav\": 7}]",                                  # 'wav' que no es nombre
])
def test_bench_manifest_roto_es_fallo_en_personaje(monkeypatch, tmp_path, capsys, contenido):
    _con_settings(monkeypatch, _settings())
    modelo = _WhisperComoAv19()
    _modelos(monkeypatch, tiny=modelo)
    _bench_dir(tmp_path, [("abre spotify", None)])
    (tmp_path / "manifest.json").write_bytes(contenido)

    assert _bench(tmp_path, "tiny") == 1
    out = capsys.readouterr().out
    assert "FALLO" in out and "manifest" in out
    assert modelo.audios == []


# --- 4. record: sin micrófono, "FALLO" ------------------------------------------

class _PortAudioError(Exception):
    """Como sounddevice.PortAudioError: hereda de Exception, NO de OSError."""


def _sounddevice_que_lanza(error_cls, mensaje: str = "Error querying device -1"):
    sd = types.ModuleType("sounddevice")

    def _input_stream(*args, **kwargs):
        raise error_cls(mensaje)

    sd.PortAudioError = _PortAudioError  # type: ignore[attr-defined]
    sd.InputStream = _input_stream  # type: ignore[attr-defined]
    return sd


class _SinPortAudio:
    """Finder que hace que `import sounddevice` falle como sin libportaudio."""

    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return None


@pytest.mark.parametrize("como", ["sin_mic", "dispositivo_inexistente", "sin_portaudio",
                                  "sin_modulo"])
def test_record_sin_microfono_dice_fallo(monkeypatch, tmp_path, capsys, como):
    """Solo se atrapaba VoiceUnavailable: sin micrófono (PortAudioError), con un
    [voice].input_device con nombre que no está (ValueError de sounddevice) o sin
    libportaudio (OSError), `record` soltaba un traceback en vez de "FALLO"."""
    from scripts import stt_harness

    voice = {"vad_backend": "energy"}
    if como == "dispositivo_inexistente":
        voice["input_device"] = "crotolamo_aec_source"
    _con_settings(monkeypatch, _settings(voice=voice))
    if como == "sin_mic":
        monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice_que_lanza(_PortAudioError))
    elif como == "dispositivo_inexistente":
        monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice_que_lanza(
            ValueError, "No input device matching 'crotolamo_aec_source'"))
    elif como == "sin_portaudio":
        monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_SinPortAudio(), *sys.meta_path])
    else:
        monkeypatch.setitem(sys.modules, "sounddevice", None)  # ModuleNotFoundError
    monkeypatch.setattr("builtins.input", lambda *a: "")
    out_dir = tmp_path / "bench"

    assert stt_harness.cmd_record(argparse.Namespace(
        dir=str(out_dir), n=2, max_seconds=1.0, force=False)) == 1
    out = capsys.readouterr().out
    assert "FALLO" in out
    assert not (out_dir / "manifest.json").exists()


# --- 5. la regla del veredicto ---------------------------------------------------

@pytest.mark.parametrize(("ok", "total", "media", "pasa"), [
    (9, 10, 2.99, True),    # justo 90% y bajo 3 s
    (10, 10, 3.0, False),   # media de 3 s ya no: "< 3 s"
    (8, 10, 0.5, False),    # 80%
    (0, 0, 0.1, False),     # sin esperadas no hay forma de validar intents
])
def test_passes_rule_umbrales(ok, total, media, pasa):
    from scripts import stt_harness

    assert stt_harness._passes_rule(
        {"ok": ok, "total": total, "mean_latency": media}) is pasa


def test_bench_sugiere_el_mas_chico_que_cumple(monkeypatch, tmp_path, capsys):
    """tiny falla el intent; base, small y uno desconocido cumplen: gana base (el
    más chico conocido); los desconocidos van al final del orden de tamaño."""
    _con_settings(monkeypatch, _settings())
    _modelos(monkeypatch, tiny=_WhisperComoAv19("hola"),
             base=_WhisperComoAv19("pon musica"), small=_WhisperComoAv19("pon musica"))
    _modelos(monkeypatch, **{"x-custom": _WhisperComoAv19("pon musica")})
    _bench_dir(tmp_path, [("pon musica", "music_control")])

    assert _bench(tmp_path, "x-custom,small,tiny,base") == 0
    out = capsys.readouterr().out
    assert "Sugerido: 'base'" in out


def test_bench_desconocido_va_despues_de_los_conocidos(monkeypatch, tmp_path, capsys):
    _con_settings(monkeypatch, _settings())
    _modelos(monkeypatch, small=_WhisperComoAv19("pon musica"))
    _modelos(monkeypatch, **{"x-custom": _WhisperComoAv19("pon musica")})
    _bench_dir(tmp_path, [("pon musica", "music_control")])

    assert _bench(tmp_path, "x-custom,small") == 0
    assert "Sugerido: 'small'" in capsys.readouterr().out


@pytest.mark.parametrize("esperada", ["music_control", None])
def test_bench_ningun_modelo_cumple(monkeypatch, tmp_path, capsys, esperada):
    """Con todos fallando el intent, o sin ninguna esperada (total == 0), no se
    sugiere nada."""
    _con_settings(monkeypatch, _settings())
    _modelos(monkeypatch, tiny=_WhisperComoAv19("hola"), base=_WhisperComoAv19("hola"))
    _bench_dir(tmp_path, [("pon musica", esperada)])

    assert _bench(tmp_path, "tiny,base") == 0
    out = capsys.readouterr().out
    assert "Ningún modelo cumple" in out and "Sugerido" not in out


# --- 6. wake_debug: input_device con nombre inexistente --------------------------

class _DetectorFalso:
    model_name = "crotolamo"
    threshold = 0.5

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def _get_model(self):
        return object()

    def score(self, audio) -> float:
        if self.error is not None:
            raise self.error
        return 0.0

    def stream_score(self, audio):
        return self.score(audio), True


class _StreamUnFrame:
    """InputStream falso que abre bien y sirve un frame de silencio."""

    def __init__(self, *, samplerate, channels, dtype, device=None) -> None:
        self.dtype = dtype

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        return np.zeros((n, 1), dtype=self.dtype), False


def _wake_debug(monkeypatch, detector, sd):
    from scripts import wake_debug

    settings = types.SimpleNamespace(voice={"input_device": "crotolamo_aec_source"}, wake={})
    monkeypatch.setattr(wake_debug, "get_settings", lambda: settings)
    monkeypatch.setattr(wake_debug.WakeWordDetector, "from_settings",
                        classmethod(lambda cls, s: detector))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return wake_debug


@pytest.mark.parametrize("como_produccion", [False, True])
def test_wake_debug_input_device_inexistente_dice_fallo(monkeypatch, capsys, como_produccion):
    """[voice].input_device = "crotolamo_aec_source" con el AEC apagado: sounddevice
    lanza ValueError ('No input device matching'), no PortAudioError, y el script
    de calibración moría con traceback aunque su comentario decía cubrirlo."""
    sd = _sounddevice_que_lanza(ValueError, "No input device matching 'crotolamo_aec_source'")
    wake_debug = _wake_debug(monkeypatch, _DetectorFalso(), sd)

    assert wake_debug.run(None, False, como_produccion=como_produccion) == 1
    out = capsys.readouterr().out
    assert "FALLO: el micrófono no respondió" in out and "crotolamo_aec_source" in out
    assert "RESUMEN" not in out


def test_wake_debug_valueerror_de_la_medicion_no_se_disfraza_de_microfono(monkeypatch):
    """El ValueError que se atrapa es el de ABRIR el micrófono; uno que salga ya
    midiendo (un bug del detector) se ve tal cual, no como "el micrófono no
    respondió"."""
    sd = types.ModuleType("sounddevice")
    sd.PortAudioError = _PortAudioError  # type: ignore[attr-defined]
    sd.InputStream = _StreamUnFrame  # type: ignore[attr-defined]
    wake_debug = _wake_debug(monkeypatch, _DetectorFalso(ValueError("forma rara")), sd)

    with pytest.raises(ValueError, match="forma rara"):
        wake_debug.run(None, False)
