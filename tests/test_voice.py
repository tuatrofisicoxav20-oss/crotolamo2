"""Fase 5: normalización MX (pura) + que los módulos de voz importen sin las deps."""

import importlib

from crotolamo.voice.normalize import normalize_text


def test_normalize_whisper_errors():
    assert normalize_text("abre you tube") == "abre youtube"
    assert normalize_text("busca en git hub") == "busca en github"
    assert normalize_text("pon espoti fai") == "pon spotify"


def test_normalize_carpeta_intent():
    assert normalize_text("crear una carpeta nueva") == "crea una carpeta nueva"
    assert normalize_text("qué es una carpeta llamada x") == "crea una carpeta llamada x"


def test_normalize_collapses_spaces_and_lowercases():
    assert normalize_text("  ABRE   YouTube  ") == "abre youtube"


def test_voice_modules_import_without_deps():
    # Los imports pesados son perezosos: importar no debe exigir faster-whisper.
    for mod in ("crotolamo.voice.stt", "crotolamo.voice.tts",
                "crotolamo.voice.normalize", "interfaces.listener"):
        assert importlib.import_module(mod) is not None


def test_tts_speak_without_voice_is_graceful(tmp_path, capsys):
    from crotolamo.voice.tts import TTS

    tts = TTS(tmp_path / "no_existe.onnx")
    assert tts.available() is False
    tts.speak("hola")  # no debe lanzar
    assert "voz desactivada" in capsys.readouterr().out


def test_stt_tts_satisfy_protocols(tmp_path):
    # L4: las clases concretas cumplen las interfaces (costura para Wyoming futuro).
    from crotolamo.voice.interfaces import SpeechToText, TextToSpeech
    from crotolamo.voice.stt import STT
    from crotolamo.voice.tts import TTS

    assert isinstance(STT(), SpeechToText)
    assert isinstance(TTS(tmp_path / "v.onnx"), TextToSpeech)


def test_stt_requires_deps_raises_clear_error():
    from crotolamo.voice import stt as stt_mod

    try:
        stt_mod._require("modulo_que_no_existe_xyz")
        raised = False
    except stt_mod.VoiceUnavailable:
        raised = True
    assert raised
