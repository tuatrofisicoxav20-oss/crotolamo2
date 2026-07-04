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


class _FakeWhisperModel:
    """Modelo falso: registra los kwargs que recibe model.transcribe()."""

    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return iter(()), None  # (segments, info)


def test_transcribe_propaga_hotwords_al_modelo(tmp_path):
    """El wrapper STT.transcribe() propaga `hotwords` a model.transcribe()
    (faster-whisper >= 1.0.2) SIN hardcodearlo: quien no lo pasa — la ruta de
    WAKE difuso llama transcribe(path) a secas — manda hotwords=None (sin
    sesgo, para no inflar falsos despertares)."""
    from crotolamo.voice import stt as stt_mod

    fake = _FakeWhisperModel()
    stt_mod._models["_fake_hotwords_"] = fake
    try:
        stt = stt_mod.STT(model_size="_fake_hotwords_")
        wav = tmp_path / "x.wav"
        wav.touch()

        # Ruta de COMANDOS: el caller pasa las hotwords y llegan al modelo.
        stt.transcribe(wav, hotwords="Crotolamo, Tletl, Huevonitis")
        assert fake.calls[-1]["hotwords"] == "Crotolamo, Tletl, Huevonitis"

        # Ruta de WAKE: transcribe(path) sin argumento -> el modelo recibe
        # hotwords=None (exactamente el comportamiento previo, sin sesgo).
        stt.transcribe(wav)
        assert fake.calls[-1]["hotwords"] is None
        # El resto de la llamada no cambió (anti-alucinación intacta).
        assert fake.calls[-1]["temperature"] == 0.0
        assert fake.calls[-1]["condition_on_previous_text"] is False
    finally:
        stt_mod._models.pop("_fake_hotwords_", None)
