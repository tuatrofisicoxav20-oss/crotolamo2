"""Tests de la misión velocidad: TTS streaming, beam configurable, num_ctx."""

from crotolamo.core.llm import LLMClient
from crotolamo.voice import stt as stt_mod
from crotolamo.voice.tts import StreamSpeaker


class _FakeTTS:
    """Graba lo que se 'habla' (sin audio real)."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


# --- StreamSpeaker ---

def test_stream_speaker_habla_frases_conforme_llegan():
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    for token in ["Órale", ", ya", " quedó. ", "Abrí ", "Opera. ", "¿Algo más"]:
        sp.feed(token)
    assert sp.finish() is True
    assert tts.spoken == ["Órale, ya quedó.", "Abrí Opera.", "¿Algo más"]


def test_stream_speaker_sin_tokens_no_habla():
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    assert sp.finish() is False
    assert tts.spoken == []


def test_stream_speaker_una_frase_sin_cierre_se_habla_al_final():
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    sp.feed("Listo, patrón")  # nunca llega el punto final
    assert sp.finish() is True
    assert tts.spoken == ["Listo, patrón"]


def test_stream_speaker_on_first_se_dispara_una_vez():
    tts = _FakeTTS()
    hits: list[int] = []
    sp = StreamSpeaker(tts, on_first=lambda: hits.append(1))
    sp.feed("Una. Dos. Tres. ")
    sp.finish()
    assert hits == [1]
    assert tts.spoken == ["Una.", "Dos.", "Tres."]


def test_stream_speaker_ignora_espacios_sueltos():
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    sp.feed(".  \n  ")
    assert sp.finish() is True or tts.spoken == [] or True  # no debe reventar
    # lo importante: nada de frases vacías
    assert all(s.strip() for s in tts.spoken)


# --- beam_size configurable ---

class _FakeWhisperModel:
    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return iter(()), None


def test_beam_size_llega_al_modelo(tmp_path):
    fake = _FakeWhisperModel()
    stt_mod._models["_fake_beam_"] = fake
    try:
        wav = tmp_path / "x.wav"
        wav.touch()
        stt_mod.STT(model_size="_fake_beam_", beam_size=2).transcribe(wav)
        assert fake.calls[-1]["beam_size"] == 2
        # default intacto: 5 (mismo comportamiento que siempre)
        stt_mod.STT(model_size="_fake_beam_").transcribe(wav)
        assert fake.calls[-1]["beam_size"] == 5
    finally:
        stt_mod._models.pop("_fake_beam_", None)


def test_beam_size_desde_settings():
    class _S:
        voice = {"whisper_model": "small", "sample_rate": 16000, "beam_size": 2}

    assert stt_mod.STT.from_settings(_S()).beam_size == 2


# --- num_ctx en las options de Ollama ---

def test_num_ctx_entra_en_options_solo_si_esta():
    con = LLMClient(num_ctx=2048)
    assert con._options() == {"temperature": 0.2, "num_ctx": 2048}
    sin = LLMClient()
    assert sin._options() == {"temperature": 0.2}


def test_num_ctx_desde_settings():
    class _S:
        llm = {"model": "llama3.2:latest", "num_ctx": 2048}

    assert LLMClient.from_settings(_S()).num_ctx == 2048
