"""Tests de la misión velocidad: TTS streaming, beam configurable, num_ctx."""

import random

from crotolamo.core.llm import LLMClient
from crotolamo.voice import stt as stt_mod
from crotolamo.voice import tts as tts_mod
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
    sp.finish()  # no debe reventar
    # lo importante: nada de frases vacías. Un "." suelto sí puede llegar al TTS:
    # limpiar_para_voz lo deja en "" y TTS.speak no suena nada.
    assert all(s.strip() for s in tts.spoken)
    assert tts.spoken in ([], ["."])


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


# --- Segmentación unificada (T4): decimales y abreviaturas no parten frase ---

def test_split_sentences_no_parte_decimales_ni_abreviaturas():
    from crotolamo.voice.tts import split_sentences

    assert split_sentences("Tienes 3.5 GB, patrón. Y 40. 5 más.") == [
        "Tienes 3.5 GB, patrón.", "Y 40. 5 más."]
    assert split_sentences("Llama al Dr. López.") == ["Llama al Dr. López."]
    assert split_sentences("Primero esto. Luego lo otro.") == [
        "Primero esto.", "Luego lo otro."]
    assert split_sentences("Hay manzanas, peras, etc. en el frutero.") == [
        "Hay manzanas, peras, etc. en el frutero."]
    assert split_sentences("La cita es en la Av. Juárez. Llega temprano.") == [
        "La cita es en la Av. Juárez.", "Llega temprano."]
    # "No. 5" es número; "Claro que no." sí cierra frase.
    assert split_sentences("Es el No. 5 de la lista.") == ["Es el No. 5 de la lista."]
    assert split_sentences("Claro que no. Pero lo intento.") == [
        "Claro que no.", "Pero lo intento."]


def test_stream_speaker_no_parte_cifras_a_medias():
    """En streaming, un '.' al final del buffer no se corta hasta VER el
    siguiente carácter: puede ser un decimal partido ('40.' + ' 5 GB')."""
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    for token in ["Tienes 40.", " 5 GB libres.", " Listo."]:
        sp.feed(token)
    assert sp.finish() is True
    assert tts.spoken == ["Tienes 40. 5 GB libres.", "Listo."]


def test_stream_speaker_no_parte_abreviaturas():
    tts = _FakeTTS()
    sp = StreamSpeaker(tts)
    for token in ["Llama al Dr.", " López.", " Ya le avisé."]:
        sp.feed(token)
    assert sp.finish() is True
    assert tts.spoken == ["Llama al Dr. López.", "Ya le avisé."]


# --- SentenceFeeder: UNA segmentación en streaming (la de StreamSpeaker, tal cual) ---
# Causa raíz voice-pipeline-forked-audio-owned-per-call, paso 5 del diseño: la
# usan StreamSpeaker (loop simple) y el BrainThread ([voice].stream_speak_concurrent).

def _ref_stream_speaker(tokens: list[str]) -> list[str]:
    """Copia LITERAL de StreamSpeaker.feed/finish antes del paso (la referencia de
    "mismas frases en el mismo orden"), con la memoria de lista de la respuesta
    (revisión final x1voz): sin ella, el buffer que se reinicia tras cada corte
    tomaba por viñeta el número que abre una frase a media línea."""
    buffer = ""
    frases: list[str] = []
    lista = tts_mod._Lista()
    for token in tokens:
        buffer += token
        pieces, buffer = tts_mod._split(buffer, final=False, lista=lista)
        frases.extend(pieces)
    tail = buffer.strip()
    if tail:
        frases.append(tail)
    return frases


_TROZOS = ["Hola", "patrón", "Dr", "40", "5", "GB", "etc", "No", "Av", "ya", "1", "12",
           ".", ".", ". ", " ", " ", "\n", "!", "?", "¿", "¡", ",", "**", "_", "á", "ñ"]


def _tokens_al_azar(rng: random.Random) -> list[str]:
    texto = "".join(rng.choice(_TROZOS) + rng.choice(["", " "]) for _ in range(rng.randint(1, 30)))
    tokens, i = [], 0
    while i < len(texto):
        n = rng.randint(1, 6)
        tokens.append(texto[i:i + n])
        i += n
    return tokens


def test_sentence_feeder_corta_igual_que_stream_speaker_antes():
    rng = random.Random(20261114)
    for _ in range(3000):
        tokens = _tokens_al_azar(rng)
        frases: list[str] = []
        feeder = tts_mod.SentenceFeeder(frases.append)
        for token in tokens:
            feeder.feed(token)
        feeder.flush()
        assert frases == _ref_stream_speaker(tokens), tokens
        assert feeder.fed == len(frases)


class _TtsQueAnota:
    """TTS falso que anota lo hablado y lo pedido a la pre-síntesis, en orden."""

    def __init__(self) -> None:
        self.eventos: list[tuple[str, str]] = []

    def speak(self, text: str) -> None:
        self.eventos.append(("habla", text))

    def prepare(self, text: str) -> None:
        self.eventos.append(("pide", text))


def test_stream_speaker_habla_lo_mismo_y_pide_cada_frase_al_encolarla():
    rng = random.Random(7)
    for _ in range(200):
        tokens = _tokens_al_azar(rng)
        tts = _TtsQueAnota()
        sp = StreamSpeaker(tts)  # type: ignore[arg-type]
        for token in tokens:
            sp.feed(token)
        sp.finish()
        esperado = _ref_stream_speaker(tokens)
        assert [t for e, t in tts.eventos if e == "habla"] == esperado
        assert [t for e, t in tts.eventos if e == "pide"] == esperado
        # Cada frase se pide antes de hablarla.
        for frase in esperado:
            assert tts.eventos.index(("pide", frase)) < tts.eventos.index(("habla", frase))
