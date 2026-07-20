"""Tests del endpointing inteligente (pausa de pensar vs oración terminada)."""

from crotolamo.voice import stt as stt_mod
from crotolamo.voice.endpoint import seems_incomplete


# --- heurística ---

def test_frases_completas_no_se_extienden():
    for frase in [
        "abre youtube",
        "qué hora es",
        "borra eso",
        "pausa",
        "siguiente",
        "mueve el reporte a descargas",
        "cuánto espacio queda en el disco?",
        "recuérdame que mañana hay tianguis",
    ]:
        assert not seems_incomplete(frase), frase


def test_conectores_al_final_son_pausa_de_pensar():
    for frase in [
        "mueve el archivo de",
        "ábreme la",
        "pon una canción de",
        "busca información sobre",
        "manda el reporte a",
        "crea una nota que diga que",
        "abre el proyecto y",
        "quiero que busques en",
    ]:
        assert seems_incomplete(frase), frase


def test_puntuacion_de_trailing_off():
    assert seems_incomplete("mueve el archivo,")
    assert seems_incomplete("ponme...")
    assert seems_incomplete("busca…")


def test_verbo_transitivo_colgado_solo():
    assert seems_incomplete("abre")
    assert seems_incomplete("mueve")
    assert seems_incomplete("búscame")
    # pero un verbo completo por sí mismo NO
    assert not seems_incomplete("pausa")


def test_acentos_y_mayusculas_no_importan():
    assert seems_incomplete("Ábreme LA")
    assert seems_incomplete("pon música DE")


def test_vacio_no_es_incompleto():
    assert not seems_incomplete("")
    assert not seems_incomplete("   ")


# --- listen_smart ---

class _ScriptedSTT(stt_mod.STT):
    """STT con _listen_transcribe guionizado: devuelve la secuencia dada y
    registra los kwargs de cada llamada (sin mic ni Whisper reales)."""

    def __init__(self, script):
        super().__init__(model_size="fake")
        self.script = list(script)
        self.calls: list[dict] = []

    def _listen_transcribe(self, hotwords=None, **vad_kwargs):
        self.calls.append({"hotwords": hotwords, **vad_kwargs})
        return self.script.pop(0) if self.script else ""


def test_listen_smart_concatena_continuacion():
    stt = _ScriptedSTT(["mueve el archivo de", "huevonitis a descargas"])
    assert stt.listen_smart(rounds=2) == "mueve el archivo de huevonitis a descargas"
    assert len(stt.calls) == 2


def test_listen_smart_completa_no_reabre():
    stt = _ScriptedSTT(["abre youtube"])
    assert stt.listen_smart(rounds=2) == "abre youtube"
    assert len(stt.calls) == 1


def test_listen_smart_continuacion_vacia_se_queda_con_lo_que_hay():
    stt = _ScriptedSTT(["ábreme la", ""])
    assert stt.listen_smart(rounds=2) == "ábreme la"
    assert len(stt.calls) == 2


def test_listen_smart_respeta_max_rounds():
    stt = _ScriptedSTT(["pon una canción de", "los tigres del norte y", "la sonora"])
    text = stt.listen_smart(rounds=2)
    assert text == "pon una canción de los tigres del norte y la sonora"
    assert len(stt.calls) == 3  # 1 inicial + 2 rondas máximo


def test_listen_smart_rounds_cero_es_listen_once():
    stt = _ScriptedSTT(["mueve el archivo de"])
    assert stt.listen_smart(rounds=0) == "mueve el archivo de"
    assert len(stt.calls) == 1


def test_listen_smart_usa_timeout_corto_en_continuacion():
    stt = _ScriptedSTT(["ábreme la", "carpeta de descargas"])
    stt.listen_smart(rounds=2, start_timeout_s=6.0, continue_timeout_s=2.5,
                     hotwords="Crotolamo")
    assert stt.calls[0]["start_timeout_s"] == 6.0
    assert stt.calls[1]["start_timeout_s"] == 2.5
    # hotwords llegan a TODAS las escuchas del comando
    assert all(c["hotwords"] == "Crotolamo" for c in stt.calls)


# --- min_audio_s: silencio puro NO se transcribe (mic sordo mínimo) ---

def test_min_audio_salta_whisper_en_silencio(monkeypatch):
    import numpy as np

    stt = stt_mod.STT(model_size="fake")

    def _fake_record(**kwargs):
        # 1 muestra: lo que produce _frames_to_audio cuando nadie habló
        return np.zeros(1, dtype=np.float32)

    def _boom(audio, hotwords=None):
        raise AssertionError("no debía transcribir silencio")

    monkeypatch.setattr(stt, "record_audio_until_silence", _fake_record)
    monkeypatch.setattr(stt, "transcribe", _boom)
    assert stt.listen_once(min_audio_s=0.3) == ""


def test_min_audio_cero_transcribe_como_siempre(monkeypatch):
    import numpy as np

    stt = stt_mod.STT(model_size="fake")
    audio = np.zeros(16000, dtype=np.float32)  # 1s de audio

    monkeypatch.setattr(stt, "record_audio_until_silence", lambda **k: audio)
    monkeypatch.setattr(stt, "transcribe", lambda a, hotwords=None: "hola")
    assert stt.listen_once(min_audio_s=0.3) == "hola"
