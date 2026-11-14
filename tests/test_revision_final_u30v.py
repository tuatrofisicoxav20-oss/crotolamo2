"""Revisión final, unidad 30v: crotolamo/voice/endpoint.py (y clean.py/interfaces.py).

1. Los interrogativos TÓNICOS cierran la frase. _norm quita la tilde y "qué",
   "cómo", "cuándo", "dónde" quedaban iguales a los conectores átonos "que",
   "como", "cuando", "donde": "¿Por qué?", "¿a dónde?", "¿hasta cuándo?" o
   "dime cómo" contaban como frase a medias y listen_smart reabría el micro
   endpoint_continue_s (2.5 s) y pegaba a la orden lo que sonara en esa
   ventana ("¿Por qué? ruido de la tele"). "¿Cómo?" y "¿Qué?" a secas solo se
   salvaban por accidente (el "¿" quedaba pegado a la palabra): ahora una
   pregunta o exclamación de UNA palabra se reconoce a propósito.

2. Un bloque de código se leía en voz alta línea a línea. limpiar_para_voz solo
   descarta una cerca ```…``` que le llega ENTERA, pero los tres caminos que
   hablan segmentan antes (speak_sentences, BrainThread con split_sentences +
   speak, StreamSpeaker/SentenceFeeder) y la segmentación cortaba DENTRO del
   bloque: en cada línea en blanco, cada sangría tras un salto y cada '.\n'.
   Cada trozo traía una cerca suelta, _CERCA_SUELTA quitaba el marcador y el
   código sonaba ("import json", "with open..."). El arreglo vive en
   crotolamo/voice/tts.py (_split no corta con un número impar de cercas
   antes del candidato) y crotolamo/core/streaming.py (la sonda incremental
   de "sentence" lleva la misma paridad): fuera de esta unidad.
"""

from __future__ import annotations

import unicodedata

import pytest

from crotolamo.voice import stt as stt_mod
from crotolamo.voice.clean import limpiar_para_voz
from crotolamo.voice.endpoint import seems_incomplete
from crotolamo.voice.tts import TTS, StreamSpeaker, split_sentences

# --- 1. interrogativos tónicos ------------------------------------------------

_PREGUNTAS_CERRADAS = [
    "¿Por qué?",
    "¿y por qué?",
    "¿Y por qué?",
    "¿Para qué?",
    "¿Y eso qué?",
    "¿Y eso para qué?",
    "¿a dónde?",
    "¿A dónde?",
    "¿hasta cuándo?",
    "¿Desde cuándo?",
    "dime cómo",
    "no sé cómo",
    "no sé por qué",
    "¿por QUÉ?",
]


@pytest.mark.parametrize("frase", _PREGUNTAS_CERRADAS)
def test_interrogativo_tonico_cierra_la_frase(frase):
    assert not seems_incomplete(frase), frase


def test_interrogativo_tonico_en_nfd_tambien_cierra():
    """Whisper puede entregar la tilde como carácter combinante suelto."""
    for frase in ("¿Por qué?", "¿a dónde?", "¿hasta cuándo?", "dime cómo"):
        nfd = unicodedata.normalize("NFD", frase)
        assert nfd != frase
        assert not seems_incomplete(nfd), frase


def test_la_forma_atona_sigue_siendo_pausa_de_pensar():
    """Sin tilde es el conector: "dime que", "pon la de cuando" siguen abiertas."""
    for frase in ("dime que", "avísame cuando", "ponla como", "llévalo donde"):
        assert seems_incomplete(frase), frase
    # La coma de trailing-off manda aunque la palabra cierre.
    assert seems_incomplete("¿por qué,")


@pytest.mark.parametrize("frase", ["¿Cómo?", "¿Qué?", "¿Y?", "¡Abre!", "¿Dónde?", "¿Como?"])
def test_pregunta_de_una_palabra_cierra_a_proposito(frase):
    """Una pregunta o exclamación de una sola palabra ya está abierta y cerrada
    por Whisper (¿…? / ¡…!): no reabre, sea conector, verbo colgado o tónica."""
    assert not seems_incomplete(frase), frase


class _ScriptedSTT(stt_mod.STT):
    """STT con _listen_transcribe guionizado (sin mic ni Whisper reales)."""

    def __init__(self, script):
        super().__init__(model_size="fake")
        self.script = list(script)
        self.calls: list[dict] = []

    def _listen_transcribe(self, hotwords=None, **vad_kwargs):
        self.calls.append({"hotwords": hotwords, **vad_kwargs})
        return self.script.pop(0) if self.script else ""


@pytest.mark.parametrize("respuesta", ["¿Por qué?", "¿a dónde?", "¿y por qué?", "dime cómo"])
def test_listen_smart_no_reabre_tras_un_interrogativo_tonico(respuesta):
    """Antes: una segunda grabación (2.5 s más de espera) y "¿Por qué? ruido de
    la tele" como orden."""
    stt = _ScriptedSTT([respuesta, "ruido de la tele"])
    assert stt.listen_smart(rounds=2) == respuesta
    assert len(stt.calls) == 1


# --- 2. un bloque de código llega ENTERO a la limpieza --------------------------

_RESPUESTAS_CON_CODIGO = {
    "linea en blanco": (
        "Así lees un archivo en Python:\n\n```python\nimport json\n\n"
        "with open('datos.json') as f:\n    datos = json.load(f)\n```\n\nListo, patrón."
    ),
    "sangria sin linea en blanco": "Así:\n```python\nfor x in lista:\n    print(x)\n```\nListo.",
    "def con return": "Así:\n```python\ndef suma(a, b):\n    return a + b\n```\nListo.",
    "comentario con punto": (
        "Así:\n```python\n# Lee el archivo.\ndatos = open('x').read()\n```\nListo."
    ),
    "bash tras frase cerrada": (
        "Corre esto.\n\n```bash\n# Instala el paquete.\nsudo dnf install x\n```\n\nListo."
    ),
}
# Palabras que solo están DENTRO de los bloques: ninguna debe sonar.
_CODIGO = ("import", "json", "datos", "print", "return", "suma", "sudo", "dnf", "Instala",
           "Lee el archivo")


@pytest.fixture
def tts_que_anota(tmp_path):
    """TTS real (limpieza, turno de voz) con el audio cambiado por una lista."""
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    hablado: list[str] = []
    tts._speak_streaming = lambda t, _abort=None: hablado.append(t) or True
    return tts, hablado


def _sin_codigo(hablado, respuesta):
    dicho = " ".join(hablado)
    assert not [w for w in _CODIGO if w in dicho], hablado
    # Segmentar no cambia lo que se dice: lo mismo que la respuesta de una pieza.
    assert dicho == limpiar_para_voz(respuesta), hablado


@pytest.mark.parametrize("respuesta", _RESPUESTAS_CON_CODIGO.values(),
                         ids=_RESPUESTAS_CON_CODIGO.keys())
def test_de_una_pieza_el_bloque_ya_se_descartaba(respuesta, tts_que_anota):
    """La referencia: speak() con la respuesta entera nunca leyó el código."""
    tts, hablado = tts_que_anota
    tts.speak(respuesta)
    assert len(hablado) == 1
    assert not [w for w in _CODIGO if w in hablado[0]], hablado


@pytest.mark.parametrize("respuesta", _RESPUESTAS_CON_CODIGO.values(),
                         ids=_RESPUESTAS_CON_CODIGO.keys())
def test_speak_sentences_no_lee_el_codigo(respuesta, tts_que_anota):
    tts, hablado = tts_que_anota
    tts.speak_sentences(respuesta)
    _sin_codigo(hablado, respuesta)


@pytest.mark.parametrize("respuesta", _RESPUESTAS_CON_CODIGO.values(),
                         ids=_RESPUESTAS_CON_CODIGO.keys())
def test_split_sentences_mas_speak_no_lee_el_codigo(respuesta, tts_que_anota):
    """El camino del BrainThread sin streaming: split_sentences(reply) + speak."""
    tts, hablado = tts_que_anota
    for frase in split_sentences(respuesta):
        tts.speak(frase)
    _sin_codigo(hablado, respuesta)


@pytest.mark.parametrize("respuesta", _RESPUESTAS_CON_CODIGO.values(),
                         ids=_RESPUESTAS_CON_CODIGO.keys())
@pytest.mark.parametrize("paso", [1, 3, 7])
def test_stream_speaker_no_lee_el_codigo(respuesta, paso, tts_que_anota):
    """StreamSpeaker/SentenceFeeder con tokens de `paso` caracteres."""
    tts, hablado = tts_que_anota
    speaker = StreamSpeaker(tts)
    for i in range(0, len(respuesta), paso):
        speaker.feed(respuesta[i:i + paso])
    assert speaker.finish(timeout_s=10.0)
    _sin_codigo(hablado, respuesta)
