"""Arreglos de la auditoría en la salida de voz (TTS Piper, segmentación, limpieza).

Sin audio ni Piper reales: el `piper` que ve `TTS._get_voice` es un módulo
falso en `sys.modules` y el motor de síntesis se sustituye donde hace falta.

- El signo de grado '°' tiene categoría Unicode So, la misma que los emojis, y
  el sanitizador lo borraba: "25°C" llegaba a Piper como "25C" ("veinticinco
  ce" en vez de "veinticinco grados ce"). Se conserva; los emojis siguen fuera.
- `_cierra_frase` copiaba y escaneaba con regex TODO el prefijo por cada
  candidato a fin de frase: segmentar era cuadrático en la longitud de la
  respuesta (cúbico por token en streaming con una lista numerada sin punto
  final). Ahora solo mira la cola (la línea actual y la última palabra) con
  resultados idénticos: se compara contra una copia de la versión anterior.
- La carga perezosa de Piper se hace UNA vez aunque dos hilos la pidan a la vez
  (WarmVoice + el primer say()): antes cada hilo cargaba su copia del modelo.
- V4: un stop() del barge-in que caía justo antes de _speak_streaming se perdía
  (la bandera se limpia al entrar) y la frase del turno abortado sonaba entera.
  speak() acepta `should_abort` (atado al turno por quien llama) y el corte real
  por chunks -> OutputStream, que no tenía pruebas, queda cubierto.
- C6: StreamSpeaker.finish() vencía a los 120 s TOTALES y dejaba el hilo vivo
  hablando la cola encima de la escucha siguiente. Ahora el plazo es por frase y,
  si vence, el hilo se cancela (corta la frase que suena y no habla más).
"""

from __future__ import annotations

import random
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from crotolamo.voice import tts as tts_mod
from crotolamo.voice.clean import limpiar_para_voz
from crotolamo.voice.tts import TTS, StreamSpeaker, split_sentences

# --- el signo de grado se pronuncia ("grados"): no es un emoji ----------------


def test_grados_se_conservan():
    assert limpiar_para_voz("Hay 25°C en Monterrey.") == "Hay 25°C en Monterrey."
    assert limpiar_para_voz("Giro de 90°.") == "Giro de 90°."
    assert limpiar_para_voz("Mínima de 18 °C y máxima de 31 °C.") == (
        "Mínima de 18 °C y máxima de 31 °C.")


def test_grados_junto_a_emojis_solo_se_van_los_emojis():
    assert limpiar_para_voz("☀️ Hoy 25°C 🎉👍🏽") == "Hoy 25°C"


def test_grado_suelto_sigue_siendo_solo_puntuacion():
    # Un '°' sin nada pronunciable alrededor no merece síntesis.
    assert limpiar_para_voz("°") == ""


def test_speak_le_pasa_el_grado_a_piper(tmp_path, monkeypatch):
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    hablado: list[str] = []
    monkeypatch.setattr(tts, "_speak_streaming", lambda t, _abort=None: hablado.append(t) or True)
    tts.speak("**Hay 25°C** en Monterrey 🌡️.")
    assert hablado == ["Hay 25°C en Monterrey."]


# --- segmentación: misma salida que antes, sin escanear todo el prefijo --------

# Copia LITERAL de la versión anterior a la auditoría: la referencia de
# "resultados idénticos" para el arreglo de coste.
_REF_MARCADOR_LISTA = re.compile(r"(?:^|\n)[ \t]*\d{1,2}\.$")
_REF_PALABRA_ANTES = re.compile(r"([a-záéíóúüñ]+(?:\.[a-záéíóúüñ]+)*)\.$", re.IGNORECASE)


def _ref_cierra_frase(text: str, start: int, end: int) -> bool:
    if text[start - 1] != ".":
        return True
    despues = text[end] if end < len(text) else ""
    antes = text[:start]
    if len(antes) >= 2 and antes[-2].isdigit() and despues.isdigit():
        return False
    if _REF_MARCADOR_LISTA.search(antes):
        return False
    m = _REF_PALABRA_ANTES.search(antes)
    if m:
        palabra = m.group(1).lower()
        if palabra in tts_mod._ABREVIATURAS:
            return False
        if palabra == "no" and despues.isdigit():
            return False
    return True


_TROZOS = [
    "a", "b", "z", "Z", "á", "Ñ", "ü", "É", "ſ", "K", "ı", "é", "1", "2", "9", "²", "٣",
    ".", ".", ".", " ", " ", "\n", "\t", "*", "_", "~", "!", "?", "¿", "¡", ",", "-",
    "Dr", "dr", "Sra", "etc", "p.ej", "P.EJ", "vs", "Av", "no", "No", "NO", "10", "99",
    "123", "x.y", "é",
]


def test_cierra_frase_identica_a_la_version_anterior():
    rng = random.Random(20261113)
    for _ in range(4000):
        texto = "".join(rng.choice(_TROZOS) for _ in range(rng.randint(1, 40)))
        for m in tts_mod._CANDIDATO.finditer(texto):
            esperado = _ref_cierra_frase(texto, m.start(), m.end())
            assert tts_mod._cierra_frase(texto, m.start(), m.end()) is esperado, (
                repr(texto), m.start())


def test_split_identico_en_casos_reales():
    casos = [
        "Tienes 3.5 GB, patrón. Y 40. 5 más.",
        "Llama al Dr. López. Ya le avisé.",
        "Pasos:\n1. Abre la app. Luego cierra.\n  12. Otro paso. Fin.",
        "Es el No. 5 de la lista. Claro que no. Pero lo intento.",
        "Hay manzanas, peras, etc. en el frutero. Vale p.ej. esto. Y P.EJ. aquello.",
        "**Listo.** ¿Qué tal? ¡Órale! Sigue.\n\nOtra línea.",
        "".join(f"{i}. Abre la aplicación número {i} y revisa el estado\n" for i in range(1, 40)),
    ]
    for texto in casos:
        for final in (True, False):
            nuevo = tts_mod._split(texto, final=final)
            # La referencia: el mismo _split con la _cierra_frase anterior.
            piezas: list[str] = []
            inicio = 0
            for m in tts_mod._CANDIDATO.finditer(texto):
                if m.end() >= len(texto) and not final and texto[m.start() - 1] == ".":
                    break
                if not _ref_cierra_frase(texto, m.start(), m.end()):
                    continue
                piezas.append(texto[inicio:m.start()])
                inicio = m.end()
            esperado = ([p.strip() for p in piezas if p.strip()], texto[inicio:])
            assert nuevo == esperado, repr(texto)


def _mejor_de(n: int, fn) -> float:
    mejor = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def test_split_sentences_de_respuesta_larga_es_lineal():
    """~49 KB y 800 frases. Antes: ~4 s (cuadrático: cada candidato copiaba y
    escaneaba todo el prefijo). Ahora: decenas de ms. El tope deja 1 s de
    holgura para una máquina cargada sin dejar pasar la versión cuadrática."""
    texto = "Órale patrón, ya quedó la música pausada y el volumen abajo. " * 800
    piezas: list[list[str]] = []
    dt = _mejor_de(2, lambda: piezas.append(split_sentences(texto)))
    assert len(piezas[-1]) == 800
    assert dt < 1.0, f"split_sentences tardó {dt:.2f} s"


class _Mudo:
    def speak(self, text: str) -> None:
        pass


def test_streaming_de_lista_numerada_sin_punto_final_no_es_cubico():
    """Una lista '1. ... \\n2. ...' sin punto final nunca cierra frase: el buffer
    crece y cada token re-segmenta todo. Antes: ~2 s de CPU en el hilo que
    recibe los tokens del LLM (cada marcador '1.' re-escaneaba el prefijo)."""
    texto = "".join(f"{i}. Abre la aplicación número {i} y revisa el estado\n"
                    for i in range(1, 100))

    def _stream() -> None:
        sp = StreamSpeaker(_Mudo())  # type: ignore[arg-type]
        for i in range(0, len(texto), 4):
            sp.feed(texto[i:i + 4])
        assert sp.finish(timeout_s=5) is True

    dt = _mejor_de(2, _stream)
    assert dt < 1.0, f"el streaming de la lista tardó {dt:.2f} s"


# --- carga perezosa de Piper: una sola vez aunque la pidan dos hilos -----------


class _SlowPiperVoice:
    """PiperVoice falso: cuenta cargas y se queda dentro de load() hasta que el
    test abre la compuerta (la ventana de la carrera dura lo que el test quiera)."""

    loads: list[str] = []
    gate: threading.Event | None = None
    fail_next: bool = False
    _guard = threading.Lock()

    @classmethod
    def load(cls, path):
        with cls._guard:
            cls.loads.append(path)
            fail = cls.fail_next
            cls.fail_next = False
        if fail:
            raise RuntimeError("onnx corrupto")
        if cls.gate is not None:
            cls.gate.wait(5)
        return object()


@pytest.fixture
def slow_piper(monkeypatch):
    fake = types.ModuleType("piper")
    fake.PiperVoice = _SlowPiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake)
    _SlowPiperVoice.loads = []
    _SlowPiperVoice.gate = None
    _SlowPiperVoice.fail_next = False
    yield _SlowPiperVoice
    if _SlowPiperVoice.gate is not None:
        _SlowPiperVoice.gate.set()  # nunca dejar un hilo colgado si el test falló


def _wait_until(cond, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_dos_hilos_cargan_piper_una_sola_vez(slow_piper):
    """WarmVoice (listener) y el primer say() piden la voz a la vez."""
    slow_piper.gate = threading.Event()
    tts = TTS(Path("voz.onnx"))
    start = threading.Barrier(2)
    got: list[object] = []

    def _load():
        start.wait(5)
        got.append(tts._get_voice())

    hilos = [threading.Thread(target=_load, daemon=True) for _ in range(2)]
    for h in hilos:
        h.start()
    # Con la carrera, el segundo hilo entra a load() mientras el primero sigue
    # dentro; con el lock, espera fuera. (Si la máquina va tan cargada que el
    # segundo hilo ni arranca en este plazo, encuentra la voz ya cacheada: el
    # test nunca falla en falso.)
    _wait_until(lambda: len(slow_piper.loads) >= 2, 0.4)
    slow_piper.gate.set()
    for h in hilos:
        h.join(5)
        assert not h.is_alive()

    assert slow_piper.loads == ["voz.onnx"]
    assert len(got) == 2 and got[0] is got[1]
    assert tts._get_voice() is got[0]


def test_carga_fallida_de_piper_no_envenena_la_siguiente(slow_piper):
    """Si la carga revienta, no queda nada cacheado ni el lock tomado."""
    slow_piper.fail_next = True
    tts = TTS(Path("voz.onnx"))
    with pytest.raises(RuntimeError):
        tts._get_voice()

    done: list[object] = []
    h = threading.Thread(target=lambda: done.append(tts._get_voice()), daemon=True)
    h.start()
    h.join(5)
    assert not h.is_alive()
    assert done and tts._get_voice() is done[0]
    assert slow_piper.loads == ["voz.onnx", "voz.onnx"]


# --- el corte REAL (_speak_streaming): stop() y should_abort --------------------
# Hasta ahora la suite solo probaba _play_interruptible (que speak() ya no usaba,
# y que la revisión final quitó):
# el camino que de verdad suena (Piper por chunks -> sd.OutputStream) no tenía
# ninguna prueba de corte. Aquí la voz Piper y el OutputStream son dobles.


class _Chunk:
    sample_rate = 22050

    def __init__(self, n: int) -> None:
        import numpy as np

        self.audio_int16_array = np.zeros(n, dtype=np.int16)


class _FakeVoice:
    """Voz Piper falsa: 3 chunks de 4096 muestras (2 escrituras de 2048 cada uno)."""

    def __init__(self) -> None:
        self.synth_calls = 0

    def synthesize(self, text):
        self.synth_calls += 1
        for _ in range(3):
            yield _Chunk(4096)


class _FakeOutputStream:
    instances: list["_FakeOutputStream"] = []
    on_write = None  # gancho: lo que "pasa" en otro hilo mientras suena

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.writes: list[int] = []
        self.calls: list[str] = []
        _FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self.calls.append("start")

    def write(self, audio) -> None:
        self.writes.append(len(audio))
        if _FakeOutputStream.on_write is not None:
            _FakeOutputStream.on_write(len(self.writes))

    def abort(self) -> None:
        self.calls.append("abort")

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        self.calls.append("close")


@pytest.fixture
def audio_falso(tmp_path, monkeypatch):
    """TTS real con voz y salida de audio falsas (sin PortAudio ni Piper)."""
    import importlib.machinery

    pytest.importorskip("numpy")
    sd = types.ModuleType("sounddevice")
    sd.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
    sd.OutputStream = _FakeOutputStream  # type: ignore[attr-defined]
    sd.stop = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    _FakeOutputStream.instances = []
    _FakeOutputStream.on_write = None
    modelo = tmp_path / "voz.onnx"
    modelo.write_bytes(b"onnx de mentira")
    tts = TTS(modelo)
    voz = _FakeVoice()
    tts._voice = voz  # type: ignore[assignment]
    yield tts, voz
    _FakeOutputStream.on_write = None


def _muestras_escritas() -> int:
    return sum(sum(s.writes) for s in _FakeOutputStream.instances)


def test_frase_completa_suena_entera_y_cierra_sin_abort(audio_falso):
    tts, _ = audio_falso
    assert tts._speak_streaming("Hola, patrón.") is True
    [stream] = _FakeOutputStream.instances
    assert stream.writes == [2048] * 6
    assert stream.calls == ["start", "stop", "close"]  # sin abort: suena hasta el final


def test_stop_a_media_frase_corta_en_una_rebanada(audio_falso):
    """Barge-in mientras suena: tras la escritura en curso no se escribe nada más
    y el stream se ABORTA (se tira lo que quedaba en el búfer). El stop cae a
    mitad de un chunk: no se espera al siguiente chunk para cortar."""
    tts, _ = audio_falso
    _FakeOutputStream.on_write = lambda n: tts.stop() if n == 1 else None
    assert tts._speak_streaming("Una frase larga que el patrón interrumpe.") is False
    [stream] = _FakeOutputStream.instances
    assert stream.writes == [2048]
    assert stream.calls == ["start", "abort", "stop", "close"]


def test_barge_in_justo_antes_de_hablar_no_suena_la_frase_vieja(audio_falso):
    """V4: el oído hace stop() + new_turn() mientras la Boca ya sacó de la cola una
    frase que ERA vigente y va camino de speak(). _speak_streaming limpia la
    bandera al entrar (si no, un corte mataría todas las frases siguientes), así
    que ese stop() se perdía y la frase del turno abortado sonaba ENTERA encima
    del patrón. Con should_abort (el turno de la frase ya no es el vigente) no se
    sintetiza ni se abre el audio."""
    tts, voz = audio_falso
    turno = {"vigente": 1}
    mi_turno = 1
    tts.stop()              # el oído corta el audio...
    turno["vigente"] = 2    # ...e invalida el turno, ANTES de que la Boca entre
    tts.speak("Segunda frase del turno viejo.",
              should_abort=lambda: turno["vigente"] != mi_turno)
    assert _muestras_escritas() == 0
    assert _FakeOutputStream.instances == []
    assert voz.synth_calls == 0  # ni siquiera se gastó CPU sintetizando


def test_should_abort_a_media_frase_corta_igual_que_stop(audio_falso):
    """El turno se invalida mientras la frase suena (aunque el stop() se haya
    perdido): se corta en la siguiente rebanada y se aborta el stream."""
    tts, _ = audio_falso
    vigente = {"ok": True}

    def _barge_in(n: int) -> None:
        if n == 3:
            vigente["ok"] = False

    _FakeOutputStream.on_write = _barge_in
    assert tts._speak_streaming("Frase que se queda vieja.",
                                should_abort=lambda: not vigente["ok"]) is False
    [stream] = _FakeOutputStream.instances
    assert stream.writes == [2048] * 3
    assert stream.calls == ["start", "abort", "stop", "close"]


def test_should_abort_falso_no_cambia_nada(audio_falso):
    tts, _ = audio_falso
    tts.speak("Hola, patrón.", should_abort=lambda: False)
    assert _muestras_escritas() == 3 * 4096
    [stream] = _FakeOutputStream.instances
    assert "abort" not in stream.calls


# --- C6: StreamSpeaker.finish() con un TTS colgado ------------------------------
# finish() esperaba 120 s EN TOTAL y, al vencer, devolvía el control con el hilo
# todavía vivo: el loop simple seguía (HUD idle, escucha del followup de 6 s) y
# el hilo hablaba el resto de la cola ENCIMA de la grabación, que el STT tomaba
# por una orden. Ahora el plazo es por frase (una respuesta larga que avanza se
# espera entera) y, si vence, el hilo se cancela: corta la frase y no habla más.


class _TTSColgado:
    """La PRIMERA frase se queda colgada (sink de audio trabado) hasta que el
    test la suelta. stop() solo se cuenta: como en el TTS real, no desbloquea
    una escritura trabada, solo marca el corte."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stops = 0
        self.suelta = threading.Event()

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        if len(self.spoken) == 1:
            self.suelta.wait(5)

    def stop(self) -> None:
        self.stops += 1


def test_finish_vencido_no_deja_el_hilo_hablando_detras_de_la_escucha():
    tts = _TTSColgado()
    sp = StreamSpeaker(tts)  # type: ignore[arg-type]
    sp.feed("Una. Dos. Tres.")
    t0 = time.monotonic()
    assert sp.finish(timeout_s=0.3) is True
    assert time.monotonic() - t0 < 3
    assert tts.stops == 1  # se cortó la frase que sonaba
    tts.suelta.set()  # el audio se destraba DESPUÉS de que finish() se rindió
    sp._thread.join(5)
    assert not sp._thread.is_alive()
    assert tts.spoken == ["Una."]  # antes: "Dos." y "Tres." encima del followup


class _TTSLento:
    """Cada frase tarda `s` segundos en sonar; nunca se cuelga."""

    def __init__(self, s: float) -> None:
        self.s = s
        self.spoken: list[str] = []
        self.stops = 0

    def speak(self, text: str) -> None:
        time.sleep(self.s)
        self.spoken.append(text)

    def stop(self) -> None:
        self.stops += 1


def test_finish_espera_entera_una_respuesta_larga_que_avanza():
    """El plazo es para un TTS COLGADO, no un tope a lo largo de la respuesta:
    30 frases de 50 ms (1.5 s en total) con un plazo de 0.75 s se hablan todas y
    finish() vuelve cuando terminan (antes volvía a los 0.75 s con la mitad de la
    cola sonando todavía encima de la escucha)."""
    tts = _TTSLento(0.05)
    frases = [f"Frase número {i}." for i in range(30)]
    sp = StreamSpeaker(tts)  # type: ignore[arg-type]
    sp.feed(" ".join(frases))
    assert sp.finish(timeout_s=0.75) is True
    assert not sp._thread.is_alive()
    assert tts.spoken == frases
    assert tts.stops == 0


def test_finish_vencido_corta_la_frase_trabada_del_tts_real(audio_falso):
    """Mismo caso con el camino real (Piper por chunks -> OutputStream): la
    escritura se traba en la primera rebanada; al vencer el plazo se marca el
    corte y, cuando el dispositivo se destraba, el stream se aborta sin escribir
    nada más ni sintetizar las frases siguientes."""
    tts, voz = audio_falso
    suelta = threading.Event()
    _FakeOutputStream.on_write = lambda n: suelta.wait(5) if n == 1 else None
    sp = StreamSpeaker(tts)
    sp.feed("Una frase. Otra frase. Y otra más.")
    assert sp.finish(timeout_s=0.3) is True
    suelta.set()
    sp._thread.join(5)
    assert not sp._thread.is_alive()
    assert voz.synth_calls == 1
    [stream] = _FakeOutputStream.instances
    assert stream.writes == [2048]
    assert stream.calls == ["start", "abort", "stop", "close"]
