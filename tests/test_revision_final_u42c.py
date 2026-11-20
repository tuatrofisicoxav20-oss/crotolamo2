"""Revisión final u42c: la pausa del panel también cierra la ventana de followup.

En el loop simple (--simple, use_oww = false o sin openWakeWord) la pausa de
"Escuchar por voz" se miraba al tope del bucle, tras un wake y dentro del wake
difuso, pero NO en la ventana de followup. Si el patrón pausaba mientras
Crotolamo pensaba, hablaba o esperaba el followup, la ventana se abría igual (con
audio_session leyendo el micrófono abierto), lo que se dijera en el cuarto se
transcribía y se EJECUTABA, y cada respuesta abría otra ventana: la pausa solo
llegaba tras 6 s de silencio, con hud_state.json diciendo ya enabled=false.

La ventana de followup es una orden nueva sin wake: se bloquea como el wake (el
turno en curso no se corta, solo no se abre otro). Todo con fakes.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from crotolamo.voice.state import SharedState

from interfaces import listener


class _GuionStt:
    """STT de la orden: frases del guion; on_listen(n) corre al grabar la n-ésima."""

    def __init__(self, texts, on_listen=None):
        self.texts = list(texts)
        self.on_listen = on_listen
        self.calls: list[str] = []

    def _next(self, metodo: str) -> str:
        self.calls.append(metodo)
        if self.on_listen is not None:
            self.on_listen(len(self.calls))
        return self.texts.pop(0) if self.texts else ""

    def listen_once(self, **kwargs):
        return self._next("listen_once")

    def listen_smart(self, **kwargs):
        return self._next("listen_smart")


class _UnWake:
    """wake_stt: un "crotolamo" y, agotado, Ctrl-C (sale del loop)."""

    def __init__(self):
        self.heard = ["crotolamo"]

    def listen_once(self, **kwargs):
        if self.heard:
            return self.heard.pop(0)
        raise KeyboardInterrupt


class _Tts:
    def speak(self, text):
        pass

    def speak_sentences(self, text):
        pass

    def beep(self):
        pass


class _Agente:
    def __init__(self, al_atender=None):
        self.commands: list[str] = []
        self.al_atender = al_atender

    def handle_turn(self, command, on_token=None):
        self.commands.append(command)
        if self.al_atender is not None:
            self.al_atender(len(self.commands))
        return "hecho, patrón."


def _correr(monkeypatch, *, flag, stt, agent, smart=False):
    """El loop simple con followup de fábrica (6 s); sale tras 3 vueltas en pausa."""
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: flag["enabled"])
    en_pausa: list[float] = []

    def sleep(s):
        if s == listener._CONTROL_POLL_S:
            en_pausa.append(s)
            if len(en_pausa) >= 3:
                raise KeyboardInterrupt

    monkeypatch.setattr(listener, "time", SimpleNamespace(sleep=sleep, time=time.time))
    pubs: list[dict] = []
    hud = SharedState(publisher=pubs.append)
    cfg = listener.ListenerConfig(threshold=0.72, variants=["crotolamo"], followup_s=6.0,
                                  ack="off", stream_speak=False, smart_endpoint=smart,
                                  post_reply_guard_s=0.0)
    rc = listener._run_simple_loop(
        agent=agent, stt=stt, wake_stt=_UnWake(), tts=_Tts(), wake_detector=None,
        use_oww=False, cfg=cfg, hud_state=hud, say=lambda text: None,
    )
    return rc, hud, pubs, en_pausa


def _tras_la_pausa(pubs: list[dict]) -> list[dict]:
    primera = next(i for i, p in enumerate(pubs) if p["enabled"] is False)
    return pubs[primera:]


@pytest.mark.parametrize("smart", [False, True])
def test_pausa_durante_el_turno_no_abre_la_ventana_de_followup(monkeypatch, smart):
    """El patrón pausa mientras Crotolamo piensa: la respuesta se da (el turno en
    curso no se corta), pero no se abre ninguna ventana de followup. Antes se
    abría y se ejecutaban "y ahora borra todo" y "manda el correo"."""
    flag = {"enabled": True}

    def pausa_en_el_primer_turno(n):
        if n == 1:
            flag["enabled"] = False  # el panel escribe control.json = false

    stt = _GuionStt(["pon musica", "y ahora borra todo", "manda el correo", ""])
    agent = _Agente(al_atender=pausa_en_el_primer_turno)
    rc, hud, pubs, en_pausa = _correr(monkeypatch, flag=flag, stt=stt, agent=agent,
                                      smart=smart)
    assert rc == 0
    assert agent.commands == ["pon musica"], "ejecutó órdenes con la escucha pausada"
    # Solo se grabó la orden tras el wake: ninguna ventana de followup.
    assert len(stt.calls) == 1
    # La pausa quedó publicada y el HUD no volvió a "ESCUCHANDO" tras ella.
    assert hud.is_enabled() is False
    assert all(p["mode"] != "listening" for p in _tras_la_pausa(pubs))
    assert pubs[-1]["mode"] == "idle" and pubs[-1]["enabled"] is False
    # Y el loop quedó en pausa (duerme al ritmo del sondeo).
    assert en_pausa == [listener._CONTROL_POLL_S] * 3


@pytest.mark.parametrize("smart", [False, True])
def test_pausa_a_media_ventana_de_followup_tira_lo_grabado(monkeypatch, smart):
    """La pausa llega con la ventana de followup ya grabando: lo que se dijo en
    ella no se ejecuta (es una orden nueva sin wake, como un wake en pausa)."""
    flag = {"enabled": True}

    def pausa_en_la_ventana(n):
        if n == 2:
            flag["enabled"] = False

    stt = _GuionStt(["pon musica", "y ahora borra todo", "manda el correo", ""],
                    on_listen=pausa_en_la_ventana)
    agent = _Agente()
    rc, hud, pubs, _ = _correr(monkeypatch, flag=flag, stt=stt, agent=agent, smart=smart)
    assert rc == 0
    assert agent.commands == ["pon musica"], "ejecutó lo oído tras la pausa"
    assert len(stt.calls) == 2  # no se abrió otra ventana detrás
    assert hud.is_enabled() is False
    assert pubs[-1]["mode"] == "idle" and pubs[-1]["enabled"] is False


def test_sin_pausa_el_followup_sigue_atendiendo_como_siempre(monkeypatch):
    """Control: con la escucha activa la ventana de followup funciona igual."""
    flag = {"enabled": True}
    stt = _GuionStt(["pon musica", "y ahora sube el volumen", ""])
    agent = _Agente()
    rc, hud, _, _ = _correr(monkeypatch, flag=flag, stt=stt, agent=agent)
    assert rc == 0
    assert agent.commands == ["pon musica", "y ahora sube el volumen"]
    assert len(stt.calls) == 3  # la orden, la 2ª y la ventana vacía que cierra
    assert hud.is_enabled() is True
