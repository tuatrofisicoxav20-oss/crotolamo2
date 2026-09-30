"""SharedState: modos, turn_id monótono y seguridad ante concurrencia (M3.2)."""

import threading

from crotolamo.voice.state import Mode, SharedState


def test_mode_transitions():
    s = SharedState()
    assert s.get_mode() is Mode.IDLE
    s.set_mode(Mode.LISTENING)
    assert s.get_mode() is Mode.LISTENING
    s.set_mode(Mode.SPEAKING)
    assert s.get_mode() is Mode.SPEAKING


def test_new_turn_increments_and_is_current():
    s = SharedState()
    assert s.turn_id == 0
    t1 = s.new_turn()
    assert t1 == 1 and s.is_current(1)
    t2 = s.new_turn()
    assert t2 == 2 and s.is_current(2)
    # El turno viejo ya no es el actual (así se descartan frases abortadas).
    assert not s.is_current(1)


def test_new_turn_limpia_el_texto_visible():
    """Al abrir un turno el HUD no debe seguir mostrando la respuesta anterior."""
    publicado: list[dict] = []
    s = SharedState(publisher=publicado.append)
    s.set_text("Listo, patrón: pausé la música.")
    s.new_turn()
    assert s.current_snapshot()["text"] == ""
    assert publicado[-1]["text"] == "" and publicado[-1]["turn_id"] == 1


def test_publish_now_publica_aunque_nada_cambie():
    """Al arrancar hay que pisar un hud_state.json rancio: publicar sin cambio."""
    publicado: list[dict] = []
    s = SharedState(publisher=publicado.append)
    s.set_enabled(True)  # mismo valor que el inicial: NO publica
    assert publicado == []
    s.publish_now()
    assert len(publicado) == 1
    assert publicado[0]["mode"] == "idle" and publicado[0]["enabled"] is True
    assert {"mode", "turn_id", "text", "enabled", "ts", "pid"} <= set(publicado[0])


def test_new_turn_is_threadsafe():
    s = SharedState()

    def hammer():
        for _ in range(100):
            s.new_turn()

    threads = [threading.Thread(target=hammer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 10 threads * 100 incrementos = 1000 exactos, sin pérdidas por race.
    assert s.turn_id == 1000
