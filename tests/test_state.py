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
