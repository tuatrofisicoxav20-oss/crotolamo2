"""Tests del publicador de estado del HUD (Misión 2).

Prueba sin micrófono, sin LLM ni audio real:
- El publicador escribe el JSON con el esquema correcto al cambiar de modo.
- La escritura es atómica (via make_file_publisher + os.replace).
- Es no-op cuando no se configura (publisher=None).
- set_text actualiza el campo text en la publicación.
- new_turn dispara una publicación.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from crotolamo.voice.state import Mode, SharedState, make_file_publisher


# ---------------------------------------------------------------------------
# 1. No-op cuando publisher=None (no rompe nada, no crea archivos)
# ---------------------------------------------------------------------------

def test_noop_when_no_publisher():
    """Sin publisher, SharedState funciona igual que antes; no escribe ningún archivo."""
    state = SharedState()  # sin publisher -> no-op
    state.set_mode(Mode.LISTENING)
    state.new_turn()
    state.set_text("hola")
    # Simplemente no debe lanzar ni crear archivos.
    assert state.get_mode() is Mode.LISTENING
    assert state.turn_id == 1


# ---------------------------------------------------------------------------
# 2. Esquema correcto al cambiar modo
# ---------------------------------------------------------------------------

def test_publisher_receives_correct_schema_on_set_mode(tmp_path):
    """Cada llamada a set_mode dispara el publicador con el esquema del HUD."""
    received: list[dict] = []

    def capture(d: dict) -> None:
        received.append(dict(d))

    state = SharedState(publisher=capture)
    state.set_mode(Mode.LISTENING)
    state.set_mode(Mode.THINKING)
    state.set_mode(Mode.SPEAKING)
    state.set_mode(Mode.IDLE)

    assert len(received) == 4

    expected_modes = ["listening", "thinking", "speaking", "idle"]
    for i, snap in enumerate(received):
        assert snap["mode"] == expected_modes[i], f"snap[{i}] mode erróneo: {snap}"
        assert isinstance(snap["turn_id"], int)
        assert isinstance(snap["text"], str)
        assert isinstance(snap["ts"], float)
        assert isinstance(snap["pid"], int)
        assert snap["pid"] == os.getpid()
        # Verificar que ts es reciente (menos de 5 s)
        assert abs(snap["ts"] - time.time()) < 5.0


# ---------------------------------------------------------------------------
# 3. Esquema correcto en new_turn
# ---------------------------------------------------------------------------

def test_publisher_fires_on_new_turn():
    """new_turn debe publicar con el nuevo turn_id."""
    received: list[dict] = []
    state = SharedState(publisher=received.append)

    t1 = state.new_turn()
    t2 = state.new_turn()

    assert len(received) == 2
    assert received[0]["turn_id"] == t1 == 1
    assert received[1]["turn_id"] == t2 == 2


# ---------------------------------------------------------------------------
# 4. set_text publica con el texto correcto
# ---------------------------------------------------------------------------

def test_publisher_fires_on_set_text():
    received: list[dict] = []
    state = SharedState(publisher=received.append)

    state.set_text("abre youtube")
    assert len(received) == 1
    assert received[0]["text"] == "abre youtube"


# ---------------------------------------------------------------------------
# 5. make_file_publisher escribe JSON atómico
# ---------------------------------------------------------------------------

def test_make_file_publisher_writes_atomic_json(tmp_path):
    """make_file_publisher escribe JSON correcto al archivo dado."""
    hud_file = tmp_path / "hud_state.json"
    pub = make_file_publisher(hud_file)
    state = SharedState(publisher=pub)

    state.set_mode(Mode.LISTENING)

    assert hud_file.exists(), "El archivo HUD debe existir tras set_mode"
    data = json.loads(hud_file.read_text(encoding="utf-8"))

    assert data["mode"] == "listening"
    assert isinstance(data["turn_id"], int)
    assert isinstance(data["text"], str)
    assert isinstance(data["ts"], float)
    assert isinstance(data["pid"], int)


# ---------------------------------------------------------------------------
# 6. Atomicidad: no hay archivos temporales sobrantes
# ---------------------------------------------------------------------------

def test_make_file_publisher_no_tmp_leftover(tmp_path):
    """Tras publicar, no deben quedar archivos .tmp en el directorio."""
    hud_file = tmp_path / "hud_state.json"
    pub = make_file_publisher(hud_file)
    state = SharedState(publisher=pub)

    for mode in (Mode.LISTENING, Mode.THINKING, Mode.SPEAKING, Mode.IDLE):
        state.set_mode(mode)

    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"Archivos temporales sobrantes: {leftover}"


# ---------------------------------------------------------------------------
# 7. El publicador crea el directorio si no existe
# ---------------------------------------------------------------------------

def test_make_file_publisher_creates_parent_dir(tmp_path):
    nested = tmp_path / "sub" / "dir" / "hud_state.json"
    pub = make_file_publisher(nested)
    state = SharedState(publisher=pub)

    state.set_mode(Mode.IDLE)

    assert nested.exists(), "Debe crear el directorio padre si no existe"


# ---------------------------------------------------------------------------
# 8. Errores del publicador no matan el loop (excepción atrapada)
# ---------------------------------------------------------------------------

def test_publisher_error_does_not_crash_state():
    """Si el publisher lanza, SharedState no propaga la excepción."""

    def bad_publisher(d: dict) -> None:
        raise RuntimeError("fallo simulado de E/S")

    state = SharedState(publisher=bad_publisher)
    # Ninguna de estas debe propagar:
    state.set_mode(Mode.LISTENING)
    state.new_turn()
    state.set_text("texto")
    assert state.get_mode() is Mode.LISTENING


# ---------------------------------------------------------------------------
# 9. Thread-safety del publicador (múltiples threads publican sin corrupción)
# ---------------------------------------------------------------------------

def test_publisher_threadsafe(tmp_path):
    """El publicador atómico desde múltiples threads no produce archivos corruptos."""
    hud_file = tmp_path / "hud_state.json"
    pub = make_file_publisher(hud_file)
    state = SharedState(publisher=pub)
    errors: list[Exception] = []

    def worker():
        try:
            for mode in (Mode.LISTENING, Mode.THINKING, Mode.SPEAKING, Mode.IDLE):
                state.set_mode(mode)
                state.new_turn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errores en threads: {errors}"
    # El archivo final debe ser JSON válido.
    data = json.loads(hud_file.read_text(encoding="utf-8"))
    assert "mode" in data


# ---------------------------------------------------------------------------
# 10. current_snapshot devuelve el esquema correcto
# ---------------------------------------------------------------------------

def test_current_snapshot_schema():
    state = SharedState()
    state.set_mode(Mode.THINKING)
    state.new_turn()
    snap = state.current_snapshot()

    assert snap["mode"] == "thinking"
    assert snap["turn_id"] == 1
    assert snap["text"] == ""
    assert isinstance(snap["ts"], float)
    assert snap["pid"] == os.getpid()
