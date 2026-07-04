import subprocess

import pytest

from crotolamo.tools import default_registry, windows


def _fake_clients() -> list[dict]:
    return [
        {
            "class": "Opera-GX",
            "title": "GitHub — Opera GX con un título larguísimo para probar el truncado a 60",
            "address": "0xaaa111",
            "workspace": {"id": 1, "name": "1"},
            "pid": 1001,
        },
        {
            "class": "kitty",
            "title": "~/Documentos/crotolamo2",
            "address": "0xbbb222",
            "workspace": {"id": 2, "name": "2"},
            "pid": 1002,
        },
        {
            "class": "kitty",
            "title": "nvim windows.py",
            "address": "0xccc333",
            "workspace": {"id": 3, "name": "3"},
            "pid": 1003,
        },
    ]


class DispatchRecorder:
    """Fake de windows._dispatch: guarda llamadas y responde returncode 0."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(args)
        return subprocess.CompletedProcess(args=["hyprctl", "dispatch", *args], returncode=0)


@pytest.fixture
def clients(monkeypatch):
    data = _fake_clients()
    monkeypatch.setattr(windows, "_clients", lambda: data)
    return data


@pytest.fixture
def dispatch(monkeypatch):
    recorder = DispatchRecorder()
    monkeypatch.setattr(windows, "_dispatch", recorder)
    return recorder


# --- registro en el registry ---------------------------------------------


def test_windows_tools_registered():
    reg = default_registry()
    for name in ("list_windows", "app_status", "focus_window", "close_window"):
        assert name in reg.names()
    assert reg.get("close_window").safe is False
    assert reg.get("focus_window").safe is True
    assert reg.get("list_windows").direct is True
    assert reg.get("app_status").direct is True
    schema = reg.get("app_status").schema()
    assert "name" in schema["function"]["parameters"]["required"]


# --- list_windows ----------------------------------------------------------


def test_list_windows_shows_workspaces_and_truncates(clients):
    out = windows.list_windows()
    assert "workspace 1: Opera-GX" in out
    assert "workspace 2: kitty" in out
    assert "3 ventanas" in out
    # el título largo se trunca (~60 chars + elipsis)
    assert "…" in out
    assert "para probar el truncado" not in out


def test_list_windows_empty(monkeypatch):
    monkeypatch.setattr(windows, "_clients", lambda: [])
    out = windows.list_windows()
    assert "ventana" in out.lower()
    assert "workspace" not in out


def test_list_windows_no_hyprctl(monkeypatch):
    monkeypatch.setattr(windows, "_clients", lambda: None)
    assert "hyprctl" in windows.list_windows()


# --- app_status -------------------------------------------------------------


def test_app_status_open_single(clients):
    out = windows.app_status("opera")
    assert "Sí" in out
    assert "workspace 1" in out


def test_app_status_open_multiple_windows(clients):
    out = windows.app_status("kitty")
    assert "2 ventanas" in out
    assert "2" in out and "3" in out  # workspaces


def test_app_status_process_without_window(clients, monkeypatch):
    monkeypatch.setattr(windows.shutil, "which", lambda _cmd: "/usr/bin/pgrep")
    monkeypatch.setattr(windows, "_pgrep", lambda name: True)
    out = windows.app_status("spotify")
    assert "proceso" in out
    assert "sin" in out or "no tiene" in out


def test_app_status_not_running(clients, monkeypatch):
    monkeypatch.setattr(windows.shutil, "which", lambda _cmd: "/usr/bin/pgrep")
    monkeypatch.setattr(windows, "_pgrep", lambda name: False)
    out = windows.app_status("gimp")
    assert "no está corriendo" in out


# --- matching ---------------------------------------------------------------


def test_matching_ignores_accents_and_case(clients, dispatch):
    out = windows.focus_window("ópera")
    assert "Opera-GX" in out
    assert dispatch.calls == [["focuswindow", "address:0xaaa111"]]


# --- focus_window -----------------------------------------------------------


def test_focus_single_match(clients, dispatch):
    out = windows.focus_window("opera")
    assert dispatch.calls == [["focuswindow", "address:0xaaa111"]]
    assert "Opera-GX" in out


def test_focus_multiple_matches_uses_first_and_says_it(clients, dispatch):
    out = windows.focus_window("kitty")
    assert dispatch.calls == [["focuswindow", "address:0xbbb222"]]
    assert "2" in out  # avisa que había 2 coincidencias
    assert "primera" in out


def test_focus_no_match(clients, dispatch):
    out = windows.focus_window("inexistente")
    assert dispatch.calls == []
    assert "No encontré" in out


# --- close_window -----------------------------------------------------------


def test_close_uses_correct_address(clients, dispatch):
    out = windows.close_window("opera")
    assert dispatch.calls == [["closewindow", "address:0xaaa111"]]
    assert "Cerrada" in out


def test_close_multiple_matches_closes_only_first(clients, dispatch):
    out = windows.close_window("kitty")
    assert dispatch.calls == [["closewindow", "address:0xbbb222"]]
    assert "SOLO la primera" in out
    assert "siguen abiertas" in out


def test_close_no_match_no_dispatch(clients, dispatch):
    out = windows.close_window("inexistente")
    assert dispatch.calls == []
    assert "Nada que cerrar" in out
