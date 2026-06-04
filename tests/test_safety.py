from pathlib import Path

import pytest

from crotolamo.safety.guard import Guard
from crotolamo.tools.base import Tool


def _tool(name="t", safe=True):
    return Tool(name=name, func=lambda **k: "ok", description="d", parameters={}, safe=safe)


@pytest.fixture
def guard(tmp_path):
    return Guard(allowed_roots=[tmp_path])


def test_path_inside_allowed_is_ok(guard, tmp_path):
    target = tmp_path / "sub" / "nota.md"
    decision = guard.check(_tool(), {"path": str(target)})
    assert decision.allowed and not decision.needs_confirmation


def test_path_outside_allowed_is_blocked(guard):
    decision = guard.check(_tool(), {"path": "/etc/passwd"})
    assert not decision.allowed
    assert "corral" in decision.reason.lower() or "permitidas" in decision.reason.lower()


def test_etc_deletion_style_path_blocked(guard):
    # El equivalente a "borra /etc": el guard rechaza la ruta, sin importar la tool.
    decision = guard.check(_tool(name="delete_file", safe=False), {"path": "/etc"})
    assert not decision.allowed


def test_unsafe_tool_needs_confirmation(guard, tmp_path):
    decision = guard.check(_tool(name="move", safe=False), {"path": str(tmp_path / "x")})
    assert decision.allowed and decision.needs_confirmation


def test_safe_tool_without_paths_runs(guard):
    decision = guard.check(_tool(name="open_url", safe=True), {"url": "https://x.com"})
    assert decision.allowed and not decision.needs_confirmation
