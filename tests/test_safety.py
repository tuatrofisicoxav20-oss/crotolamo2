
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


# --- corral ampliado (M6): tres zonas libre / confirmar / bloqueado ---

@pytest.fixture
def guard3(tmp_path):
    """Guard con zona libre y zona de confirmación separadas bajo tmp_path."""
    return Guard(
        allowed_roots=[tmp_path / "libre"],
        confirm_roots=[tmp_path / "confirmable"],
    )


def test_zone_free_runs_directly(guard3, tmp_path):
    decision = guard3.check(_tool(), {"path": str(tmp_path / "libre" / "nota.md")})
    assert decision.allowed and not decision.needs_confirmation


def test_zone_confirm_asks_first(guard3, tmp_path):
    ruta = tmp_path / "confirmable" / "nota.md"
    decision = guard3.check(_tool(), {"path": str(ruta)})
    assert decision.allowed and decision.needs_confirmation
    assert "zona libre" in decision.reason.lower()
    assert str(ruta) in decision.reason


def test_zone_outside_both_is_blocked(guard3, tmp_path):
    decision = guard3.check(_tool(), {"path": str(tmp_path / "otra" / "x.txt")})
    assert not decision.allowed
    assert "corral" in decision.reason.lower() or "permitidas" in decision.reason.lower()


def test_unsafe_tool_confirms_even_in_free_zone(guard3, tmp_path):
    # safe=False confirma SIEMPRE, aunque la ruta esté en la zona libre.
    decision = guard3.check(
        _tool(name="write_file", safe=False),
        {"path": str(tmp_path / "libre" / "x.txt")},
    )
    assert decision.allowed and decision.needs_confirmation


def test_unsafe_tool_in_confirm_zone_confirms(guard3, tmp_path):
    decision = guard3.check(
        _tool(name="delete_file", safe=False),
        {"path": str(tmp_path / "confirmable" / "x.txt")},
    )
    assert decision.allowed and decision.needs_confirmation


def test_unsafe_tool_outside_both_still_blocked(guard3, tmp_path):
    # Bloqueo gana a confirmación: fuera de ambas zonas ni preguntando.
    decision = guard3.check(
        _tool(name="delete_file", safe=False),
        {"path": str(tmp_path / "otra" / "x.txt")},
    )
    assert not decision.allowed


def test_default_confirm_roots_is_home(guard, tmp_path):
    # Sin confirm_roots explícito, el default es el home del patrón: una ruta
    # bajo ~ (fuera de la zona libre) pide confirmación en vez de bloquearse.
    from pathlib import Path

    ruta = Path.home() / "crotolamo_prueba_zona.txt"
    decision = guard.check(_tool(), {"path": str(ruta)})
    assert decision.allowed and decision.needs_confirmation


def test_etc_blocked_with_default_confirm_roots(guard):
    # /etc no está bajo ~: sigue bloqueado con el default de confirm_roots.
    decision = guard.check(_tool(), {"path": "/etc/fstab"})
    assert not decision.allowed


# --- path_inside_roots: lo que no se puede resolver se NIEGA, sin excepción ---

def test_ruta_con_byte_nulo_queda_fuera_del_corral(tmp_path):
    """Path.resolve() lanza ValueError("embedded null byte") con un '\\0' en la
    ruta; solo se capturaban OSError/RuntimeError y la excepción subía hasta el
    agente. La ruta la elige el LLM: debe negar (False), no reventar."""
    from pathlib import Path

    from crotolamo.safety.paths import path_inside_roots

    assert path_inside_roots(Path("a\0b"), [tmp_path]) is False
    assert path_inside_roots(str(tmp_path / "ok\0malo.txt"), [tmp_path]) is False
