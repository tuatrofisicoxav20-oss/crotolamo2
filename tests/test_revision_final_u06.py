"""Revisión final, unidad 06: crotolamo/core/fastpath.py.

1. "abre <app>" con una clave de [apps] que lleva '-' o '.' ("obs-studio",
   "code.oss"). fastpath._norm cambia toda puntuación por espacios y
   _known_apps guardaba las claves así: "Abre OBS-Studio." casaba con
   "obs studio" y mandaba ESE nombre a open_app, que busca con normalize_key
   (que conserva el guion) y contestaba "No tengo registrada la app 'obs
   studio'... Conozco: ..., obs-studio". Como no es un error duro, el
   fast-path lo daba por respuesta final: no se abría nada y el LLM, que sí
   podía llamar open_app("obs-studio"), ni se enteraba.
"""

from __future__ import annotations

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import fastpath
from crotolamo.tools import desktop


@pytest.fixture
def apps_cfg(monkeypatch) -> dict:
    """[apps] del singleton de settings: vacío y rellenable por cada test."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    apps: dict = {}
    monkeypatch.setitem(real.raw, "apps", apps)
    return apps


@pytest.fixture
def lanzados(monkeypatch) -> list[list[str]]:
    """Captura lo que open_app lanzaría, sin lanzar nada."""
    args: list[list[str]] = []
    monkeypatch.setattr(desktop, "run_detached", args.append)
    monkeypatch.setattr(desktop, "_flatpak_app_missing", lambda _args: False)
    return args


@pytest.mark.parametrize("clave, dicho", [
    ("obs-studio", "Abre OBS-Studio."),
    ("obs-studio", "abre obs studio"),
    ("code.oss", "Ábreme code.oss"),
], ids=["guion-tal-cual", "guion-hablado-con-espacio", "punto"])
def test_open_app_manda_la_clave_de_la_config(apps_cfg, clave, dicho):
    """El atajo casa por el nombre normalizado, pero open_app recibe la clave
    tal como la conoce (la que guarda _config_apps), no la versión sin puntuación."""
    apps_cfg[clave] = ["lo-que-sea"]
    assert fastpath.match(dicho) == ("open_app", {"name": clave})


def test_abre_una_app_de_la_config_con_guion_de_punta_a_punta(apps_cfg, lanzados):
    """El agente real con el registro de fábrica: la orden se resuelve por el
    fast-path, sin LLM, y la app SE LANZA (antes: "No tengo registrada")."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.core.turn import VIA_FASTPATH
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    class _BoomLLM:  # si el agente lo llama, el test truena
        model = "x"

        def chat(self, *a, **k):
            raise AssertionError("el fast-path no debía llegar al LLM")

        chat_stream = chat

    apps_cfg["obs-studio"] = ["obs"]
    agent = ToolAgent(
        _BoomLLM(), Conversation("sys"), registry=default_registry(),
        guard=Guard.from_settings(settings_mod.get_settings()),
        pre_hooks=[], post_hooks=[], fastpath=True,
    )
    reply = agent.handle_turn("Abre OBS-Studio.")
    assert agent.last_turn is not None and agent.last_turn.via == VIA_FASTPATH
    assert lanzados == [["obs"]]
    assert "No tengo registrada" not in reply
    assert "Abrí obs-studio" in reply


def test_clave_exacta_gana_a_la_que_solo_coincide_sin_puntuacion(apps_cfg):
    """Si el patrón cableó "obs studio" Y "obs-studio", decir "abre obs studio"
    abre la que nombró tal cual; "Abre OBS-Studio." no se distingue al oído."""
    apps_cfg["obs-studio"] = ["obs-del-guion"]
    apps_cfg["obs studio"] = ["obs-del-espacio"]
    assert fastpath.match("abre obs studio") == ("open_app", {"name": "obs studio"})


def test_claves_sin_puntuacion_no_cambian(apps_cfg):
    """Lo que ya funcionaba sigue igual: defaults, terminal y claves de la
    config con mayúsculas o acentos (open_app las normaliza igual)."""
    apps_cfg["Telegram Desktop"] = ["telegram-desktop"]
    apps_cfg["editor de vídeo"] = ["kdenlive"]
    assert fastpath.match("abre blender") == ("open_app", {"name": "blender"})
    assert fastpath.match("abre la terminal") == ("open_app", {"name": "terminal"})
    assert fastpath.match("Abre Telegram Desktop.") == (
        "open_app", {"name": "telegram desktop"})
    assert fastpath.match("abre el editor de video") == (
        "open_app", {"name": "editor de video"})
    # lo que no es una app conocida sigue cayendo al LLM
    assert fastpath.match("abre la carpeta de descargas") is None
    assert fastpath.match("abre obs studio") is None


def test_sin_config_legible_valen_los_defaults(monkeypatch):
    """Sin config, el atajo sigue conociendo APP_COMMANDS y el terminal."""
    def boom():
        raise FileNotFoundError("No encuentro la config")

    monkeypatch.setattr(settings_mod, "get_settings", boom)
    assert fastpath.match("abre blender") == ("open_app", {"name": "blender"})
    assert fastpath.match("abre la terminal") == ("open_app", {"name": "terminal"})
