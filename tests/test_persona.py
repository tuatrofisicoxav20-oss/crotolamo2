"""Tests de la personalidad configurable ([persona] en la config)."""

import logging

import pytest

from crotolamo import settings as settings_mod
from crotolamo.core import persona


@pytest.fixture
def persona_cfg(monkeypatch):
    """Fija la sección [persona] del singleton de settings para el test."""
    real = settings_mod.get_settings()

    def set_cfg(**kwargs):
        raw = {k: v for k, v in real.raw.items() if k != "persona"}
        if kwargs:
            raw["persona"] = kwargs
        monkeypatch.setattr(real, "raw", raw)
        monkeypatch.setattr(settings_mod, "_SETTINGS", real)

    return set_cfg


def test_default_style_is_desmadrosa(persona_cfg):
    # Sin sección [persona] en el toml, el default es "desmadrosa".
    persona_cfg()
    assert persona.system_prompt() == persona.STYLES["desmadrosa"]


def test_style_read_from_settings(persona_cfg):
    persona_cfg(style="clasica")
    assert persona.system_prompt() == persona.STYLES["clasica"]


def test_explicit_style_wins_over_settings(persona_cfg):
    persona_cfg(style="clasica")
    assert persona.system_prompt(style="desmadrosa") == persona.STYLES["desmadrosa"]


def test_unknown_style_falls_back_to_clasica_with_warning(persona_cfg, caplog):
    persona_cfg(style="gandalla")
    with caplog.at_level(logging.WARNING, logger="crotolamo.persona"):
        prompt = persona.system_prompt()
    assert prompt == persona.STYLES["clasica"]
    assert "gandalla" in caplog.text


def test_extra_is_appended(persona_cfg):
    persona_cfg(style="clasica", extra="Al patrón le gustan los tacos de canasta.")
    prompt = persona.system_prompt()
    assert prompt.startswith(persona.STYLES["clasica"])
    assert "tacos de canasta" in prompt


def test_extra_context_still_appended(persona_cfg):
    persona_cfg(style="clasica")
    prompt = persona.system_prompt(extra_context="usa Fedora")
    assert "Contexto que ya sabes sobre el patrón:" in prompt
    assert "usa Fedora" in prompt


def test_extra_and_extra_context_combine(persona_cfg):
    persona_cfg(style="desmadrosa", extra="Trabaja de noche.")
    prompt = persona.system_prompt(extra_context="usa Hyprland")
    assert "Trabaja de noche." in prompt
    assert "usa Hyprland" in prompt


def test_clasica_is_the_original_prompt():
    # El estilo "clasica" es el SYSTEM_PROMPT histórico, tal cual.
    assert persona.STYLES["clasica"] == persona.SYSTEM_PROMPT
    assert "sarcástico" in persona.STYLES["clasica"]


def test_desmadrosa_keeps_operational_rules():
    prompt = persona.STYLES["desmadrosa"]
    # Español y brevedad (sale por TTS).
    assert "español" in prompt.lower()
    assert "corta" in prompt.lower()
    assert "voz" in prompt.lower()
    # Reglas operativas: tools, seguridad, no inventar.
    assert "LLAMA a la tool" in prompt
    assert "Seguridad:" in prompt
    assert "No inventes" in prompt
    # El tono desmadroso está presente pero sin perder al patrón.
    assert "patrón" in prompt
    assert "órale" in prompt or "no manches" in prompt
