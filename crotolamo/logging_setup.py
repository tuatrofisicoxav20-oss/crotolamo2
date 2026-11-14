"""Logging estructurado por módulo (L1).

Separa la traza de diagnóstico (logger) de la salida conversacional al patrón
(print/voz). Nivel desde la env CROTOLAMO_LOG, o [logging].level en la config, o INFO.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Logger por módulo: get_logger('voice.tts') -> 'crotolamo.voice.tts'."""
    return logging.getLogger(f"crotolamo.{name}")


def _resolve_level() -> str:
    env = os.environ.get("CROTOLAMO_LOG")
    if env:
        return env.upper()
    try:
        from crotolamo.settings import get_settings

        return str(get_settings().raw.get("logging", {}).get("level", "INFO")).upper()
    except Exception:
        return "INFO"


def setup_logging(level: str | None = None) -> None:
    """Configura el logging raíz de crotolamo una sola vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or _resolve_level()).upper()
    # Solo NOMBRES de nivel de verdad. Con getattr(logging, lvl, INFO) un typo
    # (CROTOLAMO_LOG=trace, `level = 10` en el toml) caía a INFO sin decir nada:
    # el patrón pedía DEBUG y no había pista de por qué no salía. Y un nombre que
    # sí es atributo de logging pero no un nivel (BASIC_FORMAT) llegaba tal cual
    # a basicConfig, que reventaba al arrancar.
    numeric = logging.getLevelNamesMapping().get(lvl)
    logging.basicConfig(
        level=logging.INFO if numeric is None else numeric,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _CONFIGURED = True
    if numeric is None:
        get_logger("logging").warning(
            "nivel de log desconocido %r (CROTOLAMO_LOG o [logging].level); uso INFO. "
            "Valen DEBUG, INFO, WARNING y ERROR.", lvl,
        )
