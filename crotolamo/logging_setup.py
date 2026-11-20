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
    nivel = logging.INFO if numeric is None else numeric
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    # faster_whisper escribe en INFO «Processing audio with duration …» por CADA
    # transcripción. Con el wake difuso y gente hablando (o la tele) son unas 12
    # pasadas por minuto: cientos de líneas por hora en el journal, que crecen con el
    # audio ambiente y no con lo que pasa (por lo mismo listener._log_heard bajó su
    # línea por ventana a DEBUG). Sus avisos siguen saliendo, y con DEBUG se conserva
    # todo: ahí sí se está diagnosticando. Solo con el raíz por debajo de WARNING:
    # más arriba ya no salen, y fijarle WARNING al logger dejaría pasar sus avisos
    # con el raíz en ERROR (el nivel propio de un logger manda sobre el heredado).
    if logging.DEBUG < nivel < logging.WARNING:
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    _CONFIGURED = True
    if numeric is None:
        get_logger("logging").warning(
            "nivel de log desconocido %r (CROTOLAMO_LOG o [logging].level); uso INFO. "
            "Valen DEBUG, INFO, WARNING y ERROR.", lvl,
        )
