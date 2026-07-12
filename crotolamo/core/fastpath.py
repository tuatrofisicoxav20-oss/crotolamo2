"""Atajos de voz SIN LLM para comandos inequívocos (misión velocidad).

PROBLEMA: en CPU, un turno con tool paga ~17-22s de LLM aunque la orden sea
obvia ("pausa la música" solo puede ser music_control). Para un puñado de
comandos frecuentes e inequívocos, este módulo los resuelve con regex y la
tool se ejecuta DE INMEDIATO (~0.1s): el patrón termina de hablar y pasa.

Deliberadamente CONSERVADOR: la regla debe cubrir el comando COMPLETO
(anclada ^...$ sobre el texto normalizado, sin puntuación). Cualquier matiz
("pausa la música y dime la hora") NO matchea y cae al agente normal con
LLM, que sí razona. Un fast-path equivocado sería peor que uno lento.

Solo tools seguras/presentacionales (o open_app validada contra las apps
conocidas). Nada destructivo pasa por aquí.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _norm(text: str) -> str:
    """minúsculas, sin acentos y sin puntuación (Whisper mete '.' final)."""
    nfd = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    plain = re.sub(r"[^\w\s+]", " ", plain)
    return " ".join(plain.split())


# (regex sobre texto normalizado, tool, args). Orden = prioridad.
_RULES: list[tuple[str, str, dict[str, Any]]] = [
    # --- música (transporte MPRIS, inofensivo) ---
    (r"^pausa(la)?( la (musica|cancion|rola))?$", "music_control", {"action": "pause"}),
    (r"^(dale play|play|reanuda( la musica)?|sigue la musica)$",
     "music_control", {"action": "play"}),
    (r"^(siguiente|saltala|salta)( (cancion|rola|tema))?$|^siguiente$",
     "music_control", {"action": "next"}),
    (r"^(anterior|regresa la (cancion|rola))$", "music_control", {"action": "previous"}),
    (r"^(para|quita|deten)( la)? (musica|cancion|rola)$",
     "music_control", {"action": "stop"}),
    (r"^que ((cancion|rola|tema) )?(esta sonando|suena)( ahorita)?$|^que suena$",
     "music_now", {}),
    # --- ventanas / sistema (read-only) ---
    (r"^que (apps|ventanas|aplicaciones) tengo abiert(as|os)$|^que tengo abierto$",
     "list_windows", {}),
    (r"^(cuanta ram( tengo| queda| hay)?|como anda la ram)$", "ram_usage", {}),
    (r"^cuanto (espacio|disco)( libre)?( me)?( queda| hay| tengo)?$", "disk_usage", {}),
    (r"^(como esta|como anda|estado de(l)?) (el |la |mi )?(sistema|compu|lap(top)?)$",
     "system_status", {}),
]

_COMPILED = [(re.compile(pattern), name, args) for pattern, name, args in _RULES]

# "abre <app>": solo si <app> es una app CONOCIDA (config [apps] o defaults);
# si no, cae al LLM, que sabe distinguir apps de carpetas/URLs.
_OPEN_APP = re.compile(r"^(abre(me)?|abrime|lanza) (la |el )?(?P<app>[\w+ .]{2,30})$")


def _known_apps() -> set[str]:
    apps: set[str] = set()
    try:
        from crotolamo.settings import get_settings

        apps.update(_norm(k) for k in get_settings().raw.get("apps", {}))
    except Exception:  # noqa: BLE001 - sin config igual sirven los defaults
        pass
    try:
        from crotolamo.tools.desktop import APP_COMMANDS

        apps.update(_norm(k) for k in APP_COMMANDS)
    except Exception:  # noqa: BLE001
        pass
    # "terminal" no vive en APP_COMMANDS (open_app la resuelve vía
    # _detect_terminal), pero sigue siendo un destino válido del atajo.
    apps.add("terminal")
    return apps


def match(text: str) -> tuple[str, dict[str, Any]] | None:
    """(tool, args) si el comando COMPLETO es un atajo inequívoco; si no, None."""
    norm = _norm(text)
    if not norm:
        return None
    for pattern, name, args in _COMPILED:
        if pattern.match(norm):
            return name, dict(args)
    m = _OPEN_APP.match(norm)
    if m:
        app = m.group("app").strip()
        if app in _known_apps():
            return "open_app", {"name": app}
    return None
