"""Revisión final, unidad 39v: crotolamo/voice/wake.py (wake difuso del loop simple).

1. Un saludo antes del wake ("buenos días, crotolamo, pon las noticias") activaba
   por la cabeza de 3 palabras, pero strip_wake_word puntúa el prefijo ENTERO de
   1-3 palabras ("buenosdiascrotolamo" no llega a 0.67) y no cortaba nada:
   split_wake_command devolvía (True, "") y la orden ya dicha se perdía (el
   listener pedía otra con el acuse). Ahora, si strip no corta, se corta justo
   tras el wake DENTRO de la cabeza.

2. En la ventana de followup (orden SIN wake) strip_wake_word consideraba la
   frase ENTERA como posible wake: "¿cómo me llamo?" se parece a "croto el amo"
   y se borraba entera; el loop terminaba en silencio sin preguntar nada al
   agente. Borrar la frase entera solo vale cuando quien llama ya sabe que es un
   wake (split_wake_command, "caroto llamo" -> (True, "")).

Puro: variantes y umbral como los de config/crotolamo.toml, fijados aquí.
"""

from __future__ import annotations

import pytest

from crotolamo.voice import wake
from crotolamo.voice.normalize import normalize_text

_VARIANTES = [
    "crotolamo", "croto lamo", "gotolamo", "grotolamo", "crotoamo",
    "coto lamo", "coto y amo", "cottoy amo", "croto el amo", "crotolo amo",
]
_UMBRAL = 0.72


# --- 1: el wake en 3ª posición (tras un saludo) deja la orden pegada ---------

@pytest.mark.parametrize("heard, command", [
    ("buenos días crotolamo pon las noticias", "pon las noticias"),
    ("Buenos días, Crotolamo, pon las noticias", "pon las noticias"),
    ("buenas noches crotolamo apaga la luz", "apaga la luz"),
    ("oye entonces crotolamo pon música", "pon música"),
    ("bueno pues crotolamo pon música", "pon música"),
    ("buenos días protolamo pon música", "pon música"),
    ("buenos días, crotolamo, ¿qué hora es?", "¿qué hora es?"),
])
def test_wake_tras_un_saludo_deja_la_orden_pegada(heard, command):
    heard = normalize_text(heard)  # lo que entrega listen_once al listener
    assert wake.split_wake_command(heard, _UMBRAL, _VARIANTES) == (True, command)


@pytest.mark.parametrize("heard", [
    "buenos días crotolamo",
    "oye entonces crotolamo",
    "caroto llamo",
    "crotolamo",
])
def test_wake_sin_orden_detras_sigue_pidiendo_la_orden(heard):
    """Si tras el wake no queda nada, el listener tiene que pedir la orden."""
    assert wake.split_wake_command(heard, _UMBRAL, _VARIANTES) == (True, "")


def test_una_frase_sin_wake_en_la_cabeza_sigue_sin_activar():
    """El corte nuevo no amplía la activación: el wake sigue exigiéndose en las
    3 primeras palabras."""
    assert wake.split_wake_command(
        "buenos días a todos crotolamo pon música", _UMBRAL, _VARIANTES) == (False, "")


# --- 2: el followup que solo SE PARECE al wake llega entero al agente --------

@pytest.mark.parametrize("followup", [
    "¿cómo me llamo?",
    "cómo lo llamo",
    "otro largo",
    "gracias, crotolamo",
])
def test_el_followup_parecido_al_wake_no_se_borra(followup):
    """listener: command = strip_wake_word(listen_command(...)) en la ventana de
    followup; un "" termina la conversación sin decir nada."""
    followup = normalize_text(followup)
    assert wake.strip_wake_word(followup, _VARIANTES) == followup


def test_el_followup_con_el_wake_delante_se_sigue_limpiando():
    """Si en el followup el dueño repite el wake, se quita como siempre."""
    assert wake.strip_wake_word("crotolamo pon música", _VARIANTES) == "pon música"
    assert wake.strip_wake_word("protolamo pon música", _VARIANTES) == "pon música"
    assert wake.strip_wake_word("crotolamo", _VARIANTES) == ""


def test_la_frase_entera_como_wake_solo_si_quien_llama_lo_sabe():
    """split_wake_command ya decidió que la frase activó: ahí la frase entera
    puede ser el wake mal oído (y no hay orden). En el followup, no."""
    assert wake.strip_wake_word("caroto llamo", _VARIANTES, whole=True) == ""
    assert wake.strip_wake_word("cómo lo llamo", _VARIANTES) == "cómo lo llamo"
