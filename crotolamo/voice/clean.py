"""Limpieza de texto para el TTS: quitar markdown y formato de chat.

El LLM responde con formato de chat (negritas, listas, títulos, tablas, links,
emojis, URLs) aunque el prompt le pida que no, y el TTS lo lee literal:
"asterisco asterisco negritas asterisco asterisco". Este módulo deja el texto
como lo diría una persona.

Es una función PURA y se aplica en UN solo punto: `TTS.speak`, por donde pasa
todo lo que Crotolamo habla (StreamSpeaker, speak_sentences, MouthThread y los
avisos de say()), sea cual sea el motor de síntesis. Así ningún camino de voz
se olvida de limpiar, y un motor nuevo hereda la limpieza sin tocar nada.

Streaming: StreamSpeaker acumula tokens y solo manda a speak() frases ya
cerradas, así que un "**" partido entre dos chunks llega reensamblado aquí.
Limpiar a nivel de caracteres nunca se come palabras: solo quita símbolos,
marcadores y URLs, y colapsa espacios.
"""

from __future__ import annotations

import re
import unicodedata

# Bloques de código completos (```...```) se descartan enteros: leer código en voz
# alta no sirve de nada. Una cerca sin cerrar (frase partida por el streaming)
# solo pierde el marcador y su etiqueta de lenguaje.
_CERCA_COMPLETA = re.compile(r"```.*?```", re.DOTALL)
_CERCA_SUELTA = re.compile(r"```[\w+-]*")
_CODIGO_INLINE = re.compile(r"`([^`\n]*)`")
# Imágenes antes que links: ![alt](url) contiene un link.
_IMAGEN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# La URL termina antes de un espacio o de un cierre ( ) [ ] « » " ' y NO se lleva
# la puntuación final de la frase ("...x.com." conserva el punto).
_URL = re.compile(
    r"""(?:https?://|www\.)[^\s<>()\[\]{}«»"']*[^\s<>()\[\]{}«»"'.,;:!?]""",
    re.IGNORECASE,
)
# Marcadores al inicio de línea: títulos, viñetas (también numeradas), citas.
_TITULO = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_VINETA = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,2}[.)])[ \t]+(?=\S)", re.MULTILINE)
_CITA = re.compile(r"^[ \t]*>+[ \t]?", re.MULTILINE)
# Líneas que son solo formato: reglas horizontales (---, ***) y separadores de tabla.
_REGLA = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
_SEP_TABLA = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-{2,}:?[ \t]*\|)+[ \t]*(?::?-{2,}:?)?[ \t]*\|?[ \t]*$",
                        re.MULTILINE)
# Restos de énfasis y formato que no forman parte de ninguna palabra.
_RESTOS = re.compile(r"[*~#>]")
_A_ESPACIO = re.compile(r"[_|]")
_PARENTESIS_VACIOS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
_ESPACIO_ANTES_PUNTUACION = re.compile(r"\s+([,.;:!?])")
_ESPACIOS = re.compile(r"\s+")
_SOLO_PUNTUACION = re.compile(r"^[\W_]*$")

# Categorías Unicode que se descartan: emojis y símbolos "otros" (So), marcas
# envolventes (Me: el recuadro de los keycaps), formato invisible (Cf: ZWJ,
# selectores), y basura (Cs, Co, Cn). Las letras con acento y la ñ son "L*"
# y las marcas combinantes (Mn) se conservan: en NFC no quedan sueltas.
_CATEGORIAS_FUERA = {"So", "Me", "Cf", "Cs", "Co", "Cn"}
# Modificadores de emoji (tono de piel) y selectores de variación: no son
# pronunciables y sin ellos el emoji base ya se fue por "So".
_RANGOS_FUERA = ((0xFE00, 0xFE0F), (0x1F3FB, 0x1F3FF), (0xE0100, 0xE01EF))


def _sin_simbolos(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if unicodedata.category(ch) in _CATEGORIAS_FUERA:
            continue
        if any(lo <= cp <= hi for lo, hi in _RANGOS_FUERA):
            continue
        out.append(ch)
    return "".join(out)


def limpiar_para_voz(texto: str) -> str:
    """Devuelve `texto` sin markdown ni formato, listo para que lo lea el TTS.

    Quita: bloques de código, código inline (deja su contenido), links
    [texto](url) -> texto, URLs sueltas, títulos (#), viñetas (- * • y 1.),
    citas (>), reglas y separadores de tabla, emojis, y los restos de
    * _ ~ | > #. Colapsa los espacios. Texto normal (acentos, ñ, ¿¡, números,
    horas, decimales) pasa intacto.
    """
    if not texto:
        return ""
    text = unicodedata.normalize("NFC", texto)

    text = _CERCA_COMPLETA.sub(" ", text)
    text = _CERCA_SUELTA.sub(" ", text)
    text = _IMAGEN.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _SEP_TABLA.sub(" ", text)
    text = _REGLA.sub(" ", text)
    text = _TITULO.sub("", text)
    text = _VINETA.sub("", text)
    text = _CITA.sub("", text)
    text = _CODIGO_INLINE.sub(r"\1", text)
    text = _sin_simbolos(text)
    text = _RESTOS.sub("", text)
    text = _A_ESPACIO.sub(" ", text)
    text = _PARENTESIS_VACIOS.sub(" ", text)
    text = _ESPACIO_ANTES_PUNTUACION.sub(r"\1", text)
    text = _ESPACIOS.sub(" ", text).strip()

    if _SOLO_PUNTUACION.match(text):
        return ""
    return text
