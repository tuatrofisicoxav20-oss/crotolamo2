"""Streaming de tokens hacia el patrón, con retención de tool-calls.

Extraído de `agent.py` para que el loop agéntico quede legible; la lógica es
idéntica.
"""

from __future__ import annotations

from typing import Callable


class LiveStreamer:
    """Streamea tokens al patrón en vivo, salvo que la respuesta empiece como un
    tool-call JSON ('{', '[' o cerca de código) — en ese caso retiene, para no
    filtrar el JSON crudo de los tool-calls que qwen emite en `content`.

    `hold_until_done=True` retiene SIEMPRE. Se usa cuando a esta llamada se le
    enviaron tools: el modelo puede verbalizar su intención ANTES de pedir la tool
    ("Voy a pausar la música por ti") y en voz eso se hablaría, seguido del
    resultado real. Medido contra GLM: en streaming emite ese preámbulo; en
    no-streaming, `content` viene vacío. Si no se enviaron tools, lo que salga ES
    la respuesta final y se habla en vivo, que es el objetivo de `stream_speak`.
    """

    def __init__(self, on_token: Callable[[str], None],
                 hold_until_done: bool = False) -> None:
        self._on_token = on_token
        self._buf: list[str] = []
        self._decision: str | None = "hold" if hold_until_done else None

    def feed(self, chunk: str) -> None:
        self._buf.append(chunk)
        if self._decision == "stream":
            self._on_token(chunk)
            return
        if self._decision == "hold":
            return
        head = "".join(self._buf).lstrip()
        if len(head) < 2:
            return  # aún no hay suficiente para decidir
        if head[0] in "{[" or head.startswith("```"):
            self._decision = "hold"
        else:
            self._decision = "stream"
            self._on_token(head)  # soltamos lo acumulado de golpe y seguimos en vivo

    def flush_if_held(self, final_text: str) -> None:
        """Si retuvimos pero resultó ser texto final, lo emitimos completo."""
        if self._decision != "stream":
            self._on_token(final_text)
