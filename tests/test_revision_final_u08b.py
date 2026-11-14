"""Revisión final, unidad u08b: el keep_alive de [llm] tal como lo documenta el toml.

El comentario de [llm] en config/crotolamo.toml dice: sube a "1h" o "-1" (siempre
residente). tomllib lee `keep_alive = "-1"` como STRING y el cliente lo mandaba tal
cual. Ollama decodifica un keep_alive string con time.ParseDuration de Go, que exige
unidad ("missing unit in duration"), y el bind fallido es un 400: CADA request local
moría (el warm-up, el respaldo sin internet, todos los turnos con backend=ollama).
Un número sí lo acepta: segundos, y negativo = residente para siempre (igual que
OLLAMA_KEEP_ALIVE, que lee "-1" sin unidad como segundos).

Ni un byte de red real: `post_json` del transporte se sustituye por un Ollama de
mentiras que aplica la regla de Duration.UnmarshalJSON (api/types.go).
"""

from __future__ import annotations

import io
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from crotolamo.core.llm import LLMClient
from crotolamo.settings import Settings

MSGS = [{"role": "user", "content": "hola"}]

# time.ParseDuration de Go: "0" a secas, o uno o más <número><unidad>, con signo.
_DURACION_GO = re.compile(r"[+-]?(0|((\d+(\.\d*)?|\.\d+)(ns|us|µs|μs|ms|s|m|h))+)")


class _Resp:
    """Respuesta HTTP mínima con la superficie que usa LLMClient."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status = status
        self._buf = io.BytesIO(json.dumps(body).encode("utf-8"))

    def read(self, *args: Any) -> bytes:
        return self._buf.read(*args)

    def getheader(self, name: str, default: Any = None) -> Any:
        return default


def _ollama_falso(enviados: list[dict[str, Any]]):
    """post_json que contesta como Ollama al decodificar keep_alive."""

    def post_json(path: str, payload: dict[str, Any]) -> _Resp:
        # Lo que de verdad viaja: JSON ida y vuelta (un int y un "-1" se distinguen).
        wire = json.loads(json.dumps(payload))
        enviados.append(wire)
        ka = wire.get("keep_alive")
        if isinstance(ka, str) and not _DURACION_GO.fullmatch(ka):
            return _Resp(400, {"error": f'time: missing unit in duration "{ka}"'})
        return _Resp(200, {"message": {"role": "assistant", "content": "a la orden"},
                           "done": True})

    return post_json


def _cliente_desde_toml(texto: str) -> LLMClient:
    raw = tomllib.loads(texto)
    settings = Settings(raw=raw, user="patron", home=Path("/tmp"))
    return LLMClient.from_settings(settings)


@pytest.mark.parametrize(
    ("linea", "esperado"),
    [
        ('keep_alive = "-1"', -1),          # lo que sugiere el comentario del toml
        ('keep_alive = "3600"', 3600),
        ('keep_alive = "1.5"', 1.5),
        ('keep_alive = " -1 "', -1),
        ("keep_alive = -1", -1),            # sin comillas ya funcionaba: sigue igual
        ('keep_alive = "15m"', "15m"),      # el de fábrica: sigue saliendo igual
        ('keep_alive = "1h"', "1h"),
        ('keep_alive = "-1m"', "-1m"),
        ('keep_alive = "0"', 0),            # descarga inmediata, también como número
    ],
)
def test_keep_alive_documentado_llega_a_ollama_en_una_forma_que_acepta(linea, esperado):
    cliente = _cliente_desde_toml(f"[llm]\n{linea}\n")
    enviados: list[dict[str, Any]] = []
    cliente._transport.post_json = _ollama_falso(enviados)   # type: ignore[method-assign]

    resp = cliente.chat(MSGS)

    assert resp.content == "a la orden"
    assert enviados[0]["keep_alive"] == esperado
    assert type(enviados[0]["keep_alive"]) is type(esperado)


def test_keep_alive_menos_uno_entre_comillas_no_es_un_400_en_cada_turno():
    """El escenario del hallazgo, de punta a punta: toml -> from_settings -> chat."""
    cliente = _cliente_desde_toml('[llm]\nkeep_alive = "-1"\n')
    enviados: list[dict[str, Any]] = []
    cliente._transport.post_json = _ollama_falso(enviados)   # type: ignore[method-assign]

    for _ in range(2):
        assert cliente.chat(MSGS).content == "a la orden"
    assert [p["keep_alive"] for p in enviados] == [-1, -1]


def test_sin_keep_alive_en_el_toml_sigue_el_default():
    cliente = _cliente_desde_toml("[llm]\n")
    assert cliente._payload(MSGS, None, stream=False)["keep_alive"] == "15m"
