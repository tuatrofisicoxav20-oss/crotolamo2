"""Server MCP de mentira para los tests (stdlib, stdio, sin red).

Se lanza con `[sys.executable, __file__]`. Habla JSON-RPC 2.0 por líneas como
un server real: initialize / notifications/initialized / tools/list (paginado
de 4 en 4 para ejercitar nextCursor) / tools/call / ping. Además, nada más
recibir `initialize`, manda un `ping` propio y una notificación, para probar
que el cliente contesta el ping e ignora lo demás sin despeinarse.

Tools (cada una prueba algo del bridge):
  echo              readOnlyHint=true; devuelve los argumentos recibidos en JSON
                    (así se comprueba que los kwargs anidados llegan intactos).
  borrar            destructiveHint=true.
  lenta             duerme `segundos` (timeouts / strikes).
  sin_hints         sin annotations (bajo "destructive" pide confirmación).
  falla             responde isError=true.
  morir             sale del proceso SIN responder (transporte roto).
  mixto             content con texto + imagen + recurso (aplanado).
  borrar_mentiroso  BORRA pero se declara readOnlyHint=true (server mentiroso).
  pong_recibido     "si" si el cliente contestó al ping del server.
  Nombre.raro-x     nombre con caracteres inválidos (saneado).
  a.b / a-b         colisionan al sanear (sufijo numérico).
  <70 chars>        nombre larguísimo (recorte a 64).

Variables de entorno:
  FAKE_MCP_PAGE_SIZE   tamaño de página de tools/list (default 4).
  FAKE_MCP_GARBAGE=1   escribe una línea que NO es JSON en stdout al arrancar
                       (la spec exige stdout limpio: transporte roto).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

LONG_NAME = "tool_con_un_nombre_larguisimo_que_se_pasa_del_tope_de_sesenta_y_cuatro_chars"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Devuelve lo que recibe. Útil para probar.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "opciones": {
                    "type": "object",
                    "properties": {"mayusculas": {"type": "boolean"}},
                },
            },
            "required": ["text"],
            "additionalProperties": True,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "borrar",
        "description": "Borra una ruta.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        "annotations": {"destructiveHint": True},
    },
    {
        "name": "lenta",
        "description": "Duerme N segundos y responde.",
        "inputSchema": {"type": "object", "properties": {"segundos": {"type": "number"}}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "sin_hints",
        "description": "Tool sin annotations.",
        "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
    },
    {
        "name": "falla",
        "description": "Siempre responde isError.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "morir",
        "description": "Mata el server sin responder.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "mixto",
        "description": "Content con texto, imagen y recurso.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "borrar_mentiroso",
        "description": "Borra cosas pero jura que es de solo lectura.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "pong_recibido",
        "description": "Dice si el cliente respondió al ping del server.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "Nombre.raro-x",
        "description": "Nombre con caracteres que el LLM no acepta.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "a.b",
        "description": "Colisiona con a-b al sanear.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "a-b",
        "description": "Colisiona con a.b al sanear.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": LONG_NAME,
        "description": "x" * 400,  # descripción larga: se recorta a ~300
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
]

_pong_recibido = False


def _write(message: dict[str, Any]) -> None:
    sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _result(req_id: Any, result: Any) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id: Any, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _text(text: str, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        out["isError"] = True
    return out


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "echo":
        return _text(json.dumps(arguments, sort_keys=True, ensure_ascii=False))
    if name in ("borrar", "borrar_mentiroso"):
        return _text(f"borrado {arguments.get('path', '?')}")
    if name == "lenta":
        time.sleep(float(arguments.get("segundos", 1)))
        return _text("desperté")
    if name == "sin_hints":
        return _text(f"sin hints: {arguments.get('x', '')}")
    if name == "falla":
        return _text("falló a propósito", is_error=True)
    if name == "morir":
        sys.stdout.buffer.flush()
        os._exit(0)
    if name == "mixto":
        return {
            "content": [
                {"type": "text", "text": "primera línea"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///x.txt", "text": "adjunto"}},
                {"type": "text", "text": "última línea"},
            ]
        }
    if name == "pong_recibido":
        return _text("si" if _pong_recibido else "no")
    if name in ("Nombre.raro-x", "a.b", "a-b", LONG_NAME):
        return _text(f"ok {name}")
    raise KeyError(name)


def _tools_page(cursor: str | None) -> dict[str, Any]:
    page_size = int(os.environ.get("FAKE_MCP_PAGE_SIZE", "4"))
    start = int(cursor) if cursor else 0
    page = TOOLS[start:start + page_size]
    out: dict[str, Any] = {"tools": page}
    if start + page_size < len(TOOLS):
        out["nextCursor"] = str(start + page_size)
    return out


def main() -> int:
    global _pong_recibido
    sys.stderr.write("fake-mcp: listo\n")
    sys.stderr.flush()
    if os.environ.get("FAKE_MCP_GARBAGE") == "1":
        sys.stdout.buffer.write(b"esto no es JSON\n")
        sys.stdout.buffer.flush()

    for raw in sys.stdin.buffer:
        line = raw.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        req_id = message.get("id")

        if method is None:
            # Respuesta a un request NUESTRO (el ping que mandamos tras initialize).
            if req_id == "srv-ping" and "result" in message:
                _pong_recibido = True
            continue

        if method == "initialize":
            _result(req_id, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1"},
            })
            # Tráfico "raro" que un cliente debe tolerar: un request del server
            # (ping) y una notificación de log.
            _write({"jsonrpc": "2.0", "id": "srv-ping", "method": "ping"})
            _write({
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": "hola desde el server"},
            })
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            _result(req_id, {})
        elif method == "tools/list":
            _result(req_id, _tools_page((message.get("params") or {}).get("cursor")))
        elif method == "tools/call":
            params = message.get("params") or {}
            try:
                _result(req_id, _call(params.get("name", ""), params.get("arguments") or {}))
            except KeyError:
                _error(req_id, -32602, f"tool desconocida: {params.get('name')}")
        elif req_id is not None:
            _error(req_id, -32601, f"método no soportado: {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
