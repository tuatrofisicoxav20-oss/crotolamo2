"""Cliente MCP (Model Context Protocol) para Crotolamo, M4.

Un server MCP es un proceso externo que ofrece tools por JSON-RPC 2.0. Este
paquete lo lanza, le pide su lista de tools y las registra en el registry de
Crotolamo como si fueran tools nativas, pasando por el mismo guard y la misma
confirmación del patrón. Stdlib puro (subprocess + threading + json), como el
resto del núcleo: nada de SDKs.

- `client.py`: transporte stdio (`StdioMCPClient`) y sus errores.
- `bridge.py`: config, registro en el registry/router, timeouts y desregistro.
"""

from __future__ import annotations

from crotolamo.mcp.client import MCPError, MCPTimeout, MCPTransportError, StdioMCPClient

__all__ = ["MCPError", "MCPTimeout", "MCPTransportError", "StdioMCPClient"]
