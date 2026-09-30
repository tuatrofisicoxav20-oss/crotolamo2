# Fase M4 — Cliente MCP (Model Context Protocol) por stdio

Crotolamo puede **tomar prestadas tools de servers MCP externos**: procesos que
lanza él mismo y con los que habla por stdin/stdout en JSON-RPC 2.0. Sus tools
aparecen al LLM junto a las nativas, con el mismo formato, y pasan por el
**mismo guard de rutas y la misma confirmación del patrón**. El agente no sabe
cuáles son MCP y cuáles no.

Stdlib puro, como el resto del núcleo: `subprocess` + `threading` + `json`.
Cero SDKs, cero `pip install`.

## Arquitectura

```
[mcp] en crotolamo.toml
   └─ bridge.register_mcp_tools()      (al armar el registry, si enabled)
        ├─ StdioMCPClient.start()      subprocess + hilo lector + drenador de stderr
        ├─ initialize / initialized    handshake (protocolVersion 2025-06-18)
        ├─ tools/list (paginado)       -> Tool(name="mcp_<prefix>_<tool>", strict_args=False)
        ├─ registry.register(...)      el LLM las ve como cualquier otra
        └─ router.register_group(...)  grupo dinámico "mcp:<server>" para el routing local

tool-call del LLM -> guard (rutas, recursivo) -> confirmación -> tools/call -> texto al LLM
```

- `crotolamo/mcp/client.py` — `StdioMCPClient`: transporte, futuros por `id`,
  `ping`, timeouts (`MCPTimeout`) y transporte roto (`MCPTransportError`).
- `crotolamo/mcp/bridge.py` — config, traducción a `Tool`, política de
  confirmación, strikes, desregistro en caliente y cierre `atexit`.

## Configuración

Todo en `config/crotolamo.toml`, sección `[mcp]` (apagada por defecto):

```toml
[mcp]
enabled = false            # true lanza los servers al arrancar
timeout_s = 20             # por llamada a una tool
startup_timeout_s = 15     # presupuesto: lanzar + handshake + tools/list
confirm = "destructive"    # "destructive" | "always" | "never"

[mcp.servers.archivos]
command  = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "~/Documentos"]
keywords = ["archivo", "archivos", "documento", "carpeta"]
confirm  = "destructive"   # pisa el global
# timeout_s / startup_timeout_s / cwd / prefix también por server
[mcp.servers.archivos.env]
# MI_TOKEN = "$MI_TOKEN"   # se SUMA al entorno; ~ y $VARS se expanden
```

Los servers son **tablas nombradas** (`[mcp.servers.<nombre>]`), no un array:
`crotolamo.local.toml` se fusiona con `_deep_merge`, que es recursivo sobre
dicts pero **no** sobre listas. Así el local puede retocar un server
(`timeout_s`) o añadir otro sin repetir ni pisar los demás.

Validación: un valor inválido cae a su default con `WARNING`; un server sin
`command` (lista de strings no vacía) se ignora. Un server que **no arranca**
(binario ausente, muere al lanzarse, no responde en `startup_timeout_s`) se
salta con `WARNING`: **nunca impide que Crotolamo arranque**.

## Nombres de tools

`mcp_<prefix>_<tool>`, con `prefix` = nombre del server salvo que lo cambies.
Se sanean a `[A-Za-z0-9_]`, se recortan a 64 caracteres y si dos colisionan el
segundo lleva sufijo `_2`, `_3`... La descripción es la del server, recortada a
~300 caracteres, con el sufijo ` [MCP: <server>]` para que el modelo sepa de
dónde viene. El `inputSchema` se respeta tal cual (anidados,
`additionalProperties`...), solo se garantiza `type: object` + `properties` y
se quita `$schema`.

Por eso las tools MCP llevan `Tool.strict_args = False`: el agente **no filtra**
sus kwargs a los de primer nivel de `properties` (ese filtro existe para los
inventos del 3B sobre tools planas y mutilaría un esquema anidado).

## Política de confirmación

| `confirm`       | Comportamiento                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| `"destructive"` | (default) pide confirmación **salvo** que el server declare `readOnlyHint: true` o `destructiveHint: false` |
| `"always"`      | pide confirmación para TODAS las tools del server                              |
| `"never"`       | nunca pregunta (solo para servers de solo lectura y de fiar)                    |

> **ADVERTENCIA (léela).** Los `annotations` los declara **el server** y la
> spec los define como *advisory*: son de buena fe, no una garantía. Bajo
> `"destructive"`, un server mentiroso que marque una tool destructiva con
> `readOnlyHint: true` **se salta la confirmación**. `tests/test_mcp_bridge.py::
> test_server_mentiroso_se_salta_la_confirmacion_bajo_destructive` documenta ese
> comportamiento a propósito. Para servers que no controlas o no conoces, usa
> `confirm = "always"`: es la única defensa real. El guard de rutas sigue
> aplicando siempre (ver abajo), pero solo cubre rutas.

Además, el **guard es recursivo** (M4): recorre listas y dicts anidados de los
argumentos (hasta 8 niveles) buscando strings que parezcan rutas, con las
mismas tres zonas de siempre (libre / confirmar / bloqueado). Una ruta
escondida en `{"opciones": {"ruta": "/etc/passwd"}}` o en `paths: [...]` se
bloquea antes de llegar al server. Más hondo de 8 niveles ya no se inspecciona
(el tope existe para que un JSON patológico no reviente la pila).

## Timeouts y strikes

- Cada `tools/call` corre bajo el `timeout_s` del server. Si expira, el LLM
  recibe un **soft-error en personaje** ("El server MCP 'x' tardó demasiado en
  responder, patrón...") — texto, no excepción — y el server acumula un strike.
- Al **2º timeout consecutivo** se desregistran TODAS sus tools del registry,
  se quita su grupo del router y se cierra el proceso (`WARNING` en el log). Un
  server colgado no debe costar 20 s por cada intento del modelo.
- Una llamada **exitosa resetea** los strikes.
- Un **fallo de transporte** (proceso muerto, EOF en stdout, pipe rota, basura
  no-JSON en stdout) desregistra **de inmediato**: no hay nada que reintentar.
- Un error JSON-RPC o un `isError: true` del server llegan como soft-error con
  el texto del server, para que el modelo lo lea y reaccione.

Los soft-errors empiezan por `El server MCP '<n>'`, un prefijo que **no** está en
`tool_parsing.HARD_ERROR_PREFIXES`: son texto apto para decírselo al patrón.

Hoy no hay reconexión automática: un server desregistrado vuelve al reiniciar
Crotolamo.

## Routing (motor local)

Con GLM (nube) el modelo ve **todas** las tools, MCP incluidas. Con Ollama
local y `tool_routing = true`, cada server es un **grupo dinámico** del router
(`mcp:<server>`) con sus `keywords`. Si no defines `keywords`, se derivan del
nombre del server, su prefijo y las palabras de los nombres/descripciones de
sus tools (minúsculas, sin acentos, con tope). Como las descripciones suelen
venir en inglés y el patrón habla español, lo que de verdad enruta es el nombre
del server: **define `keywords` a mano** para cualquier server que uses en serio.

Cuando varios grupos MCP matchean la misma frase, su orden relativo **rota en
round-robin** entre llamadas, para que con el tope `max_tools` ninguno se muera
de hambre. Los grupos estáticos conservan su orden por especificidad, y cuando
**no** matchea ningún grupo MCP el routing es exactamente el de siempre.

## Cómo probarlo sin instalar nada

`tests/fake_mcp_server.py` es un server MCP de mentira (stdlib, stdio) con
tools para cada caso: `echo` (solo lectura), `borrar` (destructiva), `lenta`
(duerme N s), `sin_hints`, `falla` (`isError`), `morir` (mata el proceso),
`mixto` (texto + imagen + recurso), `borrar_mentiroso`... Para verlo en vivo:

```toml
# config/crotolamo.local.toml
[mcp]
enabled = true

[mcp.servers.fake]
command  = ["python3", "tests/fake_mcp_server.py"]   # ruta absoluta si hace falta
keywords = ["eco", "repite"]
timeout_s = 3
```

```bash
CROTOLAMO_LOG=DEBUG python -m crotolamo shell
# "repite hola con el eco"  -> mcp_fake_echo
# "borra /tmp/x con fake"   -> pide confirmación (destructiveHint)
```

Los tests (`tests/test_mcp_client.py`, `tests/test_mcp_bridge.py`) cubren
handshake, paginación, timeouts, strikes, transporte roto, política de
confirmación, guard recursivo, routing y config; con timeouts de 0.3-0.5 s
para que la suite siga rápida.

## Limitaciones (lo que NO hace)

- **Solo transporte stdio.** Nada de HTTP / SSE / Streamable HTTP: un server
  remoto habría que envolverlo con un puente local (p.ej. `mcp-remote`).
- **Solo tools.** Sin `resources`, `prompts`, `sampling`, `roots`,
  `elicitation` ni suscripciones. Los requests del server que no sean `ping`
  se contestan con `-32601` (método no soportado) y sus notificaciones se
  ignoran (`DEBUG`).
- Sin reconexión automática tras desregistrar un server.
- Sin `listChanged`: la lista de tools se lee una vez al arrancar.
- El contenido no textual (imágenes, audio) se resume en una marca
  (`[imagen image/png]`); no se le pasa al modelo.
- El server hereda tu entorno completo más `env`; si quieres aislarlo, hazlo
  desde el `command` (contenedor, `env -i`...).
- Los `annotations` no se verifican: ver la advertencia de arriba.
