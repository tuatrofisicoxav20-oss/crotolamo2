# Auditoría completa de Crotolamo — 2026-07-09

Revisión de todo el código (`crotolamo/`, `interfaces/`, `scripts/`, `tests/`),
del estado del repo y del rendimiento de la máquina. Más la migración del motor
de inferencia a GLM.

## Resumen

El código **no estaba roto**: los 283 tests pasaban antes de tocar nada. Pero había
un agujero de seguridad grave, un script completamente inutilizable, y tres fugas de
rendimiento reales. Lo que hace lenta la máquina, sin embargo, no era ningún bug: era
inferir un modelo en CPU sin GPU.

Estado tras los arreglos: **311 tests pasan (con red y sin red), `ruff` limpio,
`mypy` limpio.**

## La máquina

| Recurso | Valor | Lectura |
|---|---|---|
| RAM total | 15 GiB | |
| RAM libre | 378 MiB | ahogada |
| Swap en uso | 2.5 GiB / 8 GiB | está paginando a disco |
| Núcleos | 12 | |
| GPU | Intel UHD integrada | **sin GPU dedicada** |

El modelo realmente en uso era `llama3.2:latest` (2 GB), fijado en
`config/crotolamo.local.toml:26` — no el `qwen2.5-coder:7b` del TOML base.

## Corregido

### Críticos

**1. Inyección de comandos — `crotolamo/tools/projects.py:198`**

`launch_project` construía `bash -lc 'cd "{project}" && "{launcher}"'`. `launcher`
sale de un `rglob` del directorio del proyecto: es un nombre de archivo real del
disco, no una constante. Un archivo llamado `run";rm -rf ~;".sh` dentro de un
proyecto configurado en `[projects]` ejecutaría al decir "lanza el proyecto".

Era el **único** punto del código que pasaba datos a un intérprete de shell; los ~30
`subprocess` restantes usan listas argv, que es lo correcto. Arreglado con
`shlex.quote()` en las dos interpolaciones.

**2. `scripts/wake_debug.py` estaba totalmente roto — `UnboundLocalError`**

Un `from crotolamo.settings import get_settings` **dentro** de `run()` (línea 84)
convertía el nombre en local para toda la función, así que la línea 51
(`settings = get_settings()`) reventaba antes de llegar. El script no arrancaba.

Python decide local-vs-global en tiempo de compilación escaneando la función entera,
así que un import tardío ensombrece el global desde la primera línea. Arreglado
borrando el import redundante (ya existía a nivel de módulo, línea 32). Verificado a
nivel de bytecode: `get_settings` ya no aparece en `co_varnames` de `run()`.

**3. El modelo Silero se recargaba del disco en cada comando — `crotolamo/voice/stt.py:202`**

`_record_silero` llamaba `load_silero_vad(onnx=True)` en **cada** grabación, y hasta
3 veces por orden con `smart_endpoint` activo. Whisper sí se cacheaba (`_models`,
línea 90); Silero no. Arreglado con un cache de módulo `_get_silero_vad()`, siguiendo
el patrón ya existente.

**4. Fuga de ficheros temporales — `crotolamo/voice/loop.py:141`**

`SttThread.run` hacía `continue` cuando el turno se invalidaba (barge-in), saltándose
el `unlink` del WAV ocho líneas más abajo. Cada comando abortado dejaba un fichero
huérfano en `/tmp` para siempre.

Curiosamente el otro camino que hace lo mismo (`stt.py:299`, `_listen_transcribe`)
**sí** protegía el borrado con `try/finally`. Era una asimetría entre dos rutas
equivalentes. Arreglado envolviendo en `try/finally` (un `continue` dentro de `try`
ejecuta el `finally`).

### Medios

**5. SSRF en `read_page` — `crotolamo/tools/search.py`**

`read_page` descargaba cualquier URL que el LLM pidiera, sin bloquear direcciones
internas: `http://localhost:11434` (el propio Ollama), `169.254.169.254` (metadatos
de nube), la LAN entera. Añadido `is_public_url()`, que resuelve el hostname (para
atrapar dominios públicos apuntando a `127.0.0.1`) y niega ante la duda.

**6. El VAD del hilo de escucha nunca reseteaba su estado — `crotolamo/voice/loop.py`**

Silero es una RNN con estado oculto persistente. `_record_silero` sí llamaba
`reset_states()`; `_SileroVad` (que corre ~30 veces/segundo durante toda la sesión)
nunca. El estado se arrastraba de un turno al siguiente, degradando la precisión del
VAD conforme avanzaba la sesión. Arreglado con `reset()` al cerrar cada comando.

**7. Dos copias del modelo Silero en RAM**

`loop.py` cargaba su propio ONNX y `stt.py` otro. Ahora comparten la instancia
cacheada.

**8. El loop agéntico dejaba el historial inconsistente — `crotolamo/core/agent.py`**

Al agotar `max_iterations`, se devolvía "Me enredé en demasiados pasos" sin añadirlo
al historial. Quedaba una secuencia `tool` sin `assistant` que la cerrara. Con Ollama
pasaba desapercibido; con el contrato de OpenAI (GLM) algunos motores lo rechazan.

**9-10. `mypy` (1 error) y `ruff` (3 errores)** — todos preexistentes, ahora limpios.

## Reportado, no corregido

Nada de esto es urgente; se documenta para que exista la decisión.

- **`EarThread` no responde a `shutdown` durante `mic.read()`** (`loop.py:262`).
  `mic.read()` es una lectura PortAudio bloqueante: `stop()` agota su `join(timeout=2)`
  y el hilo (daemon) queda colgado. En el servicio lo tapa `os._exit(0)`
  (`listener.py:89`), pero un `stop()` ordinario lo deja.
- **Inyección indirecta de prompt vía `read_page`.** El texto de una página web entra
  al contexto de un LLM con tool-calling activo; una página maliciosa puede intentar
  instruirlo. Cerrar el SSRF reduce el daño, pero no se elimina sin quitar la tool.
- **TOCTOU con symlinks** entre el `guard` y la tool: ambos resuelven la ruta, pero no
  atómicamente. Riesgo bajo en single-user.
- **Fuga de tool-call en streaming** (`agent.py:302`): si el modelo emite texto y luego
  JSON de tool-call en `content`, `_LiveStreamer` ya emitió el texto al patrón.
- **Race benigna en el stream global de sounddevice** (`tts.py`): `speak`, `beep` y
  `StreamSpeaker` comparten `sd.play/sd.stop`; si solapan, se cortan el audio.

## Rendimiento: no eran bugs

La lentitud de la máquina no venía del código, sino de la configuración de modelos.

| Causa | Coste | Qué hacer |
|---|---|---|
| Inferir en CPU sin GPU | ~17-22 s por turno con tool | **backend GLM** (ya migrado) |
| `keep_alive = "15m"` | ~2.5 GB residentes | baja a `"2m"` si vas justo |
| `whisper_model = "small"` + `wake = "base"` | ~+425 MB vs los defaults | `local.toml:6-7` |
| `qwen2.5:14b` instalado y **sin usar** | **9.0 GB de disco** | `ollama rm qwen2.5:14b-instruct-q4_K_M` |

Con `backend = "glm"` el modelo local deja de residir: se liberan los ~2.5 GB de RAM
y el CPU deja de saturarse en cada turno.

### Disco (25 GB en el repo)

No afecta al rendimiento en uso, solo ocupa espacio. Por petición explícita **no se
tocó nada de esto**; solo se borraron cachés regenerables (~50 MB).

```
19 GB   wakeword_training/   datasets de entrenamiento (conservados)
5.9 GB  .venv/               (conservado)
61 MB   voices/              (conservado)
```

## Migración a GLM

`[llm].backend = "glm"` en `config/crotolamo.toml`. Requiere
`export CROTOLAMO_GLM_API_KEY=...` (gratis en <https://z.ai>). Sin key, avisa y cae
solo a Ollama.

Diseño: `crotolamo/core/glm.py` es un **adaptador**. Traduce el formato de mensajes
Ollama ↔ OpenAI dentro del cliente, así que `Conversation`, `ToolAgent` y las tools no
se enteran de con qué motor hablan. `crotolamo/core/engine.py` elige uno u otro.

Las dos incompatibilidades reales del contrato:

1. OpenAI exige `tool_call_id` en los mensajes de rol `tool`; Ollama usa `name`.
2. OpenAI manda `function.arguments` como string JSON; Ollama, como dict.

Cero dependencias nuevas (stdlib, como el resto del núcleo).

Un detalle sutil: la ventana deslizante de `Conversation._trim()` podría, en
principio, dejar un mensaje `tool` cuyo `assistant` con `tool_calls` fue descartado.
Ollama lo toleraba (correlaciona por `name`); OpenAI/GLM devuelve un **400 opaco**.
Se comprobó que `_trim()` recorta por bloques completos y nunca lo produce (probado con
`max_turns` de 1 a 20, forzando recortes), pero `to_openai_messages` **descarta** un
`tool` huérfano de todos modos: no depende de una invariante que mantiene otro módulo.
Sin esa defensa, el síntoma habría sido "GLM funciona, y revienta tras ~20 turnos de
conversación por voz".

### Verificación

- 22 tests unitarios del adaptador (`tests/test_glm.py`), incluidos el reensamblado de
  tool_calls fragmentados en el SSE y la invariante conjunta con `_trim()`.
- Prueba end-to-end del `ToolAgent` completo contra un servidor HTTP falso que habla
  el contrato OpenAI: petición → tool_call → ejecución → reinyección del historial →
  segunda llamada. Confirma que el `tool_call_id` correlaciona.
- Los 4 caminos del selector (glm con/sin key, ollama, env pisando el TOML).
- Suite completa **con red y con el DNS saboteado**, para garantizar que los tests no
  dependen de conexión. 313 tests, `ruff` y `mypy` limpios.
- `glm-4.7-flash` confirmado como **gratuito** en la página oficial de precios de Z.ai
  (input, output y cacheado a $0), junto con `glm-4.5-flash`.

### Probado contra la API real (2026-07-09)

Ya con API key, verificado contra `api.z.ai`:

- **Tool-calling nativo: SÍ.** GLM devuelve el campo `tool_calls` (3/3 aciertos en
  `music_control`). El `ToolAgent` no necesita el fallback `_coerce_text_tool_calls`.
- **Loop agéntico completo:** LLM → tool → LLM, con el historial correcto
  (`user, assistant, tool, assistant`) y el `tool_call_id` correlacionado.
- **La personalidad se conserva**: responde en el registro desmadroso del `persona`.

**Hallazgo: `glm-4.7-flash` razona por defecto.** Trae `thinking` activado, y para un
"hola" gasta ~460 tokens de `reasoning_content` antes de responder.

| | thinking ON | thinking OFF |
|---|---|---|
| Charla | 6.87 s | **1.23 s** |
| Acción con tool | 1.57 s | **1.23 s** |
| Tool-calls correctos | 3/3 | 3/3 |

Se apagó por defecto (`[llm.glm].thinking = false`). Además de la latencia, los
`reasoning_tokens` cuentan contra el rate limit del tier gratuito.

**Latencia real medida** (con thinking apagado):

| Caso | Latencia |
|---|---|
| Charla, sin tools | ~1.3 s, muy estable |
| Acción con 8 tools, ritmo de voz (6 s entre comandos) | 2-9 s, mediana ~3.9 s |
| Acción con 8 tools, ráfaga sin pausa | se degrada: 1.4 → 17 → 42 s |

El tier gratuito **estrangula las ráfagas**. El uso por voz deja pausas naturales, así
que no lo dispara; un bucle de pruebas sí. Por eso `timeout = 60` (el peor caso medido
fue 42.6 s).

Comparado con `llama3.2` en CPU local (~17-22 s por acción), sigue siendo una mejora
clara, y el modelo deja de residir en RAM. Pero la promesa honesta es **~1.3 s en
charla y ~4 s en acciones**, no "1-2 s siempre".

## Respaldo offline (`crotolamo/core/fallback.py`)

**El hueco:** `build_llm()` decidía el motor UNA vez, al arrancar, mirando solo si
existía la API key. Con la key puesta y sin internet, Crotolamo respondía *"No pude
hablar con GLM, ¿hay internet?"* en cada turno, en vez de usar el `llama3.2` que ya
está instalado. Verificado empíricamente antes de arreglarlo.

**El arreglo:** `FallbackLLM` envuelve a los dos clientes. Ante un `LLMError` del
primario (red caída, 401, 429), el turno lo atiende Ollama.

Dos decisiones no obvias:

- **Circuit breaker.** Tras un fallo no se reintenta la nube en cada turno — se pagaría
  el timeout una y otra vez. Se marca caída durante `fallback_cooldown_s` (60 s) y
  luego se reintenta. Medido: turno 1 cae a Ollama (0.7 s), turno 2 va directo a
  Ollama (0.4 s) sin tocar la red, y tras el cooldown vuelve a GLM solo.
- **No reintentar si ya se habló.** En voz, `chat_stream` va soltando tokens al TTS. Si
  la nube revienta a media frase, reintentar con Ollama repetiría el principio por el
  altavoz. Si ya se emitió un token, el error se propaga.

Verificado con la API real cortando el DNS de `api.z.ai` (dejando `localhost` vivo para
Ollama): Crotolamo respondió con `llama3.2`, en personaje, y retomó GLM al volver la
red. 10 tests en `tests/test_fallback.py`.

## El camino de VOZ (streaming), probado contra la API real

Con `stream_speak = true`, la voz usa `chat_stream` (SSE), no `chat`. Ese camino se
probó por separado, porque los tests sintéticos solo validaban el formato SSE *que yo
había supuesto*.

**Confirmado:** `_consume_sse` reensambla el `tool_call` real idéntico al que devuelve
el no-streaming. Los `arguments` de GLM llegan completos en un delta (no fragmentados),
pero el reensamblado por `index` los soporta igual.

**Hallazgo: GLM verbaliza su intención antes de pedir la tool, pero solo en streaming.**

| llamada | `content` | `tool_calls` |
|---|---|---|
| `chat()` | `''` | `music_control(pause)` |
| `chat_stream()` | `'¡Claro que sí! Voy a pausar la música por ti.'` | `music_control(pause)` |

Con `stream_speak`, esa frase se habría hablado por el altavoz, seguida del resultado
real. `_LiveStreamer` no la filtraba: su heurística busca texto que empiece por `{` o
`` ``` `` (el JSON crudo de qwen), y esto es prosa normal.

Arreglado con `_LiveStreamer(hold_until_done=bool(schemas))`: si a la llamada se le
enviaron tools, cualquier texto se retiene hasta saber si hubo `tool_call`; si no se
enviaron (charla pura), se habla en vivo, que es el objetivo de `stream_speak`. El
mismo defecto existía con Ollama, menos visible.

Verificado end-to-end contra la API real: "¿cuánto espacio libre tengo?" habla **solo**
el resultado; "hola crotolamo" responde en vivo y en personaje.

## Arranque: la key y el `.desktop`

`crotolamo.desktop` lanza `Exec=kitty -e launch.sh`, que **no** es un shell interactivo:
`~/.zshrc` nunca se carga. Con la key solo en `.zshrc`, un doble clic habría caído a
`llama3.2` en silencio — exactamente el fallo que la migración pretendía evitar.

`launch.sh` ahora carga `~/.config/crotolamo/env` (permisos `600`) al arrancar, así que
la key llega tanto por doble clic como por terminal. Verificado en un entorno vacío
(`env -i`).
