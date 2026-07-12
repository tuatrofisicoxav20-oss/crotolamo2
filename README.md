# Crotolamo 2

Asistente local **agéntico** para Fedora. Reescritura modular de Crotolamo 1.

El cambio de paradigma central:

```
Crotolamo 1:  LLM genera bash  →  ejecutas bash crudo   (inseguro, no encadena)
Crotolamo 2:  LLM elige tool(nombre, args) → función Python TIPADA y SEGURA
                → resultado de vuelta al LLM → decide siguiente paso (loop)
```

## Stack

- Python 3.11+ (núcleo: solo stdlib)
- Motor LLM conmutable: **GLM** (nube, Z.ai) u **Ollama** (local). Ambos con
  tool-calling nativo. Ver *Motor de inferencia* abajo.
- Voz (Fase 5, opcional): faster-whisper + Piper `es_MX-ald-medium`

## Motor de inferencia

Se elige con `[llm].backend` en la config: `"glm"` o `"ollama"`.

| | `glm` (nube) | `ollama` (local) |
|---|---|---|
| Latencia por turno | ~1-2 s | ~17-22 s con tool (CPU sin GPU) |
| RAM del modelo | 0 (no reside) | ~2-5 GB residentes |
| Offline | no | sí |
| Coste | $0 con `glm-4.7-flash` | $0 |

**Por qué existe la opción:** en una laptop sin GPU dedicada, inferir en CPU es lo
que hace lenta la máquina. `glm-4.7-flash` es gratuito y saca la inferencia del
equipo. Todo el andamiaje de velocidad (`tool_routing`, `fastpath`, `direct_tools`)
se construyó para sobrevivir al CPU; con GLM sigue ayudando, pero deja de ser vital.

Para usar GLM necesitas una API key (gratis en <https://z.ai>). **Nunca se guarda en
el TOML** (que va a git): va en el entorno.

La forma recomendada es un archivo con permisos `600`, que `launch.sh` carga solo:

```bash
mkdir -p ~/.config/crotolamo
echo 'export CROTOLAMO_GLM_API_KEY="tu-key"' > ~/.config/crotolamo/env
chmod 600 ~/.config/crotolamo/env
```

`launch.sh` lo lee al arrancar, así que funciona también con el `.desktop` (doble
clic), donde `~/.zshrc` **no** se carga: `Exec=kitty -e launch.sh` no es un shell
interactivo. Si además quieres la key en tus terminales, añade a `~/.zshrc`:

```bash
[ -f ~/.config/crotolamo/env ] && source ~/.config/crotolamo/env
```

O expórtala a mano para una sesión suelta:

```bash
export CROTOLAMO_GLM_API_KEY="tu-key"
python -m crotolamo shell
```

### Sin internet sigue funcionando

Con `backend = "glm"`, el cliente va envuelto en un respaldo local. Si la nube falla
**en caliente** (se cayó el wifi, caducó la key, un 429), los turnos siguientes los
atiende Ollama con el modelo local. Más lento, pero Crotolamo no se queda mudo.

Tras un fallo, no se reintenta la nube en cada turno (se pagaría el timeout una y otra
vez): se espera `[llm.glm].fallback_cooldown_s` segundos y se vuelve a probar. En
cuanto la nube responde, se retoma sola.

Una excepción deliberada: si la nube falla **a media frase** en modo voz (ya salieron
tokens por el altavoz), ese turno falla en vez de reintentar — repetir el principio de
la frase sonaría peor que un error.

Si `backend = "glm"` y no hay key **al arrancar**, ni se intenta: se usa Ollama
directo. Para forzar un motor sin editar nada:

```bash
CROTOLAMO_LLM_BACKEND=ollama python -m crotolamo shell   # la env pisa el TOML
```

Ambos clientes exponen el mismo contrato (`chat`, `chat_stream` → `ChatResponse`), así
que el agente, la memoria y las tools no distinguen cuál corre. `crotolamo/core/glm.py`
es un adaptador: traduce el formato de mensajes de Ollama al de OpenAI (que es el que
habla Z.ai) y normaliza la respuesta de vuelta.

## Uso

La forma más fácil — **el launcher con menú** (detecta el venv solo):

```bash
./launch.sh            # menú: doctor / shell / voz / smoke
./launch.sh doctor     # o directo: doctor|shell|listen|smoke|version
```

O a mano:

```bash
python -m crotolamo --version
python -m crotolamo doctor      # auditor de salud
python -m crotolamo shell       # REPL de texto
python -m crotolamo listen      # bucle de voz wake-word (requiere extra [voice])
```

Para la voz: `pip install -e ".[voice]"` (faster-whisper, sounddevice, piper-tts) y
un modelo Piper `.onnx` en `[paths].voces`.

## Configuración

Todo vive en `config/crotolamo.toml` — **cero rutas hardcodeadas**. Para overrides
locales sin tocar el archivo versionado, crea `config/crotolamo.local.toml` (ignorado
por git) con solo las claves que quieras cambiar.

## Estado por fases

- [x] **Fase 0** — andamiaje: repo, config, settings, doctor
- [x] **Fase 1** — núcleo conversacional con memoria de corto plazo (REPL)
- [x] **Fase 2** — tool-calling: el loop agéntico + tools de desktop/search seguras
- [x] **Fase 3** — tools que leen/razonan sobre proyectos + archivos seguros
- [x] **Fase 4** — memoria persistente (SQLite)
- [x] **Fase 5** — voz: implementada y verificada por **smoke test circular** (`scripts/smoke_voz.py`: TTS→STT sin micrófono); falta validación de campo con micrófono real.
- [x] **Fase 6** — extensiones: streaming token-a-token, TTS por frases, memoria fuzzy, búsqueda en proyectos, hotkeys

### Notas de voz (honestas)
- El **wake word** usa Whisper `tiny` (barato) y el **comando** usa `base`; ambos en CPU.
- La **latencia depende del CPU**: en un i5 sin GPU, el smoke (TTS+STT de una frase) ronda
  los ~12 s en frío (incluye carga de modelos); en caliente baja. Con GPU sería fluido.
- La reproducción usa `sounddevice` (no ffplay). Requiere `portaudio` en el sistema.
- Lo único no probado de forma automática es la captura por **micrófono real** (`--mic`).

## Procedencia (qué se migró de Crotolamo 1)

Fuente: `~/Documentos/chapi_assistant` (la versión viva de C1). El checkpoint más
completo de C1 está guardado como referencia en `reference/c1_checkpoint/` y
registrado como el proyecto `crotolamo1` (las tools de C2 pueden leerlo). Ver
`reference/README.md`.

| Pieza de C1 | Destino en C2 | Acción |
|---|---|---|
| `chapi_shell.py::SYSTEM` (persona) | `crotolamo/core/persona.py` | migrado + adaptado a tools |
| `chapi_shell.py::ask_ollama` | `crotolamo/core/llm.py` | reescrito con tool-calling |
| bash crudo + `DANGEROUS_PATTERNS` | — | **descartado** (lo reemplaza `safety/guard.py`) |
| `skills.py` regex `parse_*` | — | **descartado** (lo reemplaza tool-calling) |
| `skills.py` open_*/search_* | `tools/desktop.py`, `tools/search.py` | migrado a tools |
| `skills.py` funny lines | `tools/desktop.py` | conservado |
| `listener.py::wake_score` y cía. | `crotolamo/voice/wake.py` | migrado íntegro |
| `voice_out.py` (ruta hardcodeada) | `crotolamo/voice/tts.py` | ruta desde config (Fase 5) |

## Seguridad

Sin ejecución de bash arbitrario. Las tools son funciones tipadas; las que tocan
archivos validan contra una **allowlist** de rutas (`[paths].allowed_roots`). No hay
blocklist de regex frágil como en C1.
