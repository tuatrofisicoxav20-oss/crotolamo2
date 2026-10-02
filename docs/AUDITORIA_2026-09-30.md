# Auditoría completa de Crotolamo 2 — 2026-09-30

Revisión línea por línea de TODO el código vivo (`crotolamo/`, `interfaces/`,
`desktop/`, `scripts/`, `tests/`, `config/`, CI) hecha en la rama
`claude/crotolamo-2-setup-9ckuci`, con cuatro revisores independientes
(núcleo, tools/safety/persistencia, voz+suite, desktop/scripts/CI) y repro
verificado de cada hallazgo marcado como tal. Los hallazgos ya corregidos en
esta misma rama se indican con **[ARREGLADO]**; el resto queda como backlog
priorizado. Los marcados **[DECISIÓN]** cambian comportamiento visible y
esperan al dueño.

## 0. Resumen ejecutivo

- **CI llevaba 11 runs en rojo desde junio** y pytest no corría en CI: mypy
  fijado a 3.11 rechazaba los stubs de numpy ≥ 2.5 (Python 3.12), y la última
  versión de ruff activaba reglas que el proyecto nunca eligió. **[ARREGLADO]**:
  matriz 3.11/3.12, reglas de ruff explícitas, caché de pip y torch CPU.
- **El servicio systemd nunca cargaba la API key de GLM**: arrancaba SIEMPRE
  con Ollama local, en silencio, aunque la config dijera "GLM como cerebro
  único". **[ARREGLADO]**: `EnvironmentFile=` + formato sin `export`.
- **El corral tenía tres agujeros reales**: un `\0` en una ruta reventaba el
  guard y dejaba el historial roto para GLM; `allowed_roots` como string abría
  TODO el disco; y `grep` sin `--` buscaba fuera del proyecto. **[ARREGLADO]**.
- **La confirmación por voz confirmaba por subcadena** ("necesito pensarlo"
  confirmaba una acción destructiva). **[ARREGLADO]**.
- El **loop concurrente** (openWakeWord, el default de fábrica) tiene tres
  problemas de diseño sin resolver (silencio inicial, hilo del oído mortal,
  stop() que se pierde) — el modo simple (Whisper difuso), que es el que usa
  el dueño, no los sufre. Ver §2.

## 1. Discrepancia importante: el stack descrito no está en GitHub

El dueño describe Groq (`openai/gpt-oss-120b`), Parakeet (`onnx_asr`), Kokoro
(`kokoro-v1.0.onnx`, voz `em_alex`), un adaptador Soloist (`tools/_soloist.py`,
WebSocket 127.0.0.1:9090) y un gate de hablante. **Nada de eso existe en el
repositorio publicado** (verificado con `grep -ri` sobre el árbol; solo hay
menciones en `docs/` y `reference/`). El repo tiene GLM/Z.ai + Ollama, Whisper,
Piper y playerctl. Si ese código existe en la laptop, está SIN commitear allí:
`git status` en la laptop lo dirá. Todo lo hecho en esta rama es agnóstico al
motor (sanitizador en `TTS.speak`, sondas de música inyectables, doctor guiado
por config) para que ese stack encaje sin reescribir nada.

## 2. Hallazgos pendientes, por prioridad

### ALTA
| # | Dónde | Qué pasa | Sugerencia |
|---|-------|----------|------------|
| V1 | `crotolamo/voice/threads.py:306` | En LISTENING se cuenta silencio desde el primer chunk: una pausa tras "crotolamo" (>600 ms) cierra un comando VACÍO y la orden se pierde. Solo loop concurrente; en el simple lo cubre `start_timeout_s`. | Estado "esperando voz" con su propio timeout (~4 s) y no contar silencio hasta el primer chunk con voz, como hace `stt._record_silero`. |
| V2 | `crotolamo/voice/threads.py:274` | Una excepción en `mic.read()` o `wake_fn()` mata el hilo Ear; el loop sigue vivo, sordo y sin aviso. openWakeWord se carga perezoso DENTRO del Ear (descarga sin timeout). | try/except por vuelta con log; tras N fallos seguidos, `shutdown.set()` (systemd reinicia); precargar `wake_detector._get_model()` antes de `start()`. |
| V4 | `crotolamo/voice/tts.py` (`_speak_streaming` limpia `_stop_flag` al entrar) | `tts.stop()` que llega justo antes de que el Mouth entre en `_speak_streaming` se pierde: la frase del turno abortado suena entera. `sd.stop()` no afecta a `OutputStream`. | Que el Mouth limpie el flag solo al tomar una `Utterance` vigente (o un contador de generación de stop comparado con `turn_id`). No lo toca el sanitizador (que es texto previo al audio). |

### MEDIA
| # | Dónde | Qué pasa | Sugerencia |
|---|-------|----------|------------|
| V5 | `threads.py:58` | Mouth: `is_current` → `set_mode(SPEAKING)` → `speak()` no es atómico frente al barge-in (reproducido con publisher lento). | `SharedState.set_mode_if_current(turn, mode) -> bool`. |
| V6 | `threads.py:264` + `stt.py:172` | Cada falso wake manda ~600 ms de silencio a Whisper y `_frames_to_audio` normaliza el ruido a 0.9 (alucinaciones, ~1 s de CPU). | No encolar sin ningún chunk con voz; normalizar solo si `peak > 0.02`. |
| V7 | `stt.py:215` | `_record_energy` devuelve todo el audio aunque no hubo voz (vuelve la "ventana sorda"); la calibración usa los 10 primeros chunks (si hablas al instante, el umbral queda 3× tu voz). | Devolver `[]` si nunca hubo voz; recortar al pre-buffer; memorizar el fallo de Silero. |
| V8 | `stt.py:112`, `tts.py:103` | Cargas perezosas sin lock con `WarmVoice` concurrente → doble carga (RAM/segundos). SOSPECHA en frecuencia. | `threading.Lock` alrededor de las tres cargas. |
| C1 | `core/llm.py:177`, `core/glm.py:206` | `resp.read()` puede lanzar `http.client.IncompleteRead` (HTTPException, no OSError): sale una excepción cruda de `handle_turn` y no abre el breaker de `FallbackLLM`. | Añadir `http.client.HTTPException` a los `except` de `chat()`/`chat_stream()`. |
| C2 | `core/fallback.py:135` + listener | Si GLM falla A MEDIA FRASE en streaming, el error se devuelve como texto pero el `StreamSpeaker` ya habló: el patrón oye una frase truncada y silencio. | Si `reply` es un error y ya se habló, hablar igual un aviso corto. |
| D5 | `desktop/panel/window.py:342`, `interfaces/listener.py` | El switch "Escuchar por voz" se revierte solo (race 0.3 s vs tick 1.5 s en concurrente; permanente en `--simple`, que solo evalúa el flag tras un wake). | Panel: mantener el valor escrito hasta que el archivo lo refleje (~3 s); simple: `listen_for_wake(timeout_s=0.5)` y sondear entre iteraciones. |
| D6b | `interfaces/listener.py:_write_idle_hud` + `_graceful_exit` | El idle final puede perder contra una publicación concurrente (Mouth/Brain a medio `os.replace`) y `os._exit` deja `*.tmp` huérfanos en `~/.crotolamo`. | `shutdown.set()` + `tts.stop()` + join corto antes de escribir; barrer `*.tmp` al arrancar. |
| D7 | `desktop/install.sh:48` | No instala `crotolamo-hud.service`; los dos units llevan `%h/Documentos/crotolamo2` quemado (clonar en otra carpeta = servicio roto). | `sed "s#%h/Documentos/crotolamo2#$ROOT#g"` sobre los units + install/enable del HUD. |
| D3 | `pyproject.toml` (`openwakeword` sin versión) | En 3.12 pip resuelve openwakeword 0.4.0 (2023) porque tflite-runtime no tiene wheel cp312; en 3.11, 0.6.0. Por eso `wakeword.py` prueba 4 constructores. | `openwakeword>=0.6.0` y aceptar que la voz completa es 3.11. **[DECISIÓN]** |
| T10 | `safety/guard.py` + `tools/files.py:_resolve` | Rutas relativas se resuelven contra el CWD del proceso: la decisión del guard cambia entre shell y `.service`. | Anclar relativos a `settings.home` en `files._resolve`; en el guard, tratar `./`/`../` solo en args con nombre de ruta. |

### BAJA
| # | Dónde | Qué pasa | Sugerencia |
|---|-------|----------|------------|
| V9 | `listener.py:make_deny_with_notice` | Habla desde el hilo Brain con el modo en THINKING (el HUD dice "pensando"; con barge-in el aviso puede cortarse solo). | Encolar el aviso como `Utterance` del turno. |
| V10 | `wakeword.py:feed` | Log INFO por chunk (31/s) con score > 0.05. | Loguear solo al cruzar el umbral o con rate-limit. |
| V11 | `threads.py:237` | `_start_command` no resetea el VAD (arrastra el eco del TTS al comando nuevo). | `reset()` también en `_start_command`. |
| V13 | `voice/normalize.py:14,39` | Sustituciones por subcadena: "abre mis sitios web" → "abre mi escritorios web"; "qué es una carpeta…" → "crea una carpeta…" (y `make_dir` es `safe=True`). Hay un test que FIJA este comportamiento (herencia C1). | `\b` en los reemplazos y quitar la regla de "qué es una carpeta". **[DECISIÓN]** |
| V14 | `voice/interfaces.py:21` | `SpeechToText.transcribe(path)` no coincide con la firma real `(audio, hotwords=None)`. | Alinear el Protocol. |
| V16 | config ↔ código | `[wake].target` nunca se lee; `vad_silence_ms` con defaults 640/800 según módulo; `hotwords` hardcodeado como fallback en `loop.py`; el loop concurrente ignora `ack`, `stream_speak`, `smart_endpoint*`, `followup_s`, `vad_preactivation_ms` (ahora documentado en `[wake].use_oww`). | Un solo `DEFAULTS` de voz; leer `target` o borrar la clave. |
| V17 | `adapters.py:55`, `loop.py:stop` | Límite conocido B1: `stop()` cierra el `InputStream` mientras el Ear está en `read()`; join expira a 2 s. | Documentado; lo tapa `os._exit`. |
| C3 | `core/router.py:56` | Keywords por subcadena: "pon" matchea "responde", "tema" matchea "sistema". Solo mete tools de más. | Matchear por palabra (`\b`). |
| C4 | `tools/base.py:181` | `except TypeError` en `Registry.run` reporta "argumentos inválidos" también cuando el TypeError ocurre DENTRO de la tool. | Distinguir por la traza (o validar args antes de llamar). |
| C5 | `interfaces/listener.py:voice_confirm` | Transcribe la respuesta con el modelo `tiny` del wake; con el matcher por palabra, un mishear = "no" (lo seguro), pero puede exigir repetir. | Usar `stt` (base) para la confirmación. |
| C6 | `voice/tts.py:StreamSpeaker.finish` | Timeout de 120 s si el TTS se cuelga. | Bajar a ~30 s o atarlo al `turn_id`. |
| C7 | `core/hooks.py:datetime_prehook` | "[ahora: …]" en CADA mensaje de usuario queda en el historial (tokens de más; con compaction arrastra fechas). | Inyectar la fecha solo en el system prompt o en el primer turno. |
| C8 | `voice/state.py:_publish` | Publicar fuera del lock puede reordenar dos publicaciones casi simultáneas (HUD con un modo viejo hasta el siguiente cambio). | Número de secuencia en el dict y que el HUD ignore los menores. |
| T7 | `tools/projects.py:launch_project` | `safe=True` pero ejecuta scripts del disco (`launch*.sh`, en cualquier subcarpeta, incluida `reference/`) vía `bash -lc` sin confirmación. | `@tool(safe=False)` o limitar a launchers de primer nivel. **[DECISIÓN]** |
| T8 | `tools/projects.py:211` | `gtk-launch` busca el `.desktop` por ID en `applications/`, no en la ruta del proyecto; responde "Lancé" igual. | `gio launch <ruta>` o parsear `Exec=`. **[DECISIÓN]** |
| T16 | `tools/base.py:Tool.run` | Sin coerción de tipos (`limit=8.0`, `hours="6"`, `categoria=None`) → una iteración extra del LLM y se pierde el short-circuit. | Coerción ligera según `parameters.properties[].type`. **[DECISIÓN]** |
| T18/D11 | `crotolamo/__main__.py:34` + `pyproject` | `doctor` importa `scripts.*`, que no se empaqueta: solo funciona con `pip install -e`. | Mover el doctor a `crotolamo/doctor.py` y dejar el script como shim. **[DECISIÓN]** |
| D8 | `scripts/crotolamo_doctor.py` | Desfasado con "GLM como cerebro único": Ollama/modelo son fallos DUROS, no comprueba la API key ni la nube. | **En curso**: reescritura guiada por `[doctor]` (requeridos/opcionales). |
| D10 | CI | Tools sin pinear; mypy no cubre `desktop/` ni `scripts/`. | Pinear ruff/mypy; `mypy … desktop scripts` con `explicit_package_bases`. |
| D12 | `desktop/crotolamo-hud.service` | Puede reiniciar en bucle cada 3 s si `WAYLAND_DISPLAY` no está en `systemd --user` (SOSPECHA, depende del entorno). | `WantedBy=graphical-session.target` o `ConditionEnvironment=WAYLAND_DISPLAY`. |
| D13 | `desktop/panel/systemd.py:56`, `window.py:354-359` | `Popen` sin `wait` (zombis); `run()` con timeout 8 s en el hilo GTK (solo fallback); `Type=simple` → "cargando modelos" nunca se muestra; unit ausente falla en silencio. | `Gio.Subprocess`/`child_watch_add`, consultar `LoadState`, `Type=notify`. |
| D14 | `desktop/hud/app.py:18` | `--demo` es no-op si el HUD del servicio ya corre (instancia única D-Bus). | `Gio.ApplicationFlags.NON_UNIQUE` con `--demo`. |
| D15 | `desktop/hud/window.py:189` | Sin tope de líneas: una respuesta larga = bloque de media pantalla. | `set_lines(3)` + `set_ellipsize(END)`. |
| D16 | `scripts/stt_harness.py:117` | Transcribe sin `hotwords` (no mide el pipeline real). | `hotwords=get_settings().voice.get("hotwords")`. |
| D17 | `scripts/wake_debug.py:56`, `smoke_voz.py:39` | Prometen "FALLO: …" pero sueltan traceback (ModuleNotFoundError, PortAudioError, ImportError de piper). | Ampliar los `except` al abrir mic/modelos. |
| D18 | `desktop/aec.sh:19,73` | `grep -q` bajo `pipefail` (SIGPIPE) y rama muerta con `set -e` (SOSPECHA). | `[[ "$(pactl …)" == *"$AEC_SOURCE"* ]]`; `MOD=$(…) \|\| { echo FALLO; exit 1; }`. |
| D19 | `.gitignore`, `crotolamo.desktop` | Ignora `*.onnx` y `wakeword_training/` pero hay archivos versionados ahí; el `.desktop` raíz quema `/home/exitili`. | `git rm --cached` + excepción `!`; generar el `.desktop` con `$ROOT`. |
| D20 | `state.py:42`, `listener.py:40`, `desktop/common/ipc.py:20` | Tres definiciones de la ruta de `hud_state.json`/`control.json` sin test cruzado. | Una sola fuente + test de igualdad. |
| T6b | `tools/desktop.py:_open_local_browser` | Sigue hardcodeando Opera GX (ignora `[apps].opera`); el fallback a `xdg-open` ya funciona. | Contrato "navegador de URLs" en `[apps]`. **[DECISIÓN]** |

## 3. Arreglado en esta rama (con commit)

| Commit | Qué |
|--------|-----|
| `fix(shell)` | mypy limpio (`route_fn` anotado); `build_agent` loguea si cae a modo sin tools. |
| `fix(voz)` | Confirmación por voz por palabra completa (`wake.confirmation_from_answer`). |
| `feat(persona)` | Bloque de voz común a todas las personas. |
| `feat(voz)` | `limpiar_para_voz` en `TTS.speak` (punto único); segmentación tolerante a `**` y a listas numeradas. |
| `fix(ci,servicio)` | CI en 3.11/3.12 sin pin de mypy, caché, torch CPU; el servicio carga `~/.config/crotolamo/env`. |
| `fix(ci)` | Reglas de ruff explícitas. |
| `fix(hud)` | `new_turn()` limpia el texto visible. |
| `feat(mcp)` | M4: cliente MCP por stdio (`crotolamo/mcp/`), guard recursivo, `strict_args`, router con grupos dinámicos, `README_M4.md`. |
| `feat(voz)` | Umbral de wake con música + ducking (`crotolamo/voice/media_aware.py`). |
| `fix(voz,hud)` | Estado inicial publicado al arrancar; idle final con `enabled`; `available()` tolera portaudio ausente; `[wake].use_oww` documentado. |
| `fix(tools)` | 11 correcciones: `\0` en rutas, `grep -e/--`, sqlite borrado en caliente, fallback real a `xdg-open`, parser DDG con void tags, hechos más recientes al prompt, sin proxy en HA/Frigate, `\n` de playerctl, `pgrep --`, `PermissionError` en `iterdir`, TOCTOU de symlinks (nota: `delete_file`/`move_file` actúan ahora sobre el destino canónico de un enlace). |
| `fix(core,safety)` | Resultado de tool garantizado aunque el guard/confirm lancen; contenido no es ruta; `allowed_roots`/`confirm_roots` validados. |

## 4. Calidad de la suite (para el backlog)

- `tests/test_tts_stop.py` prueba `_play_interruptible`, que es código MUERTO (`speak()` usa `_speak_streaming`); la ruta real de corte no tiene tests (ahí vive V4).
- Aserciones negativas tras `time.sleep` (`test_loop.py:141/219/320/357/382`, `test_voice_threads.py:88-117`, `test_vad_hysteresis.py:64`): pasan vacíamente si el hilo aún no procesó. El repo ya sabe hacerlo bien (`test_m38_*` inyecta `time.monotonic`).
- Tautología en `test_speed.py:58` (`… or True`). Tests que no prueban lo que dicen: `test_wakeword_debug.py:56`, `test_voice.py:39`, `test_listener_confirm.py:30` (introspección de firma), `test_tools_coverage.py:44`.
- Dependencias de entorno: `test_voice_threads.py:179` requiere `torch` (sin `skipif`; es el único rojo en este entorno y pasa en CI); `numpy` no está en la extra `dev`; `test_system.py` usa `ps`/`/proc` reales; `test_safety.py:106` usa `Path.home()`.
- `get_settings()` se parchea en 6 archivos; con `crotolamo.local.toml` presente la suite corre contra la config local del autor.
- Sin cobertura: `STT._record_*`, el filtro `no_speech_prob`, `VoiceLoop._poll_control`, `WakeWordDetector._get_model`, el Ear con silencio inicial (el `_run_simple_loop` ya tiene tests desde esta rama).

## 5. Lo que está bien (conservar)

`_web.py` anti-SSRF (pin de IP, redirects validados, SNI, topes); `_hass.py` con token solo por env y sin redirects; guard por allowlist con tres zonas y `safe=False` que gana siempre; SQL parametrizado con WAL; `HTTPTransport` por hilo; `FallbackLLM` con breaker solo para fallos transitorios y sin reintentar si ya habló; `LiveStreamer` que retiene el preámbulo con tools; `turn_id` monótono + `is_current()`; publisher atómico que nunca propaga; histéresis del VAD pura y testeada; segmentación unificada streaming/no-streaming; medidas anti-alucinación en `transcribe`; y comentarios honestos sobre los límites conocidos (B1, Brain no cancelable).
