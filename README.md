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
- Ollama con `qwen2.5-coder:7b` (tool-calling nativo)
- Voz (Fase 5, opcional): faster-whisper + Piper `es_MX-ald-medium`

## Uso

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
- [x] **Fase 5** — voz (VAD real, latencia) sobre el agente nuevo
- [x] **Fase 6** — extensiones: streaming token-a-token, TTS por frases, memoria fuzzy, búsqueda en proyectos, hotkeys

## Procedencia (qué se migró de Crotolamo 1)

Fuente: `~/Documentos/chapi_assistant` (la versión viva de C1).

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
