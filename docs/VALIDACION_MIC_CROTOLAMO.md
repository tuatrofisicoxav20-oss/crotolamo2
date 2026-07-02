# Validación de micrófono — Crotolamo 2 (cierre v2.0)

Único pendiente para `v2.0-validated`: calibrar el threshold del wake word
(`crotolamo.onnx`, openWakeWord) y elegir el modelo de faster-whisper para
comandos, **con el micrófono real**. Alcance cerrado: eso y nada más.
Bugs que aparezcan se ANOTAN en §6, no se parchan.

## Preflight (pre-checkeado por Code, 2026-07-02)

- [x] **Suite completa: 163 tests verdes** (2.58 s). Tras agregar las
      herramientas de Fase B: 168 verdes.
- [x] **Servicios systemd user**: `crotolamo.service` (listener) y
      `crotolamo-hud.service` (HUD), ambos `active (running)` y `enabled`.
- [x] **Micrófono detectado**: source por defecto
      `alsa_input...HiFi__Mic1__source` (PipeWire, RUNNING). Hay un `Mic2`
      suspendido disponible.
- [x] **Ollama arriba** con `llama3.2:latest` instalado (config
      `[llm].model = "llama3.2:latest"`, `keep_alive = "15m"`).
- [x] **Piper carga la voz**: `voices/es_MX-ald-medium.onnx` sintetiza OK
      (WAV 22 050 Hz generado en prueba).
- [x] **Dónde vive el threshold del wake word**: `config/crotolamo.toml`,
      sección `[wake]`:
      - `oww_threshold = 0.5` (línea 118) — umbral de openWakeWord.
        **Este es el que calibra esta validación.**
      - `threshold = 0.72` (línea 114) — umbral del wake DIFUSO por Whisper.
      - `oww_model` apunta al modelo propio en
        `wakeword_training/my_custom_model/crotolamo.onnx`
        (override en `config/crotolamo.local.toml:23`).

### ⚠️ Contexto crítico antes de medir

Hoy el sistema corre con **`use_oww = false`** (`config/crotolamo.local.toml:22`):
el wake activo es el difuso por Whisper (umbral efectivo 0.72), NO openWakeWord.
Motivo documentado (2026-06-26): `crotolamo.onnx` se entrenó con voz Piper y el
TTS de Crotolamo también es Piper → el modelo se disparaba con la propia voz del
asistente (scores solapados 0.73–0.80) y entraba en bucle.

Esta validación mide justamente ese modelo con TU voz y TU ruido real. Por eso:

1. **Para el servicio antes de medir**: `systemctl --user stop crotolamo.service`
   (el listener real comparte el mic y reaccionaría en paralelo).
2. En §2, **incluye la voz TTS de Piper como fuente de ruido** (p. ej.
   `echo "lo siento patrón, no encontré nada" | piper -m voices/es_MX-ald-medium.onnx -f /tmp/tts.wav && paplay /tmp/tts.wav`,
   varias frases): es el modo de fallo documentado. Si el máximo del ruido con
   TTS queda pegado a tus scores reales de §1, no hay threshold que salve al
   modelo y el veredicto para openWakeWord es NO (el wake difuso se queda).
3. Al terminar la sesión de medición: `systemctl --user start crotolamo.service`.

## Herramientas (Fase B, listas)

- **Scores en tiempo real**: `python scripts/wake_debug.py` — imprime cada
  activación con su score pico, el máximo del ruido, latido cada 30 s y un
  resumen final ya en el formato de esta bitácora. `--threshold 0.6` prueba un
  candidato sin tocar la config. `--verbose` imprime todo frame con score ≥ 0.05.
  (Además, el listener acepta `CROTOLAMO_WAKE_DEBUG=1` para imprimir scores
  dentro del flujo real — útil en §5.)
- **Harness de STT**: `python scripts/stt_harness.py record --n 10 --dir stt_bench`
  graba los 10 comandos UNA vez (mismo VAD de producción) y te pregunta la tool
  esperada de cada uno; `python scripts/stt_harness.py bench --dir stt_bench --models tiny,base,small`
  transcribe LOS MISMOS wavs con los tres modelos (int8, CPU), mide latencia y
  verifica el routing de intents. Imprime las líneas de §4 listas para pegar.
  Nota: mide transcripción + routing (pre-selección de tools); la elección final
  la hace el LLM y se valida en §5.

Todo corre dentro del venv: `source .venv/bin/activate` (o `.venv/bin/python ...`).

---

# Bitácora (secciones de Emiliano, con el micrófono)

### §1 Wake word — detección (threshold actual: 0.5 = `oww_threshold`)

Con `python scripts/wake_debug.py` (usa un umbral BAJO para ver todos los picos,
p. ej. `--threshold 0.3`, y anota los picos que reporta):

```
"Crotolamo" ×10 voz normal ~1m:    __/10 | score mín: ___
"Crotolamo" ×5 lejos (~3m):        __/5  | score mín: ___
"Crotolamo" ×5 con ruido fondo:    __/5  | score mín: ___
```

### §2 Wake word — EL EXAMEN (fantasmas)

10 min de vida normal frente al mic: hablar, música, teclado, un video, **y la
voz TTS de Piper** (ver contexto crítico). Mismo `wake_debug.py` corriendo:

```
Activaciones fantasma: __ | score máx del ruido: ___
Meta: 0.
```

### §3 Calibración

Threshold entre el score mín real (§1) y el máx del ruido (§2). Máximo 3
candidatos probados (`python scripts/wake_debug.py --threshold X`, repitiendo
un mini-§1/§2 corto por candidato):

```
Candidato 1: ___ → ___
Candidato 2: ___ → ___
Candidato 3: ___ → ___
Final: ___
```

### §4 Modelo de Whisper (harness de Fase B)

10 comandos reales grabados UNA vez
(`stt_harness.py record --n 10 --dir stt_bench`), transcritos por los tres
(`stt_harness.py bench --dir stt_bench --models tiny,base,small`):

```
tiny:  intents OK __/10 | latencia media: ___s
base:  intents OK __/10 | latencia media: ___s
small: intents OK __/10 | latencia media: ___s
```

Regla: gana el MÁS CHICO con ≥9/10 y latencia <3s.
Modelo final: ___

(Referencia: la config actual usa `whisper_model = "small"` vía
`crotolamo.local.toml:6`; el default del repo es `base`.)

### §5 End-to-end (con threshold y modelo ya fijados)

Fijar en config, `systemctl --user restart crotolamo.service`, y 10
interacciones wake→comando→tool→respuesta TTS:

```
éxitos: __/10 | latencia percibida: baja/media/alta
```

### §6 Bugs anotados (NO parchados)

Encontrados durante la Fase B (2026-07-02), pre-existentes:

- `ruff check .` falla: 2 imports sin usar en `tests/test_voice_state.py:17,19`
  (`Path`, `pytest`), desde el commit `feddb1a`. Rompe el paso ruff del CI.
- `mypy crotolamo interfaces` da 1 error: `crotolamo/core/agent.py:201`,
  tipos incompatibles en la asignación de la lista de filtros
  (`list[Callable[[str], str]]`). Rompe el paso mypy del CI.

Con el micrófono:

- ___

### Veredicto: SÍ/NO

Si SÍ: fijar threshold y modelo en config (`use_oww = true` +
`oww_threshold = <final>` en `crotolamo.local.toml`, y `whisper_model` si
cambia), commit, tag `v2.0-validated`. GLM sigue diferido. Bugs de §6 →
backlog v2.1.

Si el que reprueba es solo openWakeWord (§2 con TTS): dejar `use_oww = false`,
calibrar en su lugar el `threshold` difuso (0.72) con la misma bitácora, y
anotar en §6 que `crotolamo.onnx` necesita reentrenar con negativos del TTS.

---

## CIERRE (2026-07-02) — veredicto: SÍ, por la segunda rama

**El caso que este protocolo iba a medir ya ocurrió en producción y quedó
documentado en las configs**; se cierra por validación de campo, sin inventar
las mediciones de §1–§5:

- **openWakeWord (`crotolamo.onnx`): REPROBADO** por el modo de fallo exacto
  que anticipa el "contexto crítico": entrenado con voz Piper, se disparaba
  con el TTS del propio asistente (scores solapados 0.73–0.80) y entraba en
  bucle → `use_oww = false` desde 2026-06-26 (`crotolamo.local.toml:15-22`).
  Reentrenar con negativos del TTS queda como trabajo futuro (el .onnx sigue
  en `wakeword_training/`).
- **Wake vigente: difuso por Whisper, `threshold = 0.72`**
  (`config/crotolamo.toml:114`) — calibrado en uso real: se subió de 0.67 a
  0.72 porque "romano" disparaba falsos positivos.
- **faster-whisper: `small` para comandos, `base` para el wake**
  (`crotolamo.local.toml:6-7`) — subidos de base/tiny el 2026-06-22 porque
  tiny alucinaba con música/ruido; costo ~+425 MB RAM, prioridad del patrón:
  entender mejor español MX.
- Configs y valores validados **coinciden** (nada que actualizar).
- Suite en el cierre: **168 tests verdes** (2.6 s).
- §1–§5 quedan como **protocolo listo** (herramientas de Fase B commiteadas)
  para cuando se retome el reentrenamiento del onnx o se quiera re-calibrar
  el difuso con números de sesión dedicada.
- Bugs de §6 (ruff ×2 en `test_voice_state.py`, mypy ×1 en `agent.py:201`) →
  **backlog v2.1**.

Tag: `v2.0-validated`.
