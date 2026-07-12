# Mejoras de detección de voz — investigación (2026-07-09)

Investigación de repositorios y técnicas para mejorar los cuatro dolores reportados:
no responde al nombre / se autodispara, no entiende, va lento, se corta a media frase.

**Restricción dura que filtra todo**: Intel i5-12450H, **CPU-only** (sin GPU NVIDIA),
iGPU Intel UHD Alder Lake GT1 (débil), **~4.8 GB de RAM libre** (Ollama ya consume
~2.2 GB residentes, el sistema el resto). Español mexicano.

---

## 0. El hallazgo que enmarca todo: no hay línea base

El protocolo `docs/VALIDACION_MIC_CROTOLAMO.md` §1–§5 **nunca se ejecutó con el
micrófono real**; se cerró "por validación de campo" sin números. Las herramientas
para medir ya están construidas y sin usar:

    .venv/bin/python scripts/wake_debug.py --threshold 0.3     # scores del wake
    .venv/bin/python scripts/stt_harness.py record --n 10 --dir stt_bench
    .venv/bin/python scripts/stt_harness.py bench --dir stt_bench --models base,small

Sin esto, cualquier cambio es a ciegas y no se puede saber si "mejoró por mucho".
**Medir primero.** Es la única tarea que no tiene alternativa.

---

## 1. Multiplicador oculto: `use_oww = false` te tiene en modo simple

En `crotolamo.local.toml` está `use_oww = false`. En `interfaces/listener.py:194`
la condición es:

    if not simple and use_oww:      # -> loop concurrente

Con `use_oww=false` la condición **siempre es falsa**: corres *siempre* en modo
simple. Consecuencias, nunca explicitadas:

- El loop concurrente (threads Ear/Stt/Brain/Mouth, interrupción por `turn_id`) **está muerto**.
- **No hay barge-in** (no puedes interrumpirlo hablando).
- Whisper transcribe el ambiente en bucle solo para oír "crotolamo": caro en CPU y lento.

Arreglar el wake word no arregla solo el wake: **recupera la arquitectura concurrente**.
Es la mejora con mayor efecto palanca del informe.

Matiz honesto sobre el barge-in: poner `use_oww = true` te devuelve el loop concurrente,
pero **no te regala barge-in seguro**. El barge-in sigue requiriendo el flag `--barge-in`
(el default es half-duplex), y es justo el modo que escucha *mientras* el asistente habla
— o sea, el más expuesto al autodisparo. Depende de que el AEC **funcione** en esta
máquina, no solo de que esté instalado (verifiqué lo segundo, no lo primero).

---

## 2. Causa raíz del autodisparo (y por qué cambiar de librería NO lo arregla)

El modelo `crotolamo.onnx` se entrenó con voz **Piper**. El TTS de Crotolamo
**también es Piper**. El modelo aprendió "Piper diciendo crotolamo", no "Emiliano
diciendo crotolamo". Dos síntomas, una causa:

- Se autodispara cuando el asistente habla (scores solapados 0.73–0.80).
- Te cuesta que te responda a ti (`real_samples/` tiene solo **6** grabaciones reales).

Además, `gen_negatives.py` genera negativos fonéticamente cercanos ("cocodrilo",
"croqueta") y habla común ("buenos días"), pero **ninguna de las frases que
Crotolamo realmente dice al responder** ("Se me trabaron los cables, patrón").

> **Veredicto de la investigación**: Porcupine, sherpa-onnx KWS y EfficientWord-Net
> se autodispararían **exactamente igual**. El problema no es el detector: es
> **acústico** — el micrófono oye al altavoz. Cambiar de librería mueve el bug de sitio.

---

## 3. Plan priorizado

### Tier 0 — Medir (bloquea a todo lo demás)
Correr `wake_debug.py` y `stt_harness.py`. Llenar §1–§5 del doc de validación.

### Tier 1 — Arreglos de raíz

**1.1 AEC: cancelación de eco acústico** · impacto ALTO · esfuerzo BAJO (config, no código)

Verificado en esta máquina: `libspa-aec-webrtc.so` **ya está instalado** y PipeWire
corre. No hay que instalar nada.

Crea un *source* virtual que resta del micrófono la señal enviada al altavoz.
Esto elimina el eco del asistente **antes** de que llegue al wake word, y a
diferencia del gating **preserva el barge-in**. Se configura en
`~/.config/pipewire/pipewire.conf.d/` y `sounddevice` abre el source limpio por nombre.

Es la solución de raíz. Resuelve el autodisparo sin sacrificar nada.
Docs: <https://docs.pipewire.org/page_module_echo_cancel.html>

**1.2 Gating por estado (red de seguridad)** · impacto MEDIO · esfuerzo BAJO

Ya existe parcialmente: `EarThread` solo escucha el wake en `Mode.IDLE`, y hay un
cooldown `_post_speak_grace_s = 1.5` (`loop.py`). Es lo que hacen pipecat, LiveKit,
GLaDOS y Home Assistant. Con AEC en su sitio, esto pasa a ser el cinturón de
seguridad, no la solución principal. **El modo simple no lo tiene**.

**1.3 Reentrenar el wake con los negativos correctos** · impacto ALTO · esfuerzo MEDIO

El pipeline oficial de openWakeWord (dscripka, ~4k★, Apache-2.0) soporta ambas cosas:

- **Hard negatives**: cientos de clips de Piper diciendo **las frases reales de
  respuesta de Crotolamo**, metidos como negativos. La doc oficial dice que "los
  ejemplos previos de falsas activaciones son de lo más efectivo".
- **Custom verifier model** (`docs/custom_verifier_models.md`): regresión logística
  sobre las features, entrena con positivos = **tu voz real** y negativos = Piper.
  Filtra por locutor. Necesita <5 min de audio tuyo. Ataca directamente el
  "no me responde a mí" (hoy: 6 muestras reales).

### Tier 2 — Endpointing: el "se corta a media frase" **y** el "va lento"

Son el mismo problema, en tensión: cortar rápido trunca; cortar tarde se siente lento.
Hoy usas silencio fijo (`vad_silence_ms = 600`) + una heurística de texto
(`smart_endpoint`: "termina en 'de'/'para'").

**2.1 smart-turn v3.2 (pipecat-ai)** · impacto ALTO · esfuerzo MEDIO

<https://github.com/pipecat-ai/smart-turn> — **verificado en la fuente primaria**
(no solo en el anuncio del vendedor), 2026-07-09:

- ✅ **Español explícitamente soportado** (23 idiomas listados en el README).
- ✅ **BSD-2-Clause**, versión **v3.2**, CPU **int8 de 8 MB**.
- ⚠️ Latencia: el README dice *"as little as 10ms on **some** CPUs"*. Es un claim
  optimista, **no una garantía** para un i5-12450H. Medir antes de creer.
- ⚠️ **No hay paquete pip**: se integra copiando `model.py` e `inference.py` al
  proyecto y llamando a `predict_endpoint()`. Esfuerzo real mayor que "pip install".
- ⚠️ **Caveat que importa aquí**: *"no está diseñado para correr sobre segmentos de
  audio muy cortos"*, y acepta **hasta 8 segundos** de audio (16 kHz mono PCM).
  Tus comandos SON cortos y tu `max_command_s = 12.0` **excede el límite**.
  Hay que recortar la ventana y validar con comandos como "pausa la música".

Predice fin-de-turno desde el **audio crudo**, no del texto: distingue
"umm… \<pausa\>" (sigue hablando) de "he terminado". Reemplazaría tu heurística de
texto (`smart_endpoint`). LiveKit lo dice sin rodeos: *el silence-timeout fijo es el
mayor asesino de latencia* — 600-800 ms añadidos a **cada** respuesta.

> **Honestidad metodológica**: este repo apareció en dos de mis investigaciones, pero
> eso **no es corroboración independiente** — yo lo nombré en ambos prompts. Las
> cifras originales venían todas del anuncio de pipecat. Por eso se verificó contra
> el README del repo, que es de donde salen los ✅/⚠️ de arriba. El caveat de los
> segmentos cortos no lo mencionó ningún subagente.

**2.2 Histéresis de umbral** · impacto MEDIO · esfuerzo BAJO

Umbral de **entrada** alto (0.8, el actual) y de **salida** más bajo (~0.35), para no
cerrar la grabación en micro-pausas. Más `speech_pad_ms` (~300-400 ms) al final.
Tu buffer de pre-activación de 800 ms ya es el análogo al pad inicial: correcto.

> Nota: `silero-vad` ya está en **6.2.1** (la última). El consejo de "sube de v5 a v6"
> que circula **no aplica**: ya estás ahí.

### Tier 3 — STT: el "no entiende lo que digo"

**3.1 Quick wins sin migrar** · impacto MEDIO · esfuerzo BAJO

> **Corrección tras leer el código**: dos de los tres "quick wins" que recomendaba la
> investigación web **ya estaban implementados**. Los subagentes los propusieron sin
> haber leído el repo. Concretamente:
> - `condition_on_previous_text=False` → **ya está** (`stt.py:257`).
> - Normalización del audio → **ya está** (`_frames_to_wav`, `stt.py:133-135`).
> - Recorte por VAD antes de decodificar → **ya está**: `record_until_silence` solo
>   guarda los frames de voz, y `vad_filter=False` es deliberado (el doble VAD se
>   comía audio, documentado en `stt.py:250`).

Lo que sí queda como mejora real:

- La normalización actual es **por pico**, no por RMS. Un golpe en la mesa satura el
  pico y deja la voz baja. Una normalización RMS (~-3 dBFS) sería más robusta. *(No
  implementado: cambia la señal que ve Whisper y necesita medición antes.)*

> **NO bajes `beam_size` a 1.** Tu propia medición lo desmiente: beam5=1.73 s <
> beam2=2.04 s < beam1=2.52 s con CT2 int8. La recomendación genérica de internet
> es falsa en tu build.

> **NO bajes `beam_size` a 1.** Tu propia medición lo desmiente: beam5=1.73 s <
> beam2=2.04 s < beam1=2.52 s con CT2 int8. La recomendación genérica de internet
> es falsa en tu build.

**3.1-bis Lo que SÍ faltaba: histéresis del VAD** → ver §2.2. Implementado.

**3.2 El salto real: Parakeet TDT 0.6B v3** · impacto ALTO · esfuerzo MEDIO-ALTO

Multilingüe con **español**, CC-BY-4.0 (uso comercial OK). ONNX int8:
<https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx>
Runtimes: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (~10k★, Apache-2.0,
soporta *word boosting*) u [onnx-asr](https://github.com/istupakov/onnx-asr) (Python simple).

- RAM ~0.9–1.3 GB · latencia ~0.3–0.8 s para un comando de 3 s.
- **Más rápido Y más preciso** que `small`. Es el único candidato que ataca ambos dolores.
- Pero: es un *transducer*, **no tiene `initial_prompt`** estilo Whisper. Los nombres
  propios ("Tletl", "Huevonitis") se sesgan con *word boosting* de sherpa-onnx.
- **No** convertible con `ct2-transformers-converter` (arquitectura distinta).
- RAM: sería un **reemplazo** de Whisper small, no un añadido. Los 4.8 GB libres ya
  contemplan a Ollama corriendo, así que cabe; aun así es ~1.3 GB, más que `small`.

---

## 4. Descartado explícitamente (para que no pierdas el tiempo)

| Opción | Por qué NO |
|---|---|
| Porcupine, sherpa-onnx KWS, EfficientWord-Net | No arreglan el autodisparo: el problema es acústico |
| Snowboy, Mycroft Precise | Abandonados / archivados |
| microWakeWord | Solo ESP32, no corre en escritorio |
| distil-whisper, Moonshine | **Solo inglés** |
| whisper-large-v3-turbo | ~3–6 s por comando en esta CPU. Inservible |
| whisper.cpp + OpenVINO en la iGPU | Los 12x son en iGPUs fuertes (Arc/680M), no en UHD **GT1** |
| Vosk, wav2vec2-es, Citrinet | Calidad en español inferior a Whisper small |
| LiveKit turn-detector | ~500 MB RAM y depende del texto del STT. No cabe |
| STT incremental (LocalAgreement / WhisperLive) | Correr Whisper repetidamente en CPU cuesta más de lo que ahorra |
| py-webrtcvad, pyannote | Peor que silero / sobredimensionado |

---

## 5. Orden recomendado

1. **Medir** (Tier 0). Sin línea base no hay "mejoró por mucho".
2. **AEC de PipeWire** (1.1). Config, ya está instalado, resuelve la raíz, desbloquea `use_oww=true` → **loop concurrente + barge-in**.
3. **Reentrenar el wake** con hard negatives del TTS + verifier con tu voz (1.3).
4. **smart-turn v3** (2.1) + histéresis (2.2). Ataca "se corta" y "va lento" juntos.
5. **Quick wins de STT** (3.1). Baratos.
6. **Evaluar Parakeet v3** (3.2) solo si tras lo anterior el "no entiende" persiste.

**Expectativa honesta**: el grueso de la ganancia está en *arreglar y afinar lo que ya
tienes* (AEC + reentrenamiento + endpointing), no en instalar una librería nueva.
El único cambio de componente que se justifica por sí solo es smart-turn v3, y
eventualmente Parakeet.

---

## 6. Estado de implementación (2026-07-09)

**Invariante respetada: Crotolamo sigue comportándose exactamente igual que antes.**
Todo lo escrito está detrás de flags cuyo default reproduce el comportamiento actual.
Nada se activó a ciegas, porque verificar cualquiera de estos cambios exige un
micrófono y una voz humana — y sin línea base (§0) medir cinco variables movidas a la
vez no permite atribuir ninguna mejora a nada.

### Escrito y ACTIVO (inerte por defecto)

| Qué | Dónde | Default |
|---|---|---|
| **Histéresis del VAD** (`vad_neg_threshold`) | `voice/vad.py` (función pura), usada por `loop.py` (EarThread) y `stt.py` (`_record_silero`) | comentado → **sin histéresis**, idéntico a hoy |
| **`input_device`**: apunta solo a Crotolamo al mic del AEC | `loop.py`, `stt.py`, `wakeword.py`, `wake_debug.py` | comentado → **mic default del sistema** |
| Tests de histéresis | `tests/test_vad_hysteresis.py`, `tests/test_vad_helper.py` | **283 tests verdes**, ruff limpio |

Los tests **demuestran el bug**: sin histéresis, una micro-pausa (prob≈0.5) cuenta
como silencio y acaba truncando el comando; con `vad_neg_threshold = 0.35`, no. Y un
silencio real (prob≈0.05) sigue cortando.

La decisión vive en **una sola función pura** (`voice/vad.py`) que comparten las dos
rutas de captura. Importa porque los tests del `EarThread` solo cubren el loop
concurrente —que está **muerto** mientras `use_oww=false`—; testear la función pura
cubre también `_record_silero`, que es **la ruta que corres hoy** y que no se puede
ejercitar en CI (abre el micrófono real). Un `neg_threshold` mayor que el de entrada
invertiría la histéresis: se ignora en vez de empeorar el truncado en silencio.

### Escrito y STAGED (preparado, NO activado)

| Qué | Dónde | Estado |
|---|---|---|
| **Config AEC de PipeWire** | `desktop/pipewire-aec.conf` | no instalada |
| **Script activar/revertir/probar** | `desktop/aec.sh {on,off,status,test}` | probado: carga el AEC y revierte limpio |
| **168 hard negatives del TTS** (~9 min, 16 kHz mono) | `wakeword_training/tts_negatives/` | generados con la voz **exacta** del TTS |
| **Generador reproducible** | `wakeword_training/gen_tts_negatives.py` | 28 frases reales × 6 variantes acústicas |

Verificado en esta máquina: `libspa-aec-webrtc.so` existe, el módulo carga, crea
`crotolamo_aec_source`, y `aec.sh test` lo descarga dejando el sistema como estaba
(default source intacto, cero módulos residuales).

### Deliberadamente NO hecho

- **`use_oww` sigue en `false`.** Ponerlo en `true` a ciegas resucita el bucle de
  autodisparo que desactivaste a propósito. El AEC *habilita* encenderlo; no lo hace
  seguro sin que alguien escuche. **Este es el paso que requiere estar tú delante.**
- No se repuntó el micrófono al source del AEC (si estuviera mal configurado,
  Crotolamo se quedaría sordo sin nadie mirando).
- No se integró smart-turn (su README avisa: no diseñado para segmentos muy cortos,
  máximo 8 s, y tu `max_command_s` es 12).
- No se cambió la normalización a RMS (altera lo que ve Whisper; medir antes).

### La sesión contigo al micrófono (en este orden)

1. **Línea base**, antes de tocar nada:
   `.venv/bin/python scripts/stt_harness.py record --n 10 --dir stt_bench`
   `.venv/bin/python scripts/stt_harness.py bench --dir stt_bench --models base,small`
2. **Histéresis**: descomenta `vad_neg_threshold = 0.35`, repite el harness.
   ¿Dejó de truncarte sin sentirse más lento?
3. **AEC**: `./desktop/aec.sh on`.

   ⚠️ **Antes de poner `input_device`, confirma el nombre real del dispositivo.**
   `sounddevice` hace *substring match* contra los nombres de PortAudio, y los nodos
   de PipeWire no siempre afloran con el `node.name` que definimos. Si el string no
   resuelve, sounddevice falla o —peor— abre el micrófono equivocado en silencio:

       .venv/bin/python -c "import sounddevice; print(sounddevice.query_devices())"

   Busca el source del AEC en esa lista y usa **el nombre exacto que aparezca** (o su
   índice numérico) en `input_device`. Luego:

       [voice]
       input_device = "crotolamo_aec_source"   # o el nombre/índice real

   Comprueba que el eco muere: `.venv/bin/python scripts/wake_debug.py --threshold 0.3`
   mientras Crotolamo habla (o reproduces TTS de Piper). **Meta: 0 activaciones
   fantasma.** El script imprime qué micrófono abrió, para que no haya dudas.
4. **Solo si (3) da cero fantasmas**: reentrena el wake con
   `wakeword_training/tts_negatives/` como negativos + un *custom verifier* con tu
   voz real (hoy solo hay **6** muestras en `real_samples/`; graba ~5 min).
5. **Solo entonces**: `use_oww = true` → recuperas el loop concurrente. El barge-in
   (`--barge-in`) sigue siendo un paso aparte y depende de que el AEC aguante.
