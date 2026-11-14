# Fase M3 — Loop de voz concurrente con barge-in (turn_id)

Reescritura race-free del loop de voz. La interrupción es correcta **por
construcción**: cada comando nuevo (wake o barge-in) incrementa un `turn_id`
monótono; cada frase nace con su `turn_id` y el `MouthThread` descarta las de
turnos abortados. Nada de Events de "interrupción" que se limpian (eso da races).

## Arquitectura

Cuatro threads coordinados por colas + `SharedState` (lock + turn_id):

```
EarThread  (micrófono)  → wake/VAD/barge-in → stt_queue / new_turn()
SttThread               → transcribe (Whisper) → command_queue
BrainThread             → agent.handle_turn → frases (Utterance) → tts_queue
MouthThread             → habla solo el turno vigente (descarta lo viejo)
```

- `crotolamo/voice/state.py` — `SharedState`, `Mode`, `turn_id`.
- `crotolamo/voice/threads.py` — los 4 threads (`EarThread`, `SttThread`,
  `BrainThread`, `MouthThread`) y la `Utterance` que viaja por `tts_queue`.
- `crotolamo/voice/loop.py` — `VoiceLoop` (orquestador: arma las colas y los 4
  threads y los apaga limpio).
- `crotolamo/voice/tts.py` — `TTS.stop()` + reproducción interrumpible por frase.

## Uso

```bash
python -m crotolamo listen                 # concurrente, half-duplex (default seguro)
python -m crotolamo listen --barge-in      # full-duplex (RECOMENDADO con auriculares)
python -m crotolamo listen --no-barge-in   # fuerza half-duplex
python -m crotolamo listen --simple        # fallback secuencial viejo (red de seguridad)
```

## Pruebas manuales en hardware (las hace el patrón)

La lógica de transición está cubierta por `tests/test_loop.py` con fakes (sin
micrófono). Lo que solo se prueba a mano:

1. **Básico (half-duplex):** `python -m crotolamo listen --no-barge-in`
   → di la wake word, luego un comando; debe responder hablando.
2. **Barge-in (con auriculares):** `python -m crotolamo listen --barge-in`
   → háblale encima a media respuesta; debe **callarse al instante** y atender lo
   nuevo. (Con altavoces y sin AEC puede auto-dispararse por eco; usa `--no-barge-in`.)
3. **Fallback:** `python -m crotolamo listen --simple` → el listener secuencial
   viejo sigue intacto.
4. **Apagado limpio:** Ctrl+C en cualquiera → termina solo, sin threads colgados.

## Mitigación de eco (M3.8)

Sin AEC real, el barge-in con altavoces se mitiga con 3 capas (configurables en
`[voice]`): ventana de gracia (`barge_in_grace_ms`), umbral elevado en SPEAKING
(`barge_in_threshold_margin`) y persistencia (`barge_in_min_chunks`).

El AEC real ya está en el repo, a nivel de PipeWire (módulo WebRTC):
`desktop/aec.sh on` instala `desktop/pipewire-aec.conf` y crea el source limpio
`crotolamo_aec_source`; Crotolamo lo abre con `[voice].input_device` sin cambiar el
micrófono por defecto. La salida sí cuenta: `crotolamo_aec_sink` debe ser el sink por
defecto (es la referencia que se resta). El doctor lo exige (`[doctor].aec_source` /
`aec_sink`). Ver `docs/MEJORAS_VOZ.md` §1.1.

## Pendiente (futuro)

- Validar el barge-in con altavoces sobre el AEC de PipeWire (`--barge-in` con
  `desktop/aec.sh on`); hasta entonces, auriculares o `--no-barge-in`.
- Cancelar el LLM en vuelo (hoy la inferencia vieja termina en background y solo se
  descarta su salida por turno; el audio sí para al instante).
- Entrenar un modelo de wake word propio "crotolamo" (hoy se usa un puente de
  openWakeWord, p.ej. "hey_jarvis").
