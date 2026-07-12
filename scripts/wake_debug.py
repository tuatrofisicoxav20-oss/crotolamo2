"""Calibración del wake word openWakeWord con el micrófono REAL (validación de mic).

Imprime el score en tiempo real y acumula las estadísticas que pide la bitácora
docs/VALIDACION_MIC_CROTOLAMO.md:

- §1 (detección): di "crotolamo" y anota el score PICO de cada activación; el
  resumen final da el mínimo de esos picos.
- §2 (fantasmas): déjalo correr ~10 min de vida normal (hablar, música, teclado,
  el propio TTS de Piper); el resumen da el score máximo del ruido y cuántas
  activaciones fantasma hubo.
- §3 (candidatos): prueba umbrales con --threshold sin tocar la config.

Usa SIEMPRE el modelo openWakeWord ([wake].oww_model, hoy crotolamo.onnx),
aunque use_oww=false en la config: justamente se trata de medir ese modelo.

IMPORTANTE: para el servicio antes de medir, para que el listener real no
reaccione en paralelo:  systemctl --user stop crotolamo.service

  python scripts/wake_debug.py                  # umbral de la config (oww_threshold)
  python scripts/wake_debug.py --threshold 0.6  # probar un candidato
  python scripts/wake_debug.py --verbose        # además, cada frame con score>piso

Ctrl+C termina e imprime el resumen para la bitácora.
"""

from __future__ import annotations

import argparse
import sys
import time

from crotolamo.settings import get_settings
from crotolamo.voice.stt import VoiceUnavailable
from crotolamo.voice.wakeword import _FRAME, _SAMPLE_RATE, WakeWordDetector

# Piso de reporte: por debajo de esto el score es silencio y no aporta a la
# calibración (solo inundaría la terminal en la sesión de 10 min de §2).
_FLOOR = 0.05

# Frames seguidos por debajo del umbral que cierran una activación: agrupa la
# ráfaga de frames altos de UN "crotolamo" como UN evento (1 s a 80 ms/frame).
_REFRACTORY_FRAMES = 12


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run(threshold: float | None, verbose: bool) -> int:
    settings = get_settings()
    detector = WakeWordDetector.from_settings(settings)
    if threshold is not None:
        detector.threshold = threshold

    try:
        sd = __import__("sounddevice")
        np = __import__("numpy")
        detector._get_model()  # carga aquí, para fallar ANTES de abrir el mic
    except VoiceUnavailable as error:
        print("FALLO:", error)
        return 1

    print(f"Modelo:  {detector.model_name}")
    print(f"Umbral:  {detector.threshold:.2f}  (candidato --threshold)"
          if threshold is not None else
          f"Umbral:  {detector.threshold:.2f}  ([wake].oww_threshold)")
    print("Escuchando... di 'crotolamo' (§1) o haz vida normal (§2). Ctrl+C = resumen.")

    start = time.monotonic()
    frames = 0
    activations: list[float] = []  # score pico de cada activación
    in_activation = False
    below_run = 0
    peak = 0.0            # pico de la activación en curso
    noise_max = 0.0       # score máximo FUERA de activaciones
    last_heartbeat = start
    window_max = 0.0      # máximo desde el último heartbeat

    # Respeta [voice].input_device: así puedes medir DIRECTAMENTE sobre el source
    # con cancelación de eco ("crotolamo_aec_source") y comprobar que los
    # disparos fantasma del TTS desaparecen. Ver desktop/aec.sh.
    # OJO: nada de reimportar get_settings aquí. Un `import` dentro de la función
    # convierte el nombre en LOCAL para toda ella, y la línea `settings =
    # get_settings()` de arriba reventaba con UnboundLocalError. Se usa el import
    # de módulo.
    try:
        _device = get_settings().voice.get("input_device")
    except Exception:  # noqa: BLE001
        _device = None
    if _device:
        print(f"(micrófono: {_device})")

    try:
        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16",
                            device=_device) as stream:
            while True:
                block, _ = stream.read(_FRAME)
                audio = np.squeeze(np.asarray(block, dtype=np.int16))
                mx = detector.score(audio)
                frames += 1
                now = time.monotonic()
                clock = _fmt_clock(now - start)
                window_max = max(window_max, mx)

                if verbose and mx >= _FLOOR:
                    print(f"[{clock}] score={mx:.3f}", flush=True)

                if mx >= detector.threshold:
                    below_run = 0
                    peak = max(peak, mx)
                    if not in_activation:
                        in_activation = True
                        print(f"[{clock}] <<< ACTIVACIÓN score={mx:.3f}", flush=True)
                elif in_activation:
                    below_run += 1
                    if below_run >= _REFRACTORY_FRAMES:
                        activations.append(peak)
                        print(f"[{clock}]     fin de activación, pico={peak:.3f}",
                              flush=True)
                        in_activation = False
                        peak = 0.0
                else:
                    noise_max = max(noise_max, mx)
                    if mx >= _FLOOR and not verbose:
                        print(f"[{clock}] ruido score={mx:.3f}", flush=True)

                # Latido cada 30 s: en la sesión silenciosa de §2 confirma que
                # el script sigue vivo y cuánto ha subido el ruido.
                if now - last_heartbeat >= 30:
                    print(f"[{clock}] ... {frames} frames, max ventana={window_max:.3f}, "
                          f"max ruido acumulado={noise_max:.3f}", flush=True)
                    last_heartbeat = now
                    window_max = 0.0
    except KeyboardInterrupt:
        pass

    if in_activation:  # Ctrl+C a media activación
        activations.append(peak)

    dur = time.monotonic() - start
    print("\n--- RESUMEN (para la bitácora) ---")
    print(f"duración: {dur:.0f}s | frames: {frames} | umbral probado: {detector.threshold:.2f}")
    print(f"activaciones (score>=umbral): {len(activations)}")
    if activations:
        picos = ", ".join(f"{p:.3f}" for p in activations)
        print(f"score pico por activación: {picos}")
        print(f"score MÍNIMO de las activaciones (§1): {min(activations):.3f}")
    print(f"score MÁXIMO del ruido, fuera de activaciones (§2): {noise_max:.3f}")
    print("§3: el umbral final va ENTRE el mínimo real (§1) y el máximo del ruido (§2).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scores de openWakeWord en tiempo real para calibrar el umbral."
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="candidato de umbral a probar (default: config)")
    parser.add_argument("--verbose", action="store_true",
                        help=f"imprime cada frame con score>={_FLOOR}")
    args = parser.parse_args()
    return run(args.threshold, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
