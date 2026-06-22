"""Auditor de salud de Crotolamo 2.

Reporta ✅/❌ por cada check con una sugerencia de fix. Nunca lanza:
un entorno a medias debe poder diagnosticarse.

Uso:
    python -m crotolamo doctor
    python scripts/crotolamo_doctor.py
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    from crotolamo.settings import load_settings
except ModuleNotFoundError:  # ejecutado como script suelto
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crotolamo.settings import load_settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _ollama_tags(host: str, timeout: float = 5.0) -> list[str] | None:
    """Devuelve la lista de modelos instalados, o None si Ollama no responde."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def collect_checks() -> list[Check]:
    checks: list[Check] = []

    try:
        settings = load_settings()
    except Exception as error:  # noqa: BLE001 - el doctor nunca debe morir
        return [
            Check(
                "config",
                False,
                f"No pude cargar la config: {error}",
                "Revisa config/crotolamo.toml.",
            )
        ]

    checks.append(Check("config", True, "config/crotolamo.toml cargada", ""))

    # Usuario detectado (no hardcodeado).
    checks.append(
        Check("usuario", True, f"corriendo como '{settings.user}' (home={settings.home})", "")
    )

    # Rutas críticas.
    problems = settings.validate_critical()
    checks.append(
        Check(
            "rutas",
            not problems,
            "rutas críticas OK" if not problems else "; ".join(problems),
            "Crea las carpetas o ajusta [paths] en la config.",
        )
    )

    # Ollama responde.
    host = settings.llm.get("host", "http://localhost:11434")
    model = settings.llm.get("model", "qwen2.5-coder:7b")
    tags = _ollama_tags(host)
    if tags is None:
        checks.append(
            Check(
                "ollama",
                False,
                f"no responde en {host}",
                "Arranca el servicio: `ollama serve` (o systemctl start ollama).",
            )
        )
    else:
        checks.append(Check("ollama", True, f"responde en {host}", ""))
        has_model = any(t == model or t.split(":")[0] == model.split(":")[0] for t in tags)
        checks.append(
            Check(
                "modelo",
                has_model,
                f"'{model}' instalado" if has_model else f"falta '{model}' (hay: {', '.join(tags) or 'ninguno'})",
                f"Instálalo: `ollama pull {model}`.",
            )
        )

    # Deps del núcleo: solo stdlib. Confirmamos tomllib (3.11+).
    try:
        import tomllib  # noqa: F401

        checks.append(Check("python", True, f"Python {sys.version.split()[0]} con tomllib", ""))
    except ModuleNotFoundError:
        checks.append(
            Check(
                "python",
                False,
                "falta tomllib (Python < 3.11)",
                "Usa Python 3.11+ o instala el backport `tomli`.",
            )
        )

    # Voz (opcional): voz Piper en la ruta de config.
    voces = settings.paths.get("voces")
    piper_voice = settings.voice.get("piper_voice", "")
    if voces and piper_voice:
        onnx = voces / piper_voice
        checks.append(
            Check(
                "voz-piper",
                onnx.exists(),
                f"voz en {onnx}" if onnx.exists() else f"sin voz en {onnx} (opcional hasta Fase 5)",
                "Coloca el .onnx en [paths].voces o instala con `pip install -e '.[voice]'`.",
            )
        )

    # Deps de voz (opcional, extra [voice]). La reproducción usa sounddevice
    # (no ffplay), así que portaudio es lo que de verdad hace falta.
    missing_voice = []
    for mod in ("faster_whisper", "sounddevice", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing_voice.append(mod)
    checks.append(
        Check(
            "voz-deps",
            not missing_voice,
            "faster-whisper/sounddevice/numpy importables" if not missing_voice
            else f"faltan: {', '.join(missing_voice)} (opcional hasta usar voz)",
            "Instala la voz: pip install -e '.[voice]'.",
        )
    )

    # openWakeWord (requerido para el modo concurrente de voz).
    try:
        import openwakeword  # noqa: F401

        oww_ok, oww_detail = True, "openwakeword importable (modo concurrente OK)"
    except ImportError:
        oww_ok, oww_detail = (
            False,
            "falta openwakeword (sin él, solo modo simple/difuso está disponible)",
        )
    checks.append(
        Check(
            "voz-oww",
            oww_ok,
            oww_detail,
            "pip install openwakeword  (o pip install -e '.[voice]')",
        )
    )

    # silero-vad (recomendado para el modo concurrente; hay fallback por energía si falta).
    try:
        import silero_vad  # noqa: F401

        svad_ok, svad_detail = True, "silero-vad importable (VAD neuronal activo)"
    except ImportError:
        svad_ok, svad_detail = (
            False,
            "falta silero-vad (el modo concurrente usará VAD por energía como fallback)",
        )
    checks.append(
        Check(
            "voz-silero",
            svad_ok,
            svad_detail,
            "pip install silero-vad torch  (opcional, mejora la detección de voz)",
        )
    )

    # Piper como librería persistente (la API real que usa tts.py).
    try:
        from piper import PiperVoice  # noqa: F401

        piper_ok, piper_detail = True, "piper (PiperVoice) importable"
    except ImportError:
        piper_ok, piper_detail = False, "falta piper (from piper import PiperVoice)"
    checks.append(
        Check("voz-piper-lib", piper_ok, piper_detail,
              "Instala piper: pip install -e '.[voice]'.")
    )

    # portaudio: sin él, sounddevice no reproduce ni graba. Lo sondeamos vía sounddevice.
    if "sounddevice" not in missing_voice:
        try:
            import sounddevice as _sd

            _sd.query_devices()
            portaudio_ok, portaudio_detail = True, "portaudio OK (sounddevice ve dispositivos)"
        except Exception as error:  # noqa: BLE001
            portaudio_ok = False
            portaudio_detail = f"sounddevice no consultó dispositivos: {error}"
        checks.append(
            Check("portaudio", portaudio_ok, portaudio_detail,
                  "Instala portaudio: sudo dnf install -y portaudio.")
        )

    # Entorno Wayland / ydotool (opcional, para control de ventanas futuro).
    has_ydotool = shutil.which("ydotool") is not None
    checks.append(
        Check(
            "ydotool",
            has_ydotool,
            "ydotool presente" if has_ydotool else "sin ydotool (opcional)",
            "Instala con `sudo dnf install ydotool` si quieres control de ventanas.",
        )
    )

    # Navegador para las tools de desktop.
    browser = shutil.which("xdg-open") or shutil.which("flatpak")
    checks.append(
        Check(
            "navegador",
            browser is not None,
            f"lanzador disponible ({browser})" if browser else "sin xdg-open/flatpak",
            "Instala xdg-utils: `sudo dnf install xdg-utils`.",
        )
    )

    return checks


def run_doctor() -> int:
    from crotolamo.logging_setup import get_logger, setup_logging

    setup_logging()
    log = get_logger("doctor")

    checks = collect_checks()
    log.info("doctor: %d checks evaluados", len(checks))
    for c in checks:
        log.debug("check %-14s ok=%s :: %s", c.name, c.ok, c.detail)

    print("Doctor de Crotolamo 2\n" + "=" * 40)

    # Distinguimos checks obligatorios de opcionales para el código de salida.
    # voz-oww: requerido para el modo concurrente pero el modo simple funciona sin él.
    # voz-silero: opcional — hay fallback por energía en _SileroVad (loop.py).
    optional = {"voz-piper", "voz-deps", "voz-piper-lib", "portaudio", "ydotool",
                "voz-oww", "voz-silero"}
    hard_fail = False

    for c in checks:
        mark = "✅" if c.ok else "❌"
        print(f"{mark} {c.name:12s} {c.detail}")
        if not c.ok:
            if c.fix:
                print(f"   ↳ fix: {c.fix}")
            if c.name not in optional:
                hard_fail = True

    print("=" * 40)
    if hard_fail:
        print("Hay fallos que arreglar, patrón. Mira los ❌ de arriba.")
        return 1
    print("Todo en orden, patrón. (Los ❌ opcionales no bloquean.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_doctor())
