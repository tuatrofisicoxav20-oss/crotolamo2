"""Harness para elegir el modelo de faster-whisper de los comandos de voz.

Porqué: elegir entre tiny/base/small "a oído" es adivinar. Este harness graba
una tanda de comandos reales UNA sola vez con el MISMO pipeline de producción
(STT.record_until_silence, mismo VAD, misma normalización) y luego transcribe
ESOS MISMOS wavs con cada modelo candidato, midiendo latencia y si el routing
de tools (strip_wake_word + select_tool_names) da con la tool esperada. Así la
comparación es justa: mismo audio, mismas condiciones, solo cambia el modelo.

  python scripts/stt_harness.py record --n 10 --dir stt_bench
  python scripts/stt_harness.py bench --dir stt_bench --models tiny,base,small

OJO: esto mide transcripción + routing de tools (pre-selección). La elección
final de tool la hace el LLM y se valida en el end-to-end (§5).
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

# Orden de tamaño para el veredicto: gana el MÁS chico que cumpla la regla.
_SIZE_ORDER = ["tiny", "base", "small", "medium", "large-v3"]

# Umbrales de la validación (§ regla): >=90% de intents OK y media < 3 s.
_MIN_INTENT_RATE = 0.9
_MAX_MEAN_LATENCY_S = 3.0


def _all_tool_names() -> list[str]:
    """Todas las tools de todos los grupos del router (para referencia/aviso)."""
    from crotolamo.core.router import GROUPS

    names: list[str] = []
    for group in GROUPS.values():
        for name in group["tools"]:
            if name not in names:
                names.append(name)
    return names


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / rate if rate else 0.0


# --- record ---------------------------------------------------------------

def cmd_record(args: argparse.Namespace) -> int:
    """Graba N comandos con el pipeline de producción y escribe manifest.json."""
    import shutil

    from crotolamo.settings import get_settings
    from crotolamo.voice.stt import STT, VoiceUnavailable

    out_dir = Path(args.dir)
    existing = sorted(out_dir.glob("*.wav")) if out_dir.is_dir() else []
    if existing and not args.force:
        print(f"FALLO: '{out_dir}' ya tiene {len(existing)} wav(s), patrón.")
        print("No sobrescribo grabaciones. Usa otro --dir o pasa --force.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = _all_tool_names()
    print("Tools válidas (para el campo 'esperada'):")
    print("  " + ", ".join(tools))
    print()

    stt = STT.from_settings(get_settings())
    manifest: list[dict[str, str | None]] = []
    width = max(2, len(str(args.n)))

    for i in range(1, args.n + 1):
        input(f"[{i}/{args.n}] Enter y di el comando (SOLO la orden, sin 'crotolamo')...")
        try:
            tmp_wav = stt.record_until_silence(max_seconds=args.max_seconds)
        except VoiceUnavailable as error:
            print("FALLO:", error)
            return 1

        name = f"cmd_{i:0{width}d}.wav"
        dest = out_dir / name
        shutil.move(str(tmp_wav), str(dest))
        print(f"  grabado {name} ({_wav_duration_s(dest):.1f}s)")

        expected = input("Tool esperada (enter = sin verificación): ").strip()
        if expected and expected not in tools:
            print(f"  OJO: '{expected}' no está en la lista de tools; la acepto igual.")
        manifest.append({"wav": name, "expected_tool": expected or None})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nListo, patrón: {args.n} wavs + manifest en {out_dir}/")
    print(f"Ahora: python scripts/stt_harness.py bench --dir {out_dir} --models tiny,base,small")
    return 0


# --- bench ----------------------------------------------------------------

def _bench_model(model_size: str, entries: list[dict], base_dir: Path) -> dict:
    """Transcribe todos los wavs con un modelo y devuelve las métricas."""
    import time

    from crotolamo.core.router import select_tool_names
    from crotolamo.voice.stt import STT
    from crotolamo.voice.wake import strip_wake_word

    # Idéntico a producción: language e initial_prompt por defecto.
    stt = STT(model_size=model_size, sample_rate=16000)

    # Warmup: carga el modelo y calienta int8 sin contaminar las mediciones.
    stt.transcribe(base_dir / entries[0]["wav"])

    rows: list[dict] = []
    for entry in entries:
        wav_path = base_dir / entry["wav"]
        t0 = time.perf_counter()
        texto = stt.transcribe(wav_path)
        latency = time.perf_counter() - t0

        cmd = strip_wake_word(texto)
        tools = select_tool_names(cmd)
        expected = entry.get("expected_tool")
        if expected:
            status = "OK" if expected in tools else "FALLO"
        else:
            status = "—"
        rows.append({"wav": entry["wav"], "latency": latency, "status": status, "text": texto})

    checked = [r for r in rows if r["status"] != "—"]
    ok = sum(1 for r in checked if r["status"] == "OK")
    mean_latency = sum(r["latency"] for r in rows) / len(rows)
    return {
        "model": model_size,
        "rows": rows,
        "ok": ok,
        "total": len(checked),
        "mean_latency": mean_latency,
    }


def _print_model_report(result: dict) -> None:
    print(f"\n=== modelo: {result['model']} ===")
    name_w = max(len(r["wav"]) for r in result["rows"])
    for r in result["rows"]:
        text = r["text"]
        if len(text) > 60:
            text = text[:57] + "..."
        print(f"  {r['wav']:<{name_w}}  {r['latency']:6.2f}s  {r['status']:<5}  \"{text}\"")
    print(
        f"  intents OK {result['ok']}/{result['total']} | "
        f"latencia media: {result['mean_latency']:.2f}s"
    )


def _passes_rule(result: dict) -> bool:
    if result["total"] == 0:
        return False  # sin expected_tool no hay forma de validar intents
    rate = result["ok"] / result["total"]
    return rate >= _MIN_INTENT_RATE and result["mean_latency"] < _MAX_MEAN_LATENCY_S


def cmd_bench(args: argparse.Namespace) -> int:
    """Transcribe los wavs del dir con cada modelo y saca veredicto."""
    from crotolamo.voice.stt import VoiceUnavailable

    base_dir = Path(args.dir)
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"FALLO: no encuentro {manifest_path}. ¿Ya corriste 'record', patrón?")
        return 1
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not entries:
        print("FALLO: el manifest está vacío.")
        return 1
    missing = [e["wav"] for e in entries if not (base_dir / e["wav"]).is_file()]
    if missing:
        print("FALLO: faltan wavs del manifest:", ", ".join(missing))
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("FALLO: lista de modelos vacía.")
        return 1

    results: list[dict] = []
    for model in models:
        print(f"\nCargando y midiendo '{model}' (warmup incluido, paciencia)...")
        try:
            result = _bench_model(model, entries, base_dir)
        except VoiceUnavailable as error:
            print("FALLO:", error)
            return 1
        _print_model_report(result)
        results.append(result)

    # Veredicto: el MÁS chico (tiny < base < small) que cumpla la regla.
    def size_key(res: dict) -> int:
        try:
            return _SIZE_ORDER.index(res["model"])
        except ValueError:
            return len(_SIZE_ORDER)  # modelos desconocidos, al final

    print("\n=== veredicto ===")
    winners = sorted((r for r in results if _passes_rule(r)), key=size_key)
    if winners:
        best = winners[0]
        print(
            f"Sugerido: '{best['model']}' — el más chico con >=90% intents OK "
            f"y media < {_MAX_MEAN_LATENCY_S:.0f}s."
        )
    else:
        print(
            "Ningún modelo cumple la regla (>=90% intents OK y latencia media "
            f"< {_MAX_MEAN_LATENCY_S:.0f}s). Revisa las transcripciones de arriba, patrón."
        )

    print("\nPara la bitácora:")
    name_w = max(len(r["model"]) for r in results)
    for r in results:
        print(
            f"{r['model'] + ':':<{name_w + 1}}  intents OK {r['ok']}/{r['total']} | "
            f"latencia media: {r['mean_latency']:.2f}s"
        )

    print(
        "\nNota: esto mide transcripción + routing de tools (pre-selección); la "
        "elección final de tool la hace el LLM y se valida en el end-to-end (§5)."
    )
    return 0


# --- CLI ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness para elegir el modelo de faster-whisper de comandos."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="graba N comandos con el pipeline real")
    p_record.add_argument("--n", type=int, default=10, help="cuántos comandos grabar")
    p_record.add_argument("--dir", default="stt_bench", help="directorio destino")
    p_record.add_argument("--max-seconds", type=float, default=12.0,
                          help="tope de grabación por comando")
    p_record.add_argument("--force", action="store_true",
                          help="permite grabar aunque el dir ya tenga wavs")
    p_record.set_defaults(func=cmd_record)

    p_bench = sub.add_parser("bench", help="transcribe los wavs con cada modelo")
    p_bench.add_argument("--dir", default="stt_bench", help="directorio con wavs + manifest")
    p_bench.add_argument("--models", default="tiny,base,small",
                         help="modelos a comparar, separados por coma")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
