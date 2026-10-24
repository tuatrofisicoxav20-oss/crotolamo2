#!/usr/bin/env python3
"""Evaluador de la compuerta de hablante: ¿qué embedding separa TU voz del ruido?

Para cada modelo (TitaNet-small, WeSpeaker CAM++, ERes2NetV2, o los que pases)
calcula embeddings con sherpa-onnx de tres carpetas de WAV:

    --enrol  tus frases de enrolamiento (yo_enrol)
    --yo     tus frases de prueba        (yo_test)   -> deben PASAR
    --otros  todo lo que NO eres tú      (otros)     -> deben NO pasar
             (subcarpetas = categorías: tele/, cafe/, persona1/... se reportan aparte)

y mide, por modelo y por forma de puntuar:

    margen   = mínimo de tu voz - máximo de no-tu-voz   (> 0 = umbral limpio)
    margen/σ = el margen en desviaciones típicas (comparable entre puntuaciones)
    EER      = error donde falsos aceptados = falsos rechazados (más bajo, mejor)
    FRR@FA0  = % de TU voz rechazada si el umbral no deja pasar NADA de lo otro
    r(dur)   = correlación score~log(duración) DENTRO de cada clase: el sesgo de
               duración que mediste (r=0.69). Cerca de 0 = el score ya no depende
               de cuánto dura el segmento, sino de quién habla.

Formas de puntuar (todas con validación leave-one-out: un archivo nunca se
usa para normalizar su propia puntuación, así los números no salen inflados):

    crudo     coseno contra la media del enrolamiento (lo que suele hacer comparar.py)
    centrado  se resta la media de la cohorte (los "otros") antes del coseno.
              Quita la componente común que hace que TODO puntúe alto con
              referencias largas: es lo primero que hay que probar.
    asnorm    normalización adaptativa contra la cohorte (estándar en
              verificación de hablante con frases cortas)
    +dur      regresión logística sobre [score centrado, log(duración)]: si el
              sesgo de duración es lo único que estorba, aquí desaparece.

Opcional: --vad silero_vad.onnx recorta a SOLO voz antes del embedding (con
--vad-umbral alto = VAD agresivo) y --gtcrn gtcrn_simple.onnx denoisea antes,
para comparar con y sin denoiser.

Uso (fish):
    python gate_eval.py --enrol yo_enrol --yo yo_test --otros otros \\
        --modelo titanet=modelos/nemo_en_titanet_small.onnx \\
        --modelo campp=modelos/wespeaker_en_voxceleb_CAM++.onnx \\
        --modelo eres2netv2=modelos/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx \\
        --vad modelos/silero_vad.onnx
Requiere: sherpa-onnx y numpy (soundfile opcional, para FLAC/float WAV).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000


# --------------------------------------------------------------------------- audio
def leer_audio(path: Path) -> np.ndarray:
    """WAV/FLAC -> float32 mono 16 kHz."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = data.mean(axis=1)
    except ImportError:
        with wave.open(str(path), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise ValueError(f"{path}: WAV de {8 * width} bits no soportado sin soundfile")
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        try:
            from scipy.signal import resample_poly

            g = math.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        except ImportError:
            n = int(round(len(x) * SR / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


class Vad:
    def __init__(self, model: str, umbral: float, min_silencio: float) -> None:
        import sherpa_onnx as so

        cfg = so.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = umbral
        cfg.silero_vad.min_silence_duration = min_silencio
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = SR
        self._so, self._cfg = so, cfg
        self.window = cfg.silero_vad.window_size

    def solo_voz(self, x: np.ndarray) -> np.ndarray:
        vad = self._so.VoiceActivityDetector(self._cfg, buffer_size_in_seconds=600)
        partes: list[np.ndarray] = []

        def drenar() -> None:
            while not vad.empty():
                partes.append(np.asarray(vad.front.samples, dtype=np.float32))
                vad.pop()

        for i in range(0, len(x), self.window):
            vad.accept_waveform(x[i:i + self.window])
            drenar()
        vad.flush()
        drenar()
        return np.concatenate(partes) if partes else np.zeros(0, dtype=np.float32)


class Denoiser:
    def __init__(self, model: str) -> None:
        import sherpa_onnx as so

        cfg = so.OfflineSpeechDenoiserConfig(
            model=so.OfflineSpeechDenoiserModelConfig(
                gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=model)))
        self._d = so.OfflineSpeechDenoiser(cfg)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self._d(x, SR)
        y = np.asarray(out.samples, dtype=np.float32)
        if out.sample_rate != SR:
            n = int(round(len(y) * SR / out.sample_rate))
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
        return y


# --------------------------------------------------------------------------- datos
@dataclass
class Item:
    path: Path
    clase: str            # enrol | yo | otros
    categoria: str
    audio: np.ndarray = field(repr=False)
    dur: float = 0.0      # segundos de VOZ (tras VAD si lo hay)


def recolectar(carpeta: Path, clase: str) -> list[Path]:
    files = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        sys.exit(f"ERROR: no hay audio en {carpeta} ({clase})")
    return files


def categoria(path: Path, raiz: Path) -> str:
    rel = path.relative_to(raiz)
    return rel.parts[0] if len(rel.parts) > 1 else raiz.name


# --------------------------------------------------------------------------- métricas
def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def eer(t: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """(EER, umbral) barriendo todos los scores."""
    cand = np.unique(np.concatenate([t, n]))
    best = (1.0, float(cand[0]))
    gap = 2.0
    for thr in cand:
        frr = float(np.mean(t < thr))
        far = float(np.mean(n >= thr))
        if abs(frr - far) < gap:
            gap = abs(frr - far)
            best = ((frr + far) / 2, float(thr))
    return best


def auc(t: np.ndarray, n: np.ndarray) -> float:
    return float(np.mean(t[:, None] > n[None, :]) + 0.5 * np.mean(t[:, None] == n[None, :]))


def r_intra(scores: np.ndarray, logdur: np.ndarray, etiquetas: np.ndarray) -> float:
    """Correlación score~log(dur) quitando la media de cada clase (pooled within-class)."""
    s, d = scores.astype(float).copy(), logdur.astype(float).copy()
    for c in np.unique(etiquetas):
        m = etiquetas == c
        if m.sum() > 1:
            s[m] -= s[m].mean()
            d[m] -= d[m].mean()
    den = math.sqrt(float((s * s).sum() * (d * d).sum()))
    return float((s * d).sum() / den) if den > 0 else 0.0


def logreg_loo(X: np.ndarray, y: np.ndarray, l2pen: float = 1e-2) -> np.ndarray:
    """Scores leave-one-out (log-odds) de una regresión logística con numpy."""
    mu, sd = X.mean(0), X.std(0) + 1e-9

    def fit(Xa: np.ndarray, ya: np.ndarray) -> np.ndarray:
        Xa = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]
        w = np.zeros(Xa.shape[1])
        for _ in range(500):
            p = 1 / (1 + np.exp(-Xa @ w))
            g = Xa.T @ (p - ya) / len(ya) + l2pen * np.r_[0, w[1:]]
            h = (Xa * (p * (1 - p))[:, None]).T @ Xa / len(ya) + l2pen * np.eye(len(w))
            w -= np.linalg.solve(h, g)
        return w

    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        w = fit(X[m], y[m])
        out[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return out


# --------------------------------------------------------------------------- puntuar
def puntuar(emb: dict[str, np.ndarray], items: list[Item], topk: int) -> dict[str, np.ndarray]:
    """Devuelve {variante: scores} para los items de prueba (yo + otros), en orden."""
    E = l2(np.stack([emb[str(i.path)] for i in items if i.clase == "enrol"]))
    pruebas = [i for i in items if i.clase != "enrol"]
    T = l2(np.stack([emb[str(i.path)] for i in pruebas]))
    es_otro = np.array([i.clase == "otros" for i in pruebas])
    C_all = T[es_otro]                       # cohorte = los "otros"
    idx_cohorte = np.flatnonzero(es_otro)

    crudo = T @ l2(E.mean(0))
    centrado, asnorm = np.zeros(len(T)), np.zeros(len(T))
    for j in range(len(T)):
        # leave-one-out: la cohorte nunca incluye al propio archivo evaluado
        keep = idx_cohorte != j
        C = C_all[keep]
        mu = C.mean(0)
        ref = l2(l2(E - mu).mean(0))
        x = l2(T[j] - mu)
        Cc = l2(C - mu)
        s = float(x @ ref)
        centrado[j] = s
        k = max(1, min(topk, len(Cc)))
        se = np.sort(Cc @ ref)[-k:]          # cohorte vs enrolamiento
        st = np.sort(Cc @ x)[-k:]            # cohorte vs prueba
        asnorm[j] = 0.5 * ((s - se.mean()) / (se.std() + 1e-6) + (s - st.mean()) / (st.std() + 1e-6))
    dur = np.array([max(i.dur, 1e-3) for i in pruebas])
    y = (~es_otro).astype(float)
    con_dur = logreg_loo(np.c_[centrado, np.log(dur)], y)
    return {"crudo": crudo, "centrado": centrado, "asnorm": asnorm, "+dur": con_dur}


@dataclass
class Fila:
    modelo: str
    variante: str
    eer: float
    margen: float
    margen_sigma: float
    frr_fa0: float
    auc: float
    r_dur: float
    media_yo: float
    media_otros: float
    peores_otros: list[tuple[str, float]]
    peores_yo: list[tuple[str, float]]


def analizar(modelo: str, scores: dict[str, np.ndarray], pruebas: list[Item]) -> list[Fila]:
    es_yo = np.array([i.clase == "yo" for i in pruebas])
    logdur = np.log(np.array([max(i.dur, 1e-3) for i in pruebas]))
    filas = []
    for var, s in scores.items():
        t, n = s[es_yo], s[~es_yo]
        e, _ = eer(t, n)
        margen = float(t.min() - n.max())
        sig = math.sqrt((t.var() + n.var()) / 2) or 1e-9
        frr0 = float(np.mean(t <= n.max()))
        orden_n = np.argsort(-n)[:3]
        orden_t = np.argsort(t)[:3]
        otros = [i for i in pruebas if i.clase == "otros"]
        yo = [i for i in pruebas if i.clase == "yo"]
        filas.append(Fila(
            modelo, var, e, margen, margen / sig, frr0, auc(t, n),
            r_intra(s, logdur, es_yo.astype(int)), float(t.mean()), float(n.mean()),
            [(f"{otros[k].categoria}/{otros[k].path.name} ({otros[k].dur:.1f}s)", float(n[k]))
             for k in orden_n],
            [(f"{yo[k].path.name} ({yo[k].dur:.1f}s)", float(t[k])) for k in orden_t],
        ))
    return filas


def tabla_duracion(modelo: str, s: np.ndarray, pruebas: list[Item]) -> str:
    bins = [(0, 1), (1, 2), (2, 4), (4, 1e9)]
    nombres = ["<1s", "1-2s", "2-4s", ">4s"]
    out = [f"  {'dur':6s} {'yo (n, media)':>18s} {'otros (n, media)':>20s}"]
    for (lo, hi), nom in zip(bins, nombres):
        cel = []
        for clase in ("yo", "otros"):
            v = [s[k] for k, i in enumerate(pruebas) if i.clase == clase and lo <= i.dur < hi]
            cel.append(f"{len(v):3d}, {np.mean(v):+.3f}" if v else f"{0:3d},    —  ")
        out.append(f"  {nom:6s} {cel[0]:>18s} {cel[1]:>20s}")
    return "\n".join(out)


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrol", required=True, type=Path)
    ap.add_argument("--yo", required=True, type=Path)
    ap.add_argument("--otros", required=True, type=Path)
    ap.add_argument("--modelo", action="append", required=True, metavar="NOMBRE=RUTA.onnx")
    ap.add_argument("--vad", help="silero_vad.onnx: recortar a solo voz antes del embedding")
    ap.add_argument("--vad-umbral", type=float, default=0.5, help="0.5 normal; 0.7 = agresivo")
    ap.add_argument("--vad-silencio", type=float, default=0.25)
    ap.add_argument("--gtcrn", help="gtcrn_simple.onnx: denoisear antes (compara con y sin)")
    ap.add_argument("--min-voz", type=float, default=0.0,
                    help="descartar segmentos con menos voz que esto (s); se reportan aparte")
    ap.add_argument("--topk", type=int, default=50, help="tamaño de cohorte para AS-norm")
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--json", type=Path, help="guardar todos los resultados en JSON")
    args = ap.parse_args(argv)

    import sherpa_onnx as so

    vad = Vad(args.vad, args.vad_umbral, args.vad_silencio) if args.vad else None
    den = Denoiser(args.gtcrn) if args.gtcrn else None

    items: list[Item] = []
    descartados: list[str] = []
    for carpeta, clase in ((args.enrol, "enrol"), (args.yo, "yo"), (args.otros, "otros")):
        for p in recolectar(carpeta, clase):
            x = leer_audio(p)
            if den is not None:
                x = den(x)
            if vad is not None:
                x = vad.solo_voz(x)
            dur = len(x) / SR
            if dur < max(args.min_voz, 0.25):
                descartados.append(f"{clase}/{p.name} ({dur:.2f}s de voz)")
                continue
            items.append(Item(p, clase, categoria(p, carpeta), x, dur))

    n = {c: sum(1 for i in items if i.clase == c) for c in ("enrol", "yo", "otros")}
    print("== Datos ==")
    print(f"  enrol={n['enrol']}  yo={n['yo']}  otros={n['otros']}"
          f"   VAD={'sí (umbral %.2f)' % args.vad_umbral if vad else 'no'}"
          f"   GTCRN={'sí' if den else 'no'}")
    for c in ("enrol", "yo", "otros"):
        d = [i.dur for i in items if i.clase == c]
        if d:
            print(f"  duración de voz {c:5s}: mediana {np.median(d):.1f}s  min {min(d):.1f}s  max {max(d):.1f}s")
    if descartados:
        print(f"  descartados por poca voz ({len(descartados)}): " + ", ".join(descartados[:8])
              + (" ..." if len(descartados) > 8 else ""))
    if min(n.values()) == 0:
        sys.exit("ERROR: alguna clase quedó vacía tras el VAD / --min-voz")
    if n["otros"] < 10:
        print("  AVISO: con menos de ~10 archivos en 'otros' la cohorte es chica: centrado y "
              "asnorm serán ruidosos; graba más (tele, café, otras voces).")
    pruebas = [i for i in items if i.clase != "enrol"]

    todas: list[Fila] = []
    for spec in args.modelo:
        nombre, _, ruta = spec.partition("=")
        if not ruta:
            nombre, ruta = Path(spec).stem, spec
        ext = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(model=ruta, num_threads=args.hilos))
        emb: dict[str, np.ndarray] = {}
        for it in items:
            st = ext.create_stream()
            st.accept_waveform(SR, it.audio)
            st.input_finished()
            if not ext.is_ready(st):
                sys.exit(f"ERROR: {it.path} es demasiado corto para {nombre}")
            emb[str(it.path)] = np.asarray(ext.compute(st), dtype=np.float32)
        scores = puntuar(emb, items, args.topk)
        filas = analizar(nombre, scores, pruebas)
        todas.extend(filas)

        print(f"\n== {nombre} (dim {ext.dim}) ==")
        print(f"  {'variante':9s} {'EER':>6s} {'margen':>8s} {'margen/σ':>9s} {'FRR@FA0':>8s} "
              f"{'AUC':>6s} {'r(dur)':>7s} {'yo μ':>7s} {'otros μ':>8s}")
        for f in filas:
            print(f"  {f.variante:9s} {f.eer:6.1%} {f.margen:+8.3f} {f.margen_sigma:+9.2f} "
                  f"{f.frr_fa0:8.0%} {f.auc:6.3f} {f.r_dur:+7.2f} {f.media_yo:+7.3f} {f.media_otros:+8.3f}")
        cr = next(f for f in filas if f.variante == "crudo")
        print("  puntuación crudo por duración de voz:")
        print(tabla_duracion(nombre, scores["crudo"], pruebas))
        print("  lo que más se parece a ti sin serlo (crudo): "
              + "; ".join(f"{k} {v:.3f}" for k, v in cr.peores_otros))
        print("  tu voz que peor puntúa (crudo):            "
              + "; ".join(f"{k} {v:.3f}" for k, v in cr.peores_yo))

    print("\n== Resumen (ordenado por EER) ==")
    for f in sorted(todas, key=lambda f: (f.eer, -f.margen_sigma)):
        limpio = "LIMPIO" if f.margen > 0 else ""
        print(f"  {f.modelo:12s} {f.variante:9s} EER {f.eer:6.1%}  margen/σ {f.margen_sigma:+6.2f}  "
              f"FRR@FA0 {f.frr_fa0:4.0%}  r(dur) {f.r_dur:+.2f}  {limpio}")
    limpios = [f for f in todas if f.margen > 0]
    print("\n== Veredicto ==")
    if limpios:
        g = max(limpios, key=lambda f: f.margen_sigma)
        print(f"  Separación limpia: {g.modelo} / {g.variante} (margen {g.margen:+.3f}, "
              f"{g.margen_sigma:+.2f}σ). Umbral = punto medio entre tu peor score y el mejor de 'otros'.")
        print("  Ojo: con pocos archivos un margen limpio puede ser suerte; confírmalo con más grabaciones.")
    else:
        g = min(todas, key=lambda f: f.eer)
        print(f"  Ningún modelo separa limpio. Mejor EER: {g.modelo} / {g.variante} = {g.eer:.1%} "
              f"(rechazaría {g.frr_fa0:.0%} de tu voz para no dejar pasar nada de lo otro).")
        print("  Lee la columna r(dur): si 'crudo' tiene r alto y 'centrado'/'+dur' lo bajan, el problema era")
        print("  la normalización, no el modelo. Si sigue alto, la duración de las frases manda.")
    if args.json:
        args.json.write_text(json.dumps([f.__dict__ for f in todas], ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\n  resultados en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
