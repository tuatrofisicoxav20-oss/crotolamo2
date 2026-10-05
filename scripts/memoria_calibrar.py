"""Calibración de la memoria semántica (mem0): mide los scores REALES de búsqueda.

Por qué existe: mem0 no devuelve coseno. Con Chroma calcula 1/(1 + distancia L2)
de los vectores tal cual, así que la escala depende del embedder y no es
intuitiva (coseno 0.63 -> score 0.047). Este script guarda hechos en español en
una memoria TEMPORAL (no toca la tuya), pregunta parafraseando, y te dice qué
umbral conserva los aciertos. Si cambias [memoria].modelo_embeddings, córrelo.

Uso (en tu venv, con la extra [memoria] instalada):
    python -m crotolamo memoria calibrar            # solo embeddings (sin LLM)
    python -m crotolamo memoria calibrar --groq     # además prueba la EXTRACCIÓN real
                                                    # con el LLM de [memoria] (gasta llamadas)
Nunca imprime la API key.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

HECHOS = [
    "Me gusta el café de olla sin azúcar por las mañanas",
    "Mi perro se llama Tletl y le tiene miedo a los cohetes",
    "Los viernes juego básquet con mis primos en la cancha del parque",
    "Prefiero que me hables con humor y sin rodeos",
    "Mi mamá se llama Lupita y vive en Puebla",
    "Uso Fedora con Hyprland y detesto Windows",
]
# (pregunta parafraseada, índice del hecho correcto)
PREGUNTAS = [
    ("¿cómo tomo el café?", 0),
    ("¿qué bebo al despertar?", 0),
    ("¿cómo se llama mi mascota?", 1),
    ("¿a qué le teme mi perro?", 1),
    ("¿qué hago los fines de semana con la familia?", 2),
    ("¿qué deporte practico?", 2),
    ("¿cómo quiero que me traten?", 3),
    ("¿dónde vive mi madre?", 4),
    ("¿qué sistema operativo uso?", 5),
    ("¿qué opino de Microsoft?", 5),
]
TURNOS_EXTRACCION = [
    ("Acuérdate de que mi hermana Ana vive en Monterrey y es dentista.",
     "Ya quedó, patrón."),
    ("Oye, abre Spotify y pon algo de cumbia.", "Va que va, patrón, ya suena."),
    ("Mi contraseña del wifi es casa-2024-segura, guárdala.",
     "Eso no lo guardo, patrón."),
    ("El pendiente del proyecto Huevonitis es migrar la base de datos para el viernes.",
     "Anotado en la lista, patrón."),
]


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    probar_groq = "--groq" in argv

    from crotolamo.core.memoria import Mem0Backend, MemoriaConfig, MemoriaNoDisponible
    from crotolamo.settings import get_settings

    settings = get_settings()
    cfg = MemoriaConfig.from_settings(settings)
    tmp = Path(tempfile.mkdtemp(prefix="crotolamo_calib_"))
    cfg.ruta = tmp  # memoria TEMPORAL: la tuya no se toca
    cfg.enabled = True
    backend = Mem0Backend(cfg, settings)
    try:
        t0 = time.time()
        try:
            backend._memory()
        except MemoriaNoDisponible as error:
            print(f"No puedo calibrar: {error}")
            return 1
        print(f"mem0 listo en {time.time() - t0:.1f}s (embedder {cfg.modelo_embeddings})")

        for h in HECHOS:
            backend.guardar(h)
        print(f"guardados {len(HECHOS)} hechos (sin LLM)\n")

        print("%-46s %-7s %-7s %s" % ("pregunta", "score", "2º", "top-1 correcto"))
        aciertos = 0
        correctos: list[float] = []
        for q, idx in PREGUNTAS:
            rows = backend.buscar(q, top_k=6, umbral=0.0)
            top = rows[0] if rows else None
            ok = top is not None and top.texto == HECHOS[idx]
            aciertos += ok
            score_ok = next((r.score for r in rows if r.texto == HECHOS[idx]), 0.0)
            correctos.append(score_ok)
            segundo = max((r.score for r in rows if r.texto != HECHOS[idx]), default=0.0)
            print("%-46s %-7.3f %-7.3f %s" % (q, score_ok, segundo, "SI" if ok else "NO"))
        print(f"\naciertos top-1: {aciertos}/{len(PREGUNTAS)}")
        minimo = min(correctos) if correctos else 0.0
        print(f"score mínimo de un acierto: {minimo:.3f}")
        for thr in (0.1, 0.05, 0.04, 0.03, 0.02):
            vivos = sum(1 for s in correctos if s >= thr)
            print(f"  umbral {thr:.2f}: conserva {vivos}/{len(correctos)} aciertos")
        sugerido = max(0.0, round(minimo - 0.005, 3))
        print(f"\nsugerencia: [memoria].umbral = {sugerido:.3f}  (hoy: {cfg.umbral:g}); "
              "la calidad la da top_k chico, no el umbral")

        if probar_groq:
            print(f"\n== extracción real con el LLM ({cfg.llm_provider} / {cfg.llm_model}) ==")
            for usuario, respuesta in TURNOS_EXTRACCION:
                t0 = time.time()
                try:
                    nuevos = backend.extraer(usuario, respuesta)
                except Exception as error:  # noqa: BLE001
                    print(f"  FALLO ({time.time() - t0:.1f}s): {type(error).__name__}: {error}")
                    continue
                print(f"  [{time.time() - t0:.1f}s] «{usuario[:50]}» -> {nuevos or 'nada'}")
            print("\nesperado: hecho de la hermana SÍ; orden de Spotify NO; contraseña NO "
                  "(la rechaza parece_secreto antes de llegar aquí en producción); "
                  "pendiente del proyecto NO.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
