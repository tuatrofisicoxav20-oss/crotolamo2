import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

try:
    from core.voice_in import listen_once
    from core.voice_out import speak
    from core.skills import handle_direct_skill
    from core.chapi_shell import ask_ollama, run_command
except ImportError:
    from voice_in import listen_once
    from voice_out import speak
    from skills import handle_direct_skill
    from chapi_shell import ask_ollama, run_command


WAKE_TARGET = "crotolamo"

WAKE_VARIANTS = [
    "crotolamo",
    "crótolamo",
    "coto y amo",
    "coto lamo",
    "coto amo",
    "croto lamo",
    "croto el amo",
    "corto lamo",
    "corto la mano",
    "crotolo amo",
    "control amo",
    "contro lamo",
    "cuatro lamo",
    "proto lamo",
    "troto lamo",
    "cróto lamo",
]

CONFIRM_VARIANTS = [
    "sí",
    "si",
    "confirmo",
    "ejecuta",
    "dale",
    "hazlo",
    "va",
    "correcto",
]

CANCEL_VARIANTS = [
    "no",
    "cancela",
    "cancelar",
    "nel",
    "no lo hagas",
]


def normalize_for_wake(text: str) -> str:
    text = text.lower().strip()

    # Quitar acentos.
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

    # Quitar signos raros, dejar letras/números/espacios.
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact(text: str) -> str:
    return normalize_for_wake(text).replace(" ", "")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def wake_score(text: str) -> tuple[float, str]:
    """
    Devuelve qué tan parecido fue lo escuchado a 'crotolamo'.
    """
    norm = normalize_for_wake(text)
    comp = compact(norm)

    candidates = [compact(WAKE_TARGET)] + [compact(v) for v in WAKE_VARIANTS]

    best_score = 0.0
    best_variant = ""

    # Comparación completa
    for candidate in candidates:
        score = similarity(comp, candidate)
        if score > best_score:
            best_score = score
            best_variant = candidate

    # Comparación por ventanas, por si dice: "oye crotolamo abre youtube"
    for candidate in candidates:
        n = len(candidate)
        if n == 0 or len(comp) < n:
            continue

        for i in range(0, len(comp) - n + 1):
            window = comp[i:i + n]
            score = similarity(window, candidate)

            if score > best_score:
                best_score = score
                best_variant = candidate

    return best_score, best_variant


def is_wake_word(text: str) -> bool:
    norm = normalize_for_wake(text)

    # Primero exacto/super directo.
    for variant in WAKE_VARIANTS:
        if normalize_for_wake(variant) in norm:
            return True

    score, variant = wake_score(text)
    print(f"Wake score: {score:.2f} contra variante compacta '{variant}'", flush=True)

    # 0.67 es tolerante. Si se activa de más, lo subimos a 0.74.
    return score >= 0.67


def strip_wake_word(text: str) -> str:
    """
    Intenta quitar la palabra de activación para dejar solo la orden.
    """
    norm_original = text.strip()
    lower = normalize_for_wake(norm_original)

    for wake in WAKE_VARIANTS:
        wake_norm = normalize_for_wake(wake)

        if lower.startswith(wake_norm):
            # Recorte aproximado por palabras.
            wake_words_count = len(wake_norm.split())
            original_words = norm_original.split()
            return " ".join(original_words[wake_words_count:]).strip(" ,.:;-")

    # Si no puede recortar exacto, intenta quitar primeras 1-3 palabras
    # cuando el score del inicio suena a wake word.
    words = norm_original.split()

    for n in (3, 2, 1):
        if len(words) <= n:
            continue

        prefix = " ".join(words[:n])
        if is_wake_word(prefix):
            return " ".join(words[n:]).strip(" ,.:;-")

    return norm_original


def contains_any(text: str, variants: Iterable[str]) -> bool:
    lower = normalize_for_wake(text)
    return any(normalize_for_wake(v) in lower for v in variants)


def say(text: str) -> None:
    print(text, flush=True)

    try:
        speak(text)
    except Exception as error:
        print(f"[voz falló: {error}]", flush=True)


def ask_voice_confirmation() -> bool:
    say("Patrón, necesito confirmación. Di confirmo para ejecutar o cancela para cancelar.")

    try:
        answer = listen_once(seconds=4)
    except Exception as error:
        print(f"Error escuchando confirmación: {error}", flush=True)
        say("No pude escuchar la confirmación, patrón.")
        return False

    print(f"Confirmación escuchada: {answer}", flush=True)

    if contains_any(answer, CANCEL_VARIANTS):
        say("Cancelado, patrón.")
        return False

    if contains_any(answer, CONFIRM_VARIANTS):
        return True

    say("No escuché una confirmación clara, patrón. Cancelo por seguridad.")
    return False


def process_command(command: str) -> None:
    command = command.strip()

    if not command:
        say("No escuché una orden clara, patrón.")
        return

    print(f"Orden: {command}", flush=True)

    direct_result = handle_direct_skill(command)

    if direct_result is not None:
        print(direct_result, flush=True)
        say(direct_result)
        return

    try:
        plan = ask_ollama(command)
    except Exception as error:
        print(f"Error con Ollama: {error}", flush=True)
        say("Tuve un error pensando la orden, patrón.")
        return

    explanation = plan.get("explanation", "Tengo un plan, patrón.")
    commands = plan.get("commands", [])
    safe = plan.get("safe", False)

    print("\nPlan:", flush=True)
    print(explanation, flush=True)
    say(explanation)

    if not commands:
        say("No propuse comandos, patrón.")
        return

    if not safe:
        say("No ejecuto eso, patrón. Huele a desastre.")
        return

    print("\nComandos propuestos:", flush=True)
    for cmd in commands:
        print(f"  {cmd}", flush=True)

    if not ask_voice_confirmation():
        return

    for cmd in commands:
        run_command(cmd)

    say("Hecho, patrón.")


def main() -> None:
    say("Crotolamo listener activo, patrón.")

    while True:
        try:
            print("\nEsperando palabra de activación...", flush=True)
            heard = listen_once(seconds=5)
            print(f"Escuché ambiente: {heard}", flush=True)

            if not heard:
                continue

            if not is_wake_word(heard):
                continue

            possible_inline_command = strip_wake_word(heard)

            say("Te escucho, patrón.")

            # Si dijiste “Crotolamo abre YouTube”, intenta usar lo que vino después.
            # Si solo dijiste “Crotolamo”, escucha otra orden.
            if possible_inline_command and not is_wake_word(possible_inline_command):
                command = possible_inline_command
            else:
                command = listen_once(seconds=8)

            process_command(command)

            time.sleep(1)

        except KeyboardInterrupt:
            say("Crotolamo listener apagado, patrón.")
            break
        except Exception as error:
            print(f"Error en listener: {error}", flush=True)
            say("Tuve un error en el listener, patrón.")
            time.sleep(2)


if __name__ == "__main__":
    main()
