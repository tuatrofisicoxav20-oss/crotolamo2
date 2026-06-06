"""
Cliente seguro para Ollama usado por Crotolamo.
Evita que el runtime se quede congelado 120 segundos como si eso fuera una virtud técnica.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_TIMEOUT = 30


@dataclass
class OllamaResult:
    ok: bool
    text: str
    error: str | None = None
    model: str | None = None


def ask_ollama_safe(
    prompt: str,
    model: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    system: str | None = None,
) -> OllamaResult:
    model = model or DEFAULT_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DEFAULT_OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        text = parsed.get("response") or parsed.get("message", {}).get("content") or ""
        text = str(text).strip()
        if not text:
            return OllamaResult(False, "", "Ollama respondió vacío.", model)
        return OllamaResult(True, text, None, model)

    except KeyboardInterrupt:
        raise

    except (socket.timeout, TimeoutError) as e:
        return OllamaResult(
            False,
            "",
            f"Ollama tardó más de {timeout}s. El modelo puede estar cargando, saturado o pensando demasiado para su propio bien.",
            model,
        )

    except urllib.error.URLError as e:
        return OllamaResult(
            False,
            "",
            f"No pude conectar con Ollama: {e}. Revisa `ollama serve`.",
            model,
        )

    except Exception as e:
        return OllamaResult(False, "", f"Error llamando a Ollama: {type(e).__name__}: {e}", model)


def ask_ollama_text(prompt: str, model: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    result = ask_ollama_safe(prompt, model=model, timeout=timeout)
    if result.ok:
        return result.text
    return (
        "[OLLAMA_ERROR]\n"
        f"{result.error}\n\n"
        "Sugerencias rápidas:\n"
        "- Prueba `ollama ps`\n"
        "- Prueba `ollama run qwen2.5-coder:7b 'hola'`\n"
        "- Cierra apps pesadas si la RAM está alta\n"
        "- Usa un modelo más ligero si el de 7B se pone lento"
    )
