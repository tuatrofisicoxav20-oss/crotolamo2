#!/usr/bin/env python3
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ollama_client import ask_ollama_safe

prompt = "Responde solo con: OK CROTOLAMO"
timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 20

print(f"Probando Ollama con timeout={timeout}s...")
t0 = time.time()
result = ask_ollama_safe(prompt, timeout=timeout)
dt = time.time() - t0

print(f"Tiempo: {dt:.2f}s")
print(f"OK: {result.ok}")
if result.ok:
    print(result.text)
else:
    print(result.error)
