# Crotolamo Runtime v8

Agrega:

- `core/context_engine.py`
- `core/config_manager.py`
- `plugins/context_plugin.py`
- `tools/crotolamo_context_cli.py`
- `tools/crotolamo_doctor_v8_extra.py`

## Comandos nuevos

```text
contexto
config
usar modelo llama3.2:latest
usar modelo qwen2.5-coder:7b
timeout ollama 20
contexto on
contexto off
config ollama.timeout_seconds = 25
```

## Idea

La memoria v7 ya guardaba información.
La v8 la empaqueta en un prompt útil para Ollama.
