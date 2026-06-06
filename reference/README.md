# reference/ — fuentes de referencia (no es código de C2)

## c1_checkpoint/

Checkpoint completo de **Crotolamo 1** (`chapi_assistant/`), del zip
`crotolamo_checkpoint_20260514_224547.zip`. Es el snapshot más completo de C1: además
del core viejo (chapi_shell, skills, listener) incluye los módulos que C1 evolucionó
después (`crotolamo_runtime.py`, `task_planner.py`, `project_indexer.py`,
`project_inspector.py`, `command_safety.py`, `local_memory.py`, `plugin_registry.py`,
`patch_builder.py`, `safe_executor.py`, ...).

**Para qué está aquí:** fuente canónica de migración. NO es código de C2 y NADA del
runtime de C2 lo importa. Se conserva para poder migrar capacidades concretas a tools
limpias de C2 en fases futuras (p. ej. task_planner / project_indexer).

**Cómo está conectado:** registrado como el proyecto `crotolamo1` en
`config/crotolamo.toml`, así las tools de C2 (`read_project_file`, `analyze_project`,
`list_project_tree`, `find_in_project`) pueden leerlo y razonar sobre él. El modelo de
voz `.onnx` (63 MB) se excluyó (ya vive en `voices/`).
