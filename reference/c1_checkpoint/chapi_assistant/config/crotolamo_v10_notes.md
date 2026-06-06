# Crotolamo Runtime v10 Project Indexer

Agrega indexador local de proyectos.

## Comandos

```text
proyectos
indexar
indexar huevonitis
indexar tletl
mapa
mapa huevonitis
mapa tletl
buscar archivo extractor en huevonitis
buscar archivo main en tletl
buscar texto class ExtractorApp en huevonitis
buscar texto GESTO en tletl
```

## Qué hace

- Lee rutas desde `data/memory/local_memory.json`
- Guarda índices en `data/project_index/`
- Evita carpetas pesadas como `.venv`, `.git`, `node_modules`, `__pycache__`
- No modifica proyectos. Solo lee.
