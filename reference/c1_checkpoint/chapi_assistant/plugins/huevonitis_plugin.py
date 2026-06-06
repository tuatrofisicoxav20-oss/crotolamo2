ACTIONS = [
    {
        "key": "huevonitis.audit",
        "title": "Auditoría Huevonitis",
        "description": "Compila archivos principales de Huevonitis sin ejecutar la app completa.",
        "modes": ["huevonitis"],
        "aliases": ["revisar huevonitis", "auditar huevonitis", "diagnostico huevonitis", "diagnóstico huevonitis"],
        "commands": [
            "cd {active_path} && pwd",
            "cd {active_path} && python -m py_compile main.py ui/main_window.py editor/canvas_editor.py core/features.py 2>&1 || true",
            "cd {active_path} && git status --short 2>/dev/null || true",
        ],
        "risk": "safe",
    },
    {
        "key": "huevonitis.extractor",
        "title": "Revisar extractor",
        "description": "Revisa sintaxis del extractor profesional de escritura.",
        "modes": ["huevonitis"],
        "aliases": ["revisar extractor", "compilar extractor", "extractor huevonitis"],
        "commands": ["cd {active_path} && python -m py_compile tools/handwriting_extractor_pro.py"],
        "risk": "safe",
    },
    {
        "key": "huevonitis.assets",
        "title": "Buscar bancos y assets",
        "description": "Ubica bancos de glifos, datasets e imágenes útiles sin modificar nada.",
        "modes": ["huevonitis"],
        "aliases": ["assets huevonitis", "bancos huevonitis", "glifos huevonitis"],
        "commands": [
            "cd {active_path} && find . -maxdepth 4 -type f \\( -iname '*glyph*' -o -iname '*bank*' -o -iname '*.jsonl' -o -iname '*.png' -o -iname '*.jpg' \\) | sort | head -160"
        ],
        "risk": "safe",
    },
]
