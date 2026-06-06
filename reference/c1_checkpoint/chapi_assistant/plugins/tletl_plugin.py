ACTIONS = [
    {
        "key": "tletl.audit",
        "title": "Auditoría Tletl",
        "description": "Compila módulos típicos de Tletl y muestra estado Git.",
        "modes": ["tletl"],
        "aliases": ["revisar tletl", "auditar tletl", "diagnostico tletl", "diagnóstico tletl"],
        "commands": [
            "cd {active_path} && pwd",
            "cd {active_path} && python -m py_compile main.py train_gesture_lab.py ai_duel_lab.py audit_confusions.py analyze_gesture_bank.py 2>&1 || true",
            "cd {active_path} && git status --short 2>/dev/null || true",
        ],
        "risk": "safe",
    },
    {
        "key": "tletl.camera",
        "title": "Revisar cámaras",
        "description": "Lista dispositivos de cámara disponibles en Fedora.",
        "modes": ["tletl", "fedora"],
        "aliases": ["revisar camara", "revisar cámara", "camaras tletl", "cámaras tletl"],
        "commands": ["ls -lah /dev/video* 2>/dev/null || true"],
        "risk": "safe",
    },
    {
        "key": "tletl.datasets",
        "title": "Revisar datasets",
        "description": "Busca bancos de gestos y archivos json/jsonl.",
        "modes": ["tletl"],
        "aliases": ["datasets tletl", "gestos tletl", "banco gestos"],
        "commands": ["cd {active_path} && find . -maxdepth 4 -type f \\( -iname '*.json' -o -iname '*.jsonl' -o -iname '*gesture*' \\) | sort | head -160"],
        "risk": "safe",
    },
]
