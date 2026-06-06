ACTIONS = [
    {
        "key": "laboratorio.inventario",
        "title": "Inventario de laboratorio",
        "description": "Lista archivos y notas del modo laboratorio para ubicar ideas/prototipos.",
        "modes": ["laboratorio"],
        "aliases": ["inventario laboratorio", "revisar laboratorio", "ideas laboratorio"],
        "commands": ["cd {active_path} && find . -maxdepth 3 -type f | sort | head -160"],
        "risk": "safe",
    },
    {
        "key": "laboratorio.protocolo",
        "title": "Protocolo de prototipo",
        "description": "Recordatorio operativo para prototipos de electrónica/hardware.",
        "modes": ["laboratorio"],
        "aliases": ["protocolo prototipo", "prototipo laboratorio"],
        "commands": [],
        "direct_text": "Protocolo de prototipo v6:\n1. Objetivo medible.\n2. Lista de piezas.\n3. Diagrama simple.\n4. Prueba mínima.\n5. Bitácora con foto/video.\n6. Una mejora por ciclo. Nada de construir el Iron Man completo el martes, criatura ambiciosa.",
        "risk": "safe",
    },
]
