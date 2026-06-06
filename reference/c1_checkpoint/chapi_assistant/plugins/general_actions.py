ACTIONS = [
    {
        "key": "general.doctor",
        "title": "Doctor general",
        "description": "Corre el diagnóstico principal de Crotolamo.",
        "modes": ["*"],
        "aliases": ["doctor", "diagnostico crotolamo", "diagnóstico crotolamo", "revisar crotolamo"],
        "commands": ["cd {root} && python tools/crotolamo_doctor.py"],
        "risk": "safe",
    },
    {
        "key": "general.compile",
        "title": "Compilar núcleo",
        "description": "Revisa sintaxis de runtime, seguridad, modos, plugins, UI y doctor.",
        "modes": ["*"],
        "aliases": ["compilar crotolamo", "revisar sintaxis", "pycompile crotolamo"],
        "commands": [
            "cd {root} && python -m py_compile core/command_safety.py core/system_probe.py core/project_modes.py core/plugin_registry.py core/crotolamo_runtime.py ui/crotolamo_orbital_ui.py tools/crotolamo_doctor.py launch_runtime_shell.py"
        ],
        "risk": "safe",
    },
    {
        "key": "mode.status",
        "title": "Estado del modo activo",
        "description": "Lista ruta, git y archivos principales del modo activo.",
        "modes": ["*"],
        "aliases": ["estado modo", "revisar modo", "estado proyecto", "revisar proyecto"],
        "commands": [
            "cd {active_path} && pwd",
            "cd {active_path} && git status --short 2>/dev/null || true",
            "cd {active_path} && find . -maxdepth 2 -type f | sort | head -120",
        ],
        "risk": "safe",
    },
    {
        "key": "mode.tree",
        "title": "Mapa rápido del proyecto",
        "description": "Muestra hasta 200 archivos del modo activo para ubicar basura y módulos reales.",
        "modes": ["*"],
        "aliases": ["mapa proyecto", "arbol proyecto", "árbol proyecto", "listar proyecto"],
        "commands": ["cd {active_path} && find . -maxdepth 3 -type f | sort | head -200"],
        "risk": "safe",
    },
    {
        "key": "general.logs",
        "title": "Ver logs de Crotolamo",
        "description": "Lista logs recientes del runtime y de la UI orbital.",
        "modes": ["*"],
        "aliases": ["logs crotolamo", "ver logs", "logs"],
        "commands": ["cd {root} && ls -lah data/runtime_logs data/orbital_logs 2>/dev/null || true"],
        "risk": "safe",
    },
]
