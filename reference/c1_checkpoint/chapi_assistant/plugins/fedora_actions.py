ACTIONS = [
    {
        "key": "fedora.health",
        "title": "Salud Fedora",
        "description": "Diagnóstico seguro de Fedora: versión, memoria, disco y servicios fallidos.",
        "modes": ["fedora", "*"],
        "aliases": ["salud fedora", "estado fedora", "diagnostico fedora", "diagnóstico fedora"],
        "commands": [
            "uname -a",
            "cat /etc/fedora-release 2>/dev/null || true",
            "free -h",
            "df -h ~",
            "systemctl --failed --no-pager 2>/dev/null || true",
        ],
        "risk": "safe",
    },
    {
        "key": "fedora.audio",
        "title": "Diagnóstico de audio",
        "description": "Revisa PipeWire/PulseAudio, grabación y sounddevice.",
        "modes": ["fedora", "crotolamo", "tletl"],
        "aliases": ["audio fedora", "revisar audio", "diagnostico audio", "diagnóstico audio", "microfono", "micrófono"],
        "commands": [
            "pactl info 2>/dev/null || true",
            "arecord -l 2>/dev/null || true",
            "python -c \"import sounddevice as sd; print(sd.query_devices())\"",
        ],
        "risk": "safe",
    },
]
