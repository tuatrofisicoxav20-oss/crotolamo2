from __future__ import annotations

from plugins.base_plugin import BasePlugin, PluginInfo, PluginResult


class FedoraPlugin(BasePlugin):
    info = PluginInfo(name="fedora", description="Comandos de diagnóstico seguros para Fedora", risk="safe")

    def can_handle(self, text: str) -> bool:
        lower = text.lower().strip()
        return lower in {"diagnostico fedora", "diagnóstico fedora", "estado fedora", "revisa fedora"}

    def handle(self, text: str) -> PluginResult:
        return PluginResult(
            handled=True,
            text="Diagnóstico Fedora propuesto. Nada destructivo, qué raro ver prudencia por aquí.",
            safe=True,
            commands=[
                "uname -a",
                "cat /etc/fedora-release",
                "free -h",
                "df -h ~",
                "systemctl --failed --no-pager",
            ],
        )
