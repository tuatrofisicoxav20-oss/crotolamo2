from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PluginInfo:
    name: str
    description: str
    risk: str = "safe"


@dataclass
class PluginResult:
    handled: bool
    text: str = ""
    commands: list[str] | None = None
    safe: bool = True


class BasePlugin:
    info = PluginInfo(name="base", description="Plugin base")

    def can_handle(self, text: str) -> bool:
        return False

    def handle(self, text: str) -> PluginResult:
        return PluginResult(False)
