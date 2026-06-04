"""Carga de configuración. Config-first, cero hardcodeo.

Lee config/crotolamo.toml (+ crotolamo.local.toml opcional para overrides),
expande ~ y variables de entorno, detecta el usuario actual y valida que las
rutas críticas existan.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Raíz del repo = dos niveles arriba de este archivo (crotolamo/settings.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "crotolamo.toml"
LOCAL_CONFIG = CONFIG_DIR / "crotolamo.local.toml"


def _expand(value: str) -> Path:
    """Expande ~ y variables de entorno en una ruta."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge recursivo: override gana, pero no borra claves que no menciona."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Settings:
    raw: dict[str, Any]
    user: str
    home: Path
    paths: dict[str, Path] = field(default_factory=dict)
    allowed_roots: list[Path] = field(default_factory=list)
    projects: dict[str, Path] = field(default_factory=dict)

    # --- accesos cómodos a secciones ---
    @property
    def llm(self) -> dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def memory(self) -> dict[str, Any]:
        return self.raw.get("memory", {})

    @property
    def voice(self) -> dict[str, Any]:
        return self.raw.get("voice", {})

    @property
    def wake(self) -> dict[str, Any]:
        return self.raw.get("wake", {})

    def validate_critical(self) -> list[str]:
        """Devuelve lista de problemas con rutas críticas (no lanza)."""
        problems = []
        home = self.paths.get("home")
        if home and not home.exists():
            problems.append(f"home no existe: {home}")
        return problems


def load_settings(config_path: Path | None = None) -> Settings:
    """Carga la configuración fusionando default + local."""
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"No encuentro la config: {path}")

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    if LOCAL_CONFIG.exists():
        with LOCAL_CONFIG.open("rb") as fh:
            data = _deep_merge(data, tomllib.load(fh))

    paths_raw = data.get("paths", {})
    paths = {
        key: _expand(value)
        for key, value in paths_raw.items()
        if key not in {"allowed_roots"} and isinstance(value, str)
    }

    allowed_roots = [_expand(p) for p in paths_raw.get("allowed_roots", [])]
    projects = {name: _expand(p) for name, p in data.get("projects", {}).items()}

    return Settings(
        raw=data,
        user=os.environ.get("USER") or Path.home().name,
        home=Path.home(),
        paths=paths,
        allowed_roots=allowed_roots,
        projects=projects,
    )


# Singleton perezoso para uso normal.
_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS
