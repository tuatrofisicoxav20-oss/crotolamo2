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
    # Zona de confirmación (M6): fuera de allowed_roots pero dentro de estas
    # raíces, las tools corren PREVIA confirmación del patrón. Default: su home.
    confirm_roots: list[Path] = field(default_factory=list)
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

    @property
    def persona(self) -> dict[str, Any]:
        return self.raw.get("persona", {})

    @property
    def memoria(self) -> dict[str, Any]:
        """[memoria]: memoria semántica con mem0 (ver core/memoria.py)."""
        return self.raw.get("memoria", {})

    @property
    def mcp(self) -> dict[str, Any]:
        """Sección [mcp] (M4). Los servers van como tablas nombradas
        [mcp.servers.<nombre>] para que _deep_merge los fusione con el local.toml
        (un array de tablas se pisaría entero)."""
        return self.raw.get("mcp", {})

    def validate_critical(self) -> list[str]:
        """Devuelve lista de problemas con rutas críticas (no lanza)."""
        problems = []
        home = self.paths.get("home")
        if home and not home.exists():
            problems.append(f"home no existe: {home}")
        return problems


def _path_list(paths_raw: dict, key: str, default: list[str]) -> list[Path]:
    """Lista de rutas de [paths].<key>, validada.

    Un typo como `allowed_roots = "~/Documentos"` (string en vez de lista) se
    iteraba carácter a carácter y metía "/" como raíz permitida: TODO el disco
    quedaba en zona libre sin confirmación. Mejor no arrancar que arrancar así.
    """
    raw = paths_raw.get(key, default)
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise ValueError(
            f"[paths].{key} debe ser una lista de rutas (strings), "
            f"no {type(raw).__name__}: {raw!r}"
        )
    return [_expand(p) for p in raw]


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

    allowed_roots = _path_list(paths_raw, "allowed_roots", [])
    # [paths].confirm_roots puede no existir aún en el toml: default ["~"].
    confirm_roots = _path_list(paths_raw, "confirm_roots", ["~"])
    projects = {name: _expand(p) for name, p in data.get("projects", {}).items()}

    return Settings(
        raw=data,
        user=os.environ.get("USER") or Path.home().name,
        home=Path.home(),
        paths=paths,
        allowed_roots=allowed_roots,
        confirm_roots=confirm_roots,
        projects=projects,
    )


# Singleton perezoso para uso normal.
_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS
