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
    # Ancla de las rutas RELATIVAS que reciben las tools de archivo nativas
    # ([paths].relative_base). None (de fábrica) = el CWD del proceso, como
    # siempre; con el .service y launch.sh ese CWD es el checkout del repo.
    relative_base: Path | None = None

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

    Por lo mismo, una raíz relativa ("" o "Documentos" sin el ~/, o una variable
    de entorno sin definir) tampoco vale: se resolvía contra el CWD del proceso,
    así que el corral cambiaba según desde dónde se lanzara, y con "" y el CWD
    en / era el disco entero.
    """
    raw = paths_raw.get(key, default)
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise ValueError(
            f"[paths].{key} debe ser una lista de rutas (strings), "
            f"no {type(raw).__name__}: {raw!r}"
        )
    roots = [_expand(p) for p in raw]
    relative = [p for p, root in zip(raw, roots) if not root.is_absolute()]
    if relative:
        raise ValueError(
            f"[paths].{key} solo admite rutas absolutas (o que empiecen por ~), "
            f"no relativas: {relative!r}"
        )
    return roots


def _relative_base(paths_raw: dict) -> Path | None:
    """[paths].relative_base: dónde caen las rutas relativas de las tools de
    archivo. Sin la clave (o vacía) no hay ancla y se resuelven contra el CWD,
    que es lo de siempre. Si se pone, tiene que ser absoluta (o empezar por ~),
    por la misma razón que las raíces: relativa dependería de desde dónde se
    lance Crotolamo, justo lo que la clave quiere evitar.
    """
    raw = paths_raw.get("relative_base", "")
    if not isinstance(raw, str):
        raise ValueError(f"[paths].relative_base debe ser una ruta (string), no {raw!r}")
    if not raw.strip():
        return None
    base = _expand(raw)
    if not base.is_absolute():
        raise ValueError(
            f"[paths].relative_base solo admite una ruta absoluta (o que empiece por ~), "
            f"no relativa: {raw!r}"
        )
    return base


def load_settings(config_path: Path | None = None) -> Settings:
    """Carga la configuración. Sin `config_path`: default + local.

    crotolamo.local.toml es el override del toml versionado, no de cualquier
    config: con una ruta explícita se carga ESE archivo y nada más. Fusionarlo
    siempre le echaba encima las rutas permitidas y los proyectos locales del
    patrón a quien cargaba un toml propio (los tests).
    """
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"No encuentro la config: {path}")

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    if config_path is None and LOCAL_CONFIG.exists():
        with LOCAL_CONFIG.open("rb") as fh:
            data = _deep_merge(data, tomllib.load(fh))

    paths_raw = data.get("paths", {})
    paths = {
        key: _expand(value)
        for key, value in paths_raw.items()
        if key not in {"allowed_roots", "relative_base"} and isinstance(value, str)
    }

    allowed_roots = _path_list(paths_raw, "allowed_roots", [])
    # [paths].confirm_roots puede no existir aún en el toml: default ["~"].
    confirm_roots = _path_list(paths_raw, "confirm_roots", ["~"])
    projects = {name: _expand(p) for name, p in data.get("projects", {}).items()}
    relative_base = _relative_base(paths_raw)

    return Settings(
        raw=data,
        user=os.environ.get("USER") or Path.home().name,
        home=Path.home(),
        paths=paths,
        allowed_roots=allowed_roots,
        confirm_roots=confirm_roots,
        projects=projects,
        relative_base=relative_base,
    )


# Singleton perezoso para uso normal.
_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS
