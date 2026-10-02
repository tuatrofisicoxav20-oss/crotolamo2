"""Auditor de salud de Crotolamo 2.

Audita la pila REAL que arranca Crotolamo, guiado por la sección [doctor] de la
config: la nube OpenAI-compatible de [llm.glm] (y su API key), el STT y el TTS
principales, la cancelación de eco (AEC) de PipeWire y el servicio de música.
Reporta ✅/❌/⚠️ por check, en dos bloques (REQUERIDOS y OPCIONALES), con una
sugerencia de fix debajo de cada fallo.

POR QUÉ SE REESCRIBIÓ: el doctor viejo auditaba la pila de la Fase 0 (Ollama
como motor, tomllib, Piper) y decía "todo en orden" con el stack de verdad
roto: la API key de la nube no cargaba, Crotolamo caía a Ollama en silencio y
el doctor salía verde. Ahora un REQUERIDO en rojo devuelve 1; los opcionales se
reportan pero no bloquean.

Reglas de la casa:
- Nunca lanza: cada check corre protegido; si revienta, se reporta ❌ con el
  motivo y el doctor sigue con el resto.
- Todo lo de red y subprocess lleva timeout corto (≤ 5 s).
- NUNCA se imprime ni loguea la API key (ni parte, ni su longitud): solo
  "presente"/"ausente". Por si acaso, todo texto que sale pasa por _scrub().

Uso:
    python -m crotolamo doctor
    python scripts/crotolamo_doctor.py
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from crotolamo.core.engine import GLM, resolve_backend
    from crotolamo.core.glm import API_KEY_ENVS, DEFAULT_BASE_URL, DEFAULT_MODEL
    from crotolamo.settings import LOCAL_CONFIG, Settings, load_settings
except ModuleNotFoundError:  # ejecutado como script suelto
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crotolamo.core.engine import GLM, resolve_backend
    from crotolamo.core.glm import API_KEY_ENVS, DEFAULT_BASE_URL, DEFAULT_MODEL
    from crotolamo.settings import LOCAL_CONFIG, Settings, load_settings

# Timeouts cortos: el doctor se corre a mano y debe contestar en segundos aunque
# la red esté caída o PipeWire colgado.
NET_TIMEOUT_S = 5.0   # nube y Ollama
CMD_TIMEOUT_S = 5.0   # pw-dump, systemctl
WS_TIMEOUT_S = 2.0    # WebSocket del servicio de música

# Defaults de [doctor] por si la config no trae la sección (config vieja). Reflejan
# lo que HAY en el repo; el toml documenta cómo apuntarlos a otra pila.
_DOCTOR_DEFAULTS: dict[str, Any] = {
    "stt_module": "faster_whisper",
    "stt_model": "",
    "tts_model": "",      # vacío = Piper ([paths].voces + [voice].piper_voice)
    "tts_voices": "",
    "tts_voice": "",
    "aec_source": "crotolamo_aec_source",
    "aec_sink": "crotolamo_aec_sink",
    "musica_service": "soloist.service",
    "musica_host": "127.0.0.1",
    "musica_port": 9090,
    "musica_caduca": None,  # sin fecha = el check se omite; el toml trae la suya
    "musica_aviso_dias": 14,
    "spotify_client_id_env": "SPOTIFY_CLIENT_ID",
}

# El archivo lo carga launch.sh al arrancar (también desde el .desktop, donde
# ~/.zshrc no existe). Sin `export` para que sirva igual como EnvironmentFile.
_KEY_FIX = (
    'Ponla en ~/.config/crotolamo/env como CROTOLAMO_GLM_API_KEY="..." (sin export; '
    "chmod 600) y vuelve a arrancar Crotolamo."
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    # required: un ❌ aquí hace salir con 1. Lo fija el registro de checks, no
    # cada check, para que un check que revienta herede el mismo peso.
    required: bool = False
    # warn: ok=True pero con aviso (se pinta ⚠️ y no bloquea).
    warn: bool = False

    @property
    def mark(self) -> str:
        if not self.ok:
            return "❌"
        return "⚠️" if self.warn else "✅"


@dataclass
class _Ctx:
    """Lo que comparten todos los checks: config cargada y backend resuelto."""

    settings: Settings
    doctor: dict[str, Any]
    backend: str
    today: date


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _ruta(value: str) -> Path:
    """Expande ~ y variables de entorno (mismo criterio que settings.py)."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _section(settings: Settings, name: str) -> dict[str, Any]:
    raw = settings.raw.get(name)
    return dict(raw) if isinstance(raw, dict) else {}


def _key_env() -> str | None:
    """Nombre de la env que trae la API key de la nube, o None si no hay.

    Devuelve el NOMBRE, nunca el valor: es lo único que el doctor puede decir.
    Mismo criterio que `glm._find_api_key` (orden de API_KEY_ENVS, con strip).
    """
    for env in API_KEY_ENVS:
        if (os.environ.get(env) or "").strip():
            return env
    return None


def _try_import(module: str) -> tuple[Any | None, str | None]:
    """(módulo, None) si importa; (None, motivo) si no.

    Se atrapa cualquier excepción, no solo ImportError: onnxruntime o portaudio
    pueden reventar con OSError al cargar una .so ausente, y eso también es
    diagnóstico útil.
    """
    try:
        return importlib.import_module(module), None
    except Exception as error:  # noqa: BLE001 - ver docstring
        return None, f"{type(error).__name__}: {error}"


def _import_error(module: str) -> str | None:
    return _try_import(module)[1]


def _ollama_tags(host: str, timeout: float = NET_TIMEOUT_S) -> list[str] | None:
    """Devuelve la lista de modelos instalados, o None si Ollama no responde."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def _model_ids(body: bytes) -> list[str] | None:
    """ids de un listado OpenAI (`{"data": [{"id": ...}]}`), o None si no lo es."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    return [str(m["id"]) for m in items if isinstance(m, dict) and "id" in m]


def _probe_cloud(base_url: str, model: str, key: str,
                 timeout: float = NET_TIMEOUT_S) -> Check:
    """GET {base_url}/models con la key. Devuelve el check `llm-nube`.

    Distingue los fallos que en producción se confunden entre sí (y que el
    FallbackLLM tapa cayendo a Ollama): key inválida, rate limit, sin internet,
    timeout. La key solo viaja en la cabecera: jamás entra en el detalle.
    """
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return Check("llm-nube", False, f"key inválida o caducada ({error.code} de {url})",
                         "Genera una key nueva en el proveedor y actualiza "
                         "~/.config/crotolamo/env.")
        if error.code == 429:
            return Check("llm-nube", False,
                         "rate limit (429): la key sirve pero el proveedor limita",
                         "Espera un rato o revisa el plan/cuota del proveedor.")
        return Check("llm-nube", False, f"respondió {error.code} en {url}",
                     "Revisa [llm.glm].base_url: ¿es la raíz OpenAI-compatible?")
    except urllib.error.URLError as error:
        # socket.timeout es TimeoutError desde 3.10; urlopen lo envuelve en URLError
        # cuando el timeout salta al conectar.
        if isinstance(error.reason, TimeoutError):
            return Check("llm-nube", False, f"no respondió en {timeout:.0f} s ({url})",
                         "Red lenta o proveedor caído: reintenta en un rato.")
        return Check("llm-nube", False,
                     f"sin internet o host inalcanzable ({error.reason})",
                     "Revisa la conexión y [llm.glm].base_url.")
    except TimeoutError:
        return Check("llm-nube", False, f"no respondió en {timeout:.0f} s ({url})",
                     "Red lenta o proveedor caído: reintenta en un rato.")
    except OSError as error:  # ConnectionError y compañía
        return Check("llm-nube", False, f"sin internet o host inalcanzable ({error})",
                     "Revisa la conexión y [llm.glm].base_url.")

    ids = _model_ids(body)
    if ids is None:
        return Check("llm-nube", True, f"responde (modelo configurado: {model})")
    if model in ids:
        return Check("llm-nube", True,
                     f"responde (modelo configurado: {model}, listado en /models)")
    muestra = ", ".join(ids[:6]) + ("…" if len(ids) > 6 else "")
    return Check("llm-nube", True,
                 f"responde, pero '{model}' no aparece en /models (hay {len(ids)}: {muestra})",
                 "Corrige [llm.glm].model con uno de los listados.", warn=True)


def _pw_dump() -> list[dict[str, Any]]:
    """Objetos del grafo de PipeWire (`pw-dump`, JSON). Sustituible en tests.

    Lanza FileNotFoundError si no hay pw-dump, subprocess.TimeoutExpired si se
    cuelga y RuntimeError si falla o no devuelve JSON; el check traduce cada
    caso a un mensaje en personaje.
    """
    proc = subprocess.run(["pw-dump"], capture_output=True, text=True,
                          timeout=CMD_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        motivo = proc.stderr.strip().splitlines()[:1] or ["sin detalle"]
        raise RuntimeError(f"pw-dump falló (rc={proc.returncode}): {motivo[0][:120]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"pw-dump no devolvió JSON: {error}") from error
    if not isinstance(data, list):
        raise RuntimeError("pw-dump devolvió un JSON que no es una lista")
    return data


def _node_mute(node: dict[str, Any]) -> bool | None:
    """Mute del nodo según info.params.Props, o None si pw-dump no lo trae."""
    params = (node.get("info") or {}).get("params") or {}
    for prop in params.get("Props") or []:
        if isinstance(prop, dict) and "mute" in prop:
            return bool(prop["mute"])
    return None


def _listener_pids() -> list[int]:
    """PIDs de procesos `crotolamo … listen` (sin el propio). Sustituible en tests.

    Lee /proc/*/cmdline directo: sin psutil (cero dependencias) y sin `pgrep`
    (otro subprocess). "listen" se exige como argumento suelto para no contar
    un `listener.env` o un `grep listen` casual. Si un candidato es PADRE de
    otro (launch.sh o kitty envolviendo al python real) se descarta: es el
    mismo listener, no dos.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    me = os.getpid()
    padres: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # murió entre medias o no es nuestro
        if b"listen" not in argv or not any(b"crotolamo" in arg for arg in argv):
            continue
        # /proc/<pid>/stat: "pid (comm) estado ppid …"; comm puede traer espacios.
        campos = stat.rsplit(")", 1)[-1].split()
        padres[int(entry.name)] = int(campos[1]) if len(campos) > 1 else 0
    envoltorios = set(padres.values())
    return sorted(pid for pid in padres if pid not in envoltorios)


def _as_date(raw: Any) -> date | None:
    """Fecha TOML (date/datetime) o ISO en string; None si no hay. Lanza si es basura."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and raw.strip():
        return date.fromisoformat(raw.strip())
    return None


def _piper_model_status(settings: Settings) -> Check:
    """Estado del modelo Piper de la config (lo usan `tts` y `fallback-voz`)."""
    voces = settings.paths.get("voces")
    name = str(settings.voice.get("piper_voice") or "")
    if not voces or not name:
        return Check("tts", False, "sin [paths].voces o [voice].piper_voice en la config",
                     "Define ambos en config/crotolamo.toml.")
    onnx = voces / name
    if not onnx.exists():
        return Check("tts", False, f"falta la voz Piper {onnx}",
                     "Descarga la voz .onnx y su .onnx.json a [paths].voces.")
    meta = onnx.with_name(onnx.name + ".json")
    if not meta.exists():
        return Check("tts", True,
                     f"Piper: {onnx} (falta {meta.name} al lado: Piper no podrá cargarla)",
                     "Descarga el .onnx.json de la misma voz junto al .onnx.", warn=True)
    return Check("tts", True, f"Piper: {onnx}")


def _voice_in_archive(path: Path, voice: str) -> tuple[bool | None, str]:
    """¿Está `voice` dentro del archivo de voces? (True/False, o None + motivo si
    no se pudo verificar). .json = diccionario; .bin/.npz = NPZ de numpy (Kokoro)."""
    if path.suffix == ".json":
        try:
            with path.open("rb") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as error:
            return None, f"no pude leer el JSON ({error})"
        if not isinstance(data, dict):
            return None, "el JSON de voces no es un diccionario"
        return voice in data, ""
    try:
        import numpy as np
    except ImportError:
        return None, "sin numpy"
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:  # noqa: BLE001 - formato raro o corrupto: no sabemos
        return None, f"no pude abrir el archivo ({error})"
    names = getattr(data, "files", None)
    if names is None:
        return None, "no es un archivo NPZ de voces"
    try:
        return voice in names, ""
    finally:
        data.close()


# ---------------------------------------------------------------------------
# Checks. Cada uno devuelve una lista (puede ser vacía = "no aplica"). El peso
# (requerido/opcional) lo pone el registro _CHECKS, no la función.
# ---------------------------------------------------------------------------

def _check_config() -> tuple[Settings | None, Check]:
    """Carga la config. Absorbe el viejo check `rutas` ([paths].home existe)."""
    try:
        settings = load_settings()
    except Exception as error:  # noqa: BLE001 - el doctor nunca debe morir
        return None, Check("config", False, f"no pude cargar la config: {error}",
                           "Revisa config/crotolamo.toml (y crotolamo.local.toml si existe).",
                           required=True)
    fuentes = "config/crotolamo.toml" + (" + crotolamo.local.toml" if LOCAL_CONFIG.exists()
                                         else "")
    problems = settings.validate_critical()
    if problems:
        return settings, Check("config", False, f"{fuentes} cargada, pero {'; '.join(problems)}",
                               "Crea las carpetas o ajusta [paths] en la config.", required=True)
    return settings, Check("config", True, f"{fuentes} cargada (usuario '{settings.user}')",
                           required=True)


def _nota_backend(ctx: _Ctx) -> str:
    return "" if ctx.backend == GLM else f" (backend={ctx.backend}: la nube no se usa)"


def _check_llm_key(ctx: _Ctx) -> list[Check]:
    env = _key_env()
    if env:
        return [Check("llm-key", True, f"presente en {env}{_nota_backend(ctx)}")]
    return [Check("llm-key", False,
                  f"falta {API_KEY_ENVS[0]}: Crotolamo va a caer a Ollama local"
                  f"{_nota_backend(ctx)}", _KEY_FIX)]


def _check_llm_nube(ctx: _Ctx) -> list[Check]:
    glm = ctx.settings.llm.get("glm")
    glm = glm if isinstance(glm, dict) else {}
    base_url = str(glm.get("base_url") or DEFAULT_BASE_URL)
    model = str(glm.get("model") or DEFAULT_MODEL)
    env = _key_env()
    if env is None:
        return [Check("llm-nube", False,
                      f"sin key, no pruebo la nube ({base_url}){_nota_backend(ctx)}", _KEY_FIX)]
    check = _probe_cloud(base_url, model, os.environ[env].strip())
    check.detail += _nota_backend(ctx)
    return [check]


def _check_stt(ctx: _Ctx) -> list[Check]:
    module = str(ctx.doctor.get("stt_module") or "faster_whisper")
    error = _import_error(module)
    if error:
        return [Check("stt", False, f"no puedo importar {module}: {error}",
                      f"pip install {module.replace('_', '-')}  (o pip install -e '.[voice]')")]
    detail = f"{module} importable"
    model = str(ctx.doctor.get("stt_model") or "")
    if not model:
        return [Check("stt", True, f"{detail} (sin ruta de modelo que comprobar)")]
    path = _ruta(model)
    if not path.exists():
        return [Check("stt", False, f"{detail}, pero falta el modelo {path}",
                      "Descarga el modelo a esa ruta o corrige [doctor].stt_model.")]
    return [Check("stt", True, f"{detail}; modelo en {path}")]


def _check_tts(ctx: _Ctx) -> list[Check]:
    model = str(ctx.doctor.get("tts_model") or "")
    if not model:
        return [_piper_model_status(ctx.settings)]  # Piper es el TTS principal

    path = _ruta(model)
    if not path.exists():
        return [Check("tts", False, f"falta el modelo TTS {path}",
                      "Descarga el modelo o corrige [doctor].tts_model.")]
    detail = f"modelo {path}"
    voices = str(ctx.doctor.get("tts_voices") or "")
    voice = str(ctx.doctor.get("tts_voice") or "")
    if not voices:
        return [Check("tts", True, detail)]
    vpath = _ruta(voices)
    if not vpath.exists():
        return [Check("tts", False, f"{detail}, pero falta el archivo de voces {vpath}",
                      "Descarga el archivo de voces o corrige [doctor].tts_voices.")]
    detail += f" + voces {vpath.name}"
    if not voice:
        return [Check("tts", True, detail)]
    found, motivo = _voice_in_archive(vpath, voice)
    if found is None:
        return [Check("tts", True, f"{detail}; no pude verificar la voz '{voice}': {motivo}",
                      "Instala numpy (pip install -e '.[voice]') para verificar la voz.",
                      warn=True)]
    if not found:
        return [Check("tts", False, f"{detail}, pero la voz '{voice}' no está en {vpath.name}",
                      "Corrige [doctor].tts_voice o descarga un archivo de voces que la traiga.")]
    return [Check("tts", True, f"{detail} (voz {voice} OK)")]


def _check_aec(ctx: _Ctx) -> list[Check]:
    source = str(ctx.doctor.get("aec_source") or "")
    sink = str(ctx.doctor.get("aec_sink") or "")
    if not source and not sink:
        return []  # sin AEC configurado: nada que auditar

    try:
        dump = _pw_dump()
    except FileNotFoundError:
        return [Check("aec", False, "no tengo pw-dump (pipewire-utils)",
                      "sudo dnf install pipewire-utils")]
    except subprocess.TimeoutExpired:
        return [Check("aec", False, f"pw-dump no respondió en {CMD_TIMEOUT_S:.0f} s",
                      "PipeWire colgado: systemctl --user restart pipewire wireplumber")]
    except RuntimeError as error:
        return [Check("aec", False, str(error),
                      "¿Está PipeWire corriendo? systemctl --user status pipewire")]

    nodes: dict[str, dict[str, Any]] = {}
    defaults: dict[str, str] = {}
    for obj in dump:
        kind = obj.get("type")
        if kind == "PipeWire:Interface:Node":
            name = ((obj.get("info") or {}).get("props") or {}).get("node.name")
            if isinstance(name, str):
                nodes[name] = obj
        elif kind == "PipeWire:Interface:Metadata":
            if (obj.get("props") or {}).get("metadata.name") != "default":
                continue
            for entry in obj.get("metadata") or []:
                value = entry.get("value") if isinstance(entry, dict) else None
                if isinstance(value, dict) and isinstance(value.get("name"), str):
                    defaults[str(entry.get("key"))] = value["name"]

    fallos: list[str] = []
    avisos: list[str] = []
    fixes: list[str] = []
    faltan = [n for n in (source, sink) if n and n not in nodes]
    if faltan:
        fallos.append(f"faltan los nodos {', '.join(faltan)}")
        fixes.append("Activa la cancelación de eco: desktop/aec.sh on")
    else:
        if sink and defaults.get("default.audio.sink") != sink:
            fallos.append("el AEC no tiene referencia: Crotolamo se puede activar solo con su "
                          f"propia voz (sink por defecto: "
                          f"{defaults.get('default.audio.sink') or 'ninguno'})")
            fixes.append(f"wpctl set-default {nodes[sink].get('id', '<id de ' + sink + '>')}")
        if source:
            default_source = defaults.get("default.audio.source")
            if default_source != source:
                msg = f"el source por defecto es {default_source or 'ninguno'}, no {source}"
                if ctx.settings.voice.get("input_device") == source:
                    # Crotolamo abre el AEC explícito: el default global no le afecta.
                    avisos.append(f"{msg} (Crotolamo lo abre explícito vía [voice].input_device)")
                else:
                    fallos.append(f"{msg}: el micro que abre Crotolamo trae eco")
                    fixes.append(f"wpctl set-default "
                                 f"{nodes[source].get('id', '<id de ' + source + '>')} "
                                 f'o pon [voice].input_device = "{source}"')
            mute = _node_mute(nodes[source])
            if mute:
                fallos.append("micrófono AEC en mute")
                fixes.append(f"wpctl set-mute {nodes[source].get('id', '<id de ' + source + '>')} 0")
            elif mute is None:
                avisos.append("no pude leer el mute del source (asumo que no está en mute)")

    if fallos:
        return [Check("aec", False, "; ".join(fallos + avisos), " | ".join(fixes))]
    resumen = f"nodos {source} y {sink} presentes, sink por defecto OK"
    if avisos:
        return [Check("aec", True, f"{resumen}; {'; '.join(avisos)}", warn=True)]
    return [Check("aec", True, f"{resumen}, source por defecto OK, micro sin mute")]


def _check_ollama(ctx: _Ctx) -> list[Check]:
    host = str(ctx.settings.llm.get("host", "http://localhost:11434"))
    model = str(ctx.settings.llm.get("model", "qwen2.5-coder:7b"))
    rol = "respaldo local" if ctx.backend == GLM else "motor principal"
    tags = _ollama_tags(host)
    if tags is None:
        return [Check("ollama", False, f"no responde en {host} ({rol})",
                      "Arranca el servicio: `ollama serve` (o systemctl start ollama).")]
    has_model = any(t == model or t.split(":")[0] == model.split(":")[0] for t in tags)
    return [
        Check("ollama", True, f"responde en {host} ({rol})"),
        Check("modelo", has_model,
              f"'{model}' instalado" if has_model
              else f"falta '{model}' (hay: {', '.join(tags) or 'ninguno'})",
              f"Instálalo: `ollama pull {model}`."),
    ]


def _check_fallback_voz(ctx: _Ctx) -> list[Check]:
    """Piper + faster-whisper son el respaldo cuando el TTS/STT principal es otro.
    Lo que ya revisó `tts`/`stt` como principal no se repite aquí."""
    partes: list[str] = []
    fallos: list[str] = []
    fixes: list[str] = []
    warn = False

    error = _import_error("piper")
    if error:
        fallos.append(f"piper no importa ({error})")
        fixes.append("pip install -e '.[voice]'")
    else:
        partes.append("piper importable")

    if ctx.doctor.get("tts_model"):  # el principal no es Piper: su modelo no lo vio `tts`
        piper = _piper_model_status(ctx.settings)
        (partes if piper.ok else fallos).append(piper.detail)
        warn = warn or piper.warn
        if piper.fix and (not piper.ok or piper.warn):
            fixes.append(piper.fix)

    if str(ctx.doctor.get("stt_module") or "faster_whisper") != "faster_whisper":
        error = _import_error("faster_whisper")
        if error:
            fallos.append(f"faster_whisper no importa ({error})")
            fixes.append("pip install faster-whisper")
        else:
            partes.append("faster_whisper importable")

    fix = " | ".join(dict.fromkeys(fixes))  # sin repetir el mismo fix
    if fallos:
        return [Check("fallback-voz", False, "; ".join(fallos + partes), fix)]
    return [Check("fallback-voz", True, "; ".join(partes), fix, warn=warn)]


def _check_musica_servicio(ctx: _Ctx) -> list[Check]:
    unit = str(ctx.doctor.get("musica_service") or "")
    if not unit:
        return []
    try:
        proc = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True,
                              text=True, timeout=CMD_TIMEOUT_S, check=False)
    except FileNotFoundError:
        return [Check("musica-servicio", False, f"no tengo systemctl: no puedo consultar {unit}",
                      "Sin systemd de usuario aquí; arranca el servicio de música a mano.")]
    except subprocess.TimeoutExpired:
        return [Check("musica-servicio", False,
                      f"systemctl no respondió en {CMD_TIMEOUT_S:.0f} s",
                      "systemd de usuario colgado: systemctl --user daemon-reexec")]
    # Sin sesión de usuario (SSH, contenedor) systemctl no escribe stdout: el
    # motivo viene por stderr ("Failed to connect to bus").
    estado = (proc.stdout.strip() or proc.stderr.strip() or f"rc={proc.returncode}")
    estado = estado.splitlines()[0]
    if estado == "active":
        return [Check("musica-servicio", True, f"{unit} activo")]
    return [Check("musica-servicio", False, f"{unit}: {estado}",
                  f"systemctl --user start {unit}")]


def _check_musica_ws(ctx: _Ctx) -> list[Check]:
    host = str(ctx.doctor.get("musica_host") or "127.0.0.1")
    port = int(ctx.doctor.get("musica_port") or 9090)
    unit = str(ctx.doctor.get("musica_service") or "el servicio de música")
    try:
        conn = socket.create_connection((host, port), timeout=WS_TIMEOUT_S)
    except OSError as error:
        return [Check("musica-ws", False, f"nadie acepta conexiones en {host}:{port} ({error})",
                      f"Arranca {unit} (systemctl --user start {unit}) y revisa "
                      "[doctor].musica_host/musica_port.")]
    conn.close()
    return [Check("musica-ws", True, f"acepta conexiones en {host}:{port}")]


def _check_spotify(ctx: _Ctx) -> list[Check]:
    env = str(ctx.doctor.get("spotify_client_id_env") or "")
    if not env:
        return []
    if (os.environ.get(env) or "").strip():
        return [Check("spotify", True, f"{env} presente")]
    return [Check("spotify", False, f"falta {env}: el módulo de Spotify no va a autenticar",
                  f'Ponla en ~/.config/crotolamo/env: {env}="..." (sin export).')]


def _check_musica_caduca(ctx: _Ctx) -> list[Check]:
    raw = ctx.doctor.get("musica_caduca")
    try:
        caduca = _as_date(raw)
    except ValueError:
        return [Check("musica-caduca", False, f"[doctor].musica_caduca no es una fecha: {raw!r}",
                      "Usa una fecha TOML sin comillas, p.ej. musica_caduca = 2026-11-29")]
    if caduca is None:
        return []  # sin fecha configurada: nada que vigilar
    aviso = int(ctx.doctor.get("musica_aviso_dias") or 14)
    dias = (caduca - ctx.today).days
    fix = "Renueva la credencial del servicio de música y actualiza [doctor].musica_caduca."
    if dias < 0:
        return [Check("musica-caduca", False,
                      f"caducó el {caduca.isoformat()} (hace {-dias} días)", fix)]
    if dias <= aviso:
        cuando = "HOY" if dias == 0 else f"en {dias} días"
        return [Check("musica-caduca", True, f"caduca {cuando} ({caduca.isoformat()})", fix,
                      warn=True)]
    return [Check("musica-caduca", True, f"vigente hasta {caduca.isoformat()} ({dias} días)")]


def _check_listeners(ctx: _Ctx) -> list[Check]:
    pids = _listener_pids()
    if len(pids) > 1:
        return [Check("listeners", True,
                      f"{len(pids)} listeners (pids {', '.join(map(str, pids))}): "
                      "servicio + uno manual: se pisan el micrófono",
                      "Para uno: systemctl --user stop crotolamo.service, o cierra el manual.",
                      warn=True)]
    if pids:
        return [Check("listeners", True, f"1 listener corriendo (pid {pids[0]})")]
    return [Check("listeners", True, "ningún listener corriendo")]


def _check_portaudio(ctx: _Ctx) -> list[Check]:
    """sounddevice + portaudio: sin ellos no hay micro ni altavoz (absorbe `voz-deps`)."""
    sd, error = _try_import("sounddevice")
    if sd is None:
        return [Check("portaudio", False, f"sounddevice no importa ({error})",
                      "pip install -e '.[voice]' y sudo dnf install -y portaudio")]
    try:
        n = len(sd.query_devices())
    except Exception as error:  # noqa: BLE001 - PortAudioError u otros: todo es diagnóstico
        return [Check("portaudio", False, f"sounddevice no consultó dispositivos: {error}",
                      "Instala portaudio: sudo dnf install -y portaudio.")]
    if n == 0:
        # portaudio carga pero no hay ni micro ni altavoz: en un contenedor es
        # normal; en la lap del patrón significa que PipeWire no está.
        return [Check("portaudio", True, "portaudio OK pero sounddevice no ve ningún dispositivo",
                      "¿PipeWire corriendo? systemctl --user status pipewire", warn=True)]
    return [Check("portaudio", True, f"portaudio OK (sounddevice ve {n} dispositivos)")]


def _check_oww(ctx: _Ctx) -> list[Check]:
    error = _import_error("openwakeword")
    if error:
        return [Check("voz-oww", False,
                      "falta openwakeword (sin él, solo modo simple/difuso está disponible)",
                      "pip install openwakeword  (o pip install -e '.[voice]')")]
    return [Check("voz-oww", True, "openwakeword importable (modo concurrente OK)")]


def _check_silero(ctx: _Ctx) -> list[Check]:
    error = _import_error("silero_vad")
    if error:
        return [Check("voz-silero", False,
                      "falta silero-vad (el modo concurrente usará VAD por energía como fallback)",
                      "pip install silero-vad torch  (opcional, mejora la detección de voz)")]
    return [Check("voz-silero", True, "silero-vad importable (VAD neuronal activo)")]


def _check_memoria(ctx: _Ctx) -> list[Check]:
    """Memoria semántica (mem0): deps importables, ruta escribible, telemetría off."""
    from crotolamo.core.memoria import MemoriaConfig, resolver_provider

    cfg = MemoriaConfig.from_settings(ctx.settings)
    provider = resolver_provider(cfg, ctx.settings)
    if not cfg.enabled:
        return [Check("memoria", True, "desactivada ([memoria].enabled = false)")]
    faltan = [m for m in ("mem0", "chromadb", "fastembed") if _import_error(m)]
    if provider == "groq" and _import_error("groq"):
        faltan.append("groq")
    if faltan:
        return [Check("memoria", False, f"faltan módulos: {', '.join(faltan)}",
                      "pip install -e '.[memoria]'  (versiones fijadas en pyproject)")]
    try:
        cfg.ruta.mkdir(parents=True, exist_ok=True)
        prueba = cfg.ruta / ".doctor_escribe"
        prueba.write_text("ok", encoding="utf-8")
        prueba.unlink()
    except OSError as error:
        return [Check("memoria", False, f"la ruta {cfg.ruta} no es escribible: {error}",
                      "Ajusta [memoria].ruta o los permisos de la carpeta.")]
    # core/memoria.py fija MEM0_TELEMETRY=False al importarse (antes de mem0).
    if os.environ.get("MEM0_TELEMETRY") != "False":
        return [Check("memoria", False, "la telemetría de mem0 NO está apagada",
                      "No exportes MEM0_TELEMETRY en el entorno; core/memoria.py la apaga.")]
    return [Check("memoria", True,
                  f"mem0/chromadb/fastembed OK, ruta {cfg.ruta} escribible, telemetría apagada "
                  f"(umbral {cfg.umbral:g}, top_k {cfg.top_k}, LLM {provider})")]


def _check_ydotool(ctx: _Ctx) -> list[Check]:
    has = shutil.which("ydotool") is not None
    return [Check("ydotool", has, "ydotool presente" if has else "sin ydotool (opcional)",
                  "Instala con `sudo dnf install ydotool` si quieres control de ventanas.")]


def _check_navegador(ctx: _Ctx) -> list[Check]:
    browser = shutil.which("xdg-open") or shutil.which("flatpak")
    return [Check("navegador", browser is not None,
                  f"lanzador disponible ({browser})" if browser else "sin xdg-open/flatpak",
                  "Instala xdg-utils: `sudo dnf install xdg-utils`.")]


# ---------------------------------------------------------------------------
# Registro: nombre (para el ❌ si el check revienta), función y peso.
# ---------------------------------------------------------------------------

def _siempre(ctx: _Ctx) -> bool:
    return True


def _nunca(ctx: _Ctx) -> bool:
    return False


def _si_nube(ctx: _Ctx) -> bool:
    """La nube solo es requerida si es el motor configurado."""
    return ctx.backend == GLM


def _si_local(ctx: _Ctx) -> bool:
    """Ollama es requerido cuando es el motor principal; con la nube, es el respaldo."""
    return ctx.backend != GLM


@dataclass(frozen=True)
class _Spec:
    name: str
    run: Callable[[_Ctx], list[Check]]
    required: Callable[[_Ctx], bool]


_CHECKS: tuple[_Spec, ...] = (
    _Spec("llm-key", _check_llm_key, _si_nube),
    _Spec("llm-nube", _check_llm_nube, _si_nube),
    _Spec("stt", _check_stt, _siempre),
    _Spec("tts", _check_tts, _siempre),
    _Spec("aec", _check_aec, _siempre),
    _Spec("ollama", _check_ollama, _si_local),
    _Spec("fallback-voz", _check_fallback_voz, _nunca),
    _Spec("musica-servicio", _check_musica_servicio, _nunca),
    _Spec("musica-ws", _check_musica_ws, _nunca),
    _Spec("spotify", _check_spotify, _nunca),
    _Spec("musica-caduca", _check_musica_caduca, _nunca),
    _Spec("listeners", _check_listeners, _nunca),
    _Spec("portaudio", _check_portaudio, _nunca),
    _Spec("voz-oww", _check_oww, _nunca),
    _Spec("voz-silero", _check_silero, _nunca),
    _Spec("memoria", _check_memoria, _nunca),
    _Spec("ydotool", _check_ydotool, _nunca),
    _Spec("navegador", _check_navegador, _nunca),
)


def _guarded(spec: _Spec, ctx: _Ctx) -> list[Check]:
    """Corre un check sin dejar que tumbe al doctor; fija su peso."""
    required = spec.required(ctx)
    try:
        checks = spec.run(ctx)
    except Exception as error:  # noqa: BLE001 - un check roto se reporta, no se propaga
        checks = [Check(spec.name, False,
                        f"el check reventó: {type(error).__name__}: {error}",
                        "Es un bug del doctor: repórtalo con este mensaje.")]
    for check in checks:
        check.required = required
    return checks


def _secretos(ctx: _Ctx) -> list[str]:
    """Valores de env que jamás deben salir por pantalla ni por el log."""
    envs = [*API_KEY_ENVS, str(ctx.doctor.get("spotify_client_id_env") or "")]
    valores = [(os.environ.get(env) or "").strip() for env in envs if env]
    return [v for v in valores if len(v) >= 4]  # algo tan corto no es una key


def _scrub(text: str, secretos: list[str]) -> str:
    for secreto in secretos:
        text = text.replace(secreto, "«oculto»")
    return text


def collect_checks(today: date | None = None) -> list[Check]:
    """Corre todos los checks y devuelve la lista (config primero).

    `today` se inyecta para poder probar las caducidades sin tocar el reloj.
    """
    settings, config_check = _check_config()
    checks = [config_check]
    if settings is None:
        return checks

    ctx = _Ctx(
        settings=settings,
        doctor={**_DOCTOR_DEFAULTS, **_section(settings, "doctor")},
        backend=resolve_backend(settings),
        today=today or date.today(),
    )
    for spec in _CHECKS:
        checks.extend(_guarded(spec, ctx))

    # Última barrera: ningún check debería meter una key en su texto, pero un
    # mensaje de excepción ajeno (urllib, numpy) podría. Se tapa antes de salir.
    secretos = _secretos(ctx)
    for check in checks:
        check.detail = _scrub(check.detail, secretos)
        check.fix = _scrub(check.fix, secretos)
    return checks


def run_doctor() -> int:
    from crotolamo.logging_setup import get_logger, setup_logging

    setup_logging()
    log = get_logger("doctor")

    checks = collect_checks()
    log.info("doctor: %d checks evaluados", len(checks))

    print("Doctor de Crotolamo 2\n" + "=" * 40)
    bloques = (
        ("REQUERIDOS", [c for c in checks if c.required]),
        ("OPCIONALES", [c for c in checks if not c.required]),
    )
    for titulo, subset in bloques:
        print(f"--- {titulo} ---")
        if not subset:
            print("   (nada que evaluar)")
        for c in subset:
            log.debug("check %-16s ok=%s warn=%s :: %s", c.name, c.ok, c.warn, c.detail)
            print(f"{c.mark} {c.name:16s} {c.detail}")
            if c.fix and (not c.ok or c.warn):
                print(f"   ↳ fix: {c.fix}")

    print("=" * 40)
    if any(c.required and not c.ok for c in checks):
        print("Hay fallos REQUERIDOS que arreglar, patrón.")
        return 1
    print("Todo en orden, patrón. (Los ❌/⚠️ opcionales no bloquean.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_doctor())
