"""Arreglos de la auditoría en la CI y en la documentación de entrada.

Cada bloque fija un hallazgo confirmado.

- La CI no tenía tope de tiempo por job: un test con hilos colgado retenía el
  runner hasta el límite de GitHub (6 h) y el aviso llegaba horas tarde.
- `mypy` en la CI solo cubría `crotolamo` e `interfaces`: el doctor
  (`scripts/`) y el HUD/panel (`desktop/`) no se chequeaban aunque pasan limpios.
- README_M3 situaba los cuatro threads de voz en `loop.py` (viven en
  `threads.py`) y daba por pendiente el AEC que ya trae `desktop/aec.sh`.
- README.md decía que los atajos de velocidad "con GLM siguen ayudando":
  `build_agent` los apaga con GLM de cerebro.
- El toml decía que `num_ctx = 2048` "sobra": la persona y las schemas enrutadas ya
  se comen la mayor parte de la ventana del motor local.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

from crotolamo import settings as settings_mod
from crotolamo.settings import Settings

_RAIZ = Path(__file__).resolve().parents[1]
_CI = _RAIZ / ".github" / "workflows" / "ci.yml"


def _bloque_del_job(texto: str, job: str) -> list[str]:
    """Las líneas del job `job` de un workflow (hasta el siguiente job o el final).
    Texto plano a propósito: PyYAML no es dependencia del proyecto."""
    lineas = texto.splitlines()
    inicio = lineas.index(f"  {job}:")
    bloque: list[str] = []
    for linea in lineas[inicio + 1:]:
        if linea.strip() and not linea.startswith("   "):
            break
        bloque.append(linea)
    return bloque


# --- CI: tope de tiempo por job -----------------------------------------------------

def test_ci_el_job_tiene_tope_de_tiempo():
    """Sin `timeout-minutes`, un test colgado (hay muchos con hilos) ocupa el runner
    las 6 h del límite de GitHub. El job dura ~2 min: el tope debe holgar."""
    bloque = _bloque_del_job(_CI.read_text(encoding="utf-8"), "lint-type-test")
    topes = [int(m.group(1)) for linea in bloque
             if (m := re.fullmatch(r" {4}timeout-minutes:\s*(\d+)\s*", linea))]
    assert len(topes) == 1, "el job lint-type-test no tiene timeout-minutes propio"
    assert 10 <= topes[0] <= 60


# --- CI: mypy cubre todo el código vivo ---------------------------------------------

def test_ci_mypy_cubre_scripts_y_desktop():
    """El doctor (scripts/) y el HUD/panel (desktop/) son código que se ejecuta:
    si la CI no los pasa por mypy, un error de tipos ahí llega a la lap sin aviso."""
    texto = _CI.read_text(encoding="utf-8")
    comandos = re.findall(r"^\s+run:\s*(mypy\b.*)$", texto, re.M)
    assert comandos, "la CI no corre mypy"
    cubiertos = {arg.rstrip("/") for cmd in comandos for arg in cmd.split()[1:]
                 if not arg.startswith("-")}
    for carpeta in ("crotolamo", "interfaces", "scripts", "desktop"):
        assert carpeta in cubiertos, f"mypy no chequea {carpeta}/ en la CI"
        assert (_RAIZ / carpeta).is_dir()


# --- README_M3: el mapa de módulos dice dónde vive cada pieza ------------------------

def _clases_por_modulo(carpeta: Path) -> dict[str, set[str]]:
    """{"crotolamo/voice/x.py": {clases de primer nivel}} sin importar nada (la voz
    arrastra numpy/sounddevice)."""
    out: dict[str, set[str]] = {}
    for py in sorted(carpeta.glob("*.py")):
        arbol = ast.parse(py.read_text(encoding="utf-8"))
        out[py.relative_to(_RAIZ).as_posix()] = {
            n.name for n in arbol.body if isinstance(n, ast.ClassDef)
        }
    return out


def test_readme_m3_situa_los_threads_y_el_loop_en_su_modulo():
    """README_M3 decía "`loop.py` — los 4 threads + VoiceLoop", pero Ear/Stt/Brain/
    Mouth viven en `threads.py`: quien busque el barge-in en loop.py no lo encuentra."""
    texto = (_RAIZ / "README_M3.md").read_text(encoding="utf-8")
    # Viñeta "- `ruta.py` — descripción" con sus líneas de continuación (2 espacios).
    vinetas = re.findall(r"^- `(crotolamo/[^`]+\.py)` — (.*(?:\n  .*)*)", texto, re.M)
    clases = _clases_por_modulo(_RAIZ / "crotolamo" / "voice")
    conocidas = set().union(*clases.values())

    # Cada clase que una viñeta nombra vive en el módulo de esa viñeta...
    for mod, desc in vinetas:
        nombradas = set(re.findall(r"`(\w+)`", desc)) & conocidas
        assert nombradas <= clases.get(mod, set()), f"README_M3 pone {nombradas} en {mod}"
    # ...y las piezas del loop se nombran en la viñeta de SU módulo.
    for cls in ("EarThread", "SttThread", "BrainThread", "MouthThread", "VoiceLoop"):
        suyo = next(mod for mod, nombres in clases.items() if cls in nombres)
        assert any(mod == suyo and f"`{cls}`" in desc for mod, desc in vinetas), (
            f"README_M3 no dice que {cls} vive en {suyo}")


def test_readme_m3_no_da_por_pendiente_el_aec_que_ya_existe():
    """La cancelación de eco de PipeWire (WebRTC) ya está en el repo
    (`desktop/aec.sh`, exigida por el doctor): README_M3 la listaba como futuro y no
    decía cómo activarla."""
    assert (_RAIZ / "desktop" / "aec.sh").is_file()
    texto = (_RAIZ / "README_M3.md").read_text(encoding="utf-8")
    assert "desktop/aec.sh" in texto
    pendiente = texto.split("## Pendiente", 1)[1]
    vinetas = re.findall(r"^- (.*)$", pendiente, re.M)
    assert not [v for v in vinetas if v.startswith("AEC")], vinetas


# --- README.md: los atajos de velocidad con GLM --------------------------------------

def _build_agent(monkeypatch, tmp_path, backend: str):
    from crotolamo.core import hooks
    from crotolamo.core import memoria as memoria_mod
    from crotolamo.core.fallback import FallbackLLM
    from interfaces import shell

    class _Mudo:
        model = "x"

        def chat(self, messages, tools=None):
            raise AssertionError("este test no conversa")

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": backend}}, user="test",
                                 home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    llm = FallbackLLM(_Mudo(), _Mudo()) if backend == "glm" else _Mudo()
    monkeypatch.setattr(shell, "build_llm", lambda s: llm)
    agent, _ = shell.build_agent()
    return agent, llm


def test_con_glm_los_atajos_de_velocidad_se_apagan(monkeypatch, tmp_path):
    """Lo que cuenta README.md §Motor de inferencia: con GLM de cerebro no hay
    fastpath ni short-circuit y el modelo ve TODAS las tools; si la nube cae y
    responde el respaldo local, vuelve solo el tool_routing. Si esto cambia,
    actualiza el README."""
    agent, llm = _build_agent(monkeypatch, tmp_path, "glm")
    todas = agent.registry.schemas()
    assert agent.fastpath is False
    assert agent.direct_tools == set()
    assert len(agent.route_fn("pausa la música")) == len(todas)

    llm._retry_at = time.monotonic() + 60  # breaker abierto: responde Ollama
    assert llm.degraded
    assert 0 < len(agent.route_fn("pausa la música")) < len(todas)
    assert agent.fastpath is False  # el fastpath no vuelve al degradar

    texto = (_RAIZ / "README.md").read_text(encoding="utf-8")
    parrafo = next(p for p in texto.split("\n\n") if "`direct_tools`" in p)
    assert "sigue ayudando" not in parrafo
    assert "local" in parrafo


def test_con_ollama_los_atajos_de_velocidad_siguen(monkeypatch, tmp_path):
    """La otra mitad del README: con el motor local los atajos siguen puestos."""
    agent, _ = _build_agent(monkeypatch, tmp_path, "ollama")
    assert agent.fastpath is True
    assert agent.direct_tools
    assert 0 < len(agent.route_fn("pausa la música")) < len(agent.registry.schemas())


def test_readme_cita_el_exec_real_del_lanzador():
    """README.md citaba `Exec=kitty -e launch.sh`; el `crotolamo.desktop` de la raíz
    usa `kitty --hold -e <ruta>/launch.sh`. Los flags citados deben ser los reales."""
    lanzador = _RAIZ / "crotolamo.desktop"
    real = re.search(r"^Exec=(\S+(?: -\S+)*) \S*launch\.sh$",
                     lanzador.read_text(encoding="utf-8"), re.M)
    assert real
    texto = (_RAIZ / "README.md").read_text(encoding="utf-8")
    citados = re.findall(r"`Exec=(\S+(?: -\S+)*) \S*launch\.sh`", texto)
    assert citados
    assert set(citados) == {real.group(1)}


# --- config: el comentario de [llm].num_ctx no promete una holgura que no hay --------

def _comentario_sobre(clave: str, texto: str) -> str:
    """El bloque de comentarios (#) pegado encima de `clave = ...` en el toml."""
    lineas = texto.splitlines()
    i = next(n for n, linea in enumerate(lineas) if re.match(rf"{clave}\s*=", linea))
    bloque: list[str] = []
    for linea in reversed(lineas[:i]):
        if not linea.startswith("#"):
            break
        bloque.append(linea.lstrip("#").strip())
    return " ".join(reversed(bloque))


def test_el_comentario_de_num_ctx_no_dice_que_sobra():
    """El toml decía "con el tool routing los prompts son cortos; 2048 sobra". Medido
    con el tokenizer de qwen2.5, un turno con 8 tools enrutadas ya parte de ~1 600-1 750
    tokens (persona + schemas) antes del historial, la pregunta y la respuesta, y lo que
    no cabe Ollama lo recorta en silencio. Aquí lo fijo se acota POR ABAJO a 4 chars por
    token (el real ronda 3.4): si se come más de la mitad de la ventana, el comentario
    debe avisarlo en vez de prometer holgura."""
    import json
    import tomllib

    from crotolamo.core.persona import STYLES
    from crotolamo.core.router import route_schemas
    from crotolamo.tools import build_registry

    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    data = tomllib.loads(texto)
    num_ctx = data["llm"]["num_ctx"]
    persona = STYLES[data["persona"]["style"]]
    registry = build_registry()
    schemas = max((json.dumps(route_schemas(registry, q, data["llm"]["max_tools"]),
                              ensure_ascii=False)
                   for q in ("abre la carpeta de descargas", "cierra la ventana de opera",
                             "lee el archivo notas.txt de documentos")), key=len)
    fijo_minimo = (len(persona) + len(schemas)) // 4
    if fijo_minimo <= num_ctx // 2:
        return  # con una ventana holgada, el comentario puede decir lo que quiera
    comentario = _comentario_sobre("num_ctx", texto)
    assert "sobra" not in comentario, comentario
    assert re.search(r"recorta|descarta", comentario), comentario


# --- entry-scripts-ci#25: la doc describe el wake DE FÁBRICA, no el del patrón -------
#
# El toml versionado trae `use_oww = true` + `oww_model = "hey_jarvis"`: con la extra
# [voice] (que instala openwakeword) un clon limpio despierta con "hey jarvis" y en el
# loop concurrente. README.md decía a secas "el wake word usa Whisper `tiny`", que es
# el wake difuso (`use_oww = false`, lo que usa el patrón en su local.toml). Si el
# default cambia, estos tests siguen valiendo: miran el toml, no un valor fijo.

def _wake_de_fabrica() -> dict:
    import tomllib

    return tomllib.loads(settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _notas_de_voz() -> str:
    texto = (_RAIZ / "README.md").read_text(encoding="utf-8")
    return texto.split("### Notas de voz", 1)[1].split("\n## ", 1)[0]


def test_readme_dice_que_wake_corre_de_fabrica():
    data = _wake_de_fabrica()
    notas = _notas_de_voz()
    vinetas = re.findall(r"^- (.*(?:\n  .*)*)", notas, re.M)
    wake = [v for v in vinetas if "wake word" in v]
    assert len(wake) == 1, wake
    nota = " ".join(wake[0].split())
    assert "use_oww" in nota, "la nota no dice de qué clave depende el wake"
    if data["wake"]["use_oww"]:
        # De fábrica es openWakeWord con el modelo puente: hay que decir con qué
        # frase despierta, o el patrón dice "crotolamo" y no pasa nada.
        modelo = data["wake"]["oww_model"]
        assert "openWakeWord" in nota
        assert modelo in nota or modelo.replace("_", " ") in nota, nota
    # Whisper `tiny` es el wake DIFUSO: solo con use_oww = false (o sin openwakeword).
    tiny = f"`{data['voice']['wake_whisper_model']}`"
    assert tiny in nota
    assert "use_oww = false" in nota, nota


def test_el_toml_no_pide_volver_al_wake_que_ya_es_default():
    """El comentario de [voice].input_device decía "volver a `use_oww = true`" en un
    toml que YA trae `use_oww = true` (lo escribió quien tenía false en su local)."""
    data = _wake_de_fabrica()
    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    actual = "true" if data["wake"]["use_oww"] else "false"
    comentarios = " ".join(linea.lstrip("#").strip() for linea in texto.splitlines()
                           if linea.lstrip().startswith("#"))
    assert f"volver a `use_oww = {actual}`" not in comentarios


def test_el_toml_dice_con_que_frase_despierta_openwakeword():
    """El comentario de [wake].use_oww explica qué loop corre pero no que, con true, la
    frase es la del modelo puente ("hey jarvis") y "crotolamo" no despierta nada."""
    data = _wake_de_fabrica()
    texto = settings_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    comentario = _comentario_sobre("use_oww", texto)
    frase = data["wake"]["oww_model"].replace("_", " ")
    assert frase in comentario, comentario
    assert "oww_model" in comentario, comentario
