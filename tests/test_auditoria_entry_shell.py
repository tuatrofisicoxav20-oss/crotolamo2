"""Arreglos de la auditoría en el punto de entrada, el shell y el logging.

Cada bloque fija un hallazgo confirmado.

- `crotolamo doctor` desde el console-script (o `python -m crotolamo doctor`
  fuera de la raíz del repo): `scripts` no se empaqueta y `import scripts`
  reventaba con un traceback. Ahora el doctor se carga por su ruta en el repo, y
  si el script no está se dice qué falta.
- Un nivel de log que no existe (CROTOLAMO_LOG=trace, `level = 10` en el toml)
  caía a INFO en silencio (o reventaba al arrancar): ahora avisa.
"""

from __future__ import annotations

import logging
import sys

import pytest

from crotolamo import __main__ as main_mod
from crotolamo import logging_setup
from crotolamo import settings as settings_mod
from crotolamo.settings import Settings


@pytest.fixture(autouse=True)
def _logging_ya_configurado(monkeypatch):
    """main() llama a setup_logging(): que ningún test de aquí deje el logging del
    proceso "configurado" para el resto de la suite (los de logging lo pisan)."""
    monkeypatch.setattr(logging_setup, "_CONFIGURED", True)


# --- `crotolamo doctor` fuera de la raíz del repo ---------------------------------

@pytest.fixture
def sin_scripts(monkeypatch):
    """Como con el comando `crotolamo` (console-script) o `python -m crotolamo`
    desde otra carpeta: `scripts` no se empaqueta y la raíz del repo no está en
    sys.path, así que `import scripts` lanza ModuleNotFoundError."""
    monkeypatch.setitem(sys.modules, "scripts", None)
    monkeypatch.setitem(sys.modules, "scripts.crotolamo_doctor", None)
    antes = set(sys.modules)
    yield
    # El doctor cargado por su ruta no debe quedarse para el resto de la suite.
    for nombre in set(sys.modules) - antes:
        if "doctor" in nombre:
            sys.modules.pop(nombre, None)


def _config_rota(monkeypatch) -> None:
    """Con la config ilegible el doctor REAL se queda en su primer check: ni red,
    ni binarios, ni la config del autor. Basta para ver que arrancó y terminó."""
    def rota(config_path=None):
        raise FileNotFoundError("No encuentro la config (test)")

    monkeypatch.setattr(settings_mod, "load_settings", rota)


def test_doctor_corre_aunque_scripts_no_sea_importable(sin_scripts, monkeypatch, capsys):
    with pytest.raises(ImportError):  # el doble reproduce el fallo real
        import scripts.crotolamo_doctor  # noqa: F401
    _config_rota(monkeypatch)
    # Antes: ModuleNotFoundError: No module named 'scripts' (traceback crudo).
    assert main_mod.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "Doctor de Crotolamo 2" in out
    assert "❌ config" in out and "No encuentro la config (test)" in out
    assert "Hay fallos REQUERIDOS que arreglar, patrón." in out


def test_doctor_sin_el_script_avisa_en_vez_de_un_traceback(sin_scripts, monkeypatch, tmp_path,
                                                           capsys):
    """Instalación sin scripts/ (wheel, copia parcial): se explica qué falta."""
    monkeypatch.setattr(settings_mod, "REPO_ROOT", tmp_path)
    assert main_mod.main(["doctor"]) == 1
    salida = capsys.readouterr()
    assert "scripts/crotolamo_doctor.py" in salida.err
    assert "Traceback" not in salida.out + salida.err


def test_doctor_roto_por_dentro_no_se_disfraza_de_script_ausente(sin_scripts, monkeypatch,
                                                                 tmp_path):
    """Si el script ESTÁ pero no importa (le falta un módulo), el fallo real debe
    verse: no es "no encuentro el doctor". Y no queda a medio cargar en sys.modules."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "crotolamo_doctor.py").write_text(
        "import modulo_que_no_existe_en_ningun_lado\n", encoding="utf-8")
    monkeypatch.setattr(settings_mod, "REPO_ROOT", tmp_path)
    antes = set(sys.modules)
    with pytest.raises(ModuleNotFoundError, match="modulo_que_no_existe_en_ningun_lado"):
        main_mod.main(["doctor"])
    assert not {n for n in set(sys.modules) - antes if "doctor" in n}


def test_doctor_sigue_usando_el_modulo_importable(monkeypatch, tmp_path):
    """Desde la raíz del repo nada cambia: se usa `scripts.crotolamo_doctor` tal
    cual, sin cargarlo dos veces por su ruta."""
    from scripts import crotolamo_doctor

    monkeypatch.setattr(crotolamo_doctor, "run_doctor", lambda: 7)
    # La ruta de respaldo no debe ni mirarse: aquí no hay script que cargar.
    monkeypatch.setattr(settings_mod, "REPO_ROOT", tmp_path)
    assert main_mod.main(["doctor"]) == 7


# --- nivel de log que no existe ---------------------------------------------------

@pytest.fixture
def logging_sin_configurar(monkeypatch):
    """setup_logging como en un proceso recién arrancado. Devuelve los kwargs con
    los que se llamó a basicConfig (bajo pytest el raíz ya tiene handlers y el
    basicConfig real no haría nada: no se podría ver el nivel elegido)."""
    monkeypatch.setattr(logging_setup, "_CONFIGURED", False)
    monkeypatch.delenv("CROTOLAMO_LOG", raising=False)
    llamadas: list[dict] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: llamadas.append(kwargs))
    return llamadas


def _avisos(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name.startswith("crotolamo.") and r.levelno >= logging.WARNING]


# "basic_format": getattr(logging, "BASIC_FORMAT") existe y NO es un nivel; llegaba
# tal cual a basicConfig, que reventaba al arrancar con "Unknown level".
@pytest.mark.parametrize("valor", ["trace", "verbose", "10", "basic_format"])
def test_nivel_de_log_desconocido_por_env_avisa(logging_sin_configurar, monkeypatch, caplog,
                                                valor):
    monkeypatch.setenv("CROTOLAMO_LOG", valor)
    with caplog.at_level(logging.WARNING):
        logging_setup.setup_logging()
    (llamada,) = logging_sin_configurar
    assert llamada["level"] == logging.INFO
    # Antes: INFO sin una sola línea que dijera que lo pedido no se aplicó.
    (aviso,) = _avisos(caplog)
    assert valor.upper() in aviso and "INFO" in aviso


def test_nivel_de_log_numerico_en_el_toml_avisa(logging_sin_configurar, monkeypatch, tmp_path,
                                                caplog):
    """`level = 10` en [logging] (un número en vez del nombre) también caía a INFO."""
    falso = Settings(raw={"logging": {"level": 10}}, user="test", home=tmp_path)
    monkeypatch.setattr(settings_mod, "_SETTINGS", falso)
    with caplog.at_level(logging.WARNING):
        logging_setup.setup_logging()
    assert logging_sin_configurar[0]["level"] == logging.INFO
    (aviso,) = _avisos(caplog)
    assert "'10'" in aviso


@pytest.mark.parametrize("valor, esperado", [
    ("debug", logging.DEBUG),
    ("INFO", logging.INFO),
    ("Warning", logging.WARNING),
    ("error", logging.ERROR),
])
def test_niveles_validos_se_aplican_sin_aviso(logging_sin_configurar, monkeypatch, caplog,
                                              valor, esperado):
    monkeypatch.setenv("CROTOLAMO_LOG", valor)
    with caplog.at_level(logging.WARNING):
        logging_setup.setup_logging()
    assert logging_sin_configurar[0]["level"] == esperado
    assert _avisos(caplog) == []


def test_nivel_pasado_en_minusculas_por_argumento(logging_sin_configurar, caplog):
    """setup_logging("debug"): sin normalizar, getattr(logging, "debug") es la
    FUNCIÓN logging.debug y basicConfig reventaba con "Level not an integer"."""
    with caplog.at_level(logging.WARNING):
        logging_setup.setup_logging("debug")
    assert logging_sin_configurar[0]["level"] == logging.DEBUG
    assert _avisos(caplog) == []


def test_setup_logging_sigue_configurando_una_sola_vez(logging_sin_configurar, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_LOG", "debug")
    logging_setup.setup_logging()
    logging_setup.setup_logging()
    assert len(logging_sin_configurar) == 1


# --- /historial: nombres de las tools pedidas ---------------------------------------

def _tool_de_prueba(name: str, salida: str):
    from crotolamo.tools.base import Tool

    return Tool(name=name, func=lambda: salida, description=name,
                parameters={"type": "object", "properties": {}, "required": []})


def test_historial_muestra_las_tools_pedidas_con_la_forma_que_guarda_el_agente(capsys):
    """ToolAgent guarda los tool_calls como {"function": {"name": ...}} (formato
    Ollama; GLM se normaliza a lo mismo). /historial leía c["name"] del nivel de
    arriba, que nunca existe, y pintaba "[pide tools: ?]" tras cualquier tool."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.llm import ChatResponse
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools.base import Registry
    from interfaces import shell

    reg = Registry()
    reg.register(_tool_de_prueba("abrir_cosa", "Abrí la cosa, patrón."))
    reg.register(_tool_de_prueba("mirar_cosa", "La cosa está bien, patrón."))

    def pide(*nombres: str) -> ChatResponse:
        return ChatResponse(
            content="", tool_calls=[{"name": n, "arguments": {}} for n in nombres],
            raw_message={"tool_calls": [{"function": {"name": n, "arguments": {}}}
                                        for n in nombres]},
        )

    class _Guion:
        model = "x"

        def __init__(self):
            # Nativo (raw_message) y luego por texto (coerción): las dos formas reales.
            self._r = [pide("abrir_cosa", "mirar_cosa"),
                       ChatResponse(content='{"name": "mirar_cosa", "arguments": {}}'),
                       ChatResponse(content="Hecho, patrón.")]

        def chat(self, messages, tools=None):
            return self._r.pop(0)

    conv = Conversation("SYS")
    agent = ToolAgent(_Guion(), conv, registry=reg, guard=Guard(allowed_roots=[]),
                      fastpath=False, direct_tools=set(), post_hooks=[])
    assert agent.handle_turn("abre y mira la cosa") == "Hecho, patrón."

    shell._show_history(conv)
    out = capsys.readouterr().out
    assert "[pide tools: abrir_cosa, mirar_cosa]" in out
    assert "[pide tools: mirar_cosa]" in out
    assert "?" not in out


# --- build_agent con GLM como cerebro: las cadenas de tools no se cortan ------------

def test_build_agent_con_glm_no_corta_una_cadena_en_una_tool_con_flag_direct(monkeypatch,
                                                                           tmp_path):
    """Cableado REAL de build_agent con GLM: "si firefox está abierto, tráelo al
    frente" pide app_status (Tool.direct=True) y luego focus_window. Con el flag
    colándose bajo direct_tools=set(), la salida cruda de app_status era la
    respuesta final y focus_window nunca corría. (El arreglo vive en
    ToolAgent._is_direct; esto fija que el shell sigue pasando el interruptor.)"""
    from crotolamo.core import hooks
    from crotolamo.core import memoria as memoria_mod
    from crotolamo.core.fallback import FallbackLLM
    from crotolamo.core.llm import ChatResponse
    from crotolamo.tools import windows
    from interfaces import shell

    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": "glm"}}, user="test", home=tmp_path))
    # Sin memoria semántica ni la base de hechos real del patrón.
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))

    enfocadas: list[list[str]] = []
    monkeypatch.setattr(windows, "_clients", lambda: [
        {"class": "firefox", "title": "YouTube", "workspace": {"id": 2}, "address": "0xabc"},
    ])

    def dispatch(args):
        enfocadas.append(args)
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(windows, "_dispatch", dispatch)

    def pide(name: str) -> ChatResponse:
        args = {"name": "firefox"}
        return ChatResponse(content="", tool_calls=[{"name": name, "arguments": args}],
                            raw_message={"tool_calls": [{"function": {"name": name,
                                                                      "arguments": args}}]})

    class _GLM:
        model = "glm"

        def __init__(self, r):
            self.r = list(r)
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            return self.r.pop(0)

    glm = _GLM([pide("app_status"), pide("focus_window"),
                ChatResponse(content="Listo, patrón: Firefox al frente.")])
    monkeypatch.setattr(shell, "build_llm", lambda s: FallbackLLM(glm, _GLM([])))

    agent, _ = shell.build_agent()
    reply = agent.handle_turn("si firefox está abierto, tráelo al frente")
    assert enfocadas == [["focuswindow", "address:0xabc"]]
    assert reply == "Listo, patrón: Firefox al frente."
    assert glm.calls == 3


# --- "salir" del shell justo tras contar algo: la extracción pendiente se termina ---

# run_shell REAL (build_agent con memoria semántica y su after-turn hook) en un
# proceso aparte: se cuenta un hecho y se teclea "salir" al instante. El hilo de
# extracción es daemon; si el shell saliera saltándose atexit (os._exit) o el
# vaciado de la Memoria dejara de cubrir este camino, el hecho se perdería.
_SONDA_SHELL_SALIR = r"""
import builtins, sys, time
from pathlib import Path
from crotolamo import settings as settings_mod
from crotolamo.settings import Settings
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.llm import ChatResponse
from interfaces import shell

marca = Path(sys.argv[1])
settings_mod._SETTINGS = Settings(raw={"llm": {"backend": "ollama"}}, user="test",
                                  home=marca.parent)


class Backend:  # doble de mem0: extraer tarda lo que la llamada al LLM de extracción
    def buscar(self, query, top_k, umbral):
        return []

    def extraer(self, usuario, respuesta):
        time.sleep(0.5)
        marca.write_text(usuario, encoding="utf-8")
        return [usuario]


class LLM:
    model = "fake"

    def chat(self, messages, tools=None):
        return ChatResponse(content="Anotado, patrón.")


memoria_mod.mem0_instalado = lambda: True
memoria_mod.set_memoria(memoria_mod.Memoria(memoria_mod.MemoriaConfig(enabled=True),
                                            backend=Backend()))
shell.build_llm = lambda s: LLM()
entradas = iter(["mi hermana Ana vive en Monterrey", "salir"])
builtins.input = lambda prompt="": next(entradas)
raise SystemExit(shell.run_shell([]))
"""


def test_salir_del_shell_no_pierde_la_extraccion_del_ultimo_turno(tmp_path):
    """Antes: run_shell volvía con "salir", el intérprete mataba el hilo daemon a
    media extracción y el hecho del último turno nunca se guardaba."""
    import os
    import subprocess
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "CROTOLAMO_LLM_BACKEND"}
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _SONDA_SHELL_SALIR, str(tmp_path / "marca")],
        cwd=raiz, env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Anotado, patrón." in proc.stdout
    assert "Crotolamo apagado, patrón." in proc.stdout
    marca = tmp_path / "marca"
    assert marca.exists(), "la extracción del último turno se perdió al salir del shell"
    assert marca.read_text(encoding="utf-8") == "mi hermana Ana vive en Monterrey"


# --- `python interfaces/shell.py --stream`: el script respeta sus argumentos --------

def test_shell_como_script_respeta_sus_argumentos_y_configura_el_logging(
        logging_sin_configurar, monkeypatch, capsys, tmp_path):
    """El bloque __main__ llamaba run_shell() a secas: `python interfaces/shell.py
    --stream` arrancaba SIN streaming, y como nadie llamaba a setup_logging() los
    INFO ("motor: ...") se perdían y los WARNING salían sin formato."""
    import builtins
    import runpy
    from pathlib import Path

    from crotolamo.core import engine, hooks
    from crotolamo.core import memoria as memoria_mod
    from crotolamo.core.llm import ChatResponse

    class _LLM:
        model = "fake"

        def __init__(self):
            self.streams = 0

        def chat(self, messages, tools=None):
            return ChatResponse(content="Aquí ando, patrón.")

        def chat_stream(self, messages, tools=None, on_token=None):
            self.streams += 1
            on_token("Aquí ando, patrón.")
            return ChatResponse(content="Aquí ando, patrón.")

    llm = _LLM()
    monkeypatch.delenv("CROTOLAMO_LLM_BACKEND", raising=False)
    monkeypatch.setattr(settings_mod, "_SETTINGS",
                        Settings(raw={"llm": {"backend": "ollama"}}, user="test", home=tmp_path))
    monkeypatch.setattr(memoria_mod, "get_memoria",
                        lambda: type("M", (), {"enabled": False})())
    monkeypatch.setattr(hooks, "make_facts_prehook", lambda: (lambda text: text))
    monkeypatch.setattr(hooks, "make_pinned_facts_prehook", lambda conv: (lambda text: text))
    # El script se ejecuta como módulo nuevo: el doble entra por donde lo importa.
    monkeypatch.setattr(engine, "build_llm", lambda s: llm)
    monkeypatch.setattr(sys, "argv", ["interfaces/shell.py", "--stream"])
    entradas = iter(["hola", "salir"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(entradas))

    script = Path(__file__).resolve().parents[1] / "interfaces" / "shell.py"
    with pytest.raises(SystemExit) as fin:
        runpy.run_path(str(script), run_name="__main__")
    assert fin.value.code == 0
    out = capsys.readouterr().out
    assert "Crotolamo Shell (streaming)." in out
    assert llm.streams == 1
    assert "Crotolamo > Aquí ando, patrón." in out
    # El logging se configura una vez, igual que por `python -m crotolamo shell`.
    assert len(logging_sin_configurar) == 1
