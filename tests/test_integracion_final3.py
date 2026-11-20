"""Integración de la revisión final (tanda 3): cambios pedidos en archivos de otro grupo.

Cada bloque cita el pedido (idx de la lista de integración y el id del hallazgo)
y prueba el comportamiento que el arreglo cambia.
"""

from __future__ import annotations

import builtins
import errno
import logging
import os
import pty
import select
import signal
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crotolamo import logging_setup
from interfaces import shell

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# idx 2 (final-x5reposo-fwlog): las líneas INFO de faster_whisper no llegan al journal
# ---------------------------------------------------------------------------
# faster_whisper escribe «Processing audio with duration …» en INFO por CADA
# transcripción, con el logger `faster_whisper`. setup_logging() deja el raíz en INFO
# y ese logger hereda: con el wake difuso y la tele puesta son unas 12 pasadas por
# minuto, líneas que crecen con el audio ambiente y no con lo que pasa.

class _Journal(logging.Handler):
    """Hace de stderr del servicio: lo que systemd lleva al journal."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lineas: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lineas.append(record)


@pytest.fixture
def journal(monkeypatch):
    """setup_logging como en un proceso recién arrancado, con un handler en el raíz
    que recoge lo que saldría por stderr. Deja los niveles como estaban."""
    raiz, whisper = logging.getLogger(), logging.getLogger("faster_whisper")
    nivel_raiz, nivel_whisper = raiz.level, whisper.level
    whisper.setLevel(logging.NOTSET)        # como lo deja la librería: hereda del raíz
    monkeypatch.setattr(logging_setup, "_CONFIGURED", False)
    monkeypatch.delenv("CROTOLAMO_LOG", raising=False)
    # Bajo pytest el raíz ya tiene handlers y el basicConfig real no haría nada: aquí
    # hace lo único que importa de él, fijar el nivel del raíz.
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: raiz.setLevel(kw["level"]))
    handler = _Journal()
    raiz.addHandler(handler)
    yield handler.lineas
    raiz.removeHandler(handler)
    raiz.setLevel(nivel_raiz)
    whisper.setLevel(nivel_whisper)


def _una_pasada_de_whisper() -> None:
    """Lo que faster_whisper/transcribe.py escribe por cada transcripción, con su
    mismo logger (faster_whisper.utils.get_logger)."""
    logging.getLogger("faster_whisper").info("Processing audio with duration %s", "00:04.512")


def _de_whisper(journal: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in journal if r.name == "faster_whisper"]


# None: el nivel de la config versionada (INFO). "trace": un nivel que no existe cae
# a INFO, y ahí tampoco deben salir.
@pytest.mark.parametrize("nivel", [None, "INFO", "info", "trace"])
def test_las_pasadas_de_whisper_no_escriben_en_el_journal(journal, monkeypatch, nivel):
    if nivel is not None:
        monkeypatch.setenv("CROTOLAMO_LOG", nivel)
    logging_setup.setup_logging()
    for _ in range(12):                     # un minuto de wake difuso con la tele
        _una_pasada_de_whisper()
    # Antes: 12 líneas «faster_whisper INFO: Processing audio with duration …».
    assert _de_whisper(journal) == []


def test_los_avisos_de_faster_whisper_siguen_saliendo(journal):
    """Solo se calla el ruido por pasada: un WARNING de la librería sí interesa."""
    logging_setup.setup_logging()
    logging.getLogger("faster_whisper").warning("el modelo no cargó")
    assert _de_whisper(journal) == ["el modelo no cargó"]


def test_con_debug_las_pasadas_de_whisper_se_conservan(journal, monkeypatch):
    """CROTOLAMO_LOG=DEBUG es para diagnosticar: ahí se quiere ver cada pasada."""
    monkeypatch.setenv("CROTOLAMO_LOG", "debug")
    logging_setup.setup_logging()
    _una_pasada_de_whisper()
    assert _de_whisper(journal) == ["Processing audio with duration 00:04.512"]


def test_las_lineas_info_propias_no_se_tocan(journal):
    """El arreglo es para el logger de terceros, no un raíz en WARNING."""
    logging_setup.setup_logging()
    logging_setup.get_logger("voice.stt").info("Cargando Whisper 'tiny'")
    assert [r.name for r in journal if r.levelno == logging.INFO] == ["crotolamo.voice.stt"]


@pytest.mark.parametrize("nivel", ["ERROR", "CRITICAL"])
def test_callar_a_whisper_no_le_abre_la_puerta_por_encima_del_nivel_pedido(
    journal, monkeypatch, nivel,
):
    """Con el raíz en ERROR un WARNING de faster_whisper no salía (hereda), y así
    debe seguir: fijarle WARNING al logger también ahí lo haría MÁS hablador que lo
    que el patrón pidió."""
    monkeypatch.setenv("CROTOLAMO_LOG", nivel)
    logging_setup.setup_logging()
    logging.getLogger("faster_whisper").warning("aviso de la librería")
    _una_pasada_de_whisper()
    assert _de_whisper(journal) == []
    logging.getLogger("faster_whisper").critical("esto sí")
    assert _de_whisper(journal) == ["esto sí"]


# ---------------------------------------------------------------------------
# idx 0 (final-x3vida-shell-signals): SIGHUP / SIGTERM en el shell pasan por la
# salida normal del intérprete (atexit)
# ---------------------------------------------------------------------------
# Cerrar la ventana de la terminal manda SIGHUP; su acción por defecto mataba el
# proceso sin atexit: la extracción del turno recién dicho se cortaba a medias y un
# server MCP que no sale al ver el EOF (en su propia sesión) quedaba huérfano.

# run_shell REAL en un proceso aparte: se cuenta un hecho y el patrón no escribe
# nada más. La extracción (hilo daemon) tarda 1 s; la marca es su resultado.
_SHELL = r'''
import builtins, os, signal, sys, time
from pathlib import Path

from crotolamo import settings as settings_mod
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.llm import ChatResponse
from crotolamo.settings import Settings
from interfaces import shell

marca = Path(sys.argv[1])
settings_mod._SETTINGS = Settings(raw={"llm": {"backend": "ollama"}}, user="test",
                                  home=marca.parent)
signal.signal(signal.SIGHUP, signal.SIG_DFL)  # aunque pytest corra bajo nohup


class Backend:  # doble de mem0: extraer tarda lo que la llamada al LLM de extracción
    def buscar(self, query, top_k, umbral):
        return []

    def extraer(self, usuario, respuesta):
        time.sleep(1.0)
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
'''

# El patrón deja de escribir: input() no vuelve (ni ve cerrarse nada).
_SHELL_EN_EL_PROMPT = _SHELL + r'''
entradas = iter(["mi hermana Ana vive en Monterrey"])


def entrada(prompt=""):
    try:
        return next(entradas)
    except StopIteration:
        print("en el prompt", flush=True)
        time.sleep(60)  # el patrón no escribe nada más: cierra la ventana


builtins.input = entrada
raise SystemExit(shell.run_shell([]))
'''


def _shell_que_recibe(tmp_path: Path, senales: list[tuple[float, int]]) -> tuple[int, Path, str]:
    """Arranca el shell, espera a que esté en el prompt (con la extracción del
    turno en curso) y le manda `senales`: (espera previa en s, señal). Devuelve el
    código de salida, la marca de la extracción y el stderr."""
    marca = tmp_path / "extraido"
    env = {k: v for k, v in os.environ.items() if k != "CROTOLAMO_LLM_BACKEND"}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(RAIZ))
    hijo = subprocess.Popen([sys.executable, "-c", _SHELL_EN_EL_PROMPT, str(marca)], cwd=RAIZ,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert hijo.stdout is not None
        for linea in hijo.stdout:
            if "en el prompt" in linea:
                break
        for espera, senal in senales:
            time.sleep(espera)
            hijo.send_signal(senal)
        _, errores = hijo.communicate(timeout=20)
    finally:
        if hijo.poll() is None:
            hijo.kill()
            hijo.wait()
    return hijo.returncode, marca, errores


@pytest.mark.parametrize("senal", [signal.SIGHUP, signal.SIGTERM])
def test_cerrar_la_terminal_del_shell_no_pierde_la_extraccion_del_ultimo_turno(tmp_path, senal):
    """`kitty --hold -e launch.sh`, opción 2, y el patrón cierra la ventana: SIGHUP.
    Sin manejador el proceso moría en el acto (rc = -señal), sin atexit."""
    rc, marca, errores = _shell_que_recibe(tmp_path, [(0.0, senal)])
    assert rc == 128 + senal, f"rc={rc}: la señal mató el shell sin su salida normal\n{errores}"
    assert marca.exists(), "la extracción del último turno se perdió"
    assert marca.read_text(encoding="utf-8") == "mi hermana Ana vive en Monterrey"


# Cerrar la terminal no siempre manda UN SIGHUP: medido con un pty, al proceso en
# primer plano de un bash interactivo le llegan dos con 0.2 ms de diferencia (el
# que bash reenvía a sus trabajos y el del kernel cuando muere el líder de la
# sesión), y lo mismo si el emulador hace además killpg al grupo. 0.3 s: la segunda
# llega con la salida ya en atexit, esperando la extracción (1 s).
@pytest.mark.parametrize("entre_las_dos", [0.0, 0.3])
def test_la_segunda_senal_de_la_rafaga_no_corta_la_salida_del_shell(tmp_path, entre_las_dos):
    """Con los manejadores repuestos (o vueltos a disparar) nada más pedir la
    salida, el segundo SIGHUP mataba el proceso a media extracción: el arreglo solo
    servía para la señal suelta de un `kill`."""
    rc, marca, errores = _shell_que_recibe(
        tmp_path, [(0.0, signal.SIGHUP), (entre_las_dos, signal.SIGHUP)])
    assert rc == 128 + signal.SIGHUP, f"rc={rc}: la segunda señal cortó la salida\n{errores}"
    assert marca.exists(), "la extracción del último turno se perdió"


# La ventana de verdad: un pty, con el input() REAL leyendo de él. Al cerrarse, lo
# primero que ve el shell no es el SIGHUP sino el fin de la entrada (EIO): ese
# SIGHUP es para el líder de la sesión (el bash de launch.sh), y al shell le llega
# después, cuando el líder muere, con el shell ya saliendo por su camino de EOF.
_SHELL_EN_LA_TERMINAL = _SHELL + r'''
Path(str(marca) + ".pid").write_text(str(os.getpid()), encoding="ascii")
raise SystemExit(shell.run_shell([]))
'''

# Hace del `bash launch.sh` de `kitty -e`: líder de la sesión de la terminal, con el
# shell de hijo en su grupo. Cuando la terminal cuelga tarda argv[1] s en morirse, y
# es al morir cuando el kernel manda el SIGHUP al grupo en primer plano.
_LIDER_DE_LA_TERMINAL = r'''
import fcntl, os, signal, subprocess, sys, termios, time

fcntl.ioctl(0, termios.TIOCSCTTY, 0)  # el pty, terminal de control de esta sesión


def colgaron(signum, frame):
    time.sleep(float(sys.argv[1]))
    os._exit(0)


signal.signal(signal.SIGHUP, colgaron)
subprocess.Popen([sys.executable, "-c", sys.argv[2], *sys.argv[3:]]).wait()
'''


def _vivo(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


def _esperar(condicion, plazo: float) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if condicion():
            return True
        time.sleep(0.02)
    return bool(condicion())


def test_cerrar_la_ventana_con_el_shell_en_el_prompt_no_pierde_la_extraccion(tmp_path):
    """El caso del título, entero: el patrón cuenta algo y cierra la ventana. El
    shell veía el fin de la entrada, salía por su camino de EOF (el "apagado" ya no
    se puede escribir: EIO) con los manejadores repuestos, y el SIGHUP que llega
    0.3 s después, con la salida esperando la extracción en atexit, lo mataba."""
    marca = tmp_path / "extraido"
    pidfile = tmp_path / "extraido.pid"
    env = {k: v for k, v in os.environ.items() if k != "CROTOLAMO_LLM_BACKEND"}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(RAIZ))
    maestro, esclavo = pty.openpty()
    lider = subprocess.Popen(
        [sys.executable, "-c", _LIDER_DE_LA_TERMINAL, "0.3", _SHELL_EN_LA_TERMINAL, str(marca)],
        stdin=esclavo, stdout=esclavo, stderr=esclavo, start_new_session=True, cwd=RAIZ, env=env)
    os.close(esclavo)
    pantalla = b""

    def hasta(n_prompts: int) -> None:
        nonlocal pantalla
        fin = time.monotonic() + 30
        while pantalla.count("Patrón > ".encode()) < n_prompts:
            listo, _, _ = select.select([maestro], [], [], max(0.0, fin - time.monotonic()))
            assert listo, f"el shell no llegó al prompt {n_prompts}: {pantalla!r}"
            try:
                pantalla += os.read(maestro, 4096)
            except OSError:                      # EIO: ya no queda nadie en el pty
                raise AssertionError(f"el shell murió antes del prompt: {pantalla!r}") from None

    try:
        hasta(1)
        os.write(maestro, "mi hermana Ana vive en Monterrey\n".encode())
        hasta(2)                                 # respondió y espera otra orden
        assert "Anotado, patrón.".encode() in pantalla
        os.close(maestro)                        # el patrón cierra la ventana
        maestro = -1
        lider.wait(timeout=20)
        pid = int(pidfile.read_text(encoding="ascii"))
        assert _esperar(lambda: not _vivo(pid), 20), "el shell no terminó"
    finally:
        if maestro >= 0:
            os.close(maestro)
        if lider.poll() is None:
            lider.kill()
            lider.wait()
        if pidfile.exists() and _vivo(int(pidfile.read_text(encoding="ascii"))):
            os.kill(int(pidfile.read_text(encoding="ascii")), signal.SIGKILL)
    assert marca.exists(), "la extracción del último turno se perdió al cerrar la ventana"
    assert marca.read_text(encoding="utf-8") == "mi hermana Ana vive en Monterrey"


_SENALES = (signal.SIGTERM, signal.SIGHUP)


def _manejadores() -> dict[int, Any]:
    return {s: signal.getsignal(s) for s in _SENALES}


@pytest.fixture
def senales_del_proceso():
    """Los manejadores de pytest, tal cual estaban, pase lo que pase en el test.
    SIGHUP en su acción por defecto al empezar (bajo nohup llega ignorada)."""
    antes = _manejadores()
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    yield
    for sig, previo in antes.items():
        signal.signal(sig, previo)


def _shell_de_mentira(monkeypatch, entradas: list[Any]) -> list[dict[int, Any]]:
    """run_shell sin agente de verdad. Devuelve los manejadores vistos DENTRO del
    shell: al armar el agente y en cada input()."""
    vistos: list[dict[int, Any]] = []
    cola = iter(entradas)

    def build_agent():
        vistos.append(_manejadores())
        return object(), object()

    def entrada(prompt: str = "") -> str:
        vistos.append(_manejadores())
        siguiente = next(cola)
        if isinstance(siguiente, BaseException):
            raise siguiente
        return siguiente

    monkeypatch.setattr(shell, "build_agent", build_agent)
    monkeypatch.setattr(builtins, "input", entrada)
    return vistos


@pytest.mark.parametrize("entradas", [["salir"], [EOFError()], [KeyboardInterrupt()]])
def test_run_shell_repone_los_manejadores_al_volver(monkeypatch, senales_del_proceso, entradas):
    """Varios tests llaman run_shell dentro de pytest: sin reponerlos, el proceso
    se quedaba con SIGTERM/SIGHUP -> SystemExit para el resto de la suite."""
    vistos = _shell_de_mentira(monkeypatch, entradas)
    antes = _manejadores()
    assert shell.run_shell([]) == 0
    # Puestos ANTES de armar el agente (ahí arrancan los servers MCP) y en el prompt.
    assert len(vistos) == 2
    for dentro in vistos:
        assert all(callable(dentro[s]) and dentro[s] != antes[s] for s in _SENALES), dentro
    assert _manejadores() == antes


def test_run_shell_repone_los_manejadores_si_el_arranque_revienta(monkeypatch, senales_del_proceso):
    antes = _manejadores()

    def build_agent():
        assert callable(signal.getsignal(signal.SIGTERM))
        raise RuntimeError("config rota")

    monkeypatch.setattr(shell, "build_agent", build_agent)
    with pytest.raises(RuntimeError, match="config rota"):
        shell.run_shell([])
    assert _manejadores() == antes


def test_run_shell_no_pisa_un_sighup_ignorado(monkeypatch, senales_del_proceso):
    """`nohup crotolamo shell`: SIGHUP llega ignorada, que es pedir sobrevivir al
    cierre de la terminal. No se pisa (como el listener); SIGTERM sí se atiende."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    vistos = _shell_de_mentira(monkeypatch, ["salir"])
    antes = _manejadores()
    assert shell.run_shell([]) == 0
    for dentro in vistos:
        assert dentro[signal.SIGHUP] is signal.SIG_IGN
        assert callable(dentro[signal.SIGTERM])
    assert _manejadores() == antes


def test_run_shell_fuera_del_hilo_principal_no_toca_las_senales(monkeypatch, senales_del_proceso):
    """signal.signal solo vale en el hilo principal (ValueError en cualquier otro):
    un run_shell lanzado desde un hilo corre sin manejadores, como antes, y no deja
    a medias los del proceso."""
    vistos = _shell_de_mentira(monkeypatch, ["salir"])
    antes = _manejadores()
    out: list[Any] = []

    def correr() -> None:
        try:
            out.append(shell.run_shell([]))
        except BaseException as error:  # noqa: BLE001 - al hilo del test
            out.append(error)

    hilo = threading.Thread(target=correr)
    hilo.start()
    hilo.join(10)
    assert out == [0]
    assert vistos == [antes, antes]
    assert _manejadores() == antes


def test_la_senal_sale_por_systemexit_y_la_repetida_se_ignora(monkeypatch, senales_del_proceso):
    """El manejador pide la salida normal del intérprete (SystemExit con 128+señal,
    lo que hace correr atexit) UNA vez: la repetición de la ráfaga, que llega con
    esa salida ya en marcha, no hace nada. Y tras una salida por señal los
    manejadores NO se reponen: reponer la acción por defecto dejaba a la segunda
    señal matar el proceso en pleno atexit."""
    vistos = _shell_de_mentira(monkeypatch, ["salir"])
    dentro: dict[int, Any] = {}

    def entrada(prompt: str = "") -> str:
        dentro.update(_manejadores())
        dentro[signal.SIGHUP](signal.SIGHUP, None)      # llega el SIGHUP en el prompt
        raise AssertionError("el manejador no pidió salir")

    monkeypatch.setattr(builtins, "input", entrada)
    with pytest.raises(SystemExit) as salida:
        shell.run_shell([])
    assert salida.value.code == 128 + signal.SIGHUP
    assert vistos                                        # el agente se armó
    assert _manejadores() == dentro                      # siguen puestos: la salida sigue
    for sig in _SENALES:
        assert dentro[sig](sig, None) is None            # la repetida (o la otra): nada


def test_con_la_terminal_colgada_los_manejadores_se_quedan(monkeypatch, senales_del_proceso):
    """La ventana se cerró y el shell salió por su camino de EOF antes de ver el
    SIGHUP: viene detrás. Reponer ahí la acción por defecto lo dejaba matar la
    salida; los manejadores se quedan, y ya no piden salir otra vez."""
    _shell_de_mentira(monkeypatch, [EOFError()])
    monkeypatch.setattr(shell, "_terminal_colgada", lambda: True)
    antes = _manejadores()
    assert shell.run_shell([]) == 0
    despues = _manejadores()
    for sig in _SENALES:
        assert callable(despues[sig]) and despues[sig] != antes[sig]
        assert despues[sig](sig, None) is None           # el SIGHUP rezagado: nada


@pytest.mark.parametrize("error, colgada", [
    (termios.error(errno.EIO, "Input/output error"), True),                 # tty colgado
    (termios.error(errno.ENOTTY, "Inappropriate ioctl for device"), False),  # tubería
    (termios.error(errno.EBADF, "Bad file descriptor"), False),              # stdin cerrado
    (None, False),                                                           # tty vivo
])
def test_terminal_colgada_es_solo_el_eio_de_un_tty(monkeypatch, error, colgada):
    def tcgetattr(fd):
        assert fd == 0
        if error is not None:
            raise error
        return []

    monkeypatch.setattr(termios, "tcgetattr", tcgetattr)
    assert shell._terminal_colgada() is colgada


def test_run_shell_no_pisa_un_manejador_que_no_podria_reponer(monkeypatch, senales_del_proceso):
    """getsignal da None cuando el manejador lo puso código C, y signal.signal no
    acepta None de vuelta: no habría cómo reponerlo. Esa señal no se toca."""
    getsignal, poner = signal.getsignal, signal.signal
    puestas: list[int] = []

    def signal_espia(sig, manejador):
        puestas.append(sig)
        return poner(sig, manejador)

    _shell_de_mentira(monkeypatch, ["salir"])
    monkeypatch.setattr(signal, "getsignal",
                        lambda sig: None if sig == signal.SIGTERM else getsignal(sig))
    monkeypatch.setattr(signal, "signal", signal_espia)
    assert shell.run_shell([]) == 0
    assert puestas == [signal.SIGHUP, signal.SIGHUP]     # puesta y repuesta; SIGTERM, ni tocarla
