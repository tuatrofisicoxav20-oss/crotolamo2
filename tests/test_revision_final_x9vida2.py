"""Revisión final x9 (post-arreglo del apagado): lo que 09aeedb y f460fc0 dejaron a medias.

- close() cortado por Ctrl-C / SystemExit (el SIGTERM/SIGHUP del shell): el cliente
  quedaba cerrado a medias, nadie lo seguía y el server (colgado) sobrevivía.
- El shell: la salida NO acaba cuando run_shell vuelve. Con "salir" el vaciado de la
  Memoria y el cierre de los servers MCP corren después, en atexit, con "Crotolamo
  apagado, patrón." ya en pantalla: cerrar la ventana ahí (SIGHUP con la acción por
  defecto repuesta) mataba el proceso y la extracción del último turno se perdía.
- Los hijos de los tests de señales esperaban la señal en UNA llamada bloqueante
  (signal.pause(), time.sleep(60)): la que llega justo antes de la syscall ya no la
  despierta y el test agotaba su plazo (1 de 3000 bajo carga).
"""

from __future__ import annotations

import builtins
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crotolamo.core import router
from crotolamo.mcp import bridge
from crotolamo.mcp.client import StdioMCPClient
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry
from interfaces import shell
from tests.test_integracion_final3 import _SHELL, _SHELL_EN_EL_PROMPT

RAIZ = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).parent / "fake_mcp_server.py"


def _vivo(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


def _esperar(cond, plazo: float = 5.0) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.02)
    return bool(cond())


# ---------------------------------------------------------------------------
# close() cortado a medias: el server no queda vivo
# ---------------------------------------------------------------------------
# El server que más cuesta cerrar: un wrapper (`sh -c`) cuyo nieto ignora el EOF
# (FAKE_MCP_EOF_DELAY) y el SIGTERM. Solo el SIGKILL al grupo, el último paso de
# close(), acaba con él. Deja su pid en un archivo.
_TERCO = """\
import os, runpy, signal, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
fake = sys.argv[2]
sys.argv = [fake]
runpy.run_path(fake, run_name="__main__")
"""


def _terco(tmp_path: Path) -> tuple[list[str], dict[str, str], Path]:
    script = tmp_path / "terco.py"
    script.write_text(_TERCO, encoding="utf-8")
    pidfile = tmp_path / "terco.pid"
    comando = ["sh", "-c", f"{sys.executable} {script} {pidfile} {FAKE}; true"]
    return comando, {"FAKE_MCP_EOF_DELAY": "600"}, pidfile


def _recoger(client: StdioMCPClient) -> None:
    """Lo que un close() cortado ya no hace (quien lo corta se va del proceso):
    recoger al hijo y cerrar sus tuberías. Aquí pytest sigue, y sin esto el
    recolector avisa (ResourceWarning) cuando llega al Popen."""
    proc = client._proc
    assert proc is not None
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    for hilo in (client._reader, client._stderr_thread):
        if hilo is not None:
            hilo.join(10)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except OSError:
            pass


@pytest.mark.parametrize("corte", [KeyboardInterrupt, SystemExit])
def test_un_close_cortado_no_deja_vivo_al_server(tmp_path, monkeypatch, corte):
    """El hilo principal retira un server colgado (_on_timeout -> _retire -> close)
    y en la gracia del EOF llega Ctrl-C, o el SIGTERM/SIGHUP del shell (SystemExit):
    close() ya no se puede repetir (_closed) y quien lo llamó suelta al cliente."""
    comando, env, pidfile = _terco(tmp_path)
    client = StdioMCPClient("terco", comando, env)
    client.start()
    client.initialize(timeout=10)
    pid = int(pidfile.read_text())

    def cortado(proc, timeout):
        raise corte()

    monkeypatch.setattr(client, "_wait_exit", cortado)
    try:
        with pytest.raises(corte):
            client.close()
        # Antes: ni un SIGTERM había salido aún y el nieto seguía vivo sin dueño.
        assert _esperar(lambda: not _vivo(pid), 3.0), "el server quedó huérfano"
    finally:
        if _vivo(pid):
            os.kill(pid, signal.SIGKILL)
        _recoger(client)


def test_el_sigterm_del_shell_a_medio_retiro_no_deja_vivo_al_server(tmp_path):
    """El caso entero, con la señal de verdad: el shell (hilo principal) retira un
    server colgado tras su segundo timeout y, en la gracia del EOF, llega el
    SIGTERM, que el manejador de run_shell convierte en SystemExit DENTRO de
    close(). _retire lo saca de _CLOSING y close_all(), en atexit, ya no ve nada."""
    bridge.close_all()
    comando, env, pidfile = _terco(tmp_path)
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 10,
           "servers": {"terco": {"command": comando, "env": env}}}
    bridge.register_mcp_tools(Registry(), Settings(raw={"mcp": raw}, user="t", home=Path("/tmp")))
    state = bridge._STATE["terco"]
    pid = int(pidfile.read_text())

    acabado = threading.Event()                      # la señal rezagada ya no hace nada

    def salir(signum: int, _frame: Any) -> None:     # como el de run_shell
        if not acabado.is_set():
            raise SystemExit(128 + signum)

    principal = threading.get_ident()

    def senal_a_medio_cierre() -> None:
        if _esperar(lambda: state.client in bridge._CLOSING):
            time.sleep(0.3)                          # en plena gracia del EOF (1 s)
            signal.pthread_kill(principal, signal.SIGTERM)

    previo = signal.signal(signal.SIGTERM, salir)
    hilo = threading.Thread(target=senal_a_medio_cierre, daemon=True)
    try:
        hilo.start()
        with pytest.raises(SystemExit):
            bridge._drop(state, "prueba")
        assert bridge._CLOSING == set() and bridge.connected_servers() == []
        bridge.close_all()                           # lo que haría atexit: nada que cerrar
        assert _esperar(lambda: not _vivo(pid), 3.0), "el server sobrevivió a la salida"
    finally:
        # El manejador de pytest vuelve cuando el hilo ya no puede mandar la señal.
        acabado.set()
        hilo.join(10)
        signal.signal(signal.SIGTERM, previo)
        if _vivo(pid):
            os.kill(pid, signal.SIGKILL)
        _recoger(state.client)
        for name in router.dynamic_groups():
            router.unregister_group(name)


# ---------------------------------------------------------------------------
# El shell: cerrar la ventana DESPUÉS de "salir" tampoco corta la salida
# ---------------------------------------------------------------------------
# `python interfaces/shell.py`: el bloque __main__ de verdad, que sale del proceso
# en cuanto run_shell vuelve. El patrón cuenta algo y escribe "salir". El script
# corre como módulo nuevo: el doble del LLM entra por donde lo importa. Sin esperas
# a ojo: el hijo avisa cuando su salida ya está en atexit, y la extracción del
# turno dura hasta que el test, con la señal ya mandada, le deja seguir.
_SHELL_QUE_SALE = _SHELL + r"""
import atexit, runpy
from crotolamo.core import engine

sigue = Path(str(marca) + ".sigue")


class BackendQueEspera(Backend):
    def extraer(self, usuario, respuesta):
        fin = time.monotonic() + 4.0  # menos que el plazo del vaciado al salir (5 s)
        while not sigue.exists() and time.monotonic() < fin:
            time.sleep(0.02)
        marca.write_text(usuario, encoding="utf-8")
        return [usuario]


memoria_mod.set_memoria(memoria_mod.Memoria(memoria_mod.MemoriaConfig(enabled=True),
                                            backend=BackendQueEspera()))
engine.build_llm = lambda s: LLM()
entradas = iter(["mi hermana Ana vive en Monterrey", "salir"])


def entrada(prompt=""):
    orden = next(entradas)
    if orden == "salir":
        # El último atexit apuntado corre el primero: avisa de que run_shell ya
        # volvió y lo que queda de salida (el vaciado de la Memoria) empieza.
        atexit.register(print, "en atexit", flush=True)
    return orden


builtins.input = entrada
sys.argv = ["interfaces/shell.py"]
runpy.run_path("interfaces/shell.py", run_name="__main__")
"""


# `python -m crotolamo shell` (launch.sh, el lanzador del escritorio): la misma
# salida, por main(). interfaces.shell ya está importado y con el doble del LLM.
_SHELL_QUE_SALE_POR_MAIN = _SHELL_QUE_SALE.replace(
    'sys.argv = ["interfaces/shell.py"]\n'
    'runpy.run_path("interfaces/shell.py", run_name="__main__")\n',
    'from crotolamo.__main__ import main\nraise SystemExit(main(["shell"]))\n')
assert _SHELL_QUE_SALE_POR_MAIN != _SHELL_QUE_SALE


@pytest.mark.parametrize("guion", [_SHELL_QUE_SALE, _SHELL_QUE_SALE_POR_MAIN],
                         ids=["interfaces/shell.py", "crotolamo shell"])
@pytest.mark.parametrize("senal", [signal.SIGHUP, signal.SIGTERM])
def test_cerrar_la_ventana_tras_salir_no_pierde_la_extraccion(tmp_path, senal, guion):
    """run_shell ya volvió y "Crotolamo apagado, patrón." está en pantalla, pero la
    salida sigue: la extracción del último turno (hasta 5 s) y el cierre de los
    servers MCP corren en atexit. Con los manejadores repuestos al volver, cerrar la
    ventana ahí (o un SIGTERM) mataba el proceso: rc = -señal y el hecho perdido."""
    marca = tmp_path / "extraido"
    env = {k: v for k, v in os.environ.items() if k != "CROTOLAMO_LLM_BACKEND"}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(RAIZ))
    hijo = subprocess.Popen([sys.executable, "-c", guion, str(marca)], cwd=RAIZ,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pantalla = ""
    try:
        assert hijo.stdout is not None
        for linea in hijo.stdout:
            pantalla += linea
            if "en atexit" in linea:
                break
        hijo.send_signal(senal)                   # la ventana se cierra a media salida
        (tmp_path / "extraido.sigue").touch()     # y la extracción puede acabar
        _, errores = hijo.communicate(timeout=20)
    finally:
        if hijo.poll() is None:
            hijo.kill()
            hijo.wait()
    assert "Crotolamo apagado, patrón." in pantalla, errores
    assert hijo.returncode == 0, f"rc={hijo.returncode}: la señal cortó la salida\n{errores}"
    assert marca.exists(), "la extracción del último turno se perdió"
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


def _shell_de_mentira(monkeypatch, entradas: list[Any]) -> None:
    """run_shell sin agente de verdad: `entradas` es lo que devuelve (o lanza) input()."""
    cola = iter(entradas)

    def entrada(prompt: str = "") -> str:
        siguiente = next(cola)
        if isinstance(siguiente, BaseException):
            raise siguiente
        return siguiente

    monkeypatch.setattr(shell, "build_agent", lambda: (object(), object()))
    monkeypatch.setattr(builtins, "input", entrada)


@pytest.mark.parametrize("entradas", [["salir"], [EOFError()], [KeyboardInterrupt()]])
def test_el_shell_que_es_el_proceso_entero_no_repone_los_manejadores(
    monkeypatch, senales_del_proceso, entradas,
):
    """Quien sale del proceso nada más volver run_shell (python -m crotolamo shell,
    el script) lo dice con proceso_entero=True: los manejadores se quedan puestos
    para lo que falta de salida (atexit) y ya no piden salir otra vez."""
    _shell_de_mentira(monkeypatch, entradas)
    antes = _manejadores()
    assert shell.run_shell([], proceso_entero=True) == 0
    despues = _manejadores()
    for sig in _SENALES:
        assert callable(despues[sig]) and despues[sig] != antes[sig]
        signal.raise_signal(sig)                 # la ventana que se cierra: nada


def test_el_proceso_entero_que_revienta_al_arrancar_tambien_sale_protegido(
    monkeypatch, senales_del_proceso,
):
    """El arranque falla con los servers MCP ya lanzados: el proceso sale con su
    traceback y close_all() corre en atexit. Tampoco ahí se reponen."""
    def build_agent():
        raise RuntimeError("config rota")

    monkeypatch.setattr(shell, "build_agent", build_agent)
    antes = _manejadores()
    with pytest.raises(RuntimeError, match="config rota"):
        shell.run_shell([], proceso_entero=True)
    despues = _manejadores()
    for sig in _SENALES:
        assert callable(despues[sig]) and despues[sig] != antes[sig]
        signal.raise_signal(sig)


def test_el_proceso_entero_respeta_el_sighup_ignorado(monkeypatch, senales_del_proceso):
    """`nohup crotolamo shell`: SIGHUP sigue ignorada, antes y después; SIGTERM sí
    se queda con el manejador de la salida."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _shell_de_mentira(monkeypatch, ["salir"])
    antes = _manejadores()
    assert shell.run_shell([], proceso_entero=True) == 0
    despues = _manejadores()
    assert despues[signal.SIGHUP] is signal.SIG_IGN
    assert callable(despues[signal.SIGTERM]) and despues[signal.SIGTERM] != antes[signal.SIGTERM]
    signal.raise_signal(signal.SIGTERM)


def test_sin_decirlo_run_shell_sigue_reponiendo_los_manejadores(monkeypatch, senales_del_proceso):
    """De fábrica, como antes: quien llama run_shell dentro de su proceso (los
    tests) recupera sus manejadores al volver."""
    _shell_de_mentira(monkeypatch, ["salir"])
    antes = _manejadores()
    assert shell.run_shell([]) == 0
    assert _manejadores() == antes


# ---------------------------------------------------------------------------
# Los hijos de los tests de señales no pierden la que llega justo antes de bloquear
# ---------------------------------------------------------------------------
# La carrera (1 de 3000 bajo carga): el test manda la señal en cuanto lee el aviso
# del hijo, y esta llega entre el último chequeo del intérprete y la syscall que
# bloquea (pause, clock_nanosleep). El manejador C la apunta, pero la syscall ya no
# se interrumpe y el manejador Python no corre hasta que el hilo principal vuelva
# a ejecutar bytecode: con UNA llamada bloqueante, nunca (o a los 60 s).
# Aquí sin azar: el hilo principal del hijo bloquea la señal y solo la recibe otro
# hilo, así que su syscall NUNCA se interrumpe, que es justo lo que pasa en la
# carrera. El hijo que espera en una sola llamada no sale; el que espera en tramos, sí.
_SENAL_A_OTRO_HILO = r"""
import signal as _signal, threading as _threading

_recibe = _threading.Event()


def _recibir():
    _signal.pthread_sigmask(_signal.SIG_UNBLOCK, {_signal.SIGHUP, _signal.SIGTERM})
    _recibe.set()
    _threading.Event().wait()


_signal.pthread_sigmask(_signal.SIG_BLOCK, {_signal.SIGHUP, _signal.SIGTERM})
_threading.Thread(target=_recibir, daemon=True).start()
_recibe.wait()
"""


def _hijo_que_no_ve_interrumpida_su_espera(
    guion: str, args: list[str], tmp_path: Path, aviso: str, plazo: float = 15.0,
) -> tuple[int | None, str]:
    """Corre `guion` (el hijo de un test de señales, tal cual) con la señal desviada
    a otro hilo, espera su `aviso` y le manda SIGHUP. Devuelve su código de salida
    (None si a los `plazo` s seguía esperando) y su stderr."""
    env = {k: v for k, v in os.environ.items() if k != "CROTOLAMO_LLM_BACKEND"}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(RAIZ))
    hijo = subprocess.Popen([sys.executable, "-c", _SENAL_A_OTRO_HILO + guion, *args], cwd=RAIZ,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    colgado, errores = False, ""
    try:
        assert hijo.stdout is not None
        for linea in hijo.stdout:
            if aviso in linea:
                break
        time.sleep(0.3)                           # el hijo ya está en su espera
        hijo.send_signal(signal.SIGHUP)
        try:
            _, errores = hijo.communicate(timeout=plazo)
        except subprocess.TimeoutExpired:
            colgado = True
    finally:
        if hijo.poll() is None:
            hijo.kill()
            _, errores = hijo.communicate()
    return (None if colgado else hijo.returncode), errores


def test_el_hijo_del_listener_no_pierde_la_senal_que_llega_antes_de_bloquear(tmp_path):
    """test_sighup_apaga_limpio_como_sigterm falló así una suite entera: su hijo
    hacía print("listo") y signal.pause(), y el SIGHUP que cae entre los dos ya no
    despierta a pause(): _graceful_exit no arrancaba y el test agotaba sus 15 s."""
    from tests.test_auditoria_voice_listener import _HIJO_SENALES

    rc, errores = _hijo_que_no_ve_interrumpida_su_espera(
        _HIJO_SENALES, [str(tmp_path), "normal"], tmp_path, "listo")
    assert rc is not None, f"el hijo no atendió la señal ya apuntada\n{errores}"
    assert rc == 0, errores
    assert (tmp_path / "musica_restaurada").exists()


def test_el_shell_en_el_prompt_no_pierde_la_senal_que_llega_antes_de_bloquear(tmp_path):
    """El mismo hueco en los tests de señales del shell: print("en el prompt") y
    un time.sleep(60) que la señal ya apuntada no interrumpe."""
    marca = tmp_path / "extraido"
    rc, errores = _hijo_que_no_ve_interrumpida_su_espera(
        _SHELL_EN_EL_PROMPT, [str(marca)], tmp_path, "en el prompt")
    assert rc is not None, f"el hijo no atendió la señal ya apuntada\n{errores}"
    assert rc == 128 + signal.SIGHUP, errores
    assert marca.exists(), "la extracción del último turno se perdió"
