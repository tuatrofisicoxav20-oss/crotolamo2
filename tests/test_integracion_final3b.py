"""Integración de la revisión final (tanda 3b): el apagado por señal del listener.

Los dos pedidos caen en interfaces/listener.py, en el manejador que acaba en
os._exit (se salta atexit y todos los finally):

- idx 3 (final-x3vida-flush-drain): la Memoria se vaciaba con flush() a secas.
  Con el hilo de extracción en su pausa (tres fallos seguidos), flush() esperaba
  los 2 s del apagado detrás de él y lo encolado durante la pausa se perdía sin
  intentarse. El shell, por atexit, ya la cortaba (_vaciar_al_salir).
- idx 4 (final-x3vida-kill-survivors): a los servers MCP solo se les mandaba
  SIGTERM. El que lo ignora (el nieto de un wrapper, uno colgado) sobrevivía a
  Crotolamo en su propia sesión. bridge.kill_survivors() es la otra mitad.

Contra procesos reales (el fake server tras un `sh -c`) y la Memoria de verdad.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from crotolamo.core import memoria as memoria_mod
from crotolamo.core import router
from crotolamo.core.memoria import Memoria, MemoriaConfig
from crotolamo.mcp import bridge
from crotolamo.settings import Settings
from crotolamo.tools.base import Registry
from interfaces import listener

RAIZ = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).parent / "fake_mcp_server.py"
FAKE_CMD = [sys.executable, str(FAKE)]


# El idle final de verdad, antes de que el fixture `apagado` lo sustituya.
_IDLE_DE_VERDAD = listener._final_idle_best_effort


class _Salida(BaseException):
    """Sustituto de os._exit en los tests del manejador de señal."""


def _esperar(cond: Callable[[], bool], plazo: float = 5.0) -> bool:
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _vivo(pid: int) -> bool:
    """True si el proceso existe y no es un zombi (os.kill(pid, 0) da True a un zombi)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            estado = f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return estado not in ("Z", "X")


@pytest.fixture
def apagado(monkeypatch):
    """El manejador de señal de verdad, con os._exit sustituido y sin nada vivo
    de otro test (ducker, monitor de música, compuerta del HUD, Memoria). El
    idle final solo se apunta: el de verdad escribiría en ~/.crotolamo. Como
    lanzado desde una terminal (sin el INVOCATION_ID de systemd, que el CI sí
    puede traer puesto); ver el fixture `servicio`."""
    orden: list[str] = []
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    def _exit(code):
        orden.append("exit")
        raise _Salida(code)

    monkeypatch.setattr(listener.os, "_exit", _exit)
    monkeypatch.setattr(listener, "_ACTIVE_DUCKER", None)
    monkeypatch.setattr(listener, "_ACTIVE_MEDIA", None)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", None)
    monkeypatch.setattr(listener, "_final_idle_best_effort", lambda: orden.append("idle"))
    monkeypatch.setattr(memoria_mod, "_MEMORIA", None)
    return orden


@pytest.fixture
def servicio(apagado, monkeypatch):
    """Como servicio de systemd: es ahí donde hay un TimeoutStopSec que cumplir."""
    monkeypatch.setenv("INVOCATION_ID", "0123456789abcdef0123456789abcdef")
    return apagado


def _apagar() -> float:
    """Corre _graceful_exit hasta su os._exit; devuelve lo que tardó."""
    t0 = time.monotonic()
    with pytest.raises(_Salida):
        listener._graceful_exit(signal.SIGTERM, None)
    return time.monotonic() - t0


# ---------------------------------------------------------------------------
# idx 3 (final-x3vida-flush-drain): el apagado por señal corta la pausa de la extracción
# ---------------------------------------------------------------------------

class _Proveedor:
    """Doble de mem0: la extracción falla (429) hasta que se le dice lo contrario."""

    def __init__(self) -> None:
        self.intentos: list[str] = []
        self.extraidos: list[str] = []
        self.fallar: Exception | None = RuntimeError("groq 429: rate limit")

    def extraer(self, usuario: str, respuesta: str) -> list[str]:
        self.intentos.append(usuario)
        if self.fallar is not None:
            raise self.fallar
        self.extraidos.append(usuario)
        return [usuario]


@pytest.fixture
def en_pausa(monkeypatch):
    """Una Memoria de verdad con el hilo de extracción DENTRO de su pausa (tres
    fallos seguidos), puesta como la del proceso."""
    monkeypatch.setattr(memoria_mod, "EXTRAER_PAUSA_S", 60.0)
    proveedor = _Proveedor()
    m = Memoria(MemoriaConfig(enabled=True), backend=proveedor)
    monkeypatch.setattr(memoria_mod, "_MEMORIA", m)
    for texto in ("uno", "dos", "tres"):
        assert m.extraer_en_fondo(texto, "ok")
    assert _esperar(m._en_pausa.is_set), "el hilo de extracción no llegó a pausarse"
    yield m, proveedor
    m.cerrar(timeout_s=1.0)


def test_el_apagado_por_senal_extrae_lo_encolado_durante_la_pausa(en_pausa):
    """Tres extracciones fallidas -> el hilo pausa 120 s ("lo encolado espera,
    no se pierde"). El patrón cuenta algo y el servicio se para en ese rato:
    flush() a secas esperaba sus 2 s detrás del hilo dormido y os._exit tiraba
    el turno sin intentarlo. El vaciado de salida corta la pausa, como atexit."""
    m, proveedor = en_pausa
    proveedor.fallar = None  # el proveedor ya se recuperó
    assert m.extraer_en_fondo("mi hermana Ana vive en Monterrey", "Anotado, patrón.")
    t0 = time.monotonic()
    listener._flush_memoria_best_effort()
    assert time.monotonic() - t0 < 1.0, "agotó los 2 s detrás de la pausa"
    assert proveedor.extraidos == ["mi hermana Ana vive en Monterrey"]


def test_el_apagado_entero_guarda_lo_encolado_durante_la_pausa(apagado, en_pausa, monkeypatch):
    """Lo mismo por el manejador entero (systemctl stop a media pausa): el turno
    queda extraído antes del idle y del os._exit."""
    # `apagado` va antes que `en_pausa`: el primero quita la Memoria del proceso
    # y el segundo pone la suya.
    m, proveedor = en_pausa
    assert memoria_mod._MEMORIA is m
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: apagado.append("mcp"))
    proveedor.fallar = None
    assert m.extraer_en_fondo("mi perro se llama Bruno", "Anotado, patrón.")
    assert _apagar() < 1.0
    assert proveedor.extraidos == ["mi perro se llama Bruno"]
    assert apagado == ["mcp", "idle", "exit"]


def test_el_vaciado_del_apagado_lleva_el_presupuesto_del_manejador(monkeypatch):
    """_vaciar_al_salir sin plazo espera los 5 s de atexit: el manejador tiene 2 s
    (y un join de 2,5 s). El plazo tiene que llegarle al vaciado."""
    plazos: list[float | None] = []

    class _Memoria(Memoria):
        def _vaciar_al_salir(self, timeout_s: float | None = None) -> bool:
            plazos.append(timeout_s)
            return super()._vaciar_al_salir(timeout_s)

    m = _Memoria(MemoriaConfig(enabled=True), backend=_Proveedor())
    monkeypatch.setattr(memoria_mod, "_MEMORIA", m)
    try:
        listener._flush_memoria_best_effort()
        listener._flush_memoria_best_effort(timeout_s=0.3)
    finally:
        m.cerrar(timeout_s=1.0)
    assert plazos == [2.0, 0.3]


# ---------------------------------------------------------------------------
# idx 4 (final-x3vida-kill-survivors): el apagado por señal remata al server MCP
# que ignora el SIGTERM
# ---------------------------------------------------------------------------

# El server real detrás de un wrapper: habla MCP (es el fake), ignora el
# SIGTERM y deja su pid en argv[1] para que el test lo vigile.
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


@pytest.fixture
def mcp_limpio():
    """Ningún server ni grupo dinámico sobrevive de un test a otro."""
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)
    yield
    bridge.close_all()
    for name in router.dynamic_groups():
        router.unregister_group(name)


def _registrar(servers: dict) -> list[str]:
    raw = {"enabled": True, "timeout_s": 5, "startup_timeout_s": 5, "servers": servers}
    return bridge.register_mcp_tools(
        Registry(), Settings(raw={"mcp": raw}, user="test", home=Path("/tmp")))


@pytest.fixture
def terco(tmp_path, mcp_limpio):
    """Un server tras un wrapper `sh -c` (como npx/uvx), ya conectado: el wrapper
    muere con el SIGTERM; el nieto, que ni lo atiende ni sale al ver el EOF, no.
    Devuelve el pid del nieto."""
    script = tmp_path / "terco.py"
    script.write_text(_TERCO, encoding="utf-8")
    pidfile = tmp_path / "terco.pid"
    # '; true': sh no hace exec y queda de launcher (que SÍ muere con SIGTERM).
    orden = f"{sys.executable} {script} {pidfile} {FAKE}; true"
    names = _registrar({"terco": {"command": ["sh", "-c", orden],
                                  "env": {"FAKE_MCP_EOF_DELAY": "600"}}})
    assert "mcp_terco_echo" in names
    pid = int(pidfile.read_text())
    yield pid
    if _vivo(pid):
        os.kill(pid, signal.SIGKILL)


def _murio(pid: int) -> bool:
    """El SIGKILL sale justo antes del os._exit y no se espera a que haga efecto
    (el kernel ya lo tiene): aquí sí, un instante."""
    return _esperar(lambda: not _vivo(pid), plazo=1.0)


def _timeout_stop_sec() -> float:
    """El TimeoutStopSec de la unidad: pasado ese rato, systemd mata con SIGKILL."""
    unidad = (RAIZ / "desktop" / "crotolamo.service").read_text(encoding="utf-8")
    valores = [linea.split("=", 1)[1] for linea in unidad.splitlines()
               if linea.startswith("TimeoutStopSec=")]
    assert len(valores) == 1, valores
    return float(valores[0])


def test_el_apagado_por_senal_no_deja_vivo_al_server_que_ignora_el_sigterm(terco, apagado):
    """terminate_all() no espera ni escala y os._exit se salta el close() de
    atexit: el nieto del wrapper seguía vivo tras salir Crotolamo. Con systemd lo
    acababa matando el cgroup (el stop entero tardaba el TimeoutStopSec); lanzado
    desde una terminal, nadie."""
    tardo = _apagar()
    assert _murio(terco), "el server sobrevivió a Crotolamo"
    assert apagado == ["idle", "exit"]  # el idle sigue pegado al os._exit
    # La gracia del SIGTERM (2 s) y poco más: nada parecido al TimeoutStopSec.
    assert tardo < 4.0


def test_el_remate_va_tras_vaciar_la_memoria_y_antes_del_idle(apagado, monkeypatch):
    """Tras la Memoria: lo que tarde en vaciarse ya es gracia para los servers
    (la de kill_survivors cuenta desde el SIGTERM). Antes del idle: este tiene
    que seguir pegado al os._exit, o lo que publique otro hilo entretanto lo pisa."""
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: apagado.append("mcp"))
    monkeypatch.setattr(listener, "_flush_memoria_best_effort",
                        lambda: apagado.append("memoria"))
    monkeypatch.setattr(bridge, "kill_survivors", lambda grace_s: apagado.append("remate"))
    _apagar()
    assert apagado == ["mcp", "memoria", "remate", "idle", "exit"]


def test_un_fallo_al_rematar_no_impide_el_idle_ni_la_salida(apagado, monkeypatch):
    llamadas: list[int] = []

    def revienta(grace_s: float) -> None:
        llamadas.append(1)
        raise RuntimeError("fallo al escalar")

    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: None)
    monkeypatch.setattr(bridge, "kill_survivors", revienta)
    _apagar()
    assert llamadas == [1]
    assert apagado == ["idle", "exit"]


@pytest.mark.parametrize("como_servicio", [False, True])
def test_con_servers_bien_hechos_el_remate_no_alarga_el_apagado(
    mcp_limpio, apagado, monkeypatch, como_servicio,
):
    """Los que mueren con el SIGTERM no pagan la gracia: el remate vuelve en
    cuanto han salido y a ninguno le llega el SIGKILL."""
    if como_servicio:
        monkeypatch.setenv("INVOCATION_ID", "0123456789abcdef0123456789abcdef")
    assert _registrar({n: {"command": FAKE_CMD} for n in ("a", "b")})
    procs = [bridge._STATE[n].client._proc for n in ("a", "b")]
    rematados: list[tuple[float, float]] = []
    real = bridge.kill_survivors

    def espia(grace_s: float) -> None:
        t0 = time.monotonic()
        real(grace_s=grace_s)
        rematados.append((grace_s, time.monotonic() - t0))

    monkeypatch.setattr(bridge, "kill_survivors", espia)
    assert _apagar() < 1.5
    assert len(rematados) == 1
    gracia, tardo = rematados[0]
    # La gracia de siempre (la misma que da close()): el tope no la recorta en
    # un apagado normal. Y no se agota: los servers ya habían salido.
    assert gracia == bridge.SIGTERM_GRACE_S
    assert tardo < 1.0
    for proc in procs:
        assert proc is not None and proc.poll() is not None
        assert proc.returncode != -signal.SIGKILL


def test_como_servicio_el_remate_no_espera_mas_alla_del_tope(terco, servicio, monkeypatch):
    """Lo de ANTES del remate no está acotado en conjunto: reponer la música
    cuesta 2 s por reproductor atenuado con el D-Bus colgado. Medido con cuatro:
    8 s sin remate y 10 s con la gracia entera, el TimeoutStopSec de la unidad
    (systemd mata al listener sin idle final). Con el tope, a esas alturas ya no
    se espera al server: se le mata y se sigue."""
    monkeypatch.setattr(listener, "_SHUTDOWN_KILL_BY_S", 0.4)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: time.sleep(0.3))
    tardo = _apagar()
    assert _murio(terco), "el server sobrevivió a Crotolamo"
    assert servicio == ["idle", "exit"]
    assert tardo < 1.5, "esperó la gracia entera pese a ir ya sin tiempo"


def _gracias_del_remate(monkeypatch) -> list[float]:
    """Apunta la gracia con la que se llama a kill_survivors, sin rematar a nadie."""
    gracias: list[float] = []
    monkeypatch.setattr(listener, "_terminate_mcp_best_effort", lambda: None)
    monkeypatch.setattr(bridge, "kill_survivors", lambda grace_s: gracias.append(grace_s))
    return gracias


def test_como_servicio_con_el_tope_ya_pasado_el_remate_no_espera_nada(servicio, monkeypatch):
    monkeypatch.setattr(listener, "_SHUTDOWN_KILL_BY_S", 0.1)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: time.sleep(0.3))
    gracias = _gracias_del_remate(monkeypatch)
    _apagar()
    assert gracias == [0.0]  # ni negativa: kill_survivors la suma al instante del SIGTERM


def test_como_servicio_el_tope_recorta_la_gracia_a_lo_que_queda(servicio, monkeypatch):
    monkeypatch.setattr(listener, "_SHUTDOWN_KILL_BY_S", 1.0)
    gracias = _gracias_del_remate(monkeypatch)
    _apagar()
    assert len(gracias) == 1 and 0.5 < gracias[0] <= 1.0


def test_desde_una_terminal_el_tope_no_recorta_la_gracia(apagado, monkeypatch):
    """Sin systemd no hay TimeoutStopSec que cumplir, y nadie le mandó antes un
    SIGTERM al server: recortarle la gracia sería matar sin dejarle salir a uno
    que sale bien. La gracia entera, tarde lo que tarde lo anterior."""
    monkeypatch.setattr(listener, "_SHUTDOWN_KILL_BY_S", 0.1)
    monkeypatch.setattr(listener, "_restore_media_best_effort", lambda: time.sleep(0.3))
    gracias = _gracias_del_remate(monkeypatch)
    _apagar()
    assert gracias == [bridge.SIGTERM_GRACE_S]


def test_el_tope_del_remate_deja_sitio_al_idle_bajo_el_timeoutstopsec():
    """Tras el remate solo queda el idle: la espera (acotada) al candado de la
    compuerta y una escritura. Si alguien baja el TimeoutStopSec de la unidad o
    sube el tope, que se entere aquí y no con un HUD que se queda en "HABLANDO"."""
    import inspect

    candado_s = inspect.signature(listener._HudGate.close_with_idle).parameters[
        "lock_timeout_s"].default
    assert listener._SHUTDOWN_KILL_BY_S + candado_s + 0.5 <= _timeout_stop_sec()
    # Y no recorta la gracia en un apagado normal (que llega al remate en ms).
    assert listener._SHUTDOWN_KILL_BY_S >= 2.5 + bridge.SIGTERM_GRACE_S


def test_el_peor_caso_del_apagado_cabe_en_el_timeoutstopsec_de_la_unidad(
    terco, servicio, monkeypatch, tmp_path,
):
    """Todo lo que el manejador espera tras el SIGTERM a los servers, a la vez y
    en su peor caso: una Memoria que no vuelve (el join de 2,5 s), un server que
    ignora el SIGTERM (2 s de gracia) y la compuerta del HUD con el candado
    tomado por un hilo colgado (1 s). Si pasara del TimeoutStopSec de la unidad,
    systemd mataría al listener antes del idle final.

    Las dos esperas largas se solapan (la gracia cuenta desde el SIGTERM, no
    desde que se llama al remate): ~3,5 s, no 5,5 s."""
    soltar = threading.Event()
    monkeypatch.setattr(memoria_mod, "_MEMORIA",
                        SimpleNamespace(flush=lambda timeout_s=5.0: soltar.wait(60)))
    gate = listener._HudGate(tmp_path / "hud_state.json")
    tomado = threading.Event()

    def colgado() -> None:
        with gate._lock:
            tomado.set()
            soltar.wait(60)

    hilo = threading.Thread(target=colgado, daemon=True)
    hilo.start()
    assert tomado.wait(5)
    # El idle de verdad, por la compuerta: escribe en tmp_path, no en ~/.crotolamo
    # (de donde tampoco lee el control.json del dueño).
    monkeypatch.setattr(listener, "read_control_enabled", lambda *a, **k: True)
    monkeypatch.setattr(listener, "_ACTIVE_HUD_GATE", gate)
    monkeypatch.setattr(listener, "_final_idle_best_effort",
                        lambda: (servicio.append("idle"), _IDLE_DE_VERDAD()))
    try:
        tardo = _apagar()
    finally:
        soltar.set()
        hilo.join(5)
    assert _murio(terco), "el server sobrevivió a Crotolamo"
    assert servicio == ["idle", "exit"]
    assert (tmp_path / "hud_state.json").exists()  # el idle se escribió, sin el candado
    assert tardo < _timeout_stop_sec()
    assert tardo < 5.0, "la gracia del SIGTERM se sumó a la espera de la Memoria"
