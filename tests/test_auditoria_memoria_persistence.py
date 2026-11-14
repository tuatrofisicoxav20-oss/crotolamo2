"""Arreglos de la auditoría en la memoria semántica y la persistencia.

Cada bloque fija un hallazgo confirmado. mem0 no está instalado en los tests: se
suplanta el MÓDULO `mem0` por un doble para ejercitar el `Mem0Backend` real (el
FakeBackend de test_memoria.py sustituye esa capa entera y tapaba sus fallos).

- Cualquier fallo al ARRANCAR mem0 (Chroma ilegible, falta un paquete, sin key)
  es "memoria no disponible": se marca rota una vez, las tools caen a la libreta
  SQLite y no se reconstruye mem0 en cada turno.
- El modelo de embeddings se guarda en [memoria].ruta, no en /tmp (tmpfs).
- Sin API key para el LLM de extracción la memoria arranca igual (buscar y
  recordar no usan LLM); solo se apaga la extracción automática.
- parece_secreto: prefijos de AWS/Google que no casaban nunca, "clave"/NIP/PIN
  con un valor numérico, valores cortos con dígitos y valores sin conector.
- recordar() y `memoria migrar` no duplican un hecho que ya está (mem0 crea un
  id nuevo en cada add): migrar se puede repetir.
- SQLite: la espera ante bloqueo se fija al abrir (sin un PRAGMA por operación)
  y una base nueva ya no crea la tabla session_log, que nadie usaba.
- Libreta SQLite: search() puntúa cada hecho una vez; el respaldo de
  olvidar_recuerdo no borra un hecho que no trata de lo pedido; remember_fact
  aplica el mismo filtro de secretos que recordar_de_mi.
- "No pude buscar" no es "no hay nada": si la búsqueda falla o tarda, las tools
  y la CLI lo dicen (y no borran nada) en vez de contestar "no recuerdo nada".
- Tools de hechos: `null` en la categoría opcional no las revienta, y el filtro
  por categoría no depende de mayúsculas ni acentos.
- calibrar --groq anuncia el proveedor y el modelo que de verdad usa mem0.
- Al salir del proceso (run_shell con "salir") se termina, acotada, la extracción
  pendiente del último turno en vez de morir con el hilo daemon.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from typing import Any, Callable

import pytest

from crotolamo.core import memoria as memoria_mod
from crotolamo.core.memoria import (
    Mem0Backend,
    Memoria,
    MemoriaConfig,
    MemoriaNoDisponible,
    make_memoria_prehook,
    parece_secreto,
)

_ENVS_KEY = ("CROTOLAMO_GLM_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY",
             "OPENROUTER_API_KEY", "GROQ_API_KEY")
_ROTO = "chroma: database disk image is malformed"


class _Mem0Falso:
    """Lo mínimo de `mem0.Memory` que usa Mem0Backend."""

    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.extracciones: list[Any] = []

    def add(self, mensajes: Any, user_id: str, infer: bool) -> dict[str, Any]:
        if infer:
            self.extracciones.append(mensajes)
            return {"results": [{"id": "e1", "memory": "hecho extraído", "event": "ADD"}]}
        mid = f"m{len(self.items) + 1}"
        self.items[mid] = str(mensajes)
        return {"results": [{"id": mid, "memory": str(mensajes), "event": "ADD"}]}

    def search(self, query: str, filters: dict, top_k: int, threshold: float) -> dict[str, Any]:
        return {"results": [{"id": k, "memory": v, "score": 0.5}
                            for k, v in self.items.items()][:top_k]}

    def get_all(self, filters: dict, top_k: int) -> dict[str, Any]:
        return {"results": [{"id": k, "memory": v} for k, v in self.items.items()]}

    def delete(self, memoria_id: str) -> None:
        self.items.pop(memoria_id, None)


def _suplantar_mem0(monkeypatch, from_config: Callable[[dict], Any]) -> None:
    modulo = types.ModuleType("mem0")

    class Memory:
        pass

    Memory.from_config = staticmethod(from_config)  # type: ignore[attr-defined]
    modulo.Memory = Memory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", modulo)


def _mem0_que_no_arranca(monkeypatch, error: Exception) -> dict[str, int]:
    intentos = {"n": 0}

    def from_config(conf: dict) -> Any:
        intentos["n"] += 1
        raise error

    _suplantar_mem0(monkeypatch, from_config)
    return intentos


def _mem0_como_el_real(monkeypatch) -> list[tuple[dict, _Mem0Falso]]:
    """from_config que, como mem0 + el SDK del proveedor, LANZA al construir el
    cliente del LLM si no hay key ni en la config ni en la env del SDK (medido con
    openai y groq reales: OpenAIError / GroqError "Missing credentials")."""
    arrancados: list[tuple[dict, _Mem0Falso]] = []
    envs_sdk = {"openai": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"), "groq": ("GROQ_API_KEY",)}

    def from_config(conf: dict) -> Any:
        llm = conf["llm"]
        envs = envs_sdk.get(llm["provider"], ())
        if envs and not llm["config"].get("api_key") and not any(os.environ.get(e) for e in envs):
            raise RuntimeError("Missing credentials. Please pass an `api_key`")
        mem = _Mem0Falso()
        arrancados.append((conf, mem))
        return mem

    _suplantar_mem0(monkeypatch, from_config)
    return arrancados


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch):
    # Mem0Backend fija FASTEMBED_CACHE_PATH (setdefault) al arrancar: el par
    # setenv+delenv hace que monkeypatch restaure la variable al acabar cada test.
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "x")
    monkeypatch.delenv("FASTEMBED_CACHE_PATH")
    for env in _ENVS_KEY:
        monkeypatch.delenv(env, raising=False)
    memoria_mod.set_memoria(None)
    yield
    memoria_mod.set_memoria(None)


def _settings_con_memoria(monkeypatch, **memoria: Any) -> None:
    from crotolamo import settings as settings_mod

    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "raw", {**real.raw, "memoria": {"enabled": True, **memoria}})
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)


# --- mem0 que no arranca: no disponible, una vez, con respaldo SQLite ----------

def test_fallo_de_arranque_de_mem0_es_memoria_no_disponible(monkeypatch, tmp_path):
    _mem0_que_no_arranca(monkeypatch, RuntimeError(_ROTO))
    backend = Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"))
    with pytest.raises(MemoriaNoDisponible, match="malformed"):
        backend.buscar("hola", 1, 0.03)


def test_ruta_que_no_se_puede_crear_es_memoria_no_disponible(monkeypatch, tmp_path):
    estorbo = tmp_path / "fichero"
    estorbo.write_text("no soy una carpeta", encoding="utf-8")
    _suplantar_mem0(monkeypatch, lambda conf: _Mem0Falso())
    backend = Mem0Backend(MemoriaConfig(enabled=True, ruta=estorbo / "mem"))
    with pytest.raises(MemoriaNoDisponible):
        backend.todas()


def test_arranque_roto_se_marca_una_vez_y_no_se_reintenta_en_cada_turno(
        monkeypatch, tmp_path, caplog):
    intentos = _mem0_que_no_arranca(monkeypatch, RuntimeError(_ROTO))
    m = Memoria(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"))  # Mem0Backend REAL
    hook = make_memoria_prehook(m)
    try:
        with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
            for i in range(5):
                assert hook(f"turno {i}") == f"turno {i}"
                m.extraer_en_fondo(f"turno {i}", "ok")
                assert m.flush()
        # Antes: 10 arranques de mem0 en 5 turnos (búsqueda + extracción cada uno).
        assert intentos["n"] == 1
        assert not m.enabled and "malformed" in (m.motivo_no_disponible or "")
        assert caplog.text.count("no disponible") == 1
    finally:
        m.cerrar()


def test_precalentar_con_mem0_roto_la_deja_marcada_antes_del_primer_turno(monkeypatch, tmp_path):
    """El arranque real pasa por precalentar(): si mem0 no arranca ahí, el primer
    turno ya no debe gastar su presupuesto de búsqueda reconstruyéndolo."""
    intentos = _mem0_que_no_arranca(monkeypatch, RuntimeError(_ROTO))
    m = Memoria(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"))
    try:
        m.precalentar()
        limite = time.monotonic() + 3.0
        while m.enabled and time.monotonic() < limite:
            time.sleep(0.01)
        assert not m.enabled
        assert make_memoria_prehook(m)("hola") == "hola"
        assert intentos["n"] == 1
    finally:
        m.cerrar()


def test_arranque_roto_las_tools_caen_a_la_libreta_sqlite(monkeypatch, tmp_path):
    from crotolamo.persistence import facts
    from crotolamo.tools import memoria as tools_mod

    monkeypatch.setattr(facts.db, "_default_db_path", lambda: tmp_path / "f.sqlite")
    intentos = _mem0_que_no_arranca(monkeypatch, RuntimeError(_ROTO))
    m = Memoria(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"))
    memoria_mod.set_memoria(m)
    try:
        out = tools_mod.recordar_de_mi("mi perro se llama Tletl")
        assert out.startswith("Ya quedó") and "libreta" in out
        assert [r["texto"] for r in facts.recall()] == ["mi perro se llama Tletl"]
        # recordar() también la marca rota: la siguiente llamada ni toca mem0.
        assert not m.enabled
        assert tools_mod.recordar_de_mi("me gusta el pozole").startswith("Ya quedó")
        assert intentos["n"] == 1
        assert "Tletl" in tools_mod.buscar_recuerdos("mi perro")
        assert tools_mod.olvidar_recuerdo("mi perro").startswith("Olvidado")
    finally:
        m.cerrar()


def test_cli_con_mem0_roto_avisa_en_vez_de_un_traceback(monkeypatch, tmp_path, capsys):
    _mem0_que_no_arranca(monkeypatch, RuntimeError(_ROTO))
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "mem"))
    assert memoria_mod.run_cli(["listar"]) == 1
    out = capsys.readouterr().out
    assert "Memoria no disponible" in out and "malformed" in out


# --- caché del modelo de embeddings fuera de /tmp --------------------------------

def _mem0_que_anota_el_entorno(monkeypatch) -> dict[str, Any]:
    visto: dict[str, Any] = {}

    def from_config(conf: dict) -> Any:
        visto["cache"] = os.environ.get("FASTEMBED_CACHE_PATH")
        visto["chroma"] = conf["vector_store"]["config"]["path"]
        return _Mem0Falso()

    _suplantar_mem0(monkeypatch, from_config)
    return visto


def test_cache_de_embeddings_vive_en_la_ruta_de_la_memoria(monkeypatch, tmp_path):
    """fastembed usa <tmp>/fastembed_cache si nadie fija FASTEMBED_CACHE_PATH; en
    Fedora /tmp es tmpfs: el modelo ocupaba RAM y se rebajaba en cada arranque."""
    visto = _mem0_que_anota_el_entorno(monkeypatch)
    Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem")).todas()
    assert visto["cache"] == str(tmp_path / "mem" / "modelos")


def test_cache_de_embeddings_respeta_la_variable_del_patron(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/otro/sitio")
    visto = _mem0_que_anota_el_entorno(monkeypatch)
    Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem")).todas()
    assert visto["cache"] == "/otro/sitio"


def test_calibrar_reusa_la_cache_de_modelos_de_la_memoria_real(monkeypatch, tmp_path, capsys):
    """calibrar trabaja sobre una memoria TEMPORAL que borra al acabar: los datos
    van ahí, pero el modelo (240 MB) no se puede bajar a esa carpeta cada vez."""
    from scripts import memoria_calibrar

    visto = _mem0_que_anota_el_entorno(monkeypatch)
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "real"))
    assert memoria_calibrar.main([]) == 0
    assert visto["cache"] == str(tmp_path / "real" / "modelos")
    assert not visto["chroma"].startswith(str(tmp_path / "real"))  # la tuya no se toca
    assert "aciertos top-1" in capsys.readouterr().out


# --- sin key para el LLM de extracción: la memoria NO muere -------------------

@pytest.mark.parametrize("provider", ["auto", "groq"])
def test_sin_key_del_llm_la_memoria_arranca_y_solo_se_apaga_la_extraccion(
        provider, monkeypatch, tmp_path, caplog):
    arrancados = _mem0_como_el_real(monkeypatch)
    cfg = MemoriaConfig(enabled=True, ruta=tmp_path / "mem", llm_provider=provider)
    m = Memoria(cfg)  # Mem0Backend REAL; sin settings la nube es Z.ai de fábrica
    try:
        with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
            assert m.recordar("mi perro se llama Tletl")
            assert [r.texto for r in m.buscar("mi perro", timeout_s=5.0)] == [
                "mi perro se llama Tletl"]
            assert m.extraer_en_fondo("me gusta el pozole", "va, patrón")
            assert m.flush()
        assert m.enabled
        _conf, mem = arrancados[0]
        assert len(arrancados) == 1
        assert mem.extracciones == []  # sin key no se llama al LLM
        assert caplog.text.count("extracción automática queda apagada") == 1
    finally:
        m.cerrar()


def test_sin_key_y_con_extraer_false_ni_siquiera_avisa(monkeypatch, tmp_path, caplog):
    _mem0_como_el_real(monkeypatch)
    backend = Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem", extraer=False))
    with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
        assert backend.guardar("mi perro se llama Tletl")
    assert "apagada" not in caplog.text


def test_con_key_la_extraccion_sigue_llamando_al_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "clave-de-zai")
    arrancados = _mem0_como_el_real(monkeypatch)
    backend = Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"))
    assert backend.extraer("me gusta el pozole", "va, patrón") == ["hecho extraído"]
    conf, mem = arrancados[0]
    assert conf["llm"]["config"]["api_key"] == "clave-de-zai"
    assert len(mem.extracciones) == 1


def test_key_solo_en_la_env_del_sdk_no_se_pisa_con_la_de_relleno(monkeypatch, tmp_path):
    """mem0 cae a OPENAI_API_KEY / GROQ_API_KEY si la config no trae key: eso ya
    funcionaba y debe seguir igual (la key de relleno es solo para cuando no hay).
    OPENAI_API_KEY va al base_url de la nube: solo cuenta si esa nube ES OpenAI (con
    la de Z.ai se mandaba a Z.ai; ver tests/test_revision_final_u09.py)."""
    from crotolamo.settings import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "clave-del-sdk")
    arrancados = _mem0_como_el_real(monkeypatch)
    nube_openai = Settings(raw={"llm": {"glm": {"base_url": "https://api.openai.com/v1"}}},
                           user="t", home=tmp_path)
    backend = Mem0Backend(MemoriaConfig(enabled=True, ruta=tmp_path / "mem"), nube_openai)
    assert backend.extraer("me gusta el pozole", "va, patrón") == ["hecho extraído"]
    conf, _mem = arrancados[0]
    assert "api_key" not in conf["llm"]["config"]


# --- parece_secreto: agujeros del filtro ----------------------------------------

@pytest.mark.parametrize("texto", [
    # AWS y Google no llevan `_`/`-` tras el prefijo: esas alternativas no casaban.
    "AKIAIOSFODNN7EXAMPLE",
    "mi access key de aws es AKIAIOSFODNN7EXAMPLE",
    "ya29.a0AfH6SMBx7abcdefghijklmnopqrstuvwxyz0123456789",
    "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh",
    # "clave" a secas y códigos numéricos, cuando el valor lleva dígitos.
    "la clave del banco es 482913",
    "mi clave es Gato2024!",
    "la clave de mi correo es Gato2024",
    "mi nip es 4821",
    "el pin de la tarjeta es 4821",
    "el codigo de seguridad es 9931",
    # Valores cortos con dígitos (antes hacían falta 6 caracteres).
    "mi contraseña es 1234",
    "la contraseña del wifi es 12345",
    # Valor pegado a la palabra, sin "es" ni ":".
    "usa la contraseña Gato2024! para entrar",
    "mi password Tr0ub4dor&3xx",
    "contraseña hunter2pass",
])
def test_parece_secreto_detecta_lo_que_se_escapaba(texto):
    assert parece_secreto(texto)


@pytest.mark.parametrize("texto", [
    "la palabra clave es constancia",
    "la clave del éxito es la constancia",
    "la clave para llegar temprano es salir a las 7",
    "la clave es 100% actitud",
    "la clave lada de Puebla es 222",
    "mi pin favorito es el de la bandera",
    "le puse un pin a la mochila en 2019",
    "cambié la contraseña 3 veces este año",
    "tengo un token 2FA en el llavero",
    "uso el password manager de siempre",
])
def test_parece_secreto_sigue_dejando_pasar_el_espanol_cotidiano(texto):
    assert not parece_secreto(texto)


# --- `crotolamo memoria calibrar` fuera de la raíz del repo -----------------------

def _scripts_no_importable(monkeypatch) -> None:
    """Como con el comando `crotolamo` (console-script) o `python -m crotolamo`
    desde otra carpeta: `scripts` no se empaqueta y la raíz del repo no está en
    sys.path, así que `import scripts` lanza ModuleNotFoundError."""
    monkeypatch.setitem(sys.modules, "scripts", None)
    monkeypatch.setitem(sys.modules, "scripts.memoria_calibrar", None)


def test_cli_calibrar_funciona_aunque_scripts_no_sea_importable(monkeypatch, tmp_path, capsys):
    _scripts_no_importable(monkeypatch)
    with pytest.raises(ImportError):  # el doble reproduce el fallo real
        import scripts.memoria_calibrar  # noqa: F401
    _suplantar_mem0(monkeypatch, lambda conf: _Mem0Falso())
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "real"))
    # Antes: ModuleNotFoundError: No module named 'scripts' (traceback crudo).
    assert memoria_mod.run_cli(["calibrar"]) == 0
    assert "aciertos top-1" in capsys.readouterr().out


def test_cli_calibrar_sin_el_script_avisa_en_vez_de_un_traceback(monkeypatch, tmp_path, capsys):
    """Checkout sin scripts/ (copia parcial): se explica qué falta en vez de reventar."""
    from crotolamo import settings as settings_mod

    _scripts_no_importable(monkeypatch)
    _settings_con_memoria(monkeypatch)
    monkeypatch.setattr(settings_mod, "REPO_ROOT", tmp_path)
    assert memoria_mod.run_cli(["calibrar"]) == 1
    out = capsys.readouterr().out
    assert "scripts/memoria_calibrar.py" in out and "Traceback" not in out


def test_cli_calibrar_sigue_usando_el_modulo_importable(monkeypatch, tmp_path):
    """Desde la raíz del repo nada cambia: se usa `scripts.memoria_calibrar` tal
    cual (y con sus argumentos), sin cargarlo dos veces por su ruta."""
    from crotolamo import settings as settings_mod
    from scripts import memoria_calibrar

    visto: list[list[str]] = []
    monkeypatch.setattr(memoria_calibrar, "main", lambda argv: visto.append(argv) or 7)
    # La ruta de respaldo no debe ni mirarse: aquí no hay script que cargar.
    monkeypatch.setattr(settings_mod, "REPO_ROOT", tmp_path)
    assert memoria_mod.run_cli(["calibrar", "--groq"]) == 7
    assert visto == [["--groq"]]


# --- recordar() y `memoria migrar` no duplican ------------------------------------

def _memoria_sobre_mem0_falso(monkeypatch, tmp_path) -> tuple[Memoria, _Mem0Falso]:
    """Memoria con el Mem0Backend REAL sobre un mem0 de mentiras que, como el de
    verdad, crea un id nuevo en cada add(infer=False)."""
    mem = _Mem0Falso()
    _suplantar_mem0(monkeypatch, lambda conf: mem)
    return Memoria(MemoriaConfig(enabled=True, ruta=tmp_path / "mem")), mem


def test_recordar_el_mismo_hecho_otra_vez_no_lo_duplica(monkeypatch, tmp_path):
    """Dos copias del mismo hecho ocupaban 2 de los 3 huecos del pre-hook y
    "olvida lo del pozole" borraba una sola: la otra seguía apareciendo."""
    m, mem = _memoria_sobre_mem0_falso(monkeypatch, tmp_path)
    try:
        primero = m.recordar("me gusta el pozole")
        assert primero
        assert m.recordar("me gusta el pozole") == primero
        # La misma frase con otro formato (mayúsculas, espacios, punto final).
        assert m.recordar("  Me gusta   el POZOLE. ") == primero
        assert list(mem.items.values()) == ["me gusta el pozole"]
        assert make_memoria_prehook(m)("¿qué me gusta comer?").count("pozole") == 1
        # Otro hecho, aunque se parezca, sí entra.
        assert m.recordar("me gusta el pozole rojo") != primero
        assert len(mem.items) == 2
    finally:
        m.cerrar()


def test_tool_recordar_repetida_deja_una_copia_y_olvidar_ya_no_deja_otra_viva(
        monkeypatch, tmp_path):
    from crotolamo.tools import memoria as tools_mod

    m, mem = _memoria_sobre_mem0_falso(monkeypatch, tmp_path)
    memoria_mod.set_memoria(m)
    try:
        for _ in range(2):
            assert tools_mod.recordar_de_mi("me gusta el pozole") == (
                "Ya quedó, patrón. Me acordaré: «me gusta el pozole».")
        assert len(mem.items) == 1
        assert tools_mod.olvidar_recuerdo("lo del pozole").startswith("Olvidado")
        assert mem.items == {}
    finally:
        m.cerrar()


def test_recordar_tras_olvidar_lo_vuelve_a_guardar(monkeypatch, tmp_path):
    """La comprobación es contra el backend EN VIVO, no contra una lista en
    memoria: si no, tras "olvida X" un "acuérdate de X" contestaría "ya quedó"
    sin guardar nada."""
    m, mem = _memoria_sobre_mem0_falso(monkeypatch, tmp_path)
    try:
        assert m.recordar("me gusta el pozole")
        assert m.olvidar("el pozole") is not None and mem.items == {}
        assert m.recordar("me gusta el pozole")
        assert list(mem.items.values()) == ["me gusta el pozole"]
    finally:
        m.cerrar()


def test_si_no_se_puede_mirar_lo_guardado_el_hecho_se_guarda_igual(
        monkeypatch, tmp_path, caplog):
    """Un fallo puntual al listar no puede hacer que recordar() pierda el hecho:
    mejor una copia de más que un "ya quedó" en falso."""
    m, mem = _memoria_sobre_mem0_falso(monkeypatch, tmp_path)

    def _revienta(filters: dict, top_k: int) -> dict[str, Any]:
        raise RuntimeError("chroma: timeout")

    mem.get_all = _revienta  # type: ignore[method-assign]
    try:
        with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
            assert m.recordar("me gusta el pozole")
        assert list(mem.items.values()) == ["me gusta el pozole"]
        assert m.enabled and "chroma: timeout" in caplog.text
    finally:
        m.cerrar()


def test_cli_migrar_se_puede_repetir_sin_duplicar(monkeypatch, tmp_path, capsys):
    """Re-migrar es la única vía de traer lo que las tools apuntaron en SQLite
    con la memoria caída; antes duplicaba TODO lo ya migrado."""
    from crotolamo.persistence import facts

    monkeypatch.setattr(facts.db, "_default_db_path", lambda: tmp_path / "f.sqlite")
    facts.remember("mi perro se llama Tletl")
    facts.remember("me gusta el café de olla")
    mem = _Mem0Falso()  # el MISMO almacén entre ejecuciones, como Chroma en disco
    _suplantar_mem0(monkeypatch, lambda conf: mem)
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "mem"))

    assert memoria_mod.run_cli(["migrar"]) == 0
    assert "Migrados 2 de 2 hechos" in capsys.readouterr().out

    lecturas: list[int] = []
    listar = mem.get_all

    def _contada(filters: dict, top_k: int) -> dict[str, Any]:
        lecturas.append(1)
        return listar(filters, top_k)

    mem.get_all = _contada  # type: ignore[method-assign]
    assert memoria_mod.run_cli(["migrar"]) == 0
    out = capsys.readouterr().out
    assert "Migrados 0 de 2 hechos" in out and "2 ya estaban" in out
    assert sorted(mem.items.values()) == ["me gusta el café de olla", "mi perro se llama Tletl"]
    assert len(lecturas) == 1  # lo ya migrado se salta con UNA lectura, no una por hecho

    # Lo apuntado en la libreta mientras la memoria estuvo caída: entra SOLO eso.
    facts.remember("los viernes juego básquet")
    # Repetido dentro de la propia libreta: remember() ya no deja copias, pero
    # una libreta de antes puede traerlas (add_fact es el INSERT crudo).
    facts.db.add_fact("Mi perro se llama Tletl.")
    assert memoria_mod.run_cli(["migrar"]) == 0
    out = capsys.readouterr().out
    assert "Migrados 1 de 4 hechos" in out and "3 ya estaban" in out
    assert len(mem.items) == 3


# --- SQLite: la espera ante bloqueo se fija al abrir; solo las tablas en uso ------

def test_busy_timeout_se_fija_al_abrir_sin_un_pragma_por_operacion(tmp_path, monkeypatch):
    """sqlite3.connect() ya espera 5 s de fábrica: el PRAGMA busy_timeout=3000 de
    cada operación no añadía espera, la bajaba a 3 s con una sentencia de más.
    El valor (3 s) se conserva, puesto en la apertura."""
    import sqlite3

    from crotolamo.persistence import db

    sentencias: list[str] = []
    real = sqlite3.connect

    def _espiar(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real(*args, **kwargs)
        conn.set_trace_callback(sentencias.append)
        return conn

    monkeypatch.setattr(db.sqlite3, "connect", _espiar)
    dbp = tmp_path / "t.sqlite"
    db.add_fact("uno", db_path=dbp)
    assert [f["texto"] for f in db.get_facts(db_path=dbp)] == ["uno"]
    assert any("journal_mode" in s for s in sentencias)  # el espía sí ve los PRAGMA
    assert not [s for s in sentencias if "busy_timeout" in s]
    with db.connect(dbp) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 3000


def test_una_base_nueva_solo_crea_las_tablas_que_se_usan(tmp_path):
    """session_log (y db.log_message) no tenían ningún llamador: la tabla se
    creaba vacía en cada base nueva."""
    from crotolamo.persistence import db

    with db.connect(tmp_path / "nueva.sqlite") as conn:
        tablas = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tablas - {"sqlite_sequence"} == {"shortcuts", "facts"}


def test_una_base_vieja_con_session_log_sigue_funcionando_y_no_se_toca(tmp_path):
    import sqlite3

    from crotolamo.persistence import db

    dbp = tmp_path / "vieja.sqlite"
    vieja = sqlite3.connect(dbp)
    vieja.executescript(
        "CREATE TABLE session_log (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL,"
        " content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));"
        "INSERT INTO session_log (role, content) VALUES ('user', 'hola');")
    vieja.commit()
    vieja.close()

    db.add_fact("uno", db_path=dbp)
    db.save_shortcut("x", "url", {"value": "a"}, db_path=dbp)
    assert [f["texto"] for f in db.get_facts(db_path=dbp)] == ["uno"]
    with db.connect(dbp) as conn:
        assert [r[0] for r in conn.execute("SELECT content FROM session_log")] == ["hola"]


# --- libreta SQLite: puntuar una vez, no borrar a ciegas, no guardar secretos -----

@pytest.fixture
def libreta(tmp_path, monkeypatch):
    """La libreta de hechos SQLite sobre un fichero temporal."""
    from crotolamo.persistence import facts

    monkeypatch.setattr(facts.db, "_default_db_path", lambda: tmp_path / "f.sqlite")
    return facts


def test_search_puntua_cada_hecho_una_sola_vez_con_el_mismo_resultado(libreta, monkeypatch):
    """Antes _score (SequenceMatcher, lo caro) corría dos veces por hecho: una
    para ordenar y otra para filtrar."""
    hechos = ["uso Fedora con Hyprland", "me gusta el café de olla", "mi perro se llama Tletl",
              "mi perro le teme a los cohetes", "los viernes juego básquet", "mi perro es bravo"]
    for h in hechos:
        libreta.remember(h)
    real = libreta._score
    filas = libreta.recall()
    esperado = [r["texto"] for r in sorted(
        filas, key=lambda r: real("mi perro", r["texto"]), reverse=True)][:4]

    llamadas: list[str] = []

    def _contado(query: str, texto: str) -> float:
        llamadas.append(texto)
        return real(query, texto)

    monkeypatch.setattr(libreta, "_score", _contado)
    assert [r["texto"] for r in libreta.search("mi perro", top=4)] == esperado
    assert sorted(llamadas) == sorted(hechos)  # una por hecho, no dos


_HECHOS_LIBRETA = ("uso Fedora con Hyprland", "me gusta el café de olla",
                   "mi perro se llama Tletl")


@pytest.fixture
def tools_sin_memoria(libreta):
    """Las tools de memoria con la semántica fuera de juego: usan la libreta."""
    from crotolamo.tools import memoria as tools_mod

    m = Memoria(MemoriaConfig(enabled=False))
    memoria_mod.set_memoria(m)
    for h in _HECHOS_LIBRETA:
        libreta.remember(h)
    yield tools_mod
    m.cerrar()


@pytest.mark.parametrize("descripcion", [
    "lo de mi coche",          # comparte "de" con el café y "mi" con el perro
    "la marca de mi coche",
    "xyzzy 123",
    "eso",
])
def test_respaldo_sqlite_no_borra_un_hecho_que_no_trata_de_lo_pedido(
        tools_sin_memoria, libreta, descripcion):
    """facts.search ordena por parecido pero no descarta nada (dos frases
    cualesquiera comparten una letra o un "de"): con top=1, "olvida lo de mi
    coche" BORRABA «me gusta el café de olla»."""
    out = tools_sin_memoria.olvidar_recuerdo(descripcion)
    assert out.startswith("No encontré ningún recuerdo") and descripcion in out
    assert [r["texto"] for r in libreta.recall()] == list(_HECHOS_LIBRETA)


@pytest.mark.parametrize("descripcion, borrado", [
    ("mi perro", "mi perro se llama Tletl"),
    ("olvida lo del cafe", "me gusta el café de olla"),  # sin acento, como lo deja el STT
    ("lo de FEDORA", "uso Fedora con Hyprland"),
])
def test_respaldo_sqlite_sigue_olvidando_el_hecho_que_si_encaja(
        tools_sin_memoria, libreta, descripcion, borrado):
    out = tools_sin_memoria.olvidar_recuerdo(descripcion)
    assert out.startswith(f"Olvidado, patrón: «{borrado}»")
    assert [r["texto"] for r in libreta.recall()] == [h for h in _HECHOS_LIBRETA if h != borrado]


def test_respaldo_sqlite_borra_el_que_trata_del_tema_aunque_otro_se_parezca_mas(
        tools_sin_memoria, libreta):
    """El más "parecido" por letras y palabras vacías no es el que trata del tema."""
    libreta.remember("lo mejor de mi semana es el domingo")
    libreta.remember("vendí el coche")
    assert libreta.search("lo de mi coche", top=1)[0]["texto"].startswith("lo mejor")
    out = tools_sin_memoria.olvidar_recuerdo("lo de mi coche")
    assert out.startswith("Olvidado, patrón: «vendí el coche»")
    assert "lo mejor de mi semana es el domingo" in [r["texto"] for r in libreta.recall()]


def test_respaldo_sqlite_no_toma_el_te_por_la_palabra_vacia_te(tools_sin_memoria, libreta):
    """Las palabras vacías se descartan antes de quitar acentos: "té" no es "te"."""
    libreta.remember("tomo té por las tardes")
    assert tools_sin_memoria.olvidar_recuerdo("lo que te dije de mi coche").startswith(
        "No encontré")
    out = tools_sin_memoria.olvidar_recuerdo("lo del té")
    assert out.startswith("Olvidado, patrón: «tomo té por las tardes»")


@pytest.mark.parametrize("secreto", [
    "la contraseña del wifi es casa-2024-segura",
    "mi api key de groq es gsk_abcDEF1234567890abcdef",
    "mi nip es 4821",
])
def test_remember_fact_no_guarda_secretos(libreta, secreto):
    """La tool de fábrica ([memoria].enabled = false) se saltaba el filtro que sí
    aplican recordar_de_mi y su respaldo en esta MISMA libreta: la credencial
    quedaba en claro en SQLite y el pre-hook de hechos la reenviaba al LLM (en la
    nube) al empezar cada sesión."""
    from crotolamo.tools import facts as facts_tools
    from crotolamo.tools import memoria as tools_mod

    out = facts_tools.remember_fact(secreto, "datos")
    assert out == tools_mod._recordar_sqlite(secreto)  # la misma frase, en personaje
    assert out.startswith("Eso tiene pinta") and secreto not in out
    assert libreta.recall() == []
    assert libreta.facts_context() == ""  # nada que reenviar a la nube


def test_remember_fact_sigue_guardando_lo_normal(libreta):
    from crotolamo.tools import facts as facts_tools

    assert facts_tools.remember_fact("mi editor es geany", "preferencias").startswith("Anotado")
    assert facts_tools.remember_fact("la clave del éxito es la constancia").startswith("Anotado")
    assert [r["texto"] for r in libreta.recall()] == [
        "mi editor es geany", "la clave del éxito es la constancia"]


# --- "no pude buscar" no es "no hay nada" (tools y CLI) ---------------------------

class _BackendQueNoPuedeBuscar:
    """Backend SANO (un recuerdo guardado) cuya búsqueda falla o tarda: un error
    puntual de Chroma, el modelo aún cargando... La memoria no queda marcada rota."""

    def __init__(self, error: Exception | None = None, tardar_s: float = 0.0) -> None:
        self.items = {"m1": "mi perro se llama Tletl"}
        self.error = error
        self.tardar_s = tardar_s
        self.borrados: list[str] = []

    def guardar(self, texto: str) -> list[str]:
        return []

    def extraer(self, usuario: str, respuesta: str) -> list[str]:
        return []

    def buscar(self, query: str, top_k: int, umbral: float) -> list[memoria_mod.Recuerdo]:
        if self.tardar_s:
            time.sleep(self.tardar_s)
        if self.error is not None:
            raise self.error
        return [memoria_mod.Recuerdo(k, v, 0.05) for k, v in self.items.items()][:top_k]

    def todas(self) -> list[memoria_mod.Recuerdo]:
        return [memoria_mod.Recuerdo(k, v) for k, v in self.items.items()]

    def borrar(self, memoria_id: str) -> None:
        self.borrados.append(memoria_id)
        self.items.pop(memoria_id, None)


_FALLO_PUNTUAL = "chroma: disk I/O error"


@pytest.fixture
def memoria_que_no_puede_buscar():
    backend = _BackendQueNoPuedeBuscar(RuntimeError(_FALLO_PUNTUAL))
    m = Memoria(MemoriaConfig(enabled=True), backend=backend)
    memoria_mod.set_memoria(m)
    yield m, backend
    m.cerrar()


def test_buscar_recuerdos_no_contesta_no_recuerdo_nada_si_la_busqueda_fallo(
        memoria_que_no_puede_buscar):
    """El recuerdo ESTÁ guardado: "No recuerdo nada sobre «mi perro»" era falso, y
    es lo que el modelo le repetía al patrón."""
    from crotolamo.tools import memoria as tools_mod

    m, backend = memoria_que_no_puede_buscar
    out = tools_mod.buscar_recuerdos("mi perro")
    assert "No recuerdo nada" not in out
    assert out.startswith("No pude consultar la memoria") and _FALLO_PUNTUAL in out
    assert m.enabled  # un fallo puntual no la marca rota...
    backend.error = None
    assert "Tletl" in tools_mod.buscar_recuerdos("mi perro")  # ...y la siguiente ya responde


def test_olvidar_recuerdo_no_contesta_no_encontre_si_la_busqueda_fallo(
        memoria_que_no_puede_buscar, libreta):
    from crotolamo.tools import memoria as tools_mod

    m, backend = memoria_que_no_puede_buscar
    libreta.remember("mi perro se llama Tletl")  # la libreta de respaldo NO se toca
    out = tools_mod.olvidar_recuerdo("mi perro")
    assert "No encontré" not in out and not out.startswith("Olvidado")
    assert out.startswith("No pude consultar la memoria") and _FALLO_PUNTUAL in out
    assert "no borré nada" in out
    assert backend.borrados == [] and len(libreta.recall()) == 1
    assert m.enabled
    backend.error = None
    assert tools_mod.olvidar_recuerdo("mi perro").startswith("Olvidado, patrón: «mi perro")


def test_buscar_estricto_distingue_sin_resultados_de_no_pude_buscar(caplog):
    """buscar() sigue siendo la versión silenciosa del pre-hook ([] pase lo que
    pase); buscar_estricto es para quien va a CONTESTAR con el resultado."""
    backend = _BackendQueNoPuedeBuscar(tardar_s=0.4)
    m = Memoria(MemoriaConfig(enabled=True), backend=backend)
    try:
        with caplog.at_level(logging.WARNING, logger="crotolamo.core.memoria"):
            with pytest.raises(memoria_mod.BusquedaFallida, match="tardó más de 0.1"):
                m.buscar_estricto("mi perro", timeout_s=0.1)
            assert m.buscar("mi perro", timeout_s=0.1) == []
        assert caplog.text.count("tardó más") == 2  # el WARNING de siempre, en las dos
        time.sleep(0.5)  # deja que los dos hilos lentos terminen
        backend.tardar_s = 0.0
        backend.error = RuntimeError(_FALLO_PUNTUAL)
        with pytest.raises(memoria_mod.BusquedaFallida, match="disk I/O"):
            m.buscar_estricto("mi perro")
        assert m.buscar("mi perro") == [] and m.enabled
        backend.error = None
        backend.items.clear()
        assert m.buscar_estricto("mi perro") == []  # sin resultados: lista vacía, no error
        assert m.buscar_estricto("   ") == []
    finally:
        m.cerrar()


def test_buscar_estricto_con_la_memoria_caida_lanza_no_disponible_y_la_marca_rota():
    backend = _BackendQueNoPuedeBuscar(MemoriaNoDisponible("falta mem0"))
    m = Memoria(MemoriaConfig(enabled=True), backend=backend)
    try:
        with pytest.raises(MemoriaNoDisponible, match="falta mem0"):
            m.buscar_estricto("mi perro")
        assert not m.enabled and m.motivo_no_disponible == "falta mem0"
        with pytest.raises(MemoriaNoDisponible, match="falta mem0"):
            m.buscar_estricto("mi perro")  # ya marcada: ni toca el backend
        assert m.olvidar("mi perro") is None and m.buscar("mi perro") == []
    finally:
        m.cerrar()


def _cli_con_backend(monkeypatch, backend: Any) -> None:
    _settings_con_memoria(monkeypatch)
    monkeypatch.setattr(memoria_mod, "Memoria",
                        lambda cfg, backend_=None, settings=None: Memoria(cfg, backend=backend))


@pytest.mark.parametrize("sub", [["buscar", "mi perro"], ["olvidar", "mi perro"]])
def test_cli_sin_mem0_avisa_con_codigo_1_tambien_al_buscar_y_olvidar(
        sub, monkeypatch, tmp_path, capsys):
    """`listar` y `migrar` ya decían "Memoria no disponible" con código 1; `buscar`
    no imprimía nada y `olvidar` decía "No encontré nada parecido.", los dos con 0."""
    monkeypatch.setitem(sys.modules, "mem0", None)  # import mem0 -> ImportError
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "mem"))
    assert memoria_mod.run_cli(sub) == 1
    out = capsys.readouterr().out
    assert "Memoria no disponible" in out and "falta mem0" in out
    assert "No encontré" not in out


@pytest.mark.parametrize("sub", [["buscar", "mi perro"], ["olvidar", "mi perro"]])
def test_cli_con_un_fallo_puntual_de_busqueda_lo_dice_con_codigo_1(sub, monkeypatch, capsys):
    backend = _BackendQueNoPuedeBuscar(RuntimeError(_FALLO_PUNTUAL))
    _cli_con_backend(monkeypatch, backend)
    assert memoria_mod.run_cli(sub) == 1
    out = capsys.readouterr().out
    assert "No pude buscar" in out and _FALLO_PUNTUAL in out
    assert "No encontré" not in out and backend.borrados == []


def test_cli_buscar_sin_resultados_lo_dice_en_vez_de_callar(monkeypatch, capsys):
    backend = _BackendQueNoPuedeBuscar()
    backend.items.clear()
    _cli_con_backend(monkeypatch, backend)
    assert memoria_mod.run_cli(["buscar", "mi coche"]) == 0
    assert capsys.readouterr().out.strip() == "No encontré nada parecido."


# --- tools de hechos: `null` en la categoría y filtro que no depende de la caja ----

def _registro_de_hechos():
    from crotolamo.tools import facts as facts_tools
    from crotolamo.tools.base import Registry

    reg = Registry()
    for func in (facts_tools.remember_fact, facts_tools.recall_facts):
        reg.register(func._crotolamo_tool)  # type: ignore[attr-defined]
    return reg


def test_tools_de_hechos_aceptan_null_en_la_categoria_opcional(libreta):
    """Los modelos mandan `null` en los parámetros opcionales: las dos tools
    hacían None.strip() y contestaban "La tool ... reventó" (en remember_fact, sin
    guardar el hecho)."""
    reg = _registro_de_hechos()
    assert reg.run("recall_facts", {"categoria": None}) == "No tengo hechos guardados, patrón."
    assert reg.run("remember_fact", {"texto": "uso Fedora", "categoria": None}).startswith(
        "Anotado")
    assert [(r["texto"], r["categoria"]) for r in libreta.recall()] == [("uso Fedora", "general")]
    out = reg.run("recall_facts", {"categoria": None})
    assert out.startswith("Esto es lo que recuerdo") and "uso Fedora" in out
    assert libreta.remember("otro hecho", None) > 0  # type: ignore[arg-type]


@pytest.mark.parametrize("guardada, pedida", [
    ("Preferencias", "preferencias"),
    ("Preferencias", "PREFERENCIAS"),
    ("preferencias", " Preferencias "),
    ("música", "Musica"),
    ("Música", "MÚSICA"),
    ("datos del coche", "Datos  del Coche"),
])
def test_recall_facts_encuentra_la_categoria_aunque_el_modelo_la_escriba_distinto(
        libreta, guardada, pedida):
    """La etiqueta la escribe el modelo en cada llamada; con `categoria = ?` exacto,
    recall_facts("preferencias") contestaba "No tengo hechos en la categoría" con
    el hecho guardado en "Preferencias"."""
    from crotolamo.tools import facts as facts_tools

    facts_tools.remember_fact("me gusta el pozole", guardada)
    facts_tools.remember_fact("uso Fedora", "sistema")
    out = facts_tools.recall_facts(pedida)
    assert out.startswith("Esto es lo que recuerdo")
    assert "me gusta el pozole" in out and "Fedora" not in out
    assert f"({guardada})" in out  # se muestra como se guardó: el dato no se reescribe


def test_recall_facts_sigue_filtrando_y_avisando_si_la_categoria_no_existe(libreta):
    from crotolamo.persistence import db
    from crotolamo.tools import facts as facts_tools

    facts_tools.remember_fact("me gusta el pozole", "Preferencias")
    facts_tools.remember_fact("uso Fedora", "sistema")
    assert facts_tools.recall_facts("coches") == "No tengo hechos en la categoría 'coches', patrón."
    assert facts_tools.recall_facts("pref") == "No tengo hechos en la categoría 'pref', patrón."
    assert [f["texto"] for f in db.get_facts("SISTEMA")] == ["uso Fedora"]
    assert len(db.get_facts()) == 2 and len(db.get_facts("")) == 2


# --- calibrar: sin key no revienta y anuncia el LLM que de verdad se usa ----------

def test_calibrar_sin_key_de_nube_no_revienta_ni_con_groq(monkeypatch, tmp_path, capsys):
    """Sin key, mem0 lanzaba al construir el cliente del LLM y el script moría con
    un traceback aunque el modo por defecto no usa LLM (ya arreglado en Mem0Backend
    con la key de relleno; aquí se fija para el script)."""
    from scripts import memoria_calibrar

    _mem0_como_el_real(monkeypatch)
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "real"))
    assert memoria_calibrar.main([]) == 0
    assert "aciertos top-1" in capsys.readouterr().out
    assert memoria_calibrar.main(["--groq"]) == 0
    out = capsys.readouterr().out
    assert "No hay API key para el LLM de extracción" in out and "extracción real" not in out


def test_calibrar_groq_anuncia_el_proveedor_y_el_modelo_efectivos(monkeypatch, tmp_path, capsys):
    """La cabecera imprimía [memoria].llm_provider y llm_model SIN resolver: con los
    valores de fábrica, "(auto / )" en vez del proveedor y el modelo que se prueban."""
    from scripts import memoria_calibrar

    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "clave-de-zai")
    arrancados = _mem0_como_el_real(monkeypatch)
    _settings_con_memoria(monkeypatch, ruta=str(tmp_path / "real"))
    assert memoria_calibrar.main(["--groq"]) == 0
    out = capsys.readouterr().out
    llm = arrancados[0][0]["llm"]  # lo que mem0 recibió de verdad
    assert llm["provider"] != "auto" and llm["config"]["model"]
    assert f"== extracción real con el LLM ({llm['provider']} / {llm['config']['model']}) ==" in out
    assert "(auto / )" not in out
    assert "clave-de-zai" not in out  # nunca imprime la API key


# --- respaldo SQLite de olvidar: también cuando la memoria cae EN CALIENTE ---------

def test_olvidar_con_la_memoria_caida_en_caliente_no_borra_un_hecho_ajeno(libreta):
    """Los tests del respaldo de arriba entran con [memoria] apagada; aquí está
    encendida y mem0 cae DURANTE olvidar_recuerdo (la otra entrada al respaldo).
    Con el top-1 a ciegas, "mi perro Rocky" sin ningún hecho del perro BORRABA
    «el patron toma el cafe sin azucar» y encima lo anunciaba como olvidado."""
    from crotolamo.tools import memoria as tools_mod

    hechos = ["el patron toma el cafe sin azucar", "su hermana se llama Ana"]
    for h in hechos:
        libreta.remember(h)
    backend = _BackendQueNoPuedeBuscar(MemoriaNoDisponible("chroma ilegible"))
    m = Memoria(MemoriaConfig(enabled=True), backend=backend)
    memoria_mod.set_memoria(m)
    try:
        # 1ª llamada: la búsqueda lanza MemoriaNoDisponible dentro de olvidar().
        out = tools_mod.olvidar_recuerdo("mi perro Rocky")
        assert out.startswith("No encontré ningún recuerdo") and "libreta" in out
        assert not m.enabled
        # 2ª: ya está marcada rota y entra por la rama `not memoria.enabled`.
        assert tools_mod.olvidar_recuerdo("olvida lo de mi perro Rocky").startswith("No encontré")
        assert [r["texto"] for r in libreta.recall()] == hechos
        assert backend.borrados == []
        # El hecho que sí trata de lo pedido se sigue pudiendo olvidar.
        out = tools_mod.olvidar_recuerdo("lo de mi hermana")
        assert out.startswith("Olvidado, patrón: «su hermana se llama Ana»")
        assert [r["texto"] for r in libreta.recall()] == hechos[:1]
    finally:
        m.cerrar()


# --- al salir del proceso se termina la extracción pendiente (acotado) -------------

# Lo que hace run_shell al teclear "salir": vuelve sin flush() ni cerrar() y el
# intérprete termina. El hilo de extracción es daemon: sin un vaciado al salir, el
# intérprete lo mataba a media llamada al LLM y el hecho del último turno se perdía.
_SONDA_SALIDA = r"""
import sys, time
from pathlib import Path
from crotolamo.core import memoria as memoria_mod
from crotolamo.core.memoria import Memoria, MemoriaConfig, make_memoria_posthook

marca, tardar_s = Path(sys.argv[1]), float(sys.argv[2])
if len(sys.argv) > 3:
    memoria_mod._ESPERA_EXTRACCION_AL_SALIR_S = float(sys.argv[3])


class Backend:
    def extraer(self, usuario, respuesta):
        time.sleep(tardar_s)  # la llamada al LLM de extracción (1-3 s de verdad)
        marca.write_text(usuario, encoding="utf-8")
        return [usuario]


m = Memoria(MemoriaConfig(enabled=True), backend=Backend())
make_memoria_posthook(m)("tengo una perrita que se llama Nala", "Qué bonito, patrón.")
print("saliendo", flush=True)
"""


def _salir_con_extraccion_pendiente(tmp_path, *args: str) -> tuple[Any, float]:
    import subprocess
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _SONDA_SALIDA, str(tmp_path / "marca"), *args],
        cwd=raiz, capture_output=True, text=True, timeout=60, check=False,
    )
    return proc, time.monotonic() - t0


def test_al_salir_del_proceso_se_termina_la_extraccion_pendiente(tmp_path):
    """Antes: el proceso salía en ~0.1 s y la marca de la extracción no existía."""
    proc, _ = _salir_con_extraccion_pendiente(tmp_path, "0.5")
    assert proc.returncode == 0, proc.stderr
    assert "saliendo" in proc.stdout
    marca = tmp_path / "marca"
    assert marca.exists(), "la extracción del último turno se perdió al salir"
    assert marca.read_text(encoding="utf-8") == "tengo una perrita que se llama Nala"


def test_al_salir_la_espera_por_la_extraccion_esta_acotada(tmp_path):
    """Una nube colgada (el SDK espera hasta 600 s) no puede dejar el shell sin
    terminar: se espera lo acotado y el hilo daemon muere con el proceso."""
    proc, tardo = _salir_con_extraccion_pendiente(tmp_path, "120", "0.3")
    assert proc.returncode == 0, proc.stderr
    assert tardo < 30.0
    assert not (tmp_path / "marca").exists()


def test_el_vaciado_al_salir_se_apunta_una_vez_y_cerrar_lo_retira(monkeypatch):
    """Solo se apunta la Memoria que de verdad encoló algo, una vez aunque el
    hilo se recree, y cerrar() (que ya vacía la cola) lo retira."""
    apuntados: list[Callable[[], None]] = []
    retirados: list[Callable[[], None]] = []
    monkeypatch.setattr(memoria_mod.atexit, "register", apuntados.append)
    monkeypatch.setattr(memoria_mod.atexit, "unregister", retirados.append)

    class _Backend:
        def extraer(self, usuario: str, respuesta: str) -> list[str]:
            return [usuario]

    quieta = Memoria(MemoriaConfig(enabled=True), backend=_Backend())  # type: ignore[arg-type]
    m = Memoria(MemoriaConfig(enabled=True), backend=_Backend())  # type: ignore[arg-type]
    try:
        assert m.extraer_en_fondo("me gusta el pozole", "va")
        assert m.flush()
        m._cola.put(None)  # el hilo termina; el siguiente turno crea otro
        assert m._worker is not None
        m._worker.join(timeout=3.0)
        assert m.extraer_en_fondo("uso Fedora", "va")
        assert m.flush()
        assert apuntados == [m._vaciar_al_salir]
    finally:
        m.cerrar()
        quieta.cerrar()
    assert m._vaciar_al_salir in retirados
