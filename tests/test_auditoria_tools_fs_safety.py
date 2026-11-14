"""Auditoría exhaustiva — grupo tools / files / safety / settings.

Cada test fija un hallazgo confirmado de la auditoría: falla con el código de
antes y pasa con el arreglo. Los comentarios dicen QUÉ se rompía.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crotolamo import settings as settings_mod
from crotolamo.safety.guard import Guard
from crotolamo.tools.base import Tool


def _tool(name: str = "t", safe: bool = True) -> Tool:
    return Tool(name=name, func=lambda **k: "ok", description="d", parameters={}, safe=safe)


@pytest.fixture
def zonas(tmp_path):
    """Guard con zona libre y zona de confirmación separadas bajo tmp_path."""
    libre = tmp_path / "libre"
    confirmable = tmp_path / "confirmable"
    libre.mkdir()
    confirmable.mkdir()
    return Guard(allowed_roots=[libre], confirm_roots=[confirmable]), libre, confirmable


# ---------------------------------------------------------------------------
# Guard: el texto libre de las tools de memoria no es una ruta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valor", [
    "~/proyectos es donde guardo todo",
    "/etc/hosts lo edité ayer",
    "./notas de la reunión",
])
def test_guard_no_toma_por_ruta_el_texto_libre_de_las_tools_de_memoria(tmp_path, valor):
    """`hecho` ya estaba exento, pero `pregunta` (buscar_recuerdos) y
    `descripcion` (olvidar_recuerdo) no: "olvida lo de ~/proyectos" pedía
    confirmación (cancelar, en el modo concurrente) y "/etc/hosts lo edité
    ayer" se bloqueaba como ruta fuera del corral."""
    from crotolamo.tools.memoria import memoria_tools

    guard = Guard([tmp_path])
    vistos = []
    for tool in memoria_tools():
        for param in tool.parameters["properties"]:
            decision = guard.check(tool, {param: valor})
            vistos.append(param)
            assert decision.allowed and not decision.needs_confirmation, (tool.name, param)
    assert {"hecho", "pregunta", "descripcion"} <= set(vistos)


def test_guard_no_toma_por_ruta_la_categoria_de_un_hecho(tmp_path):
    """remember_fact(categoria="/sistema") se bloqueaba: la etiqueta es texto."""
    from crotolamo.tools import facts

    guard = Guard([tmp_path])
    remember = facts.remember_fact._crotolamo_tool
    recall = facts.recall_facts._crotolamo_tool
    assert guard.check(remember, {"texto": "uso Fedora", "categoria": "/sistema"}) \
        == guard.check(remember, {"texto": "uso Fedora", "categoria": "sistema"})
    decision = guard.check(recall, {"categoria": "~/proyectos"})
    assert decision.allowed and not decision.needs_confirmation


# ---------------------------------------------------------------------------
# Guard: URIs file:// y valores con espacios/saltos de línea en las puntas
# (para las tools MCP el guard es el ÚNICO corral)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uri", [
    "file:///etc/passwd",
    "FILE:///etc/passwd",
    "file:/etc/passwd",
    "file://localhost/etc/passwd",
    "file:///etc/%70asswd",
    "  file:///etc/passwd\n",
    # Un parser de URLs quita los tabs/saltos de línea de en medio y los
    # controles de las puntas: ni "varias líneas = contenido" ni un \x01 lo eximen.
    "file:///etc/pass\nwd",
    "file:///etc/passwd#\nnota",
    "fi\tle:///etc/passwd",
    "\x01file:///etc/passwd",
])
def test_guard_bloquea_un_uri_file_fuera_del_corral(zonas, uri):
    """Un server MCP de navegador/fetch abre file:///... igual que una ruta, pero
    el valor no empieza por "/" y pasaba LIBRE bajo cualquier nombre."""
    guard, _libre, _confirmable = zonas
    for nombre in ("uri", "url", "target"):
        decision = guard.check(_tool(), {nombre: uri})
        assert not decision.allowed, (nombre, uri)
        assert "permitidas" in decision.reason


def test_guard_clasifica_un_uri_file_en_las_tres_zonas(zonas):
    guard, libre, confirmable = zonas
    dentro = guard.check(_tool(), {"url": f"file://{libre}/doc%20final.html"})
    assert dentro.allowed and not dentro.needs_confirmation
    pregunta = guard.check(_tool(), {"url": f"file://{confirmable}/doc.html"})
    assert pregunta.allowed and pregunta.needs_confirmation
    assert "zona libre" in pregunta.reason
    # Anidado en el JSON de una tool MCP también.
    assert not guard.check(_tool(), {"pasos": [{"abrir": "file:///etc/shadow"}]}).allowed


def test_guard_uri_file_bajo_nombre_de_ruta_no_depende_del_cwd(zonas, monkeypatch):
    """Bajo un nombre de ruta, "file:///etc/passwd" se resolvía como ruta
    RELATIVA al CWD (cwd/file:/etc/passwd): lanzado desde la zona libre, libre."""
    guard, libre, _confirmable = zonas
    monkeypatch.chdir(libre)
    assert not guard.check(_tool(), {"path": "file:///etc/passwd"}).allowed
    assert not guard.check(_tool(), {"paths": ["a.txt", "file:///etc/passwd"]}).allowed


def test_guard_uri_file_bajo_nombre_de_ruta_cuenta_tambien_tal_cual(zonas, monkeypatch):
    """Una tool nativa no entiende file://: abre el valor como ruta relativa al
    CWD. Que el URI apunte a la zona libre no le ahorra la pregunta (mirar el
    URI no puede aflojar lo que ya se clasificaba)."""
    guard, libre, confirmable = zonas
    monkeypatch.chdir(confirmable)
    decision = guard.check(_tool(), {"path": f"file://{libre}/x.txt"})
    assert decision.allowed and decision.needs_confirmation


@pytest.mark.parametrize("valor", [" /etc/passwd", "/etc/passwd\n", "\t/etc/passwd  ",
                                    "\n/etc/passwd"])
def test_guard_bloquea_rutas_con_espacios_o_saltos_en_las_puntas(zonas, monkeypatch, valor):
    """" /etc/passwd" no "empezaba por /" y "/etc/passwd\\n" contaba como
    contenido por llevar un salto de línea: los dos pasaban libres, y un server
    que recorte el argumento abre /etc/passwd."""
    guard, libre, _confirmable = zonas
    # Desde la zona libre, para que la ruta tal cual (relativa al CWD) no tape
    # el fallo: lo que tiene que bloquear es la ruta recortada.
    monkeypatch.chdir(libre)
    for nombre in ("target", "path"):
        decision = guard.check(_tool(), {nombre: valor})
        assert not decision.allowed, (nombre, valor)
        # El motivo (lo oye el patrón) nombra la ruta limpia, sin el relleno.
        assert "'/etc/passwd'" in decision.reason


def test_guard_con_relleno_gana_la_zona_mas_estricta(zonas, monkeypatch):
    """Las tools nativas NO recortan: " <libre>/x" lo abren relativo al CWD. El
    guard mira las dos lecturas y manda la más estricta (aquí: confirmar)."""
    guard, libre, confirmable = zonas
    monkeypatch.chdir(confirmable)
    # Bajo un nombre cualquiera ("target") la lectura relativa solo cuenta para
    # una tool MCP: una nativa no abre contra el CWD lo que no es su argumento
    # de ruta (T10, ver los tests de rutas relativas más abajo).
    remota = Tool(name="t", func=lambda **k: "ok", description="d", parameters={},
                  strict_args=False)
    for nombre, tool in (("path", _tool()), ("path", remota), ("target", remota)):
        decision = guard.check(tool, {nombre: f" {libre}/x.txt"})
        assert decision.allowed and decision.needs_confirmation, nombre
    # Sin relleno no hay dos lecturas: la ruta de la zona libre corre directa.
    limpia = guard.check(_tool(), {"path": f"{libre}/x.txt"})
    assert limpia.allowed and not limpia.needs_confirmation


def test_guard_sigue_tratando_como_contenido_lo_que_es_contenido(zonas):
    guard, libre, _confirmable = zonas
    # Varias líneas = contenido (un script), aunque el nombre no esté en la lista.
    script = guard.check(_tool(), {"codigo": "/usr/bin/env python3\nprint(1)\n"})
    assert script.allowed and not script.needs_confirmation
    # Argumento de contenido de primer nivel: ni el prefijo ni el file:// cuentan.
    nota = guard.check(_tool(), {
        "path": str(libre / "nota.md"), "content": "file:///etc/passwd es la ruta\n",
    })
    assert nota.allowed and not nota.needs_confirmation
    # Una URL web no es una ruta.
    web = guard.check(_tool(), {"url": "https://ejemplo.com/file:/x"})
    assert web.allowed and not web.needs_confirmation


def test_guard_uri_file_malformado_no_revienta_ni_pasa(zonas):
    """urlsplit lanza ValueError con un host IPv6 a medias: el guard no puede
    reventar (rompería el turno) ni dejarlo pasar por no saber leerlo."""
    guard, _libre, _confirmable = zonas
    assert not guard.check(_tool(), {"url": "file://[::1/etc/passwd"}).allowed


# ---------------------------------------------------------------------------
# Settings: crotolamo.local.toml solo pisa la config por defecto
# ---------------------------------------------------------------------------

def test_load_settings_con_ruta_explicita_no_mezcla_el_local(tmp_path, monkeypatch):
    """load_settings(ruta) fusionaba SIEMPRE config/crotolamo.local.toml: quien
    cargaba un toml propio (los tests) recibía encima las rutas permitidas y los
    proyectos locales del patrón."""
    local = tmp_path / "crotolamo.local.toml"
    local.write_text(
        '[paths]\nallowed_roots = ["~/Proyectos"]\n[projects]\nmio = "~/Proyectos/mio"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_mod, "LOCAL_CONFIG", local)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[paths]\nhome = "~"\nallowed_roots = ["~/Documentos"]\n', encoding="utf-8")

    s = settings_mod.load_settings(cfg)
    assert s.allowed_roots == [Path("~/Documentos").expanduser()]
    assert s.projects == {}


def test_load_settings_por_defecto_sigue_fusionando_el_local(tmp_path, monkeypatch):
    base = tmp_path / "crotolamo.toml"
    base.write_text(
        '[paths]\nallowed_roots = ["~/Documentos"]\n'
        '[voice]\nwhisper_model = "base"\nlang = "es"\n',
        encoding="utf-8",
    )
    local = tmp_path / "crotolamo.local.toml"
    local.write_text('[voice]\nwhisper_model = "small"\n', encoding="utf-8")
    monkeypatch.setattr(settings_mod, "DEFAULT_CONFIG", base)
    monkeypatch.setattr(settings_mod, "LOCAL_CONFIG", local)

    s = settings_mod.load_settings()
    assert s.voice == {"whisper_model": "small", "lang": "es"}
    assert s.allowed_roots == [Path("~/Documentos").expanduser()]


# ---------------------------------------------------------------------------
# _split_doc: la descripción de un parámetro puede ocupar varias líneas
# ---------------------------------------------------------------------------

def test_split_doc_une_las_lineas_de_continuacion_de_un_parametro():
    """Dentro de 'Args:' solo se guardaba la línea 'nombre: texto'; las que
    seguían la frase se tiraban y el LLM veía la descripción cortada a medias."""
    from crotolamo.tools.base import _split_doc

    doc = '''Prende, apaga o alterna una luz.

    Args:
        target: nombre amigable de la luz, como está en la config
            (por ejemplo "xbox").
        action: qué hacer con la luz: "on" (prender), "off" (apagar) o
            "toggle" (alternar).
    '''
    description, params = _split_doc(doc)
    assert description == "Prende, apaga o alterna una luz."
    assert params == {
        "target": 'nombre amigable de la luz, como está en la config (por ejemplo "xbox").',
        "action": 'qué hacer con la luz: "on" (prender), "off" (apagar) o "toggle" (alternar).',
    }


def test_split_doc_continuacion_con_pinta_de_parametro_no_abre_otro():
    """Una línea de continuación (más sangrada) que parece 'nombre: texto'
    sigue siendo parte del parámetro que continúa."""
    from crotolamo.tools.base import _split_doc

    doc = '''Hace algo.

    Args:
        modo: cómo hacerlo. Opciones
            rapido: sin comprobar nada.
        limite: tope de resultados.
    '''
    _description, params = _split_doc(doc)
    assert params == {
        "modo": "cómo hacerlo. Opciones rapido: sin comprobar nada.",
        "limite": "tope de resultados.",
    }


def test_split_doc_no_pega_a_un_parametro_lo_que_viene_tras_el_bloque():
    """La continuación acaba con una línea en blanco o con una sangría menor:
    una sección posterior del docstring no es descripción del último parámetro."""
    from crotolamo.tools.base import _split_doc

    doc = '''Hace algo.

    Args:
        ruta: archivo a leer,
            relativo al proyecto.

        Nota suelta que no documenta ningún parámetro.
    Returns:
        el contenido del archivo.
    '''
    description, params = _split_doc(doc)
    assert description == "Hace algo."
    assert params == {"ruta": "archivo a leer, relativo al proyecto."}


def test_el_esquema_de_las_tools_reales_lleva_la_descripcion_completa():
    """light_control.action perdía la opción "toggle" y camera_events.camera
    perdía cómo pedir todas las cámaras: justo lo que el modelo necesita."""
    from crotolamo.tools import cameras, home

    action = home.light_control._crotolamo_tool.parameters["properties"]["action"]
    assert "toggle" in action["description"]
    camera = cameras.camera_events._crotolamo_tool.parameters["properties"]["camera"]
    assert "todas" in camera["description"]


# ---------------------------------------------------------------------------
# Registry.run: "argumentos inválidos" solo cuando de verdad lo son (C4)
# ---------------------------------------------------------------------------

def _registry_con(func, strict_args: bool = True):
    from crotolamo.tools.base import Registry, _build_parameters

    registry = Registry()
    registry.register(Tool(
        name=func.__name__, func=func, description="d",
        parameters=_build_parameters(func, {}), strict_args=strict_args,
    ))
    return registry


def test_registry_un_typeerror_interno_no_es_culpa_de_los_argumentos(caplog):
    """Con argumentos correctos, un bug de tipos DENTRO de la tool ('a' + None)
    le llegaba al modelo como "Argumentos inválidos": reintentaba con otros
    argumentos (gastando iteraciones) en vez de informar del fallo."""
    def rota(a: str) -> str:
        return a + None  # type: ignore[operator]

    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        out = _registry_con(rota).run("rota", {"a": "x"})
    assert out.startswith("La tool 'rota' reventó, patrón:")
    assert "can only concatenate" in out
    # Y queda la traza del bug en el log (antes no se logueaba nada).
    assert "Traceback" in caplog.text and "in rota" in caplog.text


def test_registry_deja_traza_cuando_una_tool_revienta(caplog):
    def boom() -> str:
        raise RuntimeError("kaboom")

    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        out = _registry_con(boom).run("boom", {})
    assert out == "La tool 'boom' reventó, patrón: kaboom"
    assert "RuntimeError: kaboom" in caplog.text


def test_registry_argumentos_que_no_encajan_en_la_firma_son_invalidos(caplog):
    """Falta uno requerido o sobra uno: eso SÍ es culpa de los argumentos, se
    detecta sin llegar a llamar a la tool y no ensucia el log con una traza."""
    llamadas = []

    def saluda(nombre: str, veces: int = 1) -> str:
        llamadas.append(nombre)
        return "hola " * veces + nombre

    registry = _registry_con(saluda)
    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        falta = registry.run("saluda", {})
        sobra = registry.run("saluda", {"nombre": "ana", "zzz": 1})
    assert falta.startswith("Argumentos inválidos para 'saluda', patrón:")
    assert "nombre" in falta
    assert sobra.startswith("Argumentos inválidos para 'saluda', patrón:")
    assert "zzz" in sobra
    assert llamadas == [] and caplog.text == ""
    assert registry.run("saluda", {"nombre": "ana", "veces": 2}) == "hola hola ana"


@pytest.mark.parametrize("malo", ["5", None, 2.0])
def test_registry_un_argumento_de_tipo_equivocado_sigue_siendo_invalido(malo, caplog):
    """El caso frecuente (limit="5" del modelo local) conserva su etiqueta: es
    la que le dice al modelo que corrija SUS argumentos. No es un bug de la
    tool, así que tampoco deja traza."""
    def repite(limit: int = 8) -> str:
        return "x" * limit if limit > 0 else ""

    registry = _registry_con(repite)
    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        out = registry.run("repite", {"limit": malo})
    assert out.startswith("Argumentos inválidos para 'repite', patrón:")
    assert caplog.text == ""
    assert registry.run("repite", {"limit": 2}) == "xx"


def test_registry_tool_de_esquema_abierto_que_revienta_por_dentro():
    """Tools MCP (strict_args=False, **kwargs, esquema arbitrario del server):
    cualquier argumento encaja en la firma; un TypeError suyo es interno."""
    def remota(**arguments) -> str:
        return len(arguments) + "x"  # type: ignore[operator]

    registry = _registry_con(remota, strict_args=False)
    registry.get("remota").parameters = {
        "type": "object",
        "properties": {"opciones": {"type": ["object", "null"]}, "n": {"type": "integer"}},
    }
    out = registry.run("remota", {"opciones": {"a": 1}, "n": 3, "extra": "vale"})
    assert out.startswith("La tool 'remota' reventó, patrón:")


def test_registry_valida_contra_la_firma_que_de_verdad_se_llama():
    """Un envoltorio (functools.wraps) que inyecta un argumento: la firma que
    cuenta es la del envoltorio, no la de la función envuelta; y un callable
    sin firma inspeccionable se llama sin más."""
    import functools

    def interna(conexion: str, clave: str) -> str:
        return f"{conexion}:{clave}"

    @functools.wraps(interna)
    def envuelta(clave: str) -> str:
        return interna("db", clave)

    assert _registry_con(envuelta).run("interna", {"clave": "k"}) == "db:k"

    from crotolamo.tools.base import Registry

    registry = Registry()
    registry.register(Tool(name="mayus", func=str.upper, description="d", parameters={}))
    assert registry.run("mayus", {}).startswith("Argumentos inválidos para 'mayus'")
    registry.register(Tool(name="entero", func=int, description="d", parameters={}))
    assert registry.run("entero", {}) == "0"


# ---------------------------------------------------------------------------
# Settings: las raíces del corral no pueden depender del CWD del proceso
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clave", ["allowed_roots", "confirm_roots"])
@pytest.mark.parametrize("raiz", ["", ".", "Documentos", "$CROTOLAMO_NO_DEFINIDA/x"])
def test_una_raiz_relativa_del_corral_no_arranca(tmp_path, monkeypatch, clave, raiz):
    """allowed_roots = [""] (o "Documentos", sin el ~/) se resolvía contra el
    CWD: el corral cambiaba según desde dónde se lanzara, y con "" y el CWD en
    / TODO el disco quedaba en zona libre, sin confirmación."""
    monkeypatch.delenv("CROTOLAMO_NO_DEFINIDA", raising=False)
    listas = {"allowed_roots": '["~/Documentos"]', "confirm_roots": '["~"]'}
    listas[clave] = f'["~/Descargas", "{raiz}"]'
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[paths]\n" + "".join(f"{k} = {v}\n" for k, v in listas.items()), encoding="utf-8",
    )
    with pytest.raises(ValueError, match=clave):
        settings_mod.load_settings(cfg)


def test_las_raices_absolutas_con_tilde_o_con_variable_siguen_cargando(tmp_path, monkeypatch):
    monkeypatch.setenv("CROTOLAMO_RAIZ_TEST", str(tmp_path))
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[paths]\nallowed_roots = ["~/Documentos", "$CROTOLAMO_RAIZ_TEST/libre", "/srv/datos"]\n',
        encoding="utf-8",
    )
    s = settings_mod.load_settings(cfg)
    assert s.allowed_roots == [
        Path.home() / "Documentos", tmp_path / "libre", Path("/srv/datos"),
    ]
    assert s.confirm_roots == [Path.home()]


# ---------------------------------------------------------------------------
# Tools de archivo: leer sin cargar el archivo entero, escribir sin destruir,
# nombres que no son UTF-8 y mover DENTRO de una carpeta
# ---------------------------------------------------------------------------

@pytest.fixture
def corral(tmp_path, monkeypatch):
    """El corral de las tools de archivo es tmp_path (sin zona de confirmación)."""
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [tmp_path])
    monkeypatch.setattr(real, "confirm_roots", [])
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    # Resuelto: las tools hablan de la ruta canónica.
    return tmp_path.resolve()


def _leer_con_read_file(carpeta: Path, nombre: str) -> str:
    from crotolamo.tools import files

    return files.read_file(str(carpeta / nombre))


def _leer_con_read_project_file(carpeta: Path, nombre: str) -> str:
    from crotolamo.tools import projects

    real = settings_mod.get_settings()
    antes = real.projects
    real.projects = {"demo": carpeta}
    try:
        out = projects.read_project_file("demo", nombre)
    finally:
        real.projects = antes
    prefijo = f"Contenido de demo/{nombre}:\n"
    return out[len(prefijo):] if out.startswith(prefijo) else out


_LECTORES = pytest.mark.parametrize("leer", [_leer_con_read_file, _leer_con_read_project_file])


@_LECTORES
def test_leer_un_archivo_enorme_solo_lee_lo_que_devuelve(corral, leer):
    """read_text() cargaba y decodificaba el archivo ENTERO (un log de cientos
    de MB, un .iso de descargas) para quedarse con los primeros 20 000
    caracteres: segundos de CPU y el doble de su tamaño en RAM."""
    import tracemalloc

    from crotolamo.tools.base import READ_CAP

    linea = "2026-11-13 12:00:00 INFO crotolamo.voice: chunk procesado " + "x" * 40 + "\n"
    with (corral / "grande.log").open("w", encoding="utf-8") as fh:
        for _ in range(60):
            fh.write(linea * 4096)
    size = (corral / "grande.log").stat().st_size
    assert size > 20_000_000

    tracemalloc.start()
    try:
        out = leer(corral, "grande.log")
        _actual, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Antes el pico rondaba 2x el tamaño del archivo (bytes + str decodificado).
    assert pico < 2_000_000
    cabeza = (linea * (READ_CAP // len(linea) + 1))[:READ_CAP]
    # El total de la marca va en bytes: contarlo en caracteres exigiría leerlo todo.
    assert out == cabeza + f"\n...[recortado, {size} bytes en total]"


@_LECTORES
@pytest.mark.parametrize("crudo", [
    b"hola patr\xc3\xb3n\r\nsegunda l\xc3\xadnea\r\n",
    b"latin-1 suelto: canci\xf3n \xff\xfe fin\n",
    b"multibyte cortado al final \xc3",
    b"retorno de carro al final\r",
    b"",
])
def test_leer_un_archivo_corto_devuelve_lo_mismo_que_read_text(corral, leer, crudo):
    (corral / "a.txt").write_bytes(crudo)
    esperado = (corral / "a.txt").read_text(encoding="utf-8", errors="replace")
    assert leer(corral, "a.txt") == esperado


@_LECTORES
def test_leer_recorta_por_caracteres_y_no_parte_un_multibyte(corral, leer):
    from crotolamo.tools.base import READ_CAP

    (corral / "justo.txt").write_text("ñ" * READ_CAP, encoding="utf-8")
    assert leer(corral, "justo.txt") == "ñ" * READ_CAP  # cabe entero: sin marca
    (corral / "pasado.txt").write_text("ñ" * (READ_CAP + 1), encoding="utf-8")
    assert leer(corral, "pasado.txt") == (
        "ñ" * READ_CAP + f"\n...[recortado, {2 * (READ_CAP + 1)} bytes en total]"
    )


@_LECTORES
def test_leer_un_fifo_no_cuelga_el_hilo(corral, leer):
    """open() de un FIFO sin escritor bloquea para siempre: el hilo del cerebro
    se quedaba colgado, sin timeout, hasta reiniciar. Solo se leen archivos
    normales."""
    import os
    import threading

    fifo = corral / "tuberia"
    os.mkfifo(fifo)
    salida: list[str] = []
    hilo = threading.Thread(target=lambda: salida.append(leer(corral, "tuberia")), daemon=True)
    hilo.start()
    hilo.join(timeout=3)
    colgado = hilo.is_alive()
    if colgado:
        # Abrir y cerrar la otra punta desbloquea el open() del hilo (lee EOF),
        # para no dejarlo vivo después del test.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        hilo.join(timeout=3)
    assert not colgado
    assert "no es un archivo normal" in salida[0] and "patrón" in salida[0]


# Sustituto suelto: medio emoji, tal como lo deja json.loads('"\\ud83d"') cuando
# el JSON de un tool-call llega cortado.
_MEDIO_EMOJI = "\ud83d"


def test_write_file_no_vacia_el_archivo_si_el_contenido_no_se_puede_codificar(corral):
    """write_text abre con 'w' (trunca) y DESPUÉS codifica: un sustituto suelto
    en el contenido lanzaba UnicodeEncodeError con el archivo ya a 0 bytes."""
    from crotolamo.tools import default_registry

    destino = corral / "importante.txt"
    destino.write_text("contenido valioso\n" * 100, encoding="utf-8")
    out = default_registry().run(
        "write_file", {"path": str(destino), "content": f"hola {_MEDIO_EMOJI} mundo"},
    )
    assert out.startswith("Escribí 12 chars en ")
    assert destino.read_text(encoding="utf-8") == "hola \ufffd mundo"


def test_create_note_con_un_sustituto_suelto_no_revienta(corral, monkeypatch):
    """Mismo origen: la nota (o su título, que va al cuerpo por defecto) con un
    sustituto suelto reventaba tras truncar la nota que ya existiera."""
    from crotolamo.tools import default_registry

    monkeypatch.setitem(settings_mod.get_settings().paths, "notas", corral / "notas")
    out = default_registry().run("create_note", {"title": f"idea {_MEDIO_EMOJI}"})
    assert out.startswith("Nota creada, patrón:")
    (nota,) = (corral / "notas").iterdir()
    assert nota.read_text(encoding="utf-8") == "# idea \ufffd\n\n"


def _archivo_con_nombre_no_utf8(carpeta: Path, contenido: bytes = b"") -> None:
    """'Declaración.pdf' con la ó en CP850 (0xA2): así deja unzip los nombres de
    un ZIP hecho en Windows."""
    import os

    try:
        with open(os.path.join(os.fsencode(carpeta), b"Declaraci\xa2n.pdf"), "wb") as fh:
            fh.write(contenido)
    except OSError:
        pytest.skip("este sistema de archivos no admite nombres que no sean UTF-8")


def test_run_cmd_no_revienta_con_una_salida_que_no_es_utf8():
    """text=True decodifica en estricto: un solo byte raro en la salida de un
    comando lanzaba UnicodeDecodeError, que no es ni OSError ni TimeoutExpired
    (lo único que run_cmd promete lanzar y lo único que atrapan sus callers)."""
    import sys

    from crotolamo.tools.base import run_cmd

    result = run_cmd([sys.executable, "-c", r"import os; os.write(1, b'caf\xe9 listo\n')"])
    assert result.stdout == "caf\ufffd listo\n"


def test_search_files_lista_un_archivo_con_nombre_no_utf8(corral, monkeypatch):
    from crotolamo.tools import default_registry

    _archivo_con_nombre_no_utf8(corral)
    monkeypatch.setitem(settings_mod.get_settings().paths, "documentos", corral)
    out = default_registry().run("search_files", {"query": "declara"})
    assert out.startswith("Encontré esto, patrón:")
    assert out.endswith("Declaraci\ufffdn.pdf")


def test_find_in_project_no_revienta_con_un_nombre_no_utf8(corral, monkeypatch):
    from crotolamo.tools import default_registry

    _archivo_con_nombre_no_utf8(corral, b"la aguja esta aqui\n")
    monkeypatch.setattr(settings_mod.get_settings(), "projects", {"demo": corral})
    out = default_registry().run("find_in_project", {"project": "demo", "pattern": "aguja"})
    assert out.startswith("Encontré «aguja» en demo, patrón:")
    assert "Declaraci\ufffdn.pdf:1:la aguja esta aqui" in out


@pytest.mark.parametrize("tool_name, argumentos", [
    ("list_dir", {"path": "{carpeta}"}),
    ("list_project_tree", {"name": "demo"}),
    ("read_file", {"path": "{carpeta}/no_existe_\udca2.txt"}),
])
def test_el_resultado_de_una_tool_nunca_lleva_sustitutos_sueltos(
    corral, monkeypatch, tool_name, argumentos,
):
    """Python nombra con sustitutos lo que no pudo decodificar ('Declaraci\\udca2n').
    Ese str salía tal cual de la tool hacia el historial, y revienta con
    UnicodeEncodeError en cuanto alguien lo codifica (imprimirlo, mandarlo)."""
    from crotolamo.tools import default_registry

    _archivo_con_nombre_no_utf8(corral)
    monkeypatch.setattr(settings_mod.get_settings(), "projects", {"demo": corral})
    argumentos = {k: v.format(carpeta=corral) for k, v in argumentos.items()}
    out = default_registry().run(tool_name, argumentos)
    out.encode("utf-8")  # antes: "surrogates not allowed"
    assert "\ufffd" in out and "reventó" not in out


def test_registry_tampoco_deja_sustitutos_en_el_mensaje_de_una_tool_que_revienta():
    def rota() -> str:
        raise RuntimeError(f"no pude con canci{_MEDIO_EMOJI}n.mp3")

    out = _registry_con(rota).run("rota", {})
    assert out == "La tool 'rota' reventó, patrón: no pude con canci\ufffdn.mp3"


def test_registry_no_toca_un_resultado_bien_formado():
    """Solo cambia lo irrepresentable: acentos, emoji y saltos salen intactos."""
    texto = "Contenido, patrón:\n📁 música\n📄 año_ñandú.txt\t¿qué?"

    def lista() -> str:
        return texto

    assert _registry_con(lista).run("lista", {}) == texto


def test_move_file_a_una_carpeta_existente_lo_mete_dentro(corral):
    """"Mueve informe.pdf a la carpeta respaldo": con dest=<carpeta> el rename
    fallaba con "[Errno 21] Is a directory" (y el patrón ya había confirmado)."""
    from crotolamo.tools import files

    (corral / "informe.pdf").write_text("pdf")
    (corral / "respaldo").mkdir()
    out = files.move_file(str(corral / "informe.pdf"), str(corral / "respaldo"))
    assert out == f"Moví {corral / 'informe.pdf'} -> {corral / 'respaldo' / 'informe.pdf'}, patrón."
    assert (corral / "respaldo" / "informe.pdf").read_text() == "pdf"
    assert not (corral / "informe.pdf").exists()


def test_move_file_carpeta_a_carpeta_vacia_no_la_sustituye(corral):
    """rename(2) REEMPLAZA un directorio destino vacío: "mueve fotos a respaldo"
    dejaba 'fotos' llamándose 'respaldo' en vez de respaldo/fotos."""
    from crotolamo.tools import files

    (corral / "fotos").mkdir()
    (corral / "fotos" / "a.jpg").write_text("x")
    (corral / "respaldo").mkdir()
    out = files.move_file(str(corral / "fotos"), str(corral / "respaldo"))
    assert out.startswith("Moví ")
    assert (corral / "respaldo" / "fotos" / "a.jpg").read_text() == "x"
    assert not (corral / "fotos").exists()


def test_move_file_sigue_renombrando_y_pisando_un_archivo_destino(corral):
    from crotolamo.tools import files

    (corral / "a.txt").write_text("nuevo")
    (corral / "b.txt").write_text("viejo")
    files.move_file(str(corral / "a.txt"), str(corral / "b.txt"))
    assert (corral / "b.txt").read_text() == "nuevo" and not (corral / "a.txt").exists()
    # Carpeta a un nombre que no existe: renombrar, como siempre.
    (corral / "fotos").mkdir()
    files.move_file(str(corral / "fotos"), str(corral / "imagenes"))
    assert (corral / "imagenes").is_dir() and not (corral / "fotos").exists()
    # Origen y destino iguales: no se "mete dentro de sí misma".
    out = files.move_file(str(corral / "imagenes"), str(corral / "imagenes"))
    assert out.startswith("Moví ") and (corral / "imagenes").is_dir()


def test_move_file_revalida_el_corral_sobre_la_ruta_final(tmp_path, monkeypatch):
    """La ruta final (carpeta destino + nombre) pasa por el mismo cerrojo que
    las demás: si esa entrada es un enlace que sale del corral, no se toca."""
    from crotolamo.tools import files

    dentro = tmp_path / "dentro"
    fuera = tmp_path / "fuera"
    (dentro / "destino").mkdir(parents=True)
    fuera.mkdir()
    (fuera / "nota.txt").write_text("ajeno")
    (dentro / "nota.txt").write_text("mio")
    (dentro / "destino" / "nota.txt").symlink_to(fuera / "nota.txt")
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [dentro])
    monkeypatch.setattr(real, "confirm_roots", [])

    out = files.move_file(str(dentro / "nota.txt"), str(dentro / "destino"))
    assert "permitidas" in out
    assert (dentro / "nota.txt").read_text() == "mio"
    assert (fuera / "nota.txt").read_text() == "ajeno"
    assert (dentro / "destino" / "nota.txt").is_symlink()


def test_write_file_con_contenido_que_no_es_texto_sigue_siendo_argumento_invalido(corral):
    """El arreglo de arriba no cambia la etiqueta de un contenido de otro tipo
    (content=123 del modelo local): sigue diciéndole que corrija SUS argumentos,
    y el archivo que iba a pisar sigue intacto."""
    from crotolamo.tools import default_registry

    destino = corral / "importante.txt"
    destino.write_text("contenido valioso", encoding="utf-8")
    out = default_registry().run("write_file", {"path": str(destino), "content": 123})
    assert out.startswith("Argumentos inválidos para 'write_file', patrón:")
    assert destino.read_text(encoding="utf-8") == "contenido valioso"


# ---------------------------------------------------------------------------
# search_files: no bajar a .venv/.git/node_modules y no llamar "tardó" a un fallo
# ---------------------------------------------------------------------------

@pytest.fixture
def documentos(corral, monkeypatch):
    """Un Documentos con un repo dentro (entorno virtual, .git, node_modules y
    cachés) y UN archivo del patrón que casa con "test"."""
    docs = corral / "docs"
    for carpeta in (".venv/lib", ".git/objects", "node_modules/pkg", "src/__pycache__", "venv"):
        (docs / "repo" / carpeta).mkdir(parents=True)
    for i in range(30):
        (docs / "repo" / ".venv" / "lib" / f"test_{i}.py").write_text("")
    (docs / "repo" / ".git" / "objects" / "test_pack").write_text("")
    (docs / "repo" / "node_modules" / "pkg" / "test.js").write_text("")
    (docs / "repo" / "node_modules" / "pkg" / "node_modules_leeme.txt").write_text("")
    (docs / "repo" / "src" / "__pycache__" / "test.cpython-312.pyc").write_text("")
    (docs / "repo" / "venv" / "test_activate").write_text("")
    (docs / "test_de_fisica.pdf").write_text("")
    monkeypatch.setitem(settings_mod.get_settings().paths, "documentos", docs)
    return docs


def test_search_files_no_baja_a_los_entornos_ni_a_git(documentos):
    """"Busca el archivo test de física": find recorría .venv/.git/node_modules
    de cada repo bajo Documentos y sus 34 coincidencias copaban las 20 rutas
    mostradas; que saliera el archivo del patrón dependía del orden del disco."""
    from crotolamo.tools import files

    out = files.search_files("test")
    assert out == f"Encontré esto, patrón:\n{documentos / 'test_de_fisica.pdf'}"


def test_search_files_sigue_listando_la_carpeta_podada_si_es_lo_que_se_busca(documentos):
    """La poda es sobre el CONTENIDO: "busca node_modules" sigue diciendo dónde
    está la carpeta (antes salía, y tiene que seguir saliendo)."""
    from crotolamo.tools import files

    out = files.search_files("node_modules")
    assert out == f"Encontré esto, patrón:\n{documentos / 'repo' / 'node_modules'}"


def test_search_files_un_solo_find_y_fuera_de_las_carpetas_podadas_todo_igual(
    documentos, monkeypatch,
):
    import subprocess

    from crotolamo.tools import files

    (documentos / "repo" / "src" / "Test_Unitario.py").write_text("")
    (documentos / "repo" / ".config").mkdir()
    (documentos / "repo" / ".config" / "test.toml").write_text("")
    lanzados: list[list[str]] = []
    real_run = subprocess.run

    def espia(cmd, **kwargs):
        lanzados.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", espia)
    out = files.search_files("test")
    assert len(lanzados) == 1 and lanzados[0][0] == "find"
    # Sin distinguir mayúsculas y entrando en las carpetas ocultas que no son
    # de herramientas, como siempre.
    assert sorted(out.splitlines()[1:]) == sorted(str(p) for p in (
        documentos / "test_de_fisica.pdf",
        documentos / "repo" / "src" / "Test_Unitario.py",
        documentos / "repo" / ".config" / "test.toml",
    ))


def test_search_files_la_poda_no_alcanza_a_la_propia_base(corral, monkeypatch):
    """La base es donde el patrón mandó buscar: aunque se llame como una de las
    carpetas podadas, se recorre (find evalúa la poda también sobre ella)."""
    from crotolamo.tools import files

    base = corral / "venv"
    (base / "sub" / "venv").mkdir(parents=True)
    (base / "sub" / "tesis.pdf").write_text("")
    monkeypatch.setitem(settings_mod.get_settings().paths, "documentos", base)
    assert files.search_files("tesis") == f"Encontré esto, patrón:\n{base / 'sub' / 'tesis.pdf'}"


def test_search_files_si_find_no_arranca_no_dice_que_tardo(documentos, monkeypatch):
    """OSError (find ausente o sin permiso de ejecución) compartía mensaje con
    el timeout: "tardó demasiado" para una búsqueda que ni llegó a empezar."""
    import subprocess

    from crotolamo.tools import files

    def sin_find(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "find")

    monkeypatch.setattr(subprocess, "run", sin_find)
    out = files.search_files("test")
    assert "tardó" not in out
    assert "No pude" in out and "patrón" in out and "No such file" in out


def test_search_files_el_timeout_sigue_diciendo_que_tardo(documentos, monkeypatch):
    import subprocess

    from crotolamo.tools import files

    def lento(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 20))

    monkeypatch.setattr(subprocess, "run", lento)
    assert files.search_files("test") == "La búsqueda tardó demasiado, patrón."


# ---------------------------------------------------------------------------
# create_note: crear nunca es pisar, y un título largo o una carpeta rota no
# "revientan"
# ---------------------------------------------------------------------------

@pytest.fixture
def notas(corral, monkeypatch):
    monkeypatch.setitem(settings_mod.get_settings().paths, "notas", corral / "notas")
    return corral / "notas"


def test_create_note_no_pisa_una_nota_con_el_mismo_titulo(notas):
    """"Crea una nota lista de compras: leche, huevos" y días después "crea una
    nota lista de compras: tortillas": write_text sobrescribía la primera sin
    avisar, y la tool es safe=True (nadie pregunta antes)."""
    from crotolamo.tools import files

    primera = files.create_note("Lista de compras", "leche, huevos")
    segunda = files.create_note("Lista de compras", "tortillas")
    assert primera == f"Nota creada, patrón: {notas / 'Lista_de_compras.md'}"
    assert (notas / "Lista_de_compras.md").read_text(encoding="utf-8") == "leche, huevos"
    # La segunda se guarda aparte y la respuesta dice dónde y por qué.
    assert segunda.startswith(f"Nota creada, patrón: {notas / 'Lista_de_compras_2.md'}")
    assert "no la pisé" in segunda
    assert (notas / "Lista_de_compras_2.md").read_text(encoding="utf-8") == "tortillas"
    # Y la tercera no pisa ninguna de las dos.
    files.create_note("Lista de compras", "pan")
    assert sorted(p.name for p in notas.iterdir()) == [
        "Lista_de_compras.md", "Lista_de_compras_2.md", "Lista_de_compras_3.md",
    ]


def test_create_note_titulos_distintos_que_se_sanean_igual_no_se_pisan(notas):
    from crotolamo.tools import files

    files.create_note("ideas: juego", "A")
    files.create_note("ideas/ juego", "B")
    assert (notas / "ideas__juego.md").read_text(encoding="utf-8") == "A"
    assert (notas / "ideas__juego_2.md").read_text(encoding="utf-8") == "B"


def test_create_note_no_escribe_a_traves_de_un_enlace_con_el_nombre_de_la_nota(notas, corral):
    """Otra cara de lo mismo: si el nombre era un enlace, write_text escribía
    en su destino (otro archivo del patrón)."""
    from crotolamo.tools import files

    notas.mkdir()
    (corral / "diario.txt").write_text("querido diario", encoding="utf-8")
    (notas / "Diario.md").symlink_to(corral / "diario.txt")
    out = files.create_note("Diario", "hoy llovió")
    assert out.startswith(f"Nota creada, patrón: {notas / 'Diario_2.md'}")
    assert (corral / "diario.txt").read_text(encoding="utf-8") == "querido diario"


def test_create_note_el_nombre_alternativo_tambien_pasa_por_el_corral(tmp_path, monkeypatch):
    """Carpeta de notas FUERA del corral con un enlace 'Diario.md' hacia dentro:
    el cerrojo mira la ruta resuelta (dentro) y deja pasar. El nombre libre
    que se elige después (Diario_2.md) cae en la carpeta de fuera."""
    from crotolamo.tools import files

    dentro, fuera = tmp_path / "dentro", tmp_path / "fuera"
    dentro.mkdir()
    fuera.mkdir()
    (dentro / "diario.txt").write_text("querido diario", encoding="utf-8")
    (fuera / "Diario.md").symlink_to(dentro / "diario.txt")
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [dentro])
    monkeypatch.setattr(real, "confirm_roots", [])
    monkeypatch.setitem(real.paths, "notas", fuera)

    out = files.create_note("Diario", "hoy llovió")
    assert "permitidas" in out
    assert [p.name for p in fuera.iterdir()] == ["Diario.md"]
    assert (dentro / "diario.txt").read_text(encoding="utf-8") == "querido diario"


@pytest.mark.parametrize("titulo", ["a" * 300, "ñ" * 200, "anota que " + "y luego " * 40])
def test_create_note_con_un_titulo_larguisimo_no_revienta(notas, titulo):
    """Al dictar, el modelo mete la frase entera en el título: con más de 252
    bytes de nombre el open() daba "[Errno 36] File name too long" y la tool
    "reventaba". El nombre se acorta; el título va completo en la nota."""
    from crotolamo.tools import default_registry

    out = default_registry().run("create_note", {"title": titulo})
    assert out.startswith("Nota creada, patrón:") and "reventó" not in out
    (nota,) = notas.iterdir()
    assert len(nota.name.encode("utf-8")) <= 255
    assert nota.read_text(encoding="utf-8") == f"# {titulo}\n\n"
    # Y un segundo dictado igual de largo tampoco pisa ni revienta.
    out = default_registry().run("create_note", {"title": titulo, "content": "otra"})
    assert out.startswith("Nota creada, patrón:") and len(list(notas.iterdir())) == 2


def test_create_note_si_no_puede_crear_la_carpeta_responde_en_personaje(corral, monkeypatch):
    """mkdir/escritura iban sin red: cualquier OSError (aquí, la "carpeta" de
    notas cuelga de un archivo) salía como "La tool 'create_note' reventó"."""
    from crotolamo.tools import default_registry

    (corral / "estorbo").write_text("soy un archivo")
    monkeypatch.setitem(
        settings_mod.get_settings().paths, "notas", corral / "estorbo" / "notas",
    )
    out = default_registry().run("create_note", {"title": "idea"})
    assert "reventó" not in out
    assert out.startswith("No pude crear la nota, patrón:")


# ---------------------------------------------------------------------------
# list_project_tree: filtrar ANTES de recortar, y decir cuándo se recortó
# ---------------------------------------------------------------------------

@pytest.fixture
def proyecto(corral, monkeypatch):
    proj = corral / "p"
    proj.mkdir()
    monkeypatch.setattr(settings_mod.get_settings(), "projects", {"p": proj})
    return proj


def test_list_project_tree_los_ocultos_no_gastan_el_cupo_de_la_carpeta(proyecto):
    """El corte a 12 iba ANTES de descartar ocultos: una carpeta con 12
    dotfiles y luego informe.md y ventas.csv salía VACÍA."""
    from crotolamo.tools import projects

    (proyecto / "datos").mkdir()
    for i in range(12):
        (proyecto / "datos" / f".oculto{i:02d}").write_text("")
    (proyecto / "datos" / "__pycache__").mkdir()
    (proyecto / "datos" / "informe.md").write_text("")
    (proyecto / "datos" / "ventas.csv").write_text("")
    assert projects.list_project_tree("p").splitlines() == [
        "p/", "  📁 datos", "    📄 informe.md", "    📄 ventas.csv",
    ]


def test_list_project_tree_avisa_cuando_recorta_una_carpeta(proyecto):
    """tests/ con 40 archivos mostraba 12 sin ninguna marca: el modelo
    respondía "tests/ tiene 12 archivos" o "no existe test_30.py"."""
    from crotolamo.tools import projects

    (proyecto / "tests").mkdir()
    for i in range(40):
        (proyecto / "tests" / f"test_{i:02d}.py").write_text("")
    (proyecto / "tests" / ".pytest_cache").mkdir()  # no cuenta: no se muestra
    lines = projects.list_project_tree("p").splitlines()
    assert lines[:2] == ["p/", "  📁 tests"]
    assert lines[2:14] == [f"    📄 test_{i:02d}.py" for i in range(12)]
    assert lines[14:] == ["    ...y 28 más."]


@pytest.mark.parametrize("cuantos, marca", [(12, []), (13, ["    ...y 1 más."])])
def test_list_project_tree_la_marca_solo_sale_si_sobra_algo(proyecto, cuantos, marca):
    from crotolamo.tools import projects

    (proyecto / "src").mkdir()
    for i in range(cuantos):
        (proyecto / "src" / f"m{i:02d}.py").write_text("")
    assert projects.list_project_tree("p").splitlines()[14:] == marca


# ---------------------------------------------------------------------------
# find_in_project: el texto con corchetes o asteriscos también se encuentra
# ---------------------------------------------------------------------------

@pytest.fixture
def espia_grep(monkeypatch):
    """Deja correr grep de verdad y apunta cada argv lanzado."""
    from crotolamo.tools import projects

    lanzados: list[list[str]] = []
    real_run = projects.subprocess.run

    def espia(cmd, **kwargs):
        lanzados.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(projects.subprocess, "run", espia)
    return lanzados


@pytest.mark.parametrize("texto, linea", [
    ("list[str]", "tipos.py:1:def f(xs: list[str], *args) -> None:"),
    ("arr[0]", "tipos.py:2:    return arr[0] + g(*args)"),
    ("arr[", "tipos.py:2:    return arr[0] + g(*args)"),   # regex inválido: grep sale con 2
    ("g(*args)", "tipos.py:2:    return arr[0] + g(*args)"),
    ("re.sub(r\"\\s+\"", "tipos.py:3:    re.sub(r\"\\s+\", \" \", x)"),
    ("LIST[STR]", "tipos.py:1:def f(xs: list[str], *args) -> None:"),  # sigue sin mayúsculas
])
def test_find_in_project_encuentra_codigo_literal_con_metacaracteres(
    proyecto, espia_grep, texto, linea,
):
    """El modelo pasa código tal cual. grep lo leía como regex: `list[str]` es
    "lis" + una de {s,t,r}, que NO casa con "list[str]"; y `arr[` ni siquiera
    es un regex (rc=2, stdout vacío). En los dos casos: "No encontré"."""
    from crotolamo.tools import projects

    (proyecto / "tipos.py").write_text(
        "def f(xs: list[str], *args) -> None:\n"
        "    return arr[0] + g(*args)\n"
        "    re.sub(r\"\\s+\", \" \", x)\n",
        encoding="utf-8",
    )
    out = projects.find_in_project("p", texto)
    assert out == f"Encontré «{texto}» en p, patrón:\n{linea}"
    # Primero como regex (lo de siempre) y, al no salir nada, tal cual.
    assert len(espia_grep) == 2
    assert "-F" not in espia_grep[0] and "-F" in espia_grep[1]
    for argv in espia_grep:
        assert argv[argv.index("-e") + 1] == texto and argv[-2:] == ["--", str(proyecto)]


def test_find_in_project_un_regex_que_encuentra_no_se_repite_como_literal(proyecto, espia_grep):
    """Lo que ya funcionaba no cambia ni cuesta más: si el regex da resultados,
    esos son los que salen, con un solo grep."""
    from crotolamo.tools import projects

    (proyecto / "agente.py").write_text("class ToolAgent:\n    pass\n", encoding="utf-8")
    out = projects.find_in_project("p", "^class Tool[A-Z]gent")
    assert out == "Encontré «^class Tool[A-Z]gent» en p, patrón:\nagente.py:1:class ToolAgent:"
    assert len(espia_grep) == 1


@pytest.mark.parametrize("texto, greps", [
    ("no_esta_en_ningun_lado", 1),   # sin metacaracteres: el literal daría lo mismo
    ("config.toml", 1),              # el punto como regex ya casa con el punto
    ("no_esta[0]", 2),
])
def test_find_in_project_sin_resultados_no_lanza_greps_de_mas(
    proyecto, espia_grep, texto, greps,
):
    from crotolamo.tools import projects

    (proyecto / "agente.py").write_text("class ToolAgent:\n    pass\n", encoding="utf-8")
    assert projects.find_in_project("p", texto) == f"No encontré «{texto}» en p, patrón."
    assert len(espia_grep) == greps


def test_find_in_project_un_error_de_grep_no_se_cuenta_como_no_encontrado(proyecto, monkeypatch):
    """rc>=2 es "grep falló" (aquí, un archivo que no pudo leer), no "no hay
    coincidencias" (rc=1): con stdout vacío se respondía "No encontré" a secas."""
    import subprocess

    from crotolamo.tools import projects

    def grep_con_error(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=2, stdout="",
            stderr=f"grep: {proyecto}/secreto.key: Permission denied\n",
        )

    monkeypatch.setattr(projects.subprocess, "run", grep_con_error)
    out = projects.find_in_project("p", "clave")
    assert out.startswith("No encontré «clave» en p, patrón")
    assert "secreto.key: Permission denied" in out


# ---------------------------------------------------------------------------
# find_in_project: las carpetas que poda todo lo demás, y líneas con tope
# ---------------------------------------------------------------------------

def test_find_in_project_no_baja_a_ninguna_carpeta_de_skip_dirs(proyecto, espia_grep):
    """Los --exclude-dir se repetían a mano y se quedaron atrás respecto de
    SKIP_DIRS: sin `venv` ni `node_modules`, las 25 líneas que se muestran se
    llenaban de dependencias y grep recorría miles de archivos ajenos."""
    from crotolamo.tools import projects
    from crotolamo.tools.base import SKIP_DIRS

    (proyecto / "src").mkdir()
    (proyecto / "src" / "app.js").write_text("const patron = 1;\n", encoding="utf-8")
    for carpeta in SKIP_DIRS:
        for i in range(6):
            (proyecto / carpeta / f"pkg{i}").mkdir(parents=True)
            (proyecto / carpeta / f"pkg{i}" / "index.js").write_text(
                "const patron = 2;\n", encoding="utf-8",
            )
    out = projects.find_in_project("p", "patron")
    assert out == "Encontré «patron» en p, patrón:\nsrc/app.js:1:const patron = 1;"
    # Una sola fuente: lo que grep excluye es lo que podan los recorridos.
    assert len(espia_grep) == 1
    excluidas = {a.split("=", 1)[1] for a in espia_grep[0] if a.startswith("--exclude-dir=")}
    assert excluidas == SKIP_DIRS


@pytest.mark.parametrize("nombre", [".git", ".venv", "__pycache__", "node_modules", "venv"])
def test_find_in_project_un_proyecto_llamado_como_una_carpeta_podada(
    corral, monkeypatch, espia_grep, nombre,
):
    """grep aplica --exclude-dir también a la carpeta que se le pasa: con el
    nombre de la propia raíz entre las excluidas no buscaba NADA y respondía
    "No encontré". (Lo mismo que search_files hace con su base.)"""
    from crotolamo.tools import projects
    from crotolamo.tools.base import SKIP_DIRS

    proj = corral / nombre
    proj.mkdir()
    (proj / "a.txt").write_text("hola patron\n", encoding="utf-8")
    monkeypatch.setattr(settings_mod.get_settings(), "projects", {"p": proj})
    out = projects.find_in_project("p", "patron")
    assert out == "Encontré «patron» en p, patrón:\na.txt:1:hola patron"
    excluidas = {a.split("=", 1)[1] for a in espia_grep[0] if a.startswith("--exclude-dir=")}
    assert excluidas == SKIP_DIRS - {nombre}


def test_find_in_project_recorta_una_linea_kilometrica(proyecto):
    """Se limitaban las líneas (25) pero no su largo: un JSON o un .min.js de
    una sola línea iba ENTERO al LLM (3,5 MB medidos; READ_CAP son 20 000)."""
    from crotolamo.tools import projects
    from crotolamo.tools.base import READ_CAP

    linea = '{"k": "' + "patron " * 150_000 + '"}'
    (proyecto / "data.json").write_text(linea + "\n", encoding="utf-8")
    (proyecto / "z.py").write_text("patron = 1\n", encoding="utf-8")
    out = projects.find_in_project("p", "patron")
    assert len(out) < READ_CAP
    lines = out.splitlines()
    assert lines[0] == "Encontré «patron» en p, patrón:"
    # Sigue diciendo DÓNDE está la coincidencia y cuánto se dejó fuera...
    mostrada = "data.json:1:" + linea
    assert lines[1] == mostrada[:projects._LINE_CAP]
    assert lines[2] == f"...[recortado, {len(mostrada)} chars en total]"
    # ...y la que cabía sale intacta.
    assert lines[3:] == ["z.py:1:patron = 1"]


def test_find_in_project_25_lineas_kilometricas_caben_en_el_tope(proyecto):
    """El peor caso (las 25 líneas mostradas, todas recortadas) no pasa de lo
    que ya devuelve read_project_file."""
    from crotolamo.tools import projects
    from crotolamo.tools.base import READ_CAP

    for i in range(30):
        (proyecto / f"m{i:02d}.min.js").write_text("patron;" * 5_000 + "\n", encoding="utf-8")
    out = projects.find_in_project("p", "patron")
    assert len(out) < READ_CAP
    assert out.count("...[recortado, ") == 25
    assert out.endswith("\n...y 5 coincidencias más.")


@pytest.mark.parametrize("sobra, recortada", [(0, False), (1, True)])
def test_find_in_project_una_linea_que_cabe_justa_no_se_toca(proyecto, sobra, recortada):
    from crotolamo.tools import projects

    contenido = "patron" + "x" * (projects._LINE_CAP - len("a.txt:1:patron") + sobra)
    (proyecto / "a.txt").write_text(contenido + "\n", encoding="utf-8")
    out = projects.find_in_project("p", "patron")
    assert ("[recortado" in out) is recortada
    if not recortada:
        assert out == f"Encontré «patron» en p, patrón:\na.txt:1:{contenido}"


# ---------------------------------------------------------------------------
# launch_project: si lo que lanza no arranca, lo dice en personaje
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("archivo, binario", [
    ("MiJuego.desktop", "gtk-launch"),   # gtk-launch sin instalar
    ("launch.sh", "kitti"),              # terminal de [apps] mal escrito
    (None, "kitti"),                     # sin launcher: abre un terminal ahí
], ids=["desktop", "script", "sin-launcher"])
def test_launch_project_si_no_arranca_no_revienta(proyecto, monkeypatch, archivo, binario):
    """run_detached lanza el OSError de Popen cuando el binario no existe: salía
    "La tool 'launch_project' reventó, patrón: [Errno 2] No such file...".
    No se ejecuta nada de verdad: run_detached falla como fallaría Popen."""
    from crotolamo.tools import default_registry, projects

    if archivo:
        (proyecto / archivo).write_text("", encoding="utf-8")
    lanzados: list[list[str]] = []

    def no_arranca(argv):
        lanzados.append(argv)
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(projects, "run_detached", no_arranca)
    monkeypatch.setattr(projects, "terminal_exec", lambda cmd: ["kitti", "bash", "-lc", cmd])
    out = default_registry().run("launch_project", {"name": "p"})
    assert [argv[0] for argv in lanzados] == [binario]
    assert "reventó" not in out and "Lancé" not in out and "Te abrí" not in out
    assert out == f"No pude lanzar p, patrón: [Errno 2] No such file or directory: '{binario}'"


def test_launch_project_cuando_arranca_responde_lo_de_siempre(proyecto, monkeypatch):
    from crotolamo.tools import projects

    (proyecto / "launch.sh").write_text("", encoding="utf-8")
    lanzados: list[list[str]] = []
    monkeypatch.setattr(projects, "run_detached", lanzados.append)
    monkeypatch.setattr(projects, "terminal_exec", lambda cmd: ["kitty", "bash", "-lc", cmd])
    out = projects.launch_project("p")
    assert out == "Lancé p con launch.sh, patrón. Que el código tenga piedad."
    assert lanzados == [
        ["kitty", "bash", "-lc", f"cd {proyecto} && {proyecto / 'launch.sh'}; exec bash"],
    ]


# ---------------------------------------------------------------------------
# read_page: una URL que ni se puede partir no es un "reventó"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://[::1",                 # corchete de IPv6 sin cerrar
    "https://ejemplo.com／ruta",    # barra de ancho completo: NFKC la vuelve "/"
], ids=["corchete", "nfkc"])
def test_read_page_con_una_url_mal_formada_responde_en_personaje(monkeypatch, url):
    """urlparse lanza ValueError y estaba fuera de todo try: salía "La tool
    'read_page' reventó, patrón: Invalid IPv6 URL". Sin red: no hay nada que
    resolver ni a qué conectarse."""
    from crotolamo.tools import _web, default_registry

    def no_debe_llamarse(*a, **kw):
        raise AssertionError("una URL mal formada no llega a la red")

    monkeypatch.setattr(_web.socket, "getaddrinfo", no_debe_llamarse)
    monkeypatch.setattr(_web.socket, "create_connection", no_debe_llamarse)
    out = default_registry().run("read_page", {"url": url})
    assert "reventó" not in out
    assert out.startswith("Esa URL está mal escrita, patrón")


def test_read_page_el_esquema_que_no_es_http_sigue_diciendo_lo_mismo():
    from crotolamo.tools import search

    assert search.read_page("ftp://ejemplo.com/a.txt") == (
        "Solo puedo leer páginas http o https, patrón, no 'ftp://'.")


# ---------------------------------------------------------------------------
# Filtro de búsquedas: los términos se normalizan una vez, no en cada búsqueda
# ---------------------------------------------------------------------------

def test_is_blocked_query_no_renormaliza_los_terminos_en_cada_llamada(monkeypatch):
    """Cada búsqueda pasaba los 12 términos fijos por normalize_key (regex +
    NFKD) además de la consulta: 13 normalizaciones para un resultado que solo
    depende de la consulta."""
    from crotolamo.tools import search

    llamadas: list[str] = []

    def espia(text: str) -> str:
        llamadas.append(text)
        return normalize_real(text)

    normalize_real = search.normalize_key
    monkeypatch.setattr(search, "normalize_key", espia)
    assert search.is_blocked_query("Descargar  RANSOMWARE") is True
    assert search.is_blocked_query("recetas de pan") is False
    assert llamadas == ["Descargar  RANSOMWARE", "recetas de pan"]


@pytest.mark.parametrize("consulta, bloqueada", [
    ("porno", True), ("Hackear Contraseña de wifi", True), ("robar contrasena", True),
    ("un exploit para romper eso", True), ("arma casera", True), ("xvideos", True),
    ("recetas de pan", False), ("letra de la bamba", False), ("como cambiar la bombilla", False),
    ("", False),
])
def test_is_blocked_query_decide_lo_mismo_que_antes(consulta, bloqueada):
    """El filtro compara sin acentos ni mayúsculas, y eso no cambia."""
    from crotolamo.tools import search

    assert search.is_blocked_query(consulta) is bloqueada


# ---------------------------------------------------------------------------
# Rutas relativas: ni el guard ni las tools de archivo dependen del CWD (T10)
# ---------------------------------------------------------------------------

@pytest.fixture
def casa(tmp_path, monkeypatch):
    """Home de mentira como el del patrón: Documentos es la zona libre, el resto
    del home pide confirmación y el repo de Crotolamo (el CWD del .service) vive
    en Documentos. Devuelve (home, repo, guard armado como en producción)."""
    home = (tmp_path / "home").resolve()
    repo = home / "Documentos" / "crotolamo2"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("print('hola')\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))  # para que "~" también sea este home
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "home", home)
    monkeypatch.setattr(real, "allowed_roots", [home / "Documentos"])
    monkeypatch.setattr(real, "confirm_roots", [home])
    monkeypatch.setattr(real, "projects", {"crotolamo": repo})
    # El ancla es opt-in ([paths].relative_base = "~"): estos tests prueban
    # el comportamiento CON ella; el de fábrica (sin ancla) va aparte.
    monkeypatch.setattr(real, "relative_base", home)
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    return home, repo, Guard.from_settings(real)


def _cwds(home: Path, repo: Path) -> list[Path]:
    """Desde dónde arranca Crotolamo: el .service (el repo), un shell abierto en
    el home y un lanzador cualquiera (la raíz)."""
    return [repo, home, Path("/")]


def _nativa(nombre: str) -> Tool:
    from crotolamo.tools import default_registry

    tool = default_registry().get(nombre)
    assert tool is not None
    return tool


def test_guard_no_lee_contra_el_cwd_el_relative_path_de_un_proyecto(casa, monkeypatch):
    """read_project_file(relative_path="./src/main.py") lo resuelve la tool
    contra la raíz del proyecto, pero el guard lo tomaba por una ruta del CWD:
    libre desde el repo, con confirmación desde el home y BLOQUEADO desde /."""
    from crotolamo.tools import projects

    home, repo, guard = casa
    tool = _nativa("read_project_file")
    for cwd in _cwds(home, repo):
        monkeypatch.chdir(cwd)
        for relativa in ("./src/main.py", "src/main.py", "../crotolamo2/src/main.py"):
            decision = guard.check(tool, {"project": "crotolamo", "relative_path": relativa})
            assert decision.allowed and not decision.needs_confirmation, (cwd, relativa)
            assert "print('hola')" in projects.read_project_file("crotolamo", relativa)
    # Lo que se sale del proyecto lo sigue parando la propia tool, que es quien
    # conoce la raíz; y una ruta absoluta la sigue parando el guard.
    assert "se sale del proyecto" in projects.read_project_file("crotolamo", "../../x.py")
    assert not guard.check(tool, {"project": "crotolamo", "relative_path": "/etc/passwd"}).allowed


def test_guard_ancla_al_home_las_rutas_relativas_de_las_tools_de_archivo(casa, monkeypatch):
    """read_file("Documentos/a.txt") pasaba libre desde el home y se bloqueaba
    desde /; make_dir("prueba") caía libre DENTRO del repo de Crotolamo cuando
    corría como servicio. La zona de una ruta relativa no depende del CWD."""
    home, repo, guard = casa
    vistas = set()
    for cwd in _cwds(home, repo):
        monkeypatch.chdir(cwd)
        libre = guard.check(_nativa("read_file"), {"path": "Documentos/a.txt"})
        pregunta = guard.check(_nativa("make_dir"), {"path": "prueba"})
        fuera = guard.check(_nativa("read_file"), {"path": "../otro/x.txt"})
        vistas.add((
            libre.allowed, libre.needs_confirmation,
            pregunta.allowed, pregunta.needs_confirmation, pregunta.reason,
            fuera.allowed,
        ))
    # Misma decisión desde los tres sitios, y la pregunta nombra la ruta REAL
    # (con solo "prueba" el patrón no sabría dónde va a caer la carpeta).
    assert vistas == {(
        True, False,
        True, True, f"fuera de la zona libre: {home / 'prueba'}. ¿Lo hago?",
        False,
    )}


def test_las_tools_de_archivo_anclan_las_rutas_relativas_al_home(casa, monkeypatch):
    """La tool tiene que abrir la MISMA ruta que clasificó el guard: relativa al
    home, se lance desde donde se lance."""
    from crotolamo.tools import files

    home, repo, _guard = casa
    (home / "Documentos" / "a.txt").write_text("hola patrón", encoding="utf-8")
    for n, cwd in enumerate(_cwds(home, repo)):
        monkeypatch.chdir(cwd)
        assert files.read_file("Documentos/a.txt") == "hola patrón"
        assert files._resolve("Documentos/a.txt") == home / "Documentos" / "a.txt"
        assert f"{home / 'Documentos' / f'fotos{n}'}" in files.make_dir(f"Documentos/fotos{n}")
        assert (home / "Documentos" / f"fotos{n}").is_dir()
        assert "Escribí" in files.write_file(f"Documentos/nota{n}.txt", "x")
        assert (home / "Documentos" / f"nota{n}.txt").read_text(encoding="utf-8") == "x"
    # Nada cayó en el repo ni en la raíz, y lo absoluto o con ~ no cambia.
    assert sorted(p.name for p in repo.iterdir()) == ["src"]
    assert files._resolve(str(repo / "src")) == repo / "src"
    assert files._resolve("~/x.txt") == home / "x.txt"


def test_una_ruta_vacia_es_el_home_y_el_guard_tambien_la_clasifica(casa, monkeypatch):
    """list_dir(path="") es "la carpeta base". El guard se saltaba el string
    vacío: lanzado desde el home, listaba la zona de confirmación sin preguntar."""
    from crotolamo.tools import files

    home, repo, guard = casa
    for cwd in _cwds(home, repo):
        monkeypatch.chdir(cwd)
        decision = guard.check(_nativa("list_dir"), {"path": ""})
        assert decision.allowed and decision.needs_confirmation, cwd
        assert str(home) in decision.reason
        assert files.list_dir("").startswith(f"Contenido de {home}, patrón:")


def test_guard_las_relativas_de_una_tool_mcp_siguen_siendo_del_cwd(casa, monkeypatch):
    """Un server MCP resuelve una ruta relativa contra SU directorio de trabajo,
    que hereda de este proceso, y para él el guard es el único corral: ni se
    anclan al home ni se dejan de mirar bajo un nombre cualquiera."""
    home, repo, guard = casa
    remota = Tool(name="fs__leer", func=lambda **k: "ok", description="d",
                  parameters={}, strict_args=False)
    monkeypatch.chdir(repo)
    dentro = guard.check(remota, {"path": "src/main.py"})
    assert dentro.allowed and not dentro.needs_confirmation
    assert not guard.check(remota, {"relative_path": "../../../x"}).allowed
    monkeypatch.chdir(home)
    pregunta = guard.check(remota, {"relative_path": "./src/main.py"})
    assert pregunta.allowed and pregunta.needs_confirmation
    assert pregunta.reason == "fuera de la zona libre: ./src/main.py. ¿Lo hago?"
    monkeypatch.chdir("/")
    for argumentos in ({"path": "Documentos/a.txt"}, {"path": ""}, {"target": "./x"},
                       {"relative_path": "../etc/passwd"}):
        assert not guard.check(remota, argumentos).allowed, argumentos


def test_guard_nativa_lo_anidado_y_lo_absoluto_se_siguen_clasificando(casa, monkeypatch):
    """Lo que deja de mirarse es SOLO el valor relativo de un argumento de primer
    nivel sin nombre de ruta (el que la tool nativa no abre contra el CWD)."""
    home, repo, guard = casa
    tool = _tool()
    monkeypatch.chdir("/")
    assert not guard.check(tool, {"relative_path": "~/../../etc/passwd"}).allowed
    assert not guard.check(tool, {"url": "file:///etc/passwd"}).allowed
    assert not guard.check(tool, {"nombre": " /etc/passwd"}).allowed
    assert not guard.check(tool, {"items": [{"nombre": "../../etc"}]}).allowed
    en_home = guard.check(tool, {"nombre": "~/x.txt"})
    assert en_home.allowed and en_home.needs_confirmation
    # Un "~usuario" que no existe no se puede leer: no pasa.
    assert not guard.check(tool, {"path": "~no_existe_este_usuario/x"}).allowed


def test_guard_sin_ancla_resuelve_contra_el_cwd_como_siempre(zonas, monkeypatch):
    """Guard(...) a pelo (sin el home de la config) no tiene a qué anclar: una
    ruta relativa bajo un nombre de ruta es del CWD, como la resuelve el sistema."""
    guard, libre, confirmable = zonas
    assert guard.relative_base is None
    monkeypatch.chdir(libre)
    dentro = guard.check(_tool(), {"path": "a.txt"})
    assert dentro.allowed and not dentro.needs_confirmation
    monkeypatch.chdir(confirmable)
    pregunta = guard.check(_tool(), {"path": "a.txt"})
    assert pregunta.allowed and pregunta.needs_confirmation
    assert pregunta.reason == "fuera de la zona libre: a.txt. ¿Lo hago?"


def test_de_punta_a_punta_el_guard_pregunta_por_la_ruta_que_la_tool_toca(casa, monkeypatch):
    """Con el agente de verdad: lo que el guard clasifica y nombra es lo que la
    tool abre. Antes, como servicio (CWD = el repo, zona libre), make_dir("fotos")
    corría sin preguntar y dejaba la carpeta dentro del repo de Crotolamo."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.tools import default_registry

    home, repo, guard = casa
    preguntas: list[str] = []

    def confirma(reason: str) -> bool:
        preguntas.append(reason)
        return True

    agent = ToolAgent(llm=None, conversation=Conversation("SYS"),
                      registry=default_registry(), guard=guard, confirm_fn=confirma)
    monkeypatch.chdir(repo)
    assert agent._execute_call("make_dir", {"path": "fotos"}) == (
        f"Carpeta lista, patrón: {home / 'fotos'}")
    assert preguntas == [f"fuera de la zona libre: {home / 'fotos'}. ¿Lo hago?"]
    assert (home / "fotos").is_dir() and not (repo / "fotos").exists()
    # Dentro de la zona libre no se pregunta, y tampoco depende del CWD.
    monkeypatch.chdir("/")
    out = agent._execute_call("list_dir", {"path": "Documentos"})
    assert out.startswith(f"Contenido de {home / 'Documentos'}, patrón:")
    assert len(preguntas) == 1


# ---------------------------------------------------------------------------
# _split_doc: la descripción no arrastra la sangría del docstring
# ---------------------------------------------------------------------------

def test_split_doc_no_arrastra_la_sangria_a_la_descripcion():
    """Las líneas del resumen se unían con su sangría del docstring: el LLM
    recibía "contenido     existente" (cinco espacios) en el esquema de CADA
    petición. Los parámetros ya se limpiaban; el resumen no."""
    from crotolamo.tools.base import _split_doc

    doc = '''Escribe (crea o sobrescribe) un archivo de texto. Puede pisar contenido
    existente, así que pide confirmación.

        Otra línea, aún más sangrada.

    Args:
        path: ruta del archivo a escribir.
    '''
    description, params = _split_doc(doc)
    assert description == (
        "Escribe (crea o sobrescribe) un archivo de texto. Puede pisar contenido "
        "existente, así que pide confirmación. Otra línea, aún más sangrada."
    )
    assert params == {"path": "ruta del archivo a escribir."}


def test_ninguna_tool_nativa_manda_sangria_en_su_esquema():
    from crotolamo.tools import default_registry

    registry = default_registry()
    vistas = 0
    for name in registry.names():
        tool = registry.get(name)
        if not tool.strict_args:
            continue  # las MCP traen la descripción que les da su server
        vistas += 1
        textos = [tool.description] + [
            spec.get("description", "") for spec in tool.parameters["properties"].values()
        ]
        assert not [t for t in textos if "  " in t or "\n" in t], name
    assert vistas >= 30
    escribir = registry.get("write_file").description
    assert "contenido existente, así que pide confirmación." in escribir


# ---------------------------------------------------------------------------
# Un parámetro opcional en null no revienta la tool (núcleo seguro de T16)
# ---------------------------------------------------------------------------

def test_create_note_con_el_cuerpo_en_null_crea_la_nota(notas):
    """Los modelos OpenAI-compatibles mandan `"content": null` en vez de omitir
    el opcional: salía "La tool 'create_note' reventó, patrón: 'NoneType' object
    has no attribute 'strip'" y la nota no se creaba."""
    from crotolamo.tools import default_registry

    out = default_registry().run("create_note", {"title": "lista del súper", "content": None})
    assert out == f"Nota creada, patrón: {notas / 'lista_del_súper.md'}"
    # Igual que sin cuerpo: la nota lleva su título.
    assert (notas / "lista_del_súper.md").read_text(encoding="utf-8") == "# lista del súper\n\n"


def test_search_web_con_el_motor_en_null_busca_en_el_de_por_defecto(monkeypatch):
    """`"engine": null` reventaba con "'NoneType' object has no attribute
    'lower'"; es lo mismo que no decir motor (google)."""
    from crotolamo.tools import default_registry, search

    abiertas: list[str] = []
    monkeypatch.setattr(search, "open_url", lambda url: abiertas.append(url) or "abierta")
    registry = default_registry()
    assert registry.run("search_web", {"query": "gatos siameses", "engine": None}) == "abierta"
    assert registry.run("search_web", {"query": "gatos siameses"}) == "abierta"
    assert abiertas == ["https://www.google.com/search?q=gatos+siameses"] * 2
    # El motor pedido se sigue respetando.
    assert search.build_search_url("youtube", "x") == (
        "https://www.youtube.com/results?search_query=x")


# ---------------------------------------------------------------------------
# Registry.run: true/false donde el esquema pide un número no es un número, y
# un argumento del tipo equivocado con el que la tool se atraganta (None.strip())
# es culpa de los argumentos, no un "reventó"
# ---------------------------------------------------------------------------

def test_forget_fact_con_true_no_borra_el_hecho_1(tmp_path, monkeypatch):
    """bool es subclase de int: forget_fact(fact_id=True) llegaba a SQLite como
    1 y BORRABA el hecho #1 ("Olvidado el hecho #True, patrón."), uno que nadie
    había pedido olvidar."""
    from crotolamo.persistence import facts
    from crotolamo.tools import default_registry

    monkeypatch.setattr(facts.db, "_default_db_path", lambda: tmp_path / "f.sqlite")
    primero = facts.remember("uso Fedora")
    facts.remember("me gusta el café de olla")
    registry = default_registry()
    for malo in (True, False):
        out = registry.run("forget_fact", {"fact_id": malo})
        assert out.startswith("Argumentos inválidos para 'forget_fact', patrón:"), out
    assert [r["texto"] for r in facts.recall()] == ["uso Fedora", "me gusta el café de olla"]
    # Un id de verdad se sigue borrando.
    assert registry.run("forget_fact", {"fact_id": primero}) == (
        f"Olvidado el hecho #{primero}, patrón.")
    assert [r["texto"] for r in facts.recall()] == ["me gusta el café de olla"]


def test_registry_un_booleano_donde_se_pide_un_numero_no_llega_a_la_tool(caplog):
    llamadas: list = []

    def borra(fact_id: int) -> str:
        llamadas.append(fact_id)
        return f"borré {fact_id}"

    def revisa(horas: float = 6) -> str:
        llamadas.append(horas)
        return f"{horas} h"

    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        for func, clave in ((borra, "fact_id"), (revisa, "horas")):
            for malo in (True, False):
                out = _registry_con(func).run(func.__name__, {clave: malo})
                assert out.startswith(f"Argumentos inválidos para '{func.__name__}', patrón:")
                assert clave in out
    assert llamadas == [] and caplog.text == ""
    # Los números de verdad pasan (un entero también vale donde se pide number).
    assert _registry_con(borra).run("borra", {"fact_id": 3}) == "borré 3"
    assert _registry_con(revisa).run("revisa", {"horas": 2}) == "2 h"
    assert _registry_con(revisa).run("revisa", {"horas": 1.5}) == "1.5 h"
    # Las MCP (strict_args=False) se quedan como estaban: su esquema y su
    # validación son del server.
    remota = _registry_con(borra, strict_args=False)
    assert remota.run("borra", {"fact_id": True}) == "borré True"


def test_registry_un_null_con_el_que_la_tool_se_atraganta_es_argumento_invalido(caplog):
    """music_control(player=None) y compañía hacen None.strip(): AttributeError,
    que el registry contaba como bug de la tool ("La tool ... reventó" y una
    traza en el log) aunque el fallo es del argumento. Igual que con un
    TypeError (C4), se le dice al modelo que corrija SUS argumentos."""
    def reproductor(action: str, player: str = "") -> str:
        return f"{action} en {player.strip() or 'el activo'}"

    registry = _registry_con(reproductor)
    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        for malo in (None, 3):
            out = registry.run("reproductor", {"action": "toggle", "player": malo})
            assert out.startswith("Argumentos inválidos para 'reproductor', patrón:"), out
    assert caplog.text == ""
    assert registry.run("reproductor", {"action": "toggle"}) == "toggle en el activo"


def test_registry_un_attributeerror_con_argumentos_correctos_sigue_siendo_un_bug(caplog):
    def rota(texto: str) -> str:
        return texto.no_existe()  # type: ignore[attr-defined]

    with caplog.at_level("ERROR", logger="crotolamo.tools"):
        out = _registry_con(rota).run("rota", {"texto": "x"})
    assert out.startswith("La tool 'rota' reventó, patrón:")
    assert "Traceback" in caplog.text


# ---------------------------------------------------------------------------
# Registry.run: el "Argumentos inválidos" de un argumento del tipo equivocado
# dice CUÁL corregir (x-tool-surface-wiring#27)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malo, tipo", [("6", "string"), (None, "null"), (2.0, "number")])
def test_registry_el_argumento_de_tipo_equivocado_sale_por_su_nombre(malo, tipo):
    """list_processes(limit="6") le contestaba al modelo "'>' not supported
    between instances of 'str' and 'int'": el texto de Python no dice qué
    argumento corregir ni qué tipo se esperaba."""
    def procesos(filtro: str = "", limit: int = 8) -> str:
        return "\n".join(["cabecera", *["p"] * max(1, limit)])

    out = _registry_con(procesos).run("procesos", {"filtro": "py", "limit": malo})
    assert out == (
        f"Argumentos inválidos para 'procesos', patrón: 'limit' tiene que ser integer, "
        f"no {tipo}"
    )


def test_registry_write_file_con_content_null_nombra_content_no_un_parametro_interno(corral):
    """write_file(content=None): el modelo leía "data must be str, not NoneType",
    y 'data' es el parámetro de Path.write_text, no uno que él mandara."""
    from crotolamo.tools import default_registry

    destino = corral / "importante.txt"
    destino.write_text("contenido valioso", encoding="utf-8")
    out = default_registry().run("write_file", {"path": str(destino), "content": None})
    assert out == (
        "Argumentos inválidos para 'write_file', patrón: 'content' tiene que ser string, "
        "no null"
    )
    assert destino.read_text(encoding="utf-8") == "contenido valioso"


def test_registry_nombra_todos_los_argumentos_de_tipo_equivocado():
    def dos(texto: str, veces: int) -> str:
        return texto.upper() * veces

    out = _registry_con(dos).run("dos", {"texto": 3, "veces": "2"})
    assert out == (
        "Argumentos inválidos para 'dos', patrón: 'texto' tiene que ser string, no integer; "
        "'veces' tiene que ser integer, no string"
    )
    assert _registry_con(dos).run("dos", {"texto": "ab", "veces": 2}) == "ABAB"

    # Si la tool no se atraganta, no se juzga nada: coaccionar o rechazar ANTES
    # de llamar es la decisión T16 (pendiente del dueño), no esto.
    def eco(veces: int) -> str:
        return f"{veces} veces"

    assert _registry_con(eco).run("eco", {"veces": "2"}) == "2 veces"


def test_de_fabrica_las_relativas_siguen_resolviendose_contra_el_cwd(tmp_path, monkeypatch):
    """Sin [paths].relative_base (la config de fábrica) no hay ancla: una ruta
    relativa de una tool de archivo es del CWD, como antes de la auditoría. El
    ancla al home es opt-in porque cambia dónde caen los archivos y añade una
    confirmación (el home es zona de confirmación)."""
    from crotolamo.tools import files

    fabrica = settings_mod.load_settings()
    assert fabrica.relative_base is None
    assert Guard.from_settings(fabrica).relative_base is None

    libre = (tmp_path / "libre").resolve()
    libre.mkdir()
    real = settings_mod.get_settings()
    monkeypatch.setattr(real, "allowed_roots", [libre])
    monkeypatch.setattr(real, "confirm_roots", [])
    monkeypatch.setattr(real, "relative_base", None)
    monkeypatch.setattr(settings_mod, "_SETTINGS", real)
    monkeypatch.chdir(libre)
    assert files._resolve("a.txt") == libre / "a.txt"
    assert "Escribí" in files.write_file("a.txt", "hola")
    assert (libre / "a.txt").read_text(encoding="utf-8") == "hola"
    # read_file y no write_file: esa siempre confirma por destructiva (safe=False).
    decision = Guard.from_settings(real).check(_nativa("read_file"), {"path": "a.txt"})
    assert decision.allowed and not decision.needs_confirmation
    assert files.read_file("a.txt") == "hola"


@pytest.mark.parametrize("valor, esperado", [
    ("", None),
    ("   ", None),
    ("~", Path.home()),
    ("/srv/datos", Path("/srv/datos")),
])
def test_relative_base_se_lee_de_la_config(tmp_path, valor, esperado):
    toml = tmp_path / "c.toml"
    toml.write_text(f'[paths]\nallowed_roots = ["/tmp"]\nrelative_base = "{valor}"\n',
                    encoding="utf-8")
    cargada = settings_mod.load_settings(toml)
    assert cargada.relative_base == esperado
    assert "relative_base" not in cargada.paths


@pytest.mark.parametrize("valor", ['"Documentos"', '"./x"', "3"])
def test_relative_base_relativa_o_no_string_no_arranca(tmp_path, valor):
    toml = tmp_path / "c.toml"
    toml.write_text(f'[paths]\nallowed_roots = ["/tmp"]\nrelative_base = {valor}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="relative_base"):
        settings_mod.load_settings(toml)
