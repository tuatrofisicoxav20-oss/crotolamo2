"""Revisión final, unidad 25: crotolamo/tools/media.py y crotolamo/tools/memoria.py.

1. music_control sin reproductor ("pausa la música", el camino del fast-path)
   pregunta quién suena y le manda la orden con `-p <playerName>`. Pero
   playerctl 2.4.1 (playerctl-common.c, pctl_player_name_string_instance_compare)
   casa `-p X` con la instancia X y con cualquier "X.<algo>", y sin -a la orden
   se queda en el PRIMERO de los elegidos que la acepta. Dos reproductores con
   el mismo nombre (dos navegadores Chromium o una app Electron:
   chromium.instance<pid>; dos VLC o mpv: vlc.instanceN) y el primero en
   pausa: `pause` caía en el ya pausado, la música seguía y la tool (que se
   dice tal cual) contestaba "Pausado, patrón."; `toggle` arrancaba el pausado
   encima del otro. La orden va ahora a la INSTANCIA ({{playerInstance}}),
   que solo casa consigo misma.

2. El respaldo SQLite de olvidar_recuerdo (mem0 caído) borraba el PRIMER hecho
   del ranking de facts.search que compartiera CUALQUIER palabra no vacía con
   la descripción. Los verbos genéricos ("llama", "gusta", "vive") y los
   interrogativos que faltaban en la lista ("dónde", "cuál", "quién") contaban
   como tema: "cómo se llama mi gato" borraba «mi jefe se llama Ramiro» y
   "dónde vive mi mamá" (sin hechos de mamá) «mi hermana vive en Monterrey»,
   anunciándolo como "Olvidado". Borrado permanente.
"""

from __future__ import annotations

import subprocess

import pytest

from crotolamo.core import memoria as memoria_mod
from crotolamo.core.memoria import Memoria, MemoriaConfig
from crotolamo.tools import media

# --- 1. music_control: la orden va a la instancia que suena -----------------------


class _PlayerctlInstancias:
    """playerctl de mentiras con las reglas de playerctl 2.4.1 sobre instancias:

    - el nombre (`{{playerName}}`) es la instancia hasta el primer '.';
    - `-p X` elige la instancia X y toda "X.<algo>" (distingue mayúsculas);
    - sin -a, la orden va al primero de los elegidos y ahí se acaba.
    """

    def __init__(self, *reproductores: tuple[str, str]) -> None:
        self.reps = [list(r) for r in reproductores]  # [instancia, estado]
        self.argvs: list[list[str]] = []

    def estados(self) -> dict[str, str]:
        return {instancia: estado for instancia, estado in self.reps}

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.argvs.append(list(args))
        resto = list(args)
        todos = "-a" in resto
        if todos:
            resto.remove("-a")
        elegidos = self.reps
        if "-p" in resto:
            i = resto.index("-p")
            p = resto[i + 1]
            del resto[i:i + 2]
            elegidos = [r for r in self.reps
                        if r[0] == p or (r[0].startswith(p) and r[0][len(p):].startswith("."))]
        formato = ""
        if "--format" in resto:
            i = resto.index("--format")
            formato = resto[i + 1]
            del resto[i:i + 2]
        (orden,) = resto
        if not elegidos:
            return subprocess.CompletedProcess(args, 1, "", "No players found\n")
        salida = ""
        for rep in elegidos:
            if orden == "metadata":
                linea = formato
                for clave, valor in (("playerName", rep[0].split(".", 1)[0]),
                                     ("playerInstance", rep[0]), ("status", rep[1]),
                                     ("artist", "Artista"), ("title", "Cancion")):
                    linea = linea.replace("{{%s}}" % clave, valor)
                salida += linea + "\n"
            elif orden == "play-pause":
                rep[1] = "Paused" if rep[1] == "Playing" else "Playing"
            elif orden in ("pause", "stop"):
                rep[1] = {"pause": "Paused", "stop": "Stopped"}[orden]
            if not todos:
                break
        return subprocess.CompletedProcess(args, 0, salida, "")


@pytest.fixture
def playerctl(monkeypatch):
    def instala(*reproductores: tuple[str, str]) -> _PlayerctlInstancias:
        falso = _PlayerctlInstancias(*reproductores)
        monkeypatch.setattr(media, "_run", falso)
        monkeypatch.setattr(media.shutil, "which", lambda _cmd: "/usr/bin/playerctl")
        return falso

    return instala


@pytest.mark.parametrize("accion, dicho, queda", [
    ("pause", "Pausado, patrón.", "Paused"),
    ("toggle", "Listo, patrón.", "Paused"),
    ("stop", "Parado, patrón.", "Stopped"),
])
def test_music_control_con_dos_instancias_del_mismo_reproductor_va_a_la_que_suena(
        playerctl, accion, dicho, queda):
    """Dos Chromium (o un navegador y una app Electron): la pestaña de la
    primera instancia está en pausa y suena la segunda."""
    falso = playerctl(("chromium.instance1111", "Paused"), ("chromium.instance2222", "Playing"))
    assert media.music_control(accion) == dicho
    assert falso.argvs[-1][:2] == ["-p", "chromium.instance2222"]
    # La que estaba en pausa ni se toca (con toggle, antes arrancaba encima).
    assert falso.estados() == {"chromium.instance1111": "Paused", "chromium.instance2222": queda}


def test_music_control_reproductor_de_una_sola_instancia_sigue_igual(playerctl):
    """Para Spotify la instancia ES el nombre: la orden no cambia."""
    falso = playerctl(("opera", "Paused"), ("spotify", "Playing"))
    assert media.music_control("pause") == "Pausado, patrón."
    assert falso.argvs[-1] == ["-p", "spotify", "pause"]
    assert falso.estados() == {"opera": "Paused", "spotify": "Paused"}


def test_players_devuelve_la_instancia(playerctl):
    playerctl(("chromium.instance1111", "Paused"), ("spotify", "Playing"))
    assert [p[0] for p in media._players()] == ["chromium.instance1111", "spotify"]


# --- 2. respaldo SQLite de olvidar: no borra por una palabra genérica --------------

_HECHOS = (
    "mi perro se llama Tletl",
    "tengo un gato negro",
    "me gusta el café de olla",
    "mi platillo favorito es el pozole",
    "mi hermana vive en Monterrey",
    "trabajo en Querétaro",
    "mi jefe se llama Ramiro",
    "uso Fedora con Hyprland",
)


@pytest.fixture
def libreta(tmp_path, monkeypatch):
    """La libreta SQLite en un fichero temporal y la memoria semántica apagada
    (es lo mismo que mem0 marcado roto: olvidar_recuerdo usa la libreta)."""
    from crotolamo.persistence import facts

    monkeypatch.setattr(facts.db, "_default_db_path", lambda: tmp_path / "f.sqlite")
    m = Memoria(MemoriaConfig(enabled=False))
    memoria_mod.set_memoria(m)
    yield facts
    m.cerrar()


def _tools():
    from crotolamo.tools import memoria as tools_mod

    return tools_mod


def _textos(libreta) -> list[str]:
    return [r["texto"] for r in libreta.recall()]


@pytest.mark.parametrize("descripcion, borrado", [
    ("cómo se llama mi gato", "tengo un gato negro"),   # antes: el perro o el jefe
    ("que me gusta el pozole", "mi platillo favorito es el pozole"),  # antes: el café
    ("quién es mi jefe", "mi jefe se llama Ramiro"),
    ("lo de mi hermana", "mi hermana vive en Monterrey"),
    ("olvida lo del cafe", "me gusta el café de olla"),
])
def test_olvidar_libreta_borra_el_hecho_del_tema_no_el_que_comparte_un_verbo(
        libreta, descripcion, borrado):
    for h in _HECHOS:
        libreta.remember(h)
    out = _tools().olvidar_recuerdo(descripcion)
    assert out.startswith(f"Olvidado, patrón: «{borrado}»"), out
    assert _textos(libreta) == [h for h in _HECHOS if h != borrado]


@pytest.mark.parametrize("descripcion", [
    "dónde vive mi mamá",        # antes: «mi hermana vive en Monterrey»
    "lo de que me gusta el té",  # antes: «me gusta el café de olla»
    "mi color favorito",         # antes: «mi platillo favorito es el pozole»
    "cuál es mi color favorito",
    "cuándo es mi cumpleaños",
])
def test_olvidar_libreta_sin_hecho_del_tema_no_borra_nada(libreta, descripcion):
    for h in _HECHOS:
        libreta.remember(h)
    out = _tools().olvidar_recuerdo(descripcion)
    assert out.startswith("No encontré ningún recuerdo") and descripcion in out, out
    assert _textos(libreta) == list(_HECHOS)


def test_olvidar_libreta_patron_no_es_un_tema(libreta):
    """Los hechos se guardan también en tercera persona ("el patrón ..."):
    compartir "patrón" no dice que un hecho trate de lo pedido."""
    libreta.remember("el patrón toma el café sin azúcar")
    assert _tools().olvidar_recuerdo("dónde vive el patrón").startswith("No encontré")
    assert _textos(libreta) == ["el patrón toma el café sin azúcar"]


def test_olvidar_libreta_si_empatan_dos_hechos_no_borra_ninguno_y_lo_dice(libreta):
    """Dos hechos del perro y "olvida lo de mi perro": no hay uno mejor. Se
    pregunta cuál en vez de borrar uno a ciegas, y sin decir "no encontré"."""
    hechos = ["mi perro se llama Tletl", "mi perro come croquetas", "uso Fedora con Hyprland"]
    for h in hechos:
        libreta.remember(h)
    out = _tools().olvidar_recuerdo("lo de mi perro")
    assert not out.startswith(("Olvidado", "No encontré")), out
    assert "«mi perro se llama Tletl»" in out and "«mi perro come croquetas»" in out
    assert "no borré" in out
    assert _textos(libreta) == hechos


def test_olvidar_libreta_el_verbo_desempata_entre_hechos_del_tema(libreta):
    """El verbo no decide solo, pero entre dos hechos del perro sí dice cuál."""
    hechos = ["mi perro come croquetas", "mi perro se llama Tletl"]
    for h in hechos:
        libreta.remember(h)
    out = _tools().olvidar_recuerdo("cómo se llama mi perro")
    assert out.startswith("Olvidado, patrón: «mi perro se llama Tletl»"), out
    assert _textos(libreta) == ["mi perro come croquetas"]
    out = _tools().olvidar_recuerdo("que mi perro come croquetas")
    assert out.startswith("Olvidado, patrón: «mi perro come croquetas»"), out


@pytest.mark.parametrize("hecho, descripcion", [
    ("vivo en Querétaro", "dónde vivo"),
    ("me llamo Juan", "cómo me llamo"),
])
def test_olvidar_libreta_si_solo_hay_verbo_el_verbo_es_el_tema(libreta, hecho, descripcion):
    """Sin otra palabra en la descripción, la genérica sigue siendo el tema."""
    libreta.remember(hecho)
    libreta.remember("uso Fedora con Hyprland")
    out = _tools().olvidar_recuerdo(descripcion)
    assert out.startswith(f"Olvidado, patrón: «{hecho}»"), out
    assert _textos(libreta) == ["uso Fedora con Hyprland"]


def test_olvidar_libreta_un_hecho_repetido_no_es_un_empate(libreta):
    """Dos copias del mismo hecho (recordar dos veces con mem0 caído) son UN
    recuerdo: se olvida como antes, no se pregunta "¿cuál?" para siempre."""
    # Con add_fact (el INSERT crudo): remember() ya no deja copias, pero una
    # libreta de antes puede traerlas.
    for h in ("me gusta el pozole", "Me gusta el pozole", "uso Fedora con Hyprland"):
        libreta.db.add_fact(h)
    out = _tools().olvidar_recuerdo("lo del pozole")
    assert out.startswith("Olvidado, patrón: «me gusta el pozole»"), out
    assert _textos(libreta) == ["Me gusta el pozole", "uso Fedora con Hyprland"]
