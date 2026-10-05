"""Tests de music_now SIN playerctl real: se parchea media._run (el único punto
que ejecuta playerctl) y shutil.which, así no dependen del escritorio del patrón.
"""

import subprocess

from crotolamo.tools import media


def _fake_playerctl(monkeypatch, metadata="Artista — Cancion\n", status="Playing\n"):
    """playerctl de mentiras: `metadata` y `status` son sus stdout (con el "\\n"
    final real que imprime el binario)."""
    def fake_run(args):
        stdout = metadata if args[0] == "metadata" else status
        return subprocess.CompletedProcess(
            args=["playerctl", *args], returncode=0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(media, "_run", fake_run)
    monkeypatch.setattr(media.shutil, "which", lambda _cmd: "/usr/bin/playerctl")


def test_music_now_sin_salto_de_linea_final(monkeypatch):
    """playerctl termina la línea con "\\n" y strip(" —") no lo quitaba: la frase
    (que acaba en el TTS) salía con un salto de línea pegado al título."""
    _fake_playerctl(monkeypatch)
    out = media.music_now()
    assert out == "Sonando, patrón: Artista — Cancion"
    assert "\n" not in out


def test_music_now_sin_artista_limpia_el_separador(monkeypatch):
    _fake_playerctl(monkeypatch, metadata=" — Solo titulo\n", status="Paused\n")
    assert media.music_now() == "En pausa, patrón: Solo titulo"


def test_music_now_sin_metadatos(monkeypatch):
    _fake_playerctl(monkeypatch, metadata=" — \n")
    out = media.music_now()
    assert "sin metadatos" in out
    assert "\n" not in out
