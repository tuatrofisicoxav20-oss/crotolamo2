"""Revisión final, unidad 46: desktop/panel (config del modo de escucha).

Lógica pura de panel.config, sin gi: el panel lee el modo de listener.env como
lo aplica systemd (EnvironmentFile=) y sabe si el modo elegido quedó guardado.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# desktop/ no es paquete: se importa igual que en producción (sys.path)
DESKTOP_DIR = str(Path(__file__).resolve().parent.parent / "desktop")
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from panel import config  # noqa: E402


# ---------------------------------------------------------------------------
# El lector decodifica como el escritor: write_mode_args conserva a propósito
# los bytes que no son UTF-8; leerlos con UTF-8 estricto daba el modo por
# defecto aunque el servicio corriera en otro.
# ---------------------------------------------------------------------------

def test_read_mode_args_lee_un_archivo_con_bytes_que_no_son_utf8(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_bytes(b"# caf\xe9 (latin-1)\nCROTOLAMO_LISTEN_ARGS=--simple\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.read_mode_args() == "--simple"


def test_read_mode_args_tras_escribir_sobre_bytes_que_no_son_utf8(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_bytes(b"# caf\xe9 (latin-1)\nCROTOLAMO_LISTEN_ARGS=--simple\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.write_mode_args("--barge-in") is True
    assert env.read_bytes() == b"# caf\xe9 (latin-1)\nCROTOLAMO_LISTEN_ARGS=--barge-in\n"
    assert config.read_mode_args() == "--barge-in"


# ---------------------------------------------------------------------------
# Con varias asignaciones systemd aplica la ÚLTIMA: el panel lee la misma.
# ---------------------------------------------------------------------------

def test_read_mode_args_con_dos_asignaciones_lee_la_ultima(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    # p. ej. `echo CROTOLAMO_LISTEN_ARGS=--barge-in >> listener.env` a mano
    env.write_text("CROTOLAMO_LISTEN_ARGS=--simple\n"
                   "CROTOLAMO_LOG=DEBUG\n"
                   "CROTOLAMO_LISTEN_ARGS=--barge-in\n"
                   "# CROTOLAMO_LISTEN_ARGS=--no-barge-in\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.read_mode_args() == "--barge-in"


def test_read_mode_args_sin_asignacion_da_el_modo_por_defecto(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    env.write_text("# CROTOLAMO_LISTEN_ARGS=--simple\nCROTOLAMO_LOG=DEBUG\n",
                   encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.read_mode_args() == "--no-barge-in"


# ---------------------------------------------------------------------------
# write_mode_args dice si el modo quedó guardado: el panel no debe reiniciar
# el servicio (recarga de modelos, segundos sordo) por un modo que no está en
# disco.
# ---------------------------------------------------------------------------

def test_write_mode_args_devuelve_true_si_queda_guardado(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    assert config.write_mode_args("--simple") is True
    assert config.read_mode_args() == "--simple"


def test_write_mode_args_devuelve_false_si_falla_el_reemplazo(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    viejo = "CROTOLAMO_LISTEN_ARGS=--no-barge-in\n"
    env.write_text(viejo, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)

    def falla(src, dst):
        raise OSError(28, "No queda espacio en el dispositivo")

    monkeypatch.setattr(os, "replace", falla)
    assert config.write_mode_args("--barge-in") is False  # no lanza: lo dice
    assert env.read_text(encoding="utf-8") == viejo


def test_write_mode_args_devuelve_false_si_no_pudo_leer_el_archivo(tmp_path, monkeypatch):
    env = tmp_path / "listener.env"
    viejo = "CROTOLAMO_LISTEN_ARGS=--no-barge-in\n"
    env.write_text(viejo, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    real_read_bytes = Path.read_bytes

    def sin_permiso(self):
        if self == env:
            raise PermissionError(13, "Permiso denegado")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", sin_permiso)
    assert config.write_mode_args("--simple") is False  # no se pisa a ciegas
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)
    assert env.read_text(encoding="utf-8") == viejo
