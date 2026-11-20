"""Revisión final — paquete e instalación (pyproject, CI, install.sh, launch.sh).

Cada test fija un hallazgo confirmado: falla con el código de antes y pasa con el
arreglo.

- La extra [voice] trae silero-vad, que declara torch como dependencia dura. Solo
  con PyPI, en Linux x86_64 pip resuelve la build CUDA: torch + 17 ruedas
  nvidia-*/cuda-* + triton, ~3 GB que en la lap sin GPU no corren nunca (el camino
  de voz ni importa torch: runtime.py usa el silero_vad.onnx del paquete). Solo la
  CI pasaba el índice CPU de PyTorch; las instrucciones que sigue el patrón
  (install.sh sin venv, launch.sh sin venv, el comentario de pyproject) no. Los
  scripts se ejecutan DE VERDAD y el comando que sugieren se corre con un
  python3/pip de mentiras delante en el PATH: lo que importa es lo que recibe pip.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
_CI = RAIZ / ".github" / "workflows" / "ci.yml"

# python3 de mentiras: `-m venv <dir>` crea <dir>/bin/pip, que apunta sus
# argumentos (uno por línea) en $FAKE_DIR/pip.args; cualquier otra llamada sale 0.
_PYTHON3_FALSO = r"""#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    cat > "$3/bin/pip" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" > "$FAKE_DIR/pip.args"
EOF
    chmod +x "$3/bin/pip"
fi
exit 0
"""


def _indice_cpu_de_la_ci() -> str:
    """El índice que usa la CI (la instalación que sí se prueba): la referencia."""
    urls = re.findall(r"--extra-index-url\s+(\S+)", _CI.read_text(encoding="utf-8"))
    assert len(urls) == 1, urls
    assert urls[0].rstrip("/").endswith("/whl/cpu"), urls[0]
    return urls[0]


def _lineas_logicas(texto: str) -> list[str]:
    """Une las líneas que acaban en `\\` (continuación de shell) con la siguiente."""
    logicas: list[str] = []
    pendiente = ""
    for linea in texto.splitlines():
        if linea.rstrip().endswith("\\"):
            pendiente += linea.rstrip()[:-1] + " "
            continue
        logicas.append(pendiente + linea)
        pendiente = ""
    if pendiente:
        logicas.append(pendiente)
    return logicas


def _comando_sugerido(salida: str) -> str:
    """El comando de instalación que imprime el script, desde `cd`/`python3`."""
    candidatas = [linea for linea in _lineas_logicas(salida) if "pip install" in linea]
    assert len(candidatas) == 1, salida
    inicio = re.search(r"\b(cd|python3) ", candidatas[0])
    assert inicio, candidatas[0]
    return candidatas[0][inicio.start():]


def _bin_falso(tmp_path: Path) -> tuple[Path, Path]:
    binarios, fake = tmp_path / "bin", tmp_path / "fake"
    binarios.mkdir()
    fake.mkdir()
    python3 = binarios / "python3"
    python3.write_text(_PYTHON3_FALSO, encoding="utf-8")
    python3.chmod(0o755)
    return binarios, fake


def _entorno(tmp_path: Path, binarios: Path, fake: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return {**os.environ, "HOME": str(home), "FAKE_DIR": str(fake),
            "PATH": f"{binarios}{os.pathsep}{os.environ['PATH']}"}


def _pip_recibe(comando: str, cwd: Path, entorno: dict[str, str], fake: Path) -> list[str]:
    """Corre el comando sugerido tal cual (bash -c) y devuelve los args de pip."""
    out = subprocess.run(["bash", "-c", comando], cwd=cwd, env=entorno, capture_output=True,
                         text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, (comando, out.stdout, out.stderr)
    return (fake / "pip.args").read_text(encoding="utf-8").splitlines()


def _assert_indice_cpu(args: list[str], extra: str) -> None:
    assert args[:3] == ["install", "-e", extra], args
    assert "--extra-index-url" in args, (
        f"pip install -e '{extra}' sin el índice CPU de PyTorch: silero-vad arrastra la "
        f"build CUDA de torch (~3 GB de nvidia-*/triton) en una lap sin GPU: {args}")
    assert args[args.index("--extra-index-url") + 1] == _indice_cpu_de_la_ci(), args


def test_install_sh_sin_venv_sugiere_instalar_la_voz_con_el_indice_cpu(tmp_path):
    raiz = tmp_path / "repo"
    (raiz / "desktop").mkdir(parents=True)
    (raiz / "desktop/install.sh").write_bytes((RAIZ / "desktop/install.sh").read_bytes())
    binarios, fake = _bin_falso(tmp_path)
    entorno = _entorno(tmp_path, binarios, fake)
    out = subprocess.run(["bash", str(raiz / "desktop/install.sh")], env=entorno,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "No encuentro el venv" in out.stdout
    # El comando lleva su propio `cd "<repo>"`: se corre desde otra carpeta.
    args = _pip_recibe(_comando_sugerido(out.stdout), tmp_path, entorno, fake)
    _assert_indice_cpu(args, ".[voice]")
    assert (raiz / ".venv/bin/pip").exists()   # el venv quedó en el repo, no en el cwd


def test_launch_sh_sin_venv_sugiere_instalar_la_voz_con_el_indice_cpu(tmp_path):
    raiz = tmp_path / "repo"
    raiz.mkdir()
    (raiz / "launch.sh").write_bytes((RAIZ / "launch.sh").read_bytes())
    binarios, fake = _bin_falso(tmp_path)
    entorno = _entorno(tmp_path, binarios, fake)
    out = subprocess.run(["bash", str(raiz / "launch.sh"), "version"], env=entorno,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "No hay .venv" in out.stdout
    # launch.sh ya está en la raíz del repo: el comando se corre desde ahí.
    args = _pip_recibe(_comando_sugerido(out.stdout), raiz, entorno, fake)
    _assert_indice_cpu(args, ".[dev,voice]")


def test_pyproject_y_ci_instalan_la_voz_con_el_indice_cpu():
    """Ningún `pip install ... [voice]` de pyproject (el comentario es lo primero que
    lee quien instala a mano) ni de la CI puede quedarse sin el índice CPU: es justo
    el que, copiado tal cual, baja ~3 GB. Los de install.sh y launch.sh se imprimen
    en varios `echo` y por eso se prueban arriba ejecutándolos de verdad."""
    indice = _indice_cpu_de_la_ci()
    vistos = 0
    for nombre in ("pyproject.toml", ".github/workflows/ci.yml"):
        for linea in _lineas_logicas((RAIZ / nombre).read_text(encoding="utf-8")):
            if "pip install" in linea and re.search(r"[\[,]voice\]", linea):
                vistos += 1
                assert indice in linea, f"{nombre}: {linea.strip()}"
    assert vistos >= 2, vistos   # el comentario de pyproject y el paso de la CI
