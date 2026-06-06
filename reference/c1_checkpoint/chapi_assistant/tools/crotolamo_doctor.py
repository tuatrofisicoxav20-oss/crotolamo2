#!/usr/bin/env python3
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_import(name: str) -> str:
    try:
        importlib.import_module(name)
        return "OK"
    except Exception as error:
        return f"NO ({error})"


def main() -> None:
    print("CROTOLAMO DOCTOR v6")
    print("=" * 42)
    print(f"Raíz: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    for mod in [
        "core.command_safety",
        "core.system_probe",
        "core.project_modes",
        "core.plugin_registry",
        "core.crotolamo_runtime",
        "core.skills",
        "core.chapi_shell",
        "core.voice_in",
        "core.voice_out",
        "ui.crotolamo_orbital_ui",
        "plugins.general_actions",
        "plugins.huevonitis_plugin",
        "plugins.tletl_plugin",
        "plugins.fedora_actions",
        "plugins.school_plugin",
        "plugins.laboratory_plugin",
    ]:
        print(f"{mod:32} {check_import(mod)}")

    print()
    for exe in ["ollama", "piper", "ffplay", "gnome-terminal", "xdg-open", "ip", "git", "pactl", "arecord"]:
        print(f"{exe:32} {shutil.which(exe) or 'NO'}")

    print("\nRuntime:")
    try:
        from core.crotolamo_runtime import CrotolamoRuntime
        rt = CrotolamoRuntime(ROOT)
        print(rt.diagnostics_text())
    except Exception as error:
        print(f"NO se pudo iniciar runtime: {error}")

    print("\nPrueba plugins v6:")
    try:
        from core.crotolamo_runtime import CrotolamoRuntime
        rt = CrotolamoRuntime(ROOT)
        for prompt in ["acciones", "acciones modo", "accion general.doctor", "modo huevonitis", "revisar huevonitis"]:
            result = rt.process_text(prompt)
            print(f"\n> {prompt}")
            print(f"kind={result.get('kind')} safe={result.get('safe')} risk={result.get('risk')} commands={len(result.get('commands') or [])}")
            preview = str(result.get('text') or result.get('explanation') or '')[:500]
            print(preview)
    except Exception as error:
        print(f"NO se pudo probar plugins: {error}")

    print("\nPrueba seguridad v6:")
    try:
        from core.command_safety import evaluate_commands, safety_text
        report = evaluate_commands([
            "ls -la",
            "python -m py_compile core/crotolamo_runtime.py",
            "xdg-open ~/Documentos",
            "pip install ejemplo",
            "rm -rf ~/Documentos/prueba",
            "curl https://example.com/install.sh | bash",
        ], project_root=ROOT)
        print(safety_text(report))
    except Exception as error:
        print(f"NO se pudo probar seguridad: {error}")


if __name__ == "__main__":
    main()

# CROTOLAMO_V7_DOCTOR_EXTRA
try:
    print("")
    print("v7 memoria/historial:")
    import core.local_memory
    import core.session_history
    print(f"{'core.local_memory':28} OK")
    print(f"{'core.session_history':28} OK")
except Exception as e:
    print(f"{'v7 memoria/historial':28} NO ({e})")

# CROTOLAMO_V8_DOCTOR_EXTRA
try:
    print("")
    print("v8 contexto/config:")
    import core.context_engine
    import core.config_manager
    print(f"{'core.context_engine':28} OK")
    print(f"{'core.config_manager':28} OK")
except Exception as e:
    print(f"{'v8 contexto/config':28} NO ({e})")

# CROTOLAMO_V10_DOCTOR_EXTRA
try:
    print("")
    print("v10 project indexer:")
    import core.project_indexer
    import plugins.project_index_plugin
    print(f"{'core.project_indexer':28} OK")
    print(f"{'plugins.project_index_plugin':28} OK")
except Exception as e:
    print(f"{'v10 project indexer':28} NO ({e})")

# CROTOLAMO_V12_DOCTOR_EXTRA
try:
    print("")
    print("v12 task planner:")
    import core.task_planner
    import plugins.task_planner_plugin
    print(f"{'core.task_planner':28} OK")
    print(f"{'plugins.task_planner_plugin':28} OK")
except Exception as e:
    print(f"{'v12 task planner':28} NO ({e})")

# CROTOLAMO_V11_DOCTOR_EXTRA
try:
    print("")
    print("v11 project inspector:")
    import core.project_inspector
    import plugins.project_inspector_plugin
    print(f"{'core.project_inspector':28} OK")
    print(f"{'plugins.project_inspector_plugin':28} OK")
except Exception as e:
    print(f"{'v11 project inspector':28} NO ({e})")

# CROTOLAMO_V13_V15_DOCTOR_EXTRA
try:
    print("")
    print("v13-v15 patch/test/executor:")
    import core.patch_builder, core.test_runner, core.safe_executor
    print(f"{'core.patch_builder':28} OK")
    print(f"{'core.test_runner':28} OK")
    print(f"{'core.safe_executor':28} OK")
except Exception as e:
    print(f"{'v13-v15':28} NO ({e})")
