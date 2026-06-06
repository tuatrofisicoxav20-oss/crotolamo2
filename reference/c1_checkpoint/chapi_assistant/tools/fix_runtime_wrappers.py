#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "core" / "crotolamo_runtime.py"
BACKUP_DIR = ROOT / "backups" / "runtime_wrapper_fix"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

MARKERS = [
    "CROTOLAMO_V7_MEMORY_WRAPPER",
    "CROTOLAMO_RUNTIME_INTERRUPT_PATCH_v7_1",
    "CROTOLAMO_V8_CONTEXT_WRAPPER",
    "CROTOLAMO_V9_BRAIN_WRAPPER",
    "CROTOLAMO_V10_PROJECT_INDEX_WRAPPER",
    "CROTOLAMO_V11_PROJECT_INSPECTOR_WRAPPER",
    "CROTOLAMO_V12_TASK_PLANNER_WRAPPER",
    "CROTOLAMO_V13_V15_FINAL_WRAPPER",
    "CROTOLAMO_FIXED_COMBINED_RUNTIME_WRAPPER_v1",
]

FIXED_WRAPPER = r"""
# CROTOLAMO_FIXED_COMBINED_RUNTIME_WRAPPER_v1
# Wrapper corregido: process_text(self, text, *args, **kwargs)
try:
    from pathlib import Path as _CFWPath
    _CFW_ROOT = _CFWPath(__file__).resolve().parents[1]

    def _cfw_first_direct_handler(raw_text):
        handlers = []

        for modname, fname in [
            ("core.brain_engine", "handle_brain_command"),
            ("core.context_engine", "handle_context_command"),
            ("core.config_manager", "handle_config_command"),
            ("core.local_memory", "handle_memory_command"),
            ("core.session_history", "handle_history_command"),
            ("core.project_indexer", "handle_project_index_command"),
            ("core.project_inspector", "handle_project_inspector_command"),
            ("core.task_planner", "handle_task_planner_command"),
            ("core.safe_executor", "handle_executor_command"),
            ("core.test_runner", "handle_test_command"),
            ("core.patch_builder", "handle_patch_command"),
        ]:
            try:
                mod = __import__(modname, fromlist=[fname])
                handlers.append(getattr(mod, fname))
            except Exception:
                pass

        for handler in handlers:
            try:
                result = handler(raw_text, _CFW_ROOT)
                if result is not None:
                    return result
            except Exception as e:
                return f"Error en handler directo {getattr(handler, '__name__', handler)}: {type(e).__name__}: {e}"
        return None

    def _cfw_prepare_text(raw_text):
        text = str(raw_text or "")

        try:
            from core.local_memory import get_alias
            alias = get_alias(text, _CFW_ROOT)
            if alias:
                text = alias
        except Exception:
            pass

        try:
            from core.brain_engine import brain_config, is_internal_or_direct, build_brain_prompt
            cfg = brain_config(_CFW_ROOT)
            if bool(cfg.get("enabled", True)) and not is_internal_or_direct(text):
                text = build_brain_prompt(text, _CFW_ROOT)
        except Exception as e:
            text = f"{text}\n\n[AVISO: falló brain_engine: {type(e).__name__}: {e}]"

        try:
            from core.config_manager import get_value
            from core.context_engine import should_skip_context, build_enriched_prompt
            enabled = bool(get_value("ollama.use_context_engine", True, _CFW_ROOT))
            if enabled and not should_skip_context(text):
                text = build_enriched_prompt(text, _CFW_ROOT)
        except Exception as e:
            text = f"{text}\n\n[AVISO: falló context_engine: {type(e).__name__}: {e}]"

        return text

    def _cfw_wrap_method(original):
        def _wrapped(self, text, *args, **kwargs):
            raw_text = str(text or "")

            try:
                from core.session_history import log_event
                log_event("user", raw_text, _CFW_ROOT)
            except Exception:
                pass

            try:
                direct = _cfw_first_direct_handler(raw_text)
                if direct is not None:
                    try:
                        from core.session_history import log_event
                        log_event("assistant", str(direct), _CFW_ROOT, {"source": "direct_handler"})
                    except Exception:
                        pass
                    return direct

                prepared = _cfw_prepare_text(raw_text)
                result = original(self, prepared, *args, **kwargs)

                try:
                    from core.session_history import log_event
                    log_event("assistant", str(result), _CFW_ROOT)
                except Exception:
                    pass

                return result

            except KeyboardInterrupt:
                return "Operación cancelada con Ctrl+C. Crotolamo dejó de esperar sin vomitar traceback."
            except Exception as e:
                return f"Error en runtime wrapper corregido: {type(e).__name__}: {e}"

        _wrapped._crotolamo_fixed_wrapper = True
        return _wrapped

    if "CrotolamoRuntime" in globals():
        _cls = globals()["CrotolamoRuntime"]
        if hasattr(_cls, "process_text") and not getattr(_cls.process_text, "_crotolamo_fixed_wrapper", False):
            _cls.process_text = _cfw_wrap_method(_cls.process_text)

except Exception:
    pass
"""

def remove_old_blocks(text: str) -> str:
    marker_pattern = "|".join(re.escape(m) for m in MARKERS)
    pattern = re.compile(
        rf"\n# (?:{marker_pattern})\n.*?(?=\n# CROTOLAMO_|\Z)",
        re.DOTALL,
    )
    return pattern.sub("", text).rstrip() + "\n"

def main() -> int:
    if not RUNTIME.exists():
        print(f"ERROR: no existe {RUNTIME}")
        return 1

    original = RUNTIME.read_text(encoding="utf-8", errors="replace")
    backup = BACKUP_DIR / f"crotolamo_runtime_{datetime.now().strftime('%Y%m%d-%H%M%S')}.py"
    backup.write_text(original, encoding="utf-8")

    cleaned = remove_old_blocks(original)
    RUNTIME.write_text(cleaned.rstrip() + "\n\n" + FIXED_WRAPPER.lstrip(), encoding="utf-8")

    print("Runtime corregido.")
    print(f"Backup: {backup}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
