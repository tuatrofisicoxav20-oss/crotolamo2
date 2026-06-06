#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from core.safe_executor import handle_executor_command
text=" ".join(sys.argv[1:]) if len(sys.argv)>1 else "executor"
print(handle_executor_command(text,ROOT) or "Comando no reconocido.")
