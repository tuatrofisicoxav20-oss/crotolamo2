#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from core.test_runner import handle_test_command, run_project_tests, format_report
text=" ".join(sys.argv[1:]) if len(sys.argv)>1 else "test crotolamo"
print(handle_test_command(text,ROOT) or format_report(run_project_tests(text,ROOT)))
