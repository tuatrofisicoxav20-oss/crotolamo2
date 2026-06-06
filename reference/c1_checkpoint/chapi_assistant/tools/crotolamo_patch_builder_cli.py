#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from core.patch_builder import handle_patch_command, list_patches
text=" ".join(sys.argv[1:]) if len(sys.argv)>1 else "parches"
print(handle_patch_command(text,ROOT) or list_patches(ROOT))
