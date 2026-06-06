#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python tools/crotolamo_doctor.py
