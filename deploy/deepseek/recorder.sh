#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STATE=${OMLX_DS_STATE:-"$HOME/.local/share/omlx-ds-256"}
exec "$STATE/venv/bin/python" "$SCRIPT_DIR/recorder_control.py" "$@"
