#!/bin/sh
# Stable entrypoint; all persistent state lives outside this checkout.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STATE=${OMLX_DS_STATE:-"$HOME/.local/share/omlx-ds-256"}
PYTHON="$STATE/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  printf '%s\n' "Missing isolated Python environment. Run $SCRIPT_DIR/bootstrap.sh first." >&2
  exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/control.py" "$@"
