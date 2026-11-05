#!/bin/sh
# Build only. This never downloads a model or starts an inference service.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PYTHON:-}" ]; then
  BOOTSTRAP_PYTHON=$PYTHON
elif command -v python3.13 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON=$(command -v python3.13)
elif command -v uv >/dev/null 2>&1; then
  if ! BOOTSTRAP_PYTHON=$(uv python find 3.13); then
    printf '%s\n' 'Python 3.13 is required. Run: uv python install 3.13' >&2
    exit 1
  fi
else
  printf '%s\n' 'Install Python 3.13 or uv, then run: uv python install 3.13' >&2
  exit 1
fi
exec "$BOOTSTRAP_PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@"
