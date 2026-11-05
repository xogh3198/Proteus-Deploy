#!/usr/bin/env bash
set -euo pipefail
DIAG_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd -- "$DIAG_ROOT"
exec python3 diagnostics/onnx_forward.py "$@"
