#!/usr/bin/env bash
set -euo pipefail
BASELINE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASELINE_REPO_ROOT"
exec "${PYTHON:-python3}" -m proteus_baselines.deploy "$@"
