#!/usr/bin/env bash
set -euo pipefail
RESNET_TASK_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd -- "$RESNET_TASK_ROOT"
exec python3 -m experiments.resnet.host "$@"
