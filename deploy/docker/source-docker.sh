#!/usr/bin/env bash
set -euo pipefail
SOURCE_TASK_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd -- "$SOURCE_TASK_ROOT"
exec bash experiments/resnet/run.sh "$@"
