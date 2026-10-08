#!/usr/bin/env bash
# Standalone real-GPU Proteus port; never starts/calls an INFaaS service.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
if [[ -z "${PROTEUS_ENV:-}" ]]; then
  if [[ -f "$HERE/proteus-docker.local.env" ]]; then
    PROTEUS_ENV="$HERE/proteus-docker.local.env"
  else
    PROTEUS_ENV="$HERE/proteus-docker.env"
  fi
fi
export PROTEUS_ENV
get() { python3 -m proteus_realtime.deployment get "$1"; }
WORKER_IMAGE="$(get WORKER_IMAGE)"
MODEL_DIR="$(get MODEL_DIR)"
MODEL_REGISTRY="$(get MODEL_REGISTRY)"
DATA_DIR="$(get DATA_DIR)"
GPU="$(get GPU)"
APPLICATION="$(get APPLICATION)"
SLO_MS="$(get SLO_MS)"
NETWORK_NAME="$(get NETWORK_NAME)"
CONTAINER_NAME="$(get CONTAINER_NAME)"
LICENSE_FILE="$(get GUROBI_LICENSE_FILE)"
COMPOSE=(docker compose --project-name proteus-standalone -f "$HERE/generated/docker-compose.yml")
CONFIG_MOUNT=(-v "$MODEL_REGISTRY:/config/models.json:ro")
LICENSE_MOUNTS=()
if [[ -n "$LICENSE_FILE" ]]; then
  LICENSE_MOUNTS=(-v "$LICENSE_FILE:/license/gurobi.lic:ro" -e GRB_LICENSE_FILE=/license/gurobi.lic)
fi

image_required() {
  if ! docker image inspect "$WORKER_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image $WORKER_IMAGE missing. Run '$0 build' first." >&2
    exit 1
  fi
  local runtime_kind
  runtime_kind="$(docker image inspect --format '{{index .Config.Labels "org.proteus.runtime"}}' "$WORKER_IMAGE")"
  if [[ "$runtime_kind" != standalone-v1 ]]; then
    echo "ERROR: $WORKER_IMAGE is not the standalone image. Run '$0 build'; do not use the old bootstrap image." >&2
    exit 1
  fi
}

client() {
  local command="$1" image="${2:-}"
  shift 2
  image_required
  mkdir -p "$DATA_DIR/results"
  local mounts=(-v "$DATA_DIR/results:/results")
  local image_args=()
  if [[ -n "$image" ]]; then
    [[ -f "$image" ]] || { echo "Input missing: $image" >&2; exit 1; }
    image="$(realpath "$image")"
    mounts+=(-v "$image:/in/image.jpg:ro")
    image_args=(--image /in/image.jpg)
  fi
  docker run --rm --pull=never --user "$(id -u):$(id -g)" --network "$NETWORK_NAME" \
    "${mounts[@]}" "$WORKER_IMAGE" proteus_realtime.client "$command" \
    --controller proteus-worker:9000 --output /results "${image_args[@]}" "$@"
}

usage() {
  printf '%s\n' \
    "Standalone Proteus: one GPU/application, real MILP + deadline-based adaptive batching." \
    "  $0 settings | gpus | build | doctor | sample | download [--models NAME ...]" \
    "  $0 seed --from /existing/HF_HOME [--models resnet-50]" \
    "  $0 up | state [--watch 2] | logs [--follow] | ps | down" \
    "  $0 register [--models NAME ...] [--image IMAGE] [--trials 30]" \
    "  $0 query imagenet IMAGE 200 [-n 11]" \
    "  $0 test [IMAGE] [--warm-requests 10]  # fixed1 diagnostic, restores policy" \
    "  $0 policy-test [IMAGE]  # verifies actual batch>1 and MILP plan" \
    "  $0 bench [IMAGE] --rate 10 --duration 30 --concurrency 64" \
    "  $0 evaluate --manifest manifest.json  # files inside configured EVAL_DIR" \
    "  $0 mode --mode fixed|proteus | reset"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  help|-h|--help) usage ;;
  settings|gpus|gen) python3 -m proteus_realtime.deployment "$command" ;;
  build)
    docker build --build-arg "PYTHON_IMAGE=$(get PYTHON_IMAGE)" \
      --build-arg "TORCH_VERSION=$(get TORCH_VERSION)" \
      --build-arg "TORCHVISION_VERSION=$(get TORCHVISION_VERSION)" \
      -f "$HERE/worker.Dockerfile" -t "$WORKER_IMAGE" "$ROOT"
    ;;
  doctor)
    image_required
    python3 -m proteus_realtime.deployment idle
    mkdir -p "$MODEL_DIR"
    docker run --rm --pull=never --user "$(id -u):$(id -g)" --gpus "device=$GPU" \
      "${CONFIG_MOUNT[@]}" "${LICENSE_MOUNTS[@]}" -v "$MODEL_DIR:/models:ro" \
      "$WORKER_IMAGE" proteus_realtime.doctor
    ;;
  download)
    image_required
    mkdir -p "$MODEL_DIR"
    docker run --rm --pull=never --user "$(id -u):$(id -g)" \
      "${CONFIG_MOUNT[@]}" -v "$MODEL_DIR:/models" \
      -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
      "$WORKER_IMAGE" proteus_realtime.download weights "$@"
    ;;
  seed)
    image_required
    [[ $# -ge 2 && "$1" == --from && -d "$2" ]] || { usage; exit 1; }
    source_dir="$(realpath "$2")"
    shift 2
    [[ "$source_dir" != "$MODEL_DIR" ]] || { echo "Source and destination must differ." >&2; exit 1; }
    mkdir -p "$MODEL_DIR"
    docker run --rm --pull=never --user "$(id -u):$(id -g)" \
      "${CONFIG_MOUNT[@]}" -v "$source_dir:/source:ro" -v "$MODEL_DIR:/models" \
      "$WORKER_IMAGE" proteus_realtime.download seed --source /source "$@"
    ;;
  sample)
    mkdir -p "$ROOT/inputs"
    if [[ -f "$ROOT/inputs/test-dog.jpg" ]]; then
      echo "Already present: $ROOT/inputs/test-dog.jpg"
    elif [[ -f "$(get INPUT_IMAGE)" ]]; then
      cp -- "$(get INPUT_IMAGE)" "$ROOT/inputs/test-dog.jpg"
    else
      image_required
      docker run --rm --pull=never --user "$(id -u):$(id -g)" -v "$ROOT/inputs:/inputs" \
        "$WORKER_IMAGE" proteus_realtime.download sample
    fi
    ;;
  up)
    image_required
    python3 -m proteus_realtime.deployment idle
    python3 -m proteus_realtime.deployment gen
    "${COMPOSE[@]}" up -d
    echo "Waiting for GPU worker. On a fresh setup, run register before policy requests."
    for ((attempt=1; attempt<=90; attempt++)); do
      status="$(docker inspect --format '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER_NAME" 2>/dev/null || true)"
      if [[ "$status" == 'running|healthy' ]]; then client state ""; exit 0; fi
      if [[ "$status" == exited* || "$status" == dead* ]]; then
        docker logs --tail 100 "$CONTAINER_NAME"; exit 1
      fi
      sleep 2
    done
    echo "ERROR: readiness timeout. Inspect logs." >&2
    docker logs --tail 100 "$CONTAINER_NAME"
    exit 1
    ;;
  state|reset|mode|evaluate) client "$command" "" "$@" ;;
  query)
    [[ $# -ge 3 ]] || { usage; exit 1; }
    model="$1"; image="$2"; slo="$3"; shift 3
    client query "$image" --model "$model" --slo "$slo" "$@"
    ;;
  test|bench|policy-test)
    image="$ROOT/inputs/test-dog.jpg"
    if [[ $# -gt 0 && "$1" != -* ]]; then image="$1"; shift; fi
    client "$command" "$image" --model "$APPLICATION" --slo "$SLO_MS" "$@"
    ;;
  register|profile)
    image="$ROOT/inputs/test-dog.jpg"
    trials="$(get PROFILE_TRIALS)"
    models=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --models)
          shift
          while [[ $# -gt 0 && "$1" != -* ]]; do models+=("$1"); shift; done
          [[ ${#models[@]} -gt 0 ]] || { usage; exit 1; } ;;
        --image) [[ $# -ge 2 ]] || { usage; exit 1; }; image="$2"; shift 2 ;;
        --trials) [[ $# -ge 2 ]] || { usage; exit 1; }; trials="$2"; shift 2 ;;
        *) echo "Unknown profile option: $1" >&2; exit 1 ;;
      esac
    done
    model_args=()
    if [[ ${#models[@]} -gt 0 ]]; then model_args=(--models "${models[@]}"); fi
    client profile "$image" --trials "$trials" "${model_args[@]}"
    ;;
  logs) docker logs --tail 100 "$@" "$CONTAINER_NAME" ;;
  ps)
    [[ -f "$HERE/generated/docker-compose.yml" ]] || { echo "Run gen/up first."; exit 1; }
    "${COMPOSE[@]}" ps ;;
  down)
    [[ -f "$HERE/generated/docker-compose.yml" ]] || { echo "Nothing generated."; exit 0; }
    "${COMPOSE[@]}" down
    echo "Proteus stopped; standalone weights, profiles and results retained."
    ;;
  *) usage; exit 1 ;;
esac
