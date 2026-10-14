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
container_user() { python3 -m proteus_realtime.deployment user; }
run_container() {
  local selected_user
  selected_user="$(container_user)"
  docker run --user "$selected_user" "$@"
}
WORKER_IMAGE="$(get WORKER_IMAGE)"
MODEL_DIR="$(get MODEL_DIR)"
MODEL_REGISTRY="$(get MODEL_REGISTRY)"
MODEL_ENV=(-e "MODEL_NAME=$(get MODEL_NAME)" -e "MODEL_ID=$(get MODEL_ID)" \
  -e "MODEL_REVISION=$(get MODEL_REVISION)" -e "GPU_TYPE=$(get GPU_TYPE)" \
  -e "CPU_THREADS=$(get CPU_THREADS)" -e "WARMUP_QUERIES=$(get WARMUP_QUERIES)")
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
  run_container --rm --pull=never --network "$NETWORK_NAME" \
    "${mounts[@]}" "$WORKER_IMAGE" proteus_realtime.client "$command" \
    --controller proteus-worker:9000 --output /results "${image_args[@]}" "$@"
}

usage() {
  printf '%s\n' \
    "Proteus 독립 GPU 실행: MILP + 적응형 배치 (단일 GPU/애플리케이션)." \
    "기본 순서: gpus → settings → build → download → sample → doctor → up → register → query → down" \
    "설정: deploy/docker/proteus-docker.local.env / 안내: docker.md" \
    "현재 터미널의 Docker를 사용합니다. context 변경, sudo, 공용 설정 변경을 하지 않습니다." \
    "  $0 gpus | settings                           # GPU 목록 / 적용 설정" \
    "  $0 build | doctor                            # 이미지 빌드 / CUDA·Gurobi 확인" \
    "  $0 download [--models resnet-50] | sample     # 가중치 / 테스트 JPEG 준비" \
    "  $0 seed --from /existing/HF_HOME [--models resnet-50]  # 가중치 복사" \
    "  $0 up | state [--watch 2] | logs [--follow] | ps | down" \
    "  $0 register [--models resnet-50] [--image IMAGE] [--trials 30]  # GPU 프로파일" \
    "  $0 query imagenet IMAGE 200 [-n 11]          # gRPC 반복 요청" \
    "  $0 test [IMAGE] [--warm-requests 10]         # 고정 배치 1 cold/warm 진단 후 정책 복원" \
    "  $0 policy-test [IMAGE]                      # 실제 배치 > 1과 MILP 연결 검사" \
    "  $0 bench [IMAGE] --rate 10 --duration 30 --distribution uniform" \
    "  $0 evaluate --manifest manifest.json        # EVAL_DIR 안의 라벨 검증셋" \
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
    run_container --rm --pull=never --gpus "device=$GPU" \
      "${CONFIG_MOUNT[@]}" "${MODEL_ENV[@]}" "${LICENSE_MOUNTS[@]}" -v "$MODEL_DIR:/models:ro" \
      "$WORKER_IMAGE" proteus_realtime.doctor
    ;;
  download)
    image_required
    mkdir -p "$MODEL_DIR"
    run_container --rm --pull=never \
      "${CONFIG_MOUNT[@]}" "${MODEL_ENV[@]}" -v "$MODEL_DIR:/models" \
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
    run_container --rm --pull=never \
      "${CONFIG_MOUNT[@]}" "${MODEL_ENV[@]}" -v "$source_dir:/source:ro" -v "$MODEL_DIR:/models" \
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
      run_container --rm --pull=never -v "$ROOT/inputs:/inputs" \
        "$WORKER_IMAGE" proteus_realtime.download sample
    fi
    ;;
  up)
    image_required
    python3 -m proteus_realtime.deployment idle
    python3 -m proteus_realtime.deployment gen
    "${COMPOSE[@]}" up -d
    echo "GPU 워커 준비를 기다립니다. 처음 실행할 때는 요청 전에 register로 프로파일링하세요."
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
    image="$(get INPUT_IMAGE)"
    if [[ $# -gt 0 && "$1" != -* ]]; then image="$1"; shift; fi
    client "$command" "$image" --model "$APPLICATION" --slo "$SLO_MS" "$@"
    ;;
  register|profile)
    image="$(get INPUT_IMAGE)"
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
    echo "Proteus 컨테이너를 종료했습니다. 가중치·프로파일·결과는 보존합니다."
    ;;
  *) usage; exit 1 ;;
esac
