# Sommelier 비교 정책 — 실제 GPU 실행

Proteus 논문의 Sommelier 기준선을 **단일 GPU·ImageNet 애플리케이션**으로 축소한 포트입니다.
Yale Sommelier의 모델 저장소·semantic index 전체를 구현한 것이 아닙니다.

원본 `run.py`의 `Ilp(static='spec_acc')`는 장치별 애플리케이션 배치를 고정하고
그 애플리케이션의 모델 변형 선택을 허용합니다. GPU·애플리케이션이 각각 하나이면
이 배치 제약은 자동으로 만족됩니다. 따라서 같은 프로파일·정확도·제어 설정에서는
**이 Sommelier 포트와 Proteus 포트의 선택/배치 정책이 동일하게 축소됩니다.**
재배치 성능 차이를 보여 주는 실험이라고 해석하지 마세요.

원본과 공통 실행 코드는 수정하지 않고, 새 컨테이너 안에서 공통 GPU 실행 클래스를 사용합니다.
기존 Proteus/INFaaS/Clipper 서비스에 요청하거나 모델 메모리를 공유하지 않습니다.
참고: [Proteus §6.1.1 및 §6.5](https://guanh01.github.io/files/2024proteus.pdf#page=8).

## 1. 준비 및 개인 설정

`docker/v2` 저장소 루트에서 실행합니다. GPU가 가능한 Docker를 현재 터미널에서 선택하고
본인의 다른 GPU 실험은 먼저 종료하세요. 스크립트가 다른 프로젝트를 종료하지 않습니다.

```bash
cp -n deploy/docker/baselines/sommelier.env deploy/docker/baselines/sommelier.local.env
vim deploy/docker/baselines/sommelier.local.env
```

주요 기본값:

```dotenv
GPU=2
GPU_TYPE=v100
MODEL_NAME=resnet-50
MODEL_ID=
MODEL_REVISION=
MODEL_REGISTRY=deploy/docker/baselines/sommelier-models.json
SLO_MS=200
BATCH_SIZES=1,2,4,8
CPU_THREADS=2
WARMUP_QUERIES=10
PROFILE_TRIALS=30
ALLOCATION_INTERVAL=10
DATA_ROOT=data/proteus-baselines
EVAL_DIR=data/proteus-baselines/sommelier/eval
INPUT_IMAGE=inputs/test-dog.jpg
```

- 기본값은 **ResNet-50만 활성화**되어 기존 cold/warm 실험처럼 실행할 수 있습니다.
- 포트 `19084`, 이미지 `proteus-baseline-worker:sommelier-v1`, 컨테이너 `proteus-baseline-sommelier`.
- 가중치·프로파일·결과는 `DATA_ROOT/sommelier/` 아래에서 다른 실험과 분리됩니다.
- 4TB에 저장하려면 예를 들어 `DATA_ROOT=/data4tb/taeho-proteus/baselines`,
  `EVAL_DIR=/data4tb/taeho-proteus/baselines/sommelier/eval`로 지정합니다.
- `SOMMELIER_*` 터미널 변수는 `BASELINE_*`보다 우선합니다. `SOMMELIER_ENV`는 다른 개인 설정 파일을 선택합니다.
  기존 export가 남아 있으면 `settings` 출력으로 실제 적용 값을 확인하세요.
- 별도 Gurobi 라이선스가 필요한 경우 `GUROBI_LICENSE_FILE`에 유효한 파일 경로를 지정합니다.
  라이선스는 읽기 전용 마운트되며 Git/이미지에는 넣지 않습니다.

## 2. 기본 ResNet-50 GPU 실행 확인

기존 실험과 같은 JPEG를 `INPUT_IMAGE` 경로에 먼저 준비하세요.
`sample`은 기존 파일을 확인할 뿐 다운로드/덮어쓰기를 하지 않습니다.
각 명령이 성공한 뒤 다음 단계로 진행합니다.

```bash
bash deploy/docker/baseline-docker.sh sommelier settings
bash deploy/docker/baseline-docker.sh sommelier build
bash deploy/docker/baseline-docker.sh sommelier download
bash deploy/docker/baseline-docker.sh sommelier sample
bash deploy/docker/baseline-docker.sh sommelier doctor
bash deploy/docker/baseline-docker.sh sommelier up
bash deploy/docker/baseline-docker.sh sommelier register --trials 30
bash deploy/docker/baseline-docker.sh sommelier state
bash deploy/docker/baseline-docker.sh sommelier test --warm-requests 10
bash deploy/docker/baseline-docker.sh sommelier bench --rate 5 --duration 30 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh sommelier logs
bash deploy/docker/baseline-docker.sh sommelier down
```

`state`의 `baseline: sommelier`, `mode: native`, `policy_ready: true`,
`placement.fixed: true`, `plan.solver_status: optimal`을 확인하세요.
단일 모델에서는 `model_scaling_enabled: false`가 정상입니다.
정규화 정확도 100은 상대값이며 실제 Top-1 100%가 아닙니다.

`register`는 실제 GPU로 모든 활성 모델의 배치 후보를 측정합니다.
`test`는 모델을 내린 뒤 **기본 MODEL_NAME의 고정 배치 1 cold/warm**을 측정하고 native 정책을 복원합니다.
`bench`는 모델 선택·적응형 배치가 활성화된 실제 요청률 실험입니다.
모델 로딩·GPU 이동·warmup 비용은 실제 요청에 발생한 경우 지연에 포함됩니다. 오버랩은 구현하지 않았습니다.

## 3. ResNet-18/50 모델 교체 실험

실행 중이라면 기존 설정으로 `sommelier down`부터 수행합니다.

```bash
cp -n deploy/docker/baselines/sommelier-models.json deploy/docker/baselines/sommelier-models.local.json
vim deploy/docker/baselines/sommelier-models.local.json
vim deploy/docker/baselines/sommelier.local.env
```

1. 모델 JSON에서 `resnet-18`의 `enabled`를 `true`로 바꾸고 두 모델의 revision은 유지합니다.
2. 개인 env의 `MODEL_REGISTRY`를 `deploy/docker/baselines/sommelier-models.local.json`으로 바꿉니다.
3. `MODEL_ID`·`MODEL_REVISION`은 **빈 값**으로 둡니다. 값을 넣으면 단일 모델로 override됩니다.
4. 아래 두 방법 중 하나로 **같은 검증셋·같은 Top-1 기준의 사전 정확도**를 준비합니다.

### 방법 A: 출처가 있는 사전 측정값 입력

각 모델의 JSON 항목에 `accuracy`(0보다 크고 100 이하인 실제 백분율),
`accuracy_source`(측정/보고서 출처), `accuracy_dataset`(동일 검증셋 버전과 Top-1 지표 식별자)을 입력합니다.
다른 데이터셋의 수치, Top-5 또는 출력 confidence를 혼합하지 마세요.
기본 JSON의 `null`은 미측정입니다. 실행을 위해 임의 숫자를 넣지 않습니다.

```bash
bash deploy/docker/baseline-docker.sh sommelier settings
bash deploy/docker/baseline-docker.sh sommelier download
bash deploy/docker/baseline-docker.sh sommelier doctor
bash deploy/docker/baseline-docker.sh sommelier up
bash deploy/docker/baseline-docker.sh sommelier register --trials 30
bash deploy/docker/baseline-docker.sh sommelier state
```

### 방법 B: 동일 라벨 검증셋으로 직접 측정

`EVAL_DIR` 아래에 실제 검증 이미지와 `manifest.json`을 준비합니다. 예시 **형식**:

```json
[
  {"path": "images/validation-example.jpg", "label": 0}
]
```

이미지 경로·정답 label은 실제 검증셋 값으로 바꾸세요. label은 해당 HF 모델의 ImageNet class ID `0..999`이며
ILSVRC의 1-based 라벨을 그대로 넣으면 안 됩니다. 평가 전에 캐시의 모델 config를 읽어
두 모델의 class-ID 매핑이 동일한지 검사하며, 다른 매핑이면 중단합니다.
검증셋은 사용자가 준비하며 자동 다운로드/학습하지 않습니다. 한 장짜리 예시나 test-dog.jpg를
전체 ImageNet 정확도 측정으로 사용하지 마세요.

```bash
bash deploy/docker/baseline-docker.sh sommelier download
bash deploy/docker/baseline-docker.sh sommelier doctor
bash deploy/docker/baseline-docker.sh sommelier up
bash deploy/docker/baseline-docker.sh sommelier evaluate --manifest manifest.json --timeout 7200
bash deploy/docker/baseline-docker.sh sommelier register --trials 30
bash deploy/docker/baseline-docker.sh sommelier state
```

평가는 **사전 측정** 단계이며 부하 실험과 동시에 실행하지 않습니다.
활성 모델 모두를 실제 GPU에서 평가하고 correct/total, Top-1, 체크포인트, 검증셋 해시를 저장합니다.
이미지 경로가 EVAL_DIR 밖으로 나가거나 측정 중 파일이 바뀌면 중단합니다.
정확도나 프로파일이 부족하면 `policy_ready: false`와 이유가 표시되고 native 요청을 거절합니다.

## 4. 부하·배치 실험 및 결과

`policy_ready: true`, `model_scaling_enabled: true`를 확인한 뒤 실행합니다.

```bash
bash deploy/docker/baseline-docker.sh sommelier bench --rate 1 --duration 30 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh sommelier bench --rate 5 --duration 30 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh sommelier bench --rate 10 --duration 30 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh sommelier state
bash deploy/docker/baseline-docker.sh sommelier down
```

모델 교체는 실측 처리 용량을 넘어서는 수요에서 발생할 수 있습니다. 위 QPS가 낮으면 교체가 없을 수 있으므로
`register`의 처리 용량과 `plan`을 보고 부하를 정하세요. 인위적인 프로파일/정확도 값을 넣어 교체를 만들지 않습니다.
Poisson/Gamma는 `--distribution poisson`, `--distribution gamma --shape 0.05`로 지정합니다.
동일 초기 상태를 비교하려면 각 실험을 `down/up`하고 제어 간격·EWMA·초기 QPS까지 맞추세요.

- `DATA_ROOT/sommelier/results/`: 요청별 CSV, cold/warm, bench JSON.
- `profiles/profiles.json`: 모든 모델의 GPU 프로파일과 사전 평가 결과.
- `state`: 고정 장치, 선택 모델/배치/처리 용량, 모델별 성공 요청 수, 실제 모델 변경 횟수.
- bench CSV의 `variant`, `batch_size`, `load_ms`, `slo_met`는 실제 요청 결과입니다.
- bench JSON의 `effective_offline_top1_pct`는 **성공 요청 수로 가중한 사전 모델 정확도**입니다.
  현재 JPEG 요청의 정답률을 뜻하지 않으며 미측정이면 null입니다.
- `OK`는 실제 추론 성공이지 SLO 만족이 아닙니다. 실패/거절/skip과 SLO goodput도 함께 확인하세요.

기존 `all compare`는 INFaaS-Accuracy/Clipper-HA/HT **3개만** 유지합니다. Sommelier는 위처럼 명시적으로 실행합니다.
단일 모델 기준선 3개와 다중 모델 Sommelier를 동일한 모델 선택 실험으로 비교하지 마세요.
코드를 업데이트한 최초 한 번은 build, 이후 모델/배치/GPU 변경은 down → 설정 수정 → download/up/register 순서입니다.
