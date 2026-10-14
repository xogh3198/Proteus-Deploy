# Proteus · GPU 기준선 설정 및 실행

이 문서는 **실제 GPU 추론**용입니다. 저장소 루트에서 실행합니다.

## 1. 코드와 Docker 확인

```bash
git clone --branch docker/v2 --single-branch https://github.com/xogh3198/Proteus-Deploy.git Proteus
cd Proteus
docker info --format '{{.DockerRootDir}}'
nvidia-smi
```

이미 받은 폴더라면 `git status --short`로 개인 변경을 확인한 뒤 해당 브랜치를 업데이트하세요.
개인 Docker는 현재 터미널에서 선택합니다. 스크립트는 Docker 저장 위치를 바꾸지 않습니다.
같은 GPU에서는 실험을 하나씩 실행하세요.

## 2. 실행할 종류 선택

| 종류 | 명령 | 개인 설정 파일 | 포트 |
|---|---|---|---|
| Proteus | `bash deploy/docker/proteus-docker.sh` | `deploy/docker/proteus-docker.local.env` | 18081 |
| INFaaS-Accuracy | `bash deploy/docker/baseline-docker.sh infaas-accuracy` | `deploy/docker/baselines/baseline.local.env` | 19081 |
| Clipper-HA | `bash deploy/docker/baseline-docker.sh clipper-ha` | 같은 파일 | 19082 |
| Clipper-HT | `bash deploy/docker/baseline-docker.sh clipper-ht` | 같은 파일 | 19083 |

기준선 3개는 별도 컨테이너·가중치 캐시·프로파일·결과를 사용합니다.
이들은 **Proteus 내부 정책의 GPU 포트**이며, 별도 INFaaS/원본 Clipper 배포와 구분합니다.

## 3. 개인 설정 만들기

Proteus:

```bash
cp -n deploy/docker/proteus-docker.env deploy/docker/proteus-docker.local.env
vim deploy/docker/proteus-docker.local.env
```

기준선 3개:

```bash
cp -n deploy/docker/baselines/baseline.env deploy/docker/baselines/baseline.local.env
vim deploy/docker/baselines/baseline.local.env
```

필요한 항목만 수정합니다. 주석은 별도 줄에 작성하세요.

```dotenv
GPU=2
GPU_TYPE=v100
MODEL_NAME=resnet-50
MODEL_ID=microsoft/resnet-50
MODEL_REVISION=34c2154c194f829b11125337b98c8f5f9965ff19
SLO_MS=200
BATCH_SIZES=1,2,4,8
CPU_THREADS=2
WARMUP_QUERIES=10
PROFILE_TRIALS=30
INPUT_IMAGE=inputs/test-dog.jpg
```

- `MODEL_NAME`: 실험에서 부르는 이름. `MODEL_ID`: Hugging Face 체크포인트.
- `MODEL_REVISION`: 해당 모델의 40자리 commit. 위 두 항목을 모두 비우면 기존 `MODEL_REGISTRY`를 사용합니다.
- env 모델 지정은 **한 번에 한 모델 교체**입니다. HF 이미지 분류 모델만 지원합니다.
- `BATCH_SIZES`: 프로파일링할 후보. 1을 포함하는 오름차순 2의 거듭제곱, 최대 64.
  실제 배치는 정책이 선택하므로 `1,2,4,8`을 적었다고 항상 8개씩 실행되지 않습니다.

ResNet-18로 교체할 때는 모델 관련 3개 항목만 바꿉니다.

```dotenv
MODEL_NAME=resnet-18
MODEL_ID=microsoft/resnet-18
MODEL_REVISION=65a5785d9156231087c481e0c7dd33a5ff6f7e3e
```

저장 경로 변경은 선택 사항입니다. Proteus는 `MODEL_DIR`·`DATA_DIR`,
기준선은 `DATA_ROOT`를 사용합니다. 기준선 경로 밑에는 기법별 디렉터리가 자동으로 생깁니다.
개인 `.local.env`는 Git에 올라가지 않습니다.

## 4. Proteus 최초 실행

각 명령이 성공한 뒤 다음 단계로 진행하세요. 실패하면 해당 워커 로그를 확인합니다.

```bash
bash deploy/docker/proteus-docker.sh settings
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh download
bash deploy/docker/proteus-docker.sh sample
bash deploy/docker/proteus-docker.sh doctor
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register
bash deploy/docker/proteus-docker.sh state
bash deploy/docker/proteus-docker.sh test --warm-requests 10
bash deploy/docker/proteus-docker.sh bench --rate 5 --duration 30 --distribution uniform --seed 10
bash deploy/docker/proteus-docker.sh logs
bash deploy/docker/proteus-docker.sh down
```

`register`는 설정의 입력·배치 후보·반복 횟수로 실제 GPU 시간을 측정합니다. 학습이 아닙니다.
`state`의 `policy_ready: true`와 `plan.solver_status: optimal`을 확인하세요.

## 5. 기준선 최초 실행

`BASELINE`만 바꿔 동일한 순서로 실행합니다. 이미지 빌드는 3개 중 한 번만 하면 됩니다.
`INPUT_IMAGE`에 지정한 **기존 JPEG를 먼저 준비**하세요. 기준선 `sample`은 다운로드하지 않습니다.

```bash
BASELINE=infaas-accuracy
# 다른 실행: BASELINE=clipper-ha 또는 BASELINE=clipper-ht
bash deploy/docker/baseline-docker.sh "$BASELINE" settings
bash deploy/docker/baseline-docker.sh "$BASELINE" build
bash deploy/docker/baseline-docker.sh "$BASELINE" download
bash deploy/docker/baseline-docker.sh "$BASELINE" sample
bash deploy/docker/baseline-docker.sh "$BASELINE" doctor
bash deploy/docker/baseline-docker.sh "$BASELINE" up
bash deploy/docker/baseline-docker.sh "$BASELINE" register
bash deploy/docker/baseline-docker.sh "$BASELINE" state
bash deploy/docker/baseline-docker.sh "$BASELINE" test --warm-requests 10
bash deploy/docker/baseline-docker.sh "$BASELINE" bench --rate 5 --duration 30 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh "$BASELINE" logs
bash deploy/docker/baseline-docker.sh "$BASELINE" down
```

`state`의 `mode: native`, `policy_ready: true`를 확인하세요.
3개 모두 준비하고 종료한 뒤 순차 부하 비교도 가능합니다.

```bash
bash deploy/docker/baseline-docker.sh all compare --rates 1,5,10 --duration 30 --distribution uniform --seed 10
```

이 명령은 **기준선 3개만** 비교합니다. Proteus/별도 INFaaS/원본 Clipper는 포함하지 않습니다.

## 6. 설정 변경 후 재실험

**현재 실행 중인 설정으로 `down` → local.env 수정 → `settings` → 필요한 준비 → `up`** 순서입니다.
저장 경로·컨테이너 이름을 바꾸기 전에 반드시 기존 실행을 종료하세요.

| 변경 | 추가로 할 일 |
|---|---|
| 모델 | `download` → `up` → `register` |
| GPU·SLO·배치 후보·입력·CPU 스레드·warmup | `up` → `register` |
| QPS·실험 시간·분포 | `bench`의 옵션만 변경 |
| 코드·Dockerfile·라이브러리 버전 | `build` → `up` → `register` |

모델·GPU·SLO·배치 후보는 **이미지 재빌드가 필요 없습니다**. 이번 코드 업데이트 후에는 최초 한 번 재빌드하세요.
터미널 변수는 local.env보다 우선합니다. Proteus는 `GPU` 등 일반 변수,
기준선은 `BASELINE_GPU`·`BASELINE_MODEL_NAME` 등 `BASELINE_` 변수를 사용합니다.
설정을 바꿨는데 적용되지 않으면 `settings`와 기존 `export` 값을 확인하세요.

## 7. 결과 확인

- Proteus: `DATA_DIR/results/`, `DATA_DIR/profiles/`.
- 기준선: `DATA_ROOT/<기법>/results/`, `profiles/`, `logs/`.
- `test`는 **고정 배치 1 cold/warm 진단**, `bench`는 **실제 정책 부하 실험**입니다.
- `OK`는 추론 성공입니다. SLO 충족률과 실제 배치 크기는 결과 CSV/JSON으로 확인하세요.
- 모델 하나/이미지 반복은 정확도 평가가 아닙니다. 다중 모델 Proteus 설정은 [GPU_SERVING.md](GPU_SERVING.md),
  기준선 구현 범위는 [GPU_BASELINES.md](GPU_BASELINES.md)를 참고하세요.
