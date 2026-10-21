# Proteus Docker 실행 가이드

이 문서는 Proteus 전용입니다. 모델·배치 설정과 INFaaS-Accuracy/Clipper-HA/HT 실행 구분은
**[setting_readme.md](setting_readme.md)**를 참고하세요.

GPU와 모델을 설정하고, 하나의 스크립트로 **빌드 → 가중치 다운로드 → 실행 → 프로파일링 → 요청 → 종료**합니다.
기본값은 **V100 1대(GPU 2번), ResNet-50, FP32, SLO 200ms**입니다.
사전 학습 가중치로 실제 GPU 추론을 수행하며, 모델을 새로 학습하지 않습니다.

> 이 브랜치는 단일 GPU용 독립 실행 포트입니다. MILP와 적응형 배치를 연결했지만 논문의 전체 클러스터 재현은 아닙니다.
> 원본 시뮬레이터 안내는 [SIMULATOR_DOCKER.md](SIMULATOR_DOCKER.md), 구현 범위와 상세 실험은 [GPU_SERVING.md](GPU_SERVING.md)를 참고하세요.

## 기본 설정으로 빠르게 실행

아래는 Docker에서 GPU 사용이 준비된 서버의 **저장소 루트**에서 실행합니다.
GPU·모델·저장 경로를 바꾸려면 먼저 1절의 로컬 설정을 수정하세요.

```bash
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh download --models resnet-50
bash deploy/docker/proteus-docker.sh sample
bash deploy/docker/proteus-docker.sh doctor
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-50 --trials 30
bash deploy/docker/proteus-docker.sh state
bash deploy/docker/proteus-docker.sh query imagenet inputs/test-dog.jpg 200 -n 11
bash deploy/docker/proteus-docker.sh test inputs/test-dog.jpg --warm-requests 10
bash deploy/docker/proteus-docker.sh policy-test inputs/test-dog.jpg
bash deploy/docker/proteus-docker.sh logs
bash deploy/docker/proteus-docker.sh down
```

한 단계가 실패하면 다음 명령을 실행하지 말고 오류를 확인하세요.

## 실행 전 준비

서버에 NVIDIA 드라이버, Docker, Docker Compose 플러그인, GPU 사용이 가능한 NVIDIA Container Toolkit,
Bash와 Python 3가 필요합니다. PyTorch 등 추론 라이브러리는 이미지 안에 설치하므로
호스트에 원본 시뮬레이터의 `requirements.txt`를 설치할 필요는 없습니다.

새로 받는 경우:

```bash
git clone --branch docker/v4 --single-branch https://github.com/xogh3198/Proteus-Deploy.git Proteus
cd Proteus
```

이미 받은 경우에는 기존 Proteus 폴더에서 아래 단계를 실행하세요.
개인 Docker를 선택한 터미널이면 그대로 사용합니다. 스크립트는 Docker context를 바꾸거나 `sudo`를 사용하지 않습니다.

```bash
docker info --format '{{.DockerRootDir}}'
```

현재 개인 저장소 설정에서는 `/data4tb/taeho-proteus/docker-data`가 나와야 합니다.
이 경로는 **Docker 이미지·컨테이너·빌드 캐시**의 위치입니다.
모델 가중치와 실험 결과는 아래 `MODEL_DIR`·`DATA_DIR`에 별도로 저장됩니다.
저장 위치 확인만으로 rootless의 GPU 연결까지 확인된 것은 아니며, `doctor`가 성공해야 합니다.

각 단계가 실패하면 다음 단계로 넘어가지 말고 오류를 확인하세요.

## 1. GPU와 모델 설정

```bash
bash deploy/docker/proteus-docker.sh gpus
cp -n deploy/docker/proteus-docker.env deploy/docker/proteus-docker.local.env
vim deploy/docker/proteus-docker.local.env
```

`cp -n`은 기존 로컬 설정을 덮어쓰지 않습니다. 주로 아래 항목을 설정하면 됩니다.

```dotenv
GPU=2                            # 호스트 nvidia-smi의 GPU 번호 한 개
GPU_TYPE=v100                    # 장치 유형
APPLICATION=imagenet             # 요청할 애플리케이션 이름
MODEL_NAME=resnet-50              # 기본 모델
SLO_MS=200                       # 지연 목표, ms
MODEL_DIR=data/proteus/models    # Proteus 전용 가중치 캐시
DATA_DIR=data/proteus            # 프로파일과 실험 결과
PORT=18081                      # 호스트 gRPC 포트
```

가중치·결과까지 4TB 디스크에 저장하려면 **선택적으로** 경로만 바꿉니다.
코드 폴더를 옮기거나 Docker 설정을 스크립트에 넣을 필요는 없습니다.

```dotenv
MODEL_DIR=/data4tb/taeho-proteus/models
DATA_DIR=/data4tb/taeho-proteus/experiments
```

```bash
bash deploy/docker/proteus-docker.sh settings
nvidia-smi -i 2
ss -ltn 'sport = :18081'
```

GPU와 포트를 다른 실험이 사용 중이면 본인의 기존 실험을 먼저 종료하거나 설정을 바꾸세요.
다른 사용자의 프로세스를 종료하지 않습니다. GPU 번호를 바꾼 경우 이후 `nvidia-smi` 명령도 맞춥니다.
설정은 **기본 env → local.env → 현재 터미널 환경변수** 순으로 우선 적용됩니다.

## 2. 이미지 빌드와 가중치 다운로드

```bash
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh download --models resnet-50
bash deploy/docker/proteus-docker.sh sample
bash deploy/docker/proteus-docker.sh doctor
```

| 명령 | 하는 일 |
|---|---|
| `build` | Python·PyTorch/CUDA·Transformers·Gurobi·gRPC·Proteus 코드가 있는 `proteus-worker:0.2.0` 생성 |
| `download` | 사전 학습 가중치와 모델·전처리 설정을 `MODEL_DIR`에 다운로드 |
| `sample` | `inputs/test-dog.jpg` 준비. 기존 파일이 있으면 그대로 사용 |
| `doctor` | 실제 CUDA 연산과 Gurobi 동작·라이선스 확인. 모델 추론은 다음 단계에서 확인 |

**Docker 실행 이미지와 추론 입력 JPEG는 서로 다릅니다.** 가중치는 실행 이미지에 포함되지 않으며,
학습 데이터셋도 자동으로 제공하지 않습니다. INFaaS 이미지·컨테이너를 사용하지 않습니다.
비교 실험에는 INFaaS에서 사용한 것과 같은 JPEG 파일을 준비하세요.

별도 Gurobi 라이선스가 필요하면 local.env의 `GUROBI_LICENSE_FILE`에 유효한 파일의 절대 경로를 입력합니다.
라이선스는 읽기 전용으로 연결하며 Git이나 이미지에 넣지 않습니다.

## 3. 실행과 모델 프로파일링

```bash
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-50 --trials 30
bash deploy/docker/proteus-docker.sh state
```

`up`은 선택한 GPU를 연결한 독립 워커를 띄웁니다. 기본 주소는 `127.0.0.1:18081`입니다.
컨테이너 안에서는 선택한 GPU 한 장이 0번으로 보입니다.

**처음에는 `register`가 필요합니다.** 배치 1/2/4/8의 실제 처리 시간을 측정해
MILP·적응형 배치가 사용할 프로파일을 만들고 모델을 사전 로딩합니다. 학습이 아닙니다.
호환되는 기존 프로파일은 재기동 시 재사용하지만 GPU·런타임·모델이 달라지면 다시 측정해야 합니다.

`state`에서 `mode: proteus`, `policy_ready: true`, `milp_enabled: true`,
`batching: accscale`, `plan.solver_status: optimal`을 확인하세요.
`up`의 healthy 상태만으로 MILP 입력 준비까지 완료된 것은 아닙니다.

## 4. 요청 보내기와 테스트

### 4.1 gRPC 요청 보내기

```bash
bash deploy/docker/proteus-docker.sh query imagenet inputs/test-dog.jpg 200 -n 11
```

같은 JPEG를 순차적으로 11회 요청합니다. 스크립트가 GPU를 연결하지 않은 임시 클라이언트
컨테이너에서 gRPC를 사용하므로, 호스트에 `grpcio`를 별도로 설치하지 않아도 됩니다.
`imagenet`은 애플리케이션 이름이고 서버가 실행 모델을 선택합니다. 요청 SLO는 설정의 `SLO_MS`와 맞춥니다.
Proteus는 INFaaS의 `50052`·`8081`이 아니라 자체 `18081` 포트를 사용합니다.

| 출력 | 의미 |
|---|---|
| `e2e` | 클라이언트에서 측정한 요청 왕복 시간 |
| `load` | 모델 준비 시간 |
| `queue` | 서버에서 요청이 기다린 시간. 배치를 모으는 대기 포함 |
| `variant` / `batch` | 실제 실행한 모델 / 배치 크기 |
| `SLO=MET` / `MISSED` | e2e가 목표 지연을 충족했는지 |

`OK`는 추론 성공이지 SLO 만족을 뜻하지 않습니다.
`register`가 모델을 미리 올리므로 이 명령의 첫 요청이 반드시 모델 로딩 요청인 것은 아닙니다.

### 4.2 INFaaS처럼 첫 요청과 반복 요청 비교

```bash
bash deploy/docker/proteus-docker.sh test inputs/test-dog.jpg --warm-requests 10
```

모델을 GPU에서 내린 뒤 **첫 요청 1회 + 반복 요청 10회**를 측정합니다.
이 테스트만 고정 배치 1을 사용하고 끝나면 원래 정책 모드로 복원합니다.
OS 파일 캐시·CUDA 초기화까지 초기화하는 테스트는 아니며, 적응형 배치 성능 결과와 구분하세요.

### 4.3 MILP와 실제 배치 연결 확인

```bash
bash deploy/docker/proteus-docker.sh policy-test inputs/test-dog.jpg
```

동시 요청으로 실제 배치 크기 1 초과와 MILP 연결을 확인합니다.
`POLICY_TEST=PASS`는 연결 확인이며 모든 요청의 SLO 만족을 보장하지 않습니다.
실측 프로파일에서 배치 1만 허용되면 이 테스트는 통과할 수 없습니다.

### 4.4 선택 사항으로 부하 테스트

```bash
bash deploy/docker/proteus-docker.sh bench inputs/test-dog.jpg --rate 10 --duration 30 --distribution uniform
```

평균 10 QPS로 30초간 요청하고 지연·처리량·SLO 미스를 저장합니다.
Poisson/Gamma 부하와 여러 모델의 정확도 평가는 [상세 안내](GPU_SERVING.md)를 참고하세요.

## 5. 상태 조회와 로그 및 종료

```bash
bash deploy/docker/proteus-docker.sh state --watch 2
# 확인 후 Ctrl+C
bash deploy/docker/proteus-docker.sh logs
bash deploy/docker/proteus-docker.sh logs --follow
# 확인 후 Ctrl+C
bash deploy/docker/proteus-docker.sh down
nvidia-smi -i 2
```

`Ctrl+C`는 상태·로그 조회만 끝내고 워커를 종료하지 않습니다.
**GPU 메모리를 반환하려면 `down`을 실행하세요.** Proteus 컨테이너·네트워크만 종료하고
가중치·프로파일·결과는 보존합니다.

| 기본 저장 위치 | 내용 |
|---|---|
| `data/proteus/models/` | 가중치·모델 설정 |
| `data/proteus/profiles/` | GPU 배치별 실측 프로파일 |
| `data/proteus/results/` | 요청·cold/warm·부하 테스트 CSV/JSON |
| `deploy/docker/generated/` | 생성된 Compose와 적용 설정 |

## 실행이 막힐 때와 실험 범위

- `policy_ready: false`: 먼저 `register` 출력과 워커 로그를 확인하세요. 여러 모델이면 동일 검증셋의 사전 정확도도 필요합니다.
- `permission denied`: 해당 경로의 소유권과 선택한 Docker를 확인하세요. 스크립트는 rootless/rootful에 맞춰 컨테이너 사용자를 정하지만 Docker context를 강제하지 않습니다.
- rootless에서 NVIDIA 오류: 서버의 rootless GPU 연결 설정이 필요합니다. 공용 `/etc/docker`·NVIDIA 설정을 임의로 바꾸지 말고, `doctor` 오류부터 확인하세요.
- `no space left on device`: DockerRootDir와 `MODEL_DIR`·`DATA_DIR` 각각의 여유 공간을 확인하세요. 공용 `prune`은 하지 않습니다.
- 모델 하나의 정규화 정확도 100은 **실제 Top-1 100%가 아닙니다**. 단일 JPEG 반복도 정확도 평가가 아닙니다.
- 원본 `core/`·`algorithms/`·`run.py`는 그대로 두었습니다. 다중 GPU 배치와 전처리·모델 로딩 오버랩은 이 포트의 범위 밖입니다.
- INFaaS와 비교하려면 GPU·가중치·입력·전처리·CPU 스레드·SLO 측정 기준을 맞춰 순차 실행하세요. 전처리의 텐서 수준 동등성은 별도 확인이 필요합니다.
- 로컬 테스트는 테스트용 실행기로 연결을 검사합니다. 서버 V100 실측 완료 여부는 `doctor`·`register`·`query`·`policy-test` 결과로 판단하세요.
