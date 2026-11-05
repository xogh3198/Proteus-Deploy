# v1: 공개 Proteus 정책 + 실제 GPU 계측

현재 v1은 앞서 작성한 대체 solve_milp/adaptive_decision을 호출하지 않는다.
v2는 이전 GPU 포트이며 이번 소스 보존 수정은 v1에만 적용했다.

| 구현 | 실제 호출 |
|---|---|
| proteus | algorithms.ilp.Ilp.run, core.simulator.Simulator의 배치·라우팅 적용, core.executor.Executor, core.predictor.Predictor |
| infaas-accuracy | 같은 Simulator·Executor·Predictor와 원래 trigger_infaas_v2_upscaling/downscaling |

GPU에 모델을 올리고 원래 Predictor가 정한 배치 구성원을 실제로 실행한다.
시뮬레이터의 예측 END/FINISH만 실제 GPU 완료 통지로 대체하며 예측 완료를 성공 응답으로 쓰지 않는다.
새 RPC 여유·deadline guard·early drop·큐 크기 제한·lazy-drop watchdog을 추가하지 않는다.
제어는 원래 1초 수요 집계와 EWMA, 공개 config의 10초 제어 간격을 사용한다.
MILP 목적함수·제약·beta 감소·Gurobi 설정을 그대로 사용한다.
CSV 입력은 원래 로더의 50th_pct 매핑을 유지한다. 실제 반복 측정값의 중앙값과 평균을 모두 저장한다.

## 원본 대비 필요한 호환성 수정

기준: UMass-LIDS/Proteus @6967c9327e66f6d6fe94993dcaa7521efc2f72f5.
원본 정책 파일 11개의 원본/실행 SHA256을 experiments/reproduce/proteus_source.json에 기록하고 시작 시 확인한다.

- algorithms/ilp.py: configured_accelerators가 주어졌을 때만 실제 장치 목록으로 필터한다. V100 한 장을 원본 GPU_AMPERE 슬롯 0에 매핑하며 가상 CPU/GPU를 추가하지 않는다.
- core/predictor.py: 오래된 SLO 타이머 판정에서 현재 시각 대신 이벤트의 예약 시각을 비교한다. 원래 가상 시계에서는 두 값이 같아 같은 동작이며, 실제 OS wakeup 지연으로 정상 이벤트가 무시되는 것을 방지한다.

그 밖의 core·algorithms·run.py는 원본과 같다. 이 두 변경을 무수정 원본이라고 표기하지 않는다.
원본 가상시계 조건과 늦은 실제 wakeup 조건의 차이를 테스트한다.
공개 artifact는 시뮬레이터이므로 이 GPU 완료 어댑터를 논문의 비공개 production cluster 코드와 동일하다고 부르지 않는다.

모델은 공개 ONNX ResNet-v1 18·34·50·101·152 FP32, 실행은 ONNX Runtime CUDA EP다.
터미널의 모델 이름은 resnet18·resnet34·resnet50·resnet101·resnet152로 표시한다.
설정·정책·프로파일과 저장되는 JSON/CSV는 원본 식별자(resnet18_v1 등)를 유지한다. 이름 표시는 가중치나 정책을 바꾸지 않는다.
프로파일과 실제 forward 시간이 크게 다르면 [별도 ONNX 원인 진단](../../diagnostics/README.md)을 사용한다. 원래 워커 이미지와 읽기 전용 프로파일로 실행하며 rebuild/register가 필요하지 않다.
정확한 원 논문 checkpoint는 식별되지 않았으며 models.json의 checkpoint 출처·점수 차이를 기록한다.
v1은 Proteus의 추상 입력에 대응하는 미리 준비한 텐서를 사용한다. 실제 추론·H2D·결과 회수는 요청마다 실행한다.

## 공개 Twitter·급증 부하와 SLO 그대로 재생

Proteus 실행 명령에 `--workload twitter` 또는 `--workload bursty`를 붙인다.
이 옵션은 개인 local.env의 SLO_MS=200·TRACE_SPEED=2보다 우선하여 공개 ResNet 파일의 SLO 300ms·추가 배속 1을 적용한다.
요청마다 파일의 세 번째 열을 실제 RPC의 SLO로 사용하며, 도착 시각·QoS·요청 수를 변경하지 않는다.
트레이스는 아래 고정 커밋의 원본 파일이고, 재생 직전 SHA256·전체 건수·마지막 도착 시각·SLO를 검사한다.
원본 파일은 이미 가공된 실험 입력이다. 1배속은 이 파일에 추가 시간축 압축을 하지 않는다는 뜻이다.

| 옵션 | 공개 파일 (traces/twitter/asplos 아래) | 요청 수 | 마지막 도착 시각 | 평균 QPS | SLO |
|---|---|---:|---:|---:|---:|
| twitter | zipf_exponential/300/resnet.txt | 3,776 | 290.308초 | 13.01 | 300ms |
| bursty | zipf_exponential_bursty/300/resnet.txt | 8,309 | 290.444초 | 28.61 | 300ms |

출처: https://github.com/UMass-LIDS/Proteus/tree/6967c9327e66f6d6fe94993dcaa7521efc2f72f5/traces/twitter/asplos
프리셋에서 `--speed 2`처럼 추가 배속을 지정하면 요청을 전송하기 전에 거부한다.
GPU·가중치·배치 후보·CPU·제어 설정·START_STATE는 기존 설정을 유지한다. 이번 변경으로 core/algorithms나 정책·로딩 순서를 수정하지 않는다.
재현 범위는 공개 ResNet 워크로드와 SLO다. 논문의 전체 다중 모델 그룹·이기종 클러스터나 비공개 서빙 구현과 동일하다고 표기하지 않는다.
논문 §6.1.3의 압축 배율 전체를 새로 추정하거나, 임의의 요청을 추가하지 않는다.

아래 명령은 기존 Proteus 폴더에서 실행한다. 사용 중인 일반 실행은 먼저 `proteus down`으로 종료한다.

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{.DockerRootDir}}'

# Twitter: 처음 한 번 빌드·입력 준비·프로파일링
bash deploy/docker/source-docker.sh proteus settings --workload twitter
bash deploy/docker/source-docker.sh proteus down --workload twitter
bash deploy/docker/source-docker.sh proteus build --workload twitter &&
bash deploy/docker/source-docker.sh proteus setup --workload twitter &&
bash deploy/docker/source-docker.sh proteus up --workload twitter &&
bash deploy/docker/source-docker.sh proteus register --workload twitter

# 프로파일링 상태를 비운 새 프로세스에서 전체 요청 재생
bash deploy/docker/source-docker.sh proteus down --workload twitter &&
bash deploy/docker/source-docker.sh proteus up --workload twitter
bash deploy/docker/source-docker.sh proteus trace --workload twitter --dry-run
bash deploy/docker/source-docker.sh proteus trace --workload twitter --limit 0
bash deploy/docker/source-docker.sh proteus state --workload twitter
bash deploy/docker/source-docker.sh proteus down --workload twitter

# 급증 부하: 별도 프로파일·결과 경로에서 같은 준비 순서
bash deploy/docker/source-docker.sh proteus down --workload bursty
bash deploy/docker/source-docker.sh proteus build --workload bursty &&
bash deploy/docker/source-docker.sh proteus setup --workload bursty &&
bash deploy/docker/source-docker.sh proteus up --workload bursty &&
bash deploy/docker/source-docker.sh proteus register --workload bursty
bash deploy/docker/source-docker.sh proteus down --workload bursty &&
bash deploy/docker/source-docker.sh proteus up --workload bursty
bash deploy/docker/source-docker.sh proteus trace --workload bursty --dry-run
bash deploy/docker/source-docker.sh proteus trace --workload bursty --limit 0
bash deploy/docker/source-docker.sh proteus state --workload bursty
bash deploy/docker/source-docker.sh proteus down --workload bursty
```

두 조건은 기존 MODEL_DIR의 가중치와 트레이스를 공유한다. 빌드 계층도 재사용한다.
프로파일·결과·Compose 프로젝트는 기존 실험 및 두 워크로드 사이에서 분리된다. 같은 GPU에서는 한 번에 하나만 실행한다.
모든 lifecycle 명령에 같은 `--workload`를 붙인다. 생략하면 기존 일반 실험의 설정·경로를 사용한다.
기존 200ms 프로파일을 새 프리셋에 복사하지 말고 새 경로에서 register한다.
JSON에는 원본 URL·해시·SLO 출처·실행 설정을, CSV에는 source_slo_ms와 실제 요청의 request_slo_ms를 기록한다.
START_STATE=cold 기본값이면 측정 첫 요청의 실제 모델 로딩도 포함된다. 모델 다운로드·프로파일링은 측정 요청에 포함하지 않는다.
미전송과 전송 지연도 함께 확인한다. 원본 요청이 모두 전송되지 않으면 같은 부하를 재현했다고 해석하지 않는다.

이미 register가 끝나 policy_ready=true인 워커에서는 build/setup/up/register를 반복하지 않고 아래 trace 검증·측정으로 이어간다.
`This experiment is already running`은 같은 Compose 프로젝트의 중복 시작을 막았다는 뜻이다.
코드를 갱신하거나 재빌드할 때는 위의 workload별 down을 먼저 실행한다. 이 명령은 가중치·프로파일·결과를 삭제하지 않는다.
실행 중인 서비스는 자동 종료·교체하지 않는다.

기본 저장 경로:

```text
/data4tb/taeho-proteus/experiments/paired-resnet/proteus/source-v1/published-twitter/results/
/data4tb/taeho-proteus/experiments/paired-resnet/proteus/source-v1/published-bursty/results/
```

DATA_DIR을 개인 설정에서 바꿨다면 그 경로 아래에 같은 하위 디렉터리를 만든다.

## 실행 순서

기존 수정 파일이 있는 폴더는 그대로 두고 v1 커밋을 별도 폴더에서 실행한다.
기존 저장소 폴더에서 아래를 실행하며, 새 폴더 이름은 저장소별로 다르게 지정한다.

```bash
git fetch origin refs/heads/v1
git worktree add --detach ../Proteus-source-v1 FETCH_HEAD
cd ../Proteus-source-v1

systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{.DockerRootDir}}'
nvidia-smi -i 2
```

저장 위치는 /data4tb/taeho-proteus/docker-data. V100에 다른 작업이 있으면 해당 작업을 먼저 종료한다.
스크립트가 다른 실험을 종료하거나 Docker daemon 설정을 바꾸지 않는다.

```bash
# 실행할 구현을 아래에서 하나 지정한다.
ENGINE=proteus

bash deploy/docker/source-docker.sh "$ENGINE" settings
bash deploy/docker/source-docker.sh "$ENGINE" build &&
bash deploy/docker/source-docker.sh "$ENGINE" setup &&
bash deploy/docker/source-docker.sh "$ENGINE" up &&
bash deploy/docker/source-docker.sh "$ENGINE" register &&
bash deploy/docker/source-docker.sh "$ENGINE" state

# 프로파일링 종료 후 측정용 프로세스·정책 상태를 새로 시작한다.
bash deploy/docker/source-docker.sh "$ENGINE" down &&
bash deploy/docker/source-docker.sh "$ENGINE" up

bash deploy/docker/source-docker.sh "$ENGINE" trace --dry-run
bash deploy/docker/source-docker.sh "$ENGINE" trace --limit 0 --speed 2 --concurrency 256 --timeout 10
bash deploy/docker/source-docker.sh "$ENGINE" state
bash deploy/docker/source-docker.sh "$ENGINE" down
```

설정 변경은 experiments/resnet/local.env에 필요한 KEY=VALUE만 쓴다. 없으면 settings.env 기본값을 사용한다.
기존 local.env가 있으면 그것이 우선이므로 settings 출력으로 적용 값을 확인한다.
GPU=2, SLO_MS=200, WORKLOAD=twitter, TRACE_SPEED=2가 현재 비교 조건이다.
공개 ResNet trace의 원래 SLO는 300ms이며 200ms는 기존 실험에서 선택한 조건이다. 논문의 SLO를 그대로 재현한 값이라고 쓰지 않는다.
Twitter 목록은 3,776건이고 2배속이면 145.154초·평균 26.01 QPS다. 파일의 도착 시각·QoS·원 SLO와 파일 해시를 기록한다.
배속으로 요청 수가 늘어나지는 않는다.

## 결과 해석

JSON/CSV에 원래 클라이언트 E2E, 성공/늦은 성공/실패·거절/미전송, 요청별 모델·배치·원래 stage timings를 저장한다.
SLO 위반율은 실제 클라이언트 E2E와 실패를 기준으로 계산한다. 성공 응답 지연 분위수는 성공 응답만의 분포다.

measurement_boundaries에는 전처리 wall time, 전처리 완료 뒤 worker 결과까지의 구간, E2E−전처리 wall time을 별도로 저장한다.
DALI 전처리는 GPU 연산을 포함할 수 있으므로 전처리 wall time 전체를 CPU 연산 시간이라고 부르지 않는다.
E2E−전처리는 같은 실행의 산술 분해이며, 전처리를 실제로 제거했을 때의 결과나 원래 SLO 위반율을 대신하지 않는다.
전처리 앞에 있던 큐 대기·모델 로딩은 after_preprocessing_tail에 포함되지 않는다. 원래 배치/전처리 순서가 달라 이 값만으로 전체 서빙을 비교하지 않는다.
모델 로딩은 원래 timings/이벤트에 별도로 기록한다. 측정 구간을 맞추려고 원래 deadline이나 거절 조건을 바꾸지 않는다.

실측 정확도는 측정하지 않는다. 표시되는 모델 점수는 supplied Proteus reference이고 현재 공개 체크포인트의 정답률과 구분한다.
정규화 점수는 다섯 후보의 최고 기준 점수 79.22를 공통 분모로 사용한다. 성공 응답 전체와 SLO 내 성공 응답을 나누어 기록한다.
단일 V100·ResNet 부분 실험이며, 논문 전체 클러스터 결과의 수치 재현이라고 주장하지 않는다.
로컬 CPU 연결·소스 차이 검사는 수행하지만 Docker 빌드와 GPU 실행은 GPU 서버에서 확인해야 한다.

결과 기본 경로:

```text
/data4tb/taeho-proteus/experiments/paired-resnet/<proteus 또는 infaas-accuracy>/source-v1/results/
```
