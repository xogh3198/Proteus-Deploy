# Proteus 독립 GPU 실행 — 0.2.0

처음 실행할 때는 [한국어 Docker 실행 가이드](docker.md)를 따라가세요.
이 문서는 전환 절차, 정책 구현 범위와 상세 측정 기준을 설명합니다.

## 구현 범위

원본 `core/`, `algorithms/`, `run.py`, 시뮬레이터 Dockerfile과 requirements는
수정하지 않았다. 추가 패키지 `proteus_realtime/`가 실제 CUDA 실행을 담당한다.

- 독립 PyTorch/Hugging Face 런타임. INFaaS 이미지 상속·코드 import·RPC 호출 없음.
- 단일 GPU / 단일 ImageNet 애플리케이션으로 축소한 MILP를 실제 Gurobi로 풂.
  결과 모델을 실제 GPU에 배치하고 요청을 처리한다. 한 장이므로 장치 간 분배는 1이다.
- 원본 accscale의 배치 크기·deadline 규칙을 wall-clock 큐에 포팅.
  실제 배치 텐서를 만들어 GPU forward를 수행한다. 가상 실행시간을 결과로 쓰지 않는다.
- 실측 배치 프로파일, gRPC, 상태·로그, cold/warm 진단, 부하 생성, CSV/JSON.
- 여러 모델을 활성화하면 동일 검증셋의 사전 Top-1 정확도로 모델 선택 가능.
  기본은 ResNet-50 하나이므로 정확도 스케일링의 효과는 검증할 수 없다.

이는 **단일 GPU용 실험적 포트**이지 논문의 전체 클러스터 구현을 그대로 실행한 것이 아니다.
원본 MILP/배치 클래스는 시뮬레이터 이벤트에 의존하므로 직접 GPU 워커로 사용하지 않는다.
새 policy.py에 제한된 경우의 수식과 배치 규칙을 옮겼다. 다중 장치·다중 애플리케이션,
전처리/모델 로딩 오버랩, INFaaS native QueryOnline(:50052)은 구현 범위 밖이다.

## 1. 이전 0.1.0 설치에서 전환

`docker/v4`를 새로 받은 경우에는 아래 증분 패치를 적용하지 않고 [설정 및 실행 순서](setting_readme.md)로 진행합니다.
아래 전환 절차는 이전 0.1.0 bootstrap을 설치해 둔 경우에만 해당합니다.

**기존 스크립트가 남아 있을 때 먼저** 이전 Proteus 워커를 중지한다.
새 버전은 Compose 프로젝트를 proteus-standalone으로 분리하므로 이전 프로젝트를
자동 종료하지 않는다. INFaaS 등 다른 실험은 이 스크립트가 종료하지 않는다.

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
bash deploy/docker/proteus-docker.sh down
```

그다음 전달받은 증분 패치를 확인·적용한다. 패치는 기존 bootstrap이 설치된 트리 기준이다.

```bash
git apply --check /home/xsailor4/taeho/Proteus_GPU_standalone.patch
git apply /home/xsailor4/taeho/Proteus_GPU_standalone.patch
```

이전 로컬 env에 BASE_IMAGE / MODEL_DIR=auto / INFAAS_ENV_FILE 등이 있으면 새 설정과
호환되지 않는다. 기존 파일은 보존하고 새 기본값을 사용한다.

```bash
if [ -f deploy/docker/proteus-docker.local.env ]; then
  mv -n deploy/docker/proteus-docker.local.env deploy/docker/proteus-docker.local.env.bootstrap-backup
fi
```

백업 경로가 이미 있어서 mv가 기존 파일을 남겼다면 다른 백업 이름을 선택한다.
원본 코드나 결과·가중치 파일을 지우거나 git reset을 실행할 필요는 없다.

## 2. 독립 설정·이미지·가중치

기본 env로 시작할 수 있다. 변경할 때만 기존 파일이 없는지 확인하고 복사한다.

```bash
cp -n deploy/docker/proteus-docker.env deploy/docker/proteus-docker.local.env
bash deploy/docker/proteus-docker.sh settings
bash deploy/docker/proteus-docker.sh gpus
bash deploy/docker/proteus-docker.sh build
```

주요 기본값:

| 설정 | 값 |
|---|---|
| GPU / GPU_TYPE | 2 / v100 |
| WORKER_IMAGE | proteus-worker:0.2.0 |
| PyTorch / torchvision | 2.11.0+cu126 / 0.26.0+cu126 |
| APPLICATION / MODEL_NAME | imagenet / resnet-50 |
| MODEL_DIR | data/proteus/models (Proteus 전용 HF_HOME) |
| DATA_DIR | data/proteus (profiles/results) |
| MODEL_REGISTRY | deploy/docker/models.json |
| PORT | 18081, 호스트 127.0.0.1만 공개 |
| SLO_MS / BATCH_SIZES | 200 / 1,2,4,8 |
| CPU_THREADS / WARMUP_QUERIES | 2 / 10 |
| ALLOCATION_INTERVAL / EWMA_ALPHA | 1초 / 0.7 |
| DEADLINE_GUARD_MS | 2 |
| ALLOW_GPU_SHARING | 0 (다른 GPU 프로세스가 있으면 기동 거부) |

이미지는 Python 기본 이미지에 CUDA 지원 PyTorch 등 실행 라이브러리를 설치한다.
모델 가중치는 이미지에 넣지 않는다. 컨테이너의 CUDA 런타임과 nvidia-smi의
드라이버 지원 표시는 별개이며, 실제 호환성은 doctor에서 CUDA 연산으로 확인한다.

**가중치 준비는 아래 둘 중 하나만** 선택한다.

```bash
# 새 전용 캐시에 다운로드
bash deploy/docker/proteus-docker.sh download --models resnet-50

# 또는 기존 체크포인트를 전용 캐시에 복사 (공유 마운트 아님)
bash deploy/docker/proteus-docker.sh seed --from /data/podexec-models --models resnet-50
```

seed는 원본을 읽기 전용으로 읽고 별도 복사한다. 대상에 이미 캐시가 있으면 덮어쓰지 않고
실패한다. 학습은 하지 않는다. 데이터셋도 자동으로 받지 않는다.
ResNet-50 revision은 이전 실험에서 확인된 체크포인트 SHA로 고정했다.
serve 시 가중치 볼륨은 읽기 전용이고 HF offline 모드다.

이전 실험과 같은 JPEG가 이미 inputs/test-dog.jpg에 있으면 그대로 사용한다.
없으면 직접 복사하거나 sample을 실행한다.

```bash
# 기존 JPEG가 없다면:
cp -n /home/xsailor4/taeho/infaas_deploy/inputs/test-dog.jpg inputs/test-dog.jpg
# 또는 PyTorch Hub 예제 JPEG 받기:
bash deploy/docker/proteus-docker.sh sample

# 워커를 띄우기 전에 CUDA 및 실제 Gurobi 라이선스 확인:
bash deploy/docker/proteus-docker.sh doctor
```

doctor가 성공해야 기동한다. 이 작은 MILP는 제한형 라이선스로 풀릴 수 있지만,
full academic 라이선스가 필요하면 로컬 env의 GUROBI_LICENSE_FILE에 유효한
라이선스 파일의 절대 경로를 지정한다. 파일은 읽기 전용 마운트하며 이미지/Git에 넣지 않는다.
호스트 종속 라이선스가 컨테이너에서도 유효한지는 doctor로 확인해야 한다.
doctor는 모델 추론 성공 여부까지 확인하는 명령은 아니다.

## 3. 실측 프로파일 → MILP → GPU 배치

```bash
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-50 --trials 30
bash deploy/docker/proteus-docker.sh state
bash deploy/docker/proteus-docker.sh policy-test inputs/test-dog.jpg
```

처음 up 직후는 GPU/API 준비만 확인된 상태다. register 없이 정책 요청을 보내면 거절된다.
register는 활성 모델의 배치 1/2/4/8을 실제 GPU에서 실행해 서비스 시간 p50/p95를 저장한다.
프로파일에는 JPEG 디코딩·전처리·H2D·forward·결과 변환이 포함되며 모델 로딩은 별도로 기록한다.
현재 입력 한 장을 반복하는 프로파일이므로 다양한 크기의 실제 데이터셋을 대표한다고 볼 수 없다.

MILP는 실행시간 p50 < SLO/2인 배치만 고려하고, 최대 유효 배치의 B/시간을 용량으로 사용한다.
관측 요청률을 EWMA로 갱신해 1초 단위로 재최적화한다. 수요를 만족하지 못하면 원본 전략과
같이 beta를 완화한다. beta는 계획상의 처리 비율이지 별도의 실제 admission rate limiter가 아니다.
최적화 실패 시 마지막 계획을 유지하고 policy_error를 표시한다.

state에서 확인할 항목:

- independent_runtime=true
- implementation=proteus-single-gpu-port
- mode=proteus, milp_enabled=true, policy_ready=true, batching=accscale
- plan.solver_status=optimal, milp_solve_count > 0
- policy-test 후 batch_size_counts에 1보다 큰 실제 배치가 있음

policy-test는 동시 요청을 보내 실제 배치 > 1과 MILP 연결을 검사한다.
프로파일상 최대 배치가 1이면 PASS하지 않는다. 모든 요청의 SLO 만족을 보장하는 검사는 아니다.
적은 요청에서는 더 모으기 위해 기다리므로 e2e가 SLO 가까이 올라갈 수 있다.
p50 기반 예측·타이머·실행 변동·네트워크 때문에 SLO 미스도 실제로 집계한다.
guard는 조기 dispatch 여유이며, 이를 조정했다고 논문 결과가 그대로 재현되는 것은 아니다.

일반 반복 요청과 부하 실험:

```bash
bash deploy/docker/proteus-docker.sh query imagenet inputs/test-dog.jpg 200 -n 11
bash deploy/docker/proteus-docker.sh bench inputs/test-dog.jpg --rate 10 --duration 30 --distribution uniform
bash deploy/docker/proteus-docker.sh bench inputs/test-dog.jpg --rate 10 --duration 30 --distribution poisson --seed 0
bash deploy/docker/proteus-docker.sh bench inputs/test-dog.jpg --rate 10 --duration 30 --distribution gamma --shape 0.05 --seed 0
```

벤치는 SLO 내 QPS, 전체 성공 QPS, e2e p50/p95/p99, 거절/실패/SLO 미스,
발송 지연/클라이언트 누락, 실험 전후 정책 상태를 CSV/JSON으로 저장한다.
랜덤 분포의 rate는 평균 요청률이다. 기본 동시성은 64이고 높일 때 클라이언트 누락을 확인한다.
QPS 분모는 응답 drain을 포함한 전체 측정 시간이다. 단일 JPEG 반복은 정확도 실험이 아니다.

## 4. INFaaS와 같은 cold/warm 진단

```bash
bash deploy/docker/proteus-docker.sh test inputs/test-dog.jpg --warm-requests 10
```

test는 일시적으로 fixed1 진단 모드로 전환해 모델을 내린 다음 첫 요청 1회와 warm 요청 10회를
측정하고 원래 정책 모드로 복원한다. **이 결과를 적응형 배치 성능으로 보고하지 않는다.**
정책 모드의 기본 모델은 register/재시작 시 사전 로딩한다.
첫 요청의 정확한 의미는 model non-resident이고 OS page cache/CUDA context까지 초기화하지 않는다.

시간 구분:

- e2e: 클라이언트 RPC 왕복. 클라이언트의 SLO 판정 기준.
- server: 서버 접수→완료, 큐/로딩/서비스 포함. 네트워크 제외.
- queue: 의도적인 배치 대기 및 앞선 작업 대기.
- load: 모델 reset·체크포인트 읽기·GPU 전송·워밍업.
- checkpoint_read_ms / model_h2d_ms / warmup_ms: 로딩 내 단계.
- startup_ms: 최초 라이브러리 import·CUDA 초기화. load와 별도.
- forward: 입력 H2D·GPU forward·argmax·D2H/동기화 포함, 순수 커널 시간 아님.
- OK: 추론 성공일 뿐 SLO 만족이라는 뜻은 아님.

배치의 단계별 시간은 여러 응답에 동일하게 들어갈 수 있으므로 요청별 값을 모두 더해
총 GPU 실행시간으로 계산하지 않는다. reset은 가중치 파일을 삭제하지 않는다.

## 5. 정확도 스케일링은 별도 검증 데이터가 필요

모델 하나에서 normalized_accuracy=100은 상대적인 기준값이다.
실제 Top-1 100%를 뜻하지 않는다. Samoyed라는 출력 라벨 또한 정확도 수치가 아니다.

여러 변형을 비교할 때는:

1. models.json을 models.local.json으로 복사하고 ResNet-18 등 enabled=true로 설정.
   로컬 env의 MODEL_REGISTRY를 deploy/docker/models.local.json으로 변경.
2. download로 새 변형의 가중치를 받는다. 재현 실험에서는 revision을 커밋 SHA로 고정한다.
3. up 전에 동일한 라벨이 있는 ImageNet 검증 이미지와 manifest를 EVAL_DIR에 준비한다.
4. 모든 활성 모델을 register한다. 아직 정확도가 없으면 policy_ready=false가 정상이다.
5. evaluate로 동일 데이터셋의 Top-1을 측정하거나, 동일 metric/dataset에서 검증된 사전
   정확도·출처를 registry에 입력한다. 이후 state에서 policy_ready=true를 확인한다.

```bash
cp -n deploy/docker/models.json deploy/docker/models.local.json
# enabled/revision을 수정하고 env의 MODEL_REGISTRY를 변경한 뒤:
bash deploy/docker/proteus-docker.sh download --models resnet-50 resnet-18
bash deploy/docker/proteus-docker.sh down
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-50 resnet-18
bash deploy/docker/proteus-docker.sh evaluate --manifest manifest.json --timeout 3600
```

manifest.json 형식 (라벨은 각 체크포인트 config.id2label과 일치하는 ImageNet ID 0..999):

```json
[
  {"path": "images/example-a.jpg", "label": 258},
  {"path": "images/example-b.jpg", "label": 281}
]
```

위는 형식 예시이지 제공되는 검증셋이 아니다. manifest와 이미지는 EVAL_DIR 내부에 둔다.
ImageNet 폴더명의 순서나 1-based 라벨을 그대로 쓰지 말고 checkpoint ID와 매핑한다.
모델별 정답수/전체수·Top-1·데이터 내용 해시·체크포인트를 기록한다.
측정 중 데이터가 변경되면 거절한다. 작은 임의 샘플은 대표적인 정확도 근거가 아니다.
정책 입력은 최고 모델 대비 정규화한 사전 정확도이며, 온라인 confidence가 아니다.

현재 벤치는 선택 variant와 계획상의 정확도를 기록하지만 **논문 Effective Accuracy
그래프를 자동 산출하지 않는다**. 같은 검증 지표를 준비한 뒤 처리 요청수 가중평균의
분모/실패 처리 기준까지 정해 공통 비교 harness에서 계산해야 한다.

## 6. 격리·상태·종료

```bash
bash deploy/docker/proteus-docker.sh state --watch 2
bash deploy/docker/proteus-docker.sh logs
bash deploy/docker/proteus-docker.sh logs --follow
bash deploy/docker/proteus-docker.sh down
nvidia-smi -i 2
```

up 이후 프로세스가 계속 떠 있고, 모델 상주 시 GPU 메모리를 유지한다.
실험이 끝나면 down한다. 이 Proteus Compose 프로젝트만 종료하며 캐시/프로파일/결과는 보존한다.
Docker prune, 다른 프로젝트 종료, GPU 전력/클럭 변경, 시스템 라이브러리 변경은 하지 않는다.
ALLOWSHARING 기본 거부는 기동 시 확인일 뿐, 나중에 다른 사용자가 GPU를 쓰는 것까지 막지 못한다.
같은 GPU의 INFaaS/Clipper 실험은 순차 실행하고, 실제 nvidia-smi 상태를 확인한다.

기존 podexec gRPC 클라이언트도 127.0.0.1:18081로 요청 가능하다.
단일 변형에서만 resnet-50 별칭을 허용하며 다중 변형은 model="imagenet"으로 보내야 정책이 선택한다.
관리 API는 인증이 없으므로 외부 공개하지 않는다.

## 7. 검증 상태와 비교 실험 주의

개발 환경: 실제 Gurobi 단위 검사 + 테스트용 GPU 실행기의 로컬 gRPC 통합 검사,
Bash 명령 분기 검사. 테스트의 지연 수치를 실험 결과로 사용하지 않는다.
서버 자동 SSH 인증이 되지 않아 Docker 빌드·V100 모델 추론은 아직 검증하지 못했다.
서버의 doctor/register/policy-test/state 결과를 확보한 뒤 실측 완료로 보고한다.

원본 core/algorithms 없이도 이미지가 빌드되므로 원본 시뮬레이터 requirements를 설치하지 않는다.
PyTorch/FP32/체크포인트는 기존 조건에 맞췄지만 전처리는 독립 PIL RGB + HF slow processor다.
INFaaS의 DALI/PIL fallback과 **텐서 수준의 동등성을 아직 검증하지 않았다**.
공정한 비교에는 실제 checkpoint, 입력, 전처리, CPU thread/워밍업, GPU 클럭,
SLO 시작 시점과 동일 요청 부하/집계 기준을 맞춰야 한다.
공통 부하 harness 및 논문 전체 재현은 후속 작업이다.

정책 근거: [원본 Proteus](https://github.com/UMass-LIDS/Proteus), commit 6967c932,
algorithms/ilp.py 및 core/predictor.py의 accscale 경로.
이 추가 코드의 출처는 해당 규칙의 단일 GPU 포트이며, INFaaS 원본 알고리즘을 사용하지 않는다.
