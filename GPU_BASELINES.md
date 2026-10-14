# Proteus 내부 비교 기준선 — 독립 GPU 실행

기존 INFaaS·Proteus·독립 Clipper 배포는 그대로 둡니다. 이 추가 구성은 Proteus 저장소의
**INFaaS-Accuracy / Clipper-HA / Clipper-HT 기준선**을 각각 별도 컨테이너에서 실행합니다.
Stanford INFaaS 서비스나 원본 C++ Clipper 서비스의 별칭이 아닙니다.

## 1. 범위와 동일 조건

| 항목 | 설정 |
|---|---|
| 자원 | 호스트 GPU 2, V100 32GB 한 대, 세 기준선을 하나씩 실행 |
| 모델·가중치 | microsoft/resnet-50 FP32, revision `34c2154c194f829b11125337b98c8f5f9965ff19` |
| 입력·SLO | 기존 test-dog.jpg 그대로, 200ms |
| 전처리·런타임 | 기존 Proteus GPU 포트의 PIL RGB + HF slow AutoImageProcessor, CPU 스레드 2 |
| 소프트웨어 | Python 3.11, Torch 2.11.0+cu126, torchvision 0.26.0+cu126, 기존 GPU requirements |
| 공통 경로 | gRPC → 원본 기준선 정책 → 실제 CUDA forward → 실제 완료 뒤 응답 |
| 모델 준비 | 기존 Proteus와 동일한 WARMUP_QUERIES=10, CUDA math 검사, TF32/cuDNN benchmark 끔 |

ResNet-50 하나뿐이면 HA와 HT 모두 같은 모델을 선택하고 INFaaS의 상대 정확도 비용은 0입니다.
`normalized_accuracy=100`은 실제 Top-1 100%가 아닙니다. 모델 선택/accuracy scaling/복제는 비활성화됩니다.
이번 범위는 **단일 모델 GPU 기능·배치·SLO 검증**이며 논문의 전체 클러스터/정확도 비교 재현이 아닙니다.
GPU에서 분류는 실제 수행하지만 같은 JPEG 반복으로 분류 정확도를 측정하지는 않습니다.

기존 독립 INFaaS는 gRPC/DALI, 독립 Clipper는 HTTP/PIL/Adaptive Batching v2 경로입니다.
이 기준선은 공통 Proteus GPU 백엔드를 쓰므로 환경을 통일한 정책 실험과 독립 시스템 실험을 구분하세요.
GPU 실행 코드를 import하여 재사용하지만 기존 Proteus 워커·이미지·컨테이너에 요청을 보내지 않습니다.

## 2. 원본 정책 사용과 어댑터 경계

- 원본 `core/predictor.py`, `core/executor.py`, `algorithms/clipper.py`, 기존 `proteus_realtime` 파일은 수정하지 않습니다.
- INFaaS-Accuracy: 원본 `infaas_v2` 업/다운 스케일 제어와 Predictor의 `infaas` 배치 메서드를 호출합니다.
- Clipper-HA/HT: 원본 Predictor의 AIMD 메서드를 호출합니다. 정적 배치는 단일 GPU·동일 모델 1개로 축소되어 원본 다중 장치 allocation 파일을 사용하지 않습니다.
- GPU batch service를 실제 측정하여 **p50 < SLO/2**인 배치만 원본 정책에 후보로 제공합니다. 기존 정책의 후보 집합을 제한한 단일 GPU 포트라는 점을 명시합니다.
- 원본의 가상 END_REQUEST/FINISH_BATCH는 배치 구성에만 사용합니다. 프로파일 값만큼 sleep한 뒤 성공하는 시뮬레이션이 아닙니다. 응답 시간·완료·SLO는 실제 GPU 결과를 기준으로 기록합니다.
- 원본 AIMD feedback은 프로파일 기반 예상 완료 시간 판정을 유지합니다. 실제 SLO miss는 별도 집계합니다. 이를 실제 지연 피드백 AIMD라고 부르지 않습니다.
- 원본 AIMD의 `aimd_expiring_callback`은 no-op입니다. 부분 배치를 임의로 실행해 숨기지 않습니다. 어댑터의 watchdog이 원본 `drop_expired_requests`의 **2*SLO 초과 lazy-drop**을 호출하여 무한 대기를 방지합니다. 이 watchdog은 원본 시뮬레이터와 다른 wall-clock 생존성 보완입니다.
- 단일 GPU가 이미 할당되어 추가 장치/replica 가용량은 0입니다. CPU/VPU/다른 GPU로 몰래 실행하거나 가짜 처리 용량을 입력하지 않습니다.
- 원본 파일과 공통 GPU 백엔드의 SHA256을 build/start에서 검사합니다. 변경이 있으면 덮어쓰지 않고 중단합니다.

지난 Proteus 실측에서는 배치 1만 100ms(SLO/2) 미만이었습니다. 다시 측정해도 최대 배치가 1이면
세 기준선의 배치 차이가 나타나지 않을 수 있습니다. 배치 크기 >1을 확인하려고 SLO/프로파일 값을 조작하지 마세요.

## 3. 독립 구성

| 이름 | 컨테이너·네트워크 | 호스트 포트 | 개별 데이터 디렉터리 |
|---|---|---|---|
| INFaaS-Accuracy | proteus-baseline-infaas-accuracy | 127.0.0.1:19081 | DATA_ROOT/infaas-accuracy |
| Clipper-HA | proteus-baseline-clipper-ha | 127.0.0.1:19082 | DATA_ROOT/clipper-ha |
| Clipper-HT | proteus-baseline-clipper-ht | 127.0.0.1:19083 | DATA_ROOT/clipper-ht |

신규 이미지 `proteus-baseline-worker:0.1.0`는 세 기준선의 실행 코드를 포함합니다.
각 컨테이너는 별도 프로세스·GPU 상태·가중치 캐시·프로파일·로그·결과를 사용합니다.
이미지의 읽기 전용 코드 계층 공유는 서로의 서비스에 의존하는 것을 뜻하지 않습니다.

설정은 `deploy/docker/baselines/baseline.env`이고 개인 변경은 `baseline.local.env`로 저장합니다.
기존 MODEL_DIR/DATA_DIR/PORT 환경변수가 섞이지 않도록 이 구성의 변경 변수는 `BASELINE_` 접두사를 사용합니다.
스크립트는 Docker context/data-root/호스트 NVIDIA 설정을 바꾸지 않습니다.

## 4. 최초 실행 — INFaaS-Accuracy부터

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
export DOCKER_CONTEXT=rootless
export BASELINE_DATA_ROOT=/data4tb/taeho-proteus/baselines

docker info --format '{{.DockerRootDir}}'
nvidia-smi -i 2

bash deploy/docker/baseline-docker.sh infaas-accuracy settings
bash deploy/docker/baseline-docker.sh infaas-accuracy build
bash deploy/docker/baseline-docker.sh infaas-accuracy download --models resnet-50
bash deploy/docker/baseline-docker.sh infaas-accuracy sample
bash deploy/docker/baseline-docker.sh infaas-accuracy doctor
bash deploy/docker/baseline-docker.sh infaas-accuracy up
bash deploy/docker/baseline-docker.sh infaas-accuracy register --trials 30
bash deploy/docker/baseline-docker.sh infaas-accuracy state
bash deploy/docker/baseline-docker.sh infaas-accuracy test --warm-requests 10
bash deploy/docker/baseline-docker.sh infaas-accuracy query -n 11
bash deploy/docker/baseline-docker.sh infaas-accuracy bench --rate 1 --duration 10 --distribution uniform --seed 10
bash deploy/docker/baseline-docker.sh infaas-accuracy logs
bash deploy/docker/baseline-docker.sh infaas-accuracy down
```

GPU가 다른 작업에 점유되어 있으면 doctor/up은 멈춥니다. 기존 Clipper/Proteus/INFaaS는 자동으로 종료하지 않습니다.
본인이 실행한 서비스만 해당 저장소의 down 명령으로 먼저 종료하세요. 공용 Docker prune/data-root 변경은 필요 없습니다.
처음 up 직후 `policy_ready=false`는 프로파일이 없다는 뜻이며, `register` 후 true가 되어야 정책 요청을 보낼 수 있습니다.

`sample`은 기존 JPEG 존재 여부·해시만 확인합니다. 새 이미지를 받거나 입력을 덮어쓰지 않습니다.
다른 위치에 있다면 `BASELINE_INPUT_IMAGE`에 **기존 JPEG의 절대 경로**를 지정하세요.
`seed --from /existing/HF_HOME`으로 기존 가중치를 읽기 전용 마운트하여 새 개별 캐시로 복사할 수도 있습니다.
학습·Gurobi 최적화는 실행하지 않으므로 이 기준선 서버에는 별도 Gurobi 라이선스가 필요하지 않습니다.

## 5. Clipper-HA / HT

위 명령에서 `infaas-accuracy`를 `clipper-ha` 또는 `clipper-ht`로 바꾸면 됩니다.
이미지는 최초 한 번만 build하면 되고, 각 기준선의 download/up/register/test/bench/down을 따로 수행합니다.
동일 V100에서 세 컨테이너를 동시에 켜면 분리된 성능 비교가 아닙니다.

## 6. 세 기준선 순차 비교 자동 실행

각 기준선의 첫 실행을 확인한 뒤 모두 down하고 실행하세요.

```bash
bash deploy/docker/baseline-docker.sh all compare \
  --rates 1,5,10 --duration 30 --distribution uniform --seed 10
```

기준선마다 독립 가중치 준비 → up → 실제 GPU profile → 고정 배치 1 cold/warm 진단 → 원본 정책 부하 실험 → down 순서입니다.
compare 실행 전에 동일 이름의 기준선 컨테이너가 남아 있으면 중단합니다. 실패 시에도 compare가 시작한 자기 컨테이너만 종료합니다.
기존 다른 시스템은 실행/종료/수정하지 않습니다. 측정에서 거절·SLO miss가 있어도 결과를 보존하며, 보고서 생성 전 오류는 중단합니다.

다른 분포는 `--distribution poisson` 또는 `--distribution gamma --shape 0.05`로 지정합니다.
Uniform은 고정 간격, Poisson은 지수 inter-arrival, Gamma는 같은 평균 QPS의 gamma inter-arrival입니다.
같은 seed로 요청 예정 시점 SHA256을 기록합니다. 동일 SHA256은 **예정 스케줄의 일치**이며 실제 전송 시점의 일치는 아닙니다.
유한 client concurrency 때문에 skipped arrivals/send lag가 클 수 있으므로 그 값을 반드시 확인하세요.

Cold/warm `test`는 명시적으로 고정 배치 1이며 끝나면 원본 기준선 정책을 복원합니다.
bench는 고정 배치 1로 모델을 준비한 뒤 원본 정책 상태를 다시 초기화해 실행합니다. 정책 부하 실험과 진단 결과를 섞지 않습니다.

## 7. 결과와 판정

- 각 기준선의 `results/`에 요청별 CSV/JSON, `profiles/`에 실제 GPU profile, `logs/`에 원본 Predictor 로그를 보존합니다.
- DATA_ROOT의 `comparison-*.csv/.json`은 성공률·SLO 충족률·goodput·p50/p95/p99·스케줄 해시·입력 해시·client lag/skips를 모읍니다.
- actual_batch_counts는 실제 CUDA forward의 배치입니다. 프로파일의 후보 크기나 가상 batch counter를 GPU 실행 증거로 대체하지 않습니다.
- 성공 응답이 SLO 만족을 뜻하지 않습니다. 실패/거절을 제외한 평균만으로 비교하지 마세요.
- e2e는 공통 gRPC 왕복 시간, server_ms는 수신부터 실제 GPU 완료까지, load_ms는 모델 준비·GPU 이동·warmup을 포함합니다.
- `classification_accuracy=null`은 미측정입니다. 실행 중 선택 모델의 사전 정확도를 평가하려면 별도 다중 모델·동일 라벨 검증셋 실험이 필요합니다.

Windows 로컬 검증은 실제 gRPC와 원본 정책 + **테스트용 가짜 GPU**입니다. 서버 V100 실측 완료는
build/doctor/register/test/bench 출력과 JSON 파일로 따로 확인해야 합니다.

```bash
python3 -m unittest discover -s tests_baselines -v
```

## 8. 빌드 오류: `core/__init__.py` not found

원본 Proteus의 `core/`에는 `__init__.py`가 없습니다. Python 3의 namespace package로
불러올 수 있으므로 원본에 파일을 추가하지 않고, Docker COPY에 실제 존재하는 파일만 지정합니다.
초기 기준선 패치의 잘못된 COPY 목록은 추가 빌드 수정 패치로 바로잡습니다.

이 단계가 실패하면 이미지가 생성되지 않아 download/doctor/up/register도 뒤이어 실패합니다.
수정 후 먼저 `bash deploy/docker/baseline-docker.sh infaas-accuracy build`의 성공을 확인하고
다음 단계로 진행하세요. 기존 torch 설치 레이어를 재사용할 수 있으므로 캐시 삭제나
`--no-cache` 빌드는 필요 없습니다.

테스트는 모든 COPY 원본의 존재 여부와 복사된 파일만으로 원본 정책·기준선 서버를
import할 수 있는지도 검사합니다. 이 검사는 실제 Docker 빌드·GPU 추론을 대체하지 않습니다.
