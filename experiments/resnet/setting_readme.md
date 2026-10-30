# 1차 ResNet 실험: v1 / v2

현재 실행 대상은 **ResNet-v1 18·34·50·101·152**, V100 한 대, FP32다.
51개 모델 조사는 `experiments/paper51`에 보관하며, 이 실행 경로에서는 내려받지 않는다.

| 브랜치 | 요청 시 CPU 전처리 | 실제 실행 |
| --- | --- | --- |
| `v1` | 측정 전에 만든 입력 텐서 재사용 | 요청마다 배치 구성·H2D·GPU 추론·D2H |
| `v2` | 요청마다 PIL RGB 디코딩 + HF slow 이미지 처리 | 같은 GPU 실행 경로 |

두 브랜치의 기본 설정은 같고 `phase.json`만 다르다. 같은 JPEG를 전송하며 **예측 결과는 캐시하지 않는다.**
v1도 통신·입력 확인·큐·배치 구성 비용은 포함한다. 새 이미지로 바꾸면 서버를 재시작해 준비 입력을 다시 만든다.

## 정밀도와 모델 출처

- 논문 §6.1.2·§6.1.5는 ONNX / ONNX Runtime을 명시하지만 **FP32/FP16은 명시하지 않는다.** 공개 프로파일에도 dtype 열은 없다.
- 이번 모델은 공개 ONNX Model Zoo의 ResNet-v1 다섯 개다. SHA256·크기·실제 FLOAT 가중치·FP32 입력·동적 배치를 검사한다. FP16 변환이나 양자화는 하지 않는다.
- 정확한 원 논문 체크포인트는 공개 artifact에서 식별할 수 없다. **이번 FP32 설정을 논문에서 확인한 값이라고 표현하면 안 된다.**
- 정책 입력은 기존 Proteus의 사전 기준 정확도 70.93 / 74.37 / 77.36 / 78.34 / 79.22를 유지한다. 현재 ONNX 체크포인트의 공개 Top-1은 이와 다르며 `models.json`에 별도로 기록했다. 결과의 평균 기준 정확도는 실제 정답률이 아니다.
- 실제 엔진은 네 구현 모두 ONNX Runtime CUDA EP 1.22.1이다. 논문의 참조 버전 1.8.1과는 다르다. Torch 2.11.0+cu126·torchvision 0.26.0+cu126·transformers 5.19.0·Pillow 12.1.1은 기존 실험 버전을 유지한다.

## 1. 서버에서 브랜치 선택

각 저장소 폴더에서 실행한다. 기존 수정 파일은 먼저 `git status`로 확인하며 삭제하거나 강제 초기화하지 않는다.

```bash
git status --short
git fetch origin refs/heads/v1
git switch -c v1 FETCH_HEAD
```

이미 로컬 `v1`이 있다면 마지막 줄 대신 `git switch v1` 후 `git merge --ff-only FETCH_HEAD`를 사용한다.

현재 터미널에서 개인 Docker를 선택한다. 스크립트는 daemon/context/data-root를 변경하지 않는다.

```bash
systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{.DockerRootDir}}'
```

예상 출력: `/data4tb/taeho-proteus/docker-data`.
GPU에 기존 서빙이 있으면 해당 실험의 기존 종료 스크립트로 먼저 종료한다. 새 스크립트는 다른 실험을 자동 종료하지 않는다.

## 2. 구현 선택과 설정

| 저장소 | `ENGINE` | 모델 선택 |
| --- | --- | --- |
| Proteus-Deploy | `proteus` | 5개 후보에서 기존 MILP 선택 |
| Proteus-Deploy | `infaas-accuracy` | 5개 후보에서 기존 INFaaS-v2 비교 정책 |
| INFaaS-Deploy | `infaas` | 5개 후보에서 기존 단일-GPU 비용 정책 |
| Clipper-Deploy | `clipper` | 기본 HA: ResNet-152 고정, native C++ 적응형 배치 |

```bash
# 위 표에서 현재 실행할 구현 하나를 지정
ENGINE=proteus

cp -n experiments/resnet/settings.env experiments/resnet/local.env
vim experiments/resnet/local.env
bash experiments/resnet/run.sh "$ENGINE" settings
```

주요 기본값:

| 설정 | 값 |
| --- | --- |
| `GPU` | 호스트 V100 `2`번, 컨테이너 안에서는 `0`번 |
| `MODEL_DIR` | `/data4tb/taeho-proteus/paired-resnet-models` — 네 구현·두 브랜치 공통 |
| `DATA_DIR` | `/data4tb/taeho-proteus/experiments/paired-resnet` |
| `SLO_MS` | `200` — 기존 실험 유지, 공개 트레이스의 300ms를 런타임에서 대체 |
| `BATCH_SIZES` | `1,2,4,8` — 오프라인 프로파일 후보; Clipper는 native 자동 배치 |
| `PROFILE_TRIALS` | 배치별 `30`회; GPU 프로파일은 전처리 제외 |
| `PREPROCESS_WORKERS` | CPU 전처리 동시 실행 최대 `8`개, intra-op `2`, inter-op `1` |
| `TRACE_SPEED` | `2` |
| `TRACE_CONCURRENCY` | 클라이언트 미완료 요청 상한 `256`; CPU 작업자 수와 다름 |
| `START_STATE` | `cold`; `active`는 측정 전 관리 경로에서 모델을 준비하며 도착 요청 수를 늘리지 않음 |
| `CLIPPER_MODE` | `ha` → ResNet-152; `ht` → ResNet-18 |

기존 개인 `.local.env`, `PROTEUS_ENV`, `INFAAS_ENV`는 이 경로에서 읽지 않는다.
수정할 파일은 `experiments/resnet/local.env` 하나다. 다른 설정 파일을 지정하려면 `RESNET_ENV`를 사용한다.

정책 고유 설정은 유지한다. Proteus·Accuracy 제어 주기 기본 10초, INFaaS 비용 포트 1초.
Proteus는 기존 실측 기반 RPC 여유 6ms·guard 2ms를 유지한다. 이것은 논문에서 확인한 값이 아니다.
RPC 처리 스레드는 Proteus·Accuracy 128, INFaaS 32, Clipper native query 16이다.
INFaaS 전처리는 기존 RPC 처리 스레드에서 실행하며 최대 동시 실행 수만 8로 맞춘다. 별도 CPU 작업자 풀로 교체하지 않는다.
INFaaS 비용 입력의 GPU 메모리는 ORT가 사용한 장치 메모리 증가분을 측정한다. PyTorch 할당량이나 추정 가중치 크기로 대신하지 않으며, 순간 최대 할당량을 뜻하지 않는다.
Clipper의 예측 캐시는 0, native 대기 큐는 기존처럼 제한을 추가하지 않는다.
기존 정책·프로파일 해석 차이를 제거한 실험이라고 주장하지 않는다.

## 3. 빌드·가중치 준비·서빙 시작

```bash
bash experiments/resnet/run.sh "$ENGINE" build
bash experiments/resnet/run.sh "$ENGINE" setup
bash experiments/resnet/run.sh "$ENGINE" up
bash experiments/resnet/run.sh "$ENGINE" register
bash experiments/resnet/run.sh "$ENGINE" state

# 프로파일 측정에 쓰인 정책 상태·카운터를 초기화하고 측정용 서버 시작
bash experiments/resnet/run.sh "$ENGINE" down
bash experiments/resnet/run.sh "$ENGINE" up
```

- `build`: 이 실험 전용 이미지 생성. Clipper는 기존 C++ 빌드 경로와 pinned submodule을 사용한다.
- `setup`: 모델 약 0.66GB와 공개 트레이스를 내려받고 검증한다. GPU 추론이나 워크로드는 실행하지 않는다.
- `up`: 서빙만 시작한다. 모델 파일 검사·CUDA 초기화·측정 외 CPU 입력 준비를 수행한다.
- `register`: 실제 GPU 모델·배치 프로파일을 새로 측정한다. Clipper는 별도 예측 프로파일을 대신 넣지 않고 native 온라인 배치 측정을 사용한다.

기존 HF/PyTorch 또는 FP16 프로파일은 재사용하지 않는다. 프로파일과 결과는 구현·브랜치별로 분리된다.
`cold`는 GPU 모델 세션 비상주를 뜻한다. 다운로드·CUDA 초기화는 제외하며 OS 파일 캐시는 초기화하지 않는다.
모델 로딩 비용은 ONNX 세션/장치 준비와 내부 워밍업을 포함한다. 요청 시 CPU 전처리와 모델 로딩의 새 오버랩은 추가하지 않았다.

## 4. 트레이스 실행

```bash
# 파일·설정 확인만 수행. 연결이나 추론 요청 없음.
bash experiments/resnet/run.sh "$ENGINE" trace --dry-run

# 처음 100건 확인이 필요할 때
bash experiments/resnet/run.sh "$ENGINE" trace --limit 100

# 100건을 실행했다면 전체 측정 전에 정책 상태도 다시 초기화
bash experiments/resnet/run.sh "$ENGINE" down
bash experiments/resnet/run.sh "$ENGINE" up

# 전체 실행: 이 명령 직전에 cold/active 조건을 다시 맞춘다.
bash experiments/resnet/run.sh "$ENGINE" trace --limit 0
bash experiments/resnet/run.sh "$ENGINE" state
bash experiments/resnet/run.sh "$ENGINE" down
```

기본 Twitter 목록은 공개 `zipf_exponential/300/resnet.txt` 원본이다.
**3,776건·2배속·145.154초·평균 26.01 QPS**이며, 도착 시각의 원래 0점을 보존한다.
QoS는 기록만 하고 단일 SLO 실험에서는 구분하지 않는다. 요청 수는 배속으로 늘어나지 않는다.
같은 JPEG를 반복하며, v1/v2에 같은 파일 SHA256이 기록됐는지 확인한다.

`WORKLOAD=uniform`, `gamma`, `bursty`로 바꾸면 해당 공개 ResNet 목록을 사용한다.
파일별 요청 수와 평균 QPS는 다를 수 있다. 모델 전환을 만들려고 정책을 변경하지 않는다.
2배속이 각 모델의 GPU 용량을 넘는다는 보장은 없으며, 전환 여부는 실제 결과로 확인한다.

## 5. v2와 다음 구현

현재 실험을 **현재 브랜치에서 `down`한 다음** 브랜치를 변경한다.

```bash
git fetch origin refs/heads/v2
git switch -c v2 FETCH_HEAD

bash experiments/resnet/run.sh "$ENGINE" build
bash experiments/resnet/run.sh "$ENGINE" setup
bash experiments/resnet/run.sh "$ENGINE" up
bash experiments/resnet/run.sh "$ENGINE" register
bash experiments/resnet/run.sh "$ENGINE" down
bash experiments/resnet/run.sh "$ENGINE" up
bash experiments/resnet/run.sh "$ENGINE" trace --limit 0
bash experiments/resnet/run.sh "$ENGINE" down
```

이미 `v2`가 있으면 `git switch v2` 후 `git merge --ff-only FETCH_HEAD`를 사용한다.
GPU를 비운 뒤 다음 구현을 같은 순서로 실행한다. 저장소·브랜치가 달라도 가중치와 로컬 설정값은 동일하게 유지한다.

## 6. 결과

위 기본 경로 기준:

```text
/data4tb/taeho-proteus/experiments/paired-resnet/<ENGINE>/<v1 또는 v2>/results/
  trace-<UTC시각>-<ID>.json
  trace-<UTC시각>-<ID>.csv
```

JSON: 전체 요청 수, 미전송, 200ms 내 성공, 늦은 성공, 실패·거절, E2E 평균/P50/P95/P99,
SLO 내 처리량, 모델·배치 분포, 실제 로딩 사건과 시간, 전후 정책 상태, 이미지 ID·소스 커밋·모델 해시.
CSV: 요청별 전송 여부·모델·배치·클라이언트/서버 지연·CPU/GPU 시간·실패 이유.

실패·거절은 SLO 미충족에 포함하고 미전송은 별도 집계한다. 지연 분위수는 성공 응답만의 값이다.
주 비교 지표는 클라이언트 E2E이며, 서버 내부 지연은 구현별 경계가 달라 보조 지표로 분리한다.
전송 누락이나 지연이 있으면 `trace_timing_valid`와 전송 지연 수치를 먼저 확인한다.
같은 GPU로 한 번에 한 구현을 실행하고, 비교 측정은 같은 설정으로 반복한다.

출처: [Proteus 논문](https://groups.cs.umass.edu/ramesh/wp-content/uploads/sites/3/2023/09/ljWiUS-asplos24spring-final29.pdf),
[공개 코드·트레이스](https://github.com/UMass-LIDS/Proteus/tree/6967c9327e66f6d6fe94993dcaa7521efc2f72f5),
[ONNX ResNet 모델](https://github.com/onnx/models/tree/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/classification/resnet).
