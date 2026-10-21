# CPU 전처리와 GPU 실행 분리

`docker/v2`에서 분기한 `docker/v4`의 Proteus·INFaaS-Accuracy·Clipper-HA·HT·Sommelier GPU 경로에 적용합니다.
`docker/v4` 기준 보정 패치에서는 기존 큐 한도 의미를 복구했습니다.
전체 실험 설정과 유지/변경 범위는 [EXPERIMENT_SETTINGS.md](EXPERIMENT_SETTINGS.md)에 정리했습니다.

## GPU 서버에서 코드 받기

현재 실행 중인 본인 실험은 기존 설정으로 먼저 `down`합니다. 아래 `git status`에서 기존 수정이
보이면 보존한 뒤 진행하세요. 체크아웃 충돌을 강제 해제하지 않으며 `.local.env`와 가중치는 유지합니다.

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
git status --short
git fetch origin refs/heads/docker/v4
git checkout -b docker/v4 FETCH_HEAD
```

이미 이 로컬 브랜치를 만들었다면 마지막 두 명령 대신 다음을 사용합니다.

```bash
git checkout docker/v4
git pull --ff-only origin docker/v4
```

단일 브랜치 clone/구형 Git에서도 쓸 수 있도록 최초 생성에는 `FETCH_HEAD`를 사용합니다.

```text
JPEG 수신(최초 마감시각 설정)
 → CPU 대기 → PIL RGB + HF slow AutoImageProcessor
 → 준비된 CPU 텐서 큐 → 기존 모델·배치 정책
 → 텐서 결합·H2D·GPU forward·D2H → 응답
```

- 앞 요청의 GPU 실행/모델 준비와 다음 요청의 CPU 전처리가 겹칠 수 있습니다.
- CPU 전처리가 끝나도 SLO를 다시 시작하지 않습니다. 대기·전처리·로딩·실행을 모두 포함합니다.
- 기존 MILP·accscale·INFaaS-Accuracy·AIMD 정책은 유지합니다. 논문 원본 시뮬레이터를 수정한 것은 아닙니다.
- Proteus의 `MAX_QUEUE_SIZE=256`은 기존처럼 **대기 중인 요청** 한도입니다.
  분리된 CPU 준비 대기/진행 + ready 큐를 포함하고, 이미 실행 중인 GPU 배치는 제외합니다.
  내부 기준선 3개는 기존처럼 전체 미완료 요청을 계산합니다. Sommelier 포트는 Proteus와 같습니다.
- 여러 모델을 바꿔 쓸 때는 전처리 설정이 같아야 합니다. 다르면 시작 단계에서 거부합니다.
  현재 pinned ResNet-18/50은 동일 설정입니다. 전처리 결과를 모델 변경 후 잘못 재사용하지 않게 검사합니다.

## 개인 설정 변경

기존 `.local.env`의 모델·가중치·데이터 경로는 보존하고 아래 값만 추가/갱신합니다.
기존 설정으로 먼저 해당 서비스의 `down`을 실행한 뒤 변경하세요.

```dotenv
CPU_THREADS=2
CPU_INTEROP_THREADS=1
PREPROCESS_WORKERS=1
```

`CPU_THREADS`는 PyTorch intra-op 스레드 수, `PREPROCESS_WORKERS`는 동시 CPU 준비 작업 수입니다.
CPU 코어 2개를 독점한다는 뜻이 아니며, 기본 1 작업자로도 CPU/GPU 단계는 겹칩니다.
기존에도 배치 실행 경로의 CPU 준비는 작업자 1개가 담당했습니다. 이 수를 늘리는 튜닝은 하지 않습니다.

이미지 태그도 바꿔 이전 이미지를 보존합니다.

| 실행 종류 | 설정 파일의 WORKER_IMAGE |
|---|---|
| Proteus | `proteus-worker:cpu-ready-v1` |
| INFaaS-Accuracy·Clipper-HA·HT | `proteus-baseline-worker:cpu-ready-v1` |
| Sommelier | `proteus-baseline-worker:sommelier-cpu-ready-v1` |

## 재실행

현재 터미널의 Docker를 그대로 사용합니다. `docker info`로 개인 data-root를 확인하고,
GPU 2의 다른 본인 실험을 종료한 뒤 한 종류씩 실행합니다. 가중치는 기존 캐시를 재사용합니다.

```bash
bash deploy/docker/proteus-docker.sh settings
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh doctor
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-18 resnet-50 --trials 30
bash deploy/docker/proteus-docker.sh state
```

위 두 모델은 registry에서 모두 enabled일 때만 지정하세요. 단일 모델이면 `--models resnet-50`입니다.
기존 `replay-trace.sh`는 같은 이미지/trace/SLO로 다시 실행합니다.

내부 기준선은 기존 실행 스크립트를 그대로 씁니다. 한 종류 완료 후 `down`하고 다음 종류를 실행합니다.

```bash
bash deploy/docker/baseline-docker.sh infaas-accuracy build
bash deploy/docker/baseline-docker.sh infaas-accuracy doctor
bash deploy/docker/baseline-docker.sh infaas-accuracy up
bash deploy/docker/baseline-docker.sh infaas-accuracy register --trials 30
bash deploy/docker/baseline-docker.sh infaas-accuracy test --warm-requests 10
bash deploy/docker/baseline-docker.sh infaas-accuracy down
```

`infaas-accuracy`를 `clipper-ha`, `clipper-ht`, `sommelier`로 바꿔 실행합니다.
각 기준선의 모델 후보/정책 범위는 기존과 같습니다. INFaaS-Accuracy·HA·HT의 현재 GPU 어댑터는
단일 checkpoint 조건이며 이 변경만으로 다중 모델 선택 기능이 추가되지는 않습니다.

## 프로파일과 결과 확인

- **반드시 register 재실행**: 이전 전처리 포함 프로파일을 재사용하면 안 됩니다.
- 새 프로파일은 CPU 전처리를 제외한 `pack + H2D + forward + D2H/sync`입니다.
  CPU 준비는 별도로 `cpu_preparation_ms`에 기록합니다. 프로파일과 실제 GPU 실행 스레드를 맞췄습니다.
  이 측정 경계 변경은 전처리가 끝난 큐의 **남은 실행 시간**을 예측하기 위한 파이프라인 변경의 일부입니다.
  기존 전처리 포함 프로파일과 동일한 측정이라고 부르지 않습니다. P50 선택·반복 30회·후보 크기는 유지합니다.
- Proteus: `DATA_DIR/profiles/cpu-ready-v1/profiles.json`.
- 기준선: `DATA_ROOT/<기법>/profiles/cpu-ready-v1/profiles.json`.
- 이전 프로파일·가중치·결과는 삭제하지 않습니다. 새 fingerprint와 경로로 구분합니다.
- `state`: `pipeline=cpu-ready-v1`, `cpu_gpu_pipeline=true`, CPU/ready 큐 확인.
- 요청 `timings`: `worker.cpu_queue`, `worker.cpu_service`, `worker.ready_queue`, `worker.gpu_service`.
  `worker.service`는 CPU 준비 + GPU 서비스이며, 새 GPU 프로파일과 비교할 값은 `worker.gpu_service`입니다.

CPU 병목·모델 로딩·실행시간 예측 오차와 응답 오버헤드는 남습니다. 파이프라인 분리 자체가
SLO 보장이나 처리량 개선을 보장하지 않습니다. 기존 전처리 포함 실험과 새 수치는 별도 버전으로 보고합니다.
