# 실험 설정 정리 — 기존 설정 보존 보정 패치

이 문서는 **docker/v4 보정 반영 버전의 코드 기본값**입니다.
GPU 서버의 개인 .local.env, 환경변수, 실제 실행 상태까지 확인한 결과가 아닙니다.
성능을 보고 스레드·배치·가드·제어 주기를 최적화하는 작업은 하지 않습니다.

## 1. 유지 원칙

- 유지: GPU/모델/가중치/입력/SLO/정책/배치 후보/제어 주기/warmup/프로파일 반복 횟수.
- 미지정 값 명시: PyTorch intra-op 2는 유지, inter-op 미지정 구현은 1로 명시.
  독립 Clipper의 inter-op 1은 이미 있던 값입니다.
- 요청된 구조 변경: CPU 전처리 → 준비된 CPU 텐서 큐 → 기존 모델·배치 정책 → GPU 실행.
  INFaaS는 원래 이 순서였으므로 기존 요청 처리 스레드 전처리를 유지합니다.
- 복구: INFaaS 단일 CPU 풀 제거, 비교 preset PROFILE_MODE=service 복구.
- 복구: Proteus MAX_QUEUE_SIZE는 실행 중 GPU 배치를 제외한 대기 요청 한도.
- 복구: 독립 Clipper에 새로 강제했던 256개 한도를 제거. MAX_QUEUE_SIZE=0은 원래처럼 명시적 제한 없음.
- 모델/VM 배치 정책 원본 소스는 수정하지 않습니다. Clipper의 최초 deadline 전달 어댑터는 유지합니다.
- 이전 가중치·실험 결과는 삭제하지 않습니다. 개인 설정도 기본 env 전체로 덮어쓰지 않습니다.

## 2. 공통 실행 조건

| 항목 | 조건 |
|---|---|
| GPU | 비교 preset: 호스트 GPU 2, Tesla V100 32GB 1대. 동시에 한 실험만 실행 |
| 추론 | FP32, GPU 실행. CPU 추론 대체 없음 |
| 입력 | 기존 JPEG 유지. 실험 JPEG SHA256: f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a |
| 전처리 | 서버 CPU의 PIL RGB + HF slow AutoImageProcessor |
| PyTorch CPU | intra-op 2 / inter-op 1. PIL·NumPy 전체 스레드 제한이나 CPU 코어 독점이 아님 |
| SLO | 기존 비교 설정 200ms. 전처리 후 마감시각을 다시 시작하지 않음 |
| ML 버전 | Python 3.11, Torch 2.11.0+cu126, torchvision 0.26.0+cu126, CUDA 12.6 |
| 라이브러리 | Transformers 5.19.0, Pillow 12.1.1, NumPy 2.2.6 |
| GPU 옵션 | TF32 비활성, cuDNN benchmark 비활성 |
| 저장소 | 현재 터미널에서 선택한 Docker daemon 사용. 공용 data-root 변경 없음 |

PREPROCESS_WORKERS=1은 Proteus 계열/독립 Clipper의 분리된 CPU 단계 기본값입니다.
INFaaS에는 적용하지 않습니다. 기존의 요청 처리 스레드가 전처리를 담당합니다.
작업자 수나 intra-op 값을 최적값이라고 검증한 것은 아닙니다.

## 3. 각 구현에서 유지하는 설정

| 구현 | 모델 조건 | GPU 배치 | 제어/갱신 주기 | 모델 준비 warmup |
|---|---|---|---|---|
| 독립 INFaaS | 기존 MODEL_NAME/등록 모델 유지, 비교 preset ResNet-50 | batch 1 고정 | 모델 1초 / VM 2초 | 10회 |
| Proteus | 기존 registry 유지. 코드 기본은 ResNet-50만 활성 | 후보 1,2,4,8 / accscale | 1초 | 10회 |
| 내부 INFaaS-Accuracy | 현재 GPU 포트는 singleton checkpoint | 후보 1,2,4,8 / infaas 정책 | 10초 | 10회 |
| 내부 Clipper-HA | 현재 GPU 포트는 singleton checkpoint | 후보 1,2,4,8 / AIMD | controller 값 10초, Clipper 배치 조절은 이벤트 기반 | 10회 |
| 내부 Clipper-HT | 현재 GPU 포트는 singleton checkpoint | 후보 1,2,4,8 / AIMD | controller 값 10초, Clipper 배치 조절은 이벤트 기반 | 10회 |
| Sommelier 비교 포트 | 기존 registry 유지. 기본 ResNet-50 활성/18 비활성 | 후보 1,2,4,8 / accscale | 10초 | 10회 |
| 독립 Clipper | 사용자가 지정한 단일 모델, 기본 ResNet-50 | BATCH_SIZE=-1 / native Adaptive Batching v2 | 고정 자원 재배치 주기 없음, 배치 피드백 기반 | 모델 로딩 후 내부 warmup 없음 |

| 구현 | 요청 처리/CPU 준비 | 큐 한도 |
|---|---|---|
| 독립 INFaaS | frontend 64 / worker RPC 32, 기존 요청 처리 스레드에서 CPU 준비 | 모델별 GPU 큐 크기 제한 미지정, 별도 단일 CPU 풀 없음 |
| Proteus·Sommelier | RPC 128 / 분리 CPU 작업자 1 | MAX_QUEUE_SIZE=256: CPU 준비 대기·진행 + ready 요청. 실행 중 GPU 배치는 제외 |
| 내부 기준선 3개 | RPC 128 / 분리 CPU 작업자 1 | MAX_QUEUE_SIZE=256: 기존처럼 전체 미완료 요청 |
| 독립 Clipper | C++ query 16 / 분리 CPU 작업자 1 / 모델 RPC 실행 흐름 유지 | MAX_QUEUE_SIZE=0: 기존처럼 명시적 한도 없음 |

큐 제한·warmup·제어 주기 차이는 각 구현의 기존 조건이므로 무조건 같은 값으로 바꾸지 않습니다.
무제한 큐는 과부하에서 메모리 증가 가능성이 있습니다. 신규 제한을 자동으로 추가하지 않았습니다.
DEADLINE_GUARD_MS=2, EWMA_ALPHA=0.7, INITIAL_RATE=1은 해당 Proteus/Sommelier 기본값을 유지합니다.

## 4. 모델과 정확도 설정

- pinned ResNet-50: microsoft/resnet-50 @ 34c2154c194f829b11125337b98c8f5f9965ff19.
- 두 모델 실험의 ResNet-18: microsoft/resnet-18 @ 65a5785d9156231087c481e0c7dd33a5ff6f7e3e.
- 서버의 기존 models.two-model-reference.json과 enabled 설정을 그대로 유지합니다.
  이 파일과 trace_replay.py/replay-trace.sh/workloads는 서버 미추적 파일이며 이번 패치가 추가·덮어쓰지 않습니다.
- 77.36 / 70.93은 사용자가 넣은 Proteus 원본 reference 점수입니다.
  현재 HF checkpoint의 정확도를 직접 측정한 값이 아니고, 코드 기본 registry의 accuracy는 null입니다.
- HA/HT/INFaaS-Accuracy의 현재 singleton GPU 포트는 이 패치만으로 다중 모델 선택 기능이 생기지 않습니다.
- ResNet-18/50은 processor 설정이 같아 공유 텐서 큐를 사용할 수 있습니다.
  전처리 규격이 다른 후보를 추가하면 별도 설계가 필요합니다.

## 5. 프로파일과 측정 정의

| 구현 | 유지/구분하는 측정 |
|---|---|
| INFaaS 비교 preset | 기존 service 모드 복구. JPEG 디코딩·전처리·GPU·후처리 포함 평균 지연, 실측 포화 QPS. 30회 / 포화 측정 5초·동시성 4 유지 |
| INFaaS 일반 기본 env | 기존 original 옵션 유지. 비교 preset과 다르므로 실제 선택 env를 확인 |
| Proteus·내부 기준선·Sommelier | GPU 준비 텐서 기준 pack+H2D+forward+D2H/sync, 배치별 P50/P95, 30회 |
| 독립 Clipper | 기존 런타임 배치 실행 피드백. 사전 register 프로파일 표를 강제로 추가하지 않음 |

Proteus 계열의 GPU 구간 프로파일은 전처리 완료 큐에서 **앞으로 남은 실행시간**을 예측하기 위한
요청된 파이프라인 분리의 일부입니다. 기존 CPU 포함 프로파일과 측정 경계까지 같다고 부르면 안 됩니다.
P50을 P95로 바꾸거나 guard·SLO/2 조건을 튜닝하지 않았습니다.
필요한 재프로파일링은 새 구조의 시간을 측정하는 것이지, 최적 설정을 탐색하는 실험이 아닙니다.

CPU 준비/ready 대기/GPU 서비스는 따로 기록합니다. 최종 비교 지표는 전체 e2e와 SLO 미달/거절을 포함합니다.
예측값이 바뀌어 실제 선택 모델/배치가 달라질 수 있으므로, 후보 목록이 같다고 실행 결과까지 같지는 않습니다.

## 6. 실행 전 확인

1. 기존 env로 현재 본인 서비스만 down. 모델 캐시/결과는 삭제하지 않습니다.
2. docker/v4에서 `git pull --ff-only origin docker/v4`로 갱신. 개인 .local.env는 자동으로 덮어쓰지 않습니다.
3. INFaaS 비교 env: PREPROCESS_WORKERS 삭제, PROFILE_MODE=service, 새 프로파일/로그 디렉터리 선택.
4. 독립 Clipper env: 앞선 안내로 MAX_QUEUE_SIZE=256을 추가했다면 0으로 복구.
5. 기존 GPU·모델·SLO·배치 후보·주기·warmup은 유지하고 settings 출력 확인.
6. build → doctor → up. 프로파일을 쓰는 구현은 register 재실행 후 요청/trace 재생.
7. 서버에서 실제 적용 값을 확인한 뒤 실험 조건을 확정합니다.

코드의 기본값 표만 보고 서버의 effective setting이 같다고 판단하지 마세요.
같은 JPEG·기존 trace 시각/speed/seed/요청 수를 유지하고, 원래 cold/warm 분리도 유지합니다.
기존 replay-trace.sh는 Proteus 대상입니다. 다른 endpoint용 trace 클라이언트는 이번 패치에서 새로 만들지 않았습니다.
