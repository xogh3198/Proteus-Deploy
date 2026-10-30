# Proteus E2E SLO · 별도 RPC 시간 예산

실제 GPU 포트의 선택형 보정입니다. 원본 Proteus 시뮬레이터 기능으로 설명하지 않습니다.

## 변경 범위

`RPC_BUDGET_MS`는 **클라이언트 E2E − 워커 내부 지연**에 해당하는 추가 시간의 고정 예산입니다.
요청 전달·직렬화·응답 반환·워커 계측 밖의 작업을 포함하며, 순수 네트워크 시간만은 아닙니다.

| 설정 | 기존 동작 | 이번 검증 |
|---|---:|---:|
| `SLO_MS` | 200 | 200 |
| `DEADLINE_GUARD_MS` | 2 | 2 |
| `RPC_BUDGET_MS` | **0 (기본값)** | **6 (선택값)** |
| 워커에 배정하는 예산 | 200ms | 194ms |

계산은 다음과 같습니다.

```text
워커 처리 예산 = 요청 SLO − RPC_BUDGET_MS
배치 실행 시작 시점 = 워커 요청 접수 시점 + 워커 처리 예산 − GPU 실행 예측 − 기존 guard
MILP 후보의 실행시간 제한 = 워커 처리 예산 / 2
```

- 정확도 목적함수·연속 실행 기반 처리 용량·GPU 실행 함수·CPU 파이프라인·큐 제한은 유지합니다.
- 이미 경과한 CPU 전처리와 대기는 마감 예산을 소비합니다. 전처리 후 타이머를 새로 시작하지 않습니다.
- `fixed` 진단 모드는 즉시 배치 1 실행을 유지하며, 별도 예산으로 대기시키지 않습니다.
- GPU 실행 프로파일에 RPC 시간이나 배치 대기를 더하지 않습니다. 프로파일 파일·fingerprint도 바꾸지 않습니다.
- 이 방식은 **왕복 추가 시간 전체를 한 번 차감하는 근사 방식**입니다. 절대 클라이언트 마감시각을 전달하거나, 호스트 시계 동기화를 가정하지 않습니다.
- 향후 클라이언트의 남은 마감시간을 직접 전달한다면 이 총예산을 그대로 중복 차감하면 안 됩니다.
- 모델 로딩을 미리 예측하는 새 정책이나 원본 정책의 전면 재현은 이번 수정에 포함하지 않습니다.

6ms는 2026-10-26의 1QPS·30건에서 측정된 추가 시간 P95 **5.806ms**를 반올림한 **초기 검증값**입니다.
동일 조건에서 새 실험으로 검증하고, 다른 부하·입력 크기에서는 다시 확인합니다. 모든 부하에서 위반 0건을 보장하지 않습니다.

## 적용과 실행

기존 개인 env의 모델·GPU·경로는 유지하고, 같은 터미널에서 아래 값만 지정합니다.

```bash
export PROTEUS_ENV="$PWD/deploy/docker/proteus-docker.local.env"
export WORKER_IMAGE=proteus-worker:rpc-budget-v1
export PROFILE_STRATEGY=idle-aware-v1
export PROFILE_IDLE_GAPS_MS=0,100,1000
export PROFILE_PREDICTION_PERCENTILE=50
export SLO_MS=200 DEADLINE_GUARD_MS=2 RPC_BUDGET_MS=6

bash deploy/docker/proteus-docker.sh settings
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh up
```

`up`은 같은 프로젝트 워커를 새 이미지/설정으로 재생성합니다. 다른 프로젝트 컨테이너는 종료하지 않습니다.
현재 터미널에서 개인 Docker가 선택되어 있는지 먼저 확인합니다. data-root/context를 스크립트가 바꾸지 않습니다.
이미 유효한 idle-aware 프로파일이 있으면 다시 `register`하거나 가중치를 다시 다운로드할 필요가 없습니다.
`policy_ready: true`, 기존 프로파일 경로, `deadline_budget.rpc_budget_ms: 6`을 확인합니다.
`policy_ready: false`이면 먼저 오류를 확인하며, `fixed`로 우회하거나 자동으로 프로파일을 덮어쓰지 않습니다.

```bash
bash deploy/docker/proteus-docker.sh bench \
  --rate 1 --duration 30 --distribution uniform --seed 1

bash deploy/docker/proteus-docker.sh bench-summary
```

`bench` 터미널 출력은 요약만 보여주고, 전체 runtime/MILP 기록은 기존처럼 결과 JSON에 저장합니다.
`bench-summary`는 가장 최근 `bench-*.csv`를 선택하고 파일명을 출력합니다. 특정 결과는 다음처럼 지정합니다.

```bash
bash deploy/docker/proteus-docker.sh bench-summary \
  --file data/proteus/results/bench-20261026T074633.csv
```

분석 명령은 CSV/JSON을 변경하지 않습니다. Docker/GPU를 사용하지 않고 예전 결과도 재집계할 수 있습니다.
서버 시간이 누락된 성공 응답은 0ms/성공으로 처리하지 않으며, 해당 서버 SLO 비율은 `null`로 표시합니다.

## 지표 해석

| 결과 키 | 범위 |
|---|---|
| `slo_metrics.client_e2e` | RPC 호출 직전 → 응답 수신 완료. 최종 서빙 비교의 주지표 |
| `slo_metrics.server_internal` | 워커 접수 → GPU 추론 완료. 전체 프런트엔드 지연과 다름 |
| `slo_metrics.rpc_and_unmeasured_extra` | 같은 요청의 E2E − 워커 내부 지연 |
| `slo_metrics.gpu_prediction_error` | 실제 GPU 실행 − 예측 실행시간 |
| 기존 `slo_met`, `slo_miss_ratio_among_replies` 등 | **기존과 동일하게 클라이언트 E2E**, 의미를 바꾸지 않음 |
| `server.budget_missed`, `server_budget_met` | 줄어든 워커 예산(이번 예: 194ms) 충족 여부. 200ms SLO 지표와 별개 |

두 범위 모두 요청별 SLO를 사용합니다. `slo_unmet_ratio_of_all_requests`에는 실패·거절도 포함합니다.
지연 통계는 성공 응답 기준이며, 요청을 아예 전송하지 못한 경우는 `client_skipped_scheduled_arrivals`로 별도 확인합니다.
이 분석 모듈은 트레이스의 미전송 행을 전체 부하에 대한 서버 SLO로 재해석하지 않습니다.
정확도는 반복 JPEG의 정답률로 측정하지 않으며, 이번 패치에서 점수를 새로 만들지 않습니다.

비교는 `RPC_BUDGET_MS=0` 대 `6`을 같은 이미지·프로파일·SLO·부하에서 새로 실행하고,
각각 3회 이상 반복한 뒤 13QPS와 전체 트레이스로 확장합니다. 검증 결과를 보고 별도 실험에서 예산을 확정합니다.
6ms를 보정값으로 택한 원래 CSV는 보존하며, 그 CSV 자체를 수정 후 검증 결과라고 보고하지 않습니다.

## 다른 실험과의 관계

이번 활성 예산 설정은 **독립 Proteus GPU 워커**에만 연결합니다.
공유 클라이언트를 재사용하는 내부 기준선의 결과에는 두 범위 지표가 추가될 수 있으나, 정책/컨테이너 예산은 자동으로 켜지지 않습니다.
독립 INFaaS/C++ Clipper 저장소와 외부 `trace_replay.py`/`replay-trace.sh`는 수정하지 않습니다.
트레이스의 요청은 같은 워커를 통과하므로 예산은 적용되지만, 기존 트레이스 요약 키는 그대로 E2E 기준입니다.
서버별 계측 시작·종료 지점과 동일한 보정 적용 여부를 맞추기 전에는 이를 전체 시스템 공정 비교 완료라고 표현하지 않습니다.

## 원래 동작으로 비교

```bash
export RPC_BUDGET_MS=0
bash deploy/docker/proteus-docker.sh up
```

동일 이미지와 프로파일로 예산 차감만 끕니다. `DEADLINE_GUARD_MS=2`는 그대로입니다.
테스트를 마친 뒤 계속 사용하려면 개인 `.local.env`에 `RPC_BUDGET_MS=6`을 명시합니다.
