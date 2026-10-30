# 휴지 후 실행을 반영한 Proteus 프로파일링

## 무엇을 고치나

CPU의 동적 주파수 조절은 정상적인 전력·성능 관리 동작입니다. 이번 측정에서는
같은 GPU 실행 스레드의 명령어 수는 거의 같지만, 1초 간격 실행의 상대 주파수 지표가
연속 실행의 약 51~53%로 낮았고 실행시간은 약 7ms에서 14~15ms로 늘었습니다.
이 관측은 CPU 주파수 변화가 주요 원인이라는 강한 근거이며, 특정 governor만이 유일한 원인이라는 증명은 아닙니다.

원본 `core/simulator.py`는 성능표의 `50th_pct`를 읽고, `core/predictor.py`는
이 값을 배치 처리시간으로 사용합니다. 시뮬레이터가 실제 CPU의 주파수를 재현하는 것은 아닙니다.
이번 수정은 추가한 **실시간 GPU 포트의 프로파일링**이며 원본 시뮬레이터는 수정하지 않습니다.

## 측정과 사용 값

| 용도 | 값 |
|---|---|
| 처리용량 계산 | 연속 실행의 P50. 기존 `p50_ms` 유지 |
| 배치 실행시간 예측 | 기본은 조건별 P50 중 최댓값. `prediction_ms`에 별도 저장 (P95 선택 가능) |
| SLO/2 실행시간 조건 | 위 실행시간 예측 사용 |
| 실제 배치 대기 | 기존 `요청 마감 - 실행시간 예측 - DEADLINE_GUARD_MS` 유지 |

모델별·배치별로 준비된 CPU 텐서를 사용해 측정합니다.
각 조건에서 추가 워밍업 3회 후 30회 실행하며, 휴지 조건은 첫 측정 호출 전에도 기다립니다.
`0,100,1000`은 **직전 호출 완료 후 다음 호출까지 기다릴 시간(ms)**이지 요청률이 아닙니다.
대기시간은 실행시간 타이머 밖이므로 1초를 추론 지연에 더하지 않습니다.

측정 범위는 기존과 같은 **배치 결합 + H2D + 모델 실행 + 결과 CPU 전달·동기화**입니다.
CPU 전처리·큐 대기·RPC·모델 로딩은 포함하지 않습니다. 서빙 전체 응답시간과 구분하세요.
측정은 실제 GPU 실행 스레드에서 수행하며 학습·GPU 모의 실행이 아닙니다.
일반 요청은 `register` 중 차단됩니다. 같은 JPEG를 다시 전처리하는 방식으로 바꾸지 않습니다.

## 새 설정

```dotenv
WORKER_IMAGE=proteus-worker:idle-profile-v1
PROFILE_STRATEGY=idle-aware-v1
PROFILE_IDLE_GAPS_MS=0,100,1000
PROFILE_PREDICTION_PERCENTILE=50
PROFILE_TIMEOUT_S=900
```

- 기본 `PROFILE_STRATEGY=continuous`는 기존 방식과 연속 P50 예측을 유지합니다.
- 기본 `PROFILE_PREDICTION_PERCENTILE=50`은 기존 통계값을 유지한 채 측정 조건만 추가하기 위한 선택입니다.
  이는 기존 연속 P50과도 다릅니다. `95`를 별도로 선택하면 조건별 P95의 최댓값을 사용합니다.
  P95는 실행시간 꼬리를 일부 반영하지만 별도 실험으로 구분해야 하며 SLO 보장은 아닙니다.
- 새 설정은 모델·GPU·SLO·CPU 스레드·작업자 수·큐 제한·제어 주기·guard를 바꾸지 않습니다.
- CPU governor/주파수/perf 권한을 변경하지 않고, sudo도 필요하지 않습니다.
- INFaaS-Accuracy·Clipper-HA·HT와 독립 INFaaS·Clipper는 이번에 활성화하거나 수정하지 않습니다.
  Sommelier는 공유 Profile RPC의 timeout 인자 호환만 유지합니다. 기법 간 최종 비교는 프로파일 조건도 통일해야 합니다.

## 서버에서 재실행

패치를 적용한 뒤 저장소 루트에서 실행합니다. 기존 개인 env와 모델 registry를 유지합니다.
아래 export는 현재 터미널에만 적용됩니다. 검증 후 같은 항목을 개인 env에 저장하세요.

```bash
export PROTEUS_ENV="$PWD/deploy/docker/proteus-docker.local.env"
export WORKER_IMAGE=proteus-worker:idle-profile-v1
export PROFILE_STRATEGY=idle-aware-v1
export PROFILE_IDLE_GAPS_MS=0,100,1000
export PROFILE_PREDICTION_PERCENTILE=50
export PROFILE_TIMEOUT_S=900

bash deploy/docker/proteus-docker.sh settings
bash deploy/docker/proteus-docker.sh down
bash deploy/docker/proteus-docker.sh build
bash deploy/docker/proteus-docker.sh up
bash deploy/docker/proteus-docker.sh register --models resnet-18 resnet-50 --trials 30 --timeout 900
bash deploy/docker/proteus-docker.sh profile-summary
```

2개 모델·4개 배치·30회·0/100/1000ms 조건이면 **휴지시간만 총 264초**입니다.
모델 로딩·워밍업·실행까지 포함하면 대략 5분 이상 걸릴 수 있습니다.
SLO 200ms는 요청의 제한이고 `--timeout 900`은 관리용 Profile RPC의 제한입니다. 둘은 다릅니다.

## 보존·확인

- 기존: `DATA_DIR/profiles/cpu-ready-v1/profiles.json` (새 방식으로 덮어쓰지 않음).
- 새 방식: `DATA_DIR/profiles/cpu-ready-idle-v1/profiles.json`.
- 같은 새 파일을 재측정하면 이전 파일을 `profiles.before-register-<고유값>.json`으로 백업합니다.
- 측정 또는 저장 실패 시 이전 메모리 프로파일·원본 파일을 유지합니다.
- 측정 전략·휴지 조건·통계값이 달라진 캐시는 거부하고 `register`를 요구합니다.
- 원본 샘플과 조건별 P50/P95는 프로파일과 `results/profile-*.json`에 보존합니다.
  터미널에는 모델·배치별 요약만 출력합니다.
- 실제 응답의 `timings`에는 `worker.predicted_gpu_service`,
  `worker.profile_continuous_p50`, `worker.gpu_prediction_error`가 추가됩니다.
  실제 `worker.gpu_service`와 같은 실행 범위인지 확인할 수 있습니다.

## 검증 순서와 한계

1. `profile-summary`: 연속 값과 새 예측값이 구분되는지 확인합니다. 14ms를 하드코딩하지 않습니다.
2. `mode --mode fixed`에서 1QPS로 실제 `worker.gpu_service`와 예측값을 비교합니다.
3. `mode --mode proteus`에서 1QPS, 일정한 13QPS를 같은 조건으로 비교합니다.
4. 전체 트레이스: SLO 충족률·처리량·실제 모델/배치·미전송 요청을 확인합니다.

새 프로파일은 지연 예측 보정입니다. 실제 서빙의 모든 실행 상태를 재현하거나 미래 지연을 보장하지 않습니다.
현재 2ms guard는 바꾸지 않으므로 응답 전달 지연·OS 타이머 오차로 일부 SLO 위반이 남을 수 있습니다.
CPU 전처리 병목·순간 과부하·모델 전환 비용도 별도입니다. 평균 QPS가 낮아도 순간 부하는 높을 수 있습니다.
라이브러리/가중치/GPU/CPU 설정을 바꾸면 다시 측정해야 합니다.

기존 방식으로 돌아갈 때는 `PROFILE_STRATEGY=continuous`와 기존 이미지 태그로 `up`하세요.
개인 설정 파일이나 다른 터미널의 export에 새 값이 남았는지는 `settings`로 확인합니다.
