# CPU 전처리 공유 잠금·만료 처리 보정

idle-aware 프로파일링과 RPC 예산 패치가 적용된 Proteus GPU 포트용입니다.
원본 논문의 추가 기법이 아니라, 우리가 추가한 실제 GPU 실행 경로의 보정입니다.

## 변경 범위

- CPU 스레드마다 같은 설정의 HF slow 이미지 처리기를 한 번 복제해 사용합니다.
  RGB 디코딩·resize/crop·정규화·입력 크기·FP32·체크포인트는 바꾸지 않습니다.
- 기본 서버는 `EARLY_DEADLINE_DROP=1`로 CPU/준비 큐의 만료 요청을 반환합니다.
  CPU 시작 직전과 완료 후에도 같은 마감을 확인합니다.
- 마감은 기존과 동일하게 `worker arrival + SLO - RPC_BUDGET_MS`입니다.
  CPU 준비 후 시간을 새로 시작하지 않고, RPC 예산을 중복 차감하지 않습니다.
- 실행 중인 CPU 함수나 GPU 배치를 강제로 중단하지 않습니다.
  만료된 CPU 함수가 나중에 완료돼도 결과를 GPU 큐에 넣지 않습니다.
- CPU 함수가 실제로 끝나기 전에는 reset/register 같은 관리 작업을 허용하지 않습니다.
- 유효한 요청의 FIFO 순서를 유지하고, 만료된 앞 요청만 제거합니다.
- 거절은 여전히 `ok=false`, `deadline_drop`입니다. 빠른 거절을 성공으로 집계하지 않습니다.

MILP 목표·요청률 계산·적응형 배치 계산·CPU 작업자 수·CPU 스레드 설정·SLO·RPC
타임아웃·클라이언트 동시 요청 한도는 바꾸지 않습니다. 기존 GPU 프로파일의
측정 범위와 식별 정보도 유지합니다. 이번 보정만으로 프로파일 재등록은 필수가 아닙니다.
서빙 중 GPU 예측 오차는 별도로 확인해야 합니다.

`EARLY_DEADLINE_DROP=0`은 조기 만료를 끕니다. 새 이미지의 전처리기는 여전히
스레드별 객체이므로, 완전한 이전 방식과 비교하려면 보존한 이전 이미지를 사용하세요.
fixed 진단 모드는 조기 만료를 적용하지 않고 지연을 그대로 측정합니다.

## 공유 코드·비교군 주의

배포 대상은 Proteus의 새 이미지 하나입니다. 독립 INFaaS·Clipper 저장소와
실행 이미지는 수정하지 않습니다. Proteus 내부 기준선은 기본적으로 조기 만료를
활성화하지 않습니다. 향후 내부 기준선 이미지를 다시 빌드하면 공통
`GpuRuntime`의 스레드별 전처리기는 함께 사용됩니다. 비교할 때 버전·CPU 조건을 기록하세요.
공유 소스의 검증 해시는 갱신하지만 원본 정책 검사 자체를 우회하지 않습니다.

같은 요청의 전처리·모델 로딩 오버랩, 클라이언트 절대 마감 전파,
CPU 처리 용량을 반영하는 새로운 모델 선택 정책은 이번 범위가 아닙니다.

## 서버 적용 후 실행

기존 세 모델·개인 Docker·경로 설정을 유지한 같은 터미널에서 실행합니다.
새 태그를 써서 기존 `proteus-worker:rpc-budget-v1` 이미지는 보존합니다.

```bash
export WORKER_IMAGE=proteus-worker:cpu-deadline-v1
export EARLY_DEADLINE_DROP=1
export PREPROCESS_WORKERS=4
export CPU_THREADS=2 CPU_INTEROP_THREADS=1
export SLO_MS=200 DEADLINE_GUARD_MS=2 RPC_BUDGET_MS=6

bash deploy/docker/proteus-docker.sh down &&
bash deploy/docker/proteus-docker.sh build &&
bash deploy/docker/proteus-docker.sh preprocess-check &&
bash deploy/docker/proteus-docker.sh up
```

`preprocess-check`는 GPU·인터넷·가중치 로딩 없이 로컬 전처리 설정과 JPEG를 읽어,
기존 직렬 결과와 병렬 결과의 CPU 텐서를 `torch.equal`로 비교합니다.
`equivalent: true`를 확인하세요. 지연시간이나 GPU 성능 검사는 아닙니다.

state에서 다음을 확인하세요.

- `preprocess_processor_ownership: thread-local-v1`, `preprocess_shared_lock: false`
- `early_deadline_drop: true`, `early_deadline_drop_active: true`
- `preprocess_workers: 4`, `policy_ready: true`, `policy_error: null`
- `deadline_budget.rpc_budget_ms: 6`, `server_budget_ms: 194`

같은 4배속·작업자 4개 조건을 먼저 재측정합니다. 클라이언트 옵션에 `--slo`를
추가하지 마세요. 래퍼가 `SLO_MS`를 읽습니다.

```bash
bash replay-trace.sh --limit 0 --speed 4 --concurrency 256 --timeout 10
```

전송 3,776건·미전송 0·실제 전송 지연부터 확인한 다음, 실패/거절을 포함한 SLO
미충족률과 SLO 내 성공 건수를 비교하세요. 큐 정리는 CPU 용량을 무한히 늘리지
않으며, SLO 개선이나 자동 모델 전환을 보장하지 않습니다.
이번 패치에는 실제 V100 성능 측정 결과가 포함되지 않습니다.

## 거절 시점 계측

state의 `deadline_drop_stages`는 `cpu_queue`, `cpu_service`, `ready_queue`,
`before_cpu`, `after_cpu`, `batch_prediction`별 거절 수입니다.
응답의 `worker.rejection_stage.<단계>=1`로 요청별 분석이 가능합니다.

`worker.cpu_queue`, `worker.cpu_service`, `worker.ready_queue`는 **거절 시점의
스냅샷**입니다. `worker.cpu_started=0`이면 CPU 실행 전입니다.
`worker.cpu_service_complete=0`이면 전처리가 끝난 실측 시간이 아니라 해당
시점까지의 경과시간입니다. 성공 요청의 완결된 service 표본과 섞지 마세요.
거절된 요청의 GPU 실행·로딩은 0, 배치는 0이며 추론 모델/정확도를 부여하지 않습니다.

`server_slo_misses`는 기존처럼 완료된 GPU 응답의 시간 초과 카운터입니다.
거절·미전송을 포함한 미충족률은 클라이언트 결과에서 확인해야 합니다.
RPC 앞단 대기는 여전히 worker 시간 밖에 있습니다. E2E에서 worker 시간을 뺀 값은
순수 네트워크 시간이 아닙니다.
