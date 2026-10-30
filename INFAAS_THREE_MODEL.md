# INFaaS-Accuracy: 세 모델과 2배속 트레이스

기존 Proteus·단일 모델 기준선을 보존하는 선택적 GPU 어댑터 확장입니다.
원본 core/executor.py, core/predictor.py, algorithms/ilp.py는 수정하지 않습니다.

## 설정

| 항목 | 값 |
|---|---|
| GPU | 호스트 GPU 2, V100 1대, FP32 |
| 후보 | ResNet-18 / 50 / 101, Proteus와 같은 고정 HF revision |
| 사전 정확도 | 70.93 / 77.36 / 78.34, 현재 체크포인트 실측 아님 |
| 전처리 | PIL RGB + HF slow → CPU 텐서 → 원본 정책 큐 → GPU |
| CPU | intra-op 2, inter-op 1, 전처리 작업자 8 |
| SLO | 클라이언트 E2E 200ms, 실패·거절도 미충족 |
| 배치 | 후보 1,2,4,8, 원본 INFaaS 배치 제어 |
| 정책 주기 | 기존 비교군의 10초 유지 |
| 프로파일 | 각 후보·배치 GPU 구간 연속 실행 30회, CPU 준비 시간 별도 |
| 워크로드 | 기존 동일 3,776건, 2배속 145.154초, 평균 26.014 QPS |
| 클라이언트 | 동시 요청 256, RPC 타임아웃 10초, 동일 JPEG |

Proteus의 idle-aware 예측, RPC 예산 6ms, 조기 만료 패치를 이 정책에 자동 적용하지
않습니다. 독립 INFaaS는 요청 핸들러 32개에서 전처리하는 기존 구조를 유지하며,
이 비교군의 전처리 풀 8개와 동시성 구조가 다릅니다. 내부 설정 전체가 같다는 뜻은 아닙니다.

## 원본 정책의 실제 동작

- 초기 후보는 원본 Executor.add_predictor()가 최저 사전 정확도 모델로 정합니다.
- 원본 trigger_infaas_v2_upscaling() / trigger_infaas_v2_downscaling()을 호출합니다.
- 현재 공개 downscaling 코드에는 least_cost_found 선언 후 후보 비용을 검색하는 루프가
  없고, selected_variant가 None이면 최저 정확도 후보를 선택하는 분기가 있습니다.
  따라서 세 후보를 주어도 ResNet-18 유지가 나올 수 있습니다. 정확도 최대화나
  SLO 위반율 기반 정책으로 대체하지 않았습니다.
- GPU 슬롯은 하나입니다. GPU 실행 중 제어 주기가 오면 적용을 배치 완료까지 미룹니다.
  교체 시 큐 요청의 기존 도착시각·마감·ID를 유지하여 넘깁니다.
- model_selection_history는 선택 기록, model_switch_events는 실제 로딩 기록입니다.
  model_switches_after_initial_load는 서빙 중 실제 교체 횟수입니다. 프로파일 로딩 제외.
- 로딩 비용을 지연에서 제거하거나 후보를 모두 GPU에 미리 상주시키지 않습니다.

## 실행

기존 Proteus와 다른 GPU 실험을 먼저 종료한 동일 rootless Docker 터미널에서 실행합니다.

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
export BASELINE_ENV="$PWD/deploy/docker/baselines/baseline.three-model.env"
bash deploy/docker/baseline-docker.sh infaas-accuracy settings
bash deploy/docker/baseline-docker.sh infaas-accuracy build &&
bash deploy/docker/baseline-docker.sh infaas-accuracy download &&
bash deploy/docker/baseline-docker.sh infaas-accuracy up &&
bash deploy/docker/baseline-docker.sh infaas-accuracy register --trials 30
```

새 독립 모델 폴더에 기존 Proteus 캐시를 복사하려면 download 대신 아래를 한 번
사용할 수 있습니다. 대상이 이미 있으면 덮어쓰지 않으므로 그때는 download를 사용합니다.

```bash
bash deploy/docker/baseline-docker.sh infaas-accuracy seed \
  --from "$(python3 -m proteus_realtime.deployment get MODEL_DIR)"
```

프로파일 완료 후 선택된 초기 모델이 GPU에 상주합니다. test/bench는 별도 워밍업과
정책 초기화를 수행하므로 아래 트레이스 앞에 끼워 넣지 않습니다.

```bash
bash deploy/docker/baseline-docker.sh infaas-accuracy trace \
  --trace "$PWD/workloads/twitter-resnet-slo200.txt" \
  --limit 0 --speed 2 --concurrency 256 --timeout 10
bash deploy/docker/baseline-docker.sh infaas-accuracy state
```

결과: /data4tb/taeho-proteus/experiments/infaas-accuracy-three-model-v1/infaas-accuracy/results/trace/

재생기는 기존 Proteus 클라이언트와 같은 open-loop 스케줄러·요약 정의를 사용합니다.
미전송과 전송 지연도 기록합니다. 재생 검증 실패 시 종료 코드 3이지만 결과는 보존합니다.
사전 모델 점수 평균은 이미지 정답률이 아닙니다.

```bash
bash deploy/docker/baseline-docker.sh infaas-accuracy down
```

로컬 검증은 fake GPU와 실제 원본 제어 함수·gRPC·Docker 명령 검증입니다.
V100 성능, 실제 체크포인트 로딩·전처리 호환성·SLO는 서버 결과로 확인해야 합니다.
기존 Clipper singleton·Sommelier 설정과 동작은 보존합니다.
