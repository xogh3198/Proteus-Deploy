# ONNX forward 지연 원인 진단

2026-11-05 Twitter v1 측정에서 ResNet-152의 프로파일 P50은 배치 1/8에서 6.97/25.98ms,
실제 실행 P50은 약 172/153ms였다. performance governor 적용 뒤에도 forward 중앙값은 거의 같았다.
이 사실만으로 CPU 주파수·GPU 연산·ONNX 내부 대기 중 어느 것이 원인인지 단정하지 않는다.

## 실행

등록이 완료되고 요청 처리가 끝난 `source-proteus-v1-twitter` 워커가 실행 중이어야 한다.
원래 GPU 이미지의 immutable image ID를 사용한다. 새 이미지 빌드·다운로드·register는 하지 않는다.
진단 코드·실행 스크립트는 기존 실험의 소스 fingerprint 범위 밖에 있어서 이 변경으로 rebuild가 필요하지 않다.

```bash
cd /home/xsailor4/taeho/serving-repos/Proteus
systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"

git fetch origin refs/heads/v1 &&
git checkout --detach FETCH_HEAD

# 조회·검증만: 컨테이너 종료/생성, 파일 쓰기, 추론 요청 없음
bash deploy/docker/diagnose-forward.sh --dry-run

# 실제 진단: 이 Twitter 워커만 잠깐 종료하고, 종료/오류 시 다시 시작
bash deploy/docker/diagnose-forward.sh
```

첫 비교에서는 CPU 전력 설정·프로파일을 추가로 변경하지 않는다.
원래 워커를 재시작하므로 누적 요청 통계·정책 이력·GPU 상주 상태는 초기화된다.
기존 가중치·프로파일·실험 결과 파일은 보존되며, 프로파일 SHA256을 실행 전후 확인한다.
진행 중 요청이 있거나 같은 GPU에 다른 프로세스가 남아 있으면 진단을 중단한다. 다른 서비스를 종료하지 않는다.

기본 조건:

- 직접 실행: ResNet-152, 배치 1·8. 완료 후 다음 호출까지 대기 0·100·300ms, 조건별 워밍업 3회·측정 10회.
- 실제 서빙: 같은 이미지의 원래 Executor·Predictor·Ilp·gRPC를 사용하여 공개 Twitter 첫 100건을 원래 시각·SLO 300ms로 재생한다. 재생 전 모델을 준비하여 최초 로딩과 반복 forward를 구분한다.
- 상세 관측 off/on을 모두 실행한다. off도 전체 실행시간·호출 스레드 CPU 시간을 측정하지만, 내부 API별 관측은 하지 않는다.
- 기존 전처리 v1·FP32·CPU 스레드·SLO·배치 후보·정책을 유지한다. 직접 실행의 인위적인 idle 간격은 진단 조건이며 논문 트레이스를 대신하지 않는다.
- 직접 실행은 저장된 프로파일을 새로 쓰지 않는다. 실제 서빙에서는 저장된 기존 프로파일과 원래 모델 선택·적응형 배치를 사용한다.

필요하면 직접 실행 배치를 추가할 수 있다. RPC 단계의 배치 후보·모델 선택 정책은 바뀌지 않는다.

```bash
bash deploy/docker/diagnose-forward.sh --batches 1,2,4,8 --trials 20
```

### 직접 실행은 빠르고 RPC에서만 느린 경우의 추가 진단

2026-11-05 첫 진단에서는 300ms idle 뒤 직접 배치 1 실행이 약 8ms, RPC 실행은 약 170ms였다.
RPC의 추가 시간은 ort_run에 집중됐고, 같은 호출 스레드의 CPU 시간도 약 170ms였다.
이 단계에서는 네트워크·단순 idle 간격·결과 복사·마지막 synchronize 중 하나를 원인으로 단정하지 않는다.
아래 추가 모드로 모듈 import, 세션·스레드, 동시 작업, gRPC의 영향을 나눠 확인한다.

```bash
bash deploy/docker/diagnose-forward.sh --context-only --dry-run
bash deploy/docker/diagnose-forward.sh --context-only
```

| case | 비교하는 조건 |
|---|---|
| context_before_imports / context_after_imports | 같은 직접 실행 스레드에서 native/RPC 모듈 import 전후에 세션을 새로 만들어 측정 |
| context_native_gpu_thread | 실제 Executor의 GPU 스레드·세션에서 관리용 exclusive 경로로 직접 추론 |
| context_server_started | 같은 세션·GPU 스레드에서 gRPC 서버만 시작하고 요청 전송 전 측정 |
| context_submit_no_grpc | 원래 submit·큐·적응형 배치를 실행하되 gRPC 통신을 거치지 않음 |
| context_grpc | 같은 Executor·세션에 원래 gRPC 프로토콜로 요청 전송 |
| context_after_rpc | RPC 종료 후 같은 GPU 스레드·세션의 직접 실행을 다시 측정 |

각 조건의 thread ID, Torch 스레드 설정, ORT 세션의 스레드 수·provider/options·입력 형상을 기록한다.
모델 후보·원래 모델 선택 함수·배치 함수·기존 프로파일은 변경하지 않는다.
이 모드도 같은 원래 컨테이너의 종료·복원과 프로파일 SHA256 검사를 수행한다.
결과는 forward-diagnostic.json, context_submit_no_grpc.json, context_grpc.json에 저장한다.

### 같은 세션에서 배치 크기 변경 비용 확인

두 번째 진단에서는 같은 실제 GPU 스레드·세션의 직접 실행이 약 7ms였지만,
gRPC를 거치지 않고 원래 submit·큐·배치 경로를 사용해도 약 170ms였다.
서버 시작·단순 import·스레드 수 차이·gRPC 통신만으로는 이 차이가 설명되지 않았다.
직접 검사는 같은 배치를 반복하고 서빙에서는 배치 크기가 바뀐다는 차이를 다음과 같이 분리한다.

```bash
git fetch origin refs/heads/v1 &&
git checkout --detach FETCH_HEAD

bash deploy/docker/diagnose-forward.sh --batch-shapes-only --batches 1,2,4,8 --dry-run
bash deploy/docker/diagnose-forward.sh --batch-shapes-only --batches 1,2,4,8
```

새 빌드·register 없이 원래 이미지·모델·프로파일·CUDA provider 설정을 사용한다.
세션 하나와 GPU 호출 스레드 하나를 유지하고, RPC·큐·정책을 실행하지 않는 직접 검사다.
이 모드도 원래 워커만 종료했다가 복원하므로 메모리의 누적 통계·상주 상태는 초기화된다.

| case | 측정 조건 |
|---|---|
| shape_fixed | 배치별 워밍업 3회 후 같은 배치 10회 반복. 저장 프로파일과 같은 방식 |
| shape_mixed_switch | 모든 배치를 이미 사용한 같은 세션에서 다른 크기로 바뀐 첫 호출 |
| shape_mixed_repeat | 바뀐 크기로 첫 호출을 마친 직후 같은 크기의 두 번째 호출 |

혼합 순서는 `1,1,2,2,4,4,8,8`을 기본 10회 반복한다. 혼합 구간은 첫 호출도 버리지 않고 모두 기록한다.
상세 관측 off/on을 각각 측정하며, 두 조건에서도 같은 ONNX 세션을 유지한다.
결과 JSON에는 이전 배치·현재 배치·변경 여부·cycle·pair_index·세션 identity와 로드된 cuDNN 라이브러리 경로를 기록한다.

`switch`만 약 150~170ms이고 `repeat`는 프로파일 수준이면 배치 크기 변경 비용이 직접 실행에서도 재현된 것이다.
두 조건 모두 빠르면 이 가설만으로 설명할 수 없으므로 원래 큐·정책과의 동시 실행 영향을 다음으로 확인한다.
두 조건 모두 느리거나 관측 off/on 차이가 크면 해당 결과를 먼저 검토한다.

코드 근거: [ONNX Runtime 1.22.0 conv.cc](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/cuda/nn/conv.cc#L301)에서
cuDNN 9 경로는 입력 차원이 변경되면 `CreateCudnnFeExecutionPlan()`을 호출하여 실행 계획을 다시 구성한다.
배치 크기 N도 입력 차원에 포함된다. `EXHAUSTIVE`라는 설정명만으로 실제 알고리즘 벤치마크가 매번 수행된다고 단정하지 않는다.
실제 라이브러리 버전·변경/반복 결과를 함께 확인하며, 진단 단계에서는 provider 옵션이나 기존 배치 정책을 바꾸지 않는다.

## 관측값과 해석

| 관측값 | 의미 |
|---|---|
| 전체 wall P50/P95 | 원래 infer_prepared() 호출의 실제 경과시간. sleep은 제외 |
| 전체 thread CPU P50 | 호출한 한 스레드의 CPU 실행시간. ORT 작업자 스레드의 CPU 시간은 포함하지 않음 |
| 전체 process CPU P50 | 같은 시간 동안 프로세스의 모든 스레드가 쓴 CPU 시간. 동시 작업도 포함되며 wall보다 클 수 있음 |
| io_binding / bind_cpu_input / bind_output | 입력·출력 binding API 시간. 실제 H2D 전송 시간과 동일하지 않음 |
| ort_run | run_with_iobinding() 호출 시간 |
| copy_outputs_to_cpu | 결과 복사 API 시간 |
| cuda_synchronize | 원래 torch.cuda.synchronize() 호출 시간 |
| outside_observed_APIs | 배치 packing·검사·Python 처리 등 관측 API 밖의 경과시간. 단계별 중앙값의 합으로 계산하지 않고 호출마다 분해 |
| resources / ancestor_cpu_limits | CPU affinity, governor, 컨테이너·상위 cgroup 제한/통계. 카운터가 없다는 것을 제한이 없다는 증거로 해석하지 않음 |

판단 순서:

1. 연속 직접 실행도 저장 프로파일보다 느리면 현재 모델·런타임·GPU/CPU 자원 조건 차이부터 본다.
2. 연속은 빠르고 간헐 직접 실행만 느리면 실행 간격과 관련된 런타임/장치 상태를 확인한다.
3. 직접 실행은 빠르고 원래 RPC 서빙만 느리면 동시 작업·실행 스레드·스케줄링·RPC 경로를 비교한다.
4. 상세 관측에서 시간을 차지하는 API를 찾는다. wall이 크고 호출 스레드 CPU 시간이 작으면 대기·동기화·스케줄링을, CPU 시간도 크면 CPU 실행·spin 등을 추가 조사한다.
5. 관측 off/on 차이가 크면 상세 관측 수치를 원래 실행시간으로 사용하지 않는다.

이 수치만으로 CPU 주파수나 GPU kernel 연산을 확정하지 않는다. ORT 호출 시간에는 CPU와 장치 대기가 섞일 수 있다.
PyTorch 기본 스트림의 CUDA event로 ORT private stream의 순수 GPU 시간을 측정했다고 주장하지 않는다.
필요한 경우 다음 단계에서 Nsight Systems 또는 CUDA profiling을 활성화한 ONNX Runtime으로 kernel timeline을 확인한다.
공식 설명: https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html

## 저장 파일

실제 DATA_DIR 아래 다음 경로에 새 결과만 저장한다.

```text
<RUN_DIR>/results/forward-diagnostics/<UTC시각>-<토큰>/
  host.json                  원래 이미지 ID·프로파일 해시·상위 CPU cgroup 제한
  host-after.json            같은 상위 CPU cgroup의 진단 종료 시점 통계
  container.log              전체 출력·오류
  forward-diagnostic.json    실제 batch 호출별 wall/CPU/API 시간과 조건별 요약
  rpc-observer-0.json        상세 관측 off인 실제 RPC 요청 결과
  rpc-observer-1.json        상세 관측 on인 실제 RPC 요청 결과
```

진단은 원인 분리용이며, 첫 100건·warm 상태·관측 on/off 결과를 전체 논문 워크로드의 최종 성능으로 사용하지 않는다.
로컬 검증은 CPU와 명시적인 가짜 GPU backend를 사용한 연결·복원·원본 보존 검사다. 실제 GPU 원인은 서버 출력으로 판단한다.
