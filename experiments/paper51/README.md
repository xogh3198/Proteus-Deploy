# 논문 모델 목록·워크로드 출처 확인

이 디렉터리는 실험 준비 중 확보한 **출처 목록·가중치 확보·트레이스 검증 도구**다.
아직 51개 모델의 GPU 실행기나 완료된 v1/v2 실행 안내가 아니다.

## 확정 조건

| 항목 | 선택 |
| --- | --- |
| GPU | V100 32GB 한 대 |
| 정밀도 | v1·v2 모두 FP16; 기존 FP32 프로파일 재사용 금지 |
| 모델 후보 | 논문 Table 3의 9개 그룹·51개 변형 |
| v1 | 요청 시 전처리 제외; 실제 모델 로딩·GPU 추론 측정 |
| v2 | 같은 모델·워크로드에 요청 시 CPU 전처리 포함 |
| 기존 코드 | 기존 브랜치·정책 유지, 실험용 추가 경로로 구성 |

FP16이라도 T5-11B의 실행 가능 배치·입력 길이는 실제 메모리 점검이 필요하다.
CPU offload·양자화·다중 GPU로 자동 대체하지 않는다.

## 모델 목록

| 그룹 | 논문 전체 변형 | 개수 |
| --- | --- | ---: |
| ResNet | 18, 34, 50, 101, 152 | 5 |
| DenseNet | 121, 161, 169, 201 | 4 |
| ResNeSt | 14, 26, 50, 269 | 4 |
| EfficientNet | b0–b7 | 8 |
| MobileNet v1 | 폭 0.25, 0.5, 0.75, 1.0 | 4 |
| YOLOv5 | n, s, m, l, x | 5 |
| BERT 계열 | RoBERTa 2개, ALBERT 4개, BERT tiny/mini/small/medium/base/large | 12 |
| T5 | small, base, large, 3b, 11b | 5 |
| GPT-2 | base, medium, large, xl | 4 |

공개 실행 목록과 프로파일 CSV에는 EfficientNet b2/b4/b6, T5 3b/11b가 없다.
따라서 `catalog.json`은 이 5개의 공개 정확도 값을 `null`로 둔다.
다른 모델 값 복사·보간·임의의 100점 대입으로 정확도 입력을 만들지 않는다.
46개의 점수도 **공개 시뮬레이터의 사전 기준값**이며 신규 체크포인트의 실측 정확도가 아니다.

추가 5개의 정책용 점수는 `supplemental_scores.json`에 원 출처와 분리해 보완했다.

| 추가 모델 | 원 출처의 값 | 정책용 정규화 |
| --- | ---: | --- |
| EfficientNet-b2 | ImageNet Top-1 80.1 | 100 × 80.1 / 84.3 |
| EfficientNet-b4 | ImageNet Top-1 82.9 | 100 × 82.9 / 84.3 |
| EfficientNet-b6 | ImageNet Top-1 84.0 | 100 × 84.0 / 84.3 |
| T5-3B | WMT EnDe BLEU 31.8 | 100 × 31.8 / 32.1 |
| T5-11B | WMT EnDe BLEU 32.1 | 100 |

EfficientNet은 [Google 공개 체크포인트 표의 AutoAugment 행](https://github.com/google-research/sam#finetuning-efficientnet),
T5는 [원 논문 Table 14](https://jmlr.org/papers/volume21/20-074/20-074.pdf)를 사용했다.
기존 공개 프로파일에 있는 EfficientNet 5개와 T5 3개의 원 점수가 각각 이 표의 수치와 일치한다.
정규화 분모도 공개 프로파일의 기존 기준을 유지했다. T5-3B의 점수가 T5-Large보다 낮은 것은 표의 값을 그대로 반영한 것이다.

공개 저장소에는 실제 ONNX 가중치·입력 규격·토크나이저 파일이 없다.
논문과 동일한 아키텍처 이름만으로 동일 체크포인트나 동일 NLP 태스크가 보장되지 않는다.
실제 GPU 실행용 공개 가중치 출처는 `sources.json`에 고정했다.

- GluonCV 17개: ResNet·DenseNet·ResNeSt·MobileNet v1. 원본 배포의 parameter SHA1을 검증한다.
- Hugging Face 29개: EfficientNet 8개·BERT 12개·T5 5개·GPT-2 4개. 모두 40자리 커밋을 고정한다.
- YOLOv5 5개: v7.0 소스 커밋·배포 asset ID·파일 크기를 고정한다.

`weights.py`는 다운로드한 모든 파일의 SHA256을 `checkpoint.lock.json`에 기록한다.
재사용 시 출처·크기·내용을 다시 검증한다. 원본과 같은 아키텍처의 공개 가중치를 재구성하는 경로이며,
원 논문의 정확한 체크포인트가 복구됐다고 주장하지 않는다.

BERT 계열은 공개 pretrained encoder를 사용한다. 공개되지 않은 감성 분류 head를 무작위 초기화하지 않는다.
이 경로의 encoder 출력과 논문의 감성 분류 점수는 다른 것이므로 기준 점수를 실제 출력 정확도로 해석하면 안 된다.
NLP의 길이 128·greedy 생성 16토큰은 명시적인 재구성 설정이며 원 논문에서 확인한 수치가 아니다.
이러한 입력·출력 차이는 실제 실행 안내에도 표시해야 한다.

```bash
# 표준 라이브러리만으로 출처 목록 확인. 실제 GPU 로딩은 하지 않는다.
python3 -m experiments.paper51.weights list --group all

# 가중치 다운로드 도구. Hugging Face 그룹에는 huggingface_hub 설치가 필요하다.
# ONNX 변환·FP16 유한값 검사·GPU 메모리 검증은 이후 단계다.
python3 -m experiments.paper51.weights download --group mobilenet --root data/paper51/source-weights
python3 -m experiments.paper51.weights verify --group mobilenet --root data/paper51/source-weights
```

## Clipper의 공개 고정 모델

단순히 각 그룹의 최고 정확도 모델을 고르는 대신, 공개 static allocation의 양수 할당을 확인했다.
다음 목록의 모델 선택을 가져오는 것이며 원래의 약 40개 장치 배치를 V100 한 대로 재현하는 것은 아니다.

| 그룹 | HA | HT |
| --- | --- | --- |
| BERT | albert-xxlarge-v2 | prajjwal1/bert-tiny |
| DenseNet | densenet201 | densenet121 |
| EfficientNet | efficientnet-b7 | efficientnet-b0 |
| GPT-2 | gpt2-medium | gpt2 |
| MobileNet | mobilenet1.0 | mobilenet0.25 |
| ResNeSt | resnest269 | resnest14 |
| ResNet | resnet152_v1 | resnet18_v1 |
| T5 | t5-base | t5-small |
| YOLOv5 | yolov5x | yolov5n |

일반 후보 파일의 DenseNet201 점수는 99.51093951이지만 HA 전용 파일에는 100으로 기재돼 있다.
공통 카탈로그는 일반 후보 파일 값을 보존하며, 이 차이를 숨기지 않는다.

## 트레이스

Twitter 기반 기본 목록은 `zipf_exponential/300`이다.
각 줄은 `도착 시각(ms),QoS,SLO(ms)`이며 공개 파일의 SLO는 300ms이다.
기존 실험의 SLO 200ms와 구분한다. 시간 배율은 도착 시각에만 적용하고 SLO에는 적용하지 않는다.

| 그룹 | 요청 수 | 1배속 마지막 도착(초) | 2배속 평균 QPS |
| --- | ---: | ---: | ---: |
| BERT | 6,845 | 289.435 | 47.30 |
| DenseNet | 4,311 | 290.118 | 29.72 |
| EfficientNet | 13,214 | 289.774 | 91.20 |
| GPT-2 | 2,878 | 290.621 | 19.81 |
| MobileNet | 26,753 | 290.020 | 184.49 |
| ResNeSt | 9,117 | 290.482 | 62.77 |
| ResNet | 3,776 | 290.308 | 26.01 |
| T5 | 3,171 | 290.246 | 21.85 |
| YOLOv5 | 5,188 | 289.673 | 35.82 |
| 全그룹 동시 도착 | 75,253 | 290.621 | 517.88 |

기존 3,776건은 ResNet 그룹만의 목록이다. 9개 그룹을 별도로 실행한 합과
한 V100에서 혼합 요청 75,253건을 동시에 처리한 결과는 같은 실험이 아니다.
단일 V100에서의 그룹 실행 방식은 확정 후 명시한다.

```bash
# 출처·목록 확인만 수행한다. Docker/GPU/서빙 상태를 바꾸지 않는다.
python3 experiments/paper51/audit.py catalog
python3 experiments/paper51/audit.py workload --workload twitter --speed 2

# 원본 커밋에서 내려받고 바이트 SHA256·행 수·마지막 도착·SLO를 검증한다.
# 이미 있는 파일은 덮어쓰지 않고 검증한다.
python3 experiments/paper51/audit.py fetch-traces --workload twitter
python3 experiments/paper51/audit.py verify-traces --workload twitter
```

추가 공개 목록은 `--workload uniform`, `gamma`, `bursty`로 확인한다.
이 목록을 논문 특정 마이크로벤치마크와 동일하다고 간주하지 않는다.

## 출처

- [Proteus 논문 Table 3·§6](https://groups.cs.umass.edu/ramesh/wp-content/uploads/sites/3/2023/09/ljWiUS-asplos24spring-final29.pdf)
- [공개 artifact 커밋](https://github.com/UMass-LIDS/Proteus/tree/6967c9327e66f6d6fe94993dcaa7521efc2f72f5)
- [공개 후보 목록](https://github.com/UMass-LIDS/Proteus/tree/6967c9327e66f6d6fe94993dcaa7521efc2f72f5/traces/model_variants)
- [Clipper HA 할당](https://github.com/UMass-LIDS/Proteus/blob/6967c9327e66f6d6fe94993dcaa7521efc2f72f5/algorithms/clipper_solutions/clipper_highacc_optimized.txt)
- [Clipper HT 할당](https://github.com/UMass-LIDS/Proteus/blob/6967c9327e66f6d6fe94993dcaa7521efc2f72f5/algorithms/clipper_solutions/clipper_lowacc_optimized.txt)

공개 소스의 모델 로딩 시간은 `core/simulator.py`에서 0으로 초기화한다.
이를 실제 체크포인트 로딩 측정값으로 사용하지 않는다.
