# Proteus

## 실험 종류별 설정 및 실행

모델·GPU·SLO·배치 후보 변경과 단계별 명령은 **[setting_readme.md](setting_readme.md)**를 참고하세요.

| 실험 | 스크립트 | 상세 안내 |
|---|---|---|
| Proteus | `deploy/docker/proteus-docker.sh` | [Proteus Docker](docker.md) |
| INFaaS-Accuracy | `deploy/docker/baseline-docker.sh infaas-accuracy` | [기준선 GPU](GPU_BASELINES.md) |
| Clipper-HA | `deploy/docker/baseline-docker.sh clipper-ha` | [기준선 GPU](GPU_BASELINES.md) |
| Clipper-HT | `deploy/docker/baseline-docker.sh clipper-ht` | [기준선 GPU](GPU_BASELINES.md) |

GPU 실행·비교 브랜치는 `docker/v2`입니다. 기준선은 독립 INFaaS/C++ Clipper 서비스와 다릅니다.

## 실제 GPU에서 Docker로 실행하기

Proteus만 실행하려면 **[한국어 Docker 실행 가이드](docker.md)**를 따라 실행하세요.
기본 구성은 **V100 한 장 · ResNet-50 · FP32 · SLO 200ms**입니다.
하나의 스크립트로 GPU·모델 설정, 이미지 빌드, 가중치 다운로드, 워커 실행,
프로파일링, gRPC 요청, 테스트와 종료를 진행합니다. 모델 학습은 하지 않습니다.

이 경로는 독립 GPU 런타임에 단일 GPU용 MILP·적응형 배치를 연결한 실험적 포트입니다.
INFaaS 서비스에 의존하지 않으며, 논문의 전체 클러스터 재현과는 구분합니다.
원본 시뮬레이터와 아래 원본 안내는 그대로 유지합니다.

Proteus is an ML inference serving system for resource-constrained compute clusters that proposes a technique to handle workload spikes called "accuracy scaling". Accuracy scaling leverages the flexibility of ML models to adjust the accuracy of served inference requests to adapt to increased load, as opposed to traditional hardware scaling that adapts by adding hardware.

Proteus is set to appear in ASPLOS 2024. You can find it [here](https://guanh01.github.io/files/2024proteus.pdf).

This artifact describes the complete workflow to setup the simulation experiments for Proteus. We describe how to obtain the code, and then describe two methods to install the simulator. We explain how to run the experiments and plot the results. We also publicize  all the workload traces used in our paper.

## Organization

This repository is organized as follows.

- `algorithms` contains the implementation for Proteus as well as the baselines
- `configs` contains the configuration files used to drive the simulator. Each configuration file specifies a workload trace, the algorithms to use, and any relevant hyper-parameters
- `core` contains the simulator implementation files
- `figures` contains all the figures plotted by the scripts in the `plotting` directory
- `gurobi` contains the Gurobi licence used to run the optimization for Proteus. If you use obtain your own Gurobi license, it should be placed in this directory
- `logs` contains the logs gathered from the simulator for each experiment used to generate the figures
- `plotting` contains the scripts to plot results from the collected logs from the simulator
- `profiling` contains the profiled information of the model variants used by Proteus to drive its resource allocation algorithm
- `traces` contains all workload traces used in our paper
- `run.py` is the script used to start the simulator using a configuration file

## Experimentation

You can test Proteus using two methods as described below.

1. Using the Proteus Docker image with a pre-installed Gurobi trial license for quick testing

2. Installing Proteus locally using a `conda` environment and setting up a Gurobi license

We strongly recommend using Method 2 if you want to use Proteus or the simulator for your experiments. However, if you want to quickly test Proteus, we recommend Method 1.

## Method 1: Docker

Proteus uses the Gurobi optimization software to solve its MILP formulation. Although Gurobi requires a full-usage license, we have prepared a [Docker image](https://hub.docker.com/r/sohaibahmad759/proteus) with a pre-installed trial license to allow quick testing. However, we strongly recommend that you obtain a Gurobi license if you want to use Proteus.

Follow the instructions [here](SIMULATOR_DOCKER.md) to run the original Proteus simulator using Docker.

## Method 2: Local installation

To use Proteus, we recommend obtaining a Gurobi license and creating a `conda` environment.

### Obtaining repo

You can clone the GitHub repo using:

`git clone https://github.com/UMass-LIDS/Proteus.git && cd Proteus`

### Conda environment setup

Once the repo is cloned, setup the `conda` environment as follows:

```bash
conda create --name proteus python=3.9.6
conda activate proteus`
python -m pip install -r requirements.txt
```

### Gurobi license

To complete the local installation, you need to obtain a Gurobi license as following.

1. Follow the instructions [here](https://www.gurobi.com/solutions/licensing/) to get a commercial or a free academic license for Gurobi, depending on your use.

2. Once you have obtained the license, Gurobi will provide a `gurobi.lic` file.

3. Place the license file under the path `gurobi/gurobi.lic`.

### Running experiments

Once you have followed the instructions above successfully, your local installation is complete. You can now follow the instructions [here](EXAMPLES.md) to run experiments with Proteus and the comparison baselines on various traces using the simulator.


## Standalone real-GPU port (experimental, single GPU)

For setup and commands, start with the [Korean Docker quickstart](docker.md).

See [GPU_SERVING.md](GPU_SERVING.md) for the independent PyTorch/Hugging Face
runtime and the `deploy/docker/proteus-docker.sh` command interface. It connects
a single-GPU/single-application reduction of the MILP and a wall-clock port of
accscale batching to real CUDA model execution. It does not import, start, or
inherit an INFaaS runtime/image; caches and results are separate by default.
The default one-model configuration cannot demonstrate accuracy scaling.
Multiple-model selection requires comparable offline accuracy measurements.
This is an experimental port, not the paper's full cluster implementation.
The original simulator and instructions above remain unchanged. Local tests
use a real Gurobi solver and fake GPU. Verify each server configuration using
doctor/register/test before running policy benchmarks.

## Citation

If you use this artifact or find it helpful, please cite our paper.

```bash
@article{ahmad2024proteus,
  title={Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling},
  author={Ahmad, Sohaib and Guan, Hui and Friedman, Brian D. and Williams, Thomas and Sitaraman, Ramesh K. and Woo, Thomas},
  booktitle={29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’24)},
  year={2024},
  doi={https://doi.org/10.1145/3617232.3624849}
}
```
