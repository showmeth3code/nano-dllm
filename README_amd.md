# Installation Guide on AMD GPUs

This guide shows nano-vllm users how to install nano-vllm and how it performs on AMD platform.


## Installation on AMD GPUs

### Launch container environment

```bash
CONTAINER_NAME=<your container name>
IMAGE_NAME=rocm/vllm-dev:rocm6.4.1_navi_ubuntu24.04_py3.12_pytorch_2.7_vllm_0.8.5
# For AMD Instinct GPUs, users can select latest pre-built docker image:
# rocm/vllm:rocm6.4.1_vllm_0.9.1_20250715. See https://hub.docker.com/r/rocm/vllm/tags

docker run -it \
        --rm \
        --device /dev/dri \
        --device /dev/kfd \
        --network host \
        --ipc host \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        --shm-size 8G \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} /bin/bash
```

### Install through pip

```bash 
pip install --no-build-isolation git+https://github.com/GeeeekExplorer/nano-vllm.git
```


## Benchmark on AMD GPUs

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware:
    - Setup 1: Radeon RX7900XT (20GB)
    - Setup 2: Instinct MI300X (192GB) 
    - Setup 3: Ryzen AI 395(128GB unified memory)
    - Setup 4: Radeon RX9070XT (16GB) 
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Setup 1 Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 61.80    | 2167.84               |
| Nano-vLLM      | 133,966     | 65.91    | 2032.36               |

**Setup 2 Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 8.93     | 14994.98              |
| Nano-vLLM      | 133,966     | 20.17    | 6640.22               |

**Setup 3 Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 114.72    | 1167.76              |
| Nano-vLLM      | 133,966     | 123.81    | 1082.05              |

**Setup 4 Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 47.12    | 2842.8                |
| Nano-vLLM      | 133,966     | *        |        *              |

*Known issue: nano-vllm has memory access fault issue on RX9070XT to be fixed.