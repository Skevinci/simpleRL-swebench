
# Simple Reinforcement Learning for Reasoning

## Quick Start

### Installation

Our code is implemented based on OpenRLHF. Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments and install our version:

```bash
salloc --nodelist=nlpgpu04 --gpus=6 --mem=400GB --cpus-per-gpu=4 # ensure number of cpus
dockerd-rootless.sh &
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
git clone https://github.com/Skevinci/simpleRL-swebench.git
cd simpleRL-swebench/train
pip install -e .
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir /workspace/hdfs/model_hub
```

### Reproducing SimpleRL-Zero
The minimum hardware requirement for training is 6 H/A100-80G GPUs (note: this configuration has not been tested yet). To accelerate our experiments, we used 4 nodes, each equipped with 8 H/A100-80G GPUs, to train on 8K MATH examples for 120 steps over approximately 1.5 days, achieving convergence. However, our results indicate that satisfactory performance can be achieved with around 60 steps, which requires less than one day of training using 4 nodes.

The training process leverages PPO with Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:
```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```

Next, submit the training job from the master node:

```bash
cd train
# For 4 nodes:
ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
    }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_new.sh

# For 1 node:
ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{
        "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
    }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_1_node.sh

```
