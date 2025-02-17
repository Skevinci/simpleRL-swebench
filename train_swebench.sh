#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/nlp/data/sikaili/simpleRL-swebench/output/train_node04_output.txt
#SBATCH --error=/nlp/data/sikaili/simpleRL-swebench/output/train_node04_error.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=8
#SBATCH --nodelist=nlpgpu04
#SBATCH --mem=480GB
#SBATCH --cpus-per-gpu=8

dockerd-rootless.sh &
sleep 10

docker run --runtime=nvidia --rm --shm-size="10g" --cap-add=SYS_ADMIN -v /nlp/data/sikaili:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 \
bash -c "
    git clone https://github.com/Skevinci/simpleRL-swebench.git;

    cd simpleRL-swebench/train;
    pip install vllm==0.6.1;
    pip install -e .;

    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /workspace/hdfs/model_hub;

    python preprocess_swebench.py;

    ray start --head --node-ip-address 0.0.0.0 --num-gpus 8;
    ray job submit --address='http://127.0.0.1:8265' --runtime-env-json='{\"pip\": [\"ray==2.12.0\", \"latex2sympy2\", \"timeout_decorator\"]}' -- /bin/bash train_ppo_swebench_1_node.sh
"
