#!/bin/bash
#SBATCH --job-name=docker
#SBATCH --output=/nlp/data/sikaili/simpleRL-reason/output/prepare_output.txt
#SBATCH --error=/nlp/data/sikaili/simpleRL-reason/output/prepare_error.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=8
#SBATCH --mem=400GB
#SBATCH --cpus-per-gpu=4

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
sleep 43200