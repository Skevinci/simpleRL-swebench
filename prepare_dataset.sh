#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --output=/nlp/data/sikaili/simpleRL-reason/output/prepare_output.txt
#SBATCH --error=/nlp/data/sikaili/simpleRL-reason/output/prepare_error.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1

cd /nlp/data/sikaili/simpleRL-reason

/mnt/nlpgridio3/data/sikaili/SWE-bench/envs/bin/python3.10 preprocess_swebench.py