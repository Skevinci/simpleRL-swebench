
HDFS_HOME=/workspace/hdfs
RUN_NAME=swebench_ppo

python3 openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --colocate_actor_ref \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 2 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 2 \
    --pretrain $HDFS_HOME/model_hub \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 80000 \
    --generate_max_len 20000 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  data/math_level3to5_data_processed_with_qwen_prompt.json \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb 354d09b784ba4b3a3159f7eabc57d851a14ef067 \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000 

        # --vllm_gpu_memory_utilization 0.3 \
            # --enable_prefix_caching \