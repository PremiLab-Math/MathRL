#!/bin/bash

set -x

PROJECT_NAME=Qwen2.5-RL
EXPEIMENT_NAME=Qwen2.5-3B-RL-MATH
MODEL_PATH=$XLF/downloads/models/Qwen/Qwen2.5-3B
LOG_DIR=$XLF/scripts/logs/${EXPEIMENT_NAME}

mkdir -p ${LOG_DIR}
chmod -R 777 ${LOG_DIR}

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

ray start --head

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[$XLF/downloads/datasets/MathRL/train_math_level3to5.parquet] \
    data.val_files=$XLF/downloads/datasets/MathRL/test_math_oai.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPEIMENT_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${LOG_DIR} \
    trainer.save_freq=16 \
    trainer.test_freq=10 \
    trainer.total_epochs=34 \
    "$@" 2>&1 | tee ${LOG_DIR}/${EXPEIMENT_NAME}.log
