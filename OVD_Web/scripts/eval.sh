#!/usr/bin/env bash
set -eo pipefail

DATA_PATH="${DATA_PATH:-${DATA_DIR:-}}"
MODEL_PATH="${MODEL_PATH:-}"

TOTAL_STEPS="${TOTAL_STEPS:-200}"
IP="${IP:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-30006}"
SEARCH_MODE="${SEARCH_MODE:-simulate_prompt}"
SIMULATION_LLM="${SIMULATION_LLM:-}"
START_THRESHOLD="${START_THRESHOLD:-0}"
END_THRESHOLD="${END_THRESHOLD:-0.5}"
SEARCH_ENGINE="${SEARCH_ENGINE:-wiki}"
MAX_TURNS="${MAX_TURNS:-5}"
TOPK="${TOPK:-5}"
reject_threshold="${reject_threshold:-0}"
enable_speculative_decoding="${enable_speculative_decoding:-False}"
update_query="${update_query:-False}"

WAND_PROJECT="${WAND_PROJECT:-OVD}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-example_eval}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if [[ -n "$HIP_VISIBLE_DEVICES" ]]; then
  export HIP_VISIBLE_DEVICES
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    +data.drop_last_train_dataloader=False \
    +data.drop_last_val_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=True \
    trainer.logger=['wandb'] \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=300 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.max_turns=${MAX_TURNS} \
    trainer.reward_function=f1 \
    trainer.do_search=True \
    retriever.start_threshold=${START_THRESHOLD} \
    retriever.end_threshold=${END_THRESHOLD} \
    retriever.llm_ip=${IP} \
    +retriever.llm_port=${LLM_PORT} \
    retriever.reject_threshold=${reject_threshold} \
    retriever.enable_speculative_decoding=${enable_speculative_decoding} \
    retriever.search_mode=${SEARCH_MODE} \
    retriever.search_engine=${SEARCH_ENGINE} \
    retriever.topk=${TOPK} \
    retriever.simulate_llm=${SIMULATION_LLM} \
    retriever.update_query=${update_query} \
