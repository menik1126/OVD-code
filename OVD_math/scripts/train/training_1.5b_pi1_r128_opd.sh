#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=Qwen/Qwen2.5-Math-1.5B bash scripts/train/training_1.5b_pi1_r128_opd.sh
  bash scripts/train/training_1.5b_pi1_r128_opd.sh --dry-run

Optional env:
  WANDB_PROJECT              default: one_shot_rlvr
  EXPERIMENT_TAG             default: example
  EXPERIMENT_NAME            default: derived from params
  HIP_VISIBLE_DEVICES        default: (unset; uses all visible)
  N_GPUS                     default: 4

Data:
  DATA_SAMPLE                default: pi1
  REPEAT_TIME                default: 128
  TRAIN_FILE                 default: <project>/data/train/one_shot_rlvr/${DATA_SAMPLE}_r${REPEAT_TIME}.parquet
  VAL_FILE                   default: <project>/data/test/math500.parquet

Training:
  LEARNING_RATE              default: 1e-6
  ROLLOUT_N                  default: 8
  TEMPERATURE                default: 0.6
  BATCH_SIZE                 default: 128
  TENSOR_PARALLEL            default: 1

OPD (reward_model):
  USE_OPD                    default: True
  REJECT_THRESHOLD           default: 7
  LLM_IP                     default: 127.0.0.1
  LLM_PORT                   default: 30008
  SIMULATOR_MODEL            default: (empty; required when USE_OPD=True)
  REGENERATION_TEMPERATURE   default: 0.8
  MAX_CONCURRENT_REQUESTS    default: 5
  OPD_REGENERATION_MODE      full | max_entropy_suffix_onecall | random_suffix_onecall
  OPD_PIPELINE_MODE          default: batch (streaming requires customized vLLM 0.6.3)
  RANDOM_CUTOFF_SEED         default: 42
  SUFFIX_MIN_TOKENS          default: 512

Logging / checkpoints:
  ENABLE_WANDB               default: 0 (console only)
  CHECKPOINTS_DIR            default: <project>/checkpoints
  RESUME_MODE                default: (empty; disabled)
  LOG_FILE                   default: <project>/verl_demo_train.log
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  ""|--dry-run) ;;
  *) usage >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_PATH:-}"

WANDB_PROJECT="${WANDB_PROJECT:-one_shot_rlvr}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-example}"
MODEL_NAME="$(basename "$MODEL_PATH")"

N_GPUS="${N_GPUS:-4}"

DATA_SAMPLE="${DATA_SAMPLE:-pi1}"
REPEAT_TIME="${REPEAT_TIME:-128}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/data/train/one_shot_rlvr/${DATA_SAMPLE}_r${REPEAT_TIME}.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/data/test/math500.parquet}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

USE_OPD="${USE_OPD:-True}"
REJECT_THRESHOLD="${REJECT_THRESHOLD:-7}"
LLM_IP="${LLM_IP:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-30008}"
SIMULATOR_MODEL="${SIMULATOR_MODEL:-}"
REGENERATION_TEMPERATURE="${REGENERATION_TEMPERATURE:-0.8}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-5}"
OPD_REGENERATION_MODE="${OPD_REGENERATION_MODE:-full}"
OPD_PIPELINE_MODE="${OPD_PIPELINE_MODE:-batch}"
RANDOM_CUTOFF_SEED="${RANDOM_CUTOFF_SEED:-42}"
SUFFIX_MIN_TOKENS="${SUFFIX_MIN_TOKENS:-512}"
case "$OPD_REGENERATION_MODE" in
  full|max_entropy_suffix_onecall|random_suffix_onecall) ;;
  *) echo "Unsupported OPD_REGENERATION_MODE" >&2; exit 2 ;;
esac
case "$OPD_PIPELINE_MODE" in
  batch|streaming) ;;
  *) echo "Unsupported OPD_PIPELINE_MODE" >&2; exit 2 ;;
esac

ENABLE_WANDB="${ENABLE_WANDB:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$PROJECT_DIR/checkpoints}"
RESUME_MODE="${RESUME_MODE:-}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/verl_demo_train.log}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_TAG}_${MODEL_NAME}_${OPD_REGENERATION_MODE}}"
LOGGER='["console"]'
if [[ "$ENABLE_WANDB" == "1" ]]; then
  LOGGER='["console","wandb"]'
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"

cmd=(python3 -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$BATCH_SIZE"
  data.val_batch_size=530
  data.max_prompt_length=1024
  data.max_response_length=3072
  reward_model.reward_manager='naive'
  +reward_model.use_OPD="$USE_OPD"
  +reward_model.reject_threshold="$REJECT_THRESHOLD"
  +reward_model.llm_ip="$LLM_IP"
  +reward_model.llm_port="$LLM_PORT"
  +reward_model.simulator_model="$SIMULATOR_MODEL"
  +reward_model.regeneration_temperature="$REGENERATION_TEMPERATURE"
  +reward_model.max_concurrent_requests="$MAX_CONCURRENT_REQUESTS"
  +reward_model.opd_pipeline_mode="$OPD_PIPELINE_MODE"
  +reward_model.opd_regeneration_mode="$OPD_REGENERATION_MODE"
  +reward_model.random_cutoff_seed="$RANDOM_CUTOFF_SEED"
  +reward_model.suffix_min_tokens="$SUFFIX_MIN_TOKENS"
  +reward_model.suffix_max_attempts=1
  +reward_model.suffix_fallback_mode=error
  +reward_model.suffix_max_fallback_rate=0.0
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature="$TEMPERATURE"
  +actor_rollout_ref.rollout.val_temperature="$TEMPERATURE"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  +actor_rollout_ref.rollout.n_val=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.critic_warmup=0
  trainer.logger="$LOGGER"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.checkpoints_dir="$CHECKPOINTS_DIR"
  +trainer.val_before_train=True
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes=1
  trainer.save_freq=100
  trainer.test_freq=50
  trainer.default_hdfs_dir=null
  trainer.total_epochs=2000
)

if [[ -n "$RESUME_MODE" ]]; then
  cmd+=(trainer.resume_mode="$RESUME_MODE")
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
if [[ -z "$MODEL_PATH" || ( "$USE_OPD" == "True" && -z "$SIMULATOR_MODEL" ) ]]; then
  echo "Set MODEL_PATH and (for OVD) SIMULATOR_MODEL before launching." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "TRAIN_FILE and VAL_FILE must exist." >&2
  exit 2
fi
mkdir -p "$CHECKPOINTS_DIR" "$(dirname "$LOG_FILE")"
cd "$PROJECT_DIR"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
