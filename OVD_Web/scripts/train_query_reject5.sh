#!/usr/bin/env bash
set -euo pipefail

# Query Reject 5 configuration used for the four-GPU Web retraining.
# Set DATA_DIR, MODEL_PATH, CKPT_PATH, and the local simulator endpoint first.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_GPUS_PER_NODE=4
export TOTAL_STEPS=201
export TRAIN_BATCH_SIZE=64
export VAL_BATCH_SIZE=768
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE=4
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=4
export REF_LOG_PROB_MICRO_BATCH_SIZE=4
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
export SAVE_FREQ=200
export SEARCH_MODE=simulate_sft
export SEARCH_ENGINE=simulate_sft
export START_THRESHOLD=0.25
export END_THRESHOLD=0.5
export reject_threshold=5
export enable_speculative_decoding=True
export update_query=True
export SIMULATION_LLM="${SIMULATION_LLM:-sunhaonlp/Simulation_LLM_google_7B_V2}"

exec bash "$SCRIPT_DIR/train.sh"
