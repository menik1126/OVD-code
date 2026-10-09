# OVD-Math

Run from the `OVD_math` directory. Replace the paths below with your own.

```bash
export MODEL_PATH=/path/to/student_model
export SIMULATOR_MODEL=/path/to/teacher_model  # Model name exposed by the Teacher server
export LLM_IP=127.0.0.1
export LLM_PORT=30008
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/validation.parquet
export N_GPUS=4
export REJECT_THRESHOLD=7
export RESUME_MODE=disable
```

The bundled default training file is `data/train/one_shot_rlvr/pi1_r128.parquet`.
Set `TRAIN_FILE` explicitly to use another dataset (for example, DSR128).

## Full response

```bash
OPD_REGENERATION_MODE=full \
EXPERIMENT_NAME=ovd_full \
CHECKPOINTS_DIR=./checkpoints \
LOG_FILE=./logs/ovd_full.log \
bash scripts/train/training_1.5b_pi1_r128_opd.sh
```

This is the basic full-response launch. The teacher replacement is installed
in the response, input IDs, and masks before PPO computes log-probabilities.
