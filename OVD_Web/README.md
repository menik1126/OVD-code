# OVD-Web: On-policy Verbal Distillation for Web Q&A

This directory contains the **Web question answering (WebQA)** implementation of **OVD (On-policy Verbal Distillation)**.

## Method Summary (OVD)

OVD replaces token-level logit matching with **trajectory-level supervision** using **discrete verbal scores** from a teacher.

For WebQA, a trajectory looks like:

`(q1, d1, s1, q2, d2, s2, ..., qK, dK, sK)` where `q` is a search query, `d` are retrieved documents, and `s` is the reasoning step conditioned on retrieved info.

This repo implements the OVD recipe with GRPO/PPO-style optimization on top of `verl`, and supports:
- **Simulated search** via an OpenAI-compatible local server (vLLM/sglang-like) for the “environment agent”.
- **Real search** (e.g., Google/Serper) for inference-time (and optionally rollouts depending on config).

## Repository Layout

- `scripts/train.sh`: WebQA training entry (GRPO via `verl.trainer.main_ppo`)
- `scripts/eval.sh`: validation-only run
- `llm_agent/generation.py`: search+reason rollout logic + teacher scoring hooks
- `inference.py`: interactive demo (model generates `<search>...</search>` queries and gets `<information>...</information>`)

## Installation

From `OVD_Web/`:

```bash
conda create -n ovd_web python=3.10 -y
conda activate ovd_web
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Data Format

Training/eval scripts expect a directory containing:
- `train.parquet`
- `test.parquet`

Each row should follow the “RLHF parquet” schema used by `verl` (chat-style `prompt` plus `reward_model.ground_truth` for EM/F1 scoring). See `OVD_Web/verl/utils/dataset/README.md`.

## Resources (Simulation Models)

If you want to use the original ZeroSearch-style simulated retrieval, the simulator models commonly used are below (Wikipedia / Google variants). These can be used as `SIMULATION_LLM` (the model name exposed by your OpenAI-compatible server).

| Retriever | Simulation Tuning Dataset | Simulation LLMs | Policy Models |
| --------- | -------------------------- | --------------- | ------------ |
| Wikipedia | •[SimulationTuning_wiki_dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_wiki_dataset) | •[Simulation_LLM_wiki_3B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_3B_V2)<br>•[Simulation_LLM_wiki_7B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_7B_V2)<br>•[Simulation_LLM_wiki_14B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_14B_V2) | •[ZeroSearch_wiki_V2_Qwen2.5_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B)<br>•[ZeroSearch_wiki_V2_Qwen2.5_3B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B_Instruct)<br>•[ZeroSearch_wiki_V2_Llama_3.2_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B)<br>•[ZeroSearch_wiki_V2_Llama_3.2_3B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B_Instruct)<br>•[ZeroSearch_wiki_V2_Qwen2.5_7B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B)<br>•[ZeroSearch_wiki_V2_Qwen2.5_7B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B_Instruct) |
| Google | •[SimulationTuning_google_dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_google_dataset) | •[Simulation_LLM_google_3B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_3B_V2)<br>•[Simulation_LLM_google_7B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_7B_V2)<br>•[Simulation_LLM_google_14B_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_14B_V2) | •[ZeroSearch_google_V2_Qwen2.5_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B)<br>•[ZeroSearch_google_V2_Qwen2.5_3B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct)<br>•[ZeroSearch_google_V2_Llama_3.2_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Llama_3.2_3B)<br>•[ZeroSearch_google_V2_Llama_3.2_3B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Llama_3.2_3B_Instruct)<br>•[ZeroSearch_google_V2_Qwen2.5_7B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B)<br>•[ZeroSearch_google_V2_Qwen2.5_7B_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct) |

### Using `Alibaba-NLP/ZeroSearch_dataset` (recommended quick start)

You can train directly on the public dataset used by the original ZeroSearch setup. Example row:
`https://huggingface.co/datasets/Alibaba-NLP/ZeroSearch_dataset/viewer/default/train?row=0`

Prepare the dataset as `train.parquet` and `test.parquet` in one local directory.
The training script requires both files; this release does not include a dataset
conversion script.

## Training (WebQA)

```bash
DATA_DIR=ZeroSearch_dataset_parquet \
MODEL_PATH=/path/to/base_hf_model_or_checkpoint \
CKPT_PATH=/path/to/save_dir \
bash scripts/train.sh
```

Query Reject 5 (four GPUs, final step 200 only):

This command matches the recorded run settings for basic Query Reject 5, with
think distillation disabled. Web query rejection is not the Math full-response
replacement branch. This release tree is not a byte-for-byte copy of the
historical trainer.

```bash
DATA_DIR=/path/to/webqa_parquets \
MODEL_PATH=/path/to/Qwen2.5-3B-Base \
CKPT_PATH=/path/to/checkpoints \
IP=127.0.0.1 LLM_PORT=30021 \
bash scripts/train_query_reject5.sh
```

Common knobs (env vars in `scripts/train.sh`):
- `SEARCH_ENGINE` (`simulate_sft` for the run above; `wiki` / `google` for other setups)
- `SEARCH_MODE` (default: `simulate_prompt`)
- `IP` / `LLM_PORT`: OpenAI-compatible server address for the simulator / teacher
- `SIMULATION_LLM`: model name/path served by that server
- `MAX_TURNS`, `TOPK`, `reject_threshold`, `START_THRESHOLD`, `END_THRESHOLD`

## Evaluation

```bash
DATA_DIR=/path/to/webqa_parquets \
MODEL_PATH=/path/to/checkpoint_or_hf_model \
bash scripts/eval.sh
```

## Interactive Inference

```bash
python inference.py --model_path /path/to/model --search_engine wiki --topk 5
```

Notes:
- `--search_engine google` requires a search API key in `SER_API_KEY` or
  `SERPER_API_KEY`; never put a key in the source or a launch script.
- `--search_engine wiki` in `inference.py` expects a local retriever at `http://localhost:6002/retrieve` (see `inference.py` for the request format).

## Acknowledgements

This implementation builds on `verl` and follows the OVD paper’s WebQA setup (environment agent + on-policy trajectory sampling + verbal rejection sampling).
