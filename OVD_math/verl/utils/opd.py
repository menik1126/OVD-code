# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for online policy distillation (OPD).

The functions in this module deliberately keep teacher I/O separate from the
PPO update.  Streaming changes *when* a trajectory is scored, never which
trajectory belongs to an update or its position inside the rollout batch.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


_SCORING_PROMPT = """You are a STRICT mathematical examiner. Evaluate the reasoning process with VERY HIGH STANDARDS.

Check for ALL of these errors:
- Calculation errors: wrong arithmetic, sign errors, algebraic mistakes
- Unit errors: missing units, wrong units, inconsistent units
- Symbol errors: undefined variables, inconsistent notation, wrong symbols
- Logic errors: unjustified steps, skipped reasoning, circular logic
- Method errors: wrong formula, inappropriate approach, missing cases
- Precision errors: premature rounding, significant figure mistakes

SCORING CRITERIA (BE HARSH - most solutions should score 3-5):
- 9: PERFECT. Flawless solution, every step correct, optimal method, clear explanation, correct units/symbols throughout. Extremely rare.
- 8: EXCELLENT. All steps correct with at most one trivial formatting issue. No calculation or logic errors.
- 7: VERY GOOD. Correct reasoning with 1 minor issue (slightly unclear step or minor notation inconsistency).
- 6: GOOD. Sound approach but has 1-2 small errors (minor calculation slip or missing unit in one place).
- 5: ACCEPTABLE. Correct method but has 2-3 minor errors OR 1 notable error (wrong intermediate calculation).
- 4: MEDIOCRE. Has significant issues: multiple calculation errors OR one major logical gap OR unclear reasoning.
- 3: WEAK. Flawed reasoning with several errors, OR correct approach but poorly executed with many mistakes.
- 2: POOR. Wrong method OR major conceptual error OR multiple serious calculation mistakes.
- 1: VERY POOR. Fundamentally flawed approach, most steps wrong, little correct reasoning.
- 0: FAILED. Completely wrong, nonsensical, or no meaningful attempt at reasoning.

STRICT RULES:
- Any calculation error: maximum score 5
- Wrong final answer due to reasoning errors: maximum score 4
- Wrong method/formula: maximum score 3
- Multiple logic errors: maximum score 3

Do NOT be lenient. Find ALL faults, then give a score from 0-9.

---
Question: {question}

Student's solution:
{response}
---

Now evaluate the above solution strictly."""


_GENERATION_PROMPT = """
Example format:
Question: What is 15% of 80?
Solution: To find 15% of 80, I need to multiply 80 by 0.15.
80 × 0.15 = 12
Therefore, the answer is \\boxed{{12}}

Question: {question}
Please solve this math Question above step by step. Put your final answer in \\boxed{{}} format."""


_SUFFIX_GENERATION_PROMPT = """Question: {question}

Continue directly from the partial assistant response that follows after the assistant marker. Do not restart or repeat the solution, and do not emit a new <think> block. If the partial response contains an unclosed <think> block, continue its reasoning, close it exactly once with </think>, and then give the final answer in \\boxed{{}} format. If its think block is already closed, continue only the answer. If it contains no think block, continue without adding <think> or </think> markers. Use at most {max_tokens} new tokens."""


_DEEPSEEK_R1_BOS = "<｜begin▁of▁sentence｜>"
_DEEPSEEK_R1_USER = "<｜User｜>"
_DEEPSEEK_R1_ASSISTANT = "<｜Assistant｜>"
_OPD_STRUCTURE_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


class SuffixCompletionError(RuntimeError):
    """A teacher suffix is syntactically unusable but may succeed on retry."""


def normalize_math_response_text(response: str) -> str:
    """Match the trainer's historical ``</answer>`` post-processing."""
    if "</answer>" in response:
        return response.split("</answer>", 1)[0] + "</answer>"
    return response


def extract_score_from_logprobs(response_obj, default_score: int = 5) -> int:
    """Extract the most likely one-token integer score in [0, 9]."""
    top_logprobs = response_obj.choices[0].logprobs.content[0].top_logprobs
    scores = {}
    for logprob_obj in top_logprobs:
        token = logprob_obj.token.strip()
        if token.isdigit() and 0 <= int(token) <= 9:
            scores[int(token)] = logprob_obj.logprob
    if not scores:
        return default_score
    return max(scores.items(), key=lambda item: item[1])[0]


def score_math_response(client, model: str, question: str, response: str) -> int:
    """Score one response with the repository's original verbal OPD rubric."""
    messages = [
        {"role": "user", "content": _SCORING_PROMPT.format(question=question, response=response)},
        {
            "role": "assistant",
            "content": "<think>As a strict examiner, I will check every step for errors.\n\nThe final score is ",
        },
    ]
    response_obj = client.chat.completions.create(
        model=model,
        max_tokens=1,
        temperature=0.0,
        messages=messages,
        logprobs=True,
        top_logprobs=10,
        extra_body={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )
    return extract_score_from_logprobs(response_obj)


def regenerate_math_response(client, model: str, question: str, temperature: float, max_tokens: int) -> str:
    """Generate a complete replacement, including the historical ``<think>`` prefix."""
    response_obj = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": _GENERATION_PROMPT.format(question=question)},
            {"role": "assistant", "content": "<think>"},
        ],
        extra_body={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )
    return "<think>" + response_obj.choices[0].message.content


def continue_truncated_math_response(
    client,
    model: str,
    question: str,
    partial_response: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Continue one teacher response without restarting its reasoning."""
    if not partial_response:
        raise ValueError("partial_response must be non-empty")
    response_obj = client.completions.create(
        model=model,
        prompt=render_math_suffix_prompt(question, partial_response, max_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return partial_response + response_obj.choices[0].text


def render_math_suffix_prompt(
    question: str,
    prefix_text: str,
    max_tokens: int,
) -> str:
    """Render the DeepSeek-R1-Distill-Qwen prompt followed by an assistant prefill."""
    user_text = _SUFFIX_GENERATION_PROMPT.format(
        question=question,
        max_tokens=max_tokens,
    )
    return (
        _DEEPSEEK_R1_BOS
        + _DEEPSEEK_R1_USER
        + user_text
        + _DEEPSEEK_R1_ASSISTANT
        + prefix_text
    )


def validate_math_suffix_completion(prefix_text: str, continuation: str) -> None:
    """Reject malformed continuations at a retained ``<think>`` boundary."""
    if not continuation:
        raise SuffixCompletionError("teacher returned an empty suffix continuation")

    prefix_open = prefix_text.count("<think>")
    prefix_close = prefix_text.count("</think>")
    if prefix_close > prefix_open or prefix_open - prefix_close > 1:
        raise RuntimeError("actor prefix contains malformed think delimiters")

    if prefix_open == prefix_close + 1:
        if continuation.count("<think>") != 0 or continuation.count("</think>") != 1:
            raise SuffixCompletionError(
                "teacher must close the retained think block exactly once"
            )
        post_think = continuation.split("</think>", 1)[1]
        if not post_think.strip():
            raise SuffixCompletionError(
                "teacher closed the think block without a final answer"
            )
    elif prefix_close:
        if "<think>" in continuation or "</think>" in continuation:
            raise SuffixCompletionError(
                "teacher emitted a second think block after a retained </think>"
            )
    elif "<think>" in continuation or "</think>" in continuation:
        raise SuffixCompletionError(
            "teacher emitted think markers after an untagged actor prefix"
        )


def validate_math_replacement_text(response_text: str) -> None:
    """Validate the final actor-tokenized reward view after suffix splicing."""
    if not response_text.strip():
        raise SuffixCompletionError("teacher replacement became empty after tokenization")
    think_open = response_text.count("<think>")
    think_close = response_text.count("</think>")
    if think_open != think_close or think_open > 1:
        raise SuffixCompletionError(
            "teacher replacement contains an unbalanced or repeated think block"
        )
    if think_close and not response_text.split("</think>", 1)[1].strip():
        raise SuffixCompletionError(
            "teacher replacement has no final answer after the think block"
        )


def regenerate_math_suffix(
    client,
    model: str,
    question: str,
    prefix_text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate only the continuation after an actor-token prefix.

    The caller retains the original prefix token IDs.  Consequently a decoded
    prefix is used only as teacher conditioning and is never re-tokenized into
    the final actor trajectory.
    """
    # ``continue_final_message`` passes the prefix through the model's chat
    # template, which deletes everything before the final </think>.  Raw
    # completions preserve the actor prefix byte-for-byte as model context.
    conditioning_prefix = prefix_text or "<think>\n"
    response_obj = client.completions.create(
        model=model,
        prompt=render_math_suffix_prompt(question, conditioning_prefix, max_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    continuation = response_obj.choices[0].text
    # DeepSeek-R1 emits one native </think> separator even when the retained
    # Qwen actor prefix is untagged.  Remove only that structural separator;
    # all generated reasoning and answer text remain unchanged.
    if (
        "<think>" not in conditioning_prefix
        and "</think>" not in conditioning_prefix
        and "<think>" not in continuation
        and continuation.count("</think>") == 1
    ):
        continuation = continuation.replace("</think>", "", 1)
    # Validate against the prefix that was actually sent to the teacher.  In
    # particular, an empty actor prefix is conditioned with ``<think>\n`` and
    # must therefore produce exactly one closing ``</think>`` plus an answer.
    validate_math_suffix_completion(conditioning_prefix, continuation)
    if not prefix_text:
        return conditioning_prefix + continuation
    return continuation


class TeacherEvaluator:
    """Thread-local OpenAI-compatible clients for bounded parallel teacher I/O."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 600.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._local = threading.local()
        self._clients = []
        self._clients_lock = threading.Lock()

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            self._local.client = client
            with self._clients_lock:
                self._clients.append(client)
        return client

    def score(self, question: str, response: str) -> int:
        return score_math_response(self._client(), self.model, question, response)

    def regenerate_full(self, question: str, temperature: float, max_tokens: int) -> str:
        return regenerate_math_response(self._client(), self.model, question, temperature, max_tokens)

    def continue_full(
        self,
        question: str,
        partial_response: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return continue_truncated_math_response(
            self._client(),
            self.model,
            question,
            partial_response,
            temperature,
            max_tokens,
        )

    def regenerate_suffix(self, question: str, prefix_text: str, temperature: float, max_tokens: int) -> str:
        return regenerate_math_suffix(
            self._client(), self.model, question, prefix_text, temperature, max_tokens
        )

    def close(self) -> None:
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()


@dataclass(frozen=True)
class StreamingScoreStats:
    submitted: int
    first_submit_after_s: Optional[float]
    wait_after_rollout_s: float


class StreamingTeacherScorer:
    """Observe vLLM outputs and score each trajectory as soon as it finishes."""

    def __init__(self, tokenizer, evaluator: TeacherEvaluator, max_workers: int):
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="opd-teacher")
        self._futures: Dict[Tuple[int, int], Future] = {}
        self._final_order: Optional[List[Tuple[int, int]]] = None
        self._started_at = time.perf_counter()
        self._first_submit_at: Optional[float] = None
        self._closed = False

    @staticmethod
    def _key(request_output, completion) -> Tuple[int, int]:
        return int(request_output.request_id), int(completion.index)

    def _submit(self, request_output, completion) -> None:
        key = self._key(request_output, completion)
        if key in self._futures:
            return
        question = self.tokenizer.decode(request_output.prompt_token_ids, skip_special_tokens=True)
        response = self.tokenizer.decode(completion.token_ids, skip_special_tokens=True)
        response = normalize_math_response_text(response)
        if self._first_submit_at is None:
            self._first_submit_at = time.perf_counter()
        self._futures[key] = self.executor.submit(self.evaluator.score, question, response)

    def on_step(self, step_outputs: Iterable) -> None:
        """Submit newly finished completions without blocking vLLM stepping."""
        for request_output in step_outputs:
            if not hasattr(request_output, "outputs"):
                continue
            for completion in request_output.outputs:
                if request_output.finished or getattr(completion, "finish_reason", None) is not None:
                    self._submit(request_output, completion)

    def on_final_outputs(self, request_outputs: Sequence) -> None:
        """Capture exact vLLM flattening order and backfill any missed completion."""
        order = []
        for request_output in request_outputs:
            for completion in request_output.outputs:
                self._submit(request_output, completion)
                order.append(self._key(request_output, completion))
        self._final_order = order

    def finalize(self) -> Tuple[List[int], StreamingScoreStats]:
        if self._final_order is None:
            raise RuntimeError("vLLM did not provide final request order to the OPD observer")
        wait_started_at = time.perf_counter()
        scores = [int(self._futures[key].result()) for key in self._final_order]
        wait_after_rollout_s = time.perf_counter() - wait_started_at
        first_submit_after_s = (
            None if self._first_submit_at is None else self._first_submit_at - self._started_at
        )
        stats = StreamingScoreStats(
            submitted=len(self._futures),
            first_submit_after_s=first_submit_after_s,
            wait_after_rollout_s=wait_after_rollout_s,
        )
        return scores, stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.evaluator.close()


def response_structure_token_mask(
    response_ids: torch.Tensor,
    tokenizer,
    structure_tags: Sequence[str] = _OPD_STRUCTURE_TAGS,
) -> torch.Tensor:
    """Mark tokenizer specials and tokens whose decoded spans overlap structure tags."""
    if response_ids.ndim != 2:
        raise ValueError(f"response_ids must be 2-D, got shape {tuple(response_ids.shape)}")

    excluded = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in set(getattr(tokenizer, "all_special_ids", ()) or ()):
        excluded |= response_ids.eq(int(token_id))

    response_rows = response_ids.detach().cpu().tolist()
    for row_idx, original_ids in enumerate(response_rows):
        token_pieces = tokenizer.convert_ids_to_tokens(
            original_ids,
            skip_special_tokens=False,
        )
        if len(token_pieces) != len(original_ids):
            raise RuntimeError("tokenizer did not return one token piece per response ID")
        encoded_text = "".join(token_pieces)
        tag_spans = []
        for tag in structure_tags:
            search_from = 0
            while True:
                start = encoded_text.find(tag, search_from)
                if start < 0:
                    break
                tag_spans.append((start, start + len(tag)))
                search_from = start + len(tag)
        token_start = 0
        for token_idx, token_piece in enumerate(token_pieces):
            token_end = token_start + len(token_piece)
            if any(token_start < tag_end and token_end > tag_start for tag_start, tag_end in tag_spans):
                excluded[row_idx, token_idx] = True
            token_start = token_end
    return excluded


def max_entropy_token_indices(
    entropies: torch.Tensor,
    response_mask: torch.Tensor,
    excluded_token_mask: Optional[torch.Tensor] = None,
    max_start_index: Optional[int] = None,
) -> torch.Tensor:
    """Return the feasible argmax over valid, non-structural response tokens."""
    if entropies.shape != response_mask.shape:
        raise ValueError(f"entropy/mask shape mismatch: {entropies.shape} vs {response_mask.shape}")
    valid = response_mask.to(dtype=torch.bool)
    if excluded_token_mask is not None:
        if excluded_token_mask.shape != valid.shape:
            raise ValueError(
                "excluded-token/mask shape mismatch: "
                f"{excluded_token_mask.shape} vs {valid.shape}"
            )
        valid &= ~excluded_token_mask.to(device=valid.device, dtype=torch.bool)
    if max_start_index is not None:
        if not 0 <= max_start_index < valid.shape[1]:
            raise ValueError(
                f"max_start_index {max_start_index} is outside [0, {valid.shape[1]})"
            )
        positions = torch.arange(valid.shape[1], device=valid.device)
        valid &= positions.unsqueeze(0) <= max_start_index
    if not torch.all(valid.any(dim=-1)):
        raise ValueError("every trajectory must contain at least one replaceable response token")
    if not torch.isfinite(entropies[valid]).all():
        raise ValueError("replaceable response-token entropies must all be finite")
    masked = entropies.float().masked_fill(~valid, float("-inf"))
    return masked.argmax(dim=-1)


def random_token_indices(
    response_mask: torch.Tensor,
    excluded_token_mask: Optional[torch.Tensor] = None,
    max_start_index: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Uniformly sample one feasible, non-structural response token per row."""
    if response_mask.ndim != 2:
        raise ValueError(
            f"response_mask must be 2-D, got shape {tuple(response_mask.shape)}"
        )
    valid = response_mask.to(dtype=torch.bool)
    if excluded_token_mask is not None:
        if excluded_token_mask.shape != valid.shape:
            raise ValueError(
                "excluded-token/mask shape mismatch: "
                f"{excluded_token_mask.shape} vs {valid.shape}"
            )
        valid &= ~excluded_token_mask.to(device=valid.device, dtype=torch.bool)
    if max_start_index is not None:
        if not 0 <= max_start_index < valid.shape[1]:
            raise ValueError(
                f"max_start_index {max_start_index} is outside [0, {valid.shape[1]})"
            )
        positions = torch.arange(valid.shape[1], device=valid.device)
        valid &= positions.unsqueeze(0) <= max_start_index
    if not torch.all(valid.any(dim=-1)):
        raise ValueError("every trajectory must contain at least one replaceable response token")

    # Sampling on CPU makes the configured seed independent of CUDA/ROCm
    # generator state and reproducible when a checkpoint resumes the same step.
    valid_cpu = valid.detach().cpu()
    sampled = []
    for row in valid_cpu:
        candidates = torch.nonzero(row, as_tuple=False).flatten()
        candidate_offset = torch.randint(
            candidates.numel(),
            (1,),
            generator=generator,
        ).item()
        sampled.append(int(candidates[candidate_offset].item()))
    return torch.tensor(sampled, dtype=torch.long, device=response_mask.device)


def splice_response_suffix(
    original_response: torch.Tensor,
    suffix_ids: torch.Tensor,
    start_index: int,
    target_length: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Keep ``original[:start_index]`` and replace from start_index onward."""
    if original_response.ndim != 1 or suffix_ids.ndim != 1:
        raise ValueError("original_response and suffix_ids must both be 1-D")
    if not 0 <= start_index < target_length:
        raise ValueError(f"start_index {start_index} is outside [0, {target_length})")
    prefix = original_response[:start_index]
    room = target_length - start_index
    combined = torch.cat((prefix, suffix_ids.to(device=original_response.device)[:room]), dim=0)
    if combined.numel() < target_length:
        padding = torch.full(
            (target_length - combined.numel(),),
            pad_token_id,
            dtype=original_response.dtype,
            device=original_response.device,
        )
        combined = torch.cat((combined, padding), dim=0)
    return combined[:target_length]
