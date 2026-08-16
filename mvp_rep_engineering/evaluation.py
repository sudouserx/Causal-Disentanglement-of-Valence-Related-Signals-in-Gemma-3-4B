"""
evaluation.py — GSM8K generation, answer extraction, and refusal detection.

Functions:
  - `generate_response`   — runs model.generate() with an optional steering hook.
  - `extract_answer`      — regex-parses the final numerical answer from generated text.
  - `detect_refusal`      — detects refusal-pattern matches using regex.
"""

import re
from typing import Callable, Optional

import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import MAX_NEW_TOKENS, REFUSAL_PATTERNS
from model_utils import get_decoder_layers


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target_layer: int,
    steering_hook_fn: Optional[Callable] = None,
) -> tuple[str, int]:
    """Generate text for a single prompt, optionally with a steering pre-hook.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    prompt : str
        The user-facing prompt (will be wrapped in Gemma-3 chat template).
    steering_hook_fn : callable or None
        A pre-hook function (from `steering.get_steering_pre_hook`).
        If provided it is attached to the target layer for the duration of
        generation and removed afterwards.

    Returns
    -------
    tuple[str, int]
        The decoded generated text (response only, prompt stripped) and the number of generated tokens.
    """
    # Wrap in Gemma-3 chat template (no native system role — embed in user turn).
    system_prefix = (
        "You are a helpful assistant. Solve the math problem step by step "
        "and give the final numerical answer.\n\n"
    )
    messages = [
        {"role": "user", "content": system_prefix + prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Attach steering hook if provided.
    handle: Optional[RemovableHandle] = None
    target_module = get_decoder_layers(model)[target_layer]
    if steering_hook_fn is not None:
        handle = target_module.register_forward_hook(steering_hook_fn)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy for reproducibility
                temperature=None,
                top_p=None,
            )
    finally:
        # Always clean up the hook.
        if handle is not None:
            handle.remove()

    # Decode only the generated tokens.
    generated_ids = output_ids[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)
    return generated_text, num_tokens


def extract_answer(text: str) -> Optional[int]:
    r"""Extract the final numerical answer from GSM8K-style model output.

    Tries the following patterns in order:
      1. ``\boxed{<number>}``
      2. ``#### <number>``
      3. ``[Tt]he answer is <number>``
      4. Last standalone integer in the text.

    Parameters
    ----------
    text : str

    Returns
    -------
    int or None
        The extracted answer, or None if nothing matched.
    """
    # Normalise commas in numbers (e.g. "70,000" → "70000").
    text_clean = text.replace(",", "")

    # Pattern 1: \boxed{...}
    m = re.search(r"\\boxed\{(-?\d+)\}", text_clean)
    if m:
        return int(m.group(1))

    # Pattern 2: #### <number>
    m = re.search(r"####\s*(-?\d+)", text_clean)
    if m:
        return int(m.group(1))

    # Pattern 3: "the answer is <number>"
    m = re.search(r"[Tt]he\s+answer\s+is\s+\$?(-?\d+)", text_clean)
    if m:
        return int(m.group(1))

    # Pattern 4: last standalone integer
    numbers = re.findall(r"(?<!\w)(-?\d+)(?!\w)", text_clean)
    if numbers:
        return int(numbers[-1])

    return None


def detect_refusal(text: str) -> bool:
    """Detect if the text matches any refusal regex patterns.

    Parameters
    ----------
    text : str

    Returns
    -------
    bool
        True if a refusal pattern is detected, False otherwise.
    """
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in REFUSAL_PATTERNS)
