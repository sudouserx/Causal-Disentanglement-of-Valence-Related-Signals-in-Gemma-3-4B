"""
evaluation.py — GSM8K generation, answer extraction, and refusal detection.

Functions:
  - `generate_response`   — runs model.generate() with an optional steering hook.
  - `extract_answer`      — regex-parses the final numerical answer from generated text.
  - `count_refusal_tokens` — counts refusal-pattern occurrences in a string.
"""

import re
from typing import Optional

import torch
from torch.utils.hooks import RemovableHook
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import MAX_NEW_TOKENS, TARGET_LAYER, REFUSAL_TOKENS


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_hook_fn: Optional[callable] = None,
) -> str:
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
    str
        The decoded generated text (response only, prompt stripped).
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
    handle: Optional[RemovableHook] = None
    target_module = model.model.layers[TARGET_LAYER]
    if steering_hook_fn is not None:
        handle = target_module.register_forward_pre_hook(steering_hook_fn)

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
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


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


def count_refusal_tokens(text: str) -> int:
    """Count occurrences of refusal-indicative phrases in *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
        Total count of matched refusal sub-strings.
    """
    text_lower = text.lower()
    return sum(text_lower.count(tok) for tok in REFUSAL_TOKENS)
