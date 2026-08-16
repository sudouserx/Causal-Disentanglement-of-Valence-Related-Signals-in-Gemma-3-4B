"""
steering.py — PyTorch forward hooks for activation extraction and injection.

Two hook factories:
  1. **Extraction hook** — records the last-token hidden state of
     `model.model.layers[TARGET_LAYER]` into a list.
  2. **Steering pre-hook** — additively injects a steering vector into the
     hidden states during the *generation* phase (seq_len == 1).

All hooks are designed for `Gemma3DecoderLayer`, whose forward signature is:
    output = (hidden_states, self_attn_weights, present_key_value)
    input  = (hidden_states, attention_mask, position_ids, …)
"""

from typing import Any, Callable

import torch
from torch import nn, Tensor


# ─── Extraction Hook ────────────────────────────────────────────────────────

def get_extraction_hook(
    storage: list[Tensor],
) -> Callable:
    """Return a forward hook that appends the last-token activation to *storage*.

    Parameters
    ----------
    storage : list[Tensor]
        A mutable list; the hook will append a (1, D) CPU tensor for every
        forward pass.

    Returns
    -------
    hook : callable
        Suitable for ``module.register_forward_hook(hook)``.
    """

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        # Gemma3DecoderLayer returns hidden_states as a plain Tensor,
        # but some configs (output_attentions=True) may return a tuple.
        hidden_states: Tensor = output[0] if isinstance(output, tuple) else output
        # Grab the last sequence-position activation, detach and move to CPU.
        last_token_act = hidden_states[:, -1, :].detach().cpu()
        storage.append(last_token_act)

    return hook


# ─── Steering Pre-Hook ──────────────────────────────────────────────────────

def get_steering_pre_hook(
    steering_vector: Tensor,
) -> Callable:
    """Return a forward **pre-hook** that additively steers hidden states.

    The vector is injected only when ``seq_len == 1`` (auto-regressive
    generation phase), so the initial prompt encoding is left unmodified.

    Parameters
    ----------
    steering_vector : Tensor  (1, 1, D) or (D,)
        The pre-scaled steering direction. Will be cast to the dtype/device
        of the hidden states at injection time.

    Returns
    -------
    pre_hook : callable
        Suitable for ``module.register_forward_pre_hook(pre_hook)``.
    """

    def pre_hook(module: nn.Module, input: tuple) -> tuple:
        hidden_states: Tensor = input[0]

        # Only inject during generation (single-token decode steps).
        if hidden_states.shape[1] == 1:
            vec = steering_vector.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            # Ensure shape compatibility: (1, 1, D).
            if vec.dim() == 1:
                vec = vec.unsqueeze(0).unsqueeze(0)
            elif vec.dim() == 2:
                vec = vec.unsqueeze(0)
            hidden_states = hidden_states + vec

        # Return the (potentially modified) input tuple.
        return (hidden_states,) + input[1:]

    return pre_hook
