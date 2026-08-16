"""
model_utils.py — 4-bit model loading and tokenizer setup for T4 (16 GB VRAM).

Uses BitsAndBytes 4-bit NF4 quantisation with `device_map="auto"` so layers
are automatically placed across available devices.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import MODEL_ID


def load_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load Gemma-3-4B-IT in 4-bit (NF4) precision.

    Returns
    -------
    model : PreTrainedModel
        The quantised model in eval mode.
    tokenizer : PreTrainedTokenizerBase
        The associated tokenizer with left-padding for batch generation.
    """
    print(f"[model_utils] Loading {MODEL_ID} in 4-bit …")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Some models lack an explicit pad token; fall back to EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-padding is required for correct causal-LM batch generation.
    tokenizer.padding_side = "left"

    print("[model_utils] Model and tokenizer loaded successfully.")
    return model, tokenizer
