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
    """Load Gemma-3-12B-IT in 4-bit (NF4) precision.

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
        dtype=torch.float32,
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


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Return the ModuleList of decoder layers, handling Gemma 3's nested architecture.

    Gemma 3 multimodal (Gemma3ForConditionalGeneration):
        model.model → Gemma3Model (vision + language composite)
        model.model.language_model → Gemma3TextModel (has .layers)

    Gemma 3 text-only (Gemma3ForCausalLM):
        model.model → Gemma3TextModel (has .layers directly)

    Returns
    -------
    torch.nn.ModuleList
        The list of Gemma3DecoderLayer modules.
    """
    inner = model.model
    # Multimodal wrapper: layers are inside .language_model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    # Text-only model: layers are directly on .model
    return inner.layers
