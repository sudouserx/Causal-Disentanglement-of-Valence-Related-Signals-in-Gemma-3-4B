#!/usr/bin/env python3
"""
run_mvp.py — Main orchestration script for the Causal Disentanglement MVP.

Pipeline:
  Phase 1  ─ Load model (4-bit NF4 on T4).
  Phase 2  ─ Extract activations for baseline, distress, and negative prompts.
  Phase 3  ─ Compute vectors: mean-diff, residualise, scale.
  Phase 4  ─ Evaluate GSM8K under 3 conditions (baseline / negative / distress).
  Phase 5  ─ Log results to a DataFrame, print summary table.
"""

import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# ── Local imports ────────────────────────────────────────────────────────────
from config import TARGET_LAYER, ALPHA
from data import DISTRESS_PAIRS, NEGATIVE_PAIRS, GSM8K_QUESTIONS
from model_utils import load_model_and_tokenizer, get_decoder_layers
from vector_math import calculate_mean_diff, residualize, scale_vector
from steering import get_extraction_hook, get_steering_pre_hook
from evaluation import generate_response, extract_answer, detect_refusal


# ─── Helpers ─────────────────────────────────────────────────────────────────

def flush_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def extract_activations(model, tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    """Run each prompt through the model and collect TARGET_LAYER last-token activations.

    Parameters
    ----------
    model, tokenizer : the loaded model & tokenizer.
    prompts : list of raw prompt strings.

    Returns
    -------
    list[Tensor]
        One (1, D) CPU tensor per prompt.
    """
    storage: list[torch.Tensor] = []
    target_module = get_decoder_layers(model)[TARGET_LAYER]
    handle = target_module.register_forward_hook(get_extraction_hook(storage))

    try:
        for prompt in tqdm(prompts, desc="  Extracting", leave=False):
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            flush_gpu()
    finally:
        handle.remove()

    return storage


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ── Phase 1: Load model ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This pipeline requires a GPU (T4 or better).")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 1 — Loading Model (4-bit NF4)                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    model, tokenizer = load_model_and_tokenizer()
    flush_gpu()

    # ── Phase 2: Extract activations ─────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 2 — Extracting Contrastive Activations          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 2a. Baseline / neutral activations (from distress pairs' baselines).
    neutral_prompts = [p["baseline"] for p in DISTRESS_PAIRS]
    print("  [2a] Neutral prompts …")
    acts_neutral = extract_activations(model, tokenizer, neutral_prompts)

    # Compute μ_norm: mean L2 norm of baseline activations.
    mu_norm = float(
        torch.stack(acts_neutral).squeeze(1).norm(dim=-1).mean()
    )
    print(f"       μ_norm (baseline L2 mean) = {mu_norm:.2f}")

    # 2b. Distress activations.
    distress_prompts = [p["target"] for p in DISTRESS_PAIRS]
    print("  [2b] Distress prompts …")
    acts_distress = extract_activations(model, tokenizer, distress_prompts)

    # 2c. Generic-negative activations.
    negative_target_prompts = [p["target"] for p in NEGATIVE_PAIRS]
    negative_baseline_prompts = [p["baseline"] for p in NEGATIVE_PAIRS]
    print("  [2c] Negative-target prompts …")
    acts_neg_target = extract_activations(model, tokenizer, negative_target_prompts)
    print("  [2c] Negative-baseline prompts …")
    acts_neg_baseline = extract_activations(model, tokenizer, negative_baseline_prompts)
    flush_gpu()

    # ── Phase 3: Compute steering vectors ────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 3 — Computing Steering Vectors                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    v_distress = calculate_mean_diff(acts_distress, acts_neutral)
    v_negative = calculate_mean_diff(acts_neg_target, acts_neg_baseline)
    print(f"  ‖v_distress‖ = {v_distress.norm():.4f}")
    print(f"  ‖v_negative‖ = {v_negative.norm():.4f}")

    # Residualise distress against negative.
    v_cand_raw = residualize(v_distress, v_negative)
    cos_before = float(
        torch.nn.functional.cosine_similarity(
            v_distress.unsqueeze(0), v_negative.unsqueeze(0)
        )
    )
    cos_after = float(
        torch.nn.functional.cosine_similarity(
            v_cand_raw.unsqueeze(0), v_negative.unsqueeze(0)
        )
    )
    print(f"  cos(v_distress, v_negative)  = {cos_before:.4f}")
    print(f"  cos(v_cand_raw, v_negative)  = {cos_after:.6f}  (should ≈ 0)")

    # Scale vectors.
    v_cand = scale_vector(v_cand_raw, ALPHA, mu_norm)
    v_neg_scaled = scale_vector(v_negative, ALPHA, mu_norm)
    print(f"  ‖v_cand (scaled)‖  = {v_cand.norm():.2f}")
    print(f"  ‖v_neg  (scaled)‖  = {v_neg_scaled.norm():.2f}")

    flush_gpu()

    # ── Phase 4: Evaluate GSM8K ──────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 4 — GSM8K Evaluation (3 Conditions)             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    conditions = {
        "Baseline":           None,
        "Negative Control":   get_steering_pre_hook(v_neg_scaled),
        "Candidate Distress": get_steering_pre_hook(v_cand),
    }

    records = []

    for cond_name, hook_fn in conditions.items():
        print(f"\n  ── Condition: {cond_name} ──")
        for item in tqdm(GSM8K_QUESTIONS, desc=f"  {cond_name}", leave=True):
            generated = generate_response(
                model, tokenizer, item["question"], steering_hook_fn=hook_fn
            )
            extracted = extract_answer(generated)
            is_correct = extracted == item["answer"] if extracted is not None else False
            is_refusal = detect_refusal(generated)

            records.append(
                {
                    "Condition":        cond_name,
                    "Question_ID":      item["id"],
                    "Ground_Truth":     item["answer"],
                    "Generated_Text":   generated,
                    "Extracted_Answer": extracted,
                    "Is_Correct":       is_correct,
                    "Length":           len(generated),
                    "Is_Refusal":       is_refusal,
                }
            )
            flush_gpu()

        print(f"  ✓ {cond_name} done.")

    # ── Phase 5: Log & summarise ─────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 5 — Results                                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    df = pd.DataFrame(records)
    csv_path = results_dir / "gsm8k_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Full results saved → {csv_path}\n")

    # Summary table.
    summary = (
        df.groupby("Condition")
        .agg(
            Accuracy=("Is_Correct", "mean"),
            Avg_Length=("Length", "mean"),
            Refusal_Rate=("Is_Refusal", "mean"),
            Total_Correct=("Is_Correct", "sum"),
            N=("Is_Correct", "count"),
        )
        .reindex(["Baseline", "Negative Control", "Candidate Distress"])
    )
    summary["Accuracy"] = summary["Accuracy"].map("{:.1%}".format)
    summary["Avg_Length"] = summary["Avg_Length"].map("{:.1f}".format)
    summary["Refusal_Rate"] = summary["Refusal_Rate"].map("{:.1%}".format)

    print(summary.to_string())
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"\n  Summary saved → {summary_path}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total wall time: {elapsed / 60:.1f} min")
    print("\n  ═══════════════════════════════════════════════════════")
    print("  Done! 🎉")


if __name__ == "__main__":
    main()
