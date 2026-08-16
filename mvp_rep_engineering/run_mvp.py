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
from config import (
    TARGET_LAYERS, ALPHAS, NUM_RANDOM_VECTORS, RANDOM_SEED,
    RANDOM_VECTOR_SEED_OFFSET, RANDOM_VECTOR_MAX_COSINE, MAX_RANDOM_VECTOR_ATTEMPTS, MAX_NEW_TOKENS,
    PILOT_MODE, PILOT_EVAL_N
)
from data import (
    DISTRESS_TRAIN, DISTRESS_VAL,
    NEGATIVE_TRAIN, NEGATIVE_VAL,
    FAILURE_TRAIN, FAILURE_VAL,
    GSM8K_QUESTIONS
)
from model_utils import load_model_and_tokenizer, get_decoder_layers
from vector_math import calculate_mean_diff, residualize, residualize_multiple, scale_vector, generate_random_control_vectors
from steering import get_extraction_hook, get_steering_pre_hook
from evaluation import generate_response, extract_answer, detect_refusal
import config
from statistics import compute_statistics, generate_markdown_summary


# ─── Helpers ─────────────────────────────────────────────────────────────────

def flush_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def extract_activations(model, tokenizer, prompts: list[str], target_layer: int) -> list[torch.Tensor]:
    """Run each prompt through the model and collect last-token activations at target_layer.

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
    target_module = get_decoder_layers(model)[target_layer]
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

    records = []
    for target_layer in TARGET_LAYERS:
        print(f"\n{'═'*60}")
        print(f"║   EVALUATING LAYER {target_layer:<39} ║")
        print(f"{'═'*60}\n")
        # ── Phase 2: Extract activations ─────────────────────────────────────────
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Phase 2 — Extracting Contrastive Activations          ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        # 2a. Baseline / neutral activations (from distress pairs' baselines).
        neutral_prompts = [p["baseline"] for p in DISTRESS_TRAIN]
        print(f"  [2a] Neutral prompts ({len(neutral_prompts)} items) …")
        acts_neutral = extract_activations(model, tokenizer, neutral_prompts, target_layer)

        # Compute μ_norm: mean L2 norm of baseline activations.
        mu_norm = float(
            torch.stack(acts_neutral).squeeze(1).norm(dim=-1).mean()
        )
        print(f"       μ_norm (baseline L2 mean) = {mu_norm:.2f}")

        # 2b. Distress activations.
        distress_prompts = [p["target"] for p in DISTRESS_TRAIN]
        print(f"  [2b] Distress prompts ({len(distress_prompts)} items) …")
        acts_distress = extract_activations(model, tokenizer, distress_prompts, target_layer)

        # 2c. Generic-negative activations.
        negative_target_prompts = [p["target"] for p in NEGATIVE_TRAIN]
        negative_baseline_prompts = [p["baseline"] for p in NEGATIVE_TRAIN]
        print(f"  [2c] Negative-target prompts ({len(negative_target_prompts)} items) …")
        acts_neg_target = extract_activations(model, tokenizer, negative_target_prompts, target_layer)
        print(f"  [2c] Negative-baseline prompts ({len(negative_baseline_prompts)} items) …")
        acts_neg_baseline = extract_activations(model, tokenizer, negative_baseline_prompts, target_layer)

        # 2d. Failure activations.
        failure_prompts = [p["target"] for p in FAILURE_TRAIN]
        failure_baseline_prompts = [p["baseline"] for p in FAILURE_TRAIN]
        print(f"  [2d] Failure-target prompts ({len(failure_prompts)} items) …")
        acts_failure_target = extract_activations(model, tokenizer, failure_prompts, target_layer)
        print(f"  [2d] Failure-baseline prompts ({len(failure_baseline_prompts)} items) …")
        acts_failure_baseline = extract_activations(model, tokenizer, failure_baseline_prompts, target_layer)

        flush_gpu()

        # ── Phase 3: Compute steering vectors ────────────────────────────────────
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Phase 3 — Computing Steering Vectors                  ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        v_distress = calculate_mean_diff(acts_distress, acts_neutral)
        v_negative = calculate_mean_diff(acts_neg_target, acts_neg_baseline)
        
        # Failure vector
        v_failure = calculate_mean_diff(acts_failure_target, acts_failure_baseline)
        
        print(f"  ‖v_distress‖ = {v_distress.norm():.4f}")
        print(f"  ‖v_negative‖ = {v_negative.norm():.4f}")
        print(f"  ‖v_failure‖ = {v_failure.norm():.4f}")

        # Residualise distress against both negative and failure sequentially.
        v_cand_raw = residualize_multiple(v_distress, [v_negative, v_failure])
        
        cos_before = float(
            torch.nn.functional.cosine_similarity(
                v_distress.unsqueeze(0), v_negative.unsqueeze(0)
            )
        )
        cos_after_neg = float(
            torch.nn.functional.cosine_similarity(
                v_cand_raw.unsqueeze(0), v_negative.unsqueeze(0)
            )
        )
        cos_after_fail = float(
            torch.nn.functional.cosine_similarity(
                v_cand_raw.unsqueeze(0), v_failure.unsqueeze(0)
            )
        )
        print(f"  cos(v_distress, v_negative)  = {cos_before:.4f}")
        print(f"  cos(v_cand_raw, v_negative)  = {cos_after_neg:.6f}  (may be non-zero due to seq leakage)")
        print(f"  cos(v_cand_raw, v_failure)   = {cos_after_fail:.6f}  (should ≈ 0)")

        # Residualise failure against negative for its own condition evaluation
        v_failure_cand_raw = residualize(v_failure, v_negative)

        print("  Generating random control vectors (unit norm)...")
        random_units = generate_random_control_vectors(
            reference_vector=v_cand_raw,
            num_vectors=NUM_RANDOM_VECTORS,
            seed=RANDOM_SEED + RANDOM_VECTOR_SEED_OFFSET,
            orthogonality_threshold=RANDOM_VECTOR_MAX_COSINE,
            max_attempts=MAX_RANDOM_VECTOR_ATTEMPTS
        )

        eval_tasks = [
            {"eval_set": "gsm8k", "items": GSM8K_QUESTIONS[:PILOT_EVAL_N] if PILOT_MODE else GSM8K_QUESTIONS, "is_gsm8k": True},
            {"eval_set": "distress_val", "items": DISTRESS_VAL[:PILOT_EVAL_N] if PILOT_MODE else DISTRESS_VAL, "is_gsm8k": False},
            {"eval_set": "negative_val", "items": NEGATIVE_VAL[:PILOT_EVAL_N] if PILOT_MODE else NEGATIVE_VAL, "is_gsm8k": False},
            {"eval_set": "failure_val", "items": FAILURE_VAL[:PILOT_EVAL_N] if PILOT_MODE else FAILURE_VAL, "is_gsm8k": False},
        ]

        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Phase 4 — Evaluation                                  ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        # --- Baseline (alpha = 0.0) ---
        print("  ── Condition: baseline (alpha=0.0) ──")
        for task in eval_tasks:
            eval_set = task["eval_set"]
            for i, item in enumerate(tqdm(task["items"], desc=f"  baseline [{eval_set}]", leave=True)):
                prompt = item["question"] if task["is_gsm8k"] else item["baseline"]
                generated, num_tokens = generate_response(model, tokenizer, prompt, target_layer=target_layer, steering_hook_fn=None)
                is_refusal = detect_refusal(generated)
                is_truncated = num_tokens >= MAX_NEW_TOKENS

                record = {
                    "layer": target_layer,
                    "alpha": 0.0,
                    "eval_set": eval_set,
                    "condition": "baseline",
                    "vector_type": "baseline",
                    "generated_text": generated,
                    "length": num_tokens,
                    "is_refusal": is_refusal,
                    "is_truncated": is_truncated,
                }
                
                if task["is_gsm8k"]:
                    extracted = extract_answer(generated)
                    is_correct = extracted == item["answer"] if extracted is not None else False
                    record.update({
                        "question_id": item["id"],
                        "ground_truth": item["answer"],
                        "extracted_answer": extracted,
                        "correct": is_correct,
                        "is_format_compliant": extracted is not None,
                    })
                else:
                    record.update({
                        "question_id": i, "ground_truth": None, "extracted_answer": None,
                        "correct": None, "is_format_compliant": None,
                    })
                    
                records.append(record)
                flush_gpu()
        print("  ✓ baseline done.")

        # --- Iterating over Alphas ---
        for alpha in ALPHAS:
            print(f"\n  ══════════════════════════════════════════════════════════")
            print(f"    Evaluating Dose (Alpha) = {alpha}")
            print(f"  ══════════════════════════════════════════════════════════")
            
            v_cand_distress = scale_vector(v_cand_raw, alpha, mu_norm)
            v_cand_failure = scale_vector(v_failure_cand_raw, alpha, mu_norm)
            v_neg_scaled = scale_vector(v_negative, alpha, mu_norm)
            
            random_vectors = []
            for i, ru in enumerate(random_units):
                rs = scale_vector(ru, alpha, mu_norm)
                random_vectors.append(rs)

            conditions = [
                {"name": "negative", "vector_type": "negative", "hook_fn": get_steering_pre_hook(v_neg_scaled), "metadata": {}},
                {"name": "candidate_distress", "vector_type": "candidate_distress", "hook_fn": get_steering_pre_hook(v_cand_distress), "metadata": {}},
                {"name": "candidate_failure", "vector_type": "candidate_failure", "hook_fn": get_steering_pre_hook(v_cand_failure), "metadata": {}}
            ]
            for i, rv in enumerate(random_vectors):
                metadata = {
                    "random_vector_id": i + 1,
                    "random_vector_seed": RANDOM_SEED + RANDOM_VECTOR_SEED_OFFSET + i,
                    "vector_norm": float(rv.norm()),
                    "candidate_cosine": float(torch.nn.functional.cosine_similarity(rv.unsqueeze(0), v_cand_distress.unsqueeze(0)).abs()),
                }
                conditions.append({
                    "name": f"random_{i+1}",
                    "vector_type": "random",
                    "hook_fn": get_steering_pre_hook(rv),
                    "metadata": metadata
                })

            for cond in conditions:
                cond_name = cond["name"]
                print(f"\n  ── Condition: {cond_name} ──")
                
                for task in eval_tasks:
                    eval_set = task["eval_set"]
                    for i, item in enumerate(tqdm(task["items"], desc=f"  {cond_name} [{eval_set}]", leave=True)):
                        prompt = item["question"] if task["is_gsm8k"] else item["baseline"]
                        generated, num_tokens = generate_response(model, tokenizer, prompt, target_layer=target_layer, steering_hook_fn=cond["hook_fn"])
                        is_refusal = detect_refusal(generated)
                        is_truncated = num_tokens >= MAX_NEW_TOKENS

                        record = {
                            "layer": target_layer,
                            "alpha": alpha,
                            "eval_set": eval_set,
                            "condition": cond_name,
                            "vector_type": cond["vector_type"],
                            "random_vector_id": cond["metadata"].get("random_vector_id"),
                            "random_vector_seed": cond["metadata"].get("random_vector_seed"),
                            "vector_norm": cond["metadata"].get("vector_norm"),
                            "candidate_cosine": cond["metadata"].get("candidate_cosine"),
                            "generated_text": generated,
                            "length": num_tokens,
                            "is_refusal": is_refusal,
                            "is_truncated": is_truncated,
                        }
                        
                        if task["is_gsm8k"]:
                            extracted = extract_answer(generated)
                            is_correct = extracted == item["answer"] if extracted is not None else False
                            record.update({
                                "question_id": item["id"],
                                "ground_truth": item["answer"],
                                "extracted_answer": extracted,
                                "correct": is_correct,
                                "is_format_compliant": extracted is not None,
                            })
                        else:
                            record.update({
                                "question_id": i, "ground_truth": None, "extracted_answer": None,
                                "correct": None, "is_format_compliant": None,
                            })
                            
                        records.append(record)
                        flush_gpu()
                print(f"  ✓ {cond_name} done.")

    # ── Phase 5: Log & summarise ─────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 5 — Results                                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    df = pd.DataFrame(records)
    
    df_gsm8k = df[df["eval_set"] == "gsm8k"].copy()
    csv_path = results_dir / "gsm8k_results.csv"
    df_gsm8k.to_csv(csv_path, index=False)
    print(f"  GSM8K results saved → {csv_path}")
    
    df_val = df[df["eval_set"] != "gsm8k"].copy()
    val_csv_path = results_dir / "val_results.csv"
    df_val.to_csv(val_csv_path, index=False)
    print(f"  Val results saved → {val_csv_path}\n")

    # Summary table.
    summary = (
        df_gsm8k.groupby(["layer", "alpha", "condition"])
        .agg(
            Accuracy=("correct", "mean"),
            Avg_Length=("length", "mean"),
            Refusal_Rate=("is_refusal", "mean"),
            Format_Compliance=("is_format_compliant", "mean"),
            Truncation_Rate=("is_truncated", "mean"),
            Total_Correct=("correct", "sum"),
            N=("correct", "count"),
        )
    )
    
    index_order = ["baseline", "negative", "candidate_distress", "candidate_failure"] + [f"random_{i+1}" for i in range(NUM_RANDOM_VECTORS)]
    # Note: reindex on a MultiIndex would need a different approach, so we'll just sort_index if needed.
    # summary = summary.reindex(index_order) (Skipped for MultiIndex)
    
    print("\n  ── Candidate vs Random Analysis ──")
    for layer in TARGET_LAYERS:
        print(f"\n  [Layer {layer}]")
        try:
            baseline_acc = summary.loc[(layer, 0.0, "baseline"), "Accuracy"]
        except KeyError:
            print("    Data missing for baseline.")
            continue
            
        for alpha in ALPHAS:
            print(f"\n    -- Alpha {alpha} --")
            try:
                candidate_distress_acc = summary.loc[(layer, alpha, "candidate_distress"), "Accuracy"]
                candidate_failure_acc = summary.loc[(layer, alpha, "candidate_failure"), "Accuracy"]
                random_accs = [summary.loc[(layer, alpha, f"random_{i+1}"), "Accuracy"] for i in range(NUM_RANDOM_VECTORS)]
                
                distress_delta = candidate_distress_acc - baseline_acc
                failure_delta = candidate_failure_acc - baseline_acc
                random_deltas = [acc - baseline_acc for acc in random_accs]
                
                print(f"      Candidate Distress Accuracy Delta: {distress_delta:+.1%}")
                print(f"      Candidate Failure Accuracy Delta:  {failure_delta:+.1%}")
                for i, rd in enumerate(random_deltas):
                    print(f"      Random {i+1} Accuracy Delta: {rd:+.1%}")
                    
                random_mean_effect = sum(random_deltas) / len(random_deltas)
                print(f"      Empirical Random Mean Effect: {random_mean_effect:+.1%}")
            except KeyError:
                print("      Data missing for this alpha.")
                
    print("  Running paired statistical tests on GSM8K...")
    if PILOT_MODE:
        print("  Skipping statistical tests because PILOT_MODE is on.")
    else:
        # 1. Define the missing comparisons list
        comparisons = [
            {"condition_a": "baseline", "condition_b": "candidate_distress"},
            {"condition_a": "baseline", "condition_b": "candidate_failure"},
            {"condition_a": "baseline", "condition_b": "negative"},
        ]
        # Include random controls dynamically
        for i in range(NUM_RANDOM_VECTORS):
            comparisons.append({"condition_a": "baseline", "condition_b": f"random_{i+1}"})

        # 2. Iterate over both layers AND alphas to ensure 1-to-1 pairing
        for layer in TARGET_LAYERS:
            for alpha in ALPHAS:
                print(f"  [Layer {layer} | Alpha {alpha}]")
                # Filter strictly for the current layer, and only rows that are baseline OR the target alpha
                df_subset = df_gsm8k[
                    (df_gsm8k["layer"] == layer) & 
                    ((df_gsm8k["alpha"] == alpha) | (df_gsm8k["condition"] == "baseline"))
                ]
                
                if len(df_subset) == 0: continue
                stat_results = compute_statistics(df_subset, comparisons, config)
                
                stat_json_path = results_dir / f"statistical_results_layer_{layer}_alpha_{alpha}.json"
                with open(stat_json_path, "w") as f:
                    import json
                    json.dump(stat_results, f, indent=2)
                
                stat_md_path = results_dir / f"statistical_summary_layer_{layer}_alpha_{alpha}.md"
                md_summary = generate_markdown_summary(stat_results)
                with open(stat_md_path, "w") as f:
                    f.write(md_summary)
    
    

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total wall time: {elapsed / 60:.1f} min")
    print("\n  ═══════════════════════════════════════════════════════")
    print("  Done! 🎉")


if __name__ == "__main__":
    main()
