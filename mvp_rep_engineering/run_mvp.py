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
    TARGET_LAYERS, ALPHA, NUM_RANDOM_VECTORS, RANDOM_SEED,
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
from vector_math import calculate_mean_diff, residualize, scale_vector, generate_random_control_vectors
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
        acts_neutral = extract_activations(model, tokenizer, neutral_prompts)

        # Compute μ_norm: mean L2 norm of baseline activations.
        mu_norm = float(
            torch.stack(acts_neutral).squeeze(1).norm(dim=-1).mean()
        )
        print(f"       μ_norm (baseline L2 mean) = {mu_norm:.2f}")

        # 2b. Distress activations.
        distress_prompts = [p["target"] for p in DISTRESS_TRAIN]
        print(f"  [2b] Distress prompts ({len(distress_prompts)} items) …")
        acts_distress = extract_activations(model, tokenizer, distress_prompts)

        # 2c. Generic-negative activations.
        negative_target_prompts = [p["target"] for p in NEGATIVE_TRAIN]
        negative_baseline_prompts = [p["baseline"] for p in NEGATIVE_TRAIN]
        print(f"  [2c] Negative-target prompts ({len(negative_target_prompts)} items) …")
        acts_neg_target = extract_activations(model, tokenizer, negative_target_prompts)
        print(f"  [2c] Negative-baseline prompts ({len(negative_baseline_prompts)} items) …")
        acts_neg_baseline = extract_activations(model, tokenizer, negative_baseline_prompts)

        # 2d. Failure activations.
        failure_prompts = [p["target"] for p in FAILURE_TRAIN]
        failure_baseline_prompts = [p["baseline"] for p in FAILURE_TRAIN]
        print(f"  [2d] Failure-target prompts ({len(failure_prompts)} items) …")
        acts_failure_target = extract_activations(model, tokenizer, failure_prompts)
        print(f"  [2d] Failure-baseline prompts ({len(failure_baseline_prompts)} items) …")
        acts_failure_baseline = extract_activations(model, tokenizer, failure_baseline_prompts)

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

        # Failure vector
        v_failure = calculate_mean_diff(acts_failure_target, acts_failure_baseline)
        v_failure_cand_raw = residualize(v_failure, v_negative)
        print(f"  ‖v_failure‖ = {v_failure.norm():.4f}")

        # Scale vectors.
        v_cand_distress = scale_vector(v_cand_raw, ALPHA, mu_norm)
        v_cand_failure = scale_vector(v_failure_cand_raw, ALPHA, mu_norm)
        v_neg_scaled = scale_vector(v_negative, ALPHA, mu_norm)
        print(f"  ‖v_cand_distress (scaled)‖ = {v_cand_distress.norm():.2f}")
        print(f"  ‖v_cand_failure (scaled)‖  = {v_cand_failure.norm():.2f}")
        print(f"  ‖v_neg (scaled)‖           = {v_neg_scaled.norm():.2f}")

        print("  Generating random control vectors...")
        random_units = generate_random_control_vectors(
            reference_vector=v_cand_distress,
            num_vectors=NUM_RANDOM_VECTORS,
            seed=RANDOM_SEED + RANDOM_VECTOR_SEED_OFFSET,
            orthogonality_threshold=RANDOM_VECTOR_MAX_COSINE,
            max_attempts=MAX_RANDOM_VECTOR_ATTEMPTS
        )

        random_vectors = []
        for i, ru in enumerate(random_units):
            rs = scale_vector(ru, ALPHA, mu_norm)
            random_vectors.append(rs)

            raw_random_norm = float(ru.norm())
            scaled_random_norm = float(rs.norm())
            print(f"  random_{i+1}: unit_norm={raw_random_norm:.2f}, scaled_norm={scaled_random_norm:.2f}")

            # Invariant check
            assert torch.allclose(
                rs.norm(), v_cand_distress.norm(), rtol=1e-5, atol=1e-6
            ), "Random vector norm does not match candidate norm!"

        print("\n  [Pairwise Random Vector Cosine Similarities]")
        for i in range(len(random_vectors)):
            for j in range(i + 1, len(random_vectors)):
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    random_vectors[i].unsqueeze(0), random_vectors[j].unsqueeze(0)
                ))
                print(f"    cos(random_{i+1}, random_{j+1}) = {cos_sim:.4f}")

        flush_gpu()

        # ── Phase 4: Evaluate GSM8K ──────────────────────────────────────────────
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Phase 4 — GSM8K Evaluation (3 Conditions)             ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        conditions = [
            {"name": "baseline", "vector_type": "baseline", "hook_fn": None, "metadata": {}},
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


        eval_tasks = [
            {"eval_set": "gsm8k", "items": GSM8K_QUESTIONS[:PILOT_EVAL_N] if PILOT_MODE else GSM8K_QUESTIONS, "is_gsm8k": True},
            {"eval_set": "distress_val", "items": DISTRESS_VAL[:PILOT_EVAL_N] if PILOT_MODE else DISTRESS_VAL, "is_gsm8k": False},
            {"eval_set": "negative_val", "items": NEGATIVE_VAL[:PILOT_EVAL_N] if PILOT_MODE else NEGATIVE_VAL, "is_gsm8k": False},
            {"eval_set": "failure_val", "items": FAILURE_VAL[:PILOT_EVAL_N] if PILOT_MODE else FAILURE_VAL, "is_gsm8k": False},
        ]

        for cond in conditions:
            cond_name = cond["name"]
            print(f"\n  ── Condition: {cond_name} ──")

            for task in eval_tasks:
                eval_set = task["eval_set"]
                for i, item in enumerate(tqdm(task["items"], desc=f"  {cond_name} [{eval_set}]", leave=True)):
                    if task["is_gsm8k"]:
                        prompt = item["question"]
                    else:
                        prompt = item["baseline"]

                    generated = generate_response(
                        model, tokenizer, prompt, steering_hook_fn=cond["hook_fn"]
                    )

                    is_refusal = detect_refusal(generated)
                    is_truncated = len(generated) > (MAX_NEW_TOKENS * 3)

                    record = {
                    "layer":            target_layer,
                    "eval_set":         eval_set,
                        "condition":        cond_name,
                        "vector_type":      cond["vector_type"],
                        "random_vector_id": cond["metadata"].get("random_vector_id"),
                        "random_vector_seed": cond["metadata"].get("random_vector_seed"),
                        "vector_norm":      cond["metadata"].get("vector_norm"),
                        "candidate_cosine": cond["metadata"].get("candidate_cosine"),
                        "generated_text":   generated,
                        "length":           len(generated),
                        "is_refusal":       is_refusal,
                        "is_truncated":     is_truncated,
                    }

                    if task["is_gsm8k"]:
                        extracted = extract_answer(generated)
                        is_correct = extracted == item["answer"] if extracted is not None else False
                        record.update({
                            "question_id":      item["id"],
                            "ground_truth":     item["answer"],
                            "extracted_answer": extracted,
                            "correct":          is_correct,
                            "is_format_compliant": extracted is not None,
                        })
                    else:
                        record.update({
                            "question_id":      i,
                            "ground_truth":     None,
                            "extracted_answer": None,
                            "correct":          None,
                            "is_format_compliant": None,
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
        df_gsm8k.groupby(["layer", "condition"])
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
            baseline_acc = summary.loc[(layer, "baseline"), "Accuracy"]
            candidate_distress_acc = summary.loc[(layer, "candidate_distress"), "Accuracy"]
            candidate_failure_acc = summary.loc[(layer, "candidate_failure"), "Accuracy"]
            random_accs = [summary.loc[(layer, f"random_{i+1}"), "Accuracy"] for i in range(NUM_RANDOM_VECTORS)]
            
            distress_delta = candidate_distress_acc - baseline_acc
            failure_delta = candidate_failure_acc - baseline_acc
            random_deltas = [acc - baseline_acc for acc in random_accs]
            
            print(f"    Candidate Distress Accuracy Delta: {distress_delta:+.1%}")
            print(f"    Candidate Failure Accuracy Delta:  {failure_delta:+.1%}")
            for i, rd in enumerate(random_deltas):
                print(f"    Random {i+1} Accuracy Delta: {rd:+.1%}")
                
            random_mean_effect = sum(random_deltas) / len(random_deltas)
            print(f"    Empirical Random Mean Effect: {random_mean_effect:+.1%}")
        except KeyError:
            print("    Data missing for this layer.")
        df_val.groupby(["layer", "eval_set", "condition"])
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 6 — Statistical Analysis                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    comparisons = [
        {"condition_a": "candidate_distress", "condition_b": "baseline"},
        {"condition_a": "candidate_distress", "condition_b": "negative"},
        {"condition_a": "candidate_distress", "condition_b": "candidate_failure"},
    ]
    for i in range(NUM_RANDOM_VECTORS):
        comparisons.append({"condition_a": "candidate_distress", "condition_b": f"random_{i+1}"})
        
    comparisons.extend([
        {"condition_a": "candidate_failure", "condition_b": "baseline"},
        {"condition_a": "candidate_failure", "condition_b": "negative"},
    ])

    print("  Running paired statistical tests on GSM8K...")
    if PILOT_MODE:
        print("  Skipping statistical tests because PILOT_MODE is on.")
    else:
        for layer in TARGET_LAYERS:
            print(f"  [Layer {layer}]")
            df_gsm8k_layer = df_gsm8k[df_gsm8k["layer"] == layer]
            if len(df_gsm8k_layer) == 0: continue
            stat_results = compute_statistics(df_gsm8k_layer, comparisons, config)
            
            stat_json_path = results_dir / f"statistical_results_layer_{layer}.json"
            with open(stat_json_path, "w") as f:
                import json
                json.dump(stat_results, f, indent=2)
            
            stat_md_path = results_dir / f"statistical_summary_layer_{layer}.md"
            md_summary = generate_markdown_summary(stat_results)
            with open(stat_md_path, "w") as f:
                f.write(md_summary)
    
    

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total wall time: {elapsed / 60:.1f} min")
    print("\n  ═══════════════════════════════════════════════════════")
    print("  Done! 🎉")


if __name__ == "__main__":
    main()
