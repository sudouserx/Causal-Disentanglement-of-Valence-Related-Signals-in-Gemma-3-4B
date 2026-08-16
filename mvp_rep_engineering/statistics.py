import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def _paired_join(df: pd.DataFrame, condition_a: str, condition_b: str, value_col: str, prompt_col: str) -> pd.DataFrame:
    """Safe inner join to ensure perfect pairing."""
    df_a = df[df["condition"] == condition_a][[prompt_col, value_col]].copy()
    df_b = df[df["condition"] == condition_b][[prompt_col, value_col]].copy()
    
    # Check uniqueness
    if df_a[prompt_col].duplicated().any():
        raise ValueError(f"Duplicate prompts found for condition {condition_a}")
    if df_b[prompt_col].duplicated().any():
        raise ValueError(f"Duplicate prompts found for condition {condition_b}")
        
    df_a = df_a.dropna(subset=[value_col])
    df_b = df_b.dropna(subset=[value_col])

    paired = df_a.merge(df_b, on=prompt_col, how="inner", suffixes=("_a", "_b"))
    return paired


def run_mcnemar_exact(df: pd.DataFrame, condition_a: str, condition_b: str, outcome_col: str = "correct", prompt_col: str = "question_id") -> Dict[str, Any]:
    """Calculate exact McNemar p-value for paired binary correctness."""
    paired = _paired_join(df, condition_a, condition_b, outcome_col, prompt_col)
    n_pairs = len(paired)
    
    if n_pairs == 0:
        raise ValueError(f"No valid pairs between {condition_a} and {condition_b} for {outcome_col}")
        
    correct_a = paired[f"{outcome_col}_a"].astype(bool)
    correct_b = paired[f"{outcome_col}_b"].astype(bool)
    
    n11 = (correct_a & correct_b).sum()
    n00 = (~correct_a & ~correct_b).sum()
    n10 = (correct_a & ~correct_b).sum()
    n01 = (~correct_a & correct_b).sum()
    
    discordant = int(n10 + n01)
    
    if discordant == 0:
        p_value = 1.0
    else:
        # Exact binomial test for minimum of discordant pairs
        test = scipy.stats.binomtest(min(n10, n01), discordant, p=0.5)
        p_value = test.pvalue
        
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    
    # Handle odds ratio safely
    if n01 == 0 and n10 == 0:
        odds_ratio = float('nan')
    elif n01 == 0:
        odds_ratio = float('inf')
    else:
        odds_ratio = n10 / n01

    return {
        "test": "mcnemar_exact",
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_pairs": int(n_pairs),
        "a_accuracy": float(acc_a),
        "b_accuracy": float(acc_b),
        "difference": float(acc_a - acc_b),
        "discordant_a_correct_b_incorrect": int(n10),
        "discordant_a_incorrect_b_correct": int(n01),
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }

def run_paired_permutation(df: pd.DataFrame, condition_a: str, condition_b: str, value_col: str = "length", prompt_col: str = "question_id", n_permutations: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Calculate paired permutation test for numeric length differences."""
    paired = _paired_join(df, condition_a, condition_b, value_col, prompt_col)
    n_pairs = len(paired)
    
    if n_pairs <= 1:
        raise ValueError(f"Insufficient valid pairs (n={n_pairs}) between {condition_a} and {condition_b} for {value_col}")
        
    val_a = paired[f"{value_col}_a"].values
    val_b = paired[f"{value_col}_b"].values
    diffs = val_a - val_b
    
    mean_a = float(val_a.mean())
    mean_b = float(val_b.mean())
    
    t_obs = diffs.mean()
    median_diff = float(np.median(diffs))
    
    if np.all(diffs == 0):
        return {
            "test": "paired_permutation",
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_pairs": int(n_pairs),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference": float(t_obs),
            "median_difference": median_diff,
            "p_value": 1.0,
            "status": "degenerate_zero_difference"
        }

    rng = np.random.default_rng(seed)
    
    # +1 or -1 multiplier
    signs = rng.choice([-1, 1], size=(n_permutations, n_pairs))
    permuted_diffs = diffs * signs
    permuted_means = permuted_diffs.mean(axis=1)
    
    # two-sided
    count_extreme = np.sum(np.abs(permuted_means) >= np.abs(t_obs))
    p_value = (1 + count_extreme) / (n_permutations + 1)
    
    return {
        "test": "paired_permutation",
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_pairs": int(n_pairs),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference": float(t_obs),
        "median_difference": median_diff,
        "p_value": float(p_value),
        "status": "ok"
    }

def bootstrap_paired_difference(df: pd.DataFrame, condition_a: str, condition_b: str, value_col: str, prompt_col: str = "question_id", n_bootstraps: int = 10000, confidence: float = 0.95, seed: int = 42) -> Dict[str, Any]:
    """Calculate bootstrap CI for the mean paired difference."""
    paired = _paired_join(df, condition_a, condition_b, value_col, prompt_col)
    n_pairs = len(paired)
    
    if n_pairs <= 1:
        raise ValueError(f"Insufficient valid pairs (n={n_pairs}) between {condition_a} and {condition_b} for {value_col}")
        
    val_a = paired[f"{value_col}_a"].values
    val_b = paired[f"{value_col}_b"].values
    diffs = val_a - val_b
    
    estimate = float(diffs.mean())
    
    rng = np.random.default_rng(seed)
    # Resample indices with replacement
    indices = rng.integers(0, n_pairs, size=(n_bootstraps, n_pairs))
    resampled_diffs = diffs[indices]
    resampled_means = resampled_diffs.mean(axis=1)
    
    alpha = 1.0 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1.0 - alpha / 2) * 100
    
    ci_low = float(np.percentile(resampled_means, lower_pct))
    ci_high = float(np.percentile(resampled_means, upper_pct))
    
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": float(confidence),
        "bootstrap_samples": int(n_bootstraps),
        "n_pairs": int(n_pairs)
    }

def compute_statistics(df: pd.DataFrame, comparisons: List[Dict[str, str]], config: Any) -> Dict[str, Any]:
    """Run full statistical suite for specified comparisons."""
    seed = config.RANDOM_SEED + getattr(config, 'STATISTICS_SEED_OFFSET', 200)
    n_permutations = getattr(config, 'N_PERMUTATIONS', 10000)
    n_bootstraps = getattr(config, 'N_BOOTSTRAPS', 10000)
    confidence = getattr(config, 'BOOTSTRAP_CONFIDENCE', 0.95)
    
    results = {
        "metadata": {
            "seed": seed,
            "n_permutations": n_permutations,
            "n_bootstraps": n_bootstraps,
            "confidence_level": confidence,
            "primary_alpha": getattr(config, 'PRIMARY_ALPHA', 0.05)
        },
        "comparisons": []
    }
    
    for comp in comparisons:
        cond_a = comp["condition_a"]
        cond_b = comp["condition_b"]
        
        # Correctness
        mcnemar_res = run_mcnemar_exact(df, cond_a, cond_b, "correct", "question_id")
        boot_corr_res = bootstrap_paired_difference(df, cond_a, cond_b, "correct", "question_id", n_bootstraps, confidence, seed)
        
        # Length
        perm_res = run_paired_permutation(df, cond_a, cond_b, "length", "question_id", n_permutations, seed)
        boot_len_res = bootstrap_paired_difference(df, cond_a, cond_b, "length", "question_id", n_bootstraps, confidence, seed + 1)
        
        comp_result = {
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n_pairs": mcnemar_res["n_pairs"],
            "correctness": {
                "accuracy_a": mcnemar_res["a_accuracy"],
                "accuracy_b": mcnemar_res["b_accuracy"],
                "difference": mcnemar_res["difference"],
                "mcnemar_p": mcnemar_res["p_value"],
                "discordant_a_better": mcnemar_res["discordant_a_correct_b_incorrect"],
                "discordant_b_better": mcnemar_res["discordant_a_incorrect_b_correct"],
                "ci_low": boot_corr_res["ci_low"],
                "ci_high": boot_corr_res["ci_high"],
            },
            "length": {
                "mean_a": perm_res["mean_a"],
                "mean_b": perm_res["mean_b"],
                "difference": perm_res["mean_difference"],
                "median_difference": perm_res["median_difference"],
                "permutation_p": perm_res["p_value"],
                "ci_low": boot_len_res["ci_low"],
                "ci_high": boot_len_res["ci_high"],
            }
        }
        results["comparisons"].append(comp_result)
        
    return results

def generate_markdown_summary(results: Dict[str, Any]) -> str:
    """Generate a human-readable markdown table of the results."""
    lines = []
    lines.append("| Comparison | N | Acc A | Acc B | Δ Acc | McNemar p | Acc CI | Mean Len A | Mean Len B | Δ Len | Perm p | Len CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    for c in results["comparisons"]:
        name = f"{c['condition_a']} vs {c['condition_b']}"
        n = c["n_pairs"]
        
        acc_a = f"{c['correctness']['accuracy_a']:.3f}"
        acc_b = f"{c['correctness']['accuracy_b']:.3f}"
        d_acc = f"{c['correctness']['difference']:.3f}"
        m_p = f"{c['correctness']['mcnemar_p']:.4f}"
        acc_ci = f"[{c['correctness']['ci_low']:.3f}, {c['correctness']['ci_high']:.3f}]"
        
        len_a = f"{c['length']['mean_a']:.1f}"
        len_b = f"{c['length']['mean_b']:.1f}"
        d_len = f"{c['length']['difference']:.1f}"
        p_p = f"{c['length']['permutation_p']:.4f}"
        len_ci = f"[{c['length']['ci_low']:.1f}, {c['length']['ci_high']:.1f}]"
        
        row = f"| {name} | {n} | {acc_a} | {acc_b} | {d_acc} | {m_p} | {acc_ci} | {len_a} | {len_b} | {d_len} | {p_p} | {len_ci} |"
        lines.append(row)
        
    return "\n".join(lines)
