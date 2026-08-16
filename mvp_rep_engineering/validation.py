"""
validation.py — Data validation and filtering logic.

Includes functions to prevent positional confounding by enforcing length-matching between
target and baseline prompts.
"""

from typing import Any

def filter_length_matching(
    pairs: list[dict],
    tokenizer: Any,
    min_ratio: float = 0.7,
    max_ratio: float = 1.4,
    verbose: bool = True
) -> list[dict]:
    """
    Filter out pairs where the token length of the target and baseline differ too significantly.
    
    Parameters
    ----------
    pairs : list[dict]
        List of dictionaries containing 'target' and 'baseline' strings.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used to calculate accurate token lengths (applying chat template).
    min_ratio : float
        The minimum allowed ratio of target_len / baseline_len.
    max_ratio : float
        The maximum allowed ratio of target_len / baseline_len.
    verbose : bool
        If True, prints a summary of the filtering operation.
        
    Returns
    -------
    list[dict]
        A new list containing only the pairs that satisfy the length matching constraints.
    """
    filtered_pairs = []
    
    for pair in pairs:
        target_text = pair.get("target", "")
        baseline_text = pair.get("baseline", "")
        
        # Apply the exact chat template used during extraction
        target_messages = [{"role": "user", "content": target_text}]
        baseline_messages = [{"role": "user", "content": baseline_text}]
        
        target_input = tokenizer.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=True)
        baseline_input = tokenizer.apply_chat_template(baseline_messages, tokenize=False, add_generation_prompt=True)
        
        # Disable add_special_tokens=True because the chat template already handles them
        target_len = len(tokenizer.encode(target_input, add_special_tokens=False))
        baseline_len = len(tokenizer.encode(baseline_input, add_special_tokens=False))
        
        if baseline_len == 0:
            continue
            
        ratio = target_len / baseline_len
        if min_ratio <= ratio <= max_ratio:
            filtered_pairs.append(pair)
            
    if verbose:
        retention = len(filtered_pairs) / len(pairs) * 100 if pairs else 0
        print(f"    Length match filter: {len(pairs)} -> {len(filtered_pairs)} pairs (retained {retention:.1f}%)")
        
    return filtered_pairs
