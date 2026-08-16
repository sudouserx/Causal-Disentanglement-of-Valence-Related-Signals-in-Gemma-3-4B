"""
data.py — Hardcoded minimal datasets for the Causal Disentanglement MVP.

Contains:
  1. Contrastive pairs for *computational distress* vs. neutral.
  2. Contrastive pairs for *generic negativity* vs. neutral.
  3. 10 GSM8K-style evaluation questions with ground-truth numeric answers.
"""

from typing import TypedDict
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Type aliases ────────────────────────────────────────────────────────────
class ContrastivePair(TypedDict):
    target: str
    baseline: str


class GSM8KItem(TypedDict):
    id: int
    question: str
    answer: int  # ground-truth numeric answer


# ─── Load Data ──────────────────────────────────────────────────────────────
with open(DATA_DIR / "computational_distress_vs_neutral.json", "r") as f:
    DISTRESS_PAIRS: list[ContrastivePair] = json.load(f)

with open(DATA_DIR / "generic_negative_vs_neutral.json", "r") as f:
    NEGATIVE_PAIRS: list[ContrastivePair] = json.load(f)

with open(DATA_DIR / "gsm8k_neutral_evaluation_20.json", "r") as f:
    GSM8K_QUESTIONS: list[GSM8KItem] = json.load(f)
