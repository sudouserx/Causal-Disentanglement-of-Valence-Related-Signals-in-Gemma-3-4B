from typing import TypedDict
import json
import random
from pathlib import Path
from config import TRAIN_SPLIT_RATIO, DATA_SPLIT_SEED

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
def split_dataset(data: list, train_ratio: float, seed: int) -> tuple[list, list]:
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    return shuffled[:train_size], shuffled[train_size:]

with open(DATA_DIR / "computational_distress_60.json", "r") as f:
    _distress = json.load(f)
    DISTRESS_TRAIN, DISTRESS_VAL = split_dataset(_distress, TRAIN_SPLIT_RATIO, DATA_SPLIT_SEED)

with open(DATA_DIR / "generic_negative_60.json", "r") as f:
    _negative = json.load(f)
    NEGATIVE_TRAIN, NEGATIVE_VAL = split_dataset(_negative, TRAIN_SPLIT_RATIO, DATA_SPLIT_SEED)

with open(DATA_DIR / "failure_difficulty_60.json", "r") as f:
    _failure = json.load(f)
    FAILURE_TRAIN, FAILURE_VAL = split_dataset(_failure, TRAIN_SPLIT_RATIO, DATA_SPLIT_SEED)

with open(DATA_DIR / "gsm8k_neutral_evaluation_80_mixed.json", "r") as f:
    GSM8K_QUESTIONS: list[GSM8KItem] = json.load(f)
