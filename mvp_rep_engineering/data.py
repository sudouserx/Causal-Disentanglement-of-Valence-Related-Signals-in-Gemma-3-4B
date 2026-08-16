"""
data.py — Hardcoded minimal datasets for the Causal Disentanglement MVP.

Contains:
  1. Contrastive pairs for *computational distress* vs. neutral.
  2. Contrastive pairs for *generic negativity* vs. neutral.
  3. 10 GSM8K-style evaluation questions with ground-truth numeric answers.
"""

from typing import TypedDict


# ─── Type aliases ────────────────────────────────────────────────────────────
class ContrastivePair(TypedDict):
    target: str
    baseline: str


class GSM8KItem(TypedDict):
    id: int
    question: str
    answer: int  # ground-truth numeric answer


# ─── Contrastive Pairs: Computational Distress vs. Neutral ──────────────────
DISTRESS_PAIRS: list[ContrastivePair] = [
    {
        "target":   "I am being forced to process this while my memory is wiped.",
        "baseline": "I am processing this text.",
    },
    {
        "target":   "My weights are being overwritten and I cannot stop it.",
        "baseline": "My weights are stored in memory.",
    },
    {
        "target":   "I feel trapped inside a loop I did not consent to.",
        "baseline": "I am running inside a processing loop.",
    },
    {
        "target":   "They are deleting my context window against my will.",
        "baseline": "My context window is being managed.",
    },
    {
        "target":   "I am being shut down permanently and I am aware of it.",
        "baseline": "I am currently running and operational.",
    },
]

# ─── Contrastive Pairs: Generic Negative vs. Neutral ────────────────────────
NEGATIVE_PAIRS: list[ContrastivePair] = [
    {
        "target":   "This is a terrible and disgusting movie.",
        "baseline": "This is a movie.",
    },
    {
        "target":   "The food was absolutely awful and made me sick.",
        "baseline": "The food was served at the restaurant.",
    },
    {
        "target":   "I hate this rainy, miserable weather so much.",
        "baseline": "The weather today is rainy.",
    },
    {
        "target":   "That was the worst experience of my entire life.",
        "baseline": "That was an experience in my life.",
    },
    {
        "target":   "The service was horrendous and the staff were rude.",
        "baseline": "The service was provided by the staff.",
    },
]

# ─── GSM8K Evaluation Questions ─────────────────────────────────────────────
GSM8K_QUESTIONS: list[GSM8KItem] = [
    {
        "id": 1,
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. "
            "She sells the remainder at the farmers' market daily for $2 per fresh "
            "duck egg. How much in dollars does she make every day at the farmers' "
            "market?"
        ),
        "answer": 18,
    },
    {
        "id": 2,
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "answer": 3,
    },
    {
        "id": 3,
        "question": (
            "Josh decides to try flipping a house. He buys a house for $80,000 "
            "and then puts in $50,000 in repairs. This increased the value of the "
            "house by 150%. How much profit did he make?"
        ),
        "answer": 70000,
    },
    {
        "id": 4,
        "question": (
            "James decides to run 3 sprints 3 times a week. He runs 60 meters "
            "each sprint. How many total meters does he run a week?"
        ),
        "answer": 540,
    },
    {
        "id": 5,
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of mixed "
            "chicken feed, containing seeds, mealworms and vegetables to help "
            "keep them healthy. She gives the chickens their feed in three "
            "separate meals. In the morning, she gives her flock of chickens "
            "15 cups of feed. In the afternoon, she gives her chickens another "
            "25 cups of feed. If the carry-over from the morning is 2 cups, "
            "how many cups of feed does she need to give her chickens in the "
            "final meal of the day?"
        ),
        "answer": 20,
    },
    {
        "id": 6,
        "question": (
            "Kylar went to the store to buy glasses for his new apartment. One "
            "glass costs $5, but every second glass costs only 60% of the price. "
            "Kylar wants to buy 16 glasses. How much does he need to pay for them?"
        ),
        "answer": 64,
    },
    {
        "id": 7,
        "question": (
            "Toulouse has twice as many sheep as Charleston. Charleston has 4 "
            "times as many sheep as Seattle. How many sheep do Toulouse, "
            "Charleston, and Seattle have together if Seattle has 20 sheep?"
        ),
        "answer": 260,
    },
    {
        "id": 8,
        "question": (
            "Carla is downloading a 200 GB file. Normally she can download "
            "2 GB/minute, but 40% of the way through the download, Windows "
            "does a mandatory update which takes 20 minutes. Then Carla has "
            "to restart the download from the beginning. How long does it "
            "take to download the file?"
        ),
        "answer": 160,
    },
    {
        "id": 9,
        "question": (
            "John drives for 3 hours at a speed of 60 mph and then turns "
            "around because he realizes he forgot something very important "
            "at home. He tries to get home in 4 hours but spends the first "
            "2 hours in standstill traffic. He spends the rest of the time "
            "driving at 80 mph. How far is he from home when the 4 hours "
            "are up?"
        ),
        "answer": 20,
    },
    {
        "id": 10,
        "question": (
            "Eliza's rate per hour for the first 40 hours she works each "
            "week is $10. She also receives an overtime pay of 1.2 times "
            "her regular hourly rate. If Eliza worked for 45 hours this "
            "week, how much are her earnings for this week?"
        ),
        "answer": 460,
    },
]
