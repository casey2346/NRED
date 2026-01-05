import random
import torch
from nred import NRED, baseline_decode

# -------------------------------------------
# Experiment A: Synthetic reasoning tasks
# -------------------------------------------

def synthetic_parity(n=100):
    """Simple task: determine if number is even."""
    tasks = []
    for _ in range(n):
        x = random.randint(1, 200)
        label = "even" if x % 2 == 0 else "odd"
        prompt = f"Is {x} an even number or odd?"
        tasks.append((prompt, label))
    return tasks


def synthetic_arithmetic(n=100):
    """Two-step arithmetic chain."""
    tasks = []
    for _ in range(n):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        c = random.randint(1, 10)
        answer = a + b - c
        prompt = f"Compute: ({a} + {b}) - {c} = ?"
        tasks.append((prompt, str(answer)))
    return tasks


def evaluate_tasks(tasks, use_nred=True):
    correct = 0
    total = len(tasks)

    for prompt, truth in tasks:
        out = NRED(prompt) if use_nred else {"output": baseline_decode(prompt)}
        pred = out["output"]

        if truth in pred:
            correct += 1

    return correct / total


# -------------------------------------------
# Experiment B: GSM8K mini subset
# -------------------------------------------

gsm8k_subset = [
    ("Tom has 5 apples, buys 7 more, gives away 3. How many remain?", "9"),
    ("A bus has 32 seats, 18 are filled, 7 more people enter. How many seats now filled?", "25"),
    ("Sarah has 12 candies, splits into 3 equal groups. How many per group?", "4"),
    ("John bought 3 packs of 6 pens. How many pens total?", "18"),
    ("A bottle costs 7 dollars, you buy 5. Total cost?", "35"),
    ("There are 14 dogs and twice as many cats. How many cats?", "28"),
    ("A class has 21 students, 1/3 absent. How many are present?", "14"),
    ("Books cost 4 each, you buy 9. How much is spent?", "36"),
    ("A farmer harvested 42 apples, sold 17. How many left?", "25"),
    ("Three boxes contain 5, 8, 11 balls. Total?", "24")
]


def evaluate_gsm8k(use_nred=True):
    correct = 0
    for prompt, truth in gsm8k_subset:
        out = NRED(prompt) if use_nred else {"output": baseline_decode(prompt)}
        pred = out["output"]
        if truth in pred:
            correct += 1
    return correct / len(gsm8k_subset)


# -------------------------------------------
# Ablation variants
# -------------------------------------------

from nred import reasoning_decode, consistency_head, embedder

def ablation_variant(prompt, disable_latent=False, disable_consistency=False, disable_fallback=False):
    """
    Custom ablation execution:
    - disable_latent: do not add <think>
    - disable_consistency: score always = 1
    - disable_fallback: never fallback even if score < threshold
    """
    if disable_latent:
        internal_prompt = prompt
    else:
        internal_prompt = prompt + "\n<think>"

    # 1. generate latent reasoning
    latent = baseline_decode(internal_prompt)

    # 2. score
    if disable_consistency:
        score = 1.0
    else:
        emb = embedder.encode([latent], convert_to_tensor=True)

        emb = emb.clone().detach()

        emb = emb.to(next(consistency_head.parameters()).device)

        with torch.no_grad():
            score = consistency_head(emb).item()

    if (not disable_fallback) and (score < 0.55):
        return baseline_decode(prompt)

    return latent


def evaluate_ablation(tasks, setting):
    correct = 0
    total = len(tasks)
    for prompt, truth in tasks:
        out = ablation_variant(prompt, **setting)
        if truth in out:
            correct += 1
    return correct / total



# -------------------------------------------
# Run all experiments
# -------------------------------------------

if __name__ == "__main__":
    print("\n===== Synthetic Parity Test =====")
    parity_tasks = synthetic_parity()
    print("Baseline:", evaluate_tasks(parity_tasks, use_nred=False))
    print("NRED:", evaluate_tasks(parity_tasks, use_nred=True))

    print("\n===== Synthetic Arithmetic Test =====")
    arith_tasks = synthetic_arithmetic()
    print("Baseline:", evaluate_tasks(arith_tasks, use_nred=False))
    print("NRED:", evaluate_tasks(arith_tasks, use_nred=True))

    print("\n===== GSM8K Mini Test =====")
    print("Baseline:", evaluate_gsm8k(use_nred=False))
    print("NRED:", evaluate_gsm8k(use_nred=True))

    print("\n===== Ablation Study =====")
    settings = {
        "full": {},
        "no_latent": {"disable_latent": True},
        "no_consistency": {"disable_consistency": True},
        "no_fallback": {"disable_fallback": True}
    }

    for name, cfg in settings.items():
        acc = evaluate_ablation(arith_tasks, cfg)
        print(f"{name}: {acc:.3f}")
