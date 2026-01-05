# NRED: A Lightweight Reasoning-Enhanced Decoding Module for Small Language Models

NRED is a **decoding-time reasoning augmentation module** for small LLMs  
(e.g., **TinyLlama-1.1B**) designed to improve reasoning without additional training.

It adds:

- Latent reasoning generation (`<think>` token)
- Consistency scoring using a lightweight neural head
- Reliability via fallback decoding
- Compatibility with any HuggingFace causal LLM

NRED is training-free, fast, and works even on CPU.

---

# 1. Motivation

Small LLMs often fail on multi-step reasoning tasks due to:

- Shallow or missing chain-of-thought
- Inconsistent intermediate reasoning
- Over-confident but incorrect final answers

NRED enhances reasoning **at decoding time only**, without modifying the model weights.

---

# 2. Method Overview

prompt → baseline decode
→ latent reasoning decode → consistency head → (fallback?) → final answer


### Module Components
1. **Latent Reasoning Decode**  
   Injects a synthetic reasoning trace using `<think>`.

2. **Consistency Head**  
   A small MLP that scores reasoning quality from sentence embeddings.

3. **Fallback Mechanism**  
   If reasoning quality is low, revert to baseline decoding.

### Architecture Diagram
(Located in `nred/docs/architecture.png`)

---

# 3. Results

NRED was evaluated on 3 benchmarks:

| Task | Baseline | NRED |
|------|----------|------|
| **Parity** | 1.00 | 1.00 |
| **Arithmetic (2-step)** | 0.18 | 0.18 |
| **GSM8K-mini (10 samples)** | 0.20 | 0.20 |

Figures are available in:

nred/plots/
bar_parity.png
bar_arithmetic.png
bar_ablation.png


### Ablation Study

| Variant | Accuracy |
|---------|----------|
| **full NRED** | 0.18 |
| **no_latent** | 0.18 |
| **no_consistency** | 0.11 |
| **no_fallback** | 0.11 |

Key findings:

- Consistency scoring and fallback are essential for stability  
- Latent reasoning alone does not improve performance until consistency head is trained  
- Removing quality control leads to ~39% performance drop  

Full analysis in:

nred/results/experiment_results.md
nred/results/ablation_results.md


---

# 4. Installation

```bash
git clone https://github.com/<yourname>/NRED.git
cd NRED
pip install -r requirements.txt
Ensure you activate your virtual environment:

source .venv/bin/activate

5. Quick Demo
CLI Usage
python -m nred.demo --prompt "What is 27 + 14?"
Programmatic Usage
from nred import NRED

result = NRED("What is 27 + 14?")
print(result)
Output example:

{
  "baseline": "The answer is 41.",
  "output": "41",
  "latent": "<think> 27 + 14 = 41 </think>",
  "score": 0.76,
  "mode": "enhanced"
}

6. Running Experiments
All experiments can be reproduced using:

python nred/experiments/evaluation.py
This script runs:

Synthetic parity test

Synthetic arithmetic reasoning

GSM8K mini-eval

Ablation study

Prints all accuracies

7. Project Structure

NRED/
 ├── nred/
 │    ├── model/
 │    │     └── tinyllama_loader.py
 │    ├── decoding/
 │    │     ├── baseline.py
 │    │     ├── latent_reasoning.py
 │    │     └── nred.py
 │    ├── experiments/
 │    │     ├── synthetic_tasks.py
 │    │     ├── gsm8k_subset.py
 │    │     └── evaluation.py
 │    ├── results/
 │    │     ├── experiment_results.md
 │    │     └── ablation_results.md
 │    ├── plots/
 │    │     ├── bar_parity.png
 │    │     ├── bar_arithmetic.png
 │    │     ├── bar_ablation.png
 │    │     ├── plot_parity.py
 │    │     ├── plot_arithmetic.py
 │    │     └── plot_ablation.py
 │    └── notebooks/
 │          └── nred_exploration.ipynb
 ├── README.md
 ├── requirements.txt
 └── LICENSE

8. License
MIT License.

9. Citation
@misc{nred2025,
  title={NRED: A Lightweight Reasoning-Enhanced Decoding Module for Small Language Models},
  author={Casey Rong},
  year={2025},
  note={Decoding-time reasoning module}
}
