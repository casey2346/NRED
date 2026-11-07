
# NRED: Reasoning-Enhanced Decoding for Small Language Models

NRED is an experimental inference-time reasoning framework for small language models, including TinyLlama-1.1B.

It investigates whether consistency scoring and selective fallback can help manage unreliable reasoning traces without modifying the underlying language model weights.

The framework combines three components:

- Latent reasoning generation
- Lightweight neural consistency scoring
- Selective fallback to baseline decoding

The current experiments do not demonstrate an overall accuracy improvement over baseline decoding. Instead, NRED provides a modular framework for studying the behaviour and limitations of reasoning-quality control.

## 1. Research Motivation

Small language models often struggle with multi-step reasoning. Generating an intermediate reasoning trace does not necessarily produce a more accurate answer, and an incorrect trace may lead to a confidently wrong prediction.

NRED investigates three questions:

1. Can consistency scoring help identify unreliable reasoning traces?
2. When should enhanced decoding be accepted rather than reverting to baseline decoding?
3. How do latent reasoning, consistency scoring, and fallback individually affect performance?

The goal is to evaluate these mechanisms rather than assume that additional reasoning always improves accuracy.

## 2. Method

The conceptual decoding pipeline is:

```text
Input Prompt
     |
     v
Baseline Decoding
     |
     v
Latent Reasoning Generation
     |
     v
Consistency Scoring
     |
     v
Threshold Decision
     |
     +--- Low score ---> Baseline Output
     |
     +--- Accepted ----> Enhanced Output
     |
     v
Final Answer
```

### Latent Reasoning

The framework generates an intermediate reasoning trace using a structured `<think>` format.

This provides an additional reasoning candidate but does not guarantee correctness.

### Consistency Scoring

A lightweight neural head uses sentence embeddings to produce a score associated with the generated reasoning.

The score is used by the decoding pipeline to determine whether the enhanced output should be accepted.

The scoring head is separate from the underlying language model. Its initialisation, training status, and calibration must be established from the implementation before interpreting its score as a reliable measure of reasoning quality.

### Selective Fallback

The fallback mechanism returns to baseline decoding when the consistency score does not meet the acceptance criterion.

Fallback is intended to reduce reliance on uncertain reasoning traces. Its effectiveness remains an empirical question rather than a guaranteed improvement.

## 3. Experimental Evaluation

The reported evaluation includes:

- Synthetic parity reasoning
- Two-step arithmetic reasoning
- GSM8K-mini

### Baseline Comparison

| Benchmark | Baseline | Full NRED |
|---|---:|---:|
| Parity | 1.00 | 1.00 |
| Arithmetic (2-step) | 0.18 | 0.18 |
| GSM8K-mini (10 examples) | 0.20 | 0.20 |

Full NRED matched baseline accuracy on all three reported benchmarks.

These results do not show an overall improvement over baseline decoding.

### Ablation Study

| Configuration | Reported Accuracy |
|---|---:|
| Full NRED | 0.18 |
| Without latent reasoning | 0.18 |
| Without consistency scoring | 0.11 |
| Without fallback | 0.11 |

In the reported ablation setting, removing consistency scoring or fallback reduced accuracy from 0.18 to 0.11.

The relative decrease is approximately:

```text
(0.18 - 0.11) / 0.18 × 100 = 38.9%
```

This is a relative decrease compared with Full NRED within the reported ablation setting. It is not a 38.9% improvement over baseline decoding.

Removing latent reasoning did not change the reported accuracy.

The results are preliminary and require further validation before drawing conclusions about the general effectiveness of individual components.

## 4. Interpretation and Limitations

The experiments provide an initial analysis of component behaviour rather than evidence of a general improvement in reasoning accuracy.

Several limitations remain:

- Full NRED did not outperform the baseline on the reported benchmarks.
- GSM8K-mini contains only 10 evaluated examples.
- The reported results do not establish statistical significance.
- The consistency head's training status and calibration require verification against the source code.
- The experiments have not yet been independently reproduced under a fully documented configuration.
- More extensive evaluation is required to determine whether the ablation differences generalise.

An important distinction is that lower accuracy in a component-removal experiment does not, by itself, establish that the full framework outperforms the original language model.

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/casey2346/NRED.git
cd NRED
```

Create a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Installation of dependencies does not by itself verify that the project runs successfully or reproduces the reported results.

## 6. Reproducibility

A complete reproduction requires the following information to be established from the implementation:

| Component | Required configuration |
|---|---|
| Base model | Model identifier and revision |
| Python environment | Python and dependency versions |
| Decoding | Generation parameters |
| Randomness | Random seed or seeds |
| Datasets | Exact examples and selection procedure |
| Scoring head | Architecture, initialisation, training status, and weights |
| Fallback | Threshold and decision rule |
| Evaluation | Answer extraction and accuracy calculation |
| Hardware | CPU/GPU and relevant execution settings |

The evaluation entry point must also be verified against the current repository structure.

The reported results should be treated as preliminary until the experiments can be reproduced using a fixed, documented configuration.

## 7. Project Organisation

The repository includes components for:

- Loading the underlying language model
- Baseline decoding
- Latent reasoning generation
- Consistency scoring and fallback
- Synthetic reasoning tasks
- GSM8K subset evaluation
- Ablation experiments
- Result analysis and visualisation

The exact file layout and executable entry points should be taken from the current repository rather than inferred from this document.

## 8. Research Findings

The current project establishes an experimental framework for investigating how reasoning generation, consistency scoring, and fallback decisions interact.

The reported results show:

1. Full NRED matched baseline accuracy on the evaluated tasks.
2. Removing consistency scoring or fallback was associated with lower reported accuracy in the ablation setting.
3. Removing latent reasoning did not change reported ablation accuracy.
4. The available evidence does not yet establish a general improvement in reasoning performance.

These findings motivate more controlled investigation of reasoning-quality estimation and selective decoding.

## 9. Future Work

Further research will focus on:

- Evaluating larger and more diverse reasoning datasets
- Repeating experiments with controlled random seeds
- Validating consistency scores against actual answer correctness
- Establishing the scoring head's training and calibration procedure
- Measuring fallback frequency and decision quality
- Comparing inference cost and accuracy
- Testing whether enhanced decoding provides benefits under identifiable conditions

A central open question is whether lightweight reasoning-quality control can produce measurable benefits over baseline decoding without additional modifications to the underlying language model.

## 10. Current Status

NRED is an experimental research and engineering project.

The decoding approach, reported benchmark comparisons, and preliminary ablation results are documented.

The current evidence does not establish an accuracy improvement over baseline decoding. Further validation is required to assess reproducibility, scoring reliability, and generalisation.

The project is intended as a foundation for investigating the conditions under which consistency-aware decoding and fallback mechanisms may be useful.

## License

MIT License.
