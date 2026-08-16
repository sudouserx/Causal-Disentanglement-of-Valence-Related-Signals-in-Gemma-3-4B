# Causal Disentanglement of Valence-Related Signals in Gemma-3

This project provides a PyTorch-based pipeline that extracts, isolates, and evaluates internal representations in the `google/gemma-3-12b-it` model. It extracts concept vectors for "computational distress" and "failure difficulty," orthogonalizes them against a "generic negative sentiment" vector, and applies these vectors during text generation. The pipeline evaluates the model's responses under these different vector interventions using paired random control vectors.

## Theoretical Background

### Representation Engineering via Contrastive Pairs
The pipeline extracts the direction of a specific concept in the model's activation space by supplying the model with pairs of prompts (a target and a baseline). It extracts the internal hidden states for these prompts and computes the concept vector as the mean difference of these activations at a specific layer:

$$
v_{concept} = \frac{1}{N}\sum_{i=1}^{N} (act_{target}^{(i)} - act_{baseline}^{(i)})
$$

### Gram-Schmidt Orthogonalization
The pipeline computes distinct vectors ($v_{distress}$, $v_{failure}$, and $v_{negative}$) and mathematically projects the negative component out of the target vector using Gram-Schmidt orthogonalization. This produces a residual candidate vector:

$$
v_{cand} = v_{target} - \text{proj}_{v_{negative}}(v_{target})
$$

Where the projection is defined as:

$$
\text{proj}_{v_{negative}}(v_{target}) = \frac{v_{target} \cdot v_{negative}}{v_{negative} \cdot v_{negative}} v_{negative}
$$

## Pipeline Architecture

The pipeline is orchestrated in `run_mvp.py` and performs the following phases:

1. **Model Loading:** 
   Loads the `google/gemma-3-12b-it` model utilizing 4-bit NF4 quantization.
2. **Activation Extraction:** 
   Splits contrastive datasets into training and validation sets. Passes the training sets through the model and uses PyTorch forward hooks (`steering.py`) to capture the last-token hidden states at specific target layers (defined in `config.py`, e.g., 12, 15, 18, 21) for the distress, generic-negative, and failure datasets.
3. **Vector Math & Scaling:** 
   Computes mean-difference vectors for the concepts and residualizes the distress and failure vectors against the negative vector. Vectors are normalized and scaled by a hyperparameter $\alpha$ multiplied by the mean $L^2$ norm of baseline activations ($\mu_{norm}$). Multiple random unit control vectors are also generated.
4. **Causal Intervention (Steering):** 
   During the auto-regressive generation phase (`seq_len == 1`), a forward pre-hook additively injects the scaled vectors into the layer's hidden states. This is done across different conditions (baseline, negative, candidate distress, candidate failure, and random controls) and evaluated on a GSM8K subset as well as the validation splits of the contrastive datasets.
5. **Evaluation & Logging:** 
   Evaluates the model's generated responses. It parses mathematical answers using regex to verify accuracy for the GSM8K dataset, monitors truncation rates, and counts refusals by matching against predefined regex refusal patterns.
6. **Statistical Analysis:** 
   If pilot mode is disabled, it performs paired statistical testing across conditions. This includes McNemar's exact tests for binary correctness differences, paired permutation tests for generation length, and bootstrap confidence intervals. The script outputs results as CSV files, JSON metrics, and Markdown summaries.

## Repository Structure

- `data/`: Contains JSON files storing the contrastive pairs and evaluation data.
  - `computational_distress_60.json`
  - `failure_difficulty_60.json`
  - `generic_negative_60.json`
  - `gsm8k_neutral_evaluation_80_mixed.json`
- `mvp_rep_engineering/`: The Python package containing the pipeline logic.
  - `config.py`: Hyperparameters and configuration constants.
  - `data.py`: Dynamic JSON loading and dataset splitting utilities.
  - `evaluation.py`: Generation logic, GSM8K answer extraction, and refusal detection.
  - `model_utils.py`: 4-bit model and tokenizer initialization.
  - `run_mvp.py`: The main orchestration script.
  - `statistics.py`: Functions for paired statistical analysis, McNemar exact tests, and permutation tests.
  - `steering.py`: PyTorch hook factories for extraction and generation-time injection.
  - `vector_math.py`: Linear algebra functions for mean-difference extraction, Gram-Schmidt orthogonalization, and random vector generation.

## Usage

To execute the pipeline, run the orchestration script:

```bash
python mvp_rep_engineering/run_mvp.py
```

The script computes the steering vectors, performs causal interventions, evaluates the datasets, executes statistical analyses (if pilot mode is disabled), and outputs the resulting metrics and reports to a `results/` directory.
