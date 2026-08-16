# Causal Disentanglement of Valence-Related Signals in Gemma-3

This project provides a PyTorch-based Minimum Viable Product (MVP) designed to extract, isolate, and evaluate specific internal representations in the `google/gemma-3-12b-it` model. It focuses on isolating concepts such as "computational distress" and "failure difficulty" from a "generic negative sentiment" concept using Representation Engineering (RepE) techniques, and features rigorous statistical evaluation using paired random control vectors.

## Theoretical Background

### Representation Engineering via Contrastive Pairs
Large Language Models process tokens through sequential layers, building internal representations of concepts. We isolate the direction of a specific concept in the model's activation space by utilizing contrastive prompting.

By supplying the model with pairs of prompts—one eliciting the target concept and one acting as a neutral baseline—we extract the internal hidden states. The concept vector is approximated as the mean difference of these activations at a specific layer:

$$
v_{concept} = \frac{1}{N}\sum_{i=1}^{N} (act_{target}^{(i)} - act_{baseline}^{(i)})
$$

### Causal Disentanglement (Gram-Schmidt Orthogonalization)
A common challenge when targeting complex states like "computational distress" or "failure difficulty" is confounding. Eliciting these states often inadvertently elicits a generic "negative sentiment", resulting in a vector that encodes multiple concepts. 

To isolate the *pure* target concept, this pipeline employs Gram-Schmidt orthogonalization. We compute distinct vectors (e.g., $v_{distress}$, $v_{failure}$, and $v_{negative}$) and mathematically project out the negative component from the target vector, leaving a residual candidate vector representing only the unique signal:

$$
v_{cand} = v_{target} - \text{proj}_{v_{negative}}(v_{target})
$$

Where the projection is defined as:

$$
\text{proj}_{v_{negative}}(v_{target}) = \frac{v_{target} \cdot v_{negative}}{v_{negative} \cdot v_{negative}} v_{negative}
$$

## Pipeline Architecture

The end-to-end pipeline is orchestrated in `run_mvp.py` and structured into six phases:

1. **Model Loading:** 
   Loads the `google/gemma-3-12b-it` model utilizing 4-bit NF4 quantization to reduce memory footprint.
2. **Activation Extraction:** 
   Passes contrastive datasets through the model. PyTorch forward hooks (`steering.py`) capture the last-token hidden states at `TARGET_LAYER=15` for distress, generic-negative, and failure datasets.
3. **Vector Math & Scaling:** 
   Computes mean-difference vectors for the concepts and residualizes the distress and failure vectors against the negative vector. Vectors are normalized and scaled by a hyperparameter $\alpha$ multiplied by the mean $L^2$ norm of baseline activations ($\mu_{norm}$). Multiple random orthogonal control vectors are also generated.
4. **Causal Intervention (Steering):** 
   During the auto-regressive generation phase (`seq_len == 1`) on the GSM8K dataset, a forward pre-hook additively injects the scaled vectors into the layer's hidden states across different conditions (baseline, negative, candidate distress, candidate failure, and random controls).
5. **Evaluation & Logging:** 
   Evaluates the model's generated responses. It parses mathematical answers using regex to verify accuracy, monitors truncation rates, and counts refusals by matching against predefined regex refusal patterns.
6. **Statistical Analysis:** 
   Performs rigorous paired statistical testing across conditions. This includes McNemar's exact tests for binary correctness differences, paired permutation tests for generation length, and bootstrap confidence intervals, outputting results as JSON and Markdown summaries.

## Repository Structure

- `data/`: Contains JSON files storing the contrastive pairs and evaluation data.
  - `computational_distress_60.json`
  - `failure_difficulty_60.json`
  - `generic_negative_60.json`
  - `gsm8k_neutral_evaluation_80_mixed.json`
- `mvp_rep_engineering/`: The core Python package.
  - `config.py`: Hyperparameters and configuration constants.
  - `data.py`: Dynamic JSON loading utilities.
  - `evaluation.py`: Generation logic, GSM8K answer extraction, and refusal detection.
  - `model_utils.py`: 4-bit model and tokenizer initialization.
  - `run_mvp.py`: The main orchestration script.
  - `statistics.py`: Functions for paired statistical analysis, McNemar exact tests, and permutation tests.
  - `steering.py`: PyTorch hook factories for extraction and generation-time injection.
  - `vector_math.py`: Linear algebra functions for mean-difference extraction, Gram-Schmidt orthogonalization, and random vector generation.

## Usage

To execute the full MVP pipeline, simply run the orchestration script:

```bash
python mvp_rep_engineering/run_mvp.py
```

The script will automatically compute the steering vectors, perform the causal interventions, evaluate the GSM8K subset, execute statistical analyses, and output the summary metrics and reports to a `results/` directory.
