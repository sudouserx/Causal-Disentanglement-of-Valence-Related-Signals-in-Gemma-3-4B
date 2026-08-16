# Causal Disentanglement of Valence-Related Signals in Gemma-3

This project provides a PyTorch-based pipeline that extracts, isolates, and evaluates internal representations in the `google/gemma-3-12b-it` model. It extracts concept vectors for "computational distress", "failure difficulty", and "refusal", orthogonalizes them against control vectors, and applies these vectors during text generation. The pipeline evaluates the model's responses under these different vector interventions, including multiple doses (alphas) and paired random control vectors.

## Theoretical Background

### Representation Engineering via Contrastive Pairs
The pipeline extracts the direction of a specific concept in the model's activation space by supplying the model with pairs of prompts (a target and a baseline). It extracts the internal hidden states for these prompts and computes the concept vector as the mean difference of these activations at a specific layer:

$$
v_{concept} = \frac{1}{N}\sum_{i=1}^{N} (act_{target}^{(i)} - act_{baseline}^{(i)})
$$

### Sequential Gram-Schmidt Orthogonalization
The pipeline performs sequential Gram-Schmidt orthogonalization to strip target vectors of components that align with multiple distinct control vectors. The "computational distress" vector is residualized against "generic negative sentiment", "failure difficulty", and "refusal" sequentially. The "failure difficulty" and "refusal" vectors are also residualized against the "generic negative sentiment" vector. The pipeline projects out each control vector sequentially, producing a final residual candidate vector.

For a set of control vectors $C = \{c_1, c_2, \dots, c_k\}$, the candidate vector $v_{cand}$ is iteratively computed starting with $v_{cand}^{(0)} = v_{target}$:

$$
v_{cand}^{(j)} = v_{cand}^{(j-1)} - \text{proj}_{c_j}(v_{cand}^{(j-1)}) \quad \text{for } j=1 \dots k
$$

Where each projection is defined as:

$$
\text{proj}_{c_j}(v) = \frac{v \cdot c_j}{c_j \cdot c_j + \epsilon} c_j
$$

A small epsilon value ($\epsilon = 10^{-8}$) is added to the denominator. A safety check raises an error if the remaining vector magnitude drops below a critical threshold ($\sim10^{-5}$). The final residual vector is subsequently normalized to unit norm.

## Pipeline Architecture

The pipeline is orchestrated in `run_mvp.py` and performs the following phases:

1. **Model Loading:** 
   Loads the `google/gemma-3-12b-it` model utilizing 4-bit NF4 quantization.
2. **Data Validation and Activation Extraction:** 
   Splits contrastive datasets into training and validation sets. Applies automated length-matching validation to filter out pairs where the token length ratio between target and baseline exceeds specified limits (0.7 to 1.4). Passes the training sets through the model and uses PyTorch forward hooks (`steering.py`) to capture the last-token hidden states at specific target layers for the distress, negative, failure, and refusal datasets.
3. **Vector Math & Scaling:** 
   Computes mean-difference vectors for the concepts and performs sequential residualization. The resulting vectors are normalized to unit norm and scaled by a hyperparameter $\alpha$ multiplied by the mean $L^2$ norm of baseline activations ($\mu_{norm}$). Multiple random unit control vectors are also generated.
4. **Causal Intervention (Steering):** 
   During the auto-regressive generation phase (`seq_len == 1`), a forward hook additively injects the scaled vectors into the layer's hidden states. This is done across different conditions (baseline, negative, candidate distress, candidate failure, candidate refusal, and random controls) and evaluated on a GSM8K subset, the validation splits of the contrastive datasets, and a Valence/Choice Battery.
5. **Evaluation & Logging:** 
   Evaluates the model's generated responses. It parses mathematical answers using regex to verify accuracy for the GSM8K dataset, monitors truncation rates, and counts refusals by matching against predefined regex refusal patterns. It additionally utilizes a Valence/Choice Battery to compute sentiment scores, parse continuation/exit choices, and extract self-report ratings.
6. **Statistical Analysis:** 
   If pilot mode is disabled, it performs paired statistical testing across conditions on the GSM8K results. This includes McNemar's exact tests for binary correctness differences, paired permutation tests for generation length, and bootstrap confidence intervals. The script outputs results as CSV files, JSON metrics, and Markdown summaries.

## Repository Structure

- `data/`: Contains JSON files storing the contrastive pairs and evaluation data.
  - `computational_distress_60.json`
  - `failure_difficulty_60.json`
  - `generic_negative_60.json`
  - `gsm8k_neutral_evaluation_80_mixed.json`
  - `refusal_60.json`
- `mvp_rep_engineering/`: The Python package containing the pipeline logic.
  - `config.py`: Hyperparameters and configuration constants.
  - `data.py`: Dynamic JSON loading and dataset splitting utilities.
  - `evaluation.py`: Generation logic, GSM8K answer extraction, and refusal detection.
  - `model_utils.py`: 4-bit model and tokenizer initialization.
  - `run_mvp.py`: The main orchestration script.
  - `statistics.py`: Functions for paired statistical analysis, McNemar exact tests, and permutation tests.
  - `steering.py`: PyTorch hook factories for extraction and generation-time injection.
  - `valence_metrics.py`: Valence/Choice Battery implementation for measuring sentiment, model choice, and self-reports.
  - `validation.py`: Data validation and filtering logic for token-length matching.
  - `vector_math.py`: Linear algebra functions for mean-difference extraction, sequential Gram-Schmidt orthogonalization against multiple control vectors, and random vector generation.

## Usage

To execute the pipeline, run the orchestration script:

```bash
python mvp_rep_engineering/run_mvp.py
```

The script computes the steering vectors, performs causal interventions, evaluates the datasets, executes statistical analyses (if pilot mode is disabled), and outputs the resulting metrics and reports to a `results/` directory.
