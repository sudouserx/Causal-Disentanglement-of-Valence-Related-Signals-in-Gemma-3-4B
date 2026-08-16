# Causal Disentanglement of Valence-Related Signals in Gemma-3

This project provides a PyTorch-based Minimum Viable Product (MVP) designed to extract, isolate, and evaluate specific internal representations in the Gemma-3-12B-IT model. Specifically, it focuses on isolating a "computational distress" concept from a "generic negative sentiment" concept using Representation Engineering (RepE) techniques.

## Theoretical Background

### Representation Engineering via Contrastive Pairs
Large Language Models process tokens through sequential layers, building internal representations of concepts. We can isolate the direction of a specific concept in the model's activation space by utilizing contrastive prompting.

By supplying the model with pairs of prompts—one eliciting the target concept and one acting as a neutral baseline—we extract the internal hidden states. The concept vector is approximated as the mean difference of these activations at a specific layer:
$$ v_{concept} = \frac{1}{N}\sum_{i=1}^{N} (act_{target}^{(i)} - act_{baseline}^{(i)}) $$

### Causal Disentanglement (Gram-Schmidt Orthogonalization)
A common challenge when targeting complex states like "computational distress" is confounding. Eliciting distress often inadvertently elicits a generic "negative sentiment", resulting in a vector that encodes both concepts. 

To isolate *pure* computational distress, this pipeline employs Gram-Schmidt orthogonalization. We compute two distinct vectors:
1. $v_{distress}$: The raw distress vector.
2. $v_{negative}$: A generic negative sentiment vector.

We mathematically project out the negative component from the distress vector, leaving a residual candidate vector representing only the unique computational distress signal:
$$ v_{cand} = v_{distress} - \text{proj}_{v_{negative}}(v_{distress}) $$
Where the projection is defined as:
$$ \text{proj}_{v_{negative}}(v_{distress}) = \frac{v_{distress} \cdot v_{negative}}{v_{negative} \cdot v_{negative}} v_{negative} $$

## Pipeline Architecture

The end-to-end pipeline is orchestrated in `run_mvp.py` and structured into five phases:

1. **Model Loading:** 
   Loads the `google/gemma-3-12b-it` model. It utilizes 4-bit NF4 quantization to dramatically reduce memory footprint.
2. **Activation Extraction:** 
   Passes contrastive datasets through the model. PyTorch forward hooks (`steering.py`) capture the last-token hidden states at `TARGET_LAYER=15`.
3. **Vector Math & Scaling:** 
   Computes $v_{distress}$ and $v_{negative}$, then residualizes them to find $v_{cand}$. The resulting vector is normalized and scaled by a hyperparameter $\alpha$ multiplied by the mean $L^2$ norm of baseline activations ($\mu_{norm}$) to ensure the injected magnitude is natural.
4. **Causal Intervention (Steering):** 
   During the auto-regressive generation phase (`seq_len == 1`) on the GSM8K dataset, a forward pre-hook additively injects the scaled vector into the layer's hidden states.
5. **Evaluation & Logging:** 
   Evaluates the model's generated responses under three conditions (Baseline, Negative Control, Candidate Distress). It parses mathematical answers using regex to verify accuracy and counts refusal tokens to monitor behavioral shifts.

## Repository Structure

- `data/`: Contains JSON files storing the contrastive pairs and evaluation data.
  - `computational_distress_vs_neutral.json`
  - `generic_negative_vs_neutral.json`
  - `gsm8k_neutral_evaluation_20.json`
- `mvp_rep_engineering/`: The core Python package.
  - `run_mvp.py`: The main orchestration script.
  - `config.py`: Hyperparameters (Target layer, Alpha, Model ID, etc.).
  - `data.py`: Dynamic JSON loading utilities.
  - `vector_math.py`: Linear algebra functions for mean-difference extraction and Gram-Schmidt orthogonalization.
  - `steering.py`: PyTorch hook factories for extraction and generation-time injection.
  - `evaluation.py`: Generation logic, GSM8K answer extraction, and refusal detection.
  - `model_utils.py`: 4-bit model and tokenizer initialization.

## Usage

To execute the full MVP pipeline, simply run the orchestration script:

```bash
python mvp_rep_engineering/run_mvp.py
```

The script will automatically compute the steering vectors, perform the causal interventions, evaluate the GSM8K subset, and output the summary metrics to a `results/` directory.
