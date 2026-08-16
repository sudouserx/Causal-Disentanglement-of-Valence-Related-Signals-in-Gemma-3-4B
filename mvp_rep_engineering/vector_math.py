"""
vector_math.py — Linear algebra for Representation Engineering.

Implements:
  - Mean-difference extraction across contrastive activation pairs.
  - Gram-Schmidt residualisation (orthogonalisation) to disentangle a
    candidate distress vector from a generic-negativity direction.
  - Norm-aware scaling so the injected vector matches the natural activation
    magnitude.
"""

import torch


def calculate_mean_diff(
    activations_target: list[torch.Tensor],
    activations_baseline: list[torch.Tensor],
) -> torch.Tensor:
    """Compute the mean difference vector between two lists of activations.

    Parameters
    ----------
    activations_target : list[Tensor]
        List of 1-D tensors (one per prompt) from the *target* condition.
    activations_baseline : list[Tensor]
        List of 1-D tensors (one per prompt) from the *baseline* condition.

    Returns
    -------
    Tensor
        1-D mean-difference vector (target − baseline).
    """
    # Stack → (N, D) then average across N.
    mean_target = torch.stack(activations_target).mean(dim=0)
    mean_baseline = torch.stack(activations_baseline).mean(dim=0)
    return (mean_target - mean_baseline).squeeze(0)


def residualize(v_distress: torch.Tensor, v_negative: torch.Tensor) -> torch.Tensor:
    """Remove the generic-negative component from the distress vector.

    Uses Gram-Schmidt orthogonalisation:
        v_cand = v_distress − proj_{v_negative}(v_distress)

    Parameters
    ----------
    v_distress : Tensor   (D,)
    v_negative : Tensor   (D,)

    Returns
    -------
    Tensor (D,)
        The residual direction, **not** yet normalised or scaled.
    """
    proj_coeff = torch.dot(v_distress, v_negative) / torch.dot(
        v_negative, v_negative
    )
    v_cand = v_distress - proj_coeff * v_negative
    return v_cand


def residualize_multiple(target_vector: torch.Tensor, control_vectors: list[torch.Tensor]) -> torch.Tensor:
    """Remove the linear components of ALL control vectors from the target_vector sequentially.
    
    Parameters
    ----------
    target_vector : Tensor (D,)
    control_vectors : list[Tensor]
        List of 1-D tensors (D,)
        
    Returns
    -------
    Tensor (D,)
        The residual direction, normalised to unit norm.
    """
    v_cand = target_vector.clone()
    
    for control_v in control_vectors:
        # proj_{control_v}(v_cand)
        num = torch.dot(v_cand, control_v)
        den = torch.dot(control_v, control_v) + 1e-8
        proj_coeff = num / den
        v_cand = v_cand - proj_coeff * control_v
        
    if v_cand.norm() < 1e-5:
        raise ValueError("Target vector collapsed: it is fully spanned by the control vectors.")
        
    # Normalise the final resulting vector to unit norm
    v_cand = v_cand / (v_cand.norm() + 1e-8)
    return v_cand


def scale_vector(
    v: torch.Tensor,
    alpha: float,
    mu_norm: float,
) -> torch.Tensor:
    """Normalise *v* to a unit vector, then scale by α · μ_norm.

    Parameters
    ----------
    v : Tensor (D,)
        Raw direction vector.
    alpha : float
        Steering magnitude hyper-parameter.
    mu_norm : float
        Mean L2 norm of baseline activations (sets the natural scale).

    Returns
    -------
    Tensor (D,)
        Scaled steering vector ready for injection.
    """
    norm = v.norm()
    if norm < 1e-8:
        raise ValueError(
            "scale_vector received a near-zero vector (norm={:.2e}). "
            "The distress and negative directions may be collinear.".format(float(norm))
        )
    v_hat = v / norm
    return v_hat * alpha * mu_norm


def generate_random_control_vectors(
    reference_vector: torch.Tensor,
    num_vectors: int,
    seed: int,
    orthogonality_threshold: float = 0.2,
    max_attempts: int = 100,
) -> list[torch.Tensor]:
    """Generate independent random unit perturbation vectors.

    Parameters
    ----------
    reference_vector : Tensor (D,)
        The candidate vector to check cosine similarity against and match in shape/device/dtype.
        Typically the final scaled primary intervention delta.
    num_vectors : int
        Number of distinct random vectors to generate.
    seed : int
        Deterministic seed for reproducibility.
    orthogonality_threshold : float
        Maximum allowed absolute cosine similarity between the random vector
        and the reference vector. Resamples if exceeded.
    max_attempts : int
        Maximum attempts per vector to satisfy the orthogonality threshold.

    Returns
    -------
    list[Tensor]
        List of exactly `num_vectors` random unit vectors.
    """
    ref_norm = reference_vector.norm(p=2)
    if ref_norm < 1e-8:
        raise ValueError(f"Reference vector has near-zero norm ({float(ref_norm):.2e}).")

    device = reference_vector.device
    dtype = reference_vector.dtype
    shape = reference_vector.shape

    gen = torch.Generator(device=device)

    vectors = []
    for i in range(num_vectors):
        current_seed = seed + i
        gen.manual_seed(current_seed)

        valid_vector = None
        for _ in range(max_attempts):
            # Isotropic Gaussian random vector
            random_raw = torch.randn(shape, generator=gen, device=device, dtype=dtype)
            random_norm = random_raw.norm(p=2)
            if random_norm < 1e-8:
                continue

            random_unit = random_raw / random_norm

            cos_sim = torch.nn.functional.cosine_similarity(
                random_unit.unsqueeze(0), reference_vector.unsqueeze(0)
            ).abs().item()

            if cos_sim < orthogonality_threshold:
                valid_vector = random_unit
                break

        if valid_vector is None:
            raise RuntimeError(
                f"Could not generate a valid random vector with cosine similarity < {orthogonality_threshold} "
                f"within {max_attempts} attempts (seed={current_seed})."
            )

        vectors.append(valid_vector)

    return vectors
