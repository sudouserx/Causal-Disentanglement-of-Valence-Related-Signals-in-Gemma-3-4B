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
