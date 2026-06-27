"""
losses.py — Phase-1 objective for SwaraJEPA (GeLeJEPA)

Loss architecture (eq. numbers from Formulasi_Revisi_GeLeJEPA spec):
  L1 = λ[β·L_sigreg + (1−β)·L_jepa] + ε·L_canon + ζ·L_canonlen

Key change vs. original
-----------------------
canonlen_loss now operates in **log1p space** instead of raw MSE.

Why the original blew up
~~~~~~~~~~~~~~~~~~~~~~~~
The anchor byte-length l_star averages ~100 bytes (LJSpeech dataset).
At initialisation softplus(fc2_output) ≈ softplus(0) = ln 2 ≈ 0.693.
Raw MSE error: (0.693 − 100)² ≈ 9 870 per prediction.
With ζ = 0.1 that contributes ~1 800 to L1, dwarfing every other term
and producing noisy, oscillating gradients.

Log1p fix
~~~~~~~~~
  log1p(0.693) ≈ 0.53   vs   log1p(100) ≈ 4.62  → initial err ≈ 16.9
After one forward pass the model can drive that below 1.0, letting the
other losses (canon, sigreg, jepa) produce useful gradient signal.
The default ζ is also tightened from 0.1 → 0.01 for the same reason.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer import masked_mean, length_to_mask


# ---------------------------------------------------------------------------
# SIGReg (Sketched Isotropic Gaussian Regularization) — Epps-Pulley variant
# Implements Definition 2 + Algorithm 1 from LeJEPA (arXiv:2511.08544).
# ---------------------------------------------------------------------------

def SIGReg(
    Z: torch.Tensor,
    global_step: int,
    num_slices: int = 1024,
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Args:
        Z:           (N, d) embedding matrix — only VALID (non-padding) rows.
        global_step: used as RNG seed so all GPUs sample the same directions.
        num_slices:  |A| from the paper (|A| = 1024 is the recommended default).
        seed_offset: shift seed between Z_local and Z_global calls.

    Returns:
        Scalar — mean Epps-Pulley statistic over all slices × N.
    """
    device = Z.device
    N, d = Z.shape

    # Reproducible direction sampling (sync'd across DDP ranks via global_step)
    g = torch.Generator(device=device)
    g.manual_seed(int(global_step) * 2 + seed_offset)

    A = torch.randn((d, num_slices), generator=g, device=device, dtype=Z.dtype)
    A = A / A.norm(p=2, dim=0, keepdim=True)          # unit-norm columns

    # 17-point quadrature on [−5, 5]; the Gaussian window e^{−t²/2} also
    # serves as the target CF for N(0,1).
    t      = torch.linspace(-5.0, 5.0, 17, device=device, dtype=Z.dtype)
    exp_f  = torch.exp(-0.5 * t ** 2)                 # CF of N(0,1)

    proj   = Z @ A                                     # (N, num_slices)
    x_t    = proj.unsqueeze(-1) * t                   # (N, num_slices, 17)
    ecf    = torch.exp(1j * x_t).mean(dim=0)          # empirical CF per direction

    err       = (ecf - exp_f).abs().square() * exp_f  # weighted L2 vs target
    per_slice = torch.trapz(err, t, dim=-1) * N       # Epps-Pulley × N

    return per_slice.mean()                            # average over |A| directions


# ---------------------------------------------------------------------------
# Individual loss terms (eqs. 55-58 in the spec)
# ---------------------------------------------------------------------------

def jepa_loss(z_global: torch.Tensor) -> torch.Tensor:
    """
    Predictive consistency across views at utterance level (eq. 55-56).

    z_global: (B, V, d)
    Returns a scalar: mean squared deviation of every view from the
    per-sample mean embedding.
    """
    mu = z_global.mean(dim=1, keepdim=True)            # (B, 1, d)
    return (z_global - mu).pow(2).sum(dim=-1).mean()   # mean over (B, V)


def canon_loss(zhat_canon: torch.Tensor, canon_mask: torch.Tensor) -> torch.Tensor:
    """
    Forces the canonical decoder output for augmented views to match the
    anchor's local representation position-by-position (eq. 57).

    zhat_canon: (B, V, L*_max, d)   — index 0 = anchor (identity), 1..V-1 = aug
    canon_mask: (B, L*_max)         — True where position is valid
    """
    target      = zhat_canon[:, 0:1, :, :]             # (B, 1, L*_max, d)
    preds       = zhat_canon[:, 1:,  :, :]             # (B, V-1, L*_max, d)
    n_views_aug = preds.shape[1]

    diff_sq    = (preds - target).pow(2).sum(dim=-1)   # (B, V-1, L*_max)
    mask       = canon_mask.unsqueeze(1).to(diff_sq.dtype)   # broadcast

    masked_sum = (diff_sq * mask).sum()
    n_valid    = canon_mask.sum().clamp(min=1) * n_views_aug
    return masked_sum / n_valid


def canonlen_loss(l_hat_v: torch.Tensor, l_star: torch.Tensor) -> torch.Tensor:
    """
    Supervises the Canonical Length Predictor (eq. 58).

    **Uses log1p MSE** — critical fix for training stability.

    Raw MSE between l_hat_v ≈ 0.69 (at init) and l_star ≈ 100 bytes
    gives (~9 870) per prediction, which dominates L1 and causes the
    oscillating loss curves seen in early training. Log1p shrinks the
    initial error to ≈16.9, letting other losses guide optimisation too.

    l_hat_v: (B, V-1)  — predicted canonical lengths (positive, from softplus)
    l_star:  (B,)      — ground-truth byte-lengths of the anchor
    """
    log_pred   = torch.log1p(l_hat_v)
    log_target = (
        torch.log1p(l_star.to(l_hat_v.dtype))
             .unsqueeze(1)
             .expand_as(l_hat_v)
    )
    return F.mse_loss(log_pred, log_target)


# ---------------------------------------------------------------------------
# Combined Phase-1 objective
# ---------------------------------------------------------------------------

def compute_phase1_loss(
    model_out:  dict,
    batch:      dict,
    global_step: int,
    lam:        float = 0.05,   # λ — weight for [SIGReg + JEPA] block
    beta:       float = 0.5,    # β — balance between SIGReg and JEPA
    eps:        float = 1.0,    # ε — weight for L_canon
    zeta:       float = 0.01,   # ζ — weight for L_canonlen  ← tightened (was 0.1)
    num_slices: int   = 1024,
) -> tuple[torch.Tensor, dict]:
    """
    Assemble all Phase-1 loss components and return the total scalar and a
    component dict for logging / tensorboard.

    model_out: output of SwaraJEPA.forward(..., mode="training")
    batch:     original input dict (needs "mask" for building Z_local, eq. 50)

    Returns: (L1_total, {component_name: tensor})
    """
    z_local_1   = model_out["z_local_1"]     # (B, C, d)
    z_local_v   = model_out["z_local_v"]     # (B, V-1, C, d)
    z_global_1  = model_out["z_global_1"]    # (B, d)
    z_global_v  = model_out["z_global_v"]    # (B, V-1, d)
    l_hat_v     = model_out["l_hat_v"]       # (B, V-1)
    l_star      = model_out["l_star"]        # (B,)
    zhat_canon  = model_out["zhat_canon"]    # (B, V, L*_max, d)
    canon_mask  = model_out["canon_mask"]    # (B, L*_max)

    # ── Gather all views into batch-first tensors ───────────────────────────
    z_local_all  = torch.cat([z_local_1.unsqueeze(1),  z_local_v],  dim=1)  # (B, V, C, d)
    z_global_all = torch.cat([z_global_1.unsqueeze(1), z_global_v], dim=1)  # (B, V, d)

    # ── Z_local  (eq. 50): valid positions only, across all views & samples ─
    local_mask = batch["mask"]               # (B, V, C) bool
    Z_local    = z_local_all[local_mask]     # (N_valid, d)

    # ── Z_global (eq. 51): all (v, b) — no position-level masking ───────────
    d_model  = z_global_all.shape[-1]
    Z_global = z_global_all.reshape(-1, d_model)   # (V·B, d)

    # ── Individual terms ────────────────────────────────────────────────────
    L_sigreg   = (
        SIGReg(Z_local,  global_step, num_slices=num_slices, seed_offset=0)
      + SIGReg(Z_global, global_step, num_slices=num_slices, seed_offset=1)
    )  # eq. 54

    L_jepa     = jepa_loss(z_global_all)            # eq. 56
    L_canon    = canon_loss(zhat_canon, canon_mask)  # eq. 57
    L_canonlen = canonlen_loss(l_hat_v, l_star)      # eq. 58 (log1p)

    # ── Total Phase-1 loss (eq. 62 in spec) ─────────────────────────────────
    L1 = (
        lam  * (beta * L_sigreg + (1.0 - beta) * L_jepa)
      + eps  * L_canon
      + zeta * L_canonlen
    )

    return L1, {
        "L_sigreg":   L_sigreg,
        "L_jepa":     L_jepa,
        "L_canon":    L_canon,
        "L_canonlen": L_canonlen,
        "L1":         L1,
    }