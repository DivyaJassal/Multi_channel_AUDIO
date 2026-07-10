"""
SI-SDR (scale-invariant SDR) loss + Permutation Invariant Training (PIT).

Why SI-SDR: it's invariant to a global scale factor between prediction and
target, so the model isn't penalized for getting the volume slightly wrong —
only the actual waveform shape/timing/separation quality matters. This is
the de-facto standard loss/metric for source separation research
(WSJ0-mix, LibriMix, DNS challenge, etc).

Why PIT: see README section 1. Ground truth speaker order is arbitrary, so
we score every output-target permutation and train on whichever pairing was
best. For C <= ~5 speakers, brute-force over all permutations is cheap. For
larger C, we use the Hungarian algorithm (scipy) instead of factorial
brute force.
"""
import itertools
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def si_sdr(estimate, target, eps=1e-8):
    """
    estimate, target: (..., T) waveforms.
    Returns SI-SDR in dB, same leading shape as input minus the time dim.
    """
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    # projection of estimate onto target (the "signal" part)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / target_energy

    noise = estimate - proj

    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def _all_permutations(c):
    return list(itertools.permutations(range(c)))


def pit_si_sdr_loss(estimates, targets, max_brute_force=5):
    """
    estimates: (B, C, T) model outputs
    targets:   (B, C, T) ground truth sources (same, arbitrary, order)

    Returns:
        loss: scalar tensor to backprop (negative mean best SI-SDR)
        best_si_sdr: (B,) best SI-SDR per example, for logging (positive dB)
        best_perm: (B, C) the winning permutation indices, for inspection
    """
    B, C, T = estimates.shape
    device = estimates.device

    if C <= max_brute_force:
        perms = _all_permutations(C)
        # pairwise SI-SDR matrix: (B, C_est, C_tgt)
        pairwise = torch.zeros(B, C, C, device=device)
        for i in range(C):
            for j in range(C):
                pairwise[:, i, j] = si_sdr(estimates[:, i], targets[:, j])

        best_scores = torch.full((B,), -1e9, device=device)
        best_perm = torch.zeros(B, C, dtype=torch.long, device=device)
        for perm in perms:
            score = sum(pairwise[:, i, p] for i, p in enumerate(perm)) / C
            improve = score > best_scores
            best_scores = torch.where(improve, score, best_scores)
            perm_t = torch.tensor(perm, device=device)
            best_perm[improve] = perm_t
        return -best_scores.mean(), best_scores.detach(), best_perm

    else:
        # Hungarian algorithm per-example — scales to large C where
        # brute-force permutations (C!) would be infeasible.
        pairwise = torch.zeros(B, C, C, device=device)
        for i in range(C):
            for j in range(C):
                pairwise[:, i, j] = si_sdr(estimates[:, i], targets[:, j])

        best_scores = torch.zeros(B, device=device)
        best_perm = torch.zeros(B, C, dtype=torch.long, device=device)
        cost = (-pairwise).detach().cpu().numpy()  # minimize cost = maximize SI-SDR
        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(cost[b])
            best_perm[b] = torch.tensor(col_ind, device=device)
            best_scores[b] = pairwise[b, row_ind, col_ind].mean()
        return -best_scores.mean(), best_scores.detach(), best_perm

def pit_si_sdr_loss_variable(estimates, targets, active_counts, silence_weight=0.1):
    """
    This project's dataset always returns exactly `max_sources` channels,
    zero-padding when a mixture has fewer real speakers. Plugging that
    straight into pit_si_sdr_loss would compute SI-SDR against all-zero
    "silence" targets, which is mathematically undefined (zero target
    energy) and gives no useful gradient telling the model to actually
    output silence there.

    This version matches only the REAL speakers (targets[:active_count])
    against the best-fitting subset of the model's output channels
    (Hungarian algorithm), and separately pushes the leftover, unmatched
    output channels toward silence with a simple energy penalty.

    estimates:      (B, C, T) — C = max_sources, model's fixed output count
    targets:        (B, C, T) — first active_counts[b] channels are real
                    speakers, remaining channels are zero-padding
    active_counts:  (B,) — real speaker count per example (this project's
                    dataset returns this as sample["num_sources"])
    """
    B, C, T = estimates.shape
    device = estimates.device

    total_loss = 0.0
    active_sisdr = torch.zeros(B, device=device)

    for b in range(B):
        c_active = int(active_counts[b].item())
        active_targets = targets[b, :c_active]  # (c_active, T)

        pairwise = torch.zeros(C, c_active, device=device)
        for i in range(C):
            for j in range(c_active):
                pairwise[i, j] = si_sdr(estimates[b, i], active_targets[j])

        cost = (-pairwise).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)  # len = min(C, c_active)

        matched_score = pairwise[row_ind, col_ind].mean()

        matched_estimate_idx = set(row_ind.tolist())
        unmatched_idx = [i for i in range(C) if i not in matched_estimate_idx]
        if unmatched_idx:
            unmatched_est = estimates[b, unmatched_idx]
            silence_loss = (unmatched_est ** 2).mean()
        else:
            silence_loss = torch.tensor(0.0, device=device)

        total_loss = total_loss + (-matched_score + silence_weight * silence_loss)
        active_sisdr[b] = matched_score.detach()

    return total_loss / B, active_sisdr