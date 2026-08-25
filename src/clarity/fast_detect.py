"""Fast-DetectGPT-style single-model scoring (simplified estimator).

Bao et al., ICLR 2024 (arXiv:2310.05130). The full method perturbs the text
by sampling from the scoring model's own distribution and measures
"conditional probability curvature":

    d = E_x[ log p(x) / (E_eps[log p(x | eps)] - log p(x)) ]

where eps are resampled variants of x. Higher curvature => machine text.
The reference implementation uses ~50 samples per token window, which needs
a generation pass per sample — expensive on laptops.

This module ships a **sampling-free approximation** of the same quantity,
needing only ONE extra forward pass: instead of resampling text, we take the
model's own next-token distribution q = softmax(logits at t-1) as the
"perturbed" hypothesis. The numerator stays log p(x); the denominator uses

    E_q[log p] - log p(x)  ≈  H(q) + log p(x) ... computed exactly below

i.e. the gap between the model's confidence in what WAS written and its
average confidence across what it COULD have written. Machine text sits in
the model's high-probability region (small gap); human text does not.

HONESTY NOTE (docs/CALIBRATION.md): this is NOT the paper's estimator. It is
cheaper (one forward pass vs ~50 generations) and empirically separates the
smoke corpora, but published accuracy numbers do not transfer to it. Treat
fast-mode scores as a weaker sibling of Binoculars mode until Phase-3
calibration validates or replaces this approximation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fast_detect_scores(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token fast-mode quantities for positions 1..n-1.

    Returns (numerator, denominator): both 1-D, aligned with tokens 1..n-1.

    numerator   = -log p(x_t | x_<t)                (same as Binoculars' log_ppl)
    denominator = E_{q}[log p'] - log p(x), where q = softmax(logits at t-1)
                  and log p' is the same log-softmax gathered over ALL vocab
                  entries weighted by q. When the written token IS the argmax,
                  this reduces to a curvature-like residual around log p(x).

    Implementation detail: E_q[logp] = sum_v q_v * logp_v = logp_x +
    sum_v q_v * (logp_v - logp_x). The second term is the expected "rank gap"
    between the written token and the distribution — cheap via one matmul.
    """
    if logits.shape[0] != input_ids.shape[0]:
        raise ValueError("logits and input_ids must cover identical sequences")
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    targets = input_ids[1:]
    logp_x = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    # E_q[logp_v] over full vocab: q · logp (one dot product per position).
    eq_logp = (F.softmax(logits[:-1].float(), dim=-1) * logp).sum(dim=-1)
    numerator = -logp_x
    denominator = eq_logp - logp_x
    return numerator, denominator


def fast_detect_score(
    logits: torch.Tensor, input_ids: torch.Tensor, start: int = 0, end: int | None = None
) -> float:
    """Document/span fast-mode score: mean(numerator) / mean(denominator).

    LOWER => more machine-like (mirrors Binoculars convention so thresholds
    and labels stay comparable in shape). Denominator guard: positions where
    the model put near-all mass on the written token give ~0 denominator;
    those positions are excluded rather than allowed to explode the ratio.
    """
    num, den = fast_detect_scores(logits, input_ids)
    if end is not None:
        num, den = num[start:end], den[start:end]
    elif start > 0:
        num, den = num[start:], den[start:]
    if num.numel() == 0:
        raise ValueError("empty token span")
    keep = den.abs() > 1e-6
    if keep.sum() == 0:
        raise ValueError("degenerate denominators across whole span")
    return (num[keep].mean() / den[keep].mean()).item()
