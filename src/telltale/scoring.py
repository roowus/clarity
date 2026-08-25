"""Binoculars scoring math (Hans et al., ICML 2024, arXiv:2401.12070).

Score = observer perplexity / cross-perplexity between observer and performer,
computed in log space per token so spans of the document can be re-aggregated
without re-running the models.

Lower score => more likely machine-generated (both models agree the text is
predictable). Higher => more likely human.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TokenScores:
    """Per-token quantities for positions 1..n-1 (position 0 has no prediction).

    All tensors are 1-D of equal length, aligned with token indices 1..n-1 of
    the original encoding.
    """

    log_ppl: torch.Tensor  # -log p_observer(x_i | x_<i)
    log_xppl: torch.Tensor  # cross-entropy of performer's dist under observer's log-probs
    token_ids: torch.Tensor
    offsets: list[tuple[int, int]]  # char offsets into the original text, per token


def token_scores(
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> TokenScores:
    """Compute per-token log-perplexity and log-cross-perplexity.

    observer_logits / performer_logits: (n, vocab) float tensors for one document.
    input_ids: (n,) token ids. offsets: n char-span tuples.
    """
    if observer_logits.shape[0] != performer_logits.shape[0]:
        raise ValueError("observer and performer must be run on identical token sequences")
    # Predictions at position i-1 correspond to the token at position i.
    obs_logp = F.log_softmax(observer_logits[:-1].float(), dim=-1)
    perf_p = F.softmax(performer_logits[:-1].float(), dim=-1)
    targets = input_ids[1:]

    log_ppl = -obs_logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_xppl = -(perf_p * obs_logp).sum(dim=-1)
    return TokenScores(
        log_ppl=log_ppl,
        log_xppl=log_xppl,
        token_ids=targets,
        offsets=offsets[1:],
    )


def binoculars_score(scores: TokenScores, start: int = 0, end: int | None = None) -> float:
    """Binoculars score over token positions [start, end) of the scored tokens.

    The paper's B = PPL / X-PPL; means are taken in log space (mean NLL ratio),
    matching the reference implementation.
    """
    log_ppl = scores.log_ppl[start:end]
    log_xppl = scores.log_xppl[start:end]
    if log_ppl.numel() == 0:
        raise ValueError("empty token span")
    xppl_mean = log_xppl.mean().item()
    if xppl_mean == 0.0:
        raise ValueError("degenerate cross-perplexity (identical constant distributions)")
    return log_ppl.mean().item() / xppl_mean


def mean_log_ppl(scores: TokenScores, start: int = 0, end: int | None = None) -> float:
    span = scores.log_ppl[start:end]
    if span.numel() == 0:
        raise ValueError("empty token span")
    return span.mean().item()
