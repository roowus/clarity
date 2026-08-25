import pytest
import torch

from clarity.fast_detect import fast_detect_score, fast_detect_scores


def _logits(n=8, vocab=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, vocab, generator=g)


def test_output_shapes():
    logits = _logits(9)
    ids = torch.randint(0, 20, (9,))
    num, den = fast_detect_scores(logits, ids)
    assert num.shape == (8,) and den.shape == (8,)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fast_detect_scores(_logits(6), torch.randint(0, 20, (7,)))


def test_greedy_token_gives_zero_denominator():
    # If the written token IS the argmax of a sharply peaked distribution,
    # E_q[logp] - logp_x ≈ 0 → that position must be excluded, not divide.
    vocab = 50
    logits = torch.full((3, vocab), -10.0)
    logits[:, 7] = 20.0  # sharp peak on token 7
    ids = torch.tensor([0, 7, 7])
    _, den = fast_detect_scores(logits, ids)
    assert (den.abs() < 1e-5).all()


def test_span_scoring_excludes_degenerate_positions():
    # Whole span degenerate → explicit error rather than inf/nan.
    vocab = 50
    logits = torch.full((4, vocab), -10.0)
    logits[:, 3] = 20.0
    ids = torch.tensor([0, 3, 3, 3])
    with pytest.raises(ValueError):
        fast_detect_score(logits, ids)


def test_score_finite_and_signed():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(12, 30, generator=g)
    ids = torch.randint(0, 30, (12,), generator=g)
    s = fast_detect_score(logits, ids)
    assert torch.isfinite(torch.tensor(s))
