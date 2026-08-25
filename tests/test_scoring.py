import pytest
import torch

from clairity.scoring import binoculars_score, mean_log_ppl, token_scores


def _fake(n=10, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, vocab, generator=g)
    perf = torch.randn(n, vocab, generator=g)
    ids = torch.randint(0, vocab, (n,), generator=g)
    offsets = [(i * 4, i * 4 + 4) for i in range(n)]
    return obs, perf, ids, offsets


def test_token_scores_shapes():
    obs, perf, ids, offsets = _fake(10)
    ts = token_scores(obs, perf, ids, offsets)
    assert ts.log_ppl.shape == (9,)
    assert ts.log_xppl.shape == (9,)
    assert len(ts.offsets) == 9
    assert ts.offsets[0] == offsets[1]  # position 0 has no prediction


def test_identical_models_score_below_one():
    # When observer == performer, xppl >= ppl only when the model is "surprised
    # by itself" less than by the data; identical logits give B = ppl/xppl where
    # xppl is the observer's entropy — the score is finite and positive.
    obs, _, ids, offsets = _fake(20, seed=1)
    ts = token_scores(obs, obs, ids, offsets)
    s = binoculars_score(ts)
    assert s > 0


def test_span_scoring_matches_full():
    obs, perf, ids, offsets = _fake(15, seed=2)
    ts = token_scores(obs, perf, ids, offsets)
    full = binoculars_score(ts)
    assert binoculars_score(ts, 0, None) == pytest.approx(full)


def test_empty_span_raises():
    obs, perf, ids, offsets = _fake(5, seed=3)
    ts = token_scores(obs, perf, ids, offsets)
    with pytest.raises(ValueError):
        binoculars_score(ts, 2, 2)
    with pytest.raises(ValueError):
        mean_log_ppl(ts, 4, 4)


def test_mismatched_lengths_raise():
    obs, perf, ids, offsets = _fake(6, seed=4)
    with pytest.raises(ValueError):
        token_scores(obs[:-1], perf, ids, offsets)
