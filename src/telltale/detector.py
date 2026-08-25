"""Top-level detector: document verdict + per-sentence report."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .evidence import Signal, sentence_signals
from .models import ModelPair
from .scoring import TokenScores, binoculars_score, mean_log_ppl
from .sentences import SentenceSpan, map_sentences_to_tokens

# Provisionally calibrated 2026-08-25 on the default Qwen2.5-1.5B pair via
# scripts/calibrate_quick.py (10 AI docs vs 40 human stdlib docstrings):
# low=0.905 keeps human FPR at 5%; high=1.11 sits above ~all AI docs.
# Detection at that FPR is only ~40% — the 1.5B pair is convenience-grade.
# Bigger pairs need recalibration; see docs/CALIBRATION.md for numbers and method.
DEFAULT_THRESHOLD_LOW = 0.905
DEFAULT_THRESHOLD_HIGH = 1.11

MIN_RELIABLE_TOKENS = 50  # document verdicts below this are labeled unreliable
BLEND_MIN_TOKENS = 30  # sentences shorter than this are blended with neighbors


@dataclass
class SentenceReport:
    text: str
    char_start: int
    char_end: int
    score: float  # blended Binoculars score used for the verdict
    raw_score: float | None  # unblended score (None if the span alone was degenerate)
    label: str  # "ai" | "uncertain" | "human"
    blended: bool
    signals: list[Signal] = field(default_factory=list)


@dataclass
class Report:
    doc_score: float
    doc_label: str  # "ai" | "uncertain" | "human"
    reliable: bool  # False when the doc is too short for a trustworthy verdict
    truncated: bool
    n_tokens: int
    threshold_low: float
    threshold_high: float
    sentences: list[SentenceReport]

    def to_dict(self) -> dict:
        return {
            "doc_score": round(self.doc_score, 4),
            "doc_label": self.doc_label,
            "reliable": self.reliable,
            "truncated": self.truncated,
            "n_tokens": self.n_tokens,
            "thresholds": {"low": self.threshold_low, "high": self.threshold_high},
            "sentences": [
                {
                    "text": s.text,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "score": round(s.score, 4),
                    "raw_score": None if s.raw_score is None else round(s.raw_score, 4),
                    "label": s.label,
                    "blended": s.blended,
                    "signals": [
                        {"kind": g.kind, "detail": g.detail} for g in s.signals
                    ],
                }
                for s in self.sentences
            ],
        }


def _label(score: float, low: float, high: float) -> str:
    if score < low:
        return "ai"
    if score > high:
        return "human"
    return "uncertain"


def _blended_span(
    spans: list[SentenceSpan], i: int, min_tokens: int
) -> tuple[int, int, bool]:
    """Widen sentence i's token span symmetrically over neighbors until it has
    at least min_tokens scored tokens. Returns (tok_start, tok_end, blended)."""
    start, end = spans[i].tok_start, spans[i].tok_end
    if end - start >= min_tokens:
        return start, end, False
    lo, hi = i, i
    while end - start < min_tokens and (lo > 0 or hi < len(spans) - 1):
        if lo > 0:
            lo -= 1
            start = spans[lo].tok_start
        if end - start < min_tokens and hi < len(spans) - 1:
            hi += 1
            end = spans[hi].tok_end
    return start, end, True


def analyze(
    text: str,
    pair: ModelPair,
    threshold_low: float = DEFAULT_THRESHOLD_LOW,
    threshold_high: float = DEFAULT_THRESHOLD_HIGH,
) -> Report:
    scores, truncated = pair.score_text(text)
    n_tokens = scores.log_ppl.numel()
    doc_score = binoculars_score(scores)
    spans = map_sentences_to_tokens(text, scores.offsets)

    sent_log_ppls = [
        mean_log_ppl(scores, s.tok_start, s.tok_end) for s in spans
    ]

    sentences: list[SentenceReport] = []
    for i, span in enumerate(spans):
        b_start, b_end, blended = _blended_span(spans, i, BLEND_MIN_TOKENS)
        score = binoculars_score(scores, b_start, b_end)
        raw = (
            binoculars_score(scores, span.tok_start, span.tok_end)
            if span.n_tokens > 0
            else None
        )
        label = _label(score, threshold_low, threshold_high)
        neighbors = sent_log_ppls[max(0, i - 2) : i + 3]
        signals = (
            sentence_signals(
                sent_score=score,
                sent_log_ppl=sent_log_ppls[i],
                doc_score=doc_score,
                threshold=threshold_low,
                neighbor_log_ppls=neighbors,
                sent_text=span.text,
                n_tokens=span.n_tokens,
            )
            if label != "human"
            else []
        )
        sentences.append(
            SentenceReport(
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                score=score,
                raw_score=raw,
                label=label,
                blended=blended,
                signals=signals,
            )
        )

    return Report(
        doc_score=doc_score,
        doc_label=_label(doc_score, threshold_low, threshold_high),
        reliable=n_tokens >= MIN_RELIABLE_TOKENS,
        truncated=truncated,
        n_tokens=n_tokens,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        sentences=sentences,
    )
