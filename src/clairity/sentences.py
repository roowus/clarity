"""Sentence segmentation and mapping of sentences to scored-token spans."""

from __future__ import annotations

from dataclasses import dataclass

import pysbd

_SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


@dataclass
class SentenceSpan:
    text: str
    char_start: int
    char_end: int
    tok_start: int  # index into TokenScores arrays (position-1-based tokens)
    tok_end: int  # exclusive

    @property
    def n_tokens(self) -> int:
        return self.tok_end - self.tok_start


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence_text, char_start, char_end) tuples covering the text."""
    return [(s.sent, s.start, s.end) for s in _SEGMENTER.segment(text) if s.sent.strip()]


def map_sentences_to_tokens(
    text: str, offsets: list[tuple[int, int]]
) -> list[SentenceSpan]:
    """Assign each scored token to the sentence containing its start offset.

    `offsets` are the char spans of the *scored* tokens (i.e. TokenScores.offsets).
    A token belongs to sentence s if its start char falls in [s.start, s.end).
    Sentences that receive zero tokens are dropped (they carry no signal).
    """
    sents = split_sentences(text)
    spans: list[SentenceSpan] = []
    tok_i = 0
    n = len(offsets)
    for sent_text, c_start, c_end in sents:
        # Skip tokens that start before this sentence (whitespace, prior overlap).
        while tok_i < n and offsets[tok_i][0] < c_start:
            tok_i += 1
        start = tok_i
        while tok_i < n and offsets[tok_i][0] < c_end:
            tok_i += 1
        if tok_i > start:
            spans.append(
                SentenceSpan(
                    text=sent_text,
                    char_start=c_start,
                    char_end=c_end,
                    tok_start=start,
                    tok_end=tok_i,
                )
            )
    return spans
