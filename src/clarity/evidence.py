"""Evidence signals: measurable reasons attached to flagged sentences.

Every reason is derived from computed statistics — never from asking an LLM
for its opinion (LLM-as-judge is demonstrably unreliable for this task; see
docs/DESIGN.md#no-llm-judges).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

# Phrases statistically overrepresented in LLM output. Secondary signal only —
# clearly labeled as stylistic, never sufficient to flag a sentence alone.
AI_IDIOMS = [
    "delve into",
    "delves into",
    "it's important to note",
    "it is important to note",
    "in today's fast-paced world",
    "in the ever-evolving landscape",
    "tapestry of",
    "underscores the importance",
    "plays a crucial role",
    "plays a pivotal role",
    "a testament to",
    "navigate the complexities",
    "in conclusion,",
    "furthermore,",
    "moreover,",
    "harness the power",
    "unlock the potential",
    "seamlessly integrate",
]
_IDIOM_RE = re.compile("|".join(re.escape(p) for p in AI_IDIOMS), re.IGNORECASE)


@dataclass
class Signal:
    kind: str  # machine-readable, e.g. "low_perplexity"
    detail: str  # human-readable explanation with the numbers that fired


def sentence_signals(
    sent_score: float,
    sent_log_ppl: float,
    doc_score: float,
    threshold: float,
    neighbor_log_ppls: list[float],
    sent_text: str,
    n_tokens: int,
) -> list[Signal]:
    """Compute which evidence signals fire for one sentence."""
    signals: list[Signal] = []

    if sent_score < threshold:
        signals.append(
            Signal(
                kind="low_binoculars",
                detail=(
                    f"Binoculars score {sent_score:.3f} is below the AI threshold "
                    f"{threshold:.3f} — both scoring models found this sentence "
                    "predictable in the same way, a machine-generation signature."
                ),
            )
        )

    if sent_log_ppl is not None and neighbor_log_ppls:
        mean_n = statistics.fmean(neighbor_log_ppls)
        if len(neighbor_log_ppls) >= 3:
            stdev_n = statistics.stdev(neighbor_log_ppls)
            if stdev_n < 0.15 and mean_n < 2.5:
                signals.append(
                    Signal(
                        kind="low_burstiness",
                        detail=(
                            f"This sentence sits in a run of {len(neighbor_log_ppls)} "
                            f"sentences with near-identical predictability (σ={stdev_n:.2f}) "
                            "— human writing varies more sentence-to-sentence."
                        ),
                    )
                )

    m = _IDIOM_RE.search(sent_text)
    if m:
        signals.append(
            Signal(
                kind="ai_idiom",
                detail=(
                    f'Contains "{m.group(0)}", a phrase statistically overrepresented '
                    "in LLM output (stylistic signal only)."
                ),
            )
        )

    if n_tokens < 12:
        signals.append(
            Signal(
                kind="short_span",
                detail=(
                    f"Only {n_tokens} tokens — scores on short sentences are noisy; "
                    "this sentence's rating leans on its neighbors."
                ),
            )
        )
    return signals
