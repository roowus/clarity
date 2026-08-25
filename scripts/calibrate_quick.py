"""Quick provisional calibration for the default model pair.

Human corpus: docstrings from the Python 3.12 standard library on disk
(pre-LLM, human-written, license-clean). AI corpus: passages written by an
LLM (Claude) in varied registers — AI text by construction.

Small n, so this yields PROVISIONAL thresholds pending the full RAID-style
calibration (Phase 3). Output: score distributions + proposed thresholds.

Usage: .venv/bin/python scripts/calibrate_quick.py
"""

from __future__ import annotations

import glob
import io
import tokenize
from statistics import fmean

from clairity import ModelPair, analyze

N_HUMAN_DOCS = 40

AI_DOCS = [
    "Artificial intelligence has fundamentally transformed the way businesses operate in the modern era. From streamlining workflows to enhancing customer experiences, AI technologies offer unprecedented opportunities for growth and innovation. Organizations that embrace these tools position themselves at a competitive advantage.",
    "When considering a healthy lifestyle, it is important to note that balance plays a crucial role. Eating a variety of nutritious foods, exercising regularly, and getting adequate sleep are all essential components. Furthermore, maintaining hydration throughout the day can significantly impact overall well-being.",
    "The Industrial Revolution marked a pivotal turning point in human history. It reshaped economies, societies, and the daily lives of millions of people. In conclusion, the legacy of this era continues to influence the modern world in profound ways.",
    "Climate change represents one of the most pressing challenges of our time. Rising global temperatures, shifting weather patterns, and melting ice caps underscore the urgency of collective action. By working together, the international community can address these challenges.",
    "Learning a new language is a rewarding journey that opens doors to new cultures and perspectives. Consistent practice, immersion, and patience are key to success. Moreover, language learning has been shown to enhance cognitive abilities.",
    "In today's fast-paced digital world, remote work has become increasingly prevalent. This shift offers numerous benefits, including flexibility and reduced commuting time. However, it also presents challenges such as maintaining work-life boundaries.",
    "The history of jazz music is a rich tapestry of cultural influences and artistic innovation. From its origins in New Orleans to its global reach today, jazz has continually evolved while honoring its roots. It serves as a testament to the power of musical expression.",
    "Effective project management requires clear communication, well-defined goals, and adaptability. Teams that leverage agile methodologies can respond to changing requirements more effectively. Ultimately, successful projects depend on collaboration.",
    "Renewable energy sources, such as solar and wind power, play a crucial role in combating environmental degradation. As technology advances, these solutions become increasingly cost-effective and accessible. The transition to clean energy is essential for a sustainable future.",
    "Reading fiction offers numerous cognitive and emotional benefits. It enhances empathy, expands vocabulary, and provides a healthy form of escapism. Additionally, studies suggest that regular readers may experience reduced stress levels.",
]


def human_docs() -> list[str]:
    """Docstrings >= 300 chars extracted from the base Python stdlib."""
    import sys

    stdlib = glob.glob(f"{sys.base_prefix}/lib/python3.12/*.py")
    docs: list[str] = []
    for path in sorted(stdlib):
        if len(docs) >= N_HUMAN_DOCS:
            break
        try:
            with open(path, "rb") as f:
                for tok in tokenize.tokenize(f.readline):
                    if tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
                        body = tok.string[3:-3].strip()
                        if len(body) >= 300:
                            docs.append(body)
                            break
        except (tokenize.TokenError, SyntaxError, UnicodeDecodeError, OSError):
            continue
    return docs


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
    return s[idx]


def main() -> None:
    pair = ModelPair()
    print(f"device: {pair.device}")
    print(f"corpora: {len(AI_DOCS)} AI docs, ", end="")
    human = human_docs()
    print(f"{len(human)} human stdlib docstrings")

    ai_scores = [analyze(t, pair).doc_score for t in AI_DOCS]
    hu_scores = [analyze(t, pair).doc_score for t in human]

    print(f"\nAI   scores: min={min(ai_scores):.3f} mean={fmean(ai_scores):.3f} max={max(ai_scores):.3f}")
    print(f"human scores: min={min(hu_scores):.3f} mean={fmean(hu_scores):.3f} max={max(hu_scores):.3f}")
    print(f"sorted AI   : {[round(s,3) for s in sorted(ai_scores)]}")
    print(f"sorted human: {[round(s,3) for s in sorted(hu_scores)]}")

    # Fixed-FPR logic in miniature: threshold_low keeps ~95% of human above it;
    # threshold_high sits above every AI doc when possible.
    low = pct(hu_scores, 0.05)
    hi = max(ai_scores) + 0.005
    if hi <= low:
        hi = pct(hu_scores, 0.60)  # overlapping distributions: mid-band compromise
    print(f"\nproposed: threshold_low={low:.3f} threshold_high={hi:.3f}")
    fp = sum(1 for s in hu_scores if s < low) / len(hu_scores)
    det = sum(1 for s in ai_scores if s < low) / len(ai_scores)
    print(f"at threshold_low: human FPR={fp:.1%}, AI detection={det:.1%}")


if __name__ == "__main__":
    main()
