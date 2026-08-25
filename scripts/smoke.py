"""Manual smoke test: real models, real texts. Run before releases.

Usage: .venv/bin/python scripts/smoke.py
Downloads the default pair on first run (~6 GB). Asserts the AI-typical passage
scores LOWER than the human-typical passage and prints both reports.
"""

import json

from clarity import ModelPair, analyze

# Human-typical: pre-LLM prose (Twain, 1883 — public domain), bursty and idiosyncratic.
HUMAN = (
    "The Mississippi is well worth reading about. It is not a commonplace "
    "river, but on the contrary is in all ways remarkable. Considering the "
    "Missouri its main branch, it is the longest river in the world - four "
    "thousand three hundred miles. It seems safe to say that it is also the "
    "crookedest river in the world, since in one part of its journey it uses "
    "up one thousand three hundred miles to cover the same ground that the "
    "crow would fly over in six hundred and seventy-five. It discharges three "
    "times as much water as the St. Lawrence, twenty-five times as much as "
    "the Rhine, and three hundred and thirty-eight times as much as the Thames."
)

# AI-typical: the flat, formulaic register LLMs produce for generic prompts.
AI = (
    "Rivers play a crucial role in shaping our world and supporting human "
    "civilization. The Mississippi River is one of the most important rivers "
    "in the United States. It provides water for agriculture, supports diverse "
    "ecosystems, and serves as a major transportation route. Furthermore, the "
    "river plays a pivotal role in the economy of the region. It is important "
    "to note that the Mississippi River faces many environmental challenges "
    "today. In conclusion, protecting this vital waterway is essential for "
    "future generations. By working together, we can ensure that the river "
    "continues to thrive in today's fast-paced world."
)


def main() -> None:
    pair = ModelPair()
    print(f"device: {pair.device}\n")
    human_report = analyze(HUMAN, pair)
    ai_report = analyze(AI, pair)
    print(f"human passage: score={human_report.doc_score:.4f} label={human_report.doc_label}")
    print(f"ai passage:    score={ai_report.doc_score:.4f} label={ai_report.doc_label}")
    print("\nAI-passage sentence detail:")
    print(json.dumps(ai_report.to_dict()["sentences"], indent=2)[:3000])
    assert ai_report.doc_score < human_report.doc_score, (
        "SMOKE FAIL: AI passage did not score lower than human passage"
    )
    print("\nSMOKE PASS: AI passage scored lower than human passage")


if __name__ == "__main__":
    main()
