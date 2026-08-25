# Calibration

Last updated: 2026-08-25 (v0.1.0)

## Current state — be honest with yourself here

The default thresholds (`AI < 0.85`, `human > 0.92`) are **inherited from the
Binoculars paper's Falcon-7B calibration** and sanity-checked with smoke tests
on the default Qwen2.5-1.5B pair. They are NOT yet a rigorous calibration.
Treat v0.1.0 verdicts accordingly. Rigorous calibration is Phase 3 on the
roadmap (RAID subsets, fixed-FPR thresholding).

## Why thresholds are pair-specific

The Binoculars score is a ratio of two model-dependent quantities. Different
model pairs put human/AI text at different absolute score ranges. A threshold
tuned on Falcon-7B is only approximately right for Qwen-1.5B and could be
badly wrong for an arbitrary pair you bring.

## The right way to threshold (RAID methodology)

Naive thresholding is how open detectors end up with unacceptable real-world
false-positive rates. The RAID benchmark's methodology, which we adopt:

1. Score a large corpus of **known-human** text with your pair.
2. Pick the threshold at which only X% of human text falls below it
   (X = your acceptable FPR; RAID uses 5%, Binoculars targets 0.01%).
3. Report detection accuracy **at that fixed FPR** — never a raw accuracy.

## Recalibrating for your own pair

Until the automated script lands (Phase 3), the manual procedure:

```bash
# 1. Gather ≥500 known-human docs (pre-2020 text is safest) into human/*.txt
# 2. Score them:
for f in human/*.txt; do
  telltale "$f" --json --observer YOUR_OBS --performer YOUR_PERF \
    | jq .doc_score
done > human_scores.txt
# 3. threshold_low = the 5th percentile of human_scores.txt
# 4. Repeat with known-AI docs to check separation; threshold_high = a value
#    above which ~no AI docs land (e.g. 99th percentile of AI scores).
telltale essay.txt --threshold-low <yours> --threshold-high <yours>
```

Store per-pair calibrations you trust in `calibrations/` (JSON:
`{"observer": ..., "performer": ..., "low": ..., "high": ..., "fpr": ...,
"corpus": "...", "date": "..."}`). `calibrations/local/` is gitignored for
private corpora.

## Sentence-level caveat

Sentence thresholds reuse the document thresholds. This is a known
simplification: short spans have higher score variance, which we mitigate by
neighbor-blending (≥30 tokens per scored window) rather than per-length
thresholds. If Phase 3 calibration shows systematic sentence-level miscalibration,
per-window-size thresholds replace this.
