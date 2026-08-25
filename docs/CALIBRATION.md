# Calibration

Last updated: 2026-08-25 (post quick-calibration of v0.1.0)

## Current measured state (default Qwen2.5-1.5B pair)

Provisional quick calibration, 2026-08-25, via `scripts/calibrate_quick.py`:

- **AI corpus:** 10 passages written by an LLM (Claude), varied registers,
  ~60–140 tokens each.
- **Human corpus:** 40 docstrings ≥300 chars from the local Python 3.12
  standard library (pre-LLM, human-written).

| | min | mean | max |
| --- | --- | --- | --- |
| AI scores | 0.836 | 0.944 | 1.107 |
| human scores | 0.856 | 1.012 | 1.208 |

**Shipped thresholds: `low = 0.905`, `high = 1.11`.**

- At `low = 0.905`: **5% human FPR, ~40% AI detection**.
- The distributions overlap substantially at this model size. Mean separation
  exists (0.944 vs 1.012) but the 1.5B pair is **convenience-grade**: single-doc
  verdicts are weak, and much genuinely-AI text will read "uncertain". The
  paper's >90%-at-0.01%-FPR numbers are for the Falcon-7B pair.
- This is why the tool leads with per-sentence *evidence*, not a verdict.

**Known limitations of this calibration** (all fixed in Phase 3): tiny n,
single register per class, AI corpus from one generator family (Claude),
human corpus is technical rather than prose, thresholds not length-stratified.
`high = 1.11` implies only ~12% of the human corpus earns a "human" label —
honest for overlapping distributions, but better pairs/calibration will move it.

## Why thresholds are pair-specific

The Binoculars score is a ratio of two model-dependent quantities. Different
model pairs put human/AI text at different absolute score ranges — the paper's
Falcon thresholds (0.85/0.9015) mislabel on the Qwen pair (measured 2026-08-25:
a known-AI passage scored 0.963, above both Falcon cutoffs → "human"). Never
reuse thresholds across pairs.

## The right way to threshold (RAID methodology)

Naive thresholding is how open detectors end up with unacceptable real-world
false-positive rates. The RAID benchmark's methodology, which we adopt:

1. Score a large corpus of **known-human** text with your pair.
2. Pick the threshold at which only X% of human text falls below it
   (X = your acceptable FPR; RAID uses 5%, Binoculars targets 0.01%).
3. Report detection accuracy **at that fixed FPR** — never a raw accuracy.

## Recalibrating for your own pair

`scripts/calibrate_quick.py` gives provisional numbers in minutes (edit its
`AI_DOCS`, or point `human_docs()` at your own corpus — stdlib docstrings are
the default). For serious use, the manual procedure:

```bash
# 1. Gather ≥500 known-human docs (pre-2020 text is safest) into human/*.txt
# 2. Score them:
for f in human/*.txt; do
  clarity "$f" --json --observer YOUR_OBS --performer YOUR_PERF \
    | jq .doc_score
done > human_scores.txt
# 3. threshold_low = the 5th percentile of human_scores.txt
# 4. Repeat with known-AI docs to check separation; threshold_high = a value
#    above which ~no AI docs land (e.g. 99th percentile of AI scores).
clarity essay.txt --threshold-low <yours> --threshold-high <yours>
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
