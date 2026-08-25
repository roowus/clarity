# Changelog

Format: Keep a Changelog. Versions: semver. Every behavioral change lands here
in the same commit that changes the behavior.

## [Unreleased]
### Changed
- **Project renamed: telltale → clairity** (package, module paths, CLI name,
  repo URL github.com/roowus/clairity — old links redirect). No behavioral
  change; scores, thresholds, and output format are identical.
- Default thresholds recalibrated for the Qwen2.5-1.5B pair: 0.905/1.11
  (5% human FPR, measured ~40% AI detection; the Falcon-paper defaults
  0.85/0.92 mislabeled a known-AI passage as human). Added
  `scripts/calibrate_quick.py` and the measured-distribution table to
  docs/CALIBRATION.md; README limitations section now states the honest
  default-pair accuracy.

## [0.1.0] — 2026-08-25
### Added
- Binoculars scoring engine (per-token, span-aggregatable, log-space) — `scoring.py`
- BYO model pair loader with shared-tokenizer guard; default Qwen2.5-1.5B
  base+instruct; CUDA/MPS/CPU with fp16 on accelerators — `models.py`
- Sentence segmentation (pysbd) mapped to token spans via char offsets — `sentences.py`
- Document + per-sentence verdicts with two thresholds and neighbor blending
  for short sentences — `detector.py`
- Evidence signals: `low_binoculars`, `low_burstiness`, `ai_idiom`, `short_span` — `evidence.py`
- CLI with terminal heatmap, "Why flagged" section, `--json` output, BYO-pair
  and threshold flags — `cli.py`
- Unit tests (13) for model-free layers; `scripts/smoke.py` for the model path
- Docs: README, DESIGN, CALIBRATION, ROADMAP, this changelog
