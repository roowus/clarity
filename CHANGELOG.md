# Changelog

Format: Keep a Changelog. Versions: semver. Every behavioral change lands here
in the same commit that changes the behavior.

## [Unreleased]
### Added
- **Live progress bar** during analysis: `/analyze` is now job-based — POST
  returns a `job_id` immediately, `GET /analyze/{id}` polls
  `{state, progress, stage, result}`. Progress reflects REAL stages
  (tokenize → observer pass → performer pass → sentence scoring → report),
  reported from inside the engine via an optional `progress` callback on
  `analyze()` and both scorers' `score_text()`. The UI shows a determinate
  bar + stage text (indeterminate only while queued); respects
  prefers-reduced-motion.
- **Every sentence shown**: results now render ALL sentences — each carries a
  score chip in the heatmap (≈ marks neighbor-blended scores), human sentences
  get a light-green tint (previously invisible/unstyled), and a per-sentence
  table (# / sentence / score / label / why-flagged) replaces the old
  flagged-only evidence list.

## [0.2.0] — 2026-08-25 (Phase 2)
### Added
- HTTP server + web UI: `clarity-server` (extras: `pip install -e ".[serve]"`)
  serves a single-page interface at http://127.0.0.1:8390 — verdict card,
  per-sentence heatmap, evidence list, mode toggle, dark mode, ⌘/Ctrl+Enter.
  API: `POST /analyze {"text", "mode"}` → full Report JSON; `GET /api/health`.
  Localhost-bound by default; models load once and stay warm — `serve.py`,
  `web/index.html`.
- Fast mode (`--mode fast` / server `--mode fast`, `FastModel`): single-model
  experimental detector using a sampling-free Fast-DetectGPT approximation
  (`fast_detect.py`). Own score scale; thresholds −41.86/62.84 from the same
  quick calibration. Measured ~10% detection at 5% human FPR vs binoculars'
  ~40% — documented as experimental, RAM-constrained use only.
- Report JSON now includes `mode`. CLI gains `--mode`, `--model`, and optional
  threshold flags (default to per-mode calibrated values).
- 5 new unit tests (fast-mode math incl. degenerate-denominator guards).
### Fixed
- `serve.py`: AnalyzeBody moved to module level — closure-local pydantic
  classes break FastAPI body detection (field silently becomes a query param).
### Changed
- **Project renamed: telltale → clarity** (briefly misspelled "clairity" in
  an intermediate commit; package, module paths, CLI name, and repo URL all
  renamed to clarity — github.com/roowus/clarity, and the misspelled repo
  name redirects). No behavioral change; scores, thresholds, and output
  format are identical.
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
