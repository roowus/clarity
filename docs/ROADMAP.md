# Roadmap

Last updated: 2026-08-25

## Phase 1 — Core engine ✅ (v0.1.0, this release)
- [x] Binoculars scoring (per-token, span-aggregatable) — `scoring.py`
- [x] BYO model pair with shared-tokenizer guard; Qwen2.5-1.5B default — `models.py`
- [x] Sentence segmentation → token spans, neighbor blending — `sentences.py`, `detector.py`
- [x] Evidence signals (low score, low burstiness, AI idioms, short span) — `evidence.py`
- [x] CLI with terminal heatmap + `--json` — `cli.py`
- [x] Unit tests for all model-free layers; smoke script for the model path
- [x] Docs: README, DESIGN, CALIBRATION, ROADMAP

## Phase 2 — Serving & UI
- [ ] FastAPI server (`telltale serve`): POST /analyze → Report JSON; model pair
      loaded once, warm
- [ ] Web UI: paste text → heatmap highlighting, hover a sentence for its
      signals; static frontend hitting the local API
- [ ] Fast-DetectGPT single-model mode (`--mode fast`) for low-RAM machines

## Phase 3 — Rigorous calibration & eval (the credibility phase)
- [ ] `scripts/calibrate.py`: automated fixed-FPR thresholding from a
      human corpus (RAID methodology)
- [ ] Evaluate on RAID subsets (incl. adversarial: paraphrase, homoglyph);
      publish honest accuracy-at-5%-FPR numbers in the README
- [ ] Per-pair calibration registry in `calibrations/`
- [ ] Submit to the public RAID leaderboard (raid-bench.xyz)

## Phase 4 — Ensemble & explanations
- [ ] Optional DeBERTa-v3 fine-tuned classifier head (e.g. desklib detector),
      ensembled with the Binoculars score (what top RAID entries do)
- [ ] Optional LLM *verbalizer*: turns computed signals into prose. Hard rule
      from DESIGN.md: it may only restate computed evidence, never judge
- [ ] PDF/docx ingestion

## Non-goals
- Hosted service with our GPUs (this is a local-first/self-host project)
- "Humanizer" / evasion features
- Claiming courtroom-grade certainty — see README limitations
