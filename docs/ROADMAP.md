# Roadmap

Last updated: 2026-08-25 (v0.2.0 — Phase 2 complete)

## Phase 1 — Core engine ✅ (v0.1.0)
- [x] Binoculars scoring (per-token, span-aggregatable) — `scoring.py`
- [x] BYO model pair with shared-tokenizer guard; Qwen2.5-1.5B default — `models.py`
- [x] Sentence segmentation → token spans, neighbor blending — `sentences.py`, `detector.py`
- [x] Evidence signals (low score, low burstiness, AI idioms, short span) — `evidence.py`
- [x] CLI with terminal heatmap + `--json` — `cli.py`
- [x] Unit tests for all model-free layers; smoke script for the model path
- [x] Docs: README, DESIGN, CALIBRATION, ROADMAP

## Phase 2 — Serving & UI ✅ (v0.2.0)
- [x] FastAPI server (`clarity-server`): POST /analyze → Report JSON,
      GET /api/health, models loaded once at startup, localhost-first — `serve.py`
- [x] Web UI at `/`: paste text → verdict card + sentence heatmap + evidence
      list; single static page (no build step, offline-capable, dark mode,
      ⌘/Ctrl+Enter, aria-live) — `web/index.html`
- [x] Single-model mode (`--mode fast` CLI + server flag) — ships
      EXPERIMENTAL: sampling-free approximation measured at ~10% detection
      @ 5% FPR vs binoculars' ~40% (CALIBRATION.md); use only when RAM-bound
- [x] E2E verified: /api/health, /analyze (full Report JSON incl. per-sentence
      signals), and / (UI HTML) all exercised against a live server

## Phase 3 — Rigorous calibration & eval (the credibility phase)
- [x] Quick provisional calibration shipped (2026-08-25):
      `scripts/calibrate_quick.py` → thresholds 0.905/1.11 at 5% human FPR,
      measured ~40% AI detection on the default pair (docs/CALIBRATION.md)
- [ ] `scripts/calibrate.py`: automated fixed-FPR thresholding from a LARGE,
      multi-register human corpus + multi-generator AI corpus (RAID methodology);
      current numbers are n=10/n=40 single-register provisional
- [ ] Candidate pair shootout: try Llama-3.2-3B, gemma-2-2b, Falcon-7B pairs;
      ship per-pair calibrated defaults so BYO users get sane numbers free
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
- [ ] True Fast-DetectGPT sampling estimator (~50 generations/doc) to replace
      the weak approximation, then re-measure fast-mode thresholds

## Non-goals
- Hosted service with our GPUs (this is a local-first/self-host project)
- "Humanizer" / evasion features
- Claiming courtroom-grade certainty — see README limitations
