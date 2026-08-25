# clarity — Design

Last updated: 2026-08-25 (v0.2.0 — Phase 2: server, web UI, fast mode)

## Goal

An open-source AI-text detector in the spirit of GPTZero, differentiated by
**transparent evidence**: per-sentence highlighting where every flag carries a
machine-checkable reason, and a bring-your-own-scoring-model design so nobody
depends on our hosted weights or a paid API.

## Research basis (what we chose and why)

Surveyed 2026-08-25; full citations at the bottom.

| Approach | Verdict | Why |
| --- | --- | --- |
| **Binoculars (zero-shot, 2 models)** | **Chosen core** | SOTA zero-shot: >90% detection at 0.01% FPR in-paper; no training data; generalizes across generator models; ~100 lines of math over `transformers` |
| Fast-DetectGPT (zero-shot, 1 model) | Possible cheap mode later | Only needs one model, but slightly weaker; the single-model seam exists in `scoring.py` if we add it |
| Fine-tuned classifier (DeBERTa etc.) | Phase 4 ensemble member | Top RAID-leaderboard entries ensemble a trained encoder with statistical scores; but alone it generalizes poorly to unseen generators (RoBERTa-GPT2 drops from 95% to <60% off-model) |
| Perplexity + burstiness alone (classic GPTZero) | Rejected as core, kept as *evidence signals* | Superseded (GPTZero themselves moved off it in 2023); raw perplexity confounds "predictable topic" with "machine text" — Binoculars' cross-perplexity denominator fixes exactly this |
| <a name="no-llm-judges"></a>LLM-as-judge ("hey GPT, is this AI?") | **Rejected, hard rule** | Empirically unreliable: GPT-4 misclassified ~95% of *human* text as AI in one study; GPT-4-Turbo <50% accuracy across prompt styles. LLMs may only *verbalize* already-computed evidence, never make the call |

## Architecture

```
                      ┌────────────── frontends ──────────────┐
                      │  cli.py        serve.py (FastAPI+UI)   │
                      └───────────────────┬───────────────────┘
                                          ▼
                                    detector.analyze()
                                          │
              ┌───────────────────────────┴────────────────────┐
              ▼                                                ▼
   ModelPair.score_text()                        FastModel.score_text()   models.py
   (2 models; paper-faithful)                    (1 model; EXPERIMENTAL —
              │                                  sampling-free Fast-DetectGPT
              │                                   approximation, fast_detect.py;
              │                                   measured ~10% det @5% FPR vs
              │                                   binoculars' ~40% — RAM-constrained
              │                                   machines only)
              ▼                                                │
      TokenScores {log_ppl[i], log_xppl[i], offsets[i]} ◄──────┘   scoring.py
              │        (fast mode: log_xppl carries its denominator)
              ├──► binoculars_score(full doc) ──► doc verdict (per-mode thresholds)
              │
              ├──► map_sentences_to_tokens()   sentences.py (pysbd, char-span accurate)
              │        each sentence = a token span; re-aggregate the SAME arrays
              ▼
      per-sentence scores (+ neighbor blending)   detector.py
              │
              ▼
      evidence signals per flagged sentence       evidence.py
              │
              ▼
      Report → CLI heatmap / JSON / web heatmap+evidence panel
```

The server (`serve.py`) loads models ONCE at startup and serves the single-page
web UI from `clarity/web/index.html` at `/`. The `AnalyzeBody` pydantic model is
module-level on purpose: FastAPI cannot resolve closure-local annotation classes,
and the body field silently degrades to a query param (hit in testing).

Key property: the models run **once**. Document score, sentence scores, and
burstiness are all re-aggregations of the same per-token arrays, so sentence
highlighting is free.

## Design decisions & rationale

1. **Score in log space, per token, keep the arrays** (`scoring.py`).
   Binoculars is defined as PPL/X-PPL; we compute mean-NLL ratios over
   arbitrary token spans so any text region can be scored post-hoc.

2. **Sentence spans via char offsets, not re-tokenization** (`sentences.py`).
   pysbd gives char-accurate sentence spans; HF tokenizers give char offsets
   per token. A token belongs to the sentence containing its start offset.
   Re-tokenizing sentences independently would change token boundaries and
   make scores incomparable.

3. **Neighbor blending for short sentences** (`detector.py:_blended_span`).
   Spans under ~30 tokens are statistically noisy (paper's reliability
   observations are at 50+ tokens for documents). Rather than lying with a
   confident color on an 8-word sentence, we widen the window symmetrically
   over neighboring sentences until ≥30 tokens, and mark the sentence
   `blended: true` (raw score still reported in JSON).

4. **Three-way labels, two thresholds** (ai < 0.905, human > 1.11, else
   uncertain). A single cutoff manufactures false confidence near the
   boundary. Defaults are empirically calibrated for the default pair
   (2026-08-25 quick calibration, 5% human FPR; measured detection ~40% at
   that FPR — the 1.5B pair is convenience-grade, see CALIBRATION.md for the
   distributions and why the Falcon-paper thresholds must never be reused
   across pairs).

5. **Evidence signals are computed, never generated** (`evidence.py`).
   - `low_binoculars`: the span score itself, with the numbers.
   - `low_burstiness`: std-dev of neighboring sentences' mean log-perplexity
     < 0.15 with low mean — the classic "AI writes with flat predictability"
     signal, scoped locally so one flat paragraph in a varied essay still fires.
   - `ai_idiom`: curated overrepresented-phrase list; explicitly labeled
     "stylistic signal only" and never sufficient alone.
   - `short_span`: honesty marker that the sentence leaned on neighbors.
   Signals fire only on non-"human" sentences to avoid decorating clean text.

6. **BYO-model with a shared-tokenizer guard** (`models.py`). Binoculars
   requires the pair to share a vocabulary (cross-entropy is computed across
   distributions over the same token space); we verify `get_vocab()` equality
   at load and fail with an actionable message. Default pair is
   Qwen2.5-1.5B-Instruct (observer) + Qwen2.5-1.5B (performer): small enough
   for laptops (fp16 on MPS/CUDA, fp32 CPU), same base/instruct sibling
   structure the paper found optimal in Falcon.

7. **Truncation + reliability flags surfaced, not hidden.** Documents are
   scored up to 4096 tokens; beyond that `truncated: true`. Under 50 scored
   tokens ⇒ `reliable: false` and the CLI says so.

8. **Ethics in the output path.** The CLI prints the fallibility caveat on
   every run; the web UI shows it in the verdict card. The README documents
   the non-native-speaker false-positive problem. Labels say "likely", never
   certainty.

9. **Fast mode is an approximation and says so.** (Phase 2) The published
   Fast-DetectGPT resamples ~50 token variants per position; ours computes a
   sampling-free residual from one forward pass (`fast_detect.py` header has
   the math and the honesty note). Empirically on our quick corpora it
   separated far worse than binoculars (10% vs 40% detection at 5% FPR), so
   it ships EXPERIMENTAL with its own scale and thresholds, and the UI/README
   label it as weaker. A future Phase 3/4 task is to implement the true
   sampling estimator and re-evaluate.

10. **Server is local-first.** (Phase 2) `serve.py` binds 127.0.0.1 by
    default, loads models once at startup, and serves the single-page UI
    (`clarity/web/index.html`) — no build step, no CDN, system fonts, so the
    UI works offline. The UI escapes all rendered text (`esc()`) — pasted
    content is untrusted input. Dark-mode support via `prefers-color-scheme`;
    keyboard: ⌘/Ctrl+Enter to analyze; `aria-live` status region.

## Testing strategy

Math, segmentation, and evidence layers are pure functions tested without any
model download (`tests/`). Model-dependent behavior is validated by a manual
smoke script (`scripts/smoke.py`) that scores a known-human and a known-AI
passage and asserts the AI passage scores lower — run before releases, not in CI
(CI has no GPU and shouldn't download 6 GB).

## Citations

- Hans et al., *Spotting LLMs With Binoculars*, ICML 2024 — arXiv:2401.12070
- Bao et al., *Fast-DetectGPT*, ICLR 2024 — arXiv:2310.05130
- Dugan et al., *RAID benchmark*, ACL 2024 — arXiv:2405.07940
- Wu et al., survey incl. LLM-as-judge unreliability — arXiv:2310.14724
- GPTZero methodology explainers (perplexity/burstiness, sentence highlighting)
