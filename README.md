# clarity

**Evidence-first, open-source AI-text detection.** Paste text, get a calibrated
score, and — unlike black-box percentage detectors — see *which sentences* look
machine-generated and *why*, with every reason backed by a measurable statistic.

```
$ clarity essay.txt
╭──────────────────────────── clarity ────────────────────────────╮
│ Document: likely AI-generated (score 0.812; AI < 0.85, human > 0.92) │
╰───────────────────────────────────────────────────────────────────╯
<text with per-sentence red/yellow highlighting>

Why flagged:
  ▌ "In today's fast-paced world, technology plays a crucial role..." (score 0.79)
      • Binoculars score 0.790 is below the AI threshold 0.850 — both scoring
        models found this sentence predictable in the same way.
      • Contains "in today's fast-paced world", a phrase statistically
        overrepresented in LLM output (stylistic signal only).
```

## How it works

clarity implements **[Binoculars](https://arxiv.org/abs/2401.12070)** (Hans et
al., ICML 2024), a zero-shot detection method: two closely related language
models (a base model and its instruct fine-tune) both score the text, and the
ratio of perplexity to cross-perplexity separates human from machine writing.
No detection-specific training; in the paper this detects >90% of
ChatGPT-generated text at a 0.01% false-positive rate.

On top of the document score, clarity re-aggregates the same per-token
log-probabilities per sentence (with neighbor-blending for short sentences,
which are individually too noisy to score) and attaches **evidence signals**:
low Binoculars score, low local burstiness, AI-idiom phrases, short-span
warnings. See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture and
the reasoning behind every choice.

## Install

```bash
git clone https://github.com/roowus/clarity && cd clarity
uv venv && uv pip install -e .
```

First run downloads the default model pair (Qwen2.5-1.5B base + instruct,
~6 GB) from Hugging Face. Runs on CUDA, Apple Silicon (MPS), or CPU.

## Usage

Three interfaces over one engine:

### 1. CLI

```bash
clarity examples/ai-slop.txt    # rich terminal report (try both examples/)
echo "some text" | clarity -    # stdin
clarity essay.txt --json        # machine-readable full report
clarity essay.txt --mode fast   # single-model mode (experimental; see below)
clarity essay.txt --observer meta-llama/Llama-3.2-3B-Instruct \
                    --performer meta-llama/Llama-3.2-3B   # bring your own pair
```

### 2. Web UI + local API

```bash
uv pip install -e ".[serve]"
clarity-server                  # http://127.0.0.1:8390 — models load once, stay warm
# options: --mode fast · --host 0.0.0.0 to expose beyond localhost · --port N
```

Then open the printed URL: paste text, hit Analyze (or ⌘/Ctrl+Enter), get the
document verdict card, per-sentence heatmap, and evidence list. Programmatic
clients hit the same API:

```bash
curl -X POST localhost:8390/analyze \
     -H 'Content-Type: application/json' \
     -d '{"text": "some text", "mode": "binoculars"}'
# → full Report JSON (doc verdict + per-sentence scores/signals)
curl localhost:8390/api/health   # model/device status
```

The server binds **127.0.0.1 by default** — this is a personal-analysis tool;
pass `--host` only if you deliberately want it reachable from your network.

### 3. Python

```python
from clarity import ModelPair, FastModel, analyze
report = analyze(open("essay.txt").read(), ModelPair())
print(report.doc_label, report.doc_score)
for s in report.sentences:
    if s.label == "ai":
        print(s.text, [g.detail for g in s.signals])

fast_report = analyze(text, FastModel())  # 1-model experimental mode
```

## Detection modes

| | `binoculars` (default) | `fast` |
| --- | --- | --- |
| Models loaded | 2 (~6 GB fp16) | 1 (~3 GB) |
| Method | Binoculars (paper-faithful) | sampling-free Fast-DetectGPT *approximation* |
| Measured quality (default weights) | ~40% detection @ 5% human FPR | ~10% detection @ 5% human FPR |
| Use when | always, unless RAM-constrained | memory-bound machines only |

Fast mode is **experimental**: our approximation needs no generation passes,
but that changes its operating characteristics vs the published method — its
scores are on their own scale and its measured separation is far weaker.
Both modes' thresholds come from the same quick calibration
([docs/CALIBRATION.md](docs/CALIBRATION.md)).

## Bring your own models

Any Hugging Face causal-LM pair that **shares a tokenizer** works (checked at
load). Bigger pairs are more accurate; the Falcon-7B pair from the paper is the
reference. **Thresholds are pair-specific** — if you swap pairs, recalibrate:
see [docs/CALIBRATION.md](docs/CALIBRATION.md).

## Honest limitations (read this)

- **No detector is proof.** Treat output as a lead to investigate, never a
  verdict on a person. This tool deliberately shows its evidence so a human
  can disagree with it.
- **The default model pair is convenience-grade.** Measured 2026-08-25
  ([calibration](docs/CALIBRATION.md)): at a 5% human false-positive rate the
  default Qwen2.5-1.5B pair detects only ~40% of AI documents. The Binoculars
  paper's >90%-at-0.01%-FPR is for the Falcon-7B pair — bring a bigger pair on
  GPU hardware for serious screening, and expect many "uncertain" labels on
  the default pair.
- False positives are real and **disproportionately hit non-native English
  writers** (documented across the literature, including for commercial tools).
- Paraphrasing/humanizer tools reduce detection rates for every known method.
- Short texts (< ~50 tokens) cannot be reliably scored; clarity says so
  rather than guessing.
- English-centric: sentence splitting and the idiom list are English-only for
  now; the Binoculars score itself is language-agnostic in principle but
  weaker off-English.

## Project docs

| Doc | What's in it |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture, research basis, every design decision + rationale |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | How thresholds were set, how to recalibrate for your model pair |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current status and planned phases (RAID eval, web UI, ensemble) |

## Development

```bash
uv pip install -e ".[dev]"
pytest            # math/segmentation/evidence tests; no model downloads needed
```

Docs policy: behavioral changes must update the relevant doc in the same
commit — the docs above are maintained as part of the code, not after it.

## License

MIT
