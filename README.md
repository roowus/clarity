# clairity

**Evidence-first, open-source AI-text detection.** Paste text, get a calibrated
score, and — unlike black-box percentage detectors — see *which sentences* look
machine-generated and *why*, with every reason backed by a measurable statistic.

```
$ clairity essay.txt
╭──────────────────────────── clairity ────────────────────────────╮
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

clairity implements **[Binoculars](https://arxiv.org/abs/2401.12070)** (Hans et
al., ICML 2024), a zero-shot detection method: two closely related language
models (a base model and its instruct fine-tune) both score the text, and the
ratio of perplexity to cross-perplexity separates human from machine writing.
No detection-specific training; in the paper this detects >90% of
ChatGPT-generated text at a 0.01% false-positive rate.

On top of the document score, clairity re-aggregates the same per-token
log-probabilities per sentence (with neighbor-blending for short sentences,
which are individually too noisy to score) and attaches **evidence signals**:
low Binoculars score, low local burstiness, AI-idiom phrases, short-span
warnings. See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture and
the reasoning behind every choice.

## Install

```bash
git clone https://github.com/roowus/clairity && cd clairity
uv venv && uv pip install -e .
```

First run downloads the default model pair (Qwen2.5-1.5B base + instruct,
~6 GB) from Hugging Face. Runs on CUDA, Apple Silicon (MPS), or CPU.

## Usage

```bash
clairity examples/ai-slop.txt    # rich terminal report (try both examples/)
echo "some text" | clairity -    # stdin
clairity essay.txt --json        # machine-readable full report
clairity essay.txt --observer meta-llama/Llama-3.2-3B-Instruct \
                    --performer meta-llama/Llama-3.2-3B   # bring your own pair
```

Python API:

```python
from clairity import ModelPair, analyze
report = analyze(open("essay.txt").read(), ModelPair())
print(report.doc_label, report.doc_score)
for s in report.sentences:
    if s.label == "ai":
        print(s.text, [g.detail for g in s.signals])
```

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
- Short texts (< ~50 tokens) cannot be reliably scored; clairity says so
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
