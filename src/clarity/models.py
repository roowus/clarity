"""Model loading and inference for both detection modes.

- mode "binoculars": two tokenizer-compatible models (observer + performer),
  the paper-standard method.
- mode "fast": ONE model, Fast-DetectGPT-style approximation (see
  fast_detect.py for what is and isn't claimed about it). Cheaper: half the
  memory, half the compute.

BYO-model: any HF causal-LM. For binoculars the pair must share a tokenizer
(checked at load time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fast_detect import fast_detect_scores
from .scoring import TokenScores, token_scores

# Default pair: small enough for a laptop, base+instruct siblings as the
# Binoculars paper prescribes. Users can override via CLI/config.
DEFAULT_OBSERVER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-1.5B"
DEFAULT_FAST_MODEL = DEFAULT_PERFORMER  # base model scores raw text best

MAX_TOKENS = 4096  # scored per document; longer texts are truncated with a warning


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(name: str, device: str):
    dtype = torch.float16 if device != "cpu" else torch.float32
    return (
        AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        .to(device)
        .eval()
    )


@dataclass
class ModelPair:
    """Binoculars scorer (two models) — the default, paper-faithful mode."""

    observer_name: str = DEFAULT_OBSERVER
    performer_name: str = DEFAULT_PERFORMER
    device: str = field(default_factory=pick_device)
    mode: str = "binoculars"  # informational; server/CLI pick the class

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        perf_tok = AutoTokenizer.from_pretrained(self.performer_name)
        if self.tokenizer.get_vocab() != perf_tok.get_vocab():
            raise ValueError(
                f"{self.observer_name} and {self.performer_name} do not share a "
                "vocabulary; Binoculars requires a tokenizer-compatible pair "
                "(e.g. a base model and its instruct fine-tune)."
            )
        self.observer = _load_model(self.observer_name, self.device)
        self.performer = _load_model(self.performer_name, self.device)

    @torch.inference_mode()
    def score_text(self, text: str, progress=None) -> tuple[TokenScores, bool]:
        """Run both models over the text once. Returns (scores, truncated).

        `progress(pct, stage)` is called between stages when provided:
        observer pass maps to 10-40%, performer pass to 40-70%.
        """
        enc = self._encode(text)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)

        def _pct(p, s):
            if progress is not None:
                progress(p, s)

        _pct(10, "scoring with observer model")
        obs_logits = self.observer(input_ids, attention_mask=attn).logits[0]
        _pct(40, "scoring with performer model")
        perf_logits = self.performer(input_ids, attention_mask=attn).logits[0]
        _pct(70, "combining model outputs")
        offsets = [tuple(o) for o in enc.offset_mapping[0].tolist()]
        scores = token_scores(
            obs_logits.cpu(), perf_logits.cpu(), input_ids[0].cpu(), offsets
        )
        return scores, enc.truncated

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        enc.truncated = enc.input_ids.shape[1] >= MAX_TOKENS
        return enc


@dataclass
class FastModel:
    """Single-model Fast-DetectGPT-approximation scorer (mode="fast")."""

    model_name: str = DEFAULT_FAST_MODEL
    device: str = field(default_factory=pick_device)
    mode: str = "fast"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = _load_model(self.model_name, self.device)

    @torch.inference_mode()
    def score_text(self, text: str, progress=None) -> tuple[TokenScores, bool]:
        """One forward pass. Returns TokenScores where log_xppl carries the
        fast-mode DENOMINATOR (so binoculars_score's ratio shape still works:
        mean(log_ppl)/mean(denominator)). Progress maps the pass to 10-70%."""
        enc = self._encode(text)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        if progress is not None:
            progress(10, "scoring with model")
        logits = self.model(input_ids, attention_mask=attn).logits[0]
        if progress is not None:
            progress(70, "combining outputs")
        offsets = [tuple(o) for o in enc.offset_mapping[0].tolist()]
        num, den = fast_detect_scores(logits, input_ids[0])
        scores = TokenScores(
            log_ppl=num,
            log_xppl=den,
            token_ids=input_ids[0, 1:].cpu(),
            offsets=offsets[1:],
        )
        return scores, enc.truncated

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        enc.truncated = enc.input_ids.shape[1] >= MAX_TOKENS
        return enc
