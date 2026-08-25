"""Model pair loading and inference. BYO-model: any HF causal-LM pair works,
as long as both models share a tokenizer (checked at load time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scoring import TokenScores, token_scores

# Default pair: small enough for a laptop, base+instruct siblings as the
# Binoculars paper prescribes. Users can override via CLI/config.
DEFAULT_OBSERVER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-1.5B"

MAX_TOKENS = 4096  # scored per document; longer texts are truncated with a warning


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelPair:
    observer_name: str = DEFAULT_OBSERVER
    performer_name: str = DEFAULT_PERFORMER
    device: str = field(default_factory=pick_device)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        perf_tok = AutoTokenizer.from_pretrained(self.performer_name)
        if self.tokenizer.get_vocab() != perf_tok.get_vocab():
            raise ValueError(
                f"{self.observer_name} and {self.performer_name} do not share a "
                "vocabulary; Binoculars requires a tokenizer-compatible pair "
                "(e.g. a base model and its instruct fine-tune)."
            )
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name, torch_dtype=dtype
        ).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name, torch_dtype=dtype
        ).to(self.device).eval()

    @torch.inference_mode()
    def score_text(self, text: str) -> tuple[TokenScores, bool]:
        """Run both models over the text once. Returns (scores, truncated)."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        truncated = enc.input_ids.shape[1] >= MAX_TOKENS
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        obs_logits = self.observer(input_ids, attention_mask=attn).logits[0]
        perf_logits = self.performer(input_ids, attention_mask=attn).logits[0]
        offsets = [tuple(o) for o in enc.offset_mapping[0].tolist()]
        scores = token_scores(
            obs_logits.cpu(), perf_logits.cpu(), input_ids[0].cpu(), offsets
        )
        return scores, truncated
