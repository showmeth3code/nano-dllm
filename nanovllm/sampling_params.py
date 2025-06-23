from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    logits_k: int = 0  # >0: top-k logits, <0: random-k logits, 0: no logits
