import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, logits_k=0):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        tokens = torch.where(temperatures == 0, greedy_tokens, sample_tokens)
        # logits_k > 0: return top-k logits for distillation, logits_k < 0: return random-k
        if logits_k == 0: return tokens
        indices = torch.multinomial(probs, num_samples=abs(logits_k), replacement=True) if logits_k < 0 else torch.topk(logits, logits_k)[1]
        return tokens, logits.gather(-1, indices), indices
