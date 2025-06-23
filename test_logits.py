#!/usr/bin/env python3

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.sampler import Sampler

# Direct sampler test
sampler = Sampler()
logits = torch.randn(2, 1000) * 2.0
temperatures = torch.tensor([0.8, 1.0])

# Test standard sampling
result = sampler(logits, temperatures, logits_k=0)
assert isinstance(result, torch.Tensor)
assert result.shape == (2,)

# Test top-k
result = sampler(logits, temperatures, logits_k=5)
assert isinstance(result, tuple)
tokens, k_logits, indices = result
assert k_logits.shape == (2, 5)
assert indices.shape == (2, 5)

# Test random-k
result = sampler(logits, temperatures, logits_k=-8)
assert isinstance(result, tuple)
tokens, k_logits, indices = result
assert k_logits.shape == (2, 8)
assert indices.shape == (2, 8)

# API test
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
prompts = ["The capital of France is", "Machine learning is"]

# Standard generation
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=10, logits_k=0))
assert 'logits' not in outputs[0]

# Top-k extraction
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=10, logits_k=5))
assert 'logits' in outputs[0]
assert len(outputs[0]['logits'][0]) == 5

# Random-k extraction
outputs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=10, logits_k=-8))
assert 'logits' in outputs[0]
assert len(outputs[0]['logits'][0]) == 8

print("All tests passed")
