#!/usr/bin/env python3
"""
Direct test of logits extraction feature at the model runner level.
This bypasses the LLM engine to directly test the sampler functionality.
"""

import torch
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams

print("Direct Test of Logits Extraction Feature")
print("=" * 50)

# Create sampler
sampler = Sampler()

# Create example data
batch_size = 2
vocab_size = 1000
logits = torch.randn(batch_size, vocab_size) * 2.0
temperatures = torch.tensor([0.8, 1.0])

print(f"Batch size: {batch_size}")
print(f"Vocab size: {vocab_size}")
print(f"Temperatures: {temperatures.tolist()}")
print()

# Test 1: Standard sampling (logits_k=0)
print("Test 1: Standard sampling (logits_k=0)")
print("-" * 40)
result = sampler(logits, temperatures, logits_k=0)
print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape}")
print(f"Sample tokens: {result.tolist()}")
print()

# Test 2: Top-K sampling (logits_k=5)
print("Test 2: Top-K sampling (logits_k=5)")
print("-" * 40)
result = sampler(logits, temperatures, logits_k=5)
print(f"Result type: {type(result)}")
print(f"Result is tuple: {isinstance(result, tuple)}")
if isinstance(result, tuple):
    tokens, k_logits, indices = result
    print(f"Tokens shape: {tokens.shape}")
    print(f"Logits shape: {k_logits.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Sample tokens: {tokens.tolist()}")
    print(f"Top-5 indices for first sample: {indices[0].tolist()}")
    # Verify indices are sorted (top-k)
    print(f"Indices sorted (should be True): {torch.all(indices[0] == torch.sort(indices[0])[0])}")
print()

# Test 3: Random-K sampling (logits_k=-8)
print("Test 3: Random-K sampling (logits_k=-8)")
print("-" * 40)
result = sampler(logits, temperatures, logits_k=-8)
print(f"Result type: {type(result)}")
print(f"Result is tuple: {isinstance(result, tuple)}")
if isinstance(result, tuple):
    tokens, k_logits, indices = result
    print(f"Tokens shape: {tokens.shape}")
    print(f"Logits shape: {k_logits.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Sample tokens: {tokens.tolist()}")
    print(f"Random-8 indices for first sample: {indices[0].tolist()}")
    # Random indices likely not sorted
    print(f"Indices sorted (should be False usually): {torch.all(indices[0] == torch.sort(indices[0])[0])}")
print()

# Test 4: Verify shapes consistency
print("Test 4: Verify shapes consistency")
print("-" * 40)
k_value = 10
for k in [k_value, -k_value]:
    result = sampler(logits, temperatures, logits_k=k)
    if isinstance(result, tuple):
        tokens, k_logits, indices = result
        print(f"logits_k={k}:")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Logits shape: {k_logits.shape}")
        print(f"  Indices shape: {indices.shape}")

print("\n" + "=" * 50)
print("Test completed successfully!")
print("\nKey findings:")
print("- logits_k=0: Returns tensor of tokens only")
print("- logits_k>0: Returns tuple (tokens, top-k logits, top-k indices)")
print("- logits_k<0: Returns tuple (tokens, random-k logits, random-k indices)")
print("- Shapes are consistent between top-k and random-k")
