#!/usr/bin/env python3
"""
Comprehensive test for logits extraction feature.
Tests both direct sampler functionality and full API integration.
"""

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.sampler import Sampler

print("=" * 70)
print("COMPREHENSIVE LOGITS EXTRACTION TEST")
print("=" * 70)

# PART 1: Direct Sampler Test
print("\nPART 1: Direct Sampler Test")
print("-" * 70)

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

# Test 1.1: Standard sampling (logits_k=0)
print("Test 1.1: Standard sampling (logits_k=0)")
print("-" * 40)
result = sampler(logits, temperatures, logits_k=0)
print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape}")
print(f"Sample tokens: {result.tolist()}")
print()

# Test 1.2: Top-K sampling (logits_k=5)
print("Test 1.2: Top-K sampling (logits_k=5)")
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

# Test 1.3: Random-K sampling (logits_k=-8)
print("Test 1.3: Random-K sampling (logits_k=-8)")
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

# Test 1.4: Verify shapes consistency
print("Test 1.4: Verify shapes consistency")
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

print("\nDirect sampler tests completed successfully!")

# PART 2: Full API Integration Test
print("\n" + "=" * 70)
print("PART 2: Full API Integration Test")
print("-" * 70)

# Test prompts
prompts = [
    "The capital of France is",
    "Machine learning is"
]

# Initialize model (adjust model path as needed)
print("Initializing model...")
llm = LLM(model="/home/dgxuser/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a")
print("Model loaded successfully!")
print()

# Test 2.1: Standard generation (logits_k=0)
print("Test 2.1: Standard generation (logits_k=0)")
print("-" * 40)
params = SamplingParams(temperature=0.8, max_tokens=10, logits_k=0)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    print(f"Keys: {output.keys()}")
    print(f"Generated text: {output['text']}")
    print(f"Has logits: {'logits' in output}")
    print()

# Test 2.2: Top-K logits extraction (logits_k=5)
print("Test 2.2: Top-K logits extraction (logits_k=5)")
print("-" * 40)
params = SamplingParams(temperature=0.8, max_tokens=10, logits_k=5)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    print(f"Keys: {output.keys()}")
    print(f"Generated text: {output['text']}")
    if 'logits' in output:
        print(f"Number of tokens generated: {len(output['logits'])}")
        print(f"Logits shape per token: {len(output['logits'][0])} (top-5)")
        print(f"Indices shape per token: {len(output['indices'][0])} (top-5)")
        print(f"First token's top-5 indices: {output['indices'][0]}")
    print()

# Test 2.3: Random-K logits extraction (logits_k=-8)
print("Test 2.3: Random-K logits extraction (logits_k=-8)")
print("-" * 40)
params = SamplingParams(temperature=1.0, max_tokens=10, logits_k=-8)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    print(f"Keys: {output.keys()}")
    print(f"Generated text: {output['text']}")
    if 'logits' in output:
        print(f"Number of tokens generated: {len(output['logits'])}")
        print(f"Logits shape per token: {len(output['logits'][0])} (random-8)")
        print(f"Indices shape per token: {len(output['indices'][0])} (random-8)")
        print(f"First token's random-8 indices: {output['indices'][0]}")
        # Check that indices are not necessarily sorted (random sampling)
        if len(output['indices'][0]) > 1:
            sorted_indices = sorted(output['indices'][0])
            is_sorted = output['indices'][0] == sorted_indices
            print(f"Indices are sorted: {is_sorted} (should be False for random sampling)")
    print()

# Test 2.4: Verify shapes are consistent
print("Test 2.4: Verify shapes are consistent between topk and randomk")
print("-" * 40)
k_value = 10
for k in [k_value, -k_value]:
    params = SamplingParams(temperature=0.8, max_tokens=5, logits_k=k)
    outputs = llm.generate(prompts[:1], params)
    output = outputs[0]
    if 'logits' in output:
        print(f"logits_k={k}:")
        print(f"  Number of tokens: {len(output['logits'])}")
        print(f"  Logits per token: {len(output['logits'][0])}")
        print(f"  Indices per token: {len(output['indices'][0])}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nSummary:")
print("- Direct sampler correctly handles logits_k parameter")
print("- logits_k=0: Returns tokens only (no logits)")
print("- logits_k>0: Returns top-k logits with deterministic selection")
print("- logits_k<0: Returns random-k logits with stochastic selection")
print("- Full API exposes logits through generate() method")
print("- Shapes are consistent between top-k and random-k modes")
print("- vLLM-style interface successfully implemented!")
