#!/usr/bin/env python3
"""
Manual test to validate logits extraction feature.
Run this after implementing the changes to verify functionality.
"""

from nanovllm import LLM, SamplingParams

# Test prompts
prompts = [
    "The capital of France is",
    "Machine learning is"
]

# Initialize model (adjust model path as needed)
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

print("Testing Logits Extraction Feature")
print("=" * 50)

# Test 1: Standard generation (logits_k=0)
print("\nTest 1: Standard generation (logits_k=0)")
print("-" * 40)
params = SamplingParams(temperature=0.8, max_tokens=10, logits_k=0)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    print(f"Generated text: {output}")
    print()

# Test 2: Top-K logits extraction (logits_k=5)
print("\nTest 2: Top-K logits extraction (logits_k=5)")
print("-" * 40)
params = SamplingParams(temperature=0.8, max_tokens=10, logits_k=5)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"Keys: {output.keys()}")
        print(f"Tokens shape: {len(output['tokens'])}")
        print(f"Logits shape: {len(output['logits'])}, {len(output['logits'][0]) if output['logits'] else 0}")
        print(f"Indices shape: {len(output['indices'])}, {len(output['indices'][0]) if output['indices'] else 0}")
        print(f"First few tokens: {output['tokens'][:3]}")
    print()

# Test 3: Random-K logits extraction (logits_k=-8)
print("\nTest 3: Random-K logits extraction (logits_k=-8)")
print("-" * 40)
params = SamplingParams(temperature=1.0, max_tokens=10, logits_k=-8)
outputs = llm.generate(prompts, params)
for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"Keys: {output.keys()}")
        print(f"Tokens shape: {len(output['tokens'])}")
        print(f"Logits shape: {len(output['logits'])}, {len(output['logits'][0]) if output['logits'] else 0}")
        print(f"Indices shape: {len(output['indices'])}, {len(output['indices'][0]) if output['indices'] else 0}")
        print(f"First few tokens: {output['tokens'][:3]}")
        # Check that indices are not necessarily sorted (random sampling)
        if output['indices'] and len(output['indices'][0]) > 1:
            first_indices = output['indices'][0]
            print(f"First token indices (should vary): {first_indices}")
    print()

# Test 4: Verify shapes are consistent
print("\nTest 4: Verify shapes are consistent between topk and randomk")
print("-" * 40)
k_value = 10
for k in [k_value, -k_value]:
    params = SamplingParams(temperature=0.8, max_tokens=5, logits_k=k)
    outputs = llm.generate(prompts[:1], params)
    output = outputs[0]
    if isinstance(output, dict):
        print(f"logits_k={k}:")
        print(f"  Logits shape: {len(output['logits'])}, {len(output['logits'][0])}")
        print(f"  Indices shape: {len(output['indices'])}, {len(output['indices'][0])}")

print("\n" + "=" * 50)
print("Test completed!")
print("\nExpected behavior:")
print("- logits_k=0: Returns list of token IDs")
print("- logits_k>0: Returns dict with top-k logits")
print("- logits_k<0: Returns dict with random-k logits")
print("- Both topk and randomk should have same shapes")
