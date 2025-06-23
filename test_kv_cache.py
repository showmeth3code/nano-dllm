#!/usr/bin/env python3
import torch

# Test KV cache access patterns
print("Testing KV cache tensor access patterns...")

# Simulate the cache structure
block_size = 256
num_heads = 16
head_dim = 64

# Create a test cache tensor
k_cache = torch.zeros(512, block_size, num_heads, head_dim)  # (blocks, block_size, heads, head_dim)

print(f"k_cache shape: {k_cache.shape}")
print(f"k_cache[0, 0] shape: {k_cache[0, 0].shape}")  # Should be (num_heads, head_dim)
print(f"k_cache[0, 0, :, :] shape: {k_cache[0, 0, :, :].shape}")  # Should be (num_heads, head_dim)

# Test the sequence cache tensor that should receive the data
seq_len = 20
k_for_seq = torch.zeros(seq_len, num_heads, head_dim)
print(f"k_for_seq shape: {k_for_seq.shape}")
print(f"k_for_seq[0, :, :] shape: {k_for_seq[0, :, :].shape}")  # Should be (num_heads, head_dim)

# Test the assignment - this is what should work
try:
    k_for_seq[0, :, :] = k_cache[0, 0, :, :]
    print("SUCCESS: k_for_seq[0, :, :] = k_cache[0, 0, :, :] works")
except Exception as e:
    print(f"FAILED: k_for_seq[0, :, :] = k_cache[0, 0, :, :] - {e}")

# Test what was failing
try:
    k_for_seq[0, :, :] = k_cache[0, 0]
    print("SUCCESS: k_for_seq[0, :, :] = k_cache[0, 0] works")
except Exception as e:
    print(f"FAILED: k_for_seq[0, :, :] = k_cache[0, 0] - {e}")
