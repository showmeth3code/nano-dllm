#!/usr/bin/env python3
import torch
from nanovllm.config import Config

# Test the actual cache allocation
print("Testing actual KV cache allocation...")

# Load a sample config
config = Config(model="Qwen/Qwen2.5-0.5B")
hf_config = config.hf_config

print(f"Config attributes:")
print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', 'NOT FOUND')}")
print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'NOT FOUND')}")
print(f"  head_dim: {getattr(config, 'head_dim', 'NOT FOUND')}")
print(f"  kvcache_block_size: {config.kvcache_block_size}")
print(f"  num_kvcache_blocks: {config.num_kvcache_blocks}")

print(f"\nHF Config attributes:")
print(f"  num_hidden_layers: {getattr(hf_config, 'num_hidden_layers', 'NOT FOUND')}")
print(f"  num_key_value_heads: {getattr(hf_config, 'num_key_value_heads', 'NOT FOUND')}")
print(f"  head_dim: {getattr(hf_config, 'head_dim', 'NOT FOUND')}")

# Simulate the actual allocation
block_size = config.kvcache_block_size
num_kv_heads = getattr(config, 'num_key_value_heads', 1) // 1  # world_size = 1
print(f"\nSimulating cache allocation:")
print(f"  block_size: {block_size}")
print(f"  num_kv_heads: {num_kv_heads}")

kv_cache = torch.zeros(
    2,
    getattr(config, 'num_hidden_layers', 1),
    config.num_kvcache_blocks,
    block_size,
    num_kv_heads,
    getattr(config, 'head_dim', 1),
)

print(f"Full kv_cache shape: {kv_cache.shape}")
print(f"k_cache (layer 0) shape: {kv_cache[0, 0].shape}")
print(f"k_cache[block_idx=0, pos_in_block=0] shape: {kv_cache[0, 0, 0, 0].shape}")
print(f"k_cache[block_idx=0] shape: {kv_cache[0, 0, 0].shape}")

# Test specific access patterns
k_cache = kv_cache[0, 0]  # Shape should be (num_blocks, block_size, num_heads, head_dim)
print(f"\nAfter layer assignment:")
print(f"k_cache shape: {k_cache.shape}")
print(f"k_cache[0, 0] shape: {k_cache[0, 0].shape}")  # Should be (num_heads, head_dim)
