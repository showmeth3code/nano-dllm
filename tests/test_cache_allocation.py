#!/usr/bin/env python3
import torch
from nanovllm.config import Config



def simulate_kv_cache_allocation():
    print("Testing actual KV cache allocation...")
    config = Config(model_path="Qwen/Qwen2.5-0.5B")
    hf_config = config.hf_config

    print("Config attributes:")
    print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', 'NOT FOUND')}")
    print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'NOT FOUND')}")
    print(f"  head_dim: {getattr(config, 'head_dim', 'NOT FOUND')}")
    print(f"  kvcache_block_size: {config.kvcache_block_size}")
    print(f"  num_kvcache_blocks: {config.num_kvcache_blocks}")

    print("\nHF Config attributes:")
    print(f"  num_hidden_layers: {getattr(hf_config, 'num_hidden_layers', 'NOT FOUND')}")
    print(f"  num_key_value_heads: {getattr(hf_config, 'num_key_value_heads', 'NOT FOUND')}")
    print(f"  head_dim: {getattr(hf_config, 'head_dim', 'NOT FOUND')}")

    # Simulate the actual allocation
    block_size = config.kvcache_block_size
    num_kv_heads = getattr(config, 'num_key_value_heads', 1) // 1  # world_size = 1
    print("\nSimulating cache allocation:")
    print(f"  block_size: {block_size}")
    print(f"  num_kv_heads: {num_kv_heads}")

    # Reduce allocation size for test discovery
    test_block_size = min(block_size, 8)
    test_num_blocks = min(config.num_kvcache_blocks, 2)
    test_num_layers = min(getattr(config, 'num_hidden_layers', 1), 2)
    test_num_heads = min(num_kv_heads, 2)
    test_head_dim = min(getattr(config, 'head_dim', 1), 8)

    kv_cache = torch.zeros(
        2,
        test_num_layers,
        test_num_blocks,
        test_block_size,
        test_num_heads,
        test_head_dim,
    )

    print(f"Full kv_cache shape: {kv_cache.shape}")
    print(f"k_cache (layer 0) shape: {kv_cache[0, 0].shape}")
    print(f"k_cache[block_idx=0, pos_in_block=0] shape: {kv_cache[0, 0, 0, 0].shape}")
    print(f"k_cache[block_idx=0] shape: {kv_cache[0, 0, 0].shape}")

    # Test specific access patterns
    k_cache = kv_cache[0, 0]  # Shape should be (num_blocks, block_size, num_heads, head_dim)
    print("\nAfter layer assignment:")
    print(f"k_cache shape: {k_cache.shape}")
    print(f"k_cache[0, 0] shape: {k_cache[0, 0].shape}")  # Should be (num_heads, head_dim)


def test_kv_cache_allocation():
    """Pytest-compatible test for kv cache allocation (smoke test, small size)."""
    simulate_kv_cache_allocation()


if __name__ == "__main__":
    simulate_kv_cache_allocation()
