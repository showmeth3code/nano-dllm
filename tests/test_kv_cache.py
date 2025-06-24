#!/usr/bin/env python3
import torch


def simulate_kv_cache_access():
    print("Testing KV cache tensor access patterns...")

    # Use much smaller sizes for test discovery
    block_size = 4
    num_heads = 2
    head_dim = 2

    # Create a test cache tensor
    k_cache = torch.zeros(8, block_size, num_heads, head_dim)  # (blocks, block_size, heads, head_dim)

    print(f"k_cache shape: {k_cache.shape}")
    print(f"k_cache[0, 0] shape: {k_cache[0, 0].shape}")  # Should be (num_heads, head_dim)
    print(f"k_cache[0, 0, :, :] shape: {k_cache[0, 0, :, :].shape}")  # Should be (num_heads, head_dim)

    # Test the sequence cache tensor that should receive the data
    seq_len = 2
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


def test_kv_cache_access():
    """Pytest-compatible test for kv cache access (smoke test, small size)."""
    def _import_kv_cache_deps():
        import torch
        return torch

    torch = _import_kv_cache_deps()
    simulate_kv_cache_access()


if __name__ == "__main__":
    simulate_kv_cache_access()
