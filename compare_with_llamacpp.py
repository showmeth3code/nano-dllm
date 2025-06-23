#!/usr/bin/env python
"""
Benchmark comparison between nano-vllm and llama.cpp on the same hardware.
This script specifically profiles different operations to identify bottlenecks.
"""

import time
import torch
import argparse

from nanovllm import LLM
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

# Check if we're on Apple Silicon
is_mps_available = torch.backends.mps.is_available()
device = torch.device("mps" if is_mps_available else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def profile_block_allocation(iterations=1000, block_size=256):
    """Profile block allocation and deallocation performance"""
    print("\n--- Profiling Block Allocation ---")
    block_manager = BlockManager(num_blocks=1000, block_size=block_size)
    
    # Create a test sequence
    tokens = list(range(1024))
    params = SamplingParams(temperature=0.8, max_tokens=100)
    seq = Sequence(tokens, params, "Test prompt")
    
    # Measure allocation time
    start = time.time()
    for _ in range(iterations):
        block_manager.allocate(seq)
        block_manager.deallocate(seq)
    end = time.time()
    
    print(f"Block allocation/deallocation time: {(end-start)*1000/iterations:.3f} ms per iteration")
    return (end-start)/iterations

def profile_attention_operation(batch_size=1, seq_len=1024, hidden_dim=4096, iterations=100):
    """Profile the attention operation performance"""
    print("\n--- Profiling Attention Operation ---")
    
    # Create random query, key, value tensors
    # [batch_size, seq_len, hidden_dim]
    q = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    k = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    v = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Create causal mask
    # [seq_len, seq_len]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
    
    # Warm-up
    for _ in range(5):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        _ = torch.matmul(attn_weights, v)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(iterations):
        # [batch_size, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        _ = torch.matmul(attn_weights, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    print(f"Attention operation time: {(end-start)*1000/iterations:.3f} ms per iteration")
    return (end-start)/iterations

def profile_model_load_time(model_id="Qwen/Qwen1.5-0.5B"):
    """Profile model loading time"""
    print("\n--- Profiling Model Load Time ---")
    start = time.time()
    try:
        _ = LLM(model_id)
        success = True
    except Exception as e:
        print(f"Error loading model: {e}")
        success = False
    end = time.time()
    print(f"Model load time: {(end-start)*1000:.3f} ms")
    return end-start

def profile_token_generation(model_id="Qwen/Qwen1.5-0.5B", prompt="Hello, my name is", tokens=50):
    """Profile token generation speed"""
    print("\n--- Profiling Token Generation ---")
    tokens_per_second = 0
    output_text = ""
    
    try:
        llm = LLM(model_id)
        params = SamplingParams(temperature=0.8, max_tokens=tokens)
        
        # Warmup
        _ = llm.generate(prompt, sampling_params=params)
        
        # Benchmark
        start = time.time()
        output = llm.generate(prompt, sampling_params=params)
        end = time.time()
        
        # Process output based on the structure of your LLM's response
        if isinstance(output, list) and output:
            result = output[0]
            
            # Try different output formats
            if hasattr(result, 'text'):
                output_text = result.text
                tokens_generated = tokens  # Approximate if not available
            elif hasattr(result, 'generated_tokens'):
                output_text = prompt + ' ' + ' '.join(map(str, result.generated_tokens))
                tokens_generated = len(result.generated_tokens)
            elif isinstance(result, dict):
                output_text = result.get('text', prompt + ' [generated text]')
                tokens_generated = len(result.get('generated_token_ids', [])) or tokens
            else:
                output_text = str(result)[:50] + "..."
                tokens_generated = tokens  # Fallback
                
            time_taken = end - start
            tokens_per_second = tokens_generated / time_taken
    except Exception as e:
        print(f"Error in token generation: {e}")
        time_taken = 0
        tokens_per_second = 0
        tokens_generated = 0
    
    print(f"Token generation time: {time_taken*1000:.3f} ms")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Generated sample: {output_text[:50]}...")
    
    return tokens_per_second

def run_benchmarks():
    """Run all benchmarks"""
    results = {}
    
    # System info
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Run benchmarks
    results["block_alloc"] = profile_block_allocation()
    results["attention"] = profile_attention_operation()
    results["model_load"] = profile_model_load_time()
    results["token_gen"] = profile_token_generation()
    
    # Summary
    print("\n--- Benchmark Summary ---")
    for name, value in results.items():
        print(f"{name}: {value*1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm operations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B", 
                        help="Model ID to use for benchmarking")
    args = parser.parse_args()
    
    print(f"Benchmarking with model: {args.model}")
    run_benchmarks()
