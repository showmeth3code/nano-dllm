import os
import time
from random import randint, seed
import torch
import numpy as np
from nanovllm import LLM, SamplingParams

def optimize_for_mps():
    """Apply system-level optimizations for MPS performance"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Clear MPS cache to free memory
        torch.mps.empty_cache()
        
        # Set environment variables that can help MPS performance
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable fallback for unsupported ops
        
        # Disable debug and excessive logging
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable high watermark warnings
        
        # Use synchronous execution for more reliable results
        os.environ['PYTORCH_MPS_ENABLE_DISPATCH_CLEANUP_POLICY'] = '1'
        
        return True
    return False

def main():
    try:
        seed(0)
        # Apply MPS optimizations
        using_mps = optimize_for_mps()
        
        # Very small benchmark just to test performance optimizations
        num_seqs = 2
        max_input_len = 100
        
        # Local path or Hugging Face path
        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        print(f"Loading model from {path}...")
        
        # Special options for MPS
        enforce_eager = True  # Using eager mode for MPS compatibility
        
        # Create LLM with optimized settings
        llm = LLM(path, 
                 enforce_eager=enforce_eager, 
                 max_model_len=1024)  # Reduce context size for faster inference
        
        print("Model loaded successfully")
        print(f"Running optimized test on {'MPS' if using_mps else 'CPU'} device...")
        
        # Create fixed input sequences for consistent benchmarking
        # Using fixed lengths instead of random to get more stable measurements
        prompt_token_ids = []
        for i in range(num_seqs):
            # Set a fixed seed for each sequence for reproducibility
            np.random.seed(i)
            prompt_token_ids.append([int(x) for x in np.random.randint(0, 10000, 80)])
        
        # Setting block_size values that could trigger the assertion error
        # but also keeping generation length smaller for faster testing
        sampling_params = []
        sampling_params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=50))
        sampling_params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=50))
        
        # Warmup run - important for MPS especially
        print("Running warmup...")
        warmup_prompt = ["Warm up the model: "]
        warmup_result = llm.generate(warmup_prompt, SamplingParams(temperature=0.7, max_tokens=10))
        
        # Run multiple trials to average performance
        num_trials = 3
        total_time = 0
        
        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}:")
            
            # Clear MPS cache between runs
            if using_mps:
                torch.mps.empty_cache()
                
            # Run with timing
            t = time.time()
            results = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
            elapsed = time.time() - t
            
            print(f"  Generated text lengths: {[len(result['text']) for result in results]}")
            print(f"  Time: {elapsed:.2f}s")
            
            total_time += elapsed
            
        avg_time = total_time / num_trials
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / avg_time
        
        print("\nTest results:")
        print(f"Average time: {avg_time:.2f}s for {total_tokens} tokens")
        print(f"Throughput: {throughput:.2f} tokens/s")
        
        if using_mps:
            print("\nMPS Performance Tips:")
            print("1. Consider using smaller batch sizes")
            print("2. Reduce context length when possible")
            print("3. For production, you may get better performance on CUDA devices")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
