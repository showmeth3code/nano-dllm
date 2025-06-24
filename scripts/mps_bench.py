import os
import time
import numpy as np
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.mps_optimizations import apply_mps_optimizations

def main():
    try:
        print("Setting up MPS optimized benchmark...")
        
        # Apply MPS optimizations
        apply_mps_optimizations()
        
        # Set fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Test parameters
        num_seqs = 2
        seq_length = 80  # Fixed prompt length
        output_tokens = 50  # Number of tokens to generate
        
        # Model path
        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        print(f"Loading model from {path}...")
        
        # Create the model with MPS-friendly options
        llm = LLM(path, 
                 enforce_eager=True,  # Use eager mode for MPS compatibility
                 max_model_len=1024)  # Reduced for better performance
                 
        print("Model loaded successfully")
        
        # Create fixed test prompts
        prompt_token_ids = []
        for i in range(num_seqs):
            token_ids = np.random.randint(0, 10000, seq_length).tolist()
            prompt_token_ids.append(token_ids)
            
        # Set generation parameters
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens) 
            for _ in range(num_seqs)
        ]
        
        # Run warmup pass 
        print("Running warmup...")
        warmup_prompt = ["Hello, optimized MPS model!"]
        _ = llm.generate(warmup_prompt, SamplingParams(temperature=0.7, max_tokens=10))
        
        # Make sure MPS cache is clear
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        print("Starting benchmark run...")
        t = time.time()
        results = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        elapsed = time.time() - t
        
        # Compute statistics
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / elapsed
        
        print("\nBenchmark results:")
        print(f"Generated tokens: {total_tokens}")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        
        # Compare with README benchmark numbers
        readme_throughput = 1434.13  # tokens/s from README
        ratio = throughput / readme_throughput
        
        print("\nPerformance comparison:")
        print(f"This run: {throughput:.2f} tokens/sec")
        print(f"README benchmark (RTX 4070): {readme_throughput:.2f} tokens/sec")
        print(f"Ratio (this/README): {ratio:.4f}x")
        
        if ratio < 0.1:
            print("\nExpected performance difference: MPS is typically slower than CUDA for LLMs")
            print("The README benchmark was run on an RTX 4070 which has:")
            print("- Higher memory bandwidth")
            print("- More optimized CUDA kernels for attention")
            print("- Tensor cores for matrix operations")
        
        return throughput
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    main()
