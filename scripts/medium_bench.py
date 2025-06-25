import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    try:
        seed(0)
        # Medium-sized benchmark 
        num_seqs = 16
        max_input_len = 256
        max_output_len = 64

        # Local path or Hugging Face path
        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        print(f"Loading model from {path}...")
        llm = LLM(path, enforce_eager=False, max_model_len=4096)
        print("Model loaded successfully")

        print(f"Generating {num_seqs} random sequences...")
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(50, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(32, max_output_len)) for _ in range(num_seqs)]

        print("Running warmup...")
        result = llm.generate(["Benchmark: "], SamplingParams())
        print(f"Warmup output: {result[0]['text'][:30]}...")
        
        print("Starting benchmark run...")
        t = time.time()
        _ = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        elapsed = time.time() - t
        
        print("Generation completed successfully")
        
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / elapsed
        print(f"Total: {total_tokens} tokens, Time: {elapsed:.2f}s, Throughput: {throughput:.2f} tokens/s")
        
        # Get values from the README
        readme_tokens = 133966
        readme_time = 93.41
        readme_throughput = 1434.13
        
        # Compare our results with the README
        print("\nComparison with README benchmark results:")
        print(f"README: {readme_tokens} tokens, {readme_time:.2f}s, {readme_throughput:.2f} tokens/s")
        print(f"Current: {total_tokens} tokens, {elapsed:.2f}s, {throughput:.2f} tokens/s")
        print(f"Ratio: {throughput/readme_throughput:.2f}x the throughput in README")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
