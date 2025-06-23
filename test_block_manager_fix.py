import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    try:
        seed(0)
        # Very small benchmark just to test the block_manager fix
        num_seqs = 2
        max_input_len = 100
        max_output_len = 20

        # Local path or Hugging Face path
        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        print(f"Loading model from {path}...")
        llm = LLM(path, enforce_eager=False, max_model_len=4096)
        print("Model loaded successfully")

        print(f"Running validation test with {num_seqs} sequences...")
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(50, max_input_len))] for _ in range(num_seqs)]
        
        # Setting block_size values that could trigger the assertion error
        sampling_params = []
        
        # First sequence: exactly block_size tokens (e.g., 256) to test block hash computation
        sampling_params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256))
        
        # Second sequence: block_size+1 tokens to test new block allocation
        sampling_params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=257))
        
        print("Starting test run...")
        t = time.time()
        _ = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        elapsed = time.time() - t
        
        print("✅ Test completed successfully!")
        print("The block_manager fix works - no assertion errors occurred.")
        print(f"Execution time: {elapsed:.2f}s")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        print("The fix may not be working properly.")

if __name__ == "__main__":
    main()
