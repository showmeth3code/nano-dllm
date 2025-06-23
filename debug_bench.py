import os
import time
import traceback
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    try:
        seed(0)
        num_seqs = 4  # Reduced from 256 to make debugging easier
        max_input_len = 1024
        max_ouput_len = 1024

        path = os.path.expanduser("Qwen/Qwen3-0.6B")
        llm = LLM(path, enforce_eager=False, max_model_len=4096)

        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
        # uncomment the following line for vllm
        # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

        print("Starting benchmark warmup...")
        llm.generate(["Benchmark: "], SamplingParams())
        print("Starting main benchmark run...")
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
