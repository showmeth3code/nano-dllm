import os
import sys
import time
from random import randint, seed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 1
    max_input_len = 1024
    max_ouput_len = 256

    path = os.path.expanduser("/home/tonywei/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [ "List all prime numbers within 100?" ] * num_seqs
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
