#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import argparse
from transformers import AutoTokenizer

# 进度条
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        # 极简兜底：不依赖 tqdm 也能跑
        class _Dummy:
            def __init__(self, total=None): self.n=0; self.total=total
            def update(self, n=1): self.n += n
            def close(self): pass
        return _Dummy(total=total)

# 按需调整导入路径，确保能找到你的 nanovllm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from nanovllm import LLM, SamplingParams


def load_sharegpt_prompts(path, limit=1000):
    """读取 ShareGPT 的 JSON/JSONL，返回最多 limit 条 prompt 文本。"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        items = json.load(f) if head.lstrip().startswith("[") else (json.loads(l) for l in f if l.strip())
        for item in items:
            print(item)
            conv = item.get("conversations") or item.get("conversation") or []
            parts = [turn.get("value", "") for turn in conv if turn.get("value")]
            if not parts and "text" in item:
                parts = [item["text"]]
            if parts:
                texts.append("\n".join(parts))
            if len(texts) >= limit:
                break
    return texts


def parse_args():
    p = argparse.ArgumentParser(description="Batch-by-batch ShareGPT benchmark (nanoVLLM) with tqdm")
    p.add_argument("--sharegpt-file", type=str, default=os.path.expanduser("~/nano-dllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"))
    p.add_argument("--model-path", type=str, default=os.path.expanduser("/home/tonywei/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"))
    p.add_argument("--total-requests", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1, help="每批发送的请求数 n")
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.sharegpt_file):
        raise FileNotFoundError(f"ShareGPT file not found: {args.sharegpt_file}")

    # 1) 读取 prompts（最多 total-requests 条）
    prompts = load_sharegpt_prompts(args.sharegpt_file, limit=args.total_requests)
    total = len(prompts)
    if total == 0:
        raise RuntimeError("No prompts loaded from ShareGPT file.")
    print(f"Loaded {total} prompts from ShareGPT.")
    
    tokenizer_path = os.path.expanduser("/home/tonywei/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompts = [
        (tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ))[:32]
        for prompt in prompts
    ]
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 2) 文本 -> token ids（最简实现）
    # prompt_token_ids = [simple_text_to_ids(t, max_input_len=args.max_input_len) for t in prompts_text]

    # 3) 为每条请求准备固定 SamplingParams
    # sampling_params_all = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len) for _ in range(total)]

    # 4) 初始化模型并 warmup
    llm = LLM(args.model_path, enforce_eager=False, max_model_len=4096)
    llm.generate(["Benchmark: "], SamplingParams())
    # 5) 按 batch-size 顺序发送：每次发 n 条，返回后再发下一批；显示 tqdm 进度
    sent = 0
    finish = 0
    t0 = time.time()
    pbar = tqdm(total=total, desc="Generating", unit="req")
    while finish < total:
        batch_prompts = prompts[sent: sent + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)
        tmp = 0
        for output in outputs:
            tmp += len(output['token_ids'])
            finish += 1
        sent += tmp
        print(outputs, sent)
        pbar.update(len(batch_prompts))
    pbar.close()

    dur = time.time() - t0
    total_out_tokens = sent
    throughput = total_out_tokens / dur if dur > 0 else float("nan")

    print("=== Summary ===")
    print(f"Requests: {total}")
    print(f"Output tokens/request: {sent / total}")
    print(f"Time: {dur:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
