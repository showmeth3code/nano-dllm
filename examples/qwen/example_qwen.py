import os
import sys
import json
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_sharegpt_prompts(path, limit=1000):
    """读取 ShareGPT 的 JSON/JSONL，返回最多 limit 条 prompt 文本。"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        items = json.load(f) if head.lstrip().startswith("[") else (json.loads(l) for l in f if l.strip())
        for item in items:
            conv = item.get("conversations") or item.get("conversation") or []
            parts = [turn.get("value", "") for turn in conv if turn.get("value")]
            if not parts and "text" in item:
                parts = [item["text"]]
            if parts:
                texts.append("\n".join(parts))
            if len(texts) >= limit:
                break
    return texts

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B")
    # path = os.path.expanduser("/home/tonywei/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # prompts = [
    #     "List all prime numbers within 100?",
    # ]
    prompts = load_sharegpt_prompts(os.path.expanduser("~/nano-dllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"), limit=1)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
