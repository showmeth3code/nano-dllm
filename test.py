from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=32)  # Reduced max tokens for testing
    prompt = "Say hello!"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    print("Starting generation...")
    output = llm.generate([prompt], sampling_params)[0]
    print("\nGenerated Output:")
    print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
