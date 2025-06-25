import argparse
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="A simple script to generate text using nano-vllm."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="The model to use.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="The prompt to start generation from.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature for sampling.")
    return parser.parse_args()


def generate_text():
    args = parse_args()

    # Create the LLM engine.
    llm = LLM(args.model)

    # Generate the completions.
    prompts = [args.prompt]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Sampling parameters for better text generation
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=100,
        ignore_eos=False,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    # The output is a list of dicts, each with 'text' and 'token_ids'.
    for output in outputs:
        # The prompt is not included in the output, so we retrieve it from our list
        prompt_text = prompts[outputs.index(output)]
        generated_text = output["text"]
        print(f"Prompt: {prompt_text!r}")
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    generate_text() 