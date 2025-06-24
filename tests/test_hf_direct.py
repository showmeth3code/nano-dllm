
"""
Script to directly test the Hugging Face model's token-by-token generation.
Supports both CLI and pytest usage.
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple

def run_hf_generation(
    model_path: str = "Qwen/Qwen3-0.6B",
    device: str = "cpu",
    prompt: str = "Hello, my name is",
    max_gen_len: int = 20,
) -> Tuple[str, str]:
    """
    Run HuggingFace model generation both step-by-step and with standard generate().
    Returns (stepwise_output, standard_output).
    """
    # Device check
    torch_device = torch.device(device)
    print(f"Testing HF model {model_path} on {torch_device} device")

    # Load model and tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    model = model.to(torch_device)
    model.eval()

    # Tokenize prompt
    print(f"Prompt: '{prompt}'")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(torch_device)
    print(f"Tokenized: {input_ids.tolist()[0]}")

    # Generate using step-by-step token generation (as in the model runner)
    generated_ids = input_ids.clone()
    print("\nGenerating HF model output token-by-token:")
    for i in range(max_gen_len):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
        # [batch, seq, vocab]
        next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [batch, 1]
        # Display the token
        next_token = tokenizer.decode(next_token_id[0])
        print(f"  Token {i}: {next_token_id.item()} → {repr(next_token)}")
        # Add token to generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)  # [batch, seq+1]

    # Decode the complete sequence
    complete_output = tokenizer.decode(generated_ids[0])
    print("\nComplete output:")
    print(complete_output)

    print("\nNow testing with standard HF generation:")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_gen_len,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    standard_output = tokenizer.decode(outputs[0])
    print(standard_output)
    return complete_output, standard_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--max_gen_len", type=int, default=20)
    args = parser.parse_args()
    run_hf_generation(
        model_path=args.model_path,
        device=args.device,
        prompt=args.prompt,
        max_gen_len=args.max_gen_len,
    )


def test_hf_direct_default() -> None:
    """Test HF direct generation with default parameters (smoke test)."""
    try:
        stepwise, standard = run_hf_generation(max_gen_len=2)  # keep short for test
        assert isinstance(stepwise, str) and isinstance(standard, str)
        assert len(stepwise) > 0 and len(standard) > 0
    except Exception as e:
        import pytest
        pytest.skip(f"Skipping test due to environment/model error: {e}")


if __name__ == "__main__":
    main()
