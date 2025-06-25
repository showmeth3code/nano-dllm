
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # For bfloat16 support, we use float32 mode for CPU and MPS
    dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=dtype,
                                             trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Test generation
    prompt = "Hello, my name is"
    print(f"Prompt: {prompt}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # Generate with regular HF
    print("\nGenerating with standard HF generate()...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,  # Greedy decoding
            use_cache=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {repr(text)}")

    # Now try manual auto-regressive generation
    print("\nGenerating manually token-by-token...")
    with torch.no_grad():
        # Track generated tokens
        generated = []

        # Start with the prompt tokens
        current_ids = input_ids.clone()

        for i in range(20):
            # Create position IDs matching the sequence length
            seq_len = current_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Forward pass through the model
            outputs = model(
                current_ids,
                position_ids=position_ids,
                use_cache=False
            )

            # Get the logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Get the most likely token (greedy)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Print token info
            token_item = next_token.item()
            token_text = tokenizer.decode([token_item])
            print(f"Token {i}: {token_item} -> {token_text!r}")
            generated.append(token_item)

            # Add the new token to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Decode all generated tokens
        manual_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\nManually generated: {repr(manual_text)}")

if __name__ == "__main__":
    main()
