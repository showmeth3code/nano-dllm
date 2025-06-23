"""
Script to directly test the Hugging Face model's token-by-token generation
"""

import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    args = parser.parse_args()

    # Get device
    device = torch.device(args.device)
    model_path = args.model_path
    print(f"Testing HF model {model_path} on {device} device")
    
    # Load model and tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    
    # Tokenize prompt
    prompt = args.prompt
    print(f"Prompt: '{prompt}'")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"Tokenized: {input_ids.tolist()[0]}")
    
    # Generate using step-by-step token generation (as in the model runner)
    max_gen_len = 20
    generated_ids = input_ids.clone()
    
    print("\nGenerating HF model output token-by-token:")
    for i in range(max_gen_len):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Display the token
        next_token = tokenizer.decode(next_token_id[0])
        print(f"  Token {i}: {next_token_id.item()} → {repr(next_token)}")
        
        # Add token to generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
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

if __name__ == "__main__":
    main()
