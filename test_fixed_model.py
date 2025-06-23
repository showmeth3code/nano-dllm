#!/usr/bin/env python3
"""
A minimal test script to validate the nano-vllm fixes for Qwen3 models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== TESTING FIXED MODEL: {model_name} ===\n")

# Test prompts
prompts = [
    "Who are you?", 
    "What is the capital of France?",
    "Explain quantum computing in simple terms."
]

# Load tokenizer and models
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, model_name)
nano_model.eval()

print(f"Loaded models successfully. Vocab size: {len(tokenizer)}")

for prompt in prompts:
    print("\n" + "="*80)
    print(f"Prompt: {repr(prompt)}")
    print("="*80)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # Create positions tensor
    positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    
    # Generate with HuggingFace model
    print("\n--- HuggingFace Generation ---")
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,  # Use greedy for deterministic comparison
            pad_token_id=tokenizer.eos_token_id
        )
        hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"HF Output: {repr(hf_text)}")
    
    # Generate with nano-vllm model
    print("\n--- nano-vllm Generation ---")
    curr_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for i in range(30):  # Generate 30 tokens
            input_tensor = torch.tensor(curr_ids, dtype=torch.long).unsqueeze(0)  # Add batch dim
            positions = torch.arange(len(curr_ids), dtype=torch.long).unsqueeze(0)  # Add batch dim
            
            # Forward pass through model
            logits = nano_model(input_tensor, positions)
            
            # Get next token prediction (greedy)
            next_token_id = logits[0, -1, :].argmax(dim=-1).item()
            
            # Add token to sequence
            curr_ids.append(next_token_id)
            
            # Print generated token
            token_text = tokenizer.decode([next_token_id])
            print(f"Token {i+1}: {next_token_id} -> {repr(token_text)}", end="\r")
            
            # Check if the model generated an EOS token
            if next_token_id == tokenizer.eos_token_id:
                print("\nGeneration complete - encountered EOS token")
                break
    
    # Print final output
    nano_text = tokenizer.decode(curr_ids, skip_special_tokens=True)
    print(f"\nnano-vllm Output: {repr(nano_text)}")
    
    # Compare the outputs
    print("\n--- Comparison ---")
    if hf_text == nano_text:
        print("MATCH: Outputs are identical!")
    else:
        print("Different outputs. Checking similarity...")
        
        # Calculate overlap percentage
        min_len = min(len(hf_text), len(nano_text))
        matching_chars = sum(1 for a, b in zip(hf_text[:min_len], nano_text[:min_len]) if a == b)
        overlap_percentage = (matching_chars / min_len) * 100
        
        print(f"Character overlap: {matching_chars}/{min_len} ({overlap_percentage:.2f}%)")
        
        # Compare token-wise
        hf_tokens = hf_outputs[0].tolist()
        nano_tokens = curr_ids
        
        min_tokens = min(len(hf_tokens), len(nano_tokens))
        matching_tokens = sum(1 for a, b in zip(hf_tokens[:min_tokens], nano_tokens[:min_tokens]) if a == b)
        token_overlap = (matching_tokens / min_tokens) * 100
        
        print(f"Token overlap: {matching_tokens}/{min_tokens} ({token_overlap:.2f}%)")
        
        # Show where they diverge
        divergence_idx = next((i for i, (a, b) in enumerate(zip(hf_tokens, nano_tokens)) if a != b), min_tokens)
        
        if divergence_idx < min_tokens:
            print(f"First divergence at token {divergence_idx}:")
            print(f"  HF: {hf_tokens[divergence_idx]} ({repr(tokenizer.decode([hf_tokens[divergence_idx]]))})")
            print(f"  nano: {nano_tokens[divergence_idx]} ({repr(tokenizer.decode([nano_tokens[divergence_idx]]))})")

print("\nTesting complete!")
