"""
Test script to verify that the rotary embedding changes have fixed the model output.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== MODEL OUTPUT TEST FOR {model_name} ===\n")

# Test case
prompt = "Hi!"

# Load tokenizer and models
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, model_name)
nano_model.eval()

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {input_ids[0].tolist()}")

# Create positions tensor
positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

# Compare model outputs
print("\n=== FORWARD PASS COMPARISON ===")

# HF forward pass
with torch.no_grad():
    # HF model
    hf_output = hf_model(input_ids).logits
    print(f"HF output shape: {hf_output.shape}")
    print(f"HF logits norm: {torch.norm(hf_output).item():.6f}")
    
    # nano model
    nano_output = nano_model(input_ids, positions)
    print(f"nano output shape: {nano_output.shape}")
    print(f"nano logits norm: {torch.norm(nano_output).item():.6f}")
    
    # Compare outputs
    logits_diff = torch.abs(hf_output - nano_output).mean().item()
    print(f"Mean absolute difference in logits: {logits_diff:.6f}")
    
    # Compare top tokens
    hf_next_token_logits = hf_output[0, -1, :]
    nano_next_token_logits = nano_output[0, -1, :]
    
    hf_top_k = torch.topk(hf_next_token_logits, 5)
    nano_top_k = torch.topk(nano_next_token_logits, 5)
    
    print("\nHF Top 5 Next Token Predictions:")
    for i, (token_id, score) in enumerate(zip(hf_top_k.indices.tolist(), hf_top_k.values.tolist())):
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{token_text}'): {score:.4f}")
    
    print("\nnano Top 5 Next Token Predictions:")
    for i, (token_id, score) in enumerate(zip(nano_top_k.indices.tolist(), nano_top_k.values.tolist())):
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{token_text}'): {score:.4f}")
    
    # Check for overlapping predictions
    hf_set = set(hf_top_k.indices.tolist())
    nano_set = set(nano_top_k.indices.tolist())
    overlap = hf_set.intersection(nano_set)
    print(f"\nOverlap in top predictions: {len(overlap)}/{len(hf_set)}")
    print(f"Common tokens: {overlap}")

# Test generation
print("\n=== GENERATION TEST ===")
# HuggingFace generation
with torch.no_grad():
    hf_outputs = hf_model.generate(
        input_ids, 
        max_new_tokens=10, 
        do_sample=False,  # Use greedy decoding for deterministic results
        pad_token_id=tokenizer.eos_token_id
    )
    hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
print(f"HF Generated: {repr(hf_text)}")

# nano-vllm generation (simple greedy)
print("\nGenerating with nano-vllm model...")
curr_ids = input_ids[0].tolist()

with torch.no_grad():
    for i in range(10):  # Generate 10 tokens
        input_tensor = torch.tensor(curr_ids).unsqueeze(0)  # Add batch dim
        positions = torch.arange(len(curr_ids)).unsqueeze(0)  # Add batch dim
        
        # Forward pass through model
        logits = nano_model(input_tensor, positions)
        
        # Get next token prediction (greedy)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()
        
        # Add token to sequence
        curr_ids.append(next_token_id)
        
        # Print generated token
        token_text = tokenizer.decode([next_token_id])
        print(f"Token {i+1}: {next_token_id} -> {repr(token_text)}")

# Print final output
nano_text = tokenizer.decode(curr_ids, skip_special_tokens=True)
print(f"\nnano-vllm Generated: {repr(nano_text)}")

# Compare generated text
print("\nGeneration match:", "✅" if hf_text == nano_text else "❌")
