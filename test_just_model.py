"""
Simple test to validate just the model component of nano-vllm.
This bypasses the LLMEngine, Scheduler, etc.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Test model
MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "Who are you?"

print(f"\n=== LOADING HF MODEL (REFERENCE) ===")
# Load HuggingFace model for reference
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()

# Tokenize prompt
input_ids = hf_tokenizer(PROMPT, return_tensors="pt").input_ids
print(f"Input ids: {input_ids[0].tolist()}")
print(f"Decoded: {hf_tokenizer.decode(input_ids[0])}")

# Run HF model inference
print(f"\n=== RUNNING HF MODEL INFERENCE ===")
with torch.no_grad():
    outputs = hf_model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,  # Greedy decoding for consistency
        output_scores=True,
        return_dict_in_generate=True,
    )
    
hf_output_ids = outputs.sequences[0].tolist()
hf_output_text = hf_tokenizer.decode(hf_output_ids[input_ids.shape[1]:])  # Skip prompt
print(f"HF output ids: {hf_output_ids[input_ids.shape[1]:20]}")
print(f"HF output text: {repr(hf_output_text)}")

# Now load our model
print(f"\n=== LOADING NANO_VLLM MODEL ===")
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, MODEL_NAME)
nano_model.eval()

# Run our model for one token generation
print(f"\n=== RUNNING NANO_VLLM MODEL INFERENCE ===")
input_tensor = torch.tensor(input_ids[0].tolist())  # Convert to regular tensor
positions = torch.arange(len(input_tensor))

with torch.no_grad():
    # Run forward pass
    print(f"Running forward pass for input shape: {input_tensor.shape}")
    logits = nano_model(input_tensor.unsqueeze(0), positions.unsqueeze(0))
    
    # Get next token prediction
    next_token_logits = logits[0, -1, :]
    next_token_id = next_token_logits.argmax().item()

print(f"Predicted next token id: {next_token_id}")
print(f"Predicted token: {repr(hf_tokenizer.decode([next_token_id]))}")

# Check if prediction matches HF model
expected_next_token = hf_output_ids[len(input_ids[0])]
print(f"Expected next token id: {expected_next_token}")
print(f"Expected token: {repr(hf_tokenizer.decode([expected_next_token]))}")

# Compare with top-5 tokens from HF model
with torch.no_grad():
    hf_logits = hf_model(input_ids).logits
    hf_next_token_logits = hf_logits[0, -1, :]
    hf_top_k = torch.topk(hf_next_token_logits, 5)
    hf_top_ids = hf_top_k.indices.tolist()
    
    nano_top_k = torch.topk(next_token_logits, 5)
    nano_top_ids = nano_top_k.indices.tolist()

print(f"\n=== TOP-5 TOKEN COMPARISON ===")
print(f"HF top-5: {hf_top_ids}, tokens: {[hf_tokenizer.decode([i]) for i in hf_top_ids]}")
print(f"NANO top-5: {nano_top_ids}, tokens: {[hf_tokenizer.decode([i]) for i in nano_top_ids]}")

# Auto-generate a few tokens with our model
print(f"\n=== GENERATING SEQUENCE WITH NANO_VLLM MODEL ===")
# Start with input ids
curr_ids = input_ids[0].tolist()

for i in range(10):  # Generate 10 tokens
    input_tensor = torch.tensor(curr_ids)  # Convert to tensor
    positions = torch.arange(len(input_tensor))
    
    with torch.no_grad():
        # Run forward pass
        logits = nano_model(input_tensor.unsqueeze(0), positions.unsqueeze(0))
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
    
    # Add new token to sequence
    curr_ids.append(next_token_id)
    
    # Print progress
    print(f"Token {i+1}: {next_token_id} -> {repr(hf_tokenizer.decode([next_token_id]))}")

# Print final output
final_output = hf_tokenizer.decode(curr_ids)
print(f"\nFinal output: {repr(final_output)}")
