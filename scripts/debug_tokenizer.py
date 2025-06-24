"""
Debug script to check tokenizer behavior specifically.
"""
import torch
from transformers import AutoTokenizer
import os
import json

# Model to test
model_name = "Qwen/Qwen3-0.6B"
prompt = "Who are you?"

print(f"\n=== TOKENIZER ANALYSIS FOR {model_name} ===")

# Load the tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Check tokenizer details
print(f"Tokenizer class: {type(tokenizer).__name__}")
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
print(f"Tokenizer model max length: {tokenizer.model_max_length}")
print(f"Tokenizer pad token: {tokenizer.pad_token}")
print(f"Tokenizer eos token: {tokenizer.eos_token}")

# Examine how the tokenizer handles the prompt
input_ids = tokenizer.encode(prompt)
print(f"\nPrompt: '{prompt}'")
print(f"Encoded token IDs: {input_ids}")
print(f"Decoded back: '{tokenizer.decode(input_ids)}'")

# Check how individual tokens are decoded
print(f"\nDecoding individual tokens:")
for i, token_id in enumerate(input_ids):
    token_text = tokenizer.decode([token_id])
    print(f"  Token {i} (ID {token_id}): '{token_text}'")
    
# Check if tokenizer saves/loads correctly
# Save tokenizer to a temporary directory
temp_dir = "/tmp/debug_tokenizer"
os.makedirs(temp_dir, exist_ok=True)
tokenizer.save_pretrained(temp_dir)
print(f"\nSaved tokenizer to {temp_dir}")

# Check what files were saved
print(f"Saved files:")
for file in os.listdir(temp_dir):
    print(f"  {file}")
    if file == "tokenizer_config.json":
        with open(os.path.join(temp_dir, file), 'r') as f:
            config = json.load(f)
            print(f"  tokenizer_config.json contents: {json.dumps(config, indent=2)}")

# Now load the tokenizer back
print("\nLoading tokenizer from saved files...")
loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir, trust_remote_code=True)

# Compare with original tokenizer
print(f"Loaded tokenizer class: {type(loaded_tokenizer).__name__}")
print(f"Loaded tokenizer vocab size: {len(loaded_tokenizer)}")

# Test tokenization with loaded tokenizer
loaded_ids = loaded_tokenizer.encode(prompt)
print(f"Loaded tokenizer encoding: {loaded_ids}")
print(f"Decoded back with loaded tokenizer: '{loaded_tokenizer.decode(loaded_ids)}'")

# Check if there are any differences between original and loaded tokenizer
if input_ids == loaded_ids:
    print("\nTokenization is consistent between original and loaded tokenizer.")
else:
    print("\nWARNING: Tokenization is different between original and loaded tokenizer!")
    print(f"Original: {input_ids}")
    print(f"Loaded:   {loaded_ids}")

print("\n=== TESTING BASIC GENERATION ===")
# Test simple token generation sequence with both tokenizers
print("Generating with original tokenizer:")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

with torch.no_grad():
    input_tensor = torch.tensor([input_ids])
    outputs = model.generate(input_tensor, max_new_tokens=5, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated text: '{generated_text}'")
    print(f"Generated IDs: {outputs[0].tolist()}")
