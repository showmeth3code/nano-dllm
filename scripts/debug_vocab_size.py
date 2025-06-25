"""
Debug script to compare vocab size discrepancies between Qwen3 model and tokenizer.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
import os
import numpy as np

# Model to test
model_name = "Qwen/Qwen3-0.6B"
print(f"=== VOCAB SIZE COMPARISON FOR {model_name} ===\n")

# 1. Load Hugging Face model and config
print("Loading HuggingFace model and config...")
hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"HF config vocab_size: {hf_config.vocab_size}")
try:
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    print(f"HF model.lm_head.weight shape: {hf_model.lm_head.weight.shape}")
    print(f"HF model.model.embed_tokens.weight shape: {hf_model.model.embed_tokens.weight.shape}")
except Exception as e:
    print(f"Error loading HF model: {str(e)}")

# 2. Load tokenizer directly
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"\nTokenizer class: {type(tokenizer).__name__}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    
    # Check special token IDs
    special_tokens = []
    for token_name, token_value in tokenizer.special_tokens_map.items():
        if isinstance(token_value, list):
            for t in token_value:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(t)
                    special_tokens.append((t, token_id))
                except:
                    pass
        else:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token_value)
                special_tokens.append((token_value, token_id))
            except:
                pass
    
    print("\nSpecial token IDs:")
    for token, token_id in special_tokens:
        print(f"  {token}: {token_id}")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")

# 3. Create nano-vllm model with HF config
try:
    print("\nCreating nano-vllm model with HF config...")
    nano_model = Qwen3ForCausalLM(hf_config)
    print(f"nano-vllm model.lm_head.weight shape: {nano_model.lm_head.weight.shape}")
    print(f"nano-vllm model.model.embed_tokens.weight shape: {nano_model.model.embed_tokens.weight.shape}")
except Exception as e:
    print(f"Error creating nano-vllm model: {str(e)}")

# 4. Compare the shapes of key tensors
try:
    print("\nComparing key tensor shapes:")
    
    # Load both models
    print("\nChecking for specific vocab size mismatches:")
    print(f"HF config.vocab_size: {hf_config.vocab_size}")
    print(f"len(tokenizer): {len(tokenizer)}")
    
    # Check vocab size in embedding layers
    nano_vocab_size = nano_model.model.embed_tokens.weight.shape[0]
    hf_vocab_size = hf_model.model.embed_tokens.weight.shape[0]
    
    print(f"HF model embedding vocab size: {hf_vocab_size}")
    print(f"nano-vllm model embedding vocab size: {nano_vocab_size}")
    print(f"Difference: {hf_vocab_size - nano_vocab_size}")
    
    # Print content of the HF config to see if there's anything useful
    print("\nHF config attributes:")
    for key, value in vars(hf_config).items():
        if key != "architectures" and not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
except Exception as e:
    print(f"Error comparing shapes: {str(e)}")
