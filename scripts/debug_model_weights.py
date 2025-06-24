"""
Debug script to check model weights and generation quality.

This script investigates:
1. If model weights are properly loaded (non-zero, reasonable values)
2. If model outputs reasonable logits 
3. If token decoding works as expected
4. Compares with HuggingFace directly for sanity check
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.loader import load_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM

print("=" * 80)
print("MODEL WEIGHT AND GENERATION QUALITY DIAGNOSIS")
print("=" * 80)

# Model to test
model_name = "Qwen/Qwen3-0.6B"
prompt = "Who are you?"

print(f"\nLoading model: {model_name}")
print(f"Test prompt: {repr(prompt)}")

# 1. First load HuggingFace model for comparison
print("\n=== LOADING HUGGINGFACE MODEL (REFERENCE) ===")
hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()  # Important: set to evaluation mode
print(f"HF Model loaded, params: {sum(p.numel() for p in hf_model.parameters()):,}")

# Check HF model embedding and head weights
hf_embed_norm = torch.norm(hf_model.model.embed_tokens.weight).item()
hf_head_norm = torch.norm(hf_model.lm_head.weight).item()
print(f"HF Embedding weight norm: {hf_embed_norm:.4f}")
print(f"HF LM head weight norm: {hf_head_norm:.4f}")

# 2. Load nano-vllm model
print("\n=== LOADING NANO-VLLM MODEL ===")
config = Config(model_path=model_name)
nano_model = Qwen3ForCausalLM(config.hf_config).to(torch.device("cpu"))
load_model(nano_model, model_name)  # Explicitly load model weights
print(f"nano-vllm Model loaded, params: {sum(p.numel() for p in nano_model.parameters()):,}")

# Check nano-vllm model embedding and head weights
nano_embed_norm = torch.norm(nano_model.model.embed_tokens.weight).item()
nano_head_norm = torch.norm(nano_model.lm_head.weight).item()
print(f"nano-vllm Embedding weight norm: {nano_embed_norm:.4f}")
print(f"nano-vllm LM head weight norm: {nano_head_norm:.4f}")

# Compare some random weights
print("\n=== WEIGHT COMPARISON (SAMPLE) ===")
# Check embedding similarity (not exact due to tensor parallelism)
hf_sample = hf_model.model.embed_tokens.weight[:10, :10].detach().cpu().float().numpy()
nano_sample = nano_model.model.embed_tokens.weight[:10, :10].detach().cpu().float().numpy()
print(f"HF embedding sample:\n{hf_sample}")
print(f"nano-vllm embedding sample:\n{nano_sample}")
weight_diff = np.abs(hf_sample - nano_sample).mean()
print(f"Mean absolute difference: {weight_diff:.8f}")

# 3. Run inference and compare outputs
print("\n=== INFERENCE COMPARISON ===")
# HF Inference
input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
print(f"Input token IDs: {input_ids[0].tolist()}")
print(f"Decoded input: {hf_tokenizer.decode(input_ids[0])}")

with torch.no_grad():
    # Get HF output
    hf_outputs = hf_model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        pad_token_id=hf_tokenizer.eos_token_id
    )
    hf_text = hf_tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"HF Generated text: {repr(hf_text)}")
    
    # Get logits for first token prediction (for comparison)
    hf_logits = hf_model(input_ids).logits
    hf_next_token_logits = hf_logits[0, -1, :]
    hf_top_k = torch.topk(hf_next_token_logits, 5)
    hf_top_tokens = hf_top_k.indices.tolist()
    hf_top_values = hf_top_k.values.tolist()
    
    print("\nHF Top 5 next token predictions:")
    for i, (token_id, score) in enumerate(zip(hf_top_tokens, hf_top_values)):
        token_text = hf_tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}")

# nano-vllm inference
print("\n=== NANO-VLLM INFERENCE ===")
nano_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
engine = LLMEngine(config=config, engine_args={}, tokenizer=nano_tokenizer)
sampling_params = SamplingParams(temperature=0.7, max_tokens=20, ignore_eos=False)

print("Running nano-vllm generate...")
outputs = engine.generate([prompt], sampling_params, use_tqdm=False)
for output in outputs:
    token_ids = output.get("token_ids", [])
    flat_token_ids = engine._flatten_token_ids(token_ids)
    text = output.get("text", "")
    print(f"nano-vllm token IDs: {flat_token_ids}")
    print(f"nano-vllm output: {repr(text)}")

# Compare model behavior - Run one token through nano_model
print("\n=== DIRECT MODEL COMPARISON ===")
prompt_tokens = torch.tensor(nano_tokenizer.encode(prompt), dtype=torch.long)
positions = torch.arange(len(prompt_tokens))

with torch.no_grad():
    # nano-vllm logits
    nano_logits = nano_model(prompt_tokens.unsqueeze(0), positions.unsqueeze(0))
    nano_next_token_logits = nano_logits[0, -1, :]
    nano_top_k = torch.topk(nano_next_token_logits, 5)
    nano_top_tokens = nano_top_k.indices.tolist()
    nano_top_values = nano_top_k.values.tolist()
    
    print("\nNano-VLLM Top 5 next token predictions:")
    for i, (token_id, score) in enumerate(zip(nano_top_tokens, nano_top_values)):
        token_text = nano_tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}")

# Check for nan/inf values in model parameters
print("\n=== CHECKING FOR NAN/INF VALUES ===")
has_nan = False
for name, p in nano_model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        has_nan = True
        print(f"Found NaN or Inf in {name}")
print("NaN/Inf check: " + ("FAIL" if has_nan else "PASS"))

print("\n=== DIAGNOSIS COMPLETE ===")
