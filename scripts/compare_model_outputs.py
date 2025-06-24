"""
Debug script to compare Qwen3 HuggingFace model output with nano-vllm model output
directly at the implementation level, focusing on matching the tokenizer.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
import torch.nn.functional as F

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== MODEL OUTPUT COMPARISON FOR {model_name} ===\n")

# Test cases
test_cases = [
    "Who are you?",
    "Explain quantum computing in simple terms.",
    "Hello, how can I help you today?"
]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Tokenizer loaded: {type(tokenizer).__name__} with {len(tokenizer)} tokens")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# Load HuggingFace model
print(f"\nLoading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()
print(f"HuggingFace model loaded: {type(hf_model).__name__}")
print(f"HF model embedding shape: {hf_model.model.embed_tokens.weight.shape}")

# Load nano-vllm model
print(f"\nLoading nano-vllm model...")
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, model_name)
nano_model.eval()
print(f"nano-vllm model loaded with vocab size: {nano_model.model.embed_tokens.weight.shape[0]}")

# Test embedding lookup for specific tokens
print("\n=== EMBEDDING COMPARISON ===")
check_tokens = [
    tokenizer.eos_token_id,  # EOS
    15191,  # 'Who'
    30,     # '?'
    3555    # ' What'
]

print("\nChecking embeddings for specific tokens:")
for token_id in check_tokens:
    token_text = tokenizer.decode([token_id])
    hf_embed = hf_model.model.embed_tokens.weight[token_id]
    nano_embed = nano_model.model.embed_tokens.weight[token_id]
    
    # Compare embeddings
    diff = torch.abs(hf_embed - nano_embed).mean().item()
    print(f"Token {token_id} ('{token_text}'):")
    print(f"  - HF Embedding norm: {torch.norm(hf_embed).item():.4f}")
    print(f"  - nano Embedding norm: {torch.norm(nano_embed).item():.4f}")
    print(f"  - Mean absolute difference: {diff:.6f}")

# Direct output comparison for prompts
print("\n=== MODEL OUTPUT COMPARISON ===")
for i, prompt in enumerate(test_cases):
    print(f"\nTest Case {i+1}: '{prompt}'")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # Get next token predictions from both models
    with torch.no_grad():
        # HuggingFace predictions
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
        hf_next_token_logits = hf_logits[0, -1, :]
        hf_top_k = torch.topk(hf_next_token_logits, 5)
        
        # nano-vllm predictions
        positions = torch.arange(input_ids.size(1)).unsqueeze(0)
        nano_logits = nano_model(input_ids, positions)
        nano_next_token_logits = nano_logits[0, -1, :]
        nano_top_k = torch.topk(nano_next_token_logits, 5)
    
    print("\nHuggingFace Top 5 Predictions:")
    for idx, (token_id, score) in enumerate(zip(hf_top_k.indices.tolist(), hf_top_k.values.tolist())):
        token_text = tokenizer.decode([token_id])
        print(f"  {idx+1}. Token {token_id} ('{token_text}'): {score:.4f}")
    
    print("\nnano-vllm Top 5 Predictions:")
    for idx, (token_id, score) in enumerate(zip(nano_top_k.indices.tolist(), nano_top_k.values.tolist())):
        token_text = tokenizer.decode([token_id])
        print(f"  {idx+1}. Token {token_id} ('{token_text}'): {score:.4f}")

    # Calculate metrics
    hf_probs = F.softmax(hf_next_token_logits, dim=0)
    nano_probs = F.softmax(nano_next_token_logits, dim=0)
    
    # KL divergence
    kl_div = F.kl_div(
        nano_probs.log(), 
        hf_probs,
        reduction='sum'
    ).item()
    
    # Calculate overlap in top-K predictions
    topk = 10
    hf_topk = set(torch.topk(hf_next_token_logits, topk).indices.tolist())
    nano_topk = set(torch.topk(nano_next_token_logits, topk).indices.tolist())
    overlap = len(hf_topk.intersection(nano_topk))
    
    print(f"\nMetrics:")
    print(f"  KL divergence: {kl_div:.6f}")
    print(f"  Top-{topk} overlap: {overlap}/{topk} tokens ({overlap/topk*100:.1f}%)")

print("\n=== ANALYSIS COMPLETE ===")
