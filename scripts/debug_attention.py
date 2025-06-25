"""
Debug script to find and fix the root cause of the model generation issues.
This script focuses on examining the attention mechanism and rotary embeddings
which seem to be the main source of differences between HF and nano-vllm.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== DEEP DEBUGGING FOR {model_name} ===\n")

# Test case
prompt = "Who are you?"

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
print(f"Decoded tokens: {[tokenizer.decode([id]) for id in input_ids[0].tolist()]}")

# Create positions tensor
positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
print(f"Positions: {positions}")

# Step-by-step debug through the model
print("\n=== ROTARY EMBEDDING COMPARISON ===")

# Compare rotary embeddings
with torch.no_grad():
    # HuggingFace rotary embeddings
    # Extract the rope_embed instance from the model
    hf_rope = hf_model.model.layers[0].self_attn.rotary_emb
    
    # Get the cos/sin from HF
    q_len = positions.size(1)
    
    # Examine Hugging Face implementation
    print(f"HF rotary implementation type: {type(hf_rope).__name__}")
    
    # Try to extract parameters for comparison
    try:
        hf_base = getattr(hf_rope, "base", 10000.0)
        print(f"HF rotary base: {hf_base}")
    except Exception as e:
        print(f"Could not extract HF rotary base parameter: {e}")
    
    # Get the cos/sin from nano-vllm
    nano_cos, nano_sin = nano_model.model.get_rotary_embedding(positions)
    
    # Print details about the cos/sin
    print(f"nano-vllm cos shape: {nano_cos.shape}, sin shape: {nano_sin.shape}")
    print(f"nano-vllm cos norm: {torch.norm(nano_cos).item():.6f}, sin norm: {torch.norm(nano_sin).item():.6f}")
    
    # Generate direct HF cos/sin for comparison if possible
    try:
        # Attempt to get cos/sin from HF
        hf_cos, hf_sin = hf_rope(positions.shape[1], positions.device)
        print(f"HF cos shape: {hf_cos.shape}, sin shape: {hf_sin.shape}")
        print(f"HF cos norm: {torch.norm(hf_cos).item():.6f}, sin norm: {torch.norm(hf_sin).item():.6f}")
        
        # Compare the cos/sin values
        if hf_cos.shape == nano_cos.shape:
            cos_diff = torch.abs(hf_cos - nano_cos).mean().item()
            sin_diff = torch.abs(hf_sin - nano_sin).mean().item()
            print(f"Rotary cos mean abs difference: {cos_diff:.6f}")
            print(f"Rotary sin mean abs difference: {sin_diff:.6f}")
        else:
            print("Shapes of rotary embeddings do not match!")
            print(f"HF cos shape: {hf_cos.shape}, nano-vllm cos shape: {nano_cos.shape}")
    except Exception as e:
        print(f"Could not directly compare rotary embeddings: {e}")

print("\n=== STEP-BY-STEP MODEL COMPARISON ===")

# Compare embeddings
with torch.no_grad():
    # HuggingFace model
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"HF embeddings norm: {torch.norm(hf_embeds).item():.6f}")
    
    # nano-vllm model
    nano_embeds = nano_model.model.embed_tokens(input_ids)
    print(f"nano embeddings shape: {nano_embeds.shape}")
    print(f"nano embeddings norm: {torch.norm(nano_embeds).item():.6f}")
    
    # Compare
    embed_diff = torch.abs(hf_embeds - nano_embeds).mean().item()
    print(f"Embeddings mean abs difference: {embed_diff:.6f}")

# Debug QKV projections
print("\n=== QKV PROJECTIONS COMPARISON ===")
with torch.no_grad():
    # HF normalization and QKV
    hf_hidden = hf_model.model.layers[0].input_layernorm(hf_embeds)
    hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_hidden)
    hf_k = hf_model.model.layers[0].self_attn.k_proj(hf_hidden)
    hf_v = hf_model.model.layers[0].self_attn.v_proj(hf_hidden)
    
    print(f"HF q shape: {hf_q.shape}, norm: {torch.norm(hf_q).item():.6f}")
    print(f"HF k shape: {hf_k.shape}, norm: {torch.norm(hf_k).item():.6f}")
    print(f"HF v shape: {hf_v.shape}, norm: {torch.norm(hf_v).item():.6f}")
    
    # nano normalization and QKV 
    nano_hidden = nano_model.model.layers[0].input_layernorm(nano_embeds)
    nano_q = nano_model.model.layers[0].self_attn.q_proj(nano_hidden)
    nano_k = nano_model.model.layers[0].self_attn.k_proj(nano_hidden)
    nano_v = nano_model.model.layers[0].self_attn.v_proj(nano_hidden)
    
    print(f"nano q shape: {nano_q.shape}, norm: {torch.norm(nano_q).item():.6f}")
    print(f"nano k shape: {nano_k.shape}, norm: {torch.norm(nano_k).item():.6f}")
    print(f"nano v shape: {nano_v.shape}, norm: {torch.norm(nano_v).item():.6f}")
    
    # Compare projections
    q_diff = torch.abs(hf_q - nano_q).mean().item()
    k_diff = torch.abs(hf_k - nano_k).mean().item()
    v_diff = torch.abs(hf_v - nano_v).mean().item()
    
    print(f"Q projection mean abs difference: {q_diff:.6f}")
    print(f"K projection mean abs difference: {k_diff:.6f}")
    print(f"V projection mean abs difference: {v_diff:.6f}")

# Debug QK normalization specific to Qwen models
print("\n=== QK NORMALIZATION COMPARISON ===")
with torch.no_grad():
    try:
        # Reshape heads to match expected input format for normalization
        head_dim = config.hidden_size // config.num_attention_heads
        
        # HF reshape and normalize
        q_batch_size, q_seq_len = hf_q.shape[:2]
        hf_q_heads = hf_q.view(q_batch_size, q_seq_len, config.num_attention_heads, head_dim)
        hf_k_heads = hf_k.view(q_batch_size, q_seq_len, config.num_key_value_heads, head_dim)
        
        # Check for q_norm and k_norm in HF model
        if hasattr(hf_model.model.layers[0].self_attn, "q_norm") and hasattr(hf_model.model.layers[0].self_attn, "k_norm"):
            hf_q_norm = hf_model.model.layers[0].self_attn.q_norm(hf_q_heads)
            hf_k_norm = hf_model.model.layers[0].self_attn.k_norm(hf_k_heads)
            
            print(f"HF q_norm shape: {hf_q_norm.shape}, norm: {torch.norm(hf_q_norm).item():.6f}")
            print(f"HF k_norm shape: {hf_k_norm.shape}, norm: {torch.norm(hf_k_norm).item():.6f}")
        else:
            print("HF model doesn't have separate q_norm/k_norm")
            hf_q_norm, hf_k_norm = hf_q_heads, hf_k_heads
        
        # nano reshape and normalize
        nano_q_heads = nano_q.view(q_batch_size, q_seq_len, config.num_attention_heads, head_dim)
        nano_k_heads = nano_k.view(q_batch_size, q_seq_len, config.num_key_value_heads, head_dim)
        
        nano_q_norm = nano_model.model.layers[0].self_attn.q_norm(nano_q_heads)
        nano_k_norm = nano_model.model.layers[0].self_attn.k_norm(nano_k_heads)
        
        print(f"nano q_norm shape: {nano_q_norm.shape}, norm: {torch.norm(nano_q_norm).item():.6f}")
        print(f"nano k_norm shape: {nano_k_norm.shape}, norm: {torch.norm(nano_k_norm).item():.6f}")
        
        # Compare normalized q and k
        q_norm_diff = torch.abs(hf_q_norm - nano_q_norm).mean().item()
        k_norm_diff = torch.abs(hf_k_norm - nano_k_norm).mean().item()
        
        print(f"Q norm mean abs difference: {q_norm_diff:.6f}")
        print(f"K norm mean abs difference: {k_norm_diff:.6f}")
    except Exception as e:
        print(f"Error in QK normalization comparison: {e}")

# Final output comparison
print("\n=== FINAL OUTPUT COMPARISON ===")
with torch.no_grad():
    # Run through both models
    hf_output = hf_model(input_ids).logits
    nano_output = nano_model(input_ids, positions)
    
    print(f"HF output shape: {hf_output.shape}")
    print(f"HF output norm: {torch.norm(hf_output).item():.6f}")
    print(f"nano output shape: {nano_output.shape}")
    print(f"nano output norm: {torch.norm(nano_output).item():.6f}")
    
    # Compare final logits
    logits_diff = torch.abs(hf_output - nano_output).mean().item()
    print(f"Final logits mean absolute difference: {logits_diff:.6f}")
    
    # Compare next token predictions
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

# Also verify end-to-end generation
print("\n=== GENERATION TEST ===")
# HuggingFace generation
with torch.no_grad():
    hf_outputs = hf_model.generate(
        input_ids, 
        max_new_tokens=20, 
        do_sample=False,  # Use greedy decoding for deterministic results
        pad_token_id=tokenizer.eos_token_id
    )
    hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
print(f"HF Generated: {repr(hf_text)}")

# nano-vllm generation (simple greedy)
print("\nGenerating with nano-vllm model...")
curr_ids = input_ids[0].tolist()

with torch.no_grad():
    for i in range(20):  # Generate 20 tokens
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

# Fix suggestions based on findings
print("\n=== RECOMMENDATIONS FOR FIXING ISSUES ===")
print("1. Make sure the rotary embeddings are correctly generated with matching shape and base")
print("2. Check that q_norm and k_norm are applied correctly with the same parameters")
print("3. Ensure that attention heads are reshaped correctly before and after operations")
print("4. Verify that when grouping q/k/v heads, the indexing and reshaping matches the HF version")
print("5. Double-check the attention scaling factor (1/sqrt(head_dim))")
print("6. Verify that attention output projections have the correct weights")
