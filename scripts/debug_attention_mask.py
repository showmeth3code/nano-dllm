"""
Debug script to inspect the attention masks during model inference.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== ATTENTION MASK DEBUG FOR {model_name} ===\n")

# Define debug mode
debug_masks = True

# Monkey patch the attention forward to debug masks
def debug_attention_mask(model):
    """Patch the forward method of attention modules to inspect masks"""
    from nanovllm.layers.attention import SelfAttention
    original_forward = SelfAttention.forward
    
    def patched_forward(self, hidden_states, cos, sin, cu_seqlens, max_s, layer_past=None, use_cache=False):
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project Q, K, V using the respective projection matrices
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RMSNorm to query and key projections - Qwen3 specific feature
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q, cos, sin)
        k = self.rotary_emb(k, cos, sin)
        
        # Handle key-value cache
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        present = (k, v) if use_cache else None
        
        # Handle grouped-query attention
        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)
        
        # Transpose for batched matrix multiplication
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        q_float = q.to(torch.float32)
        k_float = k.to(torch.float32)
        
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q_float, k_float.transpose(-1, -2)) * scale
        
        # Debug attention mask creation
        q_len, k_len = q.size(-2), k.size(-2)
        print(f"Attention scores shape: {attn_scores.shape}, q_len={q_len}, k_len={k_len}")
        
        if layer_past is None:
            # Create causal mask
            causal_mask = torch.ones((q_len, k_len), device=hidden_states.device) < 0
            causal_mask = torch.triu(causal_mask, diagonal=1)
            
            # Debug mask details
            print(f"Causal mask shape: {causal_mask.shape}")
            print(f"Causal mask (True means masked):\n{causal_mask}")
            
            # For additional clarity, show which positions will be set to -inf
            print("Mask pattern (1 = will be set to -inf, 0 = kept):")
            print(causal_mask.int().cpu().numpy())
            
            # Create broadcasted mask for adding to attention scores
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Apply mask to attention scores (change masked positions to -inf)
            mask_value = torch.finfo(attn_scores.dtype).min
            masked_attn_scores = attn_scores.masked_fill(causal_mask, mask_value)
            
            # Show sample before/after masking
            print(f"Before masking - sample attention scores:\n{attn_scores[0, 0]}")
            print(f"After masking - sample attention scores:\n{masked_attn_scores[0, 0]}")
            
            attn_scores = masked_attn_scores
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Show sample attention weights after softmax
        print(f"Attention weights after softmax:\n{attn_weights[0, 0]}")
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to match hidden states dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Combine all heads
        input_shape = (batch_size, seq_len)
        attn_output = attn_output.reshape(*input_shape, -1)
        
        # Final output projection
        output = self.o_proj(attn_output)
        
        return output, present
    
    # Apply the patch
    if debug_masks:
        SelfAttention.forward = patched_forward
    
    return original_forward

# Test case
prompt = "Hi!"

# Load tokenizer and models
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
hf_model.eval()

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)

# Patch attention for debugging
original_forward = debug_attention_mask(nano_model)

load_model(nano_model, model_name)
nano_model.eval()

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Prompt: '{prompt}'")
print(f"Token IDs: {input_ids[0].tolist()}")

# Create positions tensor
positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

# Run forward pass on nano model with debug info
print("\n=== RUNNING MODEL WITH DEBUG INFO ===")
with torch.no_grad():
    # Run model
    outputs = nano_model(input_ids, positions)
    print(f"Model outputs shape: {outputs.shape}")

# Restore original forward method
if debug_masks:
    from nanovllm.layers.attention import SelfAttention
    SelfAttention.forward = original_forward
    print("\nRestored original attention implementation.")
