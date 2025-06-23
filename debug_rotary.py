"""
Debug script to investigate the rotary embedding differences
between HuggingFace and nano-vllm.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, rotate_half

# Set model name
model_name = "Qwen/Qwen3-0.6B"
print(f"=== ROTARY EMBEDDING DEBUGGING FOR {model_name} ===\n")

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
print(f"Decoded tokens: {[tokenizer.decode([id]) for id in input_ids[0].tolist()]}")

# Create positions tensor
positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
print(f"Positions: {positions}")

# Step-by-step debug of rotary embeddings
print("\n=== ROTARY EMBEDDING COMPARISON ===")

with torch.no_grad():
    # Extract cos/sin from both models
    print("\n1. Examining the rotary embedding generation")
    
    # Get the cos/sin from HF model
    # For HF models, we need to find the rotary embedding module and call it
    # This is specific to Qwen3
    rotary_emb = None
    for name, module in hf_model.named_modules():
        if isinstance(module, torch.nn.Module) and any(x in name for x in ["rotary"]):
            rotary_emb = module
            print(f"Found rotary module: {name}, type: {type(module).__name__}")
            break
    
    if rotary_emb is None:
        print("Could not find rotary embedding module in HF model")
        # Try to find it in a different way - look directly in the first layer's attention
        try:
            rotary_emb = hf_model.model.layers[0].self_attn.rotary_emb
            print(f"Found rotary module in first layer attention: {type(rotary_emb).__name__}")
        except:
            print("Could not find rotary in first layer attention either")
            
            # Try one more approach - use the Qwen3RotaryEmbedding directly
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
            rotary_emb = Qwen3RotaryEmbedding(config)
            print(f"Created new rotary embedding module: {type(rotary_emb).__name__}")
            
    # Get HF cos/sin
    print("Generating HF cos/sin")
    hf_cos, hf_sin = rotary_emb(torch.ones_like(input_ids), positions)
    print(f"HF cos shape: {hf_cos.shape}, dtype: {hf_cos.dtype}")
    print(f"HF sin shape: {hf_sin.shape}, dtype: {hf_sin.dtype}")
    
    # Get nano cos/sin
    print("Generating nano cos/sin")
    nano_cos, nano_sin = nano_model.model.get_rotary_embedding(positions)
    print(f"nano cos shape: {nano_cos.shape}, dtype: {nano_cos.dtype}")
    print(f"nano sin shape: {nano_sin.shape}, dtype: {nano_sin.dtype}")
    
    # Compare the values
    if hf_cos.shape == nano_cos.shape:
        cos_diff = torch.abs(hf_cos - nano_cos.to(hf_cos.dtype)).mean().item()
        sin_diff = torch.abs(hf_sin - nano_sin.to(hf_sin.dtype)).mean().item()
        print(f"Mean abs difference - cos: {cos_diff:.6f}, sin: {sin_diff:.6f}")
        print(f"Max abs difference - cos: {torch.abs(hf_cos - nano_cos.to(hf_cos.dtype)).max().item():.6f}, "
              f"sin: {torch.abs(hf_sin - nano_sin.to(hf_sin.dtype)).max().item():.6f}")
    else:
        print(f"Shape mismatch - HF cos: {hf_cos.shape}, nano cos: {nano_cos.shape}")
    
    # Test rotate_half function
    print("\n2. Testing rotate_half function")
    
    # Create a test tensor
    test_tensor = torch.randn(2, 4, 128)
    
    # Apply HF's rotate_half
    hf_rotated = rotate_half(test_tensor)
    print(f"HF rotate_half output shape: {hf_rotated.shape}")
    
    # Apply nano's rotate_half (via the apply_rotary_emb function)
    from nanovllm.layers.rotary_embedding import rotate_half as nano_rotate_half
    nano_rotated = nano_rotate_half(test_tensor)
    print(f"nano rotate_half output shape: {nano_rotated.shape}")
    
    # Compare
    rotate_diff = torch.abs(hf_rotated - nano_rotated).mean().item()
    print(f"rotate_half mean abs difference: {rotate_diff:.6f}")
    
    # Now test the full rotation process
    print("\n3. Testing full rotary application")
    
    # Create test q and k
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    seq_len = 2
    head_dim = 128
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    
    # Apply HF rotary embedding
    hf_q_rotated, hf_k_rotated = apply_rotary_pos_emb(
        q, k, hf_cos, hf_sin, unsqueeze_dim=1
    )
    
    print(f"HF rotated q shape: {hf_q_rotated.shape}, k shape: {hf_k_rotated.shape}")
    
    # Apply nano rotary embedding - note difference in API
    # We may need to reshape tensors to match expectations
    q_reshaped = q.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    k_reshaped = k.transpose(1, 2)  # [batch, seq_len, num_kv_heads, head_dim]
    
    from nanovllm.layers.rotary_embedding import apply_rotary_emb
    nano_q_rotated = apply_rotary_emb(q_reshaped, nano_cos, nano_sin)
    nano_k_rotated = apply_rotary_emb(k_reshaped, nano_cos, nano_sin)
    
    # Reshape back to compare
    nano_q_rotated = nano_q_rotated.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    nano_k_rotated = nano_k_rotated.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
    
    print(f"nano rotated q shape: {nano_q_rotated.shape}, k shape: {nano_k_rotated.shape}")
    
    # Compare final results
    q_diff = torch.abs(hf_q_rotated - nano_q_rotated).mean().item()
    k_diff = torch.abs(hf_k_rotated - nano_k_rotated).mean().item()
    
    print(f"q rotated mean abs difference: {q_diff:.6f}")
    print(f"k rotated mean abs difference: {k_diff:.6f}")
    
    # Check if the output type matches the input type (important for mixed precision)
    print(f"\nType check - input q: {q.dtype}, hf rotated q: {hf_q_rotated.dtype}, nano rotated q: {nano_q_rotated.dtype}")
    print(f"Type check - input k: {k.dtype}, hf rotated k: {hf_k_rotated.dtype}, nano rotated k: {nano_k_rotated.dtype}")

print("\n=== CONCLUSION ===")
print("Based on the analysis of rotary embeddings:")
print("1. Look for shape mismatches in cos/sin tensor formatting")
print("2. Check for differences in rotate_half implementation")
print("3. Verify dtypes are preserved correctly during operations")
print("4. Confirm the scaling factor is consistent between implementations")
