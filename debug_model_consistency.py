"""
Simple debug script for comparing HuggingFace vs nano-vllm for Qwen3 model.
This focuses on just comparing the outputs without complex hooks.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Deterministic behavior for consistent debugging
torch.manual_seed(123)

# Use CPU for guaranteed precision match (MPS has different fp behavior)
device = torch.device("cpu")
print(f"Using device: {device}")

# Load models - use smaller model for quicker debugging
model_name = "Qwen/Qwen3-0.6B"
print(f"Loading models from: {model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# Load HuggingFace model for reference
print("\n=== Loading HuggingFace model ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32,  # Use float32 for CPU comparison
    trust_remote_code=True
)
hf_model = hf_model.to(device)
hf_model.eval()
print(f"HF model loaded, device: {next(hf_model.parameters()).device}")

# Load nano-vllm model
print("\n=== Loading nano-vllm model ===")
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, model_name)
nano_model = nano_model.to(device)
nano_model.eval()
print(f"nano-vllm model loaded, device: {next(nano_model.parameters()).device}")

# Simple test prompt
prompt = "Hello, how are you?"
print(f"\nTesting with prompt: {repr(prompt)}")

# Tokenize input
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"Input shape: {input_ids.shape}, tokens: {input_ids.tolist()[0]}")

# Position IDs for nano-vllm
positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
print(f"Positions: {positions}")

# Compare key model weights to identify issues
def compare_key_weights():
    """Compare key weights between models"""
    print("\n=== Comparing Key Model Weights ===")
    
    # Get state dicts
    hf_state_dict = hf_model.state_dict()
    nano_state_dict = nano_model.state_dict()
    
    # Important components to check
    key_components = [
        # Embeddings
        ('model.embed_tokens.weight', 'model.embed_tokens.weight'),
        
        # Attention components - first layer
        ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.q_proj.weight'), 
        ('model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.k_proj.weight'),
        ('model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.v_proj.weight'),
        ('model.layers.0.self_attn.o_proj.weight', 'model.layers.0.self_attn.o_proj.weight'),
        
        # Output head
        ('lm_head.weight', 'lm_head.weight'),
    ]
    
    # Check each key pair
    for hf_key, nano_key in key_components:
        print(f"\nChecking {hf_key} vs {nano_key}")
        
        # Check if keys exist
        hf_exists = hf_key in hf_state_dict
        nano_exists = nano_key in nano_state_dict
        
        if not hf_exists:
            print(f"  ❌ HF key '{hf_key}' not found in HF model")
            continue
            
        if not nano_exists:
            print(f"  ❌ Nano key '{nano_key}' not found in Nano model")
            continue
        
        # Get tensors
        hf_tensor = hf_state_dict[hf_key].float()  # Ensure same dtype
        nano_tensor = nano_state_dict[nano_key].float()
        
        # Compare shapes
        print(f"  Shapes: HF {hf_tensor.shape}, Nano {nano_tensor.shape}")
        if hf_tensor.shape != nano_tensor.shape:
            print("  ❌ Shape mismatch!")
            continue
        
        # Compare stats
        hf_mean = hf_tensor.mean().item()
        hf_std = hf_tensor.std().item()
        nano_mean = nano_tensor.mean().item()
        nano_std = nano_tensor.std().item()
        
        print(f"  Mean: HF {hf_mean:.6f}, Nano {nano_mean:.6f}, Diff {abs(hf_mean - nano_mean):.6f}")
        print(f"  Std: HF {hf_std:.6f}, Nano {nano_std:.6f}, Diff {abs(hf_std - nano_std):.6f}")
        
        # Check if tensors are close
        is_close = torch.allclose(hf_tensor, nano_tensor, rtol=1e-3, atol=1e-3)
        print(f"  Values match: {'✅' if is_close else '❌'}")
        
        if not is_close:
            # Sample some differences
            diff = (hf_tensor - nano_tensor).abs()
            max_diff = diff.max().item()
            print(f"  Max difference: {max_diff:.6f}")
            
            # Get sample of largest differences
            flat_diff = diff.flatten()
            worst_idx = torch.argmax(flat_diff).item()
            worst_value = flat_diff[worst_idx].item()
            
            # Convert to index in original tensor
            tensor_idx = torch.nonzero(diff == worst_value, as_tuple=False)[0].tolist()
            
            print(f"  Largest diff at {tensor_idx}: HF {hf_tensor[tuple(tensor_idx)]:.6f}, " 
                  f"Nano {nano_tensor[tuple(tensor_idx)]:.6f}")

# Forward pass through both models
with torch.no_grad():
    # HuggingFace forward pass
    print("\n=== Running HuggingFace model inference ===")
    hf_outputs = hf_model(input_ids)
    hf_logits = hf_outputs.logits
    
    # nano-vllm forward pass
    print("=== Running nano-vllm model inference ===")
    nano_logits = nano_model(input_ids, positions)

# Compare outputs
print("\n=== Comparing model outputs ===")
print(f"HF logits shape: {hf_logits.shape}, Nano logits shape: {nano_logits.shape}")

if hf_logits.shape == nano_logits.shape:
    # Compare logits statistics
    hf_mean = hf_logits.mean().item()
    hf_std = hf_logits.std().item()
    nano_mean = nano_logits.mean().item()
    nano_std = nano_logits.std().item()
    
    print(f"HF logits - mean: {hf_mean:.6f}, std: {hf_std:.6f}")
    print(f"Nano logits - mean: {nano_mean:.6f}, std: {nano_std:.6f}")
    
    # Check if outputs are close
    is_close = torch.allclose(hf_logits, nano_logits, rtol=1e-3, atol=1e-3)
    print(f"Outputs match within tolerance: {'✅' if is_close else '❌'}")
    
    # Calculate difference
    diff = (hf_logits - nano_logits).abs()
    print(f"Difference - mean: {diff.mean().item():.6f}, max: {diff.max().item():.6f}")
    
    # Compare last token predictions (most important for generation)
    hf_last = hf_logits[0, -1]
    nano_last = nano_logits[0, -1]
    last_diff = (hf_last - nano_last).abs()
    
    print(f"\nLast token logits - diff mean: {last_diff.mean().item():.6f}, max: {last_diff.max().item():.6f}")
    
    # Get top-5 tokens for each model
    hf_top = torch.topk(hf_last, 5)
    nano_top = torch.topk(nano_last, 5)
    
    print("\nTop-5 predictions (HF):")
    for i, (token_id, score) in enumerate(zip(hf_top.indices.tolist(), hf_top.values.tolist())):
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. {token_id} ({token!r}) - score: {score:.4f}")
    
    print("\nTop-5 predictions (nano-vllm):")
    for i, (token_id, score) in enumerate(zip(nano_top.indices.tolist(), nano_top.values.tolist())):
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. {token_id} ({token!r}) - score: {score:.4f}")
else:
    print("❌ Output shapes don't match")

# First compare model weights
compare_key_weights()

print("\n=== Debug Complete ===")
