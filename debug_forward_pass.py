#!/usr/bin/env python
"""
Debug script to perform a step-by-step comparison of forward pass between
HuggingFace and nano-vllm implementations of Qwen3.

This script focuses on tracking every tensor through the forward pass to identify
where the implementations diverge.
"""
import torch
import torch.nn.functional as F
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

# Set up debugging flags
VERBOSE = True  # Set to True for more detailed output
TRACE_TENSORS = True  # Set to True to print tensor values
MAX_TRACE_ENTRIES = 5  # Maximum number of tensor values to print in each dimension

# Set device - use CPU for more predictable behavior during debugging
USE_CPU = True
device = torch.device("cpu")
print(f"Using device: {device}")

# Test with shortest prompt for debugging
test_prompt = "Hi!"
model_name = "Qwen/Qwen3-0.6B"

# Helper function for tensor debugging
def debug_tensor(name, tensor, prefix=""):
    """Print detailed information about a tensor with optional value tracing."""
    if not VERBOSE:
        return
    
    print(f"{prefix}{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Mean: {tensor.float().mean().item():.6f}")
    print(f"  Std: {tensor.float().std().item():.6f}")
    
    if TRACE_TENSORS:
        # Handle different tensor dimensions
        if tensor.dim() == 1:
            # For 1D tensor, just print the first few values
            values = tensor.float().cpu().tolist()[:MAX_TRACE_ENTRIES]
            print(f"  Values: {values}")
        elif tensor.dim() == 2:
            # For 2D tensor, print a small corner
            corner = tensor[:min(MAX_TRACE_ENTRIES, tensor.shape[0]), 
                           :min(MAX_TRACE_ENTRIES, tensor.shape[1])]
            print(f"  Corner values:")
            for row in corner.float().cpu().tolist():
                print(f"    {row}")
        else:
            # For higher dimensions, flatten and show the first few values
            flat = tensor.flatten()
            values = flat[:MAX_TRACE_ENTRIES].float().cpu().tolist()
            print(f"  First {len(values)} values of flattened tensor: {values}")
            
            # Also show last few values to catch potential issues at the end
            last_values = flat[-MAX_TRACE_ENTRIES:].float().cpu().tolist()
            print(f"  Last {len(last_values)} values of flattened tensor: {last_values}")

# Helper function to compare tensors
def compare_tensors(name, hf_tensor, nano_tensor, rtol=1e-5, atol=1e-5):
    """Compare two tensors and print detailed comparison."""
    print(f"\n=== Comparing {name} ===")
    
    # Convert both to same dtype for comparison
    hf_tensor = hf_tensor.float()
    nano_tensor = nano_tensor.float()
    
    # Print basic info
    print(f"  HF shape: {hf_tensor.shape}, Nano shape: {nano_tensor.shape}")
    
    # Check if shapes match
    if hf_tensor.shape != nano_tensor.shape:
        print("  ❌ Shape mismatch!")
        return False
    
    # Calculate basic statistics
    hf_mean = hf_tensor.mean().item()
    nano_mean = nano_tensor.mean().item()
    hf_std = hf_tensor.std().item()
    nano_std = nano_tensor.std().item()
    
    print(f"  Mean - HF: {hf_mean:.6f}, Nano: {nano_mean:.6f}, Diff: {abs(hf_mean - nano_mean):.6f}")
    print(f"  Std - HF: {hf_std:.6f}, Nano: {nano_std:.6f}, Diff: {abs(hf_std - nano_std):.6f}")
    
    # Calculate differences
    abs_diff = (hf_tensor - nano_tensor).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    
    # Check if tensors are close
    is_close = torch.allclose(hf_tensor, nano_tensor, rtol=rtol, atol=atol)
    print(f"  Values match within tolerance: {'✅' if is_close else '❌'}")
    
    # If values don't match closely, find and print the worst differences
    if not is_close:
        flat_diff = abs_diff.flatten()
        worst_idxs = torch.topk(flat_diff, min(5, flat_diff.numel())).indices
        
        print("  Sample of worst differences:")
        for i, idx in enumerate(worst_idxs):
            # Convert flat index back to tensor indices
            tensor_idx = torch.unravel_index(idx, abs_diff.shape)
            hf_val = hf_tensor[tensor_idx].item()
            nano_val = nano_tensor[tensor_idx].item()
            diff = abs(hf_val - nano_val)
            print(f"    {i+1}. Index {tensor_idx}: HF={hf_val:.6f}, Nano={nano_val:.6f}, Diff={diff:.6f}")
    
    return is_close

# Set up tokenizer
print(f"Loading tokenizer from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Tokenize input
prompt_tokens = tokenizer.encode(test_prompt)
input_ids = torch.tensor([prompt_tokens], device=device)
positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
print(f"Prompt: '{test_prompt}'")
print(f"Token IDs: {prompt_tokens}")

print("\n=== LOADING MODELS ===")

# Load HuggingFace model
print(f"Loading HuggingFace model from {model_name}")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32,  # Use float32 for debugging
    trust_remote_code=True
)
hf_model = hf_model.to(device)
hf_model.eval()

# Load nano-vllm model
print(f"Loading nano-vllm model from {model_name}")
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
nano_model = Qwen3ForCausalLM(config)
load_model(nano_model, model_name)
nano_model = nano_model.to(device) 
nano_model.eval()

# Configure hooks to capture internal activations
hf_activations = {}
nano_activations = {}

# Define hook factories
def get_hf_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hf_activations[name] = output[0].detach()
        else:
            hf_activations[name] = output.detach()
        if VERBOSE:
            print(f"Captured HF activation: {name}")
            debug_tensor(name, hf_activations[name], prefix="  HF ")
    return hook

def get_nano_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            nano_activations[name] = output[0].detach()
        else:
            nano_activations[name] = output.detach()
        if VERBOSE:
            print(f"Captured Nano activation: {name}")
            debug_tensor(name, nano_activations[name], prefix="  Nano ")
    return hook

# Register hooks for core components
print("\nRegistering hooks for HuggingFace model...")
try:
    # Embeddings
    hf_model.model.embed_tokens.register_forward_hook(get_hf_hook('embed'))
    
    # First layer components
    hf_model.model.layers[0].input_layernorm.register_forward_hook(get_hf_hook('layer0_input_norm'))
    hf_model.model.layers[0].self_attn.q_proj.register_forward_hook(get_hf_hook('layer0_q_proj'))
    hf_model.model.layers[0].self_attn.k_proj.register_forward_hook(get_hf_hook('layer0_k_proj'))
    hf_model.model.layers[0].self_attn.v_proj.register_forward_hook(get_hf_hook('layer0_v_proj'))
    
    # Try to hook into internal attention components if available
    # These might not be accessible depending on the implementation
    try:
        # These hooks might not work depending on HF implementation
        hf_model.model.layers[0].self_attn.register_forward_hook(get_hf_hook('layer0_attn_output'))
    except:
        print("  Could not register some detailed attention hooks for HF model")
        
    hf_model.model.layers[0].self_attn.o_proj.register_forward_hook(get_hf_hook('layer0_o_proj'))
    hf_model.model.layers[0].post_attention_layernorm.register_forward_hook(get_hf_hook('layer0_post_attn_norm'))
    hf_model.model.layers[0].mlp.register_forward_hook(get_hf_hook('layer0_mlp'))
    
    # Final layer norm
    hf_model.model.norm.register_forward_hook(get_hf_hook('final_norm'))
    
    # LM head
    hf_model.lm_head.register_forward_hook(get_hf_hook('lm_head'))
    
    print("Successfully registered HF hooks")
except Exception as e:
    print(f"Error registering HF hooks: {e}")

print("\nRegistering hooks for nano-vllm model...")
try:
    # Embeddings
    nano_model.model.embed_tokens.register_forward_hook(get_nano_hook('embed'))
    
    # First layer components
    nano_model.model.layers[0].input_layernorm.register_forward_hook(get_nano_hook('layer0_input_norm'))
    nano_model.model.layers[0].self_attn.q_proj.register_forward_hook(get_nano_hook('layer0_q_proj'))
    nano_model.model.layers[0].self_attn.k_proj.register_forward_hook(get_nano_hook('layer0_k_proj'))
    nano_model.model.layers[0].self_attn.v_proj.register_forward_hook(get_nano_hook('layer0_v_proj'))
    
    # Try to hook into internal attention components if available
    try:
        nano_model.model.layers[0].self_attn.register_forward_hook(get_nano_hook('layer0_attn_output'))
    except:
        print("  Could not register some detailed attention hooks for nano model")
        
    nano_model.model.layers[0].self_attn.o_proj.register_forward_hook(get_nano_hook('layer0_o_proj'))
    nano_model.model.layers[0].post_attention_layernorm.register_forward_hook(get_nano_hook('layer0_post_attn_norm'))
    nano_model.model.layers[0].mlp.register_forward_hook(get_nano_hook('layer0_mlp'))
    
    # Final layer norm
    nano_model.model.norm.register_forward_hook(get_nano_hook('final_norm'))
    
    # LM head
    nano_model.lm_head.register_forward_hook(get_nano_hook('lm_head'))
    
    print("Successfully registered nano hooks")
except Exception as e:
    print(f"Error registering nano hooks: {e}")

# Run forward pass
print("\n=== RUNNING FORWARD PASS ===")

# Add hooks specifically to track rotary embedding application
# Track Q/K before and after rotary embedding
try:
    # For HF, try to get the rotary embedding function
    def hf_track_qk_pre_rotary(module, input, output):
        hf_activations['pre_rotary_q'] = input[0].detach()
        hf_activations['pre_rotary_k'] = input[1].detach()
    
    # For nano, get the rotary embedding function
    def nano_track_qk_post_rotary(module, input, output):
        nano_activations['post_rotary_q'] = output.detach()
    
    # This will depend on the exact model implementation
    # You may need to adapt based on the actual module structure
    try:
        # Try to hook rotary application in HF
        for module in hf_model.modules():
            if 'rotary' in str(type(module)).lower():
                module.register_forward_hook(get_hf_hook('rotary_output'))
                print(f"Found HF rotary module: {type(module)}")
    except Exception as e:
        print(f"Could not register HF rotary hook: {e}")
    
    # Try to hook rotary application in nano-vllm
    for i, layer in enumerate(nano_model.model.layers):
        try:
            layer.self_attn.rotary_emb.register_forward_hook(get_nano_hook(f'layer{i}_rotary_output'))
            print(f"Found nano rotary module in layer {i}")
        except Exception as e:
            print(f"Could not register nano rotary hook for layer {i}: {e}")
except Exception as e:
    print(f"Error setting up rotary tracking: {e}")

# Run forward pass with both models
with torch.no_grad():
    print("\nRunning HuggingFace model...")
    hf_logits = hf_model(input_ids).logits
    
    print("\nRunning nano-vllm model...")
    nano_logits = nano_model(input_ids, positions)

# Compare outputs
print("\n=== COMPARING OUTPUTS ===")
logits_match = compare_tensors("logits", hf_logits, nano_logits)

# Compare detailed activation states
print("\n=== COMPARING INTERNAL ACTIVATIONS ===")
# Find all keys that were captured by both models
common_activations = sorted(set(hf_activations.keys()).intersection(set(nano_activations.keys())))
print(f"Found {len(common_activations)} comparable activation points:")
for name in common_activations:
    compare_tensors(name, hf_activations[name], nano_activations[name])

# Check if there are any HF-only or nano-only activations
hf_only = set(hf_activations.keys()) - set(nano_activations.keys())
if hf_only:
    print(f"\nActivations only in HF model: {sorted(hf_only)}")
    
nano_only = set(nano_activations.keys()) - set(hf_activations.keys())
if nano_only:
    print(f"\nActivations only in nano model: {sorted(nano_only)}")

# Compare next token predictions
print("\n=== TOKEN PREDICTION COMPARISON ===")
# Get the next token predictions for the last position
hf_next_token_logits = hf_logits[0, -1, :]
nano_next_token_logits = nano_logits[0, -1, :]

# Apply softmax to get probabilities
hf_probs = F.softmax(hf_next_token_logits, dim=-1)
nano_probs = F.softmax(nano_next_token_logits, dim=-1)

# Get top-k predictions
k = 5
hf_top_k = torch.topk(hf_next_token_logits, k)
nano_top_k = torch.topk(nano_next_token_logits, k)

print("HF Top 5 next token predictions:")
for i, (token_id, score) in enumerate(zip(hf_top_k.indices.tolist(), hf_top_k.values.tolist())):
    prob = hf_probs[token_id].item()
    token_text = tokenizer.decode([token_id])
    print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}, prob: {prob:.6f}")

print("\nNano-VLLM Top 5 next token predictions:")
for i, (token_id, score) in enumerate(zip(nano_top_k.indices.tolist(), nano_top_k.values.tolist())):
    prob = nano_probs[token_id].item()
    token_text = tokenizer.decode([token_id])
    print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}, prob: {prob:.6f}")

# Check common top predictions
hf_top_ids = set(hf_top_k.indices.tolist())
nano_top_ids = set(nano_top_k.indices.tolist())
common_ids = hf_top_ids.intersection(nano_top_ids)
print(f"\nOverlap in top predictions: {len(common_ids)}/{k}")
print(f"Common tokens: {common_ids}")

# Calculate cross entropy between the two outputs to measure distribution divergence
kl_div_hf_nano = F.kl_div(
    F.log_softmax(nano_next_token_logits, dim=-1),
    F.softmax(hf_next_token_logits, dim=-1),
    reduction='sum'
)
print(f"\nKL divergence (nano||hf): {kl_div_hf_nano.item():.6f}")

# Print summary
print("\n=== DEBUGGING SUMMARY ===")
if logits_match:
    print("✅ Model outputs match!")
else:
    print("❌ Model outputs differ!")
    
    # Check where the divergence starts
    mismatch_points = []
    for name in common_activations:
        if not torch.allclose(
            hf_activations[name].float(), 
            nano_activations[name].float(),
            rtol=1e-4, atol=1e-4
        ):
            mismatch_points.append(name)
    
    if mismatch_points:
        print(f"First point of divergence: {mismatch_points[0]}")
        print(f"All divergence points: {mismatch_points}")
    else:
        print("No divergence found in the captured activations, but outputs differ.")
        print("Try capturing more intermediate points or check for uncaptured operations.")
