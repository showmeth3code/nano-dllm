"""
Test script for nano-vllm using the Qwen3 model.
This script compares generation from HuggingFace and nano-vllm implementations.
"""
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from nanovllm.sampling_params import SamplingParams
import torch
from nanovllm.models.qwen3 import Qwen3ForCausalLM
import torch.nn.functional as F
from nanovllm.utils.loader import load_model
import datetime
import numpy as np

# Check device availability
print("PyTorch device availability:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

# Set device - use CPU if there are known issues with MPS
USE_CPU = False  # Set to True if you want to force CPU usage
device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else 
                     "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not USE_CPU else
                     "cpu")

print(f"Will be using device: {device}")

# Test models - use smallest model available for faster testing
models = [
    "Qwen/Qwen3-0.6B",  # Use an MPS-optimized Qwen3 variant if available
]

# Test prompts - start with shortest prompt for debugging
prompts = [
    "Hi!",  # Shorter prompt for quick testing
    # "Who are you?",
]

# Helper function to compare tensors
def tensor_equal_within_tol(tensor1, tensor2, rtol=1e-3, atol=1e-5):
    """Check if two tensors are equal within a tolerance."""
    if tensor1.shape != tensor2.shape:
        return False
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

# Helper function to compare model weights more thoroughly
def compare_model_weights(hf_model, nano_model, detailed=True):
    print("\n=== MODEL WEIGHTS COMPARISON ===")
    
    # Get HF model state dict
    hf_state_dict = hf_model.state_dict()
    
    # Get nano-vllm model state dict
    nano_state_dict = nano_model.state_dict()
    
    # Compare key counts
    print(f"HF model has {len(hf_state_dict)} keys")
    print(f"nano-vllm model has {len(nano_state_dict)} keys")
    
    # Check key names
    hf_keys = set(hf_state_dict.keys())
    nano_keys = set(nano_state_dict.keys())
    
    print(f"Keys in HF but not in nano-vllm: {len(hf_keys - nano_keys)}")
    if detailed and len(hf_keys - nano_keys) > 0:
        print(f"Examples: {list(hf_keys - nano_keys)[:5]}")
        
    print(f"Keys in nano-vllm but not in HF: {len(nano_keys - hf_keys)}")
    if detailed and len(nano_keys - hf_keys) > 0:
        print(f"Examples: {list(nano_keys - hf_keys)[:5]}")
    
    # Compare some important tensor values
    key_matches = []
    key_mismatches = []
    
    # Find equivalent key pairs between models to compare
    key_mapping = {}
    
    # First look for exact matches
    for hf_key in hf_keys:
        if hf_key in nano_keys:
            key_mapping[hf_key] = hf_key
    
    # Check for stripped prefixes
    for hf_key in hf_keys:
        if hf_key.startswith('model.'):
            stripped_key = hf_key[6:]  # Remove 'model.' prefix
            if stripped_key in nano_keys and hf_key not in key_mapping:
                key_mapping[hf_key] = stripped_key
    
    # Create a more detailed mapping for specific layers
    manual_mappings = [
        # Example: ('model.embed_tokens.weight', 'embed_tokens.weight'),
        # Key Qwen components - embeddings, attention, etc.
        ('model.embed_tokens.weight', 'model.embed_tokens.weight'),
        ('lm_head.weight', 'lm_head.weight'),
        ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.q.weight'),
        ('model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.k.weight'),
        ('model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.v.weight'),
    ]
    
    for hf_key, nano_key in manual_mappings:
        if hf_key in hf_keys and nano_key in nano_keys:
            key_mapping[hf_key] = nano_key
    
    print(f"\nFound {len(key_mapping)} matching key mappings")
    
    # Check key pairs - first examine critical layers
    critical_keys = ['embed_tokens', 'lm_head', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # First check for critical keys
    print("\nChecking critical layers first...")
    for hf_key, nano_key in key_mapping.items():
        if any(critical in hf_key for critical in critical_keys):
            # Compare tensor stats
            hf_tensor = hf_state_dict[hf_key].detach().cpu().float()
            nano_tensor = nano_state_dict[nano_key].detach().cpu().float()
            
            # Skip if shapes don't match
            if hf_tensor.shape != nano_tensor.shape:
                print(f"Shape mismatch: {hf_key} ({hf_tensor.shape}) vs {nano_key} ({nano_tensor.shape})")
                continue
                
            # Get stats
            hf_mean = hf_tensor.mean().item()
            nano_mean = nano_tensor.mean().item()
            hf_std = hf_tensor.std().item()
            nano_std = nano_tensor.std().item()
            
            # Calculate difference
            mean_diff = abs(hf_mean - nano_mean)
            std_diff = abs(hf_std - nano_std)
            
            # Check exact equality
            exact_match = tensor_equal_within_tol(hf_tensor, nano_tensor)
            
            stats_info = {
                'hf_key': hf_key,
                'nano_key': nano_key,
                'shape': list(hf_tensor.shape),
                'hf_mean': hf_mean,
                'nano_mean': nano_mean, 
                'mean_diff': mean_diff,
                'hf_std': hf_std,
                'nano_std': nano_std,
                'std_diff': std_diff,
                'exact_match': exact_match
            }
            
            # Check if means and stds are close
            if mean_diff < 1e-3 and std_diff < 1e-3:
                key_matches.append(stats_info)
                print(f"  Match: {hf_key} -> {nano_key} (Exact: {exact_match})")
            else:
                key_mismatches.append(stats_info)
                print(f"  Mismatch: {hf_key} -> {nano_key} (HF: {hf_mean:.4f}, nano: {nano_mean:.4f})")
    
    # Check random sample of other keys
    count = 0
    print("\nChecking sample of other layers...")
    for hf_key, nano_key in key_mapping.items():
        if any(critical in hf_key for critical in critical_keys):
            continue  # Already checked above
            
        # Skip if already processed enough samples
        count += 1
        if count > 10:
            break
            
        # Compare tensor stats
        hf_tensor = hf_state_dict[hf_key].detach().cpu().float()
        nano_tensor = nano_state_dict[nano_key].detach().cpu().float()
        
        # Skip if shapes don't match
        if hf_tensor.shape != nano_tensor.shape:
            print(f"Shape mismatch: {hf_key} ({hf_tensor.shape}) vs {nano_key} ({nano_tensor.shape})")
            continue
            
        # Get stats
        hf_mean = hf_tensor.mean().item()
        nano_mean = nano_tensor.mean().item()
        hf_std = hf_tensor.std().item()
        nano_std = nano_tensor.std().item()
        
        # Calculate difference
        mean_diff = abs(hf_mean - nano_mean)
        std_diff = abs(hf_std - nano_std)
        
        # Check exact equality
        exact_match = tensor_equal_within_tol(hf_tensor, nano_tensor)
        
        stats_info = {
            'hf_key': hf_key,
            'nano_key': nano_key,
            'shape': list(hf_tensor.shape),
            'hf_mean': hf_mean,
            'nano_mean': nano_mean, 
            'mean_diff': mean_diff,
            'hf_std': hf_std,
            'nano_std': nano_std,
            'std_diff': std_diff,
            'exact_match': exact_match
        }
        
        # Check if means and stds are close
        if mean_diff < 1e-3 and std_diff < 1e-3:
            key_matches.append(stats_info)
            print(f"  Match: {hf_key} -> {nano_key} (Exact: {exact_match})")
        else:
            key_mismatches.append(stats_info)
            print(f"  Mismatch: {hf_key} -> {nano_key} (HF: {hf_mean:.4f}, nano: {nano_mean:.4f})")
    
    # Print summary results
    print(f"\nMatching weight statistics: {len(key_matches)}/{len(key_matches) + len(key_mismatches)} checked layers")
    print(f"Exact tensor matches: {sum(1 for i in key_matches if i['exact_match'])}/{len(key_matches)}")
    
    return len(key_mismatches) == 0

for model_name in models:
    print("\n" + "="*80)
    print(f"Testing model: {model_name}")
    print("="*80)
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

    # Set torch dtypes appropriately for the device
    if device.type == 'mps':
        torch_dtype = torch.float32  # MPS works better with float32
    else:
        torch_dtype = torch.float16  # Use float16 for CUDA or CPU
    print(f"Using torch dtype: {torch_dtype}")

    # Test with just vanilla model to avoid LLM Engine complexity
    from transformers import AutoConfig
    
    print("\n" + "="*40)
    print("Testing with direct model comparison")
    print("="*40)
    print(f"Using device: {device}")
    
    # Add debug function to capture and compare intermediate activations
    def debug_intermediate_activations(hf_model, nano_model, input_ids, positions):
        """Compare intermediate activations between HuggingFace and nano-vllm models"""
        print("\n=== DETAILED ACTIVATION COMPARISON ===")
        
        # Store activations
        hf_activations = {}
        nano_activations = {}
        
        # Register hooks for HuggingFace model
        def get_hf_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hf_activations[name] = output[0].detach() if output else None
                else:
                    hf_activations[name] = output.detach()
            return hook
        
        # Register hooks for nano-vllm model
        def get_nano_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    nano_activations[name] = output[0].detach() if output else None
                else:
                    nano_activations[name] = output.detach()
            return hook
        
        # Try to register hooks for both models
        try:
            # Register hooks for embeddings
            hf_model.model.embed_tokens.register_forward_hook(get_hf_activation('embed'))
            nano_model.model.embed_tokens.register_forward_hook(get_nano_activation('embed'))
            
            # Register hooks for first layer input norm
            hf_model.model.layers[0].input_layernorm.register_forward_hook(get_hf_activation('layer0_norm'))
            nano_model.model.layers[0].input_layernorm.register_forward_hook(get_nano_activation('layer0_norm'))
            
            # Register hooks for first layer attention
            hf_model.model.layers[0].self_attn.q_proj.register_forward_hook(get_hf_activation('layer0_qproj'))
            nano_model.model.layers[0].self_attn.q_proj.register_forward_hook(get_nano_activation('layer0_qproj'))
            
            hooks_registered = True
            print("Successfully registered hooks for both models")
        except Exception as e:
            print(f"Couldn't register all hooks: {e}")
            hooks_registered = False
        
        # Run both models
        with torch.no_grad():
            # Run HuggingFace model
            _ = hf_model(input_ids)
            
            # Run nano-vllm model
            _ = nano_model(input_ids, positions)
        
        # Compare activations
        if hooks_registered:
            for name in sorted(set(list(hf_activations.keys()) + list(nano_activations.keys()))):
                if name in hf_activations and name in nano_activations:
                    hf_act = hf_activations[name].float()
                    nano_act = nano_activations[name].float()
                    
                    # Print shapes
                    print(f"\nActivation '{name}':")
                    print(f"  HF shape: {hf_act.shape}, Nano shape: {nano_act.shape}")
                    
                    # Check if shapes match
                    if hf_act.shape != nano_act.shape:
                        print("  ❌ Shape mismatch!")
                        continue
                    
                    # Calculate statistics
                    hf_mean = hf_act.mean().item()
                    nano_mean = nano_act.mean().item()
                    hf_std = hf_act.std().item()
                    nano_std = nano_act.std().item()
                    
                    # Calculate differences
                    abs_diff = (hf_act - nano_act).abs()
                    max_diff = abs_diff.max().item()
                    mean_diff = abs_diff.mean().item()
                    
                    # Check if activations are close
                    is_close = torch.allclose(hf_act, nano_act, rtol=1e-3, atol=1e-3)
                    
                    # Print stats
                    print(f"  Mean - HF: {hf_mean:.6f}, Nano: {nano_mean:.6f}, Diff: {abs(hf_mean - nano_mean):.6f}")
                    print(f"  Std - HF: {hf_std:.6f}, Nano: {nano_std:.6f}, Diff: {abs(hf_std - nano_std):.6f}")
                    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                    print(f"  Values match: {'✅' if is_close else '❌'}")
                    
                    # If values don't match, print a sample of the differences
                    if not is_close:
                        # Find worst mismatches
                        flat_diff = abs_diff.flatten()
                        worst_idxs = torch.topk(flat_diff, 5).indices
                        
                        # Print sample of worst differences
                        print("  Sample of worst differences:")
                        for i, idx in enumerate(worst_idxs):
                            # Convert flat index back to tensor indices
                            tensor_idx = torch.unravel_index(idx, abs_diff.shape)
                            hf_val = hf_act[tensor_idx].item()
                            nano_val = nano_act[tensor_idx].item()
                            diff = abs(hf_val - nano_val)
                            print(f"    {i+1}. Index {tensor_idx}: HF={hf_val:.6f}, Nano={nano_val:.6f}, Diff={diff:.6f}")
                else:
                    print(f"\nActivation '{name}' only found in {'HF' if name in hf_activations else 'Nano'} model")
        
        return hf_activations, nano_activations
    
    # Load HuggingFace model for reference
    print("\n=== HUGGINGFACE MODEL ===")
    start_time = datetime.datetime.now()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        trust_remote_code=True
    )
    hf_model = hf_model.to(device)
    hf_model.eval()
    print(f"HF model loaded in {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds")
    print(f"HF model device: {hf_model.device}")
    
    # Load nano-vllm model
    print("\n=== NANO-VLLM MODEL ===")
    start_time = datetime.datetime.now()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    nano_model = Qwen3ForCausalLM(config)
    load_model(nano_model, model_name)
    nano_model = nano_model.to(device) 
    nano_model.eval()
    print(f"nano-vllm model device: {next(nano_model.parameters()).device}")
    print(f"nano-vllm model loaded in {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds")
    
    # Compare model weights to identify potential mismatches
    weights_match = compare_model_weights(hf_model, nano_model)
    
    # Test with first prompt
    test_prompt = prompts[0]
    print(f"\nTesting with prompt: {repr(test_prompt)}")
    
    # Tokenize
    prompt_tokens = tokenizer.encode(test_prompt)
    input_ids = torch.tensor([prompt_tokens], device=device)
    
    print(f"Prompt token IDs: {prompt_tokens}")
    print(f"Decoded prompt: {tokenizer.decode(prompt_tokens)}")
    
    # Positions tensor for nano-vllm
    positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
    
    # Run detailed comparison of intermediate activations
    debug_intermediate_activations(hf_model, nano_model, input_ids, positions)
    
    # Debug HF model internal activations
    print("\n=== COMPARING INTERNAL ACTIVATIONS ===")
    
    # Hook to capture HF model intermediate activations
    hf_activations = {}
    
    # Define hooks for HuggingFace model
    def get_hf_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                # If output is a tuple, we're interested in the first element
                hf_activations[name] = output[0].detach() if output else None
            else:
                hf_activations[name] = output.detach()
        return hook
    
    # Register hooks for some layers - this structure depends on the HF model implementation
    try:
        # Try to register hooks - different models might have different structures
        hf_model.model.embed_tokens.register_forward_hook(get_hf_activation('embed_tokens'))
        hf_model.model.layers[0].input_layernorm.register_forward_hook(get_hf_activation('layer0_input_norm'))
        hooks_registered = True
        print("Successfully registered hooks for HF model")
    except (AttributeError, IndexError) as e:
        print(f"Couldn't register hooks for HF model: {e}")
        hooks_registered = False
    
    # Get token predictions from both models with additional debugging
    with torch.no_grad():
        # HuggingFace predictions
        print("\nRunning HuggingFace model inference...")
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF logits dtype: {hf_logits.dtype}")
        print(f"HF logits mean/std: {hf_logits.float().mean().item():.6f}/{hf_logits.float().std().item():.6f}")
        
        hf_next_token_logits = hf_logits[0, -1, :]
        hf_next_token_probs = F.softmax(hf_next_token_logits, dim=-1)
        hf_top_k = torch.topk(hf_next_token_logits, 5)
        
        # Position tensor for nano-vllm
        positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
        print(f"Positions tensor: {positions.shape}, device: {positions.device}")
        
        # nano-vllm predictions
        print("\nRunning nano-vllm model inference...")
        nano_logits = nano_model(input_ids, positions)
        print(f"Nano logits shape: {nano_logits.shape}")
        print(f"Nano logits dtype: {nano_logits.dtype}")
        print(f"Nano logits mean/std: {nano_logits.float().mean().item():.6f}/{nano_logits.float().std().item():.6f}")
        
        # Check logits difference directly
        if nano_logits.shape == hf_logits.shape:
            logits_diff = (nano_logits.float() - hf_logits.float()).abs()
            print(f"Logits difference - mean: {logits_diff.mean().item():.6f}, max: {logits_diff.max().item():.6f}")
            
            # Check if specific tokens are very different
            token_diffs = (nano_logits[0, -1, :].float() - hf_logits[0, -1, :].float()).abs()
            worst_tokens = torch.topk(token_diffs, 5)
            print("Tokens with largest logit differences:")
            for i, (token_id, diff) in enumerate(zip(worst_tokens.indices.tolist(), worst_tokens.values.tolist())):
                token_text = tokenizer.decode([token_id])
                print(f"  {i+1}. Token {token_id} ({repr(token_text)}): diff={diff:.4f}")
        
        nano_next_token_logits = nano_logits[0, -1, :]
        nano_next_token_probs = F.softmax(nano_next_token_logits, dim=-1)
        nano_top_k = torch.topk(nano_next_token_logits, 5)
    
    # Compare predictions
    print("\n=== TOKEN PREDICTION COMPARISON ===")
    print("HF Top 5 next token predictions:")
    for i, (token_id, score) in enumerate(zip(hf_top_k.indices.tolist(), hf_top_k.values.tolist())):
        prob = hf_next_token_probs[token_id].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}, prob: {prob:.6f}")
    
    print("\nNano-VLLM Top 5 next token predictions:")
    for i, (token_id, score) in enumerate(zip(nano_top_k.indices.tolist(), nano_top_k.values.tolist())):
        prob = nano_next_token_probs[token_id].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ({repr(token_text)}): {score:.4f}, prob: {prob:.6f}")
    
    # Check for overlap in top predictions
    hf_top_ids = set(hf_top_k.indices.tolist())
    nano_top_ids = set(nano_top_k.indices.tolist())
    common_ids = hf_top_ids.intersection(nano_top_ids)
    print(f"\nOverlap in top predictions: {len(common_ids)}/{min(len(hf_top_ids), len(nano_top_ids))}")
    if common_ids:
        print(f"Common top tokens: {common_ids}")
    
    # Compare top logit distributions
    print("\n=== LOGIT DISTRIBUTION COMPARISON ===")
    hf_entropy = -(hf_next_token_probs * torch.log(hf_next_token_probs + 1e-10)).sum().item()
    nano_entropy = -(nano_next_token_probs * torch.log(nano_next_token_probs + 1e-10)).sum().item()
    print(f"HF entropy: {hf_entropy:.4f}")
    print(f"Nano-VLLM entropy: {nano_entropy:.4f}")
    
    # Display internal activations comparison if hooks were registered
    if hooks_registered and hf_activations:
        print("\n=== INTERNAL ACTIVATIONS ===")
        for name, activation in hf_activations.items():
            print(f"HF {name}: shape={activation.shape}, mean={activation.float().mean().item():.6f}, std={activation.float().std().item():.6f}")
    
    # Generate text with both models
    print("\n=== GENERATION COMPARISON ===")
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Always use greedy decoding for more direct comparison
    print("Generating with HuggingFace model (greedy decoding)...")
    with torch.no_grad():
        # Generate with greedy decoding
        hf_outputs = hf_model.generate(
            input_ids, 
            max_new_tokens=20, 
            do_sample=False,  # Use greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
        hf_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
        print(f"HF Generated: {repr(hf_text)}")
    
    # Reset random seed for nano-vllm
    torch.manual_seed(123)
    
    # nano-vllm generation with detailed token analysis
    print("\nGenerating with nano-vllm model (sampling)...")
    curr_ids = prompt_tokens.copy()
    generation_mismatches = []
    
    with torch.no_grad():
        for i in range(20):  # Generate 20 tokens
            input_tensor = torch.tensor(curr_ids, device=device).unsqueeze(0)  # Add batch dim
            positions = torch.arange(len(curr_ids), device=device).unsqueeze(0)  # Add batch dim
            
            # Forward pass through model
            logits = nano_model(input_tensor, positions)
            
            # Get next token prediction
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / 0.7
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Add token to sequence
            curr_ids.append(next_token_id)
            
            # Check if this matches the corresponding token in HF output
            hf_token_idx = len(prompt_tokens) + i
            if hf_token_idx < len(hf_outputs[0]):
                hf_token_id = hf_outputs[0][hf_token_idx].item()
                hf_token_text = tokenizer.decode([hf_token_id])
                matches_hf = hf_token_id == next_token_id
                
                if not matches_hf:
                    # Store mismatch details
                    generation_mismatches.append({
                        'position': i+1,
                        'hf_token_id': hf_token_id,
                        'hf_token_text': hf_token_text,
                        'nano_token_id': next_token_id,
                        'nano_token_text': tokenizer.decode([next_token_id])
                    })
                
                print(f"Token {i+1}: {next_token_id} -> {repr(tokenizer.decode([next_token_id]))} {'✓' if matches_hf else '✗'}")
            else:
                print(f"Token {i+1}: {next_token_id} -> {repr(tokenizer.decode([next_token_id]))}")
    
    # Reset random seed
    torch.manual_seed(42)
    
    # Generate with nano-vllm using greedy decoding to match with HF
    print("\nGenerating with nano-vllm model (greedy)...")
    curr_ids = prompt_tokens.copy()
    
    with torch.no_grad():
        for i in range(20):  # Generate 20 tokens
            input_tensor = torch.tensor(curr_ids, device=device).unsqueeze(0)
            positions = torch.arange(len(curr_ids), device=device).unsqueeze(0)
            
            # Forward pass through model
            logits = nano_model(input_tensor, positions)
            
            # Get next token prediction (greedy)
            next_token_id = logits[0, -1, :].argmax().item()
            
            # Add token to sequence
            curr_ids.append(next_token_id)
            
            # Check if this matches the corresponding token in HF output
            hf_token_idx = len(prompt_tokens) + i
            if hf_token_idx < len(hf_outputs[0]):
                hf_token_id = hf_outputs[0][hf_token_idx].item()
                hf_token_text = tokenizer.decode([hf_token_id])
                matches_hf = hf_token_id == next_token_id
                
                print(f"Token {i+1}: {next_token_id} -> {repr(tokenizer.decode([next_token_id]))} {'✓' if matches_hf else '✗'}")
            else:
                print(f"Token {i+1}: {next_token_id} -> {repr(tokenizer.decode([next_token_id]))}")
    
    # Print final outputs
    nano_text = tokenizer.decode(curr_ids, skip_special_tokens=True)
    
    print("\n=== GENERATION SUMMARY ===")
    print(f"HF Generated: {repr(hf_text)}")
    print(f"Nano-vllm Generated: {repr(nano_text)}")
    
    # Print token mismatch summary
    if generation_mismatches:
        print(f"\nToken mismatches: {len(generation_mismatches)}/20")
        print("First few mismatches:")
        for i, mismatch in enumerate(generation_mismatches[:5]):
            print(f"  Position {mismatch['position']}: HF={mismatch['hf_token_id']}({repr(mismatch['hf_token_text'])}) vs Nano={mismatch['nano_token_id']}({repr(mismatch['nano_token_text'])})")
    else:
        print("\nNo token mismatches found!")
    
    
    # Now test with LLMEngine to verify integration works
    def test_llm_engine(model_name, tokenizer, prompts, sampling_params):
        """Test using the LLMEngine for generation"""
        print("\n" + "="*40)
        print("Testing with LLMEngine")
        print("="*40)
        
        # Initialize configuration
        config = Config(model_path=model_name)
        config.enforce_eager = True  # Use eager mode for debugging
        
        print("\n=== MODEL CONFIGURATION ===")
        print(f"Model path: {model_name}")
        print(f"Model type: {getattr(config.hf_config, 'model_type', 'unknown')}")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"EOS token ID: {config.eos}")
        print("===============================")
        
        # Initialize LLMEngine
        print("\nInitializing LLMEngine...")
        engine = LLMEngine(config=config, engine_args={}, tokenizer=tokenizer)
        
        # Test with first prompt only
        test_prompt = prompts[0]
        print(f"\n--- Testing prompt: {repr(test_prompt)} ---")
        
        # Generate text with greedy sampling
        print("Generating text...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)  # Use greedy decoding
        
        try:
            # Try running the generate function
            outputs = engine.generate([test_prompt], sampling_params, use_tqdm=False)
            
            # Process output
            for output in outputs:
                text = output.get("text")
                
                # Decode and print output
                print("\nOutput from LLMEngine:")
                print(f"Generated text: {repr(text)}")
                
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            raise
    
    test_llm_engine(model_name, tokenizer, prompts, SamplingParams)
    
    print("\n" + "="*40)
    print("Test completed")
    print("="*40)
