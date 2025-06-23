import os
import gc
import torch
import time
from typing import Dict, Any
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.mps_optimized_attention import MPSOptimizedAttention

def patch_model_for_mps():
    """
    Monkey patch the model to use MPS-optimized attention when running on Apple Silicon
    
    Replaces standard attention with MPS-optimized attention that uses:
    - Reduced precision for weight-tensor product operations
    - Optimized tensor layouts for Metal performance
    - Specialized block-sparse attention patterns
    - More efficient MPS memory access patterns
    """
    # Store the original __init__ method
    original_init = Qwen3ForCausalLM.__init__
    
    # Define our patched initialization function
    def patched_init(self, config):
        # Call the original init first
        original_init(self, config)
        
        # Check if we're on MPS
        device_type = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Only apply MPS optimizations if we're on MPS device
        if device_type == 'mps':
            print("Applying MPS-optimized attention patches...")
            
            # Patch attention modules in each transformer layer
            for i, layer in enumerate(self.model.layers):
                # Get the original attention parameters
                orig_attn = layer.self_attn
                hidden_size = orig_attn.hidden_size
                num_heads = orig_attn.num_attention_heads
                num_kv_heads = orig_attn.num_key_value_heads
                head_dim = orig_attn.head_dim
                
                # Create optimized attention with the same parameters
                optimized_attn = MPSOptimizedAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    bias=orig_attn.q_proj.bias is not None
                )
                
                # Copy weights from original attention module
                optimized_attn.q_proj.weight.data.copy_(orig_attn.q_proj.weight.data)
                optimized_attn.k_proj.weight.data.copy_(orig_attn.k_proj.weight.data)
                optimized_attn.v_proj.weight.data.copy_(orig_attn.v_proj.weight.data)
                optimized_attn.o_proj.weight.data.copy_(orig_attn.o_proj.weight.data)
                
                # Copy bias if present
                if orig_attn.q_proj.bias is not None:
                    optimized_attn.q_proj.bias.data.copy_(orig_attn.q_proj.bias.data)
                    optimized_attn.k_proj.bias.data.copy_(orig_attn.k_proj.bias.data)
                    optimized_attn.v_proj.bias.data.copy_(orig_attn.v_proj.bias.data)
                    optimized_attn.o_proj.bias.data.copy_(orig_attn.o_proj.bias.data)
                
                # Replace the original attention with optimized version
                layer.self_attn = optimized_attn
                
                # Force a garbage collection after each layer replacement
                # This helps prevent MPS memory fragmentation
                if i % 4 == 0 and hasattr(torch.mps, 'empty_cache'):
                    gc.collect()
                    torch.mps.empty_cache()
            
            print(f"Successfully patched {len(self.model.layers)} attention layers for MPS")
            
            # Run a small warmup to initialize MPS compiled kernels
            run_mps_warmup(self)
    
    # Replace the original __init__ with our patched version
    Qwen3ForCausalLM.__init__ = patched_init

def run_mps_warmup(model: Qwen3ForCausalLM) -> None:
    """
    Run a small warm-up pass to initialize MPS compiled kernels
    
    This helps avoid the first-time compilation overhead during actual inference
    """
    print("Running MPS warmup pass...")
    
    # Make sure model is on MPS device
    model_device = next(model.parameters()).device
    if model_device.type != 'mps':
        print(f"Warning: Model is on {model_device.type}, not MPS device")
    
    try:
        # Create dummy inputs
        dummy_input = torch.ones((1, 16), dtype=torch.long, device='mps')
        dummy_positions = torch.arange(0, 16, dtype=torch.long, device='mps')
        
        # Time the warmup pass
        start_time = time.time()
        
        # Run a forward pass with dummy data
        with torch.no_grad():
            _ = model(dummy_input, dummy_positions)
            
        # Force synchronization
        torch.mps.synchronize()
        
        print(f"MPS warmup complete in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        print(f"MPS warmup failed: {e}")
    
    # Explicit cleanup
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

def optimize_mps_memory() -> None:
    """
    Apply memory optimizations for MPS device
    
    This configures memory handling on Apple Silicon to improve performance
    """
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return
        
    # Limit MPS memory high watermark
    if hasattr(torch.mps, 'set_per_process_memory_fraction'):
        # Leave some memory for system (80% usage)
        torch.mps.set_per_process_memory_fraction(0.8)
    
    # Force more aggressive garbage collection for MPS
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

def optimize_mps_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply MPS-specific optimizations to the configuration
    
    Args:
        config: Dictionary containing model/engine configuration
        
    Returns:
        Updated configuration optimized for MPS
    """
    # Don't modify the original config
    config = config.copy()
    
    # Optimized defaults for MPS
    config.setdefault("enforce_eager", True)  # Eager mode works better on MPS
    
    # Decrease KV cache block size for better memory access patterns on MPS
    config.setdefault("kvcache_block_size", 8)  
    
    # Limit batch size to avoid OOM on MPS
    config.setdefault("max_num_batched_tokens", 2048)
    
    return config

def apply_mps_optimizations():
    """Apply all MPS optimizations for the Apple Silicon environment"""
    # Set environment variables for MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Optimize memory handling
    optimize_mps_memory()
    
    # Apply model patches
    patch_model_for_mps()
    
    print("\nMPS optimization configuration complete:")
    print("- Applied optimized attention implementation")
    print("- Configured memory management")
    print("- Enabled kernel warmup")
    print("- Set fallback paths for operations not natively supported on MPS")
